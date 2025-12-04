"""
Unified Gateway for BeeKurse
Routes all services through a single port (8000) for single ngrok URL

Endpoints:
- /webhook (GET/POST) - WhatsApp webhook
- /cart/* - Cart API endpoints
- /view/cart/* - Cart web pages
- /view/wishlist/* - Wishlist web pages
- /gallery - Proxied to React frontend (port 5400)
- / - Health check
"""
import os
import json
import requests
import tempfile
import uuid
import time
import sqlite3
import re
import httpx
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Import cart manager
from .cart_manager import get_cart_manager, set_base_url

# User management
from .user_manager import get_user_manager

# Message history for contextual replies
from .message_history import get_message_history

app = FastAPI(title="BeeKurse Unified Gateway")

# ==================== Configuration ====================

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_verify_token_123")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5001/process")
VENDOR_BACKEND_URL = os.getenv("VENDOR_BACKEND_URL", "http://localhost:5001/vendor/process")
GALLERY_FRONTEND_URL = os.getenv("GALLERY_FRONTEND_URL", "http://localhost:5400")
WHATSAPP_BUSINESS_NUMBER = os.getenv("WHATSAPP_BUSINESS_NUMBER", "")

# Directory for downloaded vendor images
VENDOR_IMAGE_DIR = Path(__file__).parent.parent / "data" / "vendor_data" / "uploads"
VENDOR_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Registry paths
BASE_DIR = Path(__file__).parent.parent.absolute()
USER_REGISTRY_PATH = BASE_DIR / "data" / "user_data" / "user_registry.json"
VENDOR_REGISTRY_PATH = BASE_DIR / "data" / "vendor_data" / "vendor_registry.json"

GRAPH_API_BASE = "https://graph.facebook.com/v20.0"

# Setup templates for cart/wishlist pages
BACKEND_DIR = Path(__file__).parent
TEMPLATES_DIR = BACKEND_DIR / "templates"
STATIC_DIR = BACKEND_DIR / "static"
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Short ID DB path
SHORTID_DB_PATH = "data/databases/sql/inventory.db"
_shortid_cache = None


# ==================== Helper Functions ====================

def _normalize_phone(phone: str) -> str:
    """Normalize phone number for comparison."""
    return phone.replace("+", "").replace("-", "").replace(" ", "")


def is_registered_vendor(phone: str) -> bool:
    """Check if a phone number is registered as a vendor."""
    try:
        if not VENDOR_REGISTRY_PATH.exists():
            return False
        with open(VENDOR_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        registered = data.get("registered", [])
        normalized = _normalize_phone(phone)
        return normalized in {_normalize_phone(v) for v in registered}
    except Exception as e:
        print(f"Error checking vendor registry: {e}")
        return False


def is_registered_user(phone: str) -> bool:
    """Check if a phone number is registered as a user."""
    try:
        if not USER_REGISTRY_PATH.exists():
            return False
        with open(USER_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        normalized = _normalize_phone(phone)
        return normalized in {_normalize_phone(k) for k in data.keys()}
    except Exception as e:
        print(f"Error checking user registry: {e}")
        return False


def load_shortid_mapping():
    """Load (short_id -> product_id) mappings from SQL."""
    global _shortid_cache
    if _shortid_cache is not None:
        return _shortid_cache

    conn = sqlite3.connect(SHORTID_DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT short_id, product_id FROM product_table")
        rows = cur.fetchall()
        _shortid_cache = {short: pid for (short, pid) in rows if short}
    except Exception as e:
        print(f"Error loading short_id mapping: {e}")
        _shortid_cache = {}
    finally:
        conn.close()

    print(f"Loaded {len(_shortid_cache)} short-id mappings")
    return _shortid_cache


def resolve_short_ids_in_text(text: str) -> str:
    """Replace SHORT IDs with actual product_ids."""
    mapping = load_shortid_mapping()
    if not mapping:
        return text

    pattern = r"\b([A-Z0-9]{4})\b"

    def replacer(match):
        sid = match.group(1)
        return mapping.get(sid, sid)

    return re.sub(pattern, replacer, text.upper())


# ==================== WhatsApp Messaging Functions ====================

def send_whatsapp_text_message(
    to_number: str,
    message: str,
    reply_to_message_id: Optional[str] = None
) -> Dict[str, Any]:
    """Send a text message to WhatsApp user."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("Missing WHATSAPP_TOKEN or WHATSAPP_PHONE_NUMBER_ID")
        return {}

    url = f"{GRAPH_API_BASE}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": True, "body": message}
    }

    if reply_to_message_id:
        payload["context"] = {"message_id": reply_to_message_id}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        print(f"Error sending message: {e}")
        return {}

    if not resp.ok:
        print(f"Error sending message: {resp.status_code} {resp.text}")
    else:
        print(f"Message sent: {resp.text}")

    result = resp.json() if resp.content else {}
    if result.get("messages"):
        result["sent_message_id"] = result["messages"][0].get("id")
    return result


def send_whatsapp_image_message(to_number: str, image_url: str, caption: str = ""):
    """Send an image with caption to WhatsApp user."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID")
        return {}

    url = f"{GRAPH_API_BASE}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "image",
        "image": {"link": image_url, "caption": caption}
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        print(f"Error sending image: {e}")
        return {}

    if not resp.ok:
        print(f"Error sending image: {resp.status_code} {resp.text}")
    else:
        print(f"Image sent: {resp.text}")

    return resp.json() if resp.content else {}


def download_whatsapp_media(media_id: str) -> str:
    """Download media from WhatsApp Cloud API."""
    if not WHATSAPP_TOKEN:
        raise ValueError("WHATSAPP_TOKEN not set")

    media_url_endpoint = f"{GRAPH_API_BASE}/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    try:
        resp = requests.get(media_url_endpoint, headers=headers, timeout=30)
        resp.raise_for_status()
        media_info = resp.json()
        download_url = media_info.get("url")

        if not download_url:
            raise ValueError(f"No download URL in media response: {media_info}")

        media_resp = requests.get(download_url, headers=headers, timeout=60)
        media_resp.raise_for_status()

        content_type = media_resp.headers.get("Content-Type", "image/jpeg")
        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"

        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = VENDOR_IMAGE_DIR / filename

        with open(filepath, "wb") as f:
            f.write(media_resp.content)

        print(f"Downloaded media to: {filepath}")
        return str(filepath)

    except Exception as e:
        print(f"Error downloading media {media_id}: {e}")
        raise


# ==================== Health Check ====================

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "BeeKurse Unified Gateway",
        "port": 8000,
        "endpoints": {
            "webhook": "/webhook",
            "cart": "/cart/{user_id}",
            "wishlist": "/wishlist/{user_id}",
            "cart_page": "/view/cart/{user_id}",
            "wishlist_page": "/view/wishlist/{user_id}",
            "gallery": "/gallery?user={user_id}"
        }
    }


# ==================== WhatsApp Webhook Endpoints ====================

@app.get("/webhook")
async def verify_webhook(request: Request):
    """GET /webhook - Meta verification."""
    params = request.query_params
    mode = params.get("hub.mode")
    challenge = params.get("hub.challenge")
    token = params.get("hub.verify_token")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("Webhook verified successfully.")
        return PlainTextResponse(challenge)

    print("Webhook verification failed.")
    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request):
    """POST /webhook - Handle WhatsApp messages."""
    data = await request.json()

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" not in value:
            return JSONResponse({"status": "no_messages"}, status_code=200)

        message = value["messages"][0]
        from_number = message["from"]
        msg_type = message.get("type", "text")

        incoming_message_id = message.get("id")
        context_message_id = None
        if "context" in message:
            context_message_id = message["context"].get("id")
            print(f"Reply to message: {context_message_id}")

        is_vendor = is_registered_vendor(from_number)
        is_user = is_registered_user(from_number)

        # VENDOR FLOW
        if is_vendor:
            print(f"VENDOR detected: {from_number}")

            raw_user_text = ""
            attachments = []

            if msg_type == "text":
                raw_user_text = message["text"]["body"]
            elif msg_type == "image":
                image_info = message.get("image", {})
                media_id = image_info.get("id")
                caption = image_info.get("caption", "")

                if media_id:
                    try:
                        local_path = download_whatsapp_media(media_id)
                        attachments.append({
                            "type": "image",
                            "path": local_path,
                            "media_id": media_id,
                            "mime_type": image_info.get("mime_type", "image/jpeg")
                        })
                        raw_user_text = caption
                    except Exception as e:
                        print(f"Failed to download vendor image: {e}")
                        send_whatsapp_text_message(from_number, "Failed to process your image. Please try again.")
                        return JSONResponse({"status": "image_download_error"}, status_code=200)
            elif msg_type == "document":
                doc_info = message.get("document", {})
                media_id = doc_info.get("id")
                caption = doc_info.get("caption", "")

                if media_id:
                    try:
                        local_path = download_whatsapp_media(media_id)
                        attachments.append({
                            "type": "document",
                            "path": local_path,
                            "media_id": media_id,
                            "mime_type": doc_info.get("mime_type", "application/pdf"),
                            "filename": doc_info.get("filename", "document")
                        })
                        raw_user_text = caption
                    except Exception as e:
                        print(f"Failed to download vendor document: {e}")
                        send_whatsapp_text_message(from_number, "Failed to process your document. Please try again.")
                        return JSONResponse({"status": "document_download_error"}, status_code=200)
            else:
                send_whatsapp_text_message(from_number, f"Unsupported message type: {msg_type}. Please send text, images, or documents.")
                return JSONResponse({"status": "unsupported_vendor_type"}, status_code=200)

            try:
                backend_resp = requests.post(
                    VENDOR_BACKEND_URL,
                    json={
                        "sender": from_number,
                        "message": raw_user_text,
                        "attachments": attachments,
                        "context_id": context_message_id,
                        "incoming_message_id": incoming_message_id
                    },
                    timeout=120
                )
                backend_resp.raise_for_status()
                backend_data = backend_resp.json()
            except Exception as e:
                print(f"Error calling vendor backend: {e}")
                send_whatsapp_text_message(from_number, "Something went wrong processing your vendor request. Try again later.")
                return JSONResponse({"status": "vendor_backend_error"}, status_code=200)

            if "messages" in backend_data:
                multi_tracking = backend_data.get("multi_product_tracking", False)
                message_mappings = {}

                for msg in backend_data["messages"]:
                    if msg.get("type") == "image":
                        send_whatsapp_image_message(from_number, msg["url"], caption="")
                    elif msg.get("type") == "text":
                        result = send_whatsapp_text_message(from_number, msg["text"])

                        if multi_tracking and msg.get("requires_reply_tracking"):
                            sent_wamid = result.get("sent_message_id")
                            product_index = msg.get("product_index")
                            if sent_wamid and product_index is not None:
                                message_mappings[sent_wamid] = product_index

                if message_mappings:
                    try:
                        requests.post(
                            f"{VENDOR_BACKEND_URL}/register-message-ids",
                            json={"user_id": from_number, "mappings": message_mappings},
                            timeout=10
                        )
                    except Exception as e:
                        print(f"Failed to register message mappings: {e}")

            elif "reply" in backend_data:
                send_whatsapp_text_message(from_number, backend_data["reply"])

            return JSONResponse({"status": "vendor_handled"}, status_code=200)

        # USER FLOW
        if msg_type != "text":
            send_whatsapp_text_message(from_number, "Sorry, I only understand text messages right now.")
            return JSONResponse({"status": "unsupported_type"}, status_code=200)

        raw_user_text = message["text"]["body"]
        print(f"Incoming message from {from_number}: {raw_user_text}")

        if not is_user:
            print(f"New user detected: {from_number} - Starting user onboarding")

        user_manager = get_user_manager()
        onboarding_response = user_manager.process_message(from_number, raw_user_text)

        if onboarding_response is not None:
            print(f"Onboarding response to {from_number}")
            send_whatsapp_text_message(from_number, onboarding_response)
            return JSONResponse({"status": "onboarding"}, status_code=200)

        cart_manager = get_cart_manager()
        cart_response = cart_manager.handle_command(from_number, raw_user_text)

        if cart_response is not None:
            print(f"Cart command handled for {from_number}")
            send_whatsapp_text_message(from_number, cart_response)
            return JSONResponse({"status": "cart_handled"}, status_code=200)

        user_text = resolve_short_ids_in_text(raw_user_text)

        print(f"Routing to USER backend for {from_number}")
        try:
            backend_resp = requests.post(
                BACKEND_URL,
                json={
                    "sender": from_number,
                    "message": user_text,
                    "context_message_id": context_message_id,
                    "incoming_message_id": incoming_message_id
                },
                timeout=30
            )
            backend_resp.raise_for_status()
            backend_data = backend_resp.json()
        except Exception as e:
            print(f"Error calling backend: {e}")
            send_whatsapp_text_message(from_number, "Some error happened in my brain (backend). Try again later.")
            return JSONResponse({"status": "backend_error"}, status_code=200)

        message_history = get_message_history()

        if "text" in backend_data or "images" in backend_data:
            text = backend_data.get("text")
            gallery_link = backend_data.get("gallery_link")
            images = backend_data.get("images", [])
            product_ids = backend_data.get("product_ids", [])

            if images:
                for i, img in enumerate(images):
                    result = send_whatsapp_image_message(from_number, img["url"], caption=img.get("caption", ""))
                    if result.get("messages") and i < len(product_ids):
                        sent_msg_id = result["messages"][0].get("id")
                        if sent_msg_id:
                            message_history.register_message(from_number, sent_msg_id, product_ids[i])

                time.sleep(0.5)

            if text:
                send_whatsapp_text_message(from_number, text)
            if gallery_link:
                send_whatsapp_text_message(from_number, gallery_link)

        elif "messages" in backend_data:
            product_id = backend_data.get("product_id")
            short_id = backend_data.get("short_id")

            for msg in backend_data["messages"]:
                if msg["type"] == "image":
                    result = send_whatsapp_image_message(from_number, msg["url"], caption="")
                    if result.get("messages") and product_id:
                        sent_msg_id = result["messages"][0].get("id")
                        if sent_msg_id:
                            message_history.register_message(from_number, sent_msg_id, product_id, short_id)
                elif msg["type"] == "text":
                    result = send_whatsapp_text_message(from_number, msg["text"])
                    if result.get("sent_message_id") and product_id:
                        message_history.register_message(from_number, result["sent_message_id"], product_id, short_id)

            if product_id:
                cart_manager.set_last_viewed(from_number, product_id, short_id)

        elif "reply" in backend_data:
            reply_text = backend_data["reply"]
            print(f"Replying to {from_number}: {reply_text}")
            send_whatsapp_text_message(from_number, reply_text)

        else:
            send_whatsapp_text_message(from_number, "I couldn't understand the backend response.")

        return JSONResponse({"status": "ok"}, status_code=200)

    except Exception as e:
        print(f"Error handling webhook: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# ==================== Cart API Endpoints ====================

class AddItemRequest(BaseModel):
    product_id: str
    short_id: Optional[str] = None
    quantity: Optional[int] = 1


class RemoveItemRequest(BaseModel):
    product_id: str


@app.get("/cart/{user_id}")
def get_cart(user_id: str):
    """Get user's cart items as JSON"""
    manager = get_cart_manager()
    cart = manager.get_cart_with_products(user_id)
    return {"user_id": user_id, "cart": cart, "count": len(cart)}


@app.get("/wishlist/{user_id}")
def get_wishlist(user_id: str):
    """Get user's wishlist items as JSON"""
    manager = get_cart_manager()
    wishlist = manager.get_wishlist_with_products(user_id)
    return {"user_id": user_id, "wishlist": wishlist, "count": len(wishlist)}


@app.post("/cart/{user_id}/add")
def add_to_cart(user_id: str, item: AddItemRequest):
    """Add item to cart"""
    manager = get_cart_manager()
    success, status = manager.add_to_cart(user_id, item.product_id, item.short_id, item.quantity or 1)
    return {"success": success, "status": status, "message": "Item added to cart" if success else "Failed to add item"}


@app.post("/cart/{user_id}/remove")
def remove_from_cart(user_id: str, item: RemoveItemRequest):
    """Remove item from cart"""
    manager = get_cart_manager()
    success, status = manager.remove_from_cart(user_id, item.product_id)
    return {"success": success, "status": status, "message": "Item removed from cart" if success else "Item not found in cart"}


@app.post("/wishlist/{user_id}/add")
def add_to_wishlist(user_id: str, item: AddItemRequest):
    """Add item to wishlist"""
    manager = get_cart_manager()
    success, status = manager.add_to_wishlist(user_id, item.product_id, item.short_id)
    return {"success": success, "status": status, "message": "Item added to wishlist" if success else "Failed to add item"}


@app.post("/wishlist/{user_id}/remove")
def remove_from_wishlist(user_id: str, item: RemoveItemRequest):
    """Remove item from wishlist"""
    manager = get_cart_manager()
    success, status = manager.remove_from_wishlist(user_id, item.product_id)
    return {"success": success, "status": status, "message": "Item removed from wishlist" if success else "Item not found in wishlist"}


# ==================== Cart/Wishlist Web Pages ====================

@app.get("/view/cart/{user_id}", response_class=HTMLResponse)
async def view_cart_page(request: Request, user_id: str):
    """Render cart webpage"""
    manager = get_cart_manager()
    cart = manager.get_cart_with_products(user_id)
    return templates.TemplateResponse("cart.html", {
        "request": request,
        "user_id": user_id,
        "items": cart,
        "count": len(cart),
        "whatsapp_number": WHATSAPP_BUSINESS_NUMBER,
        "page_type": "cart"
    })


@app.get("/view/wishlist/{user_id}", response_class=HTMLResponse)
async def view_wishlist_page(request: Request, user_id: str):
    """Render wishlist webpage"""
    manager = get_cart_manager()
    wishlist = manager.get_wishlist_with_products(user_id)
    return templates.TemplateResponse("wishlist.html", {
        "request": request,
        "user_id": user_id,
        "items": wishlist,
        "count": len(wishlist),
        "whatsapp_number": WHATSAPP_BUSINESS_NUMBER,
        "page_type": "wishlist"
    })


# ==================== Images API (Proxy to Strontium) ====================

STRONTIUM_API_URL = os.getenv("STRONTIUM_API_URL", "http://localhost:5001")

@app.get("/images/{user_id}")
async def get_images(user_id: str):
    """Proxy /images/{user_id} to Strontium API for gallery frontend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{STRONTIUM_API_URL}/images/{user_id}",
                timeout=30.0
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
    except Exception as e:
        print(f"Images proxy error: {e}")
        return JSONResponse(
            content={"error": "Could not fetch images from backend", "products": []},
            status_code=500
        )


# ==================== Gallery Proxy ====================

@app.api_route("/gallery", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
@app.api_route("/gallery/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_gallery(request: Request, path: str = ""):
    """Proxy requests to the React gallery frontend (port 5400)"""
    # Vite dev server serves at /gallery/ due to base: '/gallery/' config
    # So we need to forward /gallery/... to http://localhost:5400/gallery/...
    target_url = f"{GALLERY_FRONTEND_URL}/gallery/{path}" if path else f"{GALLERY_FRONTEND_URL}/gallery/"

    # Preserve query parameters
    if request.query_params:
        target_url = f"{target_url}?{request.query_params}"

    async with httpx.AsyncClient() as client:
        try:
            # Forward the request
            response = await client.request(
                method=request.method,
                url=target_url,
                headers={k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length']},
                content=await request.body() if request.method in ["POST", "PUT", "PATCH"] else None,
                timeout=30.0
            )

            # Return proxied response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type")
            )
        except Exception as e:
            print(f"Gallery proxy error: {e}")
            return HTMLResponse(
                content=f"<h1>Gallery Unavailable</h1><p>The gallery frontend is not running on port 5400.</p><p>Start it with: cd whatsapp-frontend && npm run dev -- --port 5400</p>",
                status_code=503
            )


# ==================== Admin Endpoints ====================

@app.post("/admin/set-ngrok-url")
def set_ngrok_url_endpoint(url: str):
    """Set the ngrok URL for generating links"""
    set_base_url(url)
    return {"status": "ok", "url": url}


# ==================== Development Server ====================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("  BeeKurse Unified Gateway")
    print("=" * 70)
    print(f"  Webhook:        http://localhost:8000/webhook")
    print(f"  Cart API:       http://localhost:8000/cart/{{user_id}}")
    print(f"  Wishlist API:   http://localhost:8000/wishlist/{{user_id}}")
    print(f"  Cart Page:      http://localhost:8000/view/cart/{{user_id}}")
    print(f"  Wishlist Page:  http://localhost:8000/view/wishlist/{{user_id}}")
    print(f"  Gallery:        http://localhost:8000/gallery?user={{user_id}}")
    print("=" * 70)
    print("  One ngrok tunnel exposes all services!")
    print("  Usage: ngrok http 8000 --url YOUR_URL")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
