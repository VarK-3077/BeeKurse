# whatsapp_bot.py
import os
import json
import requests
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from dotenv import load_dotenv

# User management
from .user_manager import get_user_manager

# Cart management
from .cart_manager import get_cart_manager

# Message history for contextual replies
from .message_history import get_message_history

# Load variables from .env file
load_dotenv()

app = FastAPI()

# Environment variables
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")                  # Your Meta access token
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")       # e.g. "123456789012345"
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_verify_token_123")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5001/process")
VENDOR_BACKEND_URL = os.getenv("VENDOR_BACKEND_URL", "http://localhost:5001/vendor/process")

# Directory for downloaded vendor images
VENDOR_IMAGE_DIR = Path(__file__).parent.parent / "data" / "vendor_data" / "uploads"
VENDOR_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Registry paths
BASE_DIR = Path(__file__).parent.parent.absolute()
USER_REGISTRY_PATH = BASE_DIR / "data" / "user_data" / "user_registry.json"
VENDOR_REGISTRY_PATH = BASE_DIR / "data" / "vendor_data" / "vendor_registry.json"


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
        print(f"❌ Error checking vendor registry: {e}")
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
        print(f"❌ Error checking user registry: {e}")
        return False

GRAPH_API_BASE = "https://graph.facebook.com/v20.0"


def send_whatsapp_text_message(
    to_number: str,
    message: str,
    reply_to_message_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send a text message to a WhatsApp user using the Cloud API.

    Args:
        to_number: Recipient phone number
        message: Message text
        reply_to_message_id: Optional - ID of message to reply to (for threading)

    Returns:
        Dict with response data, including 'sent_message_id' on success
    """
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("❌ Missing WHATSAPP_TOKEN or WHATSAPP_PHONE_NUMBER_ID in environment.")
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
        "text": {
            "preview_url": True,
            "body": message
        }
    }

    # Add context for reply-to threading
    if reply_to_message_id:
        payload["context"] = {"message_id": reply_to_message_id}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        print("❌ Error sending message to WhatsApp:", e)
        return {}

    if not resp.ok:
        print("❌ Error sending message to WhatsApp:", resp.status_code, resp.text)
    else:
        print("✅ Message sent to WhatsApp:", resp.text)

    result = resp.json() if resp.content else {}

    # Extract and return sent message ID for tracking
    if result.get("messages"):
        result["sent_message_id"] = result["messages"][0].get("id")

    return result

# -------------------------------------- Image Add ------------------------------------
def send_whatsapp_image_message(to_number: str, image_url: str, caption: str = ""):
    """
    Send an image with an optional caption to WhatsApp user.
    """

    print("Does this happend??")

    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("❌ Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID")
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
        "image": {
            "link": image_url,
            "caption": caption
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        print("❌ Error sending image:", e)
        return {}

    if not resp.ok:
        print("❌ Error sending image:", resp.status_code, resp.text)
    else:
        print("📸 Image sent to WhatsApp:", resp.text)

    return resp.json() if resp.content else {}

# ----------------------------------------------------------------------------------


# ---------------------------------- Download WhatsApp Media ---------------------------
def download_whatsapp_media(media_id: str) -> str:
    """
    Download media from WhatsApp Cloud API.
    Returns the local file path where the image was saved.
    """
    if not WHATSAPP_TOKEN:
        raise ValueError("WHATSAPP_TOKEN not set")

    # Step 1: Get media URL from media ID
    media_url_endpoint = f"{GRAPH_API_BASE}/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    try:
        resp = requests.get(media_url_endpoint, headers=headers, timeout=30)
        resp.raise_for_status()
        media_info = resp.json()
        download_url = media_info.get("url")

        if not download_url:
            raise ValueError(f"No download URL in media response: {media_info}")

        # Step 2: Download the actual media file
        media_resp = requests.get(download_url, headers=headers, timeout=60)
        media_resp.raise_for_status()

        # Determine file extension from content type
        content_type = media_resp.headers.get("Content-Type", "image/jpeg")
        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"

        # Save to vendor uploads directory
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = VENDOR_IMAGE_DIR / filename

        with open(filepath, "wb") as f:
            f.write(media_resp.content)

        print(f"📥 Downloaded media to: {filepath}")
        return str(filepath)

    except Exception as e:
        print(f"❌ Error downloading media {media_id}: {e}")
        raise


# ----------------------------------------------------------------------------------


@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    GET /webhook — Meta uses this once to verify your webhook.

    Meta sends query params:
      - hub.mode
      - hub.challenge
      - hub.verify_token
    """
    params = request.query_params
    mode = params.get("hub.mode")
    challenge = params.get("hub.challenge")
    token = params.get("hub.verify_token")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Webhook verified successfully.")
        return PlainTextResponse(challenge)

    print("❌ Webhook verification failed.")
    return PlainTextResponse("Verification failed", status_code=403)


# ---------------------------------- Short ID conversion --------------------------------

import sqlite3
import re

SHORTID_DB_PATH = "data/databases/sql/inventory.db"

_shortid_cache = None

def load_shortid_mapping():
    """
    Loads (short_id -> product_id) mappings from SQL.
    Cached after first load.
    """
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
        print("❌ Error loading short_id mapping:", e)
        _shortid_cache = {}
    finally:
        conn.close()

    print(f"🔑 Loaded {len(_shortid_cache)} short-id mappings")
    return _shortid_cache


def resolve_short_ids_in_text(text: str) -> str:
    """
    Replaces any SHORT IDs (format: A1B2, X9FQ, etc.) in the user's query
    with the actual product_id so Strontium can understand it.
    """

    mapping = load_shortid_mapping()

    if not mapping:
        return text  # nothing to replace

    # SHORT ID format: 4 alphanumeric characters (uppercase)
    pattern = r"\b([A-Z0-9]{4})\b"

    def replacer(match):
        sid = match.group(1)
        return mapping.get(sid, sid)

    resolved = re.sub(pattern, replacer, text.upper())
    return resolved


# ---------------------------------------------------------------------------------------


@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    POST /webhook — called by Meta when a user sends a WhatsApp message.
    Steps:
      1. Extract text + sender number from WhatsApp payload
      2. Send it to your backend (BACKEND_URL)
      3. Backend returns JSON: { "reply": "..." }
      4. Send that reply back to the user via WhatsApp Cloud API
    """
    data = await request.json()
    # Uncomment to debug full payload
    # print("Incoming webhook payload:", data)

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        # Ignore events without messages (e.g. status updates)
        if "messages" not in value:
            return JSONResponse({"status": "no_messages"}, status_code=200)

        message = value["messages"][0]
        from_number = message["from"]                 # The user's WhatsApp number
        msg_type = message.get("type", "text")

        # ===== Extract Message Context for Reply Tracking =====
        incoming_message_id = message.get("id")  # This message's WhatsApp ID
        context_message_id = None
        if "context" in message:
            # User quoted/replied to a previous message
            context_message_id = message["context"].get("id")
            print(f"📎 Reply to message: {context_message_id}")

        # ===== Determine User Type: Vendor, User, or New User =====
        is_vendor = is_registered_vendor(from_number)
        is_user = is_registered_user(from_number)

        # ===== VENDOR FLOW (handles both text and images) =====
        if is_vendor:
            print(f"🏪 VENDOR detected: {from_number}")

            raw_user_text = ""
            attachments = []

            # Handle different message types for vendors
            if msg_type == "text":
                raw_user_text = message["text"]["body"]
                print(f"📩 Vendor text message: {raw_user_text}")

            elif msg_type == "image":
                # Download the image from WhatsApp
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
                        raw_user_text = caption  # Use caption as text if provided
                        print(f"📸 Vendor image received, saved to: {local_path}")
                    except Exception as e:
                        print(f"❌ Failed to download vendor image: {e}")
                        send_whatsapp_text_message(
                            from_number,
                            "Failed to process your image. Please try again."
                        )
                        return JSONResponse({"status": "image_download_error"}, status_code=200)

            elif msg_type == "document":
                # Handle document (PDF, etc.)
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
                        print(f"📄 Vendor document received, saved to: {local_path}")
                    except Exception as e:
                        print(f"❌ Failed to download vendor document: {e}")
                        send_whatsapp_text_message(
                            from_number,
                            "Failed to process your document. Please try again."
                        )
                        return JSONResponse({"status": "document_download_error"}, status_code=200)

            else:
                send_whatsapp_text_message(
                    from_number,
                    f"Unsupported message type: {msg_type}. Please send text, images, or documents."
                )
                return JSONResponse({"status": "unsupported_vendor_type"}, status_code=200)

            # Route to vendor backend with attachments and context
            print(f"🏪 Routing to VENDOR backend for {from_number}")
            try:
                backend_resp = requests.post(
                    VENDOR_BACKEND_URL,
                    json={
                        "sender": from_number,
                        "message": raw_user_text,
                        "attachments": attachments,
                        "context_id": context_message_id,  # ID of message being replied to
                        "incoming_message_id": incoming_message_id  # This message's ID
                    },
                    timeout=120  # Longer timeout for OCR processing
                )
                backend_resp.raise_for_status()
                backend_data = backend_resp.json()
            except Exception as e:
                print("❌ Error calling vendor backend:", e)
                send_whatsapp_text_message(
                    from_number,
                    "Something went wrong processing your vendor request. Try again later."
                )
                return JSONResponse({"status": "vendor_backend_error"}, status_code=200)

            # Handle vendor response (usually {"messages": [...]})
            if "messages" in backend_data:
                multi_tracking = backend_data.get("multi_product_tracking", False)
                message_mappings = {}

                for msg in backend_data["messages"]:
                    if msg.get("type") == "image":
                        send_whatsapp_image_message(from_number, msg["url"], caption="")
                    elif msg.get("type") == "text":
                        result = send_whatsapp_text_message(from_number, msg["text"])

                        # Track message IDs for multi-product reply handling
                        if multi_tracking and msg.get("requires_reply_tracking"):
                            sent_wamid = result.get("sent_message_id")
                            product_index = msg.get("product_index")
                            if sent_wamid and product_index is not None:
                                message_mappings[sent_wamid] = product_index
                                print(f"📍 Mapped message {sent_wamid[:20]}... -> product {product_index}")

                # Register message ID mappings with vendor backend
                if message_mappings:
                    try:
                        requests.post(
                            f"{VENDOR_BACKEND_URL}/register-message-ids",
                            json={"user_id": from_number, "mappings": message_mappings},
                            timeout=10
                        )
                        print(f"📍 Registered {len(message_mappings)} message mappings")
                    except Exception as e:
                        print(f"⚠️ Failed to register message mappings: {e}")

            elif "reply" in backend_data:
                send_whatsapp_text_message(from_number, backend_data["reply"])

            return JSONResponse({"status": "vendor_handled"}, status_code=200)
        # ===== END VENDOR FLOW =====

        # ===== USER FLOW - Only text messages =====
        if msg_type != "text":
            send_whatsapp_text_message(
                from_number,
                "Sorry, I only understand text messages right now."
            )
            return JSONResponse({"status": "unsupported_type"}, status_code=200)

        # Get raw user text (before short ID resolution)
        raw_user_text = message["text"]["body"]
        print(f"📩 Incoming message from {from_number}: {raw_user_text}")

        # ===== USER FLOW (existing user or new user - new users default to user onboarding) =====
        # Note: If not a vendor and not an existing user, they're a new user (default)
        if not is_user:
            print(f"🆕 New user detected: {from_number} - Starting user onboarding")

        # ===== User Management Check =====
        user_manager = get_user_manager()
        onboarding_response = user_manager.process_message(from_number, raw_user_text)

        if onboarding_response is not None:
            # User is in onboarding flow - send onboarding message
            print(f"🆕 Onboarding response to {from_number}")
            send_whatsapp_text_message(from_number, onboarding_response)
            return JSONResponse({"status": "onboarding"}, status_code=200)
        # ===== End User Management Check =====

        # ===== Cart/Wishlist Command Check =====
        cart_manager = get_cart_manager()
        cart_response = cart_manager.handle_command(from_number, raw_user_text)

        if cart_response is not None:
            # Cart/wishlist command handled
            print(f"🛒 Cart command handled for {from_number}")
            send_whatsapp_text_message(from_number, cart_response)
            return JSONResponse({"status": "cart_handled"}, status_code=200)
        # ===== End Cart/Wishlist Command Check =====

        # Resolve short IDs for search queries
        user_text = resolve_short_ids_in_text(raw_user_text)

        # ===== Call USER backend =====
        print(f"👤 Routing to USER backend for {from_number}")
        try:
            backend_resp = requests.post(
                BACKEND_URL,
                json={
                    "sender": from_number,
                    "message": user_text,
                    "context_message_id": context_message_id,  # ID of message being replied to
                    "incoming_message_id": incoming_message_id  # This message's ID
                },
                timeout=30
            )
            backend_resp.raise_for_status()
            backend_data = backend_resp.json()
        except Exception as e:
            print("❌ Error calling backend:", e)
            send_whatsapp_text_message(
                from_number,
                "Some error happened in my brain (backend). Try again later."
            )
            return JSONResponse({"status": "backend_error"}, status_code=200)

        # ------------------------------- Handle Multiple Response Formats --------------------------------

        print(backend_data)

        # Get message history manager for tracking
        message_history = get_message_history()

        # Format 1: Search response - {"text": str, "images": list, "product_ids": list}
        if "text" in backend_data or "images" in backend_data:
            text = backend_data.get("text")
            gallery_link = backend_data.get("gallery_link")
            images = backend_data.get("images", [])
            product_ids = backend_data.get("product_ids", [])

            # 1. Send images first and track message IDs
            if images:
                for i, img in enumerate(images):
                    result = send_whatsapp_image_message(
                        from_number,
                        img["url"],
                        caption=img.get("caption", "")
                    )
                    # Track sent message ID -> product ID
                    if result.get("messages") and i < len(product_ids):
                        sent_msg_id = result["messages"][0].get("id")
                        if sent_msg_id:
                            message_history.register_message(
                                from_number, sent_msg_id, product_ids[i]
                            )

            # 2. Then send summary text
            if text:
                send_whatsapp_text_message(from_number, text)
            if gallery_link:
                send_whatsapp_text_message(from_number, gallery_link)

        # Format 2: Detail response - {"messages": [{"type": "image"|"text", ...}]}
        elif "messages" in backend_data:
            product_id = backend_data.get("product_id")
            short_id = backend_data.get("short_id")

            for msg in backend_data["messages"]:
                if msg["type"] == "image":
                    result = send_whatsapp_image_message(
                        from_number,
                        msg["url"],
                        caption=""
                    )
                    # Track image message for this product
                    if result.get("messages") and product_id:
                        sent_msg_id = result["messages"][0].get("id")
                        if sent_msg_id:
                            message_history.register_message(
                                from_number, sent_msg_id, product_id, short_id
                            )
                elif msg["type"] == "text":
                    result = send_whatsapp_text_message(from_number, msg["text"])
                    # Also track text message for this product
                    if result.get("sent_message_id") and product_id:
                        message_history.register_message(
                            from_number, result["sent_message_id"], product_id, short_id
                        )

            # Track last viewed product for "add to cart" command
            if product_id:
                cart_manager.set_last_viewed(from_number, product_id, short_id)

        # Format 3: Chat response - {"reply": str}
        elif "reply" in backend_data:
            reply_text = backend_data["reply"]
            print(f"💬 Replying to {from_number}: {reply_text}")
            send_whatsapp_text_message(from_number, reply_text)

        else:
            # Fallback
            send_whatsapp_text_message(
                from_number,
                "I couldn't understand the backend response."
            )

        # ---------------------------------------------------------------------------

        return JSONResponse({"status": "ok"}, status_code=200)

    except Exception as e:
        print("❌ Error handling webhook:", e)
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 Starting WhatsApp Webhook")
    print("="*80)
    print(f"📍 Webhook endpoint: http://0.0.0.0:8000/webhook")
    print(f"🔑 Verify token: {VERIFY_TOKEN}")
    print(f"👤 User Backend URL: {BACKEND_URL}")
    print(f"🏪 Vendor Backend URL: {VENDOR_BACKEND_URL}")
    print(f"📋 User Registry: {USER_REGISTRY_PATH}")
    print(f"📋 Vendor Registry: {VENDOR_REGISTRY_PATH}")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
