# whatsapp_bot.py
import os
import json
import requests
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from dotenv import load_dotenv

# User management
from .user_manager import get_user_manager

# Cart management
from .cart_manager import get_cart_manager

# Load variables from .env file
load_dotenv()

app = FastAPI()

# Environment variables
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")                  # Your Meta access token
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")       # e.g. "123456789012345"
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_verify_token_123")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5001/process")
VENDOR_BACKEND_URL = os.getenv("VENDOR_BACKEND_URL", "http://localhost:5001/vendor/process")

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


def send_whatsapp_text_message(to_number: str, message: str):
    """
    Send a text message to a WhatsApp user using the Cloud API.
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

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        print("❌ Error sending message to WhatsApp:", e)
        return {}

    if not resp.ok:
        print("❌ Error sending message to WhatsApp:", resp.status_code, resp.text)
    else:
        print("✅ Message sent to WhatsApp:", resp.text)

    return resp.json() if resp.content else {}

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

        # Only handle text messages
        if msg_type != "text":
            send_whatsapp_text_message(
                from_number,
                "Sorry, I only understand text messages right now."
            )
            return JSONResponse({"status": "unsupported_type"}, status_code=200)

        # Get raw user text (before short ID resolution)
        raw_user_text = message["text"]["body"]
        print(f"📩 Incoming message from {from_number}: {raw_user_text}")

        # ===== Determine User Type: Vendor, User, or New User =====
        is_vendor = is_registered_vendor(from_number)
        is_user = is_registered_user(from_number)

        # ===== VENDOR FLOW =====
        if is_vendor:
            print(f"🏪 Routing to VENDOR backend for {from_number}")
            try:
                backend_resp = requests.post(
                    VENDOR_BACKEND_URL,
                    json={
                        "sender": from_number,
                        "message": raw_user_text,
                        "attachments": []
                    },
                    timeout=30
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
                for msg in backend_data["messages"]:
                    if msg["type"] == "image":
                        send_whatsapp_image_message(from_number, msg["url"], caption="")
                    elif msg["type"] == "text":
                        send_whatsapp_text_message(from_number, msg["text"])
            elif "reply" in backend_data:
                send_whatsapp_text_message(from_number, backend_data["reply"])

            return JSONResponse({"status": "vendor_handled"}, status_code=200)
        # ===== END VENDOR FLOW =====

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
                    "sender": from_number,   # <- IMPORTANT: matches backend_strontium.py
                    "message": user_text
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

        # Format 1: Search response - {"text": str, "images": list}
        if "text" in backend_data or "images" in backend_data:
            text = backend_data.get("text")
            images = backend_data.get("images", [])

            # 1. Send images first (each image is a separate WhatsApp message)
            print("ALIVE")
            if images:
                print("DEAD")
                for img in images:
                    send_whatsapp_image_message(
                        from_number,
                        img["url"],
                        caption=img.get("caption", "")
                    )

            # 2. Then send summary text
            if text:
                send_whatsapp_text_message(from_number, text)

        # Format 2: Detail response - {"messages": [{"type": "image"|"text", ...}]}
        elif "messages" in backend_data:
            for msg in backend_data["messages"]:
                if msg["type"] == "image":
                    send_whatsapp_image_message(
                        from_number,
                        msg["url"],
                        caption=""
                    )
                elif msg["type"] == "text":
                    send_whatsapp_text_message(from_number, msg["text"])

            # Track last viewed product for "add to cart" command
            if "product_id" in backend_data:
                cart_manager.set_last_viewed(
                    from_number,
                    backend_data["product_id"],
                    backend_data.get("short_id")
                )

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
