# terminal_bot.py
"""
Terminal-based WhatsApp Bot Simulator
Takes input from terminal instead of webhook, outputs to terminal instead of WhatsApp API.
Supports both user and vendor flows.
Includes OCR integration for vendor inventory lists (handwritten or printed).
"""
import os
import sys
import json
import requests
import uuid
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# OCR imports (optional - only loaded when needed)
_ocr_available = False
try:
    from ocr import (
        OLMOCRProcessor,
        extract_text_from_image,
        preprocess_image_for_ocr
    )
    _ocr_available = True
except ImportError as e:
    print(f"[WARNING] OCR module not available: {e}")
    print("[WARNING] Vendor image OCR processing will be disabled.")

# User management
from backend.user_manager import get_user_manager

# Cart management
from backend.cart_manager import get_cart_manager

# Message history for contextual replies
from backend.message_history import get_message_history

# Load variables from .env file
load_dotenv()

# Environment variables
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5001/process")
VENDOR_BACKEND_URL = os.getenv("VENDOR_BACKEND_URL", "http://localhost:5001/vendor/process")

# Registry paths
BASE_DIR = Path(__file__).parent.parent.absolute()
USER_REGISTRY_PATH = BASE_DIR / "data" / "user_data" / "user_registry.json"
VENDOR_REGISTRY_PATH = BASE_DIR / "data" / "vendor_data" / "vendor_registry.json"

# Track last used context for simulating replies
_last_message_id: Optional[str] = None
_last_phone: Optional[str] = None

# Pending inventory update (for OCR confirmation flow)
_pending_inventory_update: Optional[Dict[str, Any]] = None


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
        print(f"[ERROR] Error checking vendor registry: {e}")
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
        print(f"[ERROR] Error checking user registry: {e}")
        return False


# ========================= Terminal Output Functions =========================

def send_text_message(
    to_number: str,
    message: str,
    reply_to_message_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Print a text message to terminal (simulates WhatsApp text message send).
    """
    global _last_message_id

    sent_id = f"TERM_MSG_{uuid.uuid4().hex[:8]}"
    _last_message_id = sent_id

    print(f"\n{'='*60}")
    print(f"[TEXT -> {to_number}]")
    if reply_to_message_id:
        print(f"(Reply to: {reply_to_message_id})")
    print("-" * 60)
    print(message)
    print("=" * 60)

    return {"sent_message_id": sent_id, "messages": [{"id": sent_id}]}


def send_image_message(
    to_number: str,
    image_url: str,
    caption: str = ""
) -> Dict[str, Any]:
    """
    Print an image message to terminal (simulates WhatsApp image message send).
    """
    global _last_message_id

    sent_id = f"TERM_IMG_{uuid.uuid4().hex[:8]}"
    _last_message_id = sent_id

    print(f"\n{'='*60}")
    print(f"[IMAGE -> {to_number}]")
    print("-" * 60)
    print(f"  URL: {image_url}")
    if caption:
        print(f"  Caption: {caption}")
    print("=" * 60)

    return {"messages": [{"id": sent_id}]}


# ========================= Short ID Resolution =========================

import sqlite3
import re

SHORTID_DB_PATH = BASE_DIR / "data" / "databases" / "sql" / "inventory.db"
_shortid_cache = None


def load_shortid_mapping():
    """Loads (short_id -> product_id) mappings from SQL. Cached after first load."""
    global _shortid_cache
    if _shortid_cache is not None:
        return _shortid_cache

    try:
        conn = sqlite3.connect(str(SHORTID_DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT short_id, product_id FROM product_table")
        rows = cur.fetchall()
        _shortid_cache = {short: pid for (short, pid) in rows if short}
        conn.close()
        print(f"[INFO] Loaded {len(_shortid_cache)} short-id mappings")
    except Exception as e:
        print(f"[ERROR] Error loading short_id mapping: {e}")
        _shortid_cache = {}

    return _shortid_cache


def resolve_short_ids_in_text(text: str) -> str:
    """
    Replaces any SHORT IDs (format: A1B2, X9FQ, etc.) in the user's query
    with the actual product_id so Strontium can understand it.
    """
    mapping = load_shortid_mapping()
    if not mapping:
        return text

    pattern = r"\b([A-Z0-9]{4})\b"

    def replacer(match):
        sid = match.group(1)
        return mapping.get(sid, sid)

    resolved = re.sub(pattern, replacer, text.upper())
    return resolved


# ========================= OCR Inventory Processing =========================

def parse_inventory_list_text(ocr_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse OCR-extracted text from a vendor inventory list.

    Expects format like:
        Update
        Maggi, 10₹/pack, 50pcs
        Sugar, 15₹/kg, 60kgs
        ...
        Add
        Dairymilk, 80₹/pack, 15 packets
        ...

    Returns:
        {
            "update": [{"name": str, "price": float, "unit": str, "stock": float, "stock_unit": str}, ...],
            "add": [{"name": str, "price": float, "unit": str, "stock": float, "stock_unit": str}, ...]
        }
    """
    result = {"update": [], "add": []}
    current_section = None

    # Clean and split lines
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]

    for line in lines:
        line_lower = line.lower()

        # Detect section headers
        if line_lower in ['update', 'update.', 'update:']:
            current_section = "update"
            continue
        elif line_lower in ['add', 'add.', 'add:']:
            current_section = "add"
            continue

        # Skip if no section detected yet
        if current_section is None:
            # Check if line starts with update/add
            if line_lower.startswith('update'):
                current_section = "update"
                continue
            elif line_lower.startswith('add'):
                current_section = "add"
                continue
            continue

        # Parse item line
        item = parse_inventory_item(line)
        if item:
            result[current_section].append(item)

    return result


def parse_inventory_item(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single inventory line like:
        "Maggi, 10₹/pack, 50pcs"
        "Sugar, 15₹/kg, 60kgs"
        "Rice, 48₹/kg, 100Kgs"
        "Milkybar, 10₹/pcs, 100 chocolates"
        "Salt, ₹10/0.5kg, 70kg"

    Returns:
        {
            "name": "Maggi",
            "price": 10.0,
            "unit": "pack",
            "stock": 50.0,
            "stock_unit": "pcs"
        }
    """
    # Clean the line
    line = line.strip()
    if not line:
        return None

    # Try to split by comma first
    parts = [p.strip() for p in line.split(',')]

    if len(parts) < 2:
        # Try splitting by spaces/tabs
        parts = line.split()
        if len(parts) < 2:
            return None

    # Extract name (first part)
    name = parts[0].strip()

    # Initialize defaults
    price = 0.0
    unit = "piece"
    stock = 0.0
    stock_unit = "pcs"

    # Parse price and unit from remaining parts
    for part in parts[1:]:
        part = part.strip()

        # Check for price pattern: "10₹/pack", "₹10/kg", "15₹/kg", "₹10/0.5kg"
        price_match = re.search(
            r'(?:₹|Rs\.?|rs\.?)?\s*(\d+(?:\.\d+)?)\s*(?:₹|Rs\.?|rs\.?)?\s*/\s*(\d*\.?\d*)\s*(\w+)',
            part, re.IGNORECASE
        )
        if price_match:
            price = float(price_match.group(1))
            # Handle cases like "10/0.5kg" where unit has a multiplier
            unit_multiplier = price_match.group(2)
            unit = price_match.group(3).lower()
            if unit_multiplier and float(unit_multiplier) > 0:
                # e.g., "10/0.5kg" means 10 per 0.5kg
                unit = f"{unit_multiplier}{unit}"
            continue

        # Check for stock pattern: "50pcs", "60kgs", "100 chocolates", "21 pcs"
        stock_match = re.search(r'(\d+(?:\.\d+)?)\s*(\w+)', part)
        if stock_match:
            stock = float(stock_match.group(1))
            stock_unit = stock_match.group(2).lower()
            continue

    # Only return if we have at least a name
    if name:
        return {
            "name": name,
            "price": price,
            "unit": unit,
            "stock": stock,
            "stock_unit": stock_unit
        }

    return None


def apply_inventory_changes(vendor_phone: str, parsed_items: Dict[str, List[Dict]]) -> Tuple[bool, str]:
    """
    Apply parsed inventory changes (add/update items).

    Args:
        vendor_phone: Vendor's phone number
        parsed_items: {"update": [...], "add": [...]}

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Import required modules
        from backend.vendor_intake_flow import InventoryIntake
        from search_agent.database.sql_client import SQLClient
        from search_agent import config

        sql_client = SQLClient(db_path=config.SQL_DB_PATH)
        intake = InventoryIntake(sql_client)
        results = {"updated": [], "added": [], "errors": []}

        # Process updates - find existing products by name and update them
        for item in parsed_items.get("update", []):
            try:
                # Find matching product in vendor's inventory
                similar = intake.find_similar(
                    name=item["name"],
                    price=item.get("price"),
                    vendor_id=vendor_phone,
                    top_k=1
                )

                if similar and similar[0][2] > 0.7:  # Similarity threshold
                    product_id, prod_name, score, _ = similar[0]
                    print(f"[OCR] Matched '{item['name']}' to '{prod_name}' (score: {score:.2f})")

                    # Build update payload
                    payload = {
                        "price": item["price"],
                        "stock": item["stock"],
                        "quantity": 1,  # Price per 1 unit
                        "quantityunit": item["unit"]
                    }

                    result = intake.update_product(product_id, payload)
                    results["updated"].append({
                        "name": prod_name,
                        "old_stock": result["old"]["stock"],
                        "new_stock": result["new"]["stock"]
                    })
                else:
                    # No match found - add as new instead
                    print(f"[OCR] No match for '{item['name']}', adding as new product")
                    payload = {
                        "name": item["name"],
                        "price": item["price"],
                        "stock": item["stock"],
                        "quantity": 1,
                        "quantityunit": item["unit"]
                    }
                    product_id = intake.add_product(vendor_phone, payload)
                    results["added"].append(item["name"])

            except Exception as e:
                results["errors"].append(f"Error updating {item['name']}: {str(e)}")

        # Process additions - add new products
        for item in parsed_items.get("add", []):
            try:
                payload = {
                    "name": item["name"],
                    "price": item["price"],
                    "stock": item["stock"],
                    "quantity": 1,
                    "quantityunit": item["unit"]
                }
                product_id = intake.add_product(vendor_phone, payload)
                results["added"].append(item["name"])
                print(f"[OCR] Added new product: {item['name']} (ID: {product_id})")

            except Exception as e:
                results["errors"].append(f"Error adding {item['name']}: {str(e)}")

        # Build summary message
        summary_lines = ["✅ *Inventory Changes Applied*\n"]

        if results["updated"]:
            summary_lines.append(f"*Updated ({len(results['updated'])}):*")
            for update in results["updated"]:
                if isinstance(update, dict):
                    summary_lines.append(
                        f"  • {update['name']}: Stock {update['old_stock']} → {update['new_stock']}"
                    )
                else:
                    summary_lines.append(f"  • {update}")
            summary_lines.append("")

        if results["added"]:
            summary_lines.append(f"*Added ({len(results['added'])}):*")
            for name in results["added"]:
                summary_lines.append(f"  • {name}")
            summary_lines.append("")

        if results["errors"]:
            summary_lines.append(f"*Errors ({len(results['errors'])}):*")
            for error in results["errors"]:
                summary_lines.append(f"  ⚠️ {error}")

        return True, "\n".join(summary_lines)

    except ImportError as e:
        return False, f"Inventory management module not available: {str(e)}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error applying inventory changes: {str(e)}"


def handle_inventory_confirmation(vendor_phone: str, command: str) -> Optional[str]:
    """
    Handle confirm/cancel for pending inventory updates.

    Args:
        vendor_phone: Vendor's phone number
        command: User's text command

    Returns:
        Response message if handled, None if not a confirmation command
    """
    global _pending_inventory_update

    if _pending_inventory_update is None:
        return None

    if _pending_inventory_update.get("vendor") != vendor_phone:
        return None

    command_lower = command.lower().strip()

    if command_lower in ["confirm", "yes", "y", "ok"]:
        # Apply the changes
        parsed_items = _pending_inventory_update["data"]["parsed_items"]
        success, message = apply_inventory_changes(vendor_phone, parsed_items)
        _pending_inventory_update = None
        return message

    elif command_lower in ["cancel", "no", "n", "discard"]:
        _pending_inventory_update = None
        return "❌ Inventory update cancelled."

    return None


def process_vendor_inventory_image(file_path: str, vendor_phone: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Process a vendor's inventory image using OCR.

    Args:
        file_path: Path to the image file
        vendor_phone: Vendor's phone number

    Returns:
        Tuple of (success: bool, message: str, data: dict)
    """
    if not _ocr_available:
        return False, "OCR module is not available. Cannot process inventory images.", {}

    try:
        print(f"[OCR] Processing image: {file_path}")

        # Initialize OCR processor if needed
        print("[OCR] Initializing OCR processor...")
        OLMOCRProcessor.initialize(debug=True)

        # Extract text using OCR
        ocr_text = extract_text_from_image(file_path, debug=True)

        if not ocr_text or not ocr_text.strip():
            return False, "Could not extract any text from the image. Please ensure the image is clear.", {}

        print(f"[OCR] Extracted text:\n{ocr_text}")

        # Parse the inventory list
        parsed_items = parse_inventory_list_text(ocr_text)

        update_count = len(parsed_items["update"])
        add_count = len(parsed_items["add"])

        if update_count == 0 and add_count == 0:
            return False, f"Could not parse any inventory items from the image.\n\nExtracted text:\n{ocr_text}", {}

        # Build summary message
        summary_lines = ["📋 *Inventory List Detected*\n"]

        if update_count > 0:
            summary_lines.append(f"*Update ({update_count} items):*")
            for item in parsed_items["update"]:
                summary_lines.append(
                    f"  • {item['name']}: ₹{item['price']}/{item['unit']}, "
                    f"Stock: {item['stock']} {item['stock_unit']}"
                )
            summary_lines.append("")

        if add_count > 0:
            summary_lines.append(f"*Add ({add_count} items):*")
            for item in parsed_items["add"]:
                summary_lines.append(
                    f"  • {item['name']}: ₹{item['price']}/{item['unit']}, "
                    f"Stock: {item['stock']} {item['stock_unit']}"
                )

        summary = "\n".join(summary_lines)

        return True, summary, {
            "ocr_text": ocr_text,
            "parsed_items": parsed_items,
            "update_count": update_count,
            "add_count": add_count
        }

    except Exception as e:
        print(f"[ERROR] OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error processing image: {str(e)}", {}


# ========================= Terminal Input Functions =========================

def get_terminal_input() -> Dict[str, Any]:
    """
    Get message input from terminal.

    Returns dict with:
        - phone: Phone number
        - type: "text", "image", or "document"
        - message: Text content
        - file_path: Path to file (for image/document)
        - context_id: Optional message ID being replied to
    """
    global _last_phone, _last_message_id

    print("\n" + "=" * 60)
    print("Enter message details (or 'quit' to exit):")
    print("-" * 60)

    # Phone number (with default)
    default_phone = _last_phone or "917893127444"
    phone_input = input(f"Phone [{default_phone}]: ").strip()
    if phone_input.lower() == 'quit':
        return {"quit": True}
    phone = phone_input if phone_input else default_phone
    _last_phone = phone

    # Message type
    msg_type = input("Type [text/image/document] (default: text): ").strip().lower() or "text"
    if msg_type not in ["text", "image", "document"]:
        msg_type = "text"

    # Message content
    message = input("Message: ").strip()

    # File path for image/document
    file_path = None
    if msg_type in ["image", "document"]:
        file_path = input("File path: ").strip()
        if file_path and not Path(file_path).exists():
            print(f"[WARNING] File not found: {file_path}")

    # Reply context
    use_context = input(f"Reply to last message? [y/N]: ").strip().lower()
    context_id = _last_message_id if use_context == 'y' else None

    return {
        "phone": phone,
        "type": msg_type,
        "message": message,
        "file_path": file_path,
        "context_id": context_id,
        "incoming_message_id": f"TERM_IN_{uuid.uuid4().hex[:8]}"
    }


# ========================= Message Processing =========================

def process_message(input_data: Dict[str, Any]):
    """
    Process a message from terminal input.
    Mimics the WhatsApp webhook handler logic.
    """
    from_number = input_data["phone"]
    msg_type = input_data["type"]
    raw_user_text = input_data.get("message", "")
    file_path = input_data.get("file_path")
    context_message_id = input_data.get("context_id")
    incoming_message_id = input_data.get("incoming_message_id")

    print(f"\n[PROCESSING] From: {from_number}, Type: {msg_type}")

    # Determine user type
    is_vendor = is_registered_vendor(from_number)
    is_user = is_registered_user(from_number)

    # ===== VENDOR FLOW =====
    if is_vendor:
        print(f"[INFO] VENDOR detected: {from_number}")

        attachments = []

        if msg_type == "text":
            print(f"[INFO] Vendor text message: {raw_user_text}")

            # Check for pending inventory confirmation
            confirmation_response = handle_inventory_confirmation(from_number, raw_user_text)
            if confirmation_response:
                send_text_message(from_number, confirmation_response)
                return

        elif msg_type == "image":
            if file_path and Path(file_path).exists():
                print(f"[INFO] Vendor image: {file_path}")

                # Try OCR processing for inventory lists
                if _ocr_available:
                    print(f"[INFO] Attempting OCR processing for inventory list...")
                    success, message, ocr_data = process_vendor_inventory_image(file_path, from_number)

                    if success:
                        # OCR succeeded - show parsed items and ask for confirmation
                        send_text_message(from_number, message)
                        send_text_message(
                            from_number,
                            "Reply 'confirm' to apply these changes, or 'cancel' to discard."
                        )

                        # Store parsed data for confirmation
                        # (In a real system, this would be stored in a session/database)
                        global _pending_inventory_update
                        _pending_inventory_update = {
                            "vendor": from_number,
                            "data": ocr_data
                        }
                        return
                    else:
                        # OCR failed or couldn't parse - fall back to backend
                        print(f"[INFO] OCR parsing failed, falling back to backend: {message}")
                        send_text_message(from_number, f"[OCR Note] {message}")

                # Fall back to sending to vendor backend
                attachments.append({
                    "type": "image",
                    "path": file_path,
                    "media_id": f"local_{uuid.uuid4().hex[:8]}",
                    "mime_type": "image/jpeg"
                })
            else:
                send_text_message(from_number, "Failed to process your image. File not found.")
                return

        elif msg_type == "document":
            if file_path and Path(file_path).exists():
                attachments.append({
                    "type": "document",
                    "path": file_path,
                    "media_id": f"local_{uuid.uuid4().hex[:8]}",
                    "mime_type": "application/pdf",
                    "filename": Path(file_path).name
                })
                print(f"[INFO] Vendor document: {file_path}")
            else:
                send_text_message(from_number, "Failed to process your document. File not found.")
                return

        else:
            send_text_message(
                from_number,
                f"Unsupported message type: {msg_type}. Please send text, images, or documents."
            )
            return

        # Route to vendor backend
        print(f"[INFO] Routing to VENDOR backend for {from_number}")
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
            print(f"[ERROR] Error calling vendor backend: {e}")
            send_text_message(
                from_number,
                "Something went wrong processing your vendor request. Try again later."
            )
            return

        # Handle vendor response
        if "messages" in backend_data:
            for msg in backend_data["messages"]:
                if msg.get("type") == "image":
                    send_image_message(from_number, msg["url"], caption="")
                elif msg.get("type") == "text":
                    send_text_message(from_number, msg["text"])
        elif "reply" in backend_data:
            send_text_message(from_number, backend_data["reply"])

        return
    # ===== END VENDOR FLOW =====

    # ===== USER FLOW - Only text messages =====
    if msg_type != "text":
        send_text_message(
            from_number,
            "Sorry, I only understand text messages right now."
        )
        return

    print(f"[INFO] Incoming message from {from_number}: {raw_user_text}")

    # Check if new or existing user
    if not is_user:
        print(f"[INFO] New user detected: {from_number} - Starting user onboarding")

    # ===== User Management Check =====
    user_manager = get_user_manager()
    onboarding_response = user_manager.process_message(from_number, raw_user_text)

    if onboarding_response is not None:
        print(f"[INFO] Onboarding response to {from_number}")
        send_text_message(from_number, onboarding_response)
        return

    # ===== Cart/Wishlist Command Check =====
    cart_manager = get_cart_manager()
    cart_response = cart_manager.handle_command(from_number, raw_user_text)

    if cart_response is not None:
        print(f"[INFO] Cart command handled for {from_number}")
        send_text_message(from_number, cart_response)
        return

    # Resolve short IDs for search queries
    user_text = resolve_short_ids_in_text(raw_user_text)

    # ===== Call USER backend =====
    print(f"[INFO] Routing to USER backend for {from_number}")
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
        print(f"[ERROR] Error calling backend: {e}")
        send_text_message(
            from_number,
            "Some error happened in my brain (backend). Try again later."
        )
        return

    # Get message history manager for tracking
    message_history = get_message_history()

    # Format 1: Search response - {"text": str, "images": list, "product_ids": list}
    if "text" in backend_data or "images" in backend_data:
        text = backend_data.get("text")
        images = backend_data.get("images", [])
        product_ids = backend_data.get("product_ids", [])

        # Send images first and track message IDs
        if images:
            for i, img in enumerate(images):
                result = send_image_message(
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

        # Then send summary text
        if text:
            send_text_message(from_number, text)

    # Format 2: Detail response - {"messages": [{"type": "image"|"text", ...}]}
    elif "messages" in backend_data:
        product_id = backend_data.get("product_id")
        short_id = backend_data.get("short_id")

        for msg in backend_data["messages"]:
            if msg["type"] == "image":
                result = send_image_message(
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
                result = send_text_message(from_number, msg["text"])
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
        print(f"[INFO] Replying to {from_number}: {reply_text}")
        send_text_message(from_number, reply_text)

    else:
        # Fallback
        send_text_message(
            from_number,
            "I couldn't understand the backend response."
        )


# ========================= Main Loop =========================

def main():
    """Main terminal bot loop."""
    print("\n" + "=" * 60)
    print("  TERMINAL BOT - WhatsApp Simulator")
    print("=" * 60)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Vendor Backend URL: {VENDOR_BACKEND_URL}")
    print(f"User Registry: {USER_REGISTRY_PATH}")
    print(f"Vendor Registry: {VENDOR_REGISTRY_PATH}")
    print("-" * 60)
    print(f"OCR Available: {'Yes' if _ocr_available else 'No'}")
    if _ocr_available:
        print("  -> Vendor inventory images will be processed with OCR")
    print("=" * 60)
    print("\nType 'quit' at any prompt to exit.")
    print("Press Ctrl+C to force quit.\n")

    # Preload short ID mappings
    load_shortid_mapping()

    try:
        while True:
            input_data = get_terminal_input()

            if input_data.get("quit"):
                print("\nGoodbye!")
                break

            if not input_data.get("message") and not input_data.get("file_path"):
                print("[WARNING] No message or file provided. Skipping.")
                continue

            process_message(input_data)

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
