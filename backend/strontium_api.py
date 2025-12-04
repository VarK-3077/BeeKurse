"""
Strontium Backend for WhatsApp Integration
Replaces backend_dummy.py with full search engine capabilities
"""
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import Dict, Any, List, Optional

# Import local modules
from search_agent.strontium.strontium_agent import StrontiumAgent
from search_agent.orchestrator.chat_handler import ChatHandler
from search_agent.orchestrator.orchestrator import SearchOrchestrator
from search_agent.database.sql_client import SQLClient
from config.config import Config
from backend.vendor_intake_flow import VendorIntakeFlow

# to get image url
from scripts.database_operations.sql_extract import fetch_products_by_ids

# Cart manager for handling cart actions
from .cart_manager import get_cart_manager

# Use config
config = Config

# Initialize FastAPI
app = FastAPI(title="Strontium WhatsApp Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
print("🚀 Initializing Strontium Backend...")

# Strontium Agent with NVIDIA API
strontium = StrontiumAgent(
    mock_data_dir=config.USER_CONTEXT_DATA_DIR,
    enable_caching=config.ENABLE_CACHING,
    llm_cache_ttl=config.LLM_CACHE_TTL,
    kg_cache_ttl=config.KG_CACHE_TTL,
    use_nvidia=config.USE_NVIDIA_LLM,
    nvidia_api_key=config.NVIDIA_API_KEY
)

# Search Orchestrator
orchestrator = SearchOrchestrator()

# Chat Handler (with LLM support)
chat_handler = ChatHandler(use_nvidia=config.USE_NVIDIA_LLM)

# SQL Client for product details (main inventory)
sql_client = SQLClient(db_path=config.SQL_DB_PATH)

# Vendor intake flow (uses test DB if configured, otherwise main DB)
# Pass None to let VendorIntakeFlow decide based on USE_VENDOR_TEST_DB config
vendor_flow = VendorIntakeFlow(sql_client=None)

print("✅ Strontium Backend initialized successfully!")


class MessagePayload(BaseModel):
    """WhatsApp message payload from whatsapp_bot.py"""
    sender: str   # WhatsApp phone number (used as user_id)
    message: str  # User's text message
    context_message_id: Optional[str] = None  # ID of message being replied to
    incoming_message_id: Optional[str] = None  # This message's ID


class VendorMessage(BaseModel):
    """Vendor intake payload (add/update inventory)."""

    sender: str
    message: str
    attachments: Optional[List[Dict[str, Any]]] = None
    context_id: Optional[str] = None  # ID of message being replied to (for multi-product tracking)
    incoming_message_id: Optional[str] = None  # This message's WhatsApp ID



# Structure: {user_id: [{"url": str, "caption": str, "product_id": str, "short_id": str, "name": str, "price": str, "rating": str, "store": str}]}
user_current_images: Dict[str, List[Dict[str, Any]]] = {}

def set_user_images(user_id: str, products_data: List[Dict[str, Any]]):
    """
    Set/Replace all images for a user (clears previous images)
    
    Args:
        user_id: User identifier (phone number)
        products_data: List of product dicts with all details
    """
    user_current_images[user_id] = []
    
    for product in products_data:
        user_current_images[user_id].append(product)
    
    print(f"📸 Set {len(products_data)} images for user {user_id} (replaced old images)")


def get_user_images(user_id: str) -> List[Dict[str, Any]]:
    """Get current images for a user"""
    return user_current_images.get(user_id, [])


# ------------------------------ stuff for displaying additional images ends -------------------


# ---------------------------------- Image Add ---------------------------------------------------
def format_search_response(parsed: Dict[str, Any], orchestrator: SearchOrchestrator, sql_client: SQLClient, user_id: str = None) -> Dict[str, Any]:
    """
    WhatsApp-friendly search response:
      - sends up to 4 product images
      - returns a text block with product summaries
    """
    try:
        # Run Strontium search with user_id for personalized filtering
        search_results = orchestrator.search_strontium(parsed, user_id=user_id)

        # Check for no relevant results
        for res in search_results:
            if res.no_relevant_results:
                # Get subcategory for friendly message
                subcategory = "products"
                products_list = parsed.get("products", [])
                if products_list:
                    subcategory = products_list[0].get("product_subcategory", "products")

                if res.filter_reason == "gender_filter":
                    msg = f"😔 Sorry, we don't have {subcategory}s matching your preferences in our catalog."
                elif res.filter_reason == "relevance_threshold":
                    msg = f"😔 Sorry, we don't have {subcategory}s in our catalog right now."
                else:
                    msg = "😔 Sorry, no products matched your search."

                return {"text": msg, "images": []}

        # Gather product IDs
        all_product_ids = []
        for res in search_results:
            all_product_ids.extend(res.product_ids)

        if not all_product_ids:
            return {
                "text": "😔 Sorry, no products matched your search.",
                "images": []
            }

        # Fetch product info from SQL
        # products = sql_client.get_products_by_ids(all_product_ids)
        products = fetch_products_by_ids(all_product_ids)


        # Build complete product data for storage
        all_products_data = []
        for pid in all_product_ids:
            if pid in products:
                p = products[pid]
                all_products_data.append({
                    "url": p.get("image_url", ""),
                    "imageid": p.get("imageid", ""),
                    "prod_name": p.get("prod_name", ""),
                    "name": p.get("prod_name", ""),
                    "product_id": pid,
                    "short_id": p.get("short_id", ""),
                    "price": p.get("price", ""),
                    "rating": p.get("rating", "N/A"),
                    "store": p.get("store", ""),
                    "store_contact": p.get("store_contact", ""),
                    "store_location": p.get("store_location", {}),
                    "brand": p.get("brand", ""),
                    "colour": p.get("colour", ""),
                    "color": p.get("colour", ""),
                    "description": p.get("description", ""),
                    "category": p.get("category", ""),
                    "subcategory": p.get("subcategory", ""),
                    "stock": p.get("stock"),
                    "quantity": p.get("quantity", ""),
                    "quantityunit": p.get("quantityunit", ""),
                    "size": p.get("size", ""),
                    "dimensions": p.get("dimensions", {}),
                    "other_properties": p.get("other_properties", {})
                })


        # Store ALL products in backend
        set_user_images(user_id, all_products_data)


        # Pick top 4 products for images
        top_products = [products[pid] for pid in all_product_ids[:4] if pid in products]

        # ✨ Build summary text
        lines = []
        lines.append(f"🔍 Found {len(all_product_ids)} product(s):\n")

        for i, p in enumerate(top_products, start=1):
            name = p["prod_name"]
            if len(name) > 40:
                name = name[:40] + "..."

            price = p["price"] or "N/A"
            rating = p["rating"] or "N/A"
            store = p["store"] or "Unknown"
            short_id = p["short_id"]

            lines.append(f"{i}. *{name}*")
            lines.append(f"   ₹{price} | ⭐{rating}")
            lines.append(f"   Store: {store}")
            lines.append(f"   ID: {short_id}\n")

        if len(all_product_ids) > 4:
            lines.append(f"_... and {len(all_product_ids) - 4} more results_")

        lines.append("\n💡 Ask about a product using its ID")
        lines.append("Example: _\"details of A3F1\"_")

        text_message = "\n".join(lines)

        # ✨ Collect image URLs
        images = [{"url": p["image_url"], "caption": f"{p['prod_name']} (ID: {p['short_id']})"} for p in top_products]

        # Get product IDs for the images (for message tracking)
        top_product_ids = all_product_ids[:4]

        # ✨ Build gallery link message (separate)
        gallery_message = None
        if len(all_product_ids) > 4:
            gallery_message = (
                f"🖼️ *View all {len(all_product_ids)} products with images:*\n\n"
                f"http://localhost:5400?user={user_id}"
            )

        return {
            "text": text_message,
            "images": images,
            "gallery_link": gallery_message,
            "product_ids": top_product_ids  # For contextual reply tracking
        }

    except Exception as e:
        print(f"❌ Error in search: {e}")
        return {
            "text": "❌ Something went wrong while searching.",
            "images": []
        }


# ----------------------------------------------------------------------------------


# ----------------------- Image add ---------------------------------------
def format_detail_response(
    parsed: Dict[str, Any],
    orchestrator: SearchOrchestrator,
    sql_client: SQLClient
) -> Dict[str, Any]:

    try:
        # Support both new format (product_ids) and old format (product_id)
        product_ids = parsed.get("product_ids", [])
        if not product_ids:
            # Backward compatibility
            old_product_id = parsed.get("product_id")
            if old_product_id:
                product_ids = [old_product_id]

        if not product_ids:
            return {
                "messages": [
                    {"type": "text", "text": "❌ No product ID found."}
                ]
            }

        # Resolve short_ids (4-char codes like "44QM") to full product_ids
        import re
        short_id_pattern = re.compile(r'^[A-Z0-9]{4}$', re.IGNORECASE)
        resolved_ids = []
        for pid in product_ids:
            if short_id_pattern.match(pid):
                full_id = sql_client.resolve_short_id(pid)
                resolved_ids.append(full_id if full_id else pid)
            else:
                resolved_ids.append(pid)
        product_ids = resolved_ids

        # ⭐ Fetch using your improved extractor
        products = fetch_products_by_ids(product_ids)

        if not products:
            return {
                "messages": [
                    {"type": "text", "text": f"❌ No product found for ID(s) {', '.join(product_ids)}"}
                ]
            }

        # LLM detail answer
        answer = orchestrator.answer_detail_query(parsed)

        # Build response based on number of products
        messages = []

        # For single product, show image
        if len(product_ids) == 1 and product_ids[0] in products:
            product = products[product_ids[0]]
            messages.append({
                "type": "image",
                "url": product["image_url"],
            })

        # Just use the natural LLM response - no technical headers
        messages.append({
            "type": "text",
            "text": answer
        })

        # ⭐ Return multi-message response with product_ids for cart tracking
        return {
            "messages": messages,
            "product_ids": product_ids,
            "product_id": product_ids[0] if product_ids else None,  # Backward compat
            "short_id": products[product_ids[0]].get("short_id") if product_ids and product_ids[0] in products else None
        }

    except Exception as e:
        print("❌ DETAIL ERROR:", e)
        import traceback
        traceback.print_exc()
        return {
            "messages": [
                {"type": "text", "text": "❌ Could not fetch details."}
            ]
        }


# --------------------------------------------------------------------


def format_chat_response(parsed: Dict[str, Any], chat_handler: ChatHandler, user_id: str = None) -> str:
    """
    Format chat response for WhatsApp

    Args:
        parsed: Strontium parsed output
        chat_handler: ChatHandler instance
        user_id: User ID for context loading

    Returns:
        Chat response
    """
    try:
        # Get chat response with user context
        response = chat_handler.handle_chat_output(parsed, user_id)
        return response

    except Exception as e:
        print(f"❌ Error in chat: {e}")
        return "Hi! I'm Strontium, your shopping curator. How can I help you today?"


@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Strontium WhatsApp Backend",
        "version": "1.0.0",
        "nvidia_enabled": config.USE_NVIDIA_LLM
    }


def handle_cart_action(user_id: str, parsed: Dict[str, Any]) -> str:
    """
    Handle cart_action query type.
    Returns formatted response string.
    """
    cart_manager = get_cart_manager()
    action = parsed.get("action")
    target = parsed.get("target")
    product_id = parsed.get("product_id")

    # Handle "LAST_VIEWED" placeholder
    if product_id == "LAST_VIEWED":
        product_id = cart_manager.get_last_viewed(user_id)
        if not product_id:
            return "Please view a product first, then I can add it to your cart."

    if not product_id:
        return "I couldn't identify which product you're referring to. Please specify a product ID."

    # Resolve short_id (4-char code like "44QM") to full product_id
    import re
    short_id_pattern = re.compile(r'^[A-Z0-9]{4}$', re.IGNORECASE)
    if short_id_pattern.match(product_id):
        full_id = sql_client.resolve_short_id(product_id)
        if full_id:
            product_id = full_id

    # Fetch product name for response
    products = fetch_products_by_ids([product_id])
    product = products.get(product_id, {})
    name = product.get("prod_name", product_id[:8])
    if len(name) > 30:
        name = name[:30] + "..."

    if action == "add" and target == "cart":
        success, status = cart_manager.add_to_cart(user_id, product_id)
        if success:
            if status == "already_exists":
                return f"ℹ️ *{name}* is already in your cart"
            return f"✅ Added *{name}* to your cart!\n\n🛒 View cart: {cart_manager.get_cart_url(user_id)}"
        return "❌ Could not add to cart"

    elif action == "add" and target == "wishlist":
        success, status = cart_manager.add_to_wishlist(user_id, product_id)
        if success:
            if status == "already_exists":
                return f"ℹ️ *{name}* is already in your wishlist"
            return f"❤️ Saved *{name}* to your wishlist!\n\n💝 View wishlist: {cart_manager.get_wishlist_url(user_id)}"
        return "❌ Could not add to wishlist"

    elif action == "remove" and target == "cart":
        success, _ = cart_manager.remove_from_cart(user_id, product_id)
        if success:
            return f"✅ Removed *{name}* from your cart"
        return f"❌ *{name}* is not in your cart"

    elif action == "remove" and target == "wishlist":
        success, _ = cart_manager.remove_from_wishlist(user_id, product_id)
        if success:
            return f"✅ Removed *{name}* from your wishlist"
        return f"❌ *{name}* is not in your wishlist"

    return "I couldn't understand that cart action. Try 'add to cart' or 'remove from wishlist'."


def handle_cart_view(user_id: str, parsed: Dict[str, Any]) -> str:
    """
    Handle cart_view query type.
    Returns formatted response string.
    """
    cart_manager = get_cart_manager()
    target = parsed.get("target")

    if target == "cart":
        cart = cart_manager.get_cart(user_id)
        if not cart:
            return "🛒 Your cart is empty.\n\nSearch for products to add!"
        return f"🛒 You have {len(cart)} item(s) in your cart.\n\nView & manage: {cart_manager.get_cart_url(user_id)}"

    elif target == "wishlist":
        wishlist = cart_manager.get_wishlist(user_id)
        if not wishlist:
            return "❤️ Your wishlist is empty.\n\nSave products you love for later!"
        return f"❤️ You have {len(wishlist)} item(s) in your wishlist.\n\nView & manage: {cart_manager.get_wishlist_url(user_id)}"

    return "Would you like to see your cart or wishlist?"


def enhance_message_with_context(message: str, product_id: str, short_id: str = None) -> str:
    """Enhance user message with product context from reply."""
    msg_lower = message.lower().strip()
    pid = short_id or product_id

    # Check for detail patterns
    if any(p in msg_lower for p in ["more detail", "details", "tell me more", "more info", "about this"]):
        return f"details of {pid}"

    # Check for cart patterns
    if any(p in msg_lower for p in ["add to cart", "add this", "buy this", "cart"]):
        return f"add {pid} to cart"

    # Check for wishlist patterns
    if any(p in msg_lower for p in ["save", "wishlist", "save this"]):
        return f"add {pid} to wishlist"

    # Check for similar patterns
    if any(p in msg_lower for p in ["similar", "like this", "more like"]):
        return f"products similar to {pid}"

    # Default: append product reference
    return f"{message} (about {pid})"


@app.post("/process")
def process_message(payload: MessagePayload):
    """
    Process WhatsApp messages via Strontium

    Input:
        {
            "sender": "phone_number",
            "message": "user query",
            "context_message_id": "optional - ID of message being replied to",
            "incoming_message_id": "optional - this message's ID"
        }

    Output:
        {
            "reply": "formatted response"
        }
    """
    try:
        user_phone = payload.sender
        user_message = payload.message

        print(f"\n📩 Incoming: [{user_phone}] {user_message}")

        # Use phone number as user_id (clean it first)
        user_id = user_phone.replace("+", "").replace("-", "").replace(" ", "")

        # Check if this is a reply to a product message
        if payload.context_message_id:
            from .message_history import get_message_history
            message_history = get_message_history()
            product_context = message_history.get_product_from_message(
                user_id, payload.context_message_id
            )
            if product_context:
                print(f"📎 Reply context: {product_context}")
                user_message = enhance_message_with_context(
                    user_message,
                    product_context.get("product_id"),
                    product_context.get("short_id")
                )
                print(f"📝 Enhanced message: {user_message}")

        # Step 1: Parse with Strontium
        print(f"🔍 Parsing query with Strontium...")
        parsed = strontium.process_query_to_dict(user_message, user_id)

        query_type = parsed.get("query_type", "unknown")
        print(f"📊 Query type: {query_type}")

        # Step 2: Route based on query type
        if query_type == "search":
            print(f"🔎 Executing search...")
            response = format_search_response(parsed, orchestrator, sql_client, user_id=user_id)
            return response

        elif query_type == "detail":
            print(f"📦 Getting product details...")
            response = format_detail_response(parsed, orchestrator, sql_client)
            return response

        elif query_type == "chat":
            print(f"💬 Handling chat...")
            reply_text = format_chat_response(parsed, chat_handler, user_id)

        elif query_type == "cart_action":
            print(f"🛒 Processing cart action...")
            reply_text = handle_cart_action(user_id, parsed)

        elif query_type == "cart_view":
            print(f"👀 Viewing cart/wishlist...")
            reply_text = handle_cart_view(user_id, parsed)

        else:
            reply_text = (
                "🤔 I'm not sure what you're asking for. "
                "Try:\n"
                "• Search: _\"Red cotton shirt under $30\"_\n"
                "• Details: _\"What material is p-123 made of?\"_\n"
                "• Chat: _\"Hello!\"_ or _\"What can you do?\"_"
            )

        print(f"✅ Reply: {reply_text[:100]}...")

        return {"reply": reply_text}

    except Exception as e:
        print(f"❌ Error processing message: {e}")
        import traceback
        traceback.print_exc()

        # Return user-friendly error
        return {
            "reply": (
                "😕 Oops! Something went wrong on my end. "
                "Please try again or contact support if the problem persists."
            )
        }


@app.post("/vendor/process")
def process_vendor_message(payload: VendorMessage):
    """Handle vendor-facing add/update flows (Nunchi)."""

    try:
        user_phone = payload.sender
        user_message = payload.message
        attachments = payload.attachments or []
        context_id = payload.context_id  # ID of message being replied to
        incoming_message_id = payload.incoming_message_id  # This message's ID

        print(f"\n📦 Vendor intake from [{user_phone}]: {user_message}")
        if context_id:
            print(f"📎 Replying to message: {context_id}")

        reply = vendor_flow.handle(
            user_phone,
            user_message,
            attachments=attachments,
            reply_context_id=context_id,
            incoming_message_id=incoming_message_id
        )

        return reply

    except Exception as e:
        print(f"❌ Error in vendor intake: {e}")
        import traceback

        traceback.print_exc()
        return {
            "messages": [
                {
                    "type": "text",
                    "text": (
                        "We hit a snag while processing your inventory message. "
                        "Please try again in a moment."
                    ),
                }
            ]
        }


@app.post("/vendor/register-message-ids")
def register_message_ids(payload: Dict[str, Any]):
    """
    Register WhatsApp message IDs for multi-product reply tracking.
    Called by whatsapp_bot after sending product messages.
    """
    user_id = payload.get("user_id")
    mappings = payload.get("mappings", {})  # {wamid: product_index}

    if not user_id or not mappings:
        return {"status": "error", "detail": "Missing user_id or mappings"}

    session = vendor_flow.sessions.get(user_id)
    if session:
        for wamid, product_index in mappings.items():
            session.product_message_map[wamid] = product_index
        print(f"📍 Registered {len(mappings)} message mappings for {user_id}")
        return {"status": "ok", "count": len(mappings)}

    return {"status": "warning", "detail": "Session not found"}


# to display additional images
@app.get("/images/{user_id}")
def get_images(user_id: str):
    """
    Get current search images for a user
    
    Returns:
        {
            "user_id": str,
            "products": [
                {
                    "url": str,
                    "name": str,
                    "price": str,
                    "rating": str,
                    "store": str,
                    "short_id": str,
                    "product_id": str
                }
            ],
            "total": int
        }
    """
    images = get_user_images(user_id)
    
    return {
        "user_id": user_id,
        "products": images,
        "total": len(images)
    }


# Development server
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 Starting Strontium WhatsApp Backend")
    print("="*80)
    print(f"📍 Endpoint: http://localhost:5001/process")
    print(f"🔧 NVIDIA API: {'Enabled' if config.USE_NVIDIA_LLM else 'Disabled (Mock)'}")
    print(f"💾 User Data: {config.USER_CONTEXT_DATA_DIR}")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
