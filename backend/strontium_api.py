"""
Strontium Backend for WhatsApp Integration
Replaces backend_dummy.py with full search engine capabilities
"""
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from typing import Dict, Any

# Import local modules
from search_agent.strontium.strontium_agent import StrontiumAgent
from search_agent.orchestrator.chat_handler import ChatHandler
from search_agent.orchestrator.orchestrator import SearchOrchestrator
from search_agent.database.sql_client import SQLClient
from config.config import Config

# to get image url
from scripts.database_operations.sql_extract import fetch_products_by_ids

# Cart manager for handling cart actions
from .cart_manager import get_cart_manager

# Use config
config = Config

# Initialize FastAPI
app = FastAPI(title="Strontium WhatsApp Backend")

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

# Chat Handler
chat_handler = ChatHandler()

# SQL Client for product details
sql_client = SQLClient(db_path=config.SQL_DB_PATH)

print("✅ Strontium Backend initialized successfully!")


class MessagePayload(BaseModel):
    """WhatsApp message payload from whatsapp_bot.py"""
    sender: str   # WhatsApp phone number (used as user_id)
    message: str  # User's text message


# ---------------------------------- Image Add ---------------------------------------------------
def format_search_response(parsed: Dict[str, Any], orchestrator: SearchOrchestrator, sql_client: SQLClient) -> Dict[str, Any]:
    """
    WhatsApp-friendly search response:
      - sends up to 4 product images
      - returns a text block with product summaries
    """
    try:
        # Run Strontium search
        search_results = orchestrator.search_strontium(parsed)

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
        # images = [{"url": "https://plus.unsplash.com/premium_photo-1762541871245-ffaf4ab5da47?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHwyfHx8ZW58MHx8fHx8", "caption": "some random shit"}]

        return {
            "text": text_message,
            "images": images    # <= send up to 4 images
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
        product_id = parsed.get("product_id")
        if not product_id:
            return {
                "messages": [
                    {"type": "text", "text": "❌ No product ID found."}
                ]
            }

        # ⭐ Fetch using your improved extractor
        # products = sql_client.get_products_by_ids([product_id])
        products = fetch_products_by_ids([product_id])
        product = products.get(product_id)

        if not product:
            return {
                "messages": [
                    {"type": "text", "text": f"❌ No product found for ID {product_id}"}
                ]
            }

        # LLM detail answer
        answer = orchestrator.answer_detail_query(parsed)

        # Build caption
        caption = (
            f"📦 *Product Information: {product_id}*\n\n"
            f"{answer}"
        )

        # ⭐ Return multi-message response with product_id for cart tracking
        return {
            "messages": [
                {
                    "type": "image",
                    "url": product["image_url"],   # dynamic URL from extractor
                },
                {
                    "type": "text",
                    "text": caption
                }
            ],
            "product_id": product_id,
            "short_id": product.get("short_id")
        }

    except Exception as e:
        print("❌ DETAIL ERROR:", e)
        return {
            "messages": [
                {"type": "text", "text": "❌ Could not fetch details."}
            ]
        }


# --------------------------------------------------------------------


def format_chat_response(parsed: Dict[str, Any], chat_handler: ChatHandler) -> str:
    """
    Format chat response for WhatsApp

    Args:
        parsed: Strontium parsed output
        chat_handler: ChatHandler instance

    Returns:
        Chat response
    """
    try:
        # Get chat response
        response = chat_handler.handle_chat_output(parsed)
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


@app.post("/process")
def process_message(payload: MessagePayload):
    """
    Process WhatsApp messages via Strontium

    Input:
        {
            "sender": "phone_number",
            "message": "user query"
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

        # Step 1: Parse with Strontium
        print(f"🔍 Parsing query with Strontium...")
        parsed = strontium.process_query_to_dict(user_message, user_id)

        query_type = parsed.get("query_type", "unknown")
        print(f"📊 Query type: {query_type}")

        # Step 2: Route based on query type
        if query_type == "search":
            print(f"🔎 Executing search...")
            response = format_search_response(parsed, orchestrator, sql_client)
            return response

        elif query_type == "detail":
            print(f"📦 Getting product details...")
            response = format_detail_response(parsed, orchestrator, sql_client)
            return response

        elif query_type == "chat":
            print(f"💬 Handling chat...")
            reply_text = format_chat_response(parsed, chat_handler)

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
