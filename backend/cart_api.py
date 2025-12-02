"""
Cart & Wishlist API Server for BeeKurse
Serves both JSON API and web pages for cart/wishlist management
Port: 8002
"""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import cart manager
from cart_manager import get_cart_manager, set_base_url

app = FastAPI(title="BeeKurse Cart & Wishlist API")

# Setup templates and static files
BACKEND_DIR = Path(__file__).parent
TEMPLATES_DIR = BACKEND_DIR / "templates"
STATIC_DIR = BACKEND_DIR / "static"

# Create directories if they don't exist
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# WhatsApp Business Number for deep links
WHATSAPP_NUMBER = os.getenv("WHATSAPP_BUSINESS_NUMBER", "")


class AddItemRequest(BaseModel):
    product_id: str
    short_id: Optional[str] = None
    quantity: Optional[int] = 1


class RemoveItemRequest(BaseModel):
    product_id: str


# ==================== Health Check ====================

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "BeeKurse Cart & Wishlist API",
        "port": 8002
    }


# ==================== JSON API Endpoints ====================

@app.get("/cart/{user_id}")
def get_cart(user_id: str):
    """Get user's cart items as JSON"""
    manager = get_cart_manager()
    cart = manager.get_cart_with_products(user_id)
    return {
        "user_id": user_id,
        "cart": cart,
        "count": len(cart)
    }


@app.get("/wishlist/{user_id}")
def get_wishlist(user_id: str):
    """Get user's wishlist items as JSON"""
    manager = get_cart_manager()
    wishlist = manager.get_wishlist_with_products(user_id)
    return {
        "user_id": user_id,
        "wishlist": wishlist,
        "count": len(wishlist)
    }


@app.post("/cart/{user_id}/add")
def add_to_cart(user_id: str, item: AddItemRequest):
    """Add item to cart"""
    manager = get_cart_manager()
    success, status = manager.add_to_cart(user_id, item.product_id, item.short_id, item.quantity or 1)
    return {
        "success": success,
        "status": status,
        "message": "Item added to cart" if success else "Failed to add item"
    }


@app.post("/cart/{user_id}/remove")
def remove_from_cart(user_id: str, item: RemoveItemRequest):
    """Remove item from cart"""
    manager = get_cart_manager()
    success, status = manager.remove_from_cart(user_id, item.product_id)
    return {
        "success": success,
        "status": status,
        "message": "Item removed from cart" if success else "Item not found in cart"
    }


@app.post("/wishlist/{user_id}/add")
def add_to_wishlist(user_id: str, item: AddItemRequest):
    """Add item to wishlist"""
    manager = get_cart_manager()
    success, status = manager.add_to_wishlist(user_id, item.product_id, item.short_id)
    return {
        "success": success,
        "status": status,
        "message": "Item added to wishlist" if success else "Failed to add item"
    }


@app.post("/wishlist/{user_id}/remove")
def remove_from_wishlist(user_id: str, item: RemoveItemRequest):
    """Remove item from wishlist"""
    manager = get_cart_manager()
    success, status = manager.remove_from_wishlist(user_id, item.product_id)
    return {
        "success": success,
        "status": status,
        "message": "Item removed from wishlist" if success else "Item not found in wishlist"
    }


# ==================== Web Page Endpoints ====================

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
        "whatsapp_number": WHATSAPP_NUMBER,
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
        "whatsapp_number": WHATSAPP_NUMBER,
        "page_type": "wishlist"
    })


# ==================== Admin Endpoints ====================

@app.post("/admin/set-ngrok-url")
def set_ngrok_url(url: str):
    """Set the ngrok URL for generating links"""
    set_base_url(url)
    return {"status": "ok", "url": url}


# ==================== Development Server ====================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🛒 Starting BeeKurse Cart & Wishlist Server")
    print("=" * 60)
    print(f"📍 API: http://localhost:8002")
    print(f"🌐 Cart page: http://localhost:8002/view/cart/{{user_id}}")
    print(f"❤️  Wishlist: http://localhost:8002/view/wishlist/{{user_id}}")
    print(f"📱 WhatsApp: {WHATSAPP_NUMBER or 'Not configured'}")
    print("=" * 60)
    print("\n💡 To expose via ngrok: ngrok http 8002")
    print("   Then set URL: POST /admin/set-ngrok-url?url=https://xxx.ngrok.io")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
