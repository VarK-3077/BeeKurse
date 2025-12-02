"""
Cart and Wishlist Manager for BeeKurse
Handles cart/wishlist operations and WhatsApp command processing
"""
import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
import threading

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database_operations.sql_extract import fetch_products_by_ids

# Constants
USER_DATA_BASE_DIR = Path(__file__).parent.parent / "data" / "user_data"
NGROK_URL_FILE = Path(__file__).parent.parent / "data" / "ngrok_url.txt"

# Default ngrok URL (update this when ngrok starts)
DEFAULT_BASE_URL = "http://localhost:8002"


def get_base_url() -> str:
    """Get the current base URL (ngrok or localhost)"""
    if NGROK_URL_FILE.exists():
        try:
            return NGROK_URL_FILE.read_text().strip()
        except:
            pass
    return os.getenv("CART_BASE_URL", DEFAULT_BASE_URL)


def set_base_url(url: str):
    """Set the ngrok URL"""
    NGROK_URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    NGROK_URL_FILE.write_text(url.strip())


@dataclass
class CartItem:
    """Single item in cart or wishlist"""
    product_id: str
    short_id: str
    added_at: str
    quantity: int = 1


@dataclass
class UserCartData:
    """User's cart and wishlist data"""
    user_id: str
    cart: List[Dict[str, Any]] = field(default_factory=list)
    wishlist: List[Dict[str, Any]] = field(default_factory=list)
    last_viewed_product: Optional[str] = None
    last_viewed_short_id: Optional[str] = None


class CartManager:
    """Manages cart and wishlist operations"""

    def __init__(self):
        self._lock = threading.Lock()
        self.base_dir = USER_DATA_BASE_DIR
        # Load short_id mapping
        self._shortid_cache = None

    def _load_shortid_mapping(self) -> Dict[str, str]:
        """Load short_id -> product_id mapping from SQL"""
        if self._shortid_cache is not None:
            return self._shortid_cache

        import sqlite3
        db_path = Path(__file__).parent.parent / "data" / "databases" / "sql" / "inventory.db"

        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT short_id, product_id FROM product_table")
            rows = cur.fetchall()
            self._shortid_cache = {short: pid for (short, pid) in rows if short}
            conn.close()
        except Exception as e:
            print(f"Error loading short_id mapping: {e}")
            self._shortid_cache = {}

        return self._shortid_cache

    def _get_product_id_from_short(self, short_id: str) -> Optional[str]:
        """Convert short_id to product_id"""
        mapping = self._load_shortid_mapping()
        return mapping.get(short_id.upper())

    def _get_short_id_from_product(self, product_id: str) -> Optional[str]:
        """Convert product_id to short_id"""
        mapping = self._load_shortid_mapping()
        for short, pid in mapping.items():
            if pid == product_id:
                return short
        return None

    def clean_phone_number(self, phone: str) -> str:
        """Clean phone number to use as user_id"""
        return phone.replace("+", "").replace("-", "").replace(" ", "")

    def get_cart_file_path(self, user_id: str) -> Path:
        """Get path to user's cart file"""
        return self.base_dir / user_id / "cart.json"

    def load_cart_data(self, user_id: str) -> UserCartData:
        """Load user's cart data from disk"""
        cart_path = self.get_cart_file_path(user_id)

        if cart_path.exists():
            try:
                with open(cart_path, 'r') as f:
                    data = json.load(f)
                    return UserCartData(
                        user_id=user_id,
                        cart=data.get("cart", []),
                        wishlist=data.get("wishlist", []),
                        last_viewed_product=data.get("last_viewed_product"),
                        last_viewed_short_id=data.get("last_viewed_short_id")
                    )
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error loading cart data for {user_id}: {e}")

        return UserCartData(user_id=user_id)

    def save_cart_data(self, data: UserCartData):
        """Save user's cart data to disk"""
        cart_path = self.get_cart_file_path(data.user_id)
        cart_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cart_path, 'w') as f:
            json.dump({
                "user_id": data.user_id,
                "cart": data.cart,
                "wishlist": data.wishlist,
                "last_viewed_product": data.last_viewed_product,
                "last_viewed_short_id": data.last_viewed_short_id
            }, f, indent=2)

    def set_last_viewed(self, user_id: str, product_id: str, short_id: str = None):
        """Set the last viewed product for a user"""
        user_id = self.clean_phone_number(user_id)

        with self._lock:
            data = self.load_cart_data(user_id)
            data.last_viewed_product = product_id
            data.last_viewed_short_id = short_id or self._get_short_id_from_product(product_id)
            self.save_cart_data(data)

    def get_last_viewed(self, user_id: str) -> Optional[str]:
        """Get the last viewed product_id for a user"""
        user_id = self.clean_phone_number(user_id)
        data = self.load_cart_data(user_id)
        return data.last_viewed_product

    def add_to_cart(self, user_id: str, product_id: str, short_id: str = None, quantity: int = 1) -> Tuple[bool, str]:
        """Add item to cart. Returns (success, message)"""
        user_id = self.clean_phone_number(user_id)

        if not short_id:
            short_id = self._get_short_id_from_product(product_id)

        with self._lock:
            data = self.load_cart_data(user_id)

            # Check if already in cart
            for item in data.cart:
                if item["product_id"] == product_id:
                    item["quantity"] = item.get("quantity", 1) + quantity
                    self.save_cart_data(data)
                    return True, "quantity_updated"

            # Add new item
            data.cart.append({
                "product_id": product_id,
                "short_id": short_id,
                "added_at": datetime.now().isoformat(),
                "quantity": quantity
            })
            self.save_cart_data(data)
            return True, "added"

    def remove_from_cart(self, user_id: str, product_id: str) -> Tuple[bool, str]:
        """Remove item from cart"""
        user_id = self.clean_phone_number(user_id)

        with self._lock:
            data = self.load_cart_data(user_id)
            original_len = len(data.cart)
            data.cart = [item for item in data.cart if item["product_id"] != product_id]

            if len(data.cart) < original_len:
                self.save_cart_data(data)
                return True, "removed"
            return False, "not_found"

    def add_to_wishlist(self, user_id: str, product_id: str, short_id: str = None) -> Tuple[bool, str]:
        """Add item to wishlist"""
        user_id = self.clean_phone_number(user_id)

        if not short_id:
            short_id = self._get_short_id_from_product(product_id)

        with self._lock:
            data = self.load_cart_data(user_id)

            # Check if already in wishlist
            for item in data.wishlist:
                if item["product_id"] == product_id:
                    return True, "already_exists"

            # Add new item
            data.wishlist.append({
                "product_id": product_id,
                "short_id": short_id,
                "added_at": datetime.now().isoformat(),
                "quantity": 1
            })
            self.save_cart_data(data)
            return True, "added"

    def remove_from_wishlist(self, user_id: str, product_id: str) -> Tuple[bool, str]:
        """Remove item from wishlist"""
        user_id = self.clean_phone_number(user_id)

        with self._lock:
            data = self.load_cart_data(user_id)
            original_len = len(data.wishlist)
            data.wishlist = [item for item in data.wishlist if item["product_id"] != product_id]

            if len(data.wishlist) < original_len:
                self.save_cart_data(data)
                return True, "removed"
            return False, "not_found"

    def get_cart(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's cart items"""
        user_id = self.clean_phone_number(user_id)
        data = self.load_cart_data(user_id)
        return data.cart

    def get_wishlist(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's wishlist items"""
        user_id = self.clean_phone_number(user_id)
        data = self.load_cart_data(user_id)
        return data.wishlist

    def get_cart_url(self, user_id: str) -> str:
        """Get URL to view cart"""
        user_id = self.clean_phone_number(user_id)
        return f"{get_base_url()}/view/cart/{user_id}"

    def get_wishlist_url(self, user_id: str) -> str:
        """Get URL to view wishlist"""
        user_id = self.clean_phone_number(user_id)
        return f"{get_base_url()}/view/wishlist/{user_id}"

    def get_cart_with_products(self, user_id: str) -> List[Dict[str, Any]]:
        """Get cart items with full product details"""
        cart = self.get_cart(user_id)
        if not cart:
            return []

        product_ids = [item["product_id"] for item in cart]
        products = fetch_products_by_ids(product_ids)

        result = []
        for item in cart:
            pid = item["product_id"]
            if pid in products:
                product = products[pid]
                result.append({
                    **item,
                    "prod_name": product.get("prod_name"),
                    "price": product.get("price"),
                    "rating": product.get("rating"),
                    "store": product.get("store"),
                    "image_url": product.get("image_url"),
                    "brand": product.get("raw", {}).get("brand"),
                    "colour": product.get("raw", {}).get("colour"),
                    "size": product.get("raw", {}).get("size"),
                })
        return result

    def get_wishlist_with_products(self, user_id: str) -> List[Dict[str, Any]]:
        """Get wishlist items with full product details"""
        wishlist = self.get_wishlist(user_id)
        if not wishlist:
            return []

        product_ids = [item["product_id"] for item in wishlist]
        products = fetch_products_by_ids(product_ids)

        result = []
        for item in wishlist:
            pid = item["product_id"]
            if pid in products:
                product = products[pid]
                result.append({
                    **item,
                    "prod_name": product.get("prod_name"),
                    "price": product.get("price"),
                    "rating": product.get("rating"),
                    "store": product.get("store"),
                    "image_url": product.get("image_url"),
                    "brand": product.get("raw", {}).get("brand"),
                    "colour": product.get("raw", {}).get("colour"),
                    "size": product.get("raw", {}).get("size"),
                })
        return result

    def handle_command(self, phone_number: str, message: str) -> Optional[str]:
        """
        Handle cart/wishlist commands from WhatsApp.
        Returns response message if command was handled, None otherwise.
        """
        user_id = self.clean_phone_number(phone_number)
        text = message.strip().lower()

        # ===== VIEW CART =====
        if text in ["cart", "my cart", "show cart", "view cart", "open cart"]:
            cart = self.get_cart(user_id)
            if not cart:
                return "Your cart is empty! Search for products and add them to your cart."
            return f"🛒 You have {len(cart)} item(s) in your cart.\n\nView & manage: {self.get_cart_url(user_id)}"

        # ===== VIEW WISHLIST =====
        if text in ["wishlist", "my wishlist", "show wishlist", "view wishlist", "saved", "saved items", "favorites"]:
            wishlist = self.get_wishlist(user_id)
            if not wishlist:
                return "Your wishlist is empty! Search for products and save them."
            return f"❤️ You have {len(wishlist)} item(s) in your wishlist.\n\nView & manage: {self.get_wishlist_url(user_id)}"

        # ===== ADD TO CART (with short ID) =====
        # Patterns: "cart A1B2", "add A1B2 to cart", "add to cart A1B2"
        cart_patterns = [
            r"^cart\s+([A-Z0-9]{4})$",
            r"^add\s+([A-Z0-9]{4})\s+to\s+cart$",
            r"^add\s+to\s+cart\s+([A-Z0-9]{4})$",
        ]
        for pattern in cart_patterns:
            match = re.match(pattern, text.upper())
            if match:
                short_id = match.group(1)
                product_id = self._get_product_id_from_short(short_id)
                if product_id:
                    success, status = self.add_to_cart(user_id, product_id, short_id)
                    products = fetch_products_by_ids([product_id])
                    name = products.get(product_id, {}).get("prod_name", short_id)
                    if success:
                        return f"Added *{name}* to your cart!\n\n🛒 View cart: {self.get_cart_url(user_id)}"
                return f"Could not find product with ID: {short_id}"

        # ===== ADD TO CART (last viewed) =====
        if text in ["add to cart", "add cart", "cart it", "buy"]:
            data = self.load_cart_data(user_id)
            if data.last_viewed_product:
                product_id = data.last_viewed_product
                short_id = data.last_viewed_short_id
                success, status = self.add_to_cart(user_id, product_id, short_id)
                products = fetch_products_by_ids([product_id])
                name = products.get(product_id, {}).get("prod_name", short_id or product_id)
                if success:
                    return f"Added *{name}* to your cart!\n\n🛒 View cart: {self.get_cart_url(user_id)}"
            return "No product to add. Please view a product first or specify the ID like: cart A1B2"

        # ===== ADD TO WISHLIST (with short ID) =====
        # Patterns: "save A1B2", "wishlist A1B2", "add A1B2 to wishlist"
        wishlist_patterns = [
            r"^save\s+([A-Z0-9]{4})$",
            r"^wishlist\s+([A-Z0-9]{4})$",
            r"^add\s+([A-Z0-9]{4})\s+to\s+wishlist$",
            r"^add\s+to\s+wishlist\s+([A-Z0-9]{4})$",
        ]
        for pattern in wishlist_patterns:
            match = re.match(pattern, text.upper())
            if match:
                short_id = match.group(1)
                product_id = self._get_product_id_from_short(short_id)
                if product_id:
                    success, status = self.add_to_wishlist(user_id, product_id, short_id)
                    products = fetch_products_by_ids([product_id])
                    name = products.get(product_id, {}).get("prod_name", short_id)
                    if success:
                        if status == "already_exists":
                            return f"*{name}* is already in your wishlist!\n\n❤️ View wishlist: {self.get_wishlist_url(user_id)}"
                        return f"Saved *{name}* to your wishlist!\n\n❤️ View wishlist: {self.get_wishlist_url(user_id)}"
                return f"Could not find product with ID: {short_id}"

        # ===== ADD TO WISHLIST (last viewed) =====
        if text in ["add to wishlist", "save", "save it", "wishlist it", "favorite"]:
            data = self.load_cart_data(user_id)
            if data.last_viewed_product:
                product_id = data.last_viewed_product
                short_id = data.last_viewed_short_id
                success, status = self.add_to_wishlist(user_id, product_id, short_id)
                products = fetch_products_by_ids([product_id])
                name = products.get(product_id, {}).get("prod_name", short_id or product_id)
                if success:
                    if status == "already_exists":
                        return f"*{name}* is already in your wishlist!\n\n❤️ View wishlist: {self.get_wishlist_url(user_id)}"
                    return f"Saved *{name}* to your wishlist!\n\n❤️ View wishlist: {self.get_wishlist_url(user_id)}"
            return "No product to save. Please view a product first or specify the ID like: save A1B2"

        # ===== REMOVE FROM CART =====
        remove_cart_pattern = r"^remove\s+([A-Z0-9]{4})\s+from\s+cart$"
        match = re.match(remove_cart_pattern, text.upper())
        if match:
            short_id = match.group(1)
            product_id = self._get_product_id_from_short(short_id)
            if product_id:
                success, status = self.remove_from_cart(user_id, product_id)
                if success:
                    return f"Removed {short_id} from your cart."
                return f"{short_id} was not in your cart."
            return f"Could not find product with ID: {short_id}"

        # ===== REMOVE FROM WISHLIST =====
        remove_wishlist_pattern = r"^remove\s+([A-Z0-9]{4})\s+from\s+wishlist$"
        match = re.match(remove_wishlist_pattern, text.upper())
        if match:
            short_id = match.group(1)
            product_id = self._get_product_id_from_short(short_id)
            if product_id:
                success, status = self.remove_from_wishlist(user_id, product_id)
                if success:
                    return f"Removed {short_id} from your wishlist."
                return f"{short_id} was not in your wishlist."
            return f"Could not find product with ID: {short_id}"

        # No command matched
        return None


# Singleton instance
_cart_manager: Optional[CartManager] = None


def get_cart_manager() -> CartManager:
    """Get singleton CartManager instance"""
    global _cart_manager
    if _cart_manager is None:
        _cart_manager = CartManager()
    return _cart_manager
