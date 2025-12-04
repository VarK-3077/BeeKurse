"""
Message History Manager for BeeKurse
Tracks WhatsApp message IDs -> product IDs for contextual replies
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import threading

USER_DATA_BASE_DIR = Path(__file__).parent.parent / "data" / "user_data"


class MessageHistoryManager:
    """
    Manages message ID to product ID mappings for users.

    When we send a product image/details, we track the WhatsApp message ID.
    When user replies to that message, we can look up which product they're referring to.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.base_dir = USER_DATA_BASE_DIR
        self._cache: Dict[str, Dict[str, Dict]] = {}

    def clean_phone_number(self, phone: str) -> str:
        """Clean phone number to match user_id format"""
        return phone.replace("+", "").replace("-", "").replace(" ", "")

    def get_history_file_path(self, user_id: str) -> Path:
        """Get path to user's message history file"""
        return self.base_dir / user_id / "message_history.json"

    def load_history(self, user_id: str) -> Dict[str, Any]:
        """Load message history from disk"""
        history_path = self.get_history_file_path(user_id)
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Error loading message history for {user_id}: {e}")
        return {"mappings": {}, "updated_at": None}

    def save_history(self, user_id: str, history: Dict[str, Any]):
        """Save message history to disk"""
        history_path = self.get_history_file_path(user_id)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history["updated_at"] = datetime.now().isoformat()
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def register_message(
        self,
        user_id: str,
        message_id: str,
        product_id: str,
        short_id: Optional[str] = None
    ):
        """
        Register a sent message ID with its associated product ID.

        Args:
            user_id: User phone number
            message_id: WhatsApp message ID (wamid)
            product_id: Product ID (PID-xxx format)
            short_id: Short product ID (4-char like A1B2)
        """
        user_id = self.clean_phone_number(user_id)

        with self._lock:
            # Update cache
            if user_id not in self._cache:
                self._cache[user_id] = {}

            self._cache[user_id][message_id] = {
                "product_id": product_id,
                "short_id": short_id,
                "timestamp": datetime.now().isoformat()
            }

            # Persist to disk
            history = self.load_history(user_id)
            history["mappings"][message_id] = self._cache[user_id][message_id]
            self.save_history(user_id, history)

    def get_product_from_message(
        self,
        user_id: str,
        message_id: str
    ) -> Optional[Dict[str, str]]:
        """
        Get product info from a message ID (when user replies to a message).

        Args:
            user_id: User phone number
            message_id: WhatsApp message ID being replied to

        Returns:
            Dict with product_id and short_id, or None if not found
        """
        user_id = self.clean_phone_number(user_id)

        # Check cache first
        if user_id in self._cache and message_id in self._cache[user_id]:
            return self._cache[user_id][message_id]

        # Load from disk
        history = self.load_history(user_id)
        mapping = history.get("mappings", {}).get(message_id)

        if mapping:
            # Update cache
            with self._lock:
                if user_id not in self._cache:
                    self._cache[user_id] = {}
                self._cache[user_id][message_id] = mapping
            return mapping

        return None

    def clear_user_history(self, user_id: str):
        """Clear message history for a user"""
        user_id = self.clean_phone_number(user_id)
        with self._lock:
            self._cache.pop(user_id, None)
        history_path = self.get_history_file_path(user_id)
        if history_path.exists():
            history_path.unlink()


# Singleton instance
_message_history: Optional[MessageHistoryManager] = None


def get_message_history() -> MessageHistoryManager:
    """Get singleton MessageHistoryManager instance"""
    global _message_history
    if _message_history is None:
        _message_history = MessageHistoryManager()
    return _message_history
