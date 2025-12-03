"""
User Filter Integration for Strontium
Applies user preferences (gender, size, age) to search queries
"""
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Base directory for user data
USER_DATA_BASE_DIR = Path(__file__).parent.parent.parent / "data" / "user_data"

# Categories that support gender filtering
GENDER_FILTER_CATEGORIES = {
    "clothing",
    "footwear",
    "innerwear",
    "fashion",
    "accessories",
    "jewellery",
    "beauty & personal care"
}

# Categories that support size filtering
SIZE_FILTER_CATEGORIES = {
    "clothing",
    "footwear",
    "innerwear"
}


class UserFilterManager:
    """
    Manages user preference-based filtering for search queries.

    Reads user preferences from user_info.json and generates
    additional filter properties for search queries.

    Filter priority order: gender > size > age
    """

    def __init__(self):
        self.base_dir = USER_DATA_BASE_DIR
        self._cache: Dict[str, Dict[str, Any]] = {}

    def clean_phone_number(self, phone: str) -> str:
        """Clean phone number to match user_id format"""
        return phone.replace("+", "").replace("-", "").replace(" ", "")

    def load_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load user info from their user_info.json file.

        Args:
            user_id: Phone number or user ID

        Returns:
            User info dict or None if not found
        """
        user_id = self.clean_phone_number(user_id)

        # Check cache first
        if user_id in self._cache:
            return self._cache[user_id]

        # Load from file
        info_path = self.base_dir / user_id / "user_info.json"
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    user_info = json.load(f)
                    self._cache[user_id] = user_info
                    return user_info
            except Exception as e:
                print(f"Warning: Could not load user info for {user_id}: {e}")

        return None

    def clear_cache(self, user_id: str = None):
        """Clear user info cache"""
        if user_id:
            user_id = self.clean_phone_number(user_id)
            self._cache.pop(user_id, None)
        else:
            self._cache.clear()

    def get_filter_properties(
        self,
        user_id: str,
        category: str
    ) -> List[Tuple[str, float, str]]:
        """
        Get additional filter properties based on user preferences.

        Priority order: gender > size > age

        Args:
            user_id: User ID (phone number)
            category: Product category being searched

        Returns:
            List of (property_value, weight, relation_type) tuples
            Higher weight = higher priority in filtering
        """
        user_info = self.load_user_info(user_id)
        if not user_info:
            return []

        category_lower = category.lower() if category else ""
        additional_properties = []

        # 1. Gender filtering (highest priority - weight 1.8)
        gender = user_info.get("gender_preference")
        if gender and gender != "both" and category_lower in GENDER_FILTER_CATEGORIES:
            # Add gender as a property for filtering
            additional_properties.append((gender, 1.8, "HAS_GENDER"))

        # 2. Size filtering (high priority - weight 1.5)
        # Support both old format (clothing_size: str) and new format (clothing_sizes: list)
        sizes = user_info.get("clothing_sizes") or []
        if not sizes:
            # Fallback to old single-size format
            single_size = user_info.get("clothing_size")
            if single_size:
                sizes = [single_size]

        if sizes and category_lower in SIZE_FILTER_CATEGORIES:
            # Add each size as a separate property
            for size in sizes:
                if size:  # Skip None/empty values
                    additional_properties.append((size, 1.5, "HAS_SIZE"))

        # 3. Age-based style preference (lower priority - weight 0.8)
        age_range = user_info.get("age_range")
        if age_range and category_lower in GENDER_FILTER_CATEGORIES:
            age_style = self._get_age_style_preference(age_range)
            if age_style:
                additional_properties.append((age_style, 0.8, "HAS_STYLE"))

        return additional_properties

    def _get_age_style_preference(self, age_range: str) -> Optional[str]:
        """
        Map age range to style preference.

        Args:
            age_range: User's age range from onboarding

        Returns:
            Style preference string or None
        """
        age_style_map = {
            "under_18": "trendy",
            "18-25": "trendy",
            "26-35": "modern",
            "36-50": "classic",
            "50+": "classic"
        }
        return age_style_map.get(age_range)

    def get_gender_preference(self, user_id: str) -> Optional[str]:
        """
        Get user's gender preference for hard filtering.

        Args:
            user_id: User ID (phone number)

        Returns:
            Gender preference ("male", "female") or None if not set or "both"
        """
        user_info = self.load_user_info(user_id)
        if not user_info:
            return None

        gender = user_info.get("gender_preference")
        if gender and gender != "both":
            return gender
        return None

    def should_apply_gender_filter(self, category: str) -> bool:
        """Check if gender filtering applies to this category"""
        return category.lower() in GENDER_FILTER_CATEGORIES if category else False

    def should_apply_size_filter(self, category: str) -> bool:
        """Check if size filtering applies to this category"""
        return category.lower() in SIZE_FILTER_CATEGORIES if category else False


# Singleton instance
_user_filter_manager: Optional[UserFilterManager] = None


def get_user_filter_manager() -> UserFilterManager:
    """Get singleton UserFilterManager instance"""
    global _user_filter_manager
    if _user_filter_manager is None:
        _user_filter_manager = UserFilterManager()
    return _user_filter_manager
