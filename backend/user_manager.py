"""
User Management Module for BeeKurse WhatsApp Backend
Handles user registration, onboarding, and profile management
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict, field
import threading


# Constants
USER_DATA_BASE_DIR = Path(__file__).parent.parent / "data" / "user_data"
REGISTRY_FILE = USER_DATA_BASE_DIR / "user_registry.json"


class OnboardingState(Enum):
    """Onboarding progress states"""
    NOT_STARTED = "not_started"
    AWAITING_CONSENT = "awaiting_consent"
    AWAITING_GENDER = "awaiting_gender"
    AWAITING_AGE = "awaiting_age"
    AWAITING_SIZE = "awaiting_size"
    AWAITING_USAGE_CONSENT = "awaiting_usage_consent"
    COMPLETED = "completed"


@dataclass
class UserInfo:
    """User profile data structure"""
    user_id: str                    # Phone number (cleaned)
    phone_number: str               # Original phone number
    created_at: str
    onboarding_state: str
    onboarding_completed: bool

    # Consent flags
    data_collection_consent: Optional[bool] = None   # True if user opted in to share preferences
    usage_data_consent: Optional[bool] = None        # True if user allows activity tracking

    # Onboarding answers (nullable if opted out)
    gender_preference: Optional[str] = None          # "male", "female", "both"
    age_range: Optional[str] = None                  # "under_18", "18-25", "26-35", "36-50", "50+"
    clothing_size: Optional[str] = None              # "XS", "S", "M", "L", "XL", "XXL", None (skipped)

    # Metadata
    last_active: Optional[str] = None
    message_count: int = 0


class OnboardingConfig:
    """Onboarding questions and valid responses"""

    WELCOME_MESSAGE = (
        "Hey there! Welcome to BeeKurse! I'm Strontium, your curator.\n"
        "I'd love to learn a few things about you before we get started so I can find products that match your style better.\n\n"
        "Your data stays safe with us - we only use it to improve your shopping experience, nothing else.\n\n"
        "Would you like to share a few quick preferences?\n"
        "1. Yes, let's do it!\n"
        "2. No thanks, skip for now"
    )

    QUESTIONS = {
        OnboardingState.AWAITING_CONSENT: {
            "question": WELCOME_MESSAGE,
            "valid_responses": {
                "1": True, "yes": True, "y": True, "sure": True, "ok": True, "okay": True,
                "2": False, "no": False, "n": False, "skip": False, "later": False
            },
            "field": "data_collection_consent",
            "next_state_if_true": OnboardingState.AWAITING_GENDER,
            "next_state_if_false": OnboardingState.AWAITING_USAGE_CONSENT
        },
        OnboardingState.AWAITING_GENDER: {
            "question": (
                "Awesome! Let's get started.\n\n"
                "Q1/3: Do you usually shop for male or female products?\n"
                "1. Male\n"
                "2. Female\n"
                "3. Both"
            ),
            "valid_responses": {
                "1": "male", "male": "male", "m": "male", "man": "male", "men": "male",
                "2": "female", "female": "female", "f": "female", "woman": "female", "women": "female",
                "3": "both", "both": "both", "b": "both", "all": "both"
            },
            "field": "gender_preference",
            "next_state": OnboardingState.AWAITING_AGE
        },
        OnboardingState.AWAITING_AGE: {
            "question": (
                "Got it! Q2/3: What's your age range? (This helps me suggest age-appropriate styles)\n"
                "1. Under 18\n"
                "2. 18-25\n"
                "3. 26-35\n"
                "4. 36-50\n"
                "5. 50+"
            ),
            "valid_responses": {
                "1": "under_18", "under 18": "under_18", "under18": "under_18", "<18": "under_18",
                "2": "18-25", "18 to 25": "18-25", "18-25": "18-25",
                "3": "26-35", "26 to 35": "26-35", "26-35": "26-35",
                "4": "36-50", "36 to 50": "36-50", "36-50": "36-50",
                "5": "50+", "50 plus": "50+", "over 50": "50+", ">50": "50+"
            },
            "field": "age_range",
            "next_state": OnboardingState.AWAITING_SIZE
        },
        OnboardingState.AWAITING_SIZE: {
            "question": (
                "Almost done! Q3/3: What's your typical clothing size?\n"
                "1. XS\n"
                "2. S\n"
                "3. M\n"
                "4. L\n"
                "5. XL\n"
                "6. XXL\n"
                "7. Skip this one"
            ),
            "valid_responses": {
                "1": "XS", "xs": "XS",
                "2": "S", "s": "S", "small": "S",
                "3": "M", "m": "M", "medium": "M",
                "4": "L", "l": "L", "large": "L",
                "5": "XL", "xl": "XL", "extra large": "XL",
                "6": "XXL", "xxl": "XXL", "2xl": "XXL",
                "7": None, "skip": None, "later": None, "no": None
            },
            "field": "clothing_size",
            "next_state": OnboardingState.AWAITING_USAGE_CONSENT
        },
        OnboardingState.AWAITING_USAGE_CONSENT: {
            "question": (
                "One last thing! Would you like me to learn from your searches and purchases "
                "to give you even better recommendations over time?\n\n"
                "1. Yes, sounds helpful!\n"
                "2. No, don't track my activity"
            ),
            "valid_responses": {
                "1": True, "yes": True, "y": True, "sure": True, "ok": True, "okay": True,
                "2": False, "no": False, "n": False, "don't": False, "dont": False
            },
            "field": "usage_data_consent",
            "next_state": OnboardingState.COMPLETED
        }
    }

    COMPLETION_MESSAGE = (
        "You're all set! Thanks for sharing - I'll use this to find you the perfect products.\n\n"
        "Try searching with:\n"
        "• \"Red cotton shirt under 500\"\n"
        "• \"Show me trendy sneakers\"\n\n"
        "What can I help you find today?"
    )

    COMPLETION_MESSAGE_SKIPPED = (
        "No problem at all! You can always update your preferences later.\n\n"
        "I'm ready to help you find amazing products. Try searching with:\n"
        "• \"Red cotton shirt under 500\"\n"
        "• \"Show me trendy sneakers\"\n\n"
        "What can I help you find today?"
    )


class UserRegistry:
    """Thread-safe user registry for quick existence checks"""

    def __init__(self):
        self._lock = threading.Lock()
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load registry from disk"""
        if REGISTRY_FILE.exists():
            try:
                with open(REGISTRY_FILE, 'r') as f:
                    self._registry = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Registry file corrupted, starting fresh")
                self._registry = {}

    def _save(self):
        """Save registry to disk"""
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(self._registry, f, indent=2)

    def exists(self, user_id: str) -> bool:
        """Check if user exists"""
        with self._lock:
            return user_id in self._registry

    def add(self, user_id: str, metadata: Dict[str, Any]):
        """Add user to registry"""
        with self._lock:
            self._registry[user_id] = metadata
            self._save()

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user metadata from registry"""
        with self._lock:
            return self._registry.get(user_id)

    def update(self, user_id: str, metadata: Dict[str, Any]):
        """Update user metadata"""
        with self._lock:
            if user_id in self._registry:
                self._registry[user_id].update(metadata)
                self._save()


class UserManager:
    """Main user management class"""

    def __init__(self):
        self.registry = UserRegistry()
        self.base_dir = USER_DATA_BASE_DIR

    def clean_phone_number(self, phone: str) -> str:
        """Clean phone number to use as user_id"""
        return phone.replace("+", "").replace("-", "").replace(" ", "")

    def get_user_folder(self, user_id: str) -> Path:
        """Get path to user's folder"""
        return self.base_dir / user_id

    def get_user_info_path(self, user_id: str) -> Path:
        """Get path to user's info file"""
        return self.get_user_folder(user_id) / "user_info.json"

    def load_user_info(self, user_id: str) -> Optional[UserInfo]:
        """Load user info from disk"""
        info_path = self.get_user_info_path(user_id)
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    data = json.load(f)
                    return UserInfo(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️ Error loading user info for {user_id}: {e}")
                return None
        return None

    def save_user_info(self, user_info: UserInfo):
        """Save user info to disk"""
        folder = self.get_user_folder(user_info.user_id)
        folder.mkdir(parents=True, exist_ok=True)

        info_path = self.get_user_info_path(user_info.user_id)
        with open(info_path, 'w') as f:
            json.dump(asdict(user_info), f, indent=2)

    def create_new_user(self, phone_number: str) -> UserInfo:
        """Create a new user"""
        user_id = self.clean_phone_number(phone_number)
        now = datetime.now().isoformat()

        user_info = UserInfo(
            user_id=user_id,
            phone_number=phone_number,
            created_at=now,
            onboarding_state=OnboardingState.AWAITING_CONSENT.value,
            onboarding_completed=False,
            last_active=now,
            message_count=1
        )

        # Save to disk
        self.save_user_info(user_info)

        # Add to registry
        self.registry.add(user_id, {
            "created_at": now,
            "onboarding_completed": False
        })

        return user_info

    def process_message(self, phone_number: str, message_text: str) -> Optional[str]:
        """
        Process incoming message for user management

        Returns:
            - Onboarding response message if in onboarding flow
            - None if user should proceed to normal search flow
        """
        user_id = self.clean_phone_number(phone_number)

        # Check if user exists
        if not self.registry.exists(user_id):
            # New user - create and start onboarding
            user_info = self.create_new_user(phone_number)
            print(f"👋 New user registered: {user_id}")
            return self._get_onboarding_question(OnboardingState.AWAITING_CONSENT)

        # Existing user - load info
        user_info = self.load_user_info(user_id)
        if not user_info:
            # Registry entry exists but no file - recreate
            user_info = self.create_new_user(phone_number)
            print(f"🔄 Recreated user info for: {user_id}")
            return self._get_onboarding_question(OnboardingState.AWAITING_CONSENT)

        # Update activity
        user_info.last_active = datetime.now().isoformat()
        user_info.message_count += 1

        # Check if onboarding is complete
        if user_info.onboarding_completed:
            self.save_user_info(user_info)
            return None  # Proceed to normal flow

        # Process onboarding response
        return self._process_onboarding(user_info, message_text)

    def _get_onboarding_question(self, state: OnboardingState) -> str:
        """Get the question for a given onboarding state"""
        if state in OnboardingConfig.QUESTIONS:
            return OnboardingConfig.QUESTIONS[state]["question"]
        return OnboardingConfig.COMPLETION_MESSAGE

    def _process_onboarding(self, user_info: UserInfo, message_text: str) -> str:
        """Process onboarding response and return next question"""
        current_state = OnboardingState(user_info.onboarding_state)

        if current_state not in OnboardingConfig.QUESTIONS:
            # Already completed or invalid state
            user_info.onboarding_completed = True
            user_info.onboarding_state = OnboardingState.COMPLETED.value
            self.save_user_info(user_info)
            self.registry.update(user_info.user_id, {"onboarding_completed": True})
            return OnboardingConfig.COMPLETION_MESSAGE

        config = OnboardingConfig.QUESTIONS[current_state]
        normalized = message_text.strip().lower()

        # Try to match response
        answer = config["valid_responses"].get(normalized)

        # If no exact match found, check if it's an invalid response
        if answer is None and normalized not in config["valid_responses"]:
            # Invalid response - repeat question
            return (
                "I didn't quite get that. Please choose from the options:\n\n"
                + config["question"]
            )

        # Store answer
        field_name = config["field"]
        setattr(user_info, field_name, answer)

        # Determine next state
        if current_state == OnboardingState.AWAITING_CONSENT:
            # Branching logic for consent
            if answer:  # User opted in
                next_state = config["next_state_if_true"]
            else:  # User opted out
                next_state = config["next_state_if_false"]
        else:
            next_state = config["next_state"]

        user_info.onboarding_state = next_state.value

        if next_state == OnboardingState.COMPLETED:
            user_info.onboarding_completed = True
            self.save_user_info(user_info)
            self.registry.update(user_info.user_id, {"onboarding_completed": True})

            # Return appropriate completion message
            if user_info.data_collection_consent:
                return OnboardingConfig.COMPLETION_MESSAGE
            else:
                return OnboardingConfig.COMPLETION_MESSAGE_SKIPPED

        self.save_user_info(user_info)
        return self._get_onboarding_question(next_state)


# Singleton instance
_user_manager: Optional[UserManager] = None


def get_user_manager() -> UserManager:
    """Get singleton UserManager instance"""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager
