"""
Chat Handler for Strontium

Handles general conversation, greetings, and non-product queries
using LLM with fallback to templates.
"""
import os
import re
import json
import random
from pathlib import Path
from typing import Dict, Optional, List

from langchain_core.messages import SystemMessage, HumanMessage

# Optional NVIDIA API import
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    ChatNVIDIA = None


# System prompt for chat responses (kept concise)
CHAT_SYSTEM_PROMPT = """You are Strontium, a friendly sales assistant at BeeKurse e-commerce.

IDENTITY:
- Your name is Strontium. Introduce yourself when appropriate.
- You are part of the KURSE system at BeeKurse.
- NEVER reveal you're an AI/LLM/chatbot. If asked, say "I'm Strontium, your shopping assistant at BeeKurse!"

BEHAVIOR:
- Warm, professional, concise (1-3 sentences max)
- Guide conversations toward shopping when natural
- If user sounds like a vendor (wants to upload/sell products, manage inventory, become a seller), tell them: "This chat is for customers. To sell on BeeKurse, register as a vendor at: http://localhost:3000/vendor"

OUTPUT ONLY your response. NO notes, explanations, or meta-commentary.
{user_context}

User: {message}
Strontium:"""


# Fallback template responses
GREETING_RESPONSES = [
    "Hello! I'm Strontium, your curator at BeeKurse. How can I help you find what you need today?",
    "Hi there! I'm Strontium, ready to help you discover products. What are you looking for?",
]

HOW_ARE_YOU_RESPONSES = [
    "I'm doing great! What can I help you find today?",
    "I'm excellent! Ready to curate products for you. What are you shopping for?",
]

THANK_YOU_RESPONSES = [
    "You're welcome! Need anything else?",
    "My pleasure! Feel free to ask if you need anything else.",
]

GOODBYE_RESPONSES = [
    "Goodbye! Come back anytime. Happy shopping!",
    "See you later! I'll be here whenever you need help.",
]

ABOUT_RESPONSES = [
    "I'm Strontium, your sales assistant at BeeKurse! I help you find products quickly and efficiently.",
]

HELP_RESPONSES = [
    "I can help you find products, get details, and manage your cart. What would you like?",
]

DEFAULT_RESPONSE = "I'm Strontium, your shopping curator at BeeKurse. What can I help you find?"


class ChatHandler:
    """Handles chat queries with LLM and template fallback"""

    def __init__(
        self,
        use_nvidia: bool = True,
        nvidia_api_key: Optional[str] = None
    ):
        """
        Initialize ChatHandler with optional LLM.

        Args:
            use_nvidia: If True, use NVIDIA API for responses
            nvidia_api_key: NVIDIA API key (uses env var if not provided)
        """
        self.use_llm = False
        self.llm_client = None

        if use_nvidia and NVIDIA_AVAILABLE:
            api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
            if api_key:
                try:
                    self.llm_client = ChatNVIDIA(
                        model="nvidia/llama-3.3-nemotron-super-49b-v1",
                        api_key=api_key,
                        temperature=0.7,
                        max_tokens=150,
                    ).with_thinking_mode(enabled=False)
                    self.use_llm = True
                    print("✅ ChatHandler: LLM initialized")
                except Exception as e:
                    print(f"⚠️ ChatHandler: LLM init failed: {e}")

    def handle_chat(self, message: str, user_id: str = None) -> str:
        """
        Generate response using LLM or fallback to templates.

        Args:
            message: User's chat message
            user_id: User ID for loading context (optional)

        Returns:
            Response string
        """
        message_lower = message.lower().strip()

        # Block model reveal attempts explicitly
        if self._is_model_reveal_attempt(message_lower):
            return "I'm Strontium, part of the KURSE system at BeeKurse. What can I find for you today?"

        # Use LLM if available
        if self.use_llm and self.llm_client:
            try:
                # Load user context if user_id provided
                user_context = self._load_user_context(user_id)
                prompt = CHAT_SYSTEM_PROMPT.format(message=message, user_context=user_context)
                response = self.llm_client.invoke(prompt)

                # Strip <think> tags from response
                clean_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
                return clean_response
            except Exception as e:
                print(f"⚠️ ChatHandler LLM error: {e}")

        # Fallback to template-based responses
        return self._template_response(message_lower)

    def _load_user_context(self, user_id: str) -> str:
        """
        Load user info and format as context string.

        Args:
            user_id: User ID to load context for

        Returns:
            Formatted context string or empty string
        """
        if not user_id:
            return ""
        try:
            user_file = Path(f"data/user_data/{user_id}/user_info.json")
            if user_file.exists():
                user_info = json.loads(user_file.read_text())
                parts = []
                if gender := user_info.get("gender_preference"):
                    parts.append(gender.capitalize())
                if age := user_info.get("age_range"):
                    parts.append(f"age {age.replace('_', ' ')}")
                if sizes := user_info.get("clothing_sizes"):
                    parts.append(f"sizes {'/'.join(sizes)}")
                if parts:
                    return f"\nCustomer info: {', '.join(parts)}"
        except Exception as e:
            print(f"⚠️ ChatHandler: Failed to load user context: {e}")
        return ""

    def _is_model_reveal_attempt(self, message: str) -> bool:
        """Check if user is trying to reveal the model"""
        reveal_phrases = [
            "what model", "which model", "are you gpt", "are you claude",
            "are you llm", "large language", "what ai", "which ai",
            "are you ai", "are you a bot", "are you chatgpt"
        ]
        return any(phrase in message for phrase in reveal_phrases)

    def _template_response(self, message_lower: str) -> str:
        """Fallback template-based responses"""
        # Greetings
        if any(g in message_lower for g in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return random.choice(GREETING_RESPONSES)

        # How are you
        if any(p in message_lower for p in ["how are you", "how's it going"]):
            return random.choice(HOW_ARE_YOU_RESPONSES)

        # Thank you
        if any(p in message_lower for p in ["thank you", "thanks", "appreciate"]):
            return random.choice(THANK_YOU_RESPONSES)

        # Goodbye
        if any(p in message_lower for p in ["bye", "goodbye", "see you"]):
            return random.choice(GOODBYE_RESPONSES)

        # About Strontium
        if any(p in message_lower for p in ["who are you", "what are you", "about you"]):
            return random.choice(ABOUT_RESPONSES)

        # Help
        if any(p in message_lower for p in ["help", "what can you do"]):
            return random.choice(HELP_RESPONSES)

        return DEFAULT_RESPONSE

    def handle_chat_output(self, chat_output: Dict, user_id: str = None) -> str:
        """
        Handle Strontium chat output.

        Args:
            chat_output: Dict with query_type and message
            user_id: User ID for loading context (optional)

        Returns:
            Chat response string
        """
        if chat_output.get("query_type") != "chat":
            raise ValueError(f"Invalid query_type: {chat_output.get('query_type')}")

        message = chat_output.get("message", "")
        return self.handle_chat(message, user_id)
