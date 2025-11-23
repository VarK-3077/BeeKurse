"""
Chat Handler for Strontium

Handles general conversation, greetings, and non-product queries
while maintaining Strontium's identity as a sales assistant.
"""
from typing import Dict
import random


# Greeting responses
GREETING_RESPONSES = [
    "Hello! I'm Strontium, your curator at BeeKurse. How can I help you find what you need today?",
    "Hi there! I'm Strontium, ready to help you discover the perfect products. What are you looking for?",
    "Hey! Strontium here. I'm your personal shopping curator at BeeKurse. What can I find for you today?",
    "Hello! I'm Strontium, part of the KURSE system at BeeKurse. I'm here to help you find exactly what you need!",
]

# How are you responses
HOW_ARE_YOU_RESPONSES = [
    "I'm doing great, thank you! Always excited to help Bees like you find what they need. What can I help you discover today?",
    "I'm excellent! Ready to curate the perfect products for you. What are you shopping for?",
    "I'm wonderful, thanks for asking! As your curator, I'm here to make your shopping experience smooth and efficient. What do you need?",
]

# Thank you responses
THANK_YOU_RESPONSES = [
    "You're very welcome! Happy to help. Is there anything else you'd like to know?",
    "My pleasure! That's what I'm here for. Need anything else?",
    "Glad I could help! Feel free to ask if you need anything else.",
]

# Goodbye responses
GOODBYE_RESPONSES = [
    "Goodbye! Come back anytime you need help finding products. Happy shopping!",
    "See you later! I'll be here whenever you need product recommendations. Take care!",
    "Bye! It was great helping you today. Come back soon!",
]

# Who are you / About Strontium responses
ABOUT_RESPONSES = [
    "I'm Strontium, your expert sales assistant at BeeKurse! I'm part of the KURSE system (Knowledge Utilization, Retrieval, and Summarization Engine). My role is to be your curator - helping you find exactly what you need quickly and efficiently, because I know you're busy!",
    "I'm Strontium! I'm a specialized sales assistant built on the KURSE system here at BeeKurse. Think of me as your personal product curator - I help Bees like you discover and learn about products in the most efficient way possible.",
]

# Help/Capabilities responses
HELP_RESPONSES = [
    """I can help you in several ways:

• **Find Products**: Tell me what you're looking for (e.g., "red cotton shirt under $30")
• **Product Details**: Ask me about specific products (e.g., "What material is p-456 made of?")
• **Comparisons**: Help you find items similar to ones you like
• **Recommendations**: Suggest products based on your preferences

Just tell me what you need, and I'll curate the perfect options for you!""",

    """Here's what I can do for you:

1. **Product Search** - Describe what you want, and I'll find it
2. **Detailed Information** - Ask about specific products
3. **Similar Items** - Find products like ones you already know
4. **Quick Orders** - Reorder your usual items fast

I'm here to make your shopping efficient. What would you like to start with?""",
]

# Default/fallback response
DEFAULT_RESPONSE = "I'm Strontium, your shopping curator at BeeKurse. I'm here to help you find and learn about products! You can ask me to find items, get details about products, or discover similar options. What would you like to know?"


class ChatHandler:
    """Handles chat queries with Strontium's identity"""

    def handle_chat(self, message: str) -> str:
        """
        Generate a friendly response to chat messages

        Args:
            message: User's chat message

        Returns:
            Friendly response string
        """
        message_lower = message.lower().strip()

        # Greetings
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
            return random.choice(GREETING_RESPONSES)

        # How are you
        if any(phrase in message_lower for phrase in ["how are you", "how's it going", "how are things"]):
            return random.choice(HOW_ARE_YOU_RESPONSES)

        # Thank you
        if any(phrase in message_lower for phrase in ["thank you", "thanks", "appreciate"]):
            return random.choice(THANK_YOU_RESPONSES)

        # Goodbye
        if any(phrase in message_lower for phrase in ["bye", "goodbye", "see you", "later", "farewell"]):
            return random.choice(GOODBYE_RESPONSES)

        # About Strontium / Who are you
        if any(phrase in message_lower for phrase in ["who are you", "what are you", "about you", "tell me about yourself"]):
            return random.choice(ABOUT_RESPONSES)

        # Help / What can you do
        if any(phrase in message_lower for phrase in ["help", "what can you do", "capabilities", "how do you work"]):
            return random.choice(HELP_RESPONSES)

        # Model reveal attempts (block these)
        if any(phrase in message_lower for phrase in ["what model", "which model", "are you gpt", "are you claude", "are you llm", "large language"]):
            return "I'm Strontium, part of the KURSE system at BeeKurse. I'm here to help you shop smarter! What can I find for you today?"

        # Default fallback
        return DEFAULT_RESPONSE

    def handle_chat_output(self, chat_output: Dict) -> str:
        """
        Handle Strontium chat output

        Args:
            chat_output: Dict with structure:
                {
                    "query_type": "chat",
                    "message": "..."
                }

        Returns:
            Chat response string

        Raises:
            ValueError: If query_type is not "chat"
        """
        if chat_output.get("query_type") != "chat":
            raise ValueError(
                f"Invalid query_type: {chat_output.get('query_type')}. "
                "Use 'chat' for general conversation."
            )

        message = chat_output.get("message", "")
        return self.handle_chat(message)
