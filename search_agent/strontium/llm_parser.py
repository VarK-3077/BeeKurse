"""
LLM-based query parser for Strontium
Transforms natural language queries into structured JSON
"""
import json
import os
from typing import Union, Optional
from .models import (
    LLMOutput,
    SearchQueryOutput,
    DetailQueryOutput,
    ChatQueryOutput,
    CartActionOutput,
    CartViewOutput,
    ProductRequest
)

# Optional NVIDIA API import
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    ChatNVIDIA = None


# Available product categories (from database)
AVAILABLE_CATEGORIES = [
    "clothing",
    "electronics",
    "furniture",
    "grocery",
    "other"  # For products that don't fit existing categories
]


STRONTIUM_SYSTEM_PROMPT = """You are a query parsing tool. Your task is to extract structured information from natural language queries and output valid JSON.

IMPORTANT: Output ONLY valid JSON. Do not include explanations, commentary, or text outside the JSON object.

---

TASK: Analyze the user query and extract structured information.

First, determine the query type:
- "search": User wants to find new products
- "detail": User wants more information about specific product(s)
- "chat": General conversation, greetings, or non-product questions
- "cart_action": User wants to add or remove items from their cart or wishlist
- "cart_view": User wants to see their cart or wishlist

=== FOR SEARCH QUERIES ===
Extract a list of products requested. For each product:

A. product_category: High-level category (REQUIRED)
   - MUST be ONE of these exact values: clothing, electronics, furniture, grocery, other
   - Use "other" if the product doesn't fit any specific category
   - Examples:
     * "shirt" → "clothing"
     * "blouse" → "clothing"
     * "laptop" → "electronics"
     * "tomatoes" → "grocery"
     * "chair" → "furniture"
     * "rare collectible" → "other"
   - IMPORTANT: Category MUST be lowercase

B. product_subcategory: Specific product type (e.g., "shirt", "polo shirt", "pillow", "tomatoes")
   - This is the detailed product type, while category is the broad classification

C. properties: List of 3-element arrays [property_value, weight, relation_type]
   - IMPORTANT: Each property is EXACTLY 3 elements: [value, weight, relation_type]
   - Examples: [["red", 1.5, "HAS_COLOUR"], ["cotton", 1.2, "HAS_MATERIAL"]]
   - Weight scale: 0.5 (nice-to-have) to 2.0 (critical)
   - Based on how the user phrases the request
   - Infer implicit properties (e.g., "neck pain pillow" → "orthopedic", "firm support")
   - PROPERTIES are attribute values like: red, cotton, casual, Nike, soft

   Common relation types:
   - HAS_COLOUR (for colors: red, blue, black)
   - HAS_MATERIAL (for materials: cotton, leather, wood)
   - HAS_STYLE (for styles: casual, formal, modern)
   - HAS_BRAND (for brands: Nike, Adidas)
   - HAS_SIZE (for sizes: M, L, 10)
   - HAS_FEATURE (for features: waterproof, orthopedic, eco-friendly)
   - HAS_PATTERN (for patterns: striped, floral)
   - HAS_OCCASION_TYPE (for occasions: formal, casual, sports)

C. literals: List of 4-element arrays [field, operator, value, buffer]
   - LITERALS are ONLY these fields: price, rating, size
   - Format: [field, operator, value, buffer]
   - Operators: "<", ">", "=", "<=", ">="
   - Buffer: tolerance (0.0-0.2) for near-misses
   - Examples:
     * "under $30" → ["price", "<", 30.0, 0.1]
     * "rating above 4" → ["rating", ">", 4.0, 0.05]
     * "size 10" → ["size", "=", 10, 0.0]

D. prev_products: List of tuples (product_id, [properties])
   - Any product IDs user references
   - Specific properties mentioned about that product
   - If "similar to p-123" with no specifics: use empty list []

E. is_hq: Boolean
   - TRUE if: "my usual", "my regular", "what I always buy"
   - FALSE otherwise

F. sort_literal: Tuple of (field_name, direction) OR null
   - For superlatives like "cheapest", "most expensive", "highest rated"
   - Field: "price", "rating", or other literal field
   - Direction: "asc" (ascending/lowest first) or "desc" (descending/highest first)
   - Examples:
     * "cheapest shirt" → ["price", "asc"] + literal: ["price", "<", 999999, 0.95]
     * "most expensive watch" → ["price", "desc"] + literal: ["price", ">", 0, 0.95]
     * "highest rated shoes" → ["rating", "desc"] + literal: ["rating", ">", 0, 0.95]
   - IMPORTANT: When sort_literal is set, also add corresponding extreme literal constraint
   - Set to null if no sorting requested

=== FOR DETAIL QUERIES ===
A. original_query: The exact user query (for downstream LLM)
B. product_id: The specific product
C. properties_to_explain: Properties user wants to know
   - General "tell me more" → ["*"]
   - Specific "what material" → ["material"]
   - Multiple "care and washing" → ["care_instructions", "washing_method"]

D. relation_types: Relation types for those properties (same as search)
   - For "material" → ["HAS_MATERIAL"]
   - For "care instructions" → ["HAS_CARE_INSTRUCTIONS"]
   - For "*" (all) → ["*"]

E. query_keywords: Keywords to help find info in product descriptions
   - For "dry cleaning" → ["dry cleaning", "wash", "care"]
   - For "material" → ["material", "fabric", "made of"]
   - For "durability" → ["durability", "long-lasting", "quality"]

=== FOR CHAT QUERIES ===
A. message: The user's message (greeting, question, etc.)

=== FOR CART_ACTION QUERIES ===
User wants to add or remove items from their cart or wishlist.

A. action: "add" or "remove"
B. target: "cart" or "wishlist"
C. product_id: The product ID to add/remove
   - Use "LAST_VIEWED" if user says "this", "it", "that" without specifying a product
   - Look for product IDs in the format "p-xxx" or short IDs like "A1B2"

=== FOR CART_VIEW QUERIES ===
User wants to see their cart or wishlist contents.

A. target: "cart" or "wishlist"

=== EXAMPLES ===

Query: "Red cotton shirt under $30"
{
  "query_type": "search",
  "products": [{
    "product_category": "clothing",
    "product_subcategory": "shirt",
    "properties": [["red", 1.5, "HAS_COLOUR"], ["cotton", 1.2, "HAS_MATERIAL"]],
    "literals": [["price", "<", 30.0, 0.1]],
    "prev_products": [],
    "is_hq": false,
    "sort_literal": null
  }]
}

Query: "Cheapest red shirt"
{
  "query_type": "search",
  "products": [{
    "product_category": "clothing",
    "product_subcategory": "shirt",
    "properties": [["red", 1.5, "HAS_COLOUR"]],
    "literals": [["price", "<", 999999, 0.95]],
    "prev_products": [],
    "is_hq": false,
    "sort_literal": ["price", "asc"]
  }]
}

Query: "Most expensive watch"
{
  "query_type": "search",
  "products": [{
    "product_category": "electronics",
    "product_subcategory": "watch",
    "properties": [],
    "literals": [["price", ">", 0, 0.95]],
    "prev_products": [],
    "is_hq": false,
    "sort_literal": ["price", "desc"]
  }]
}

Query: "Shoes like p-123 but cheaper"
{
  "query_type": "search",
  "products": [{
    "product_category": "clothing",
    "product_subcategory": "shoes",
    "properties": [],
    "literals": [],
    "prev_products": [["p-123", []]],
    "is_hq": false,
    "sort_literal": null
  }]
}

Query: "What material is p-456 made of?"
{
  "query_type": "detail",
  "original_query": "What material is p-456 made of?",
  "product_id": "p-456",
  "properties_to_explain": ["material"],
  "relation_types": ["HAS_MATERIAL"],
  "query_keywords": ["material", "fabric", "made of"]
}

Query: "Does p-789 require dry cleaning?"
{
  "query_type": "detail",
  "original_query": "Does p-789 require dry cleaning?",
  "product_id": "p-789",
  "properties_to_explain": ["care_instructions", "washing_method"],
  "relation_types": ["HAS_CARE_INSTRUCTIONS", "HAS_WASHING_METHOD"],
  "query_keywords": ["dry cleaning", "wash", "care", "cleaning instructions"]
}

Query: "Hello!"
{
  "query_type": "chat",
  "message": "Hello!"
}

Query: "add this to my cart"
{
  "query_type": "cart_action",
  "action": "add",
  "target": "cart",
  "product_id": "LAST_VIEWED"
}

Query: "save it for later"
{
  "query_type": "cart_action",
  "action": "add",
  "target": "wishlist",
  "product_id": "LAST_VIEWED"
}

Query: "remove that from my wishlist"
{
  "query_type": "cart_action",
  "action": "remove",
  "target": "wishlist",
  "product_id": "LAST_VIEWED"
}

Query: "add p-123 to cart"
{
  "query_type": "cart_action",
  "action": "add",
  "target": "cart",
  "product_id": "p-123"
}

Query: "what's in my cart?"
{
  "query_type": "cart_view",
  "target": "cart"
}

Query: "show me my saved items"
{
  "query_type": "cart_view",
  "target": "wishlist"
}

Now process: "{query}"

Output ONLY valid JSON. Do not include any explanation or text outside the JSON object."""


class LLMParser:
    """
    Parses natural language queries using LLM
    """

    def __init__(self, llm_client=None, use_nvidia: bool = False, nvidia_api_key: Optional[str] = None):
        """
        Initialize parser

        Args:
            llm_client: Optional LLM client (OpenAI, Anthropic, etc.)
                       If None and use_nvidia=False, will use mock parsing
            use_nvidia: If True, use NVIDIA API
            nvidia_api_key: NVIDIA API key (will use env var if not provided)
        """
        if use_nvidia and NVIDIA_AVAILABLE:
            # Initialize NVIDIA client
            api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY environment variable is required when use_nvidia=True")
            self.llm_client = ChatNVIDIA(
                model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                api_key=api_key,
                temperature=0.0,  # Deterministic for consistent parsing
                top_p=0.95,
                max_tokens=4096,  # Sufficient for query parsing
            )
            self.use_mock = False
            self.client_type = "nvidia"
        elif llm_client is not None:
            self.llm_client = llm_client
            self.use_mock = False
            self.client_type = "custom"
        else:
            self.llm_client = None
            self.use_mock = True
            self.client_type = "mock"

    def parse(self, query: str) -> Union[SearchQueryOutput, DetailQueryOutput, ChatQueryOutput]:
        """
        Parse a natural language query into structured format

        Args:
            query: Natural language user query

        Returns:
            SearchQueryOutput, DetailQueryOutput, or ChatQueryOutput
        """
        if self.use_mock:
            return self._mock_parse(query)

        # Build prompt
        prompt = STRONTIUM_SYSTEM_PROMPT.replace("{query}", query)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse JSON response
        try:
            # Clean the response - extract JSON from LLM output
            response_cleaned = response.strip()

            # Remove <think> tags (used by some models for reasoning)
            if "</think>" in response_cleaned:
                # Extract everything after </think>
                response_cleaned = response_cleaned.split("</think>", 1)[1].strip()

            # Remove markdown code blocks
            if response_cleaned.startswith("```json"):
                response_cleaned = response_cleaned[7:]  # Remove ```json
            if response_cleaned.startswith("```"):
                response_cleaned = response_cleaned[3:]  # Remove ```
            if response_cleaned.endswith("```"):
                response_cleaned = response_cleaned[:-3]  # Remove trailing ```

            response_cleaned = response_cleaned.strip()

            # Try to find JSON object if still not clean
            if not response_cleaned.startswith("{"):
                # Look for first { and last }
                start_idx = response_cleaned.find("{")
                end_idx = response_cleaned.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    response_cleaned = response_cleaned[start_idx:end_idx+1]

            parsed_json = json.loads(response_cleaned)
            return self._validate_and_convert(parsed_json)
        except json.JSONDecodeError as e:
            print(f"\n[DEBUG] LLM Response:\n{response}\n")
            print(f"[DEBUG] Cleaned Response:\n{response_cleaned}\n")
            raise ValueError(f"LLM returned invalid JSON: {e}\nResponse: {response[:500]}")

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with the prompt

        Args:
            prompt: Full prompt with system instructions

        Returns:
            LLM response as string
        """
        if self.client_type == "nvidia":
            # NVIDIA API via LangChain
            response = self.llm_client.invoke(prompt)
            # LangChain returns an AIMessage object, get the content
            return response.content

        elif self.client_type == "custom":
            # For custom clients, try common interfaces
            # Try LangChain-style first
            if hasattr(self.llm_client, 'invoke'):
                response = self.llm_client.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            # Try OpenAI-style
            elif hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response.choices[0].message.content
            else:
                raise NotImplementedError(
                    f"Unknown LLM client interface. Client has methods: {dir(self.llm_client)}"
                )

        else:
            raise NotImplementedError("LLM client type not recognized")

    def _validate_and_convert(self, parsed_json: dict) -> LLMOutput:
        """
        Validate LLM output and convert to Pydantic models

        Args:
            parsed_json: Parsed JSON from LLM

        Returns:
            Validated Pydantic model
        """
        query_type = parsed_json.get("query_type")

        if query_type == "search":
            return SearchQueryOutput(**parsed_json)
        elif query_type == "detail":
            return DetailQueryOutput(**parsed_json)
        elif query_type == "chat":
            return ChatQueryOutput(**parsed_json)
        elif query_type == "cart_action":
            return CartActionOutput(**parsed_json)
        elif query_type == "cart_view":
            return CartViewOutput(**parsed_json)
        else:
            raise ValueError(f"Unknown query_type: {query_type}")

    def _mock_parse(self, query: str) -> Union[SearchQueryOutput, DetailQueryOutput, ChatQueryOutput]:
        """
        Mock parsing for testing without LLM

        Args:
            query: User query

        Returns:
            Mock parsed output
        """
        query_lower = query.lower()
        query_words = query_lower.split()

        # Check for chat query patterns (whole word match, not substring)
        greetings = ["hello", "hi", "hey"]
        phrases = ["how are you", "good morning"]
        if any(greeting in query_words for greeting in greetings) or any(phrase in query_lower for phrase in phrases):
            return ChatQueryOutput(
                query_type="chat",
                message=query
            )

        # Check for detail query patterns
        if "what" in query_lower or "tell me" in query_lower or "about p-" in query_lower or "does p-" in query_lower:
            # Extract product ID
            product_id = "p-456"  # Default
            for word in query.split():
                if word.startswith("p-"):
                    product_id = word.strip("?.,")
                    break

            # Determine properties, relation types, and keywords
            if "material" in query_lower:
                properties = ["material"]
                relation_types = ["HAS_MATERIAL"]
                keywords = ["material", "fabric", "made of"]
            elif "dry clean" in query_lower or "wash" in query_lower or "care" in query_lower:
                properties = ["care_instructions", "washing_method"]
                relation_types = ["HAS_CARE_INSTRUCTIONS", "HAS_WASHING_METHOD"]
                keywords = ["dry cleaning", "wash", "care", "cleaning instructions"]
            elif "tell me more" in query_lower or "about" in query_lower:
                properties = ["*"]
                relation_types = ["*"]
                keywords = []
            else:
                properties = ["*"]
                relation_types = ["*"]
                keywords = []

            return DetailQueryOutput(
                query_type="detail",
                original_query=query,
                product_id=product_id,
                properties_to_explain=properties,
                relation_types=relation_types,
                query_keywords=keywords
            )

        # Search query - simple mock
        subcategory = "shirt"  # Default
        properties = []
        literals = []
        prev_products = []
        is_hq = False

        # Detect subcategory
        for product_type in ["blouse", "shirt", "pillow", "shoes", "tomatoes", "pants", "watch"]:
            if product_type in query_lower:
                subcategory = product_type
                break

        # Detect properties (with relation types)
        if "black" in query_lower:
            properties.append(["black", 1.5, "HAS_COLOUR"])
        if "red" in query_lower:
            properties.append(["red", 1.5, "HAS_COLOUR"])
        if "blue" in query_lower:
            properties.append(["blue", 1.5, "HAS_COLOUR"])
        if "cotton" in query_lower:
            properties.append(["cotton", 1.2, "HAS_MATERIAL"])
        if "casual" in query_lower:
            properties.append(["casual", 1.3, "HAS_STYLE"])
        if "neck pain" in query_lower:
            properties.extend([
                ["orthopedic", 1.5, "HAS_FEATURE"],
                ["firm_support", 1.3, "HAS_FEATURE"],
                ["memory_foam", 1.0, "HAS_MATERIAL"]
            ])

        # Detect literals
        if "under" in query_lower or "<" in query_lower:
            # Extract price
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() in ["under", "$", "price"]:
                    try:
                        price = float(words[i+1].replace("$", ""))
                        literals.append(["price", "<", price, 0.1])
                    except (IndexError, ValueError):
                        pass

        # Detect prev_products
        for word in query.split():
            if word.startswith("p-"):
                product_id = word.strip("?.,")
                prev_products.append([product_id, []])

        # Detect HQ
        if any(phrase in query_lower for phrase in ["my usual", "my regular", "always buy"]):
            is_hq = True

        # Detect sort_literal (superlatives)
        sort_literal = None
        if "cheapest" in query_lower or "lowest price" in query_lower:
            sort_literal = ("price", "asc")
            literals.append(["price", "<", 999999, 0.95])
        elif "most expensive" in query_lower or "costliest" in query_lower or "highest price" in query_lower:
            sort_literal = ("price", "desc")
            literals.append(["price", ">", 0, 0.95])
        elif "highest rated" in query_lower or "best rated" in query_lower:
            sort_literal = ("rating", "desc")
            literals.append(["rating", ">", 0, 0.95])

        # Map subcategory to category (simple mapping for mock)
        category_map = {
            "shirt": "clothing",
            "blouse": "clothing",
            "pants": "clothing",
            "pillow": "furniture",
            "shoes": "footwear",
            "tomatoes": "grocery",
            "watch": "electronics"
        }
        category = category_map.get(subcategory, "other")

        return SearchQueryOutput(
            query_type="search",
            products=[ProductRequest(
                product_category=category,
                product_subcategory=subcategory,
                properties=properties,
                literals=literals,
                prev_products=prev_products,
                is_hq=is_hq,
                sort_literal=sort_literal
            )]
        )
