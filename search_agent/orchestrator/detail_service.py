"""
Product Detail Service

Answers user questions about specific products by:
1. Fetching product data from SQL for all requested product IDs
2. Formatting product info as JSON context
3. Using LLM with strict guidelines to answer based ONLY on provided info
4. Replacing product_id placeholders with vendor contact info for unanswerable queries
"""
import json
import os
import re
from typing import Dict, List, Optional
from search_agent.database.sql_client import SQLClient
from config.config import Config

# Optional NVIDIA API import
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    ChatNVIDIA = None

config = Config


STRONTIUM_DETAIL_ANSWER_PROMPT = """You are Strontium, a helpful sales assistant for BeeKurse.

Answer the customer's question in natural, conversational language. Be concise and direct.

Customer Question: {original_query}

Product Data:
{products_json}

RULES:
1. Answer ONLY using info from the Product Data above. Do NOT make up any information.
2. Write a natural response like a helpful shop assistant would - no bullet points, no product IDs, no technical formatting.
3. If the info is NOT in the data, say: "I don't have that information. For more details, please contact the vendor: <<VENDOR_CONTACT:product_id>>" (replace product_id with actual ID like <<VENDOR_CONTACT:abc-123>>)
4. Keep it short and friendly. No "Additional details", no "Further action required", no repeating info.
5. Just answer what was asked - nothing more.

STRICT - DO NOT:
- Guess or assume dimensions, materials, care instructions, or any details not in the data
- Make up product features not explicitly mentioned in the JSON
- Provide generic or placeholder information
- Add unnecessary follow-up questions or suggestions

Answer:"""


class ProductDetailService:
    """Service for answering detailed questions about products"""

    def __init__(
        self,
        sql_client: SQLClient,
        llm_client=None,
        use_nvidia: bool = True,
        nvidia_api_key: Optional[str] = None,
        **kwargs  # Accept but ignore extra kwargs for backward compatibility
    ):
        """
        Initialize detail service

        Args:
            sql_client: SQL database client
            llm_client: LLM client for generating answers (optional)
            use_nvidia: If True, use NVIDIA API for responses (default: True)
            nvidia_api_key: NVIDIA API key (uses env var if not provided)
        """
        self.sql_client = sql_client
        self.llm_client = None
        self.use_mock = True

        # Initialize NVIDIA LLM if enabled
        if use_nvidia and NVIDIA_AVAILABLE:
            api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
            if api_key:
                try:
                    self.llm_client = ChatNVIDIA(
                        model="nvidia/llama-3.3-nemotron-super-49b-v1",  # Use v1 (NOT v1.5)
                        api_key=api_key,
                        temperature=0.0,  # Deterministic for factual answers
                        max_tokens=4096,  # High token limit for detailed responses
                    ).with_thinking_mode(enabled=False)  # Disable thinking to avoid incomplete output
                    self.use_mock = False
                    print("✅ DetailService: LLM initialized")
                except Exception as e:
                    print(f"⚠️ DetailService: LLM init failed: {e}, using mock")
        elif llm_client:
            self.llm_client = llm_client
            self.use_mock = False

    def _resolve_product_ids(self, product_ids: List[str]) -> List[str]:
        """
        Resolve any short_ids in the list to full product_ids.

        Short IDs are 4-character alphanumeric (e.g., "44QM", "A1B2").
        UUIDs and other formats pass through unchanged.

        Args:
            product_ids: List of product IDs (may include short_ids)

        Returns:
            List with short_ids resolved to full product_ids
        """
        resolved = []
        short_id_pattern = re.compile(r'^[A-Z0-9]{4}$', re.IGNORECASE)

        for pid in product_ids:
            if short_id_pattern.match(pid):
                # Looks like a short_id, try to resolve it
                full_id = self.sql_client.resolve_short_id(pid)
                if full_id:
                    resolved.append(full_id)
                else:
                    # Keep original if resolution fails (will show "not found")
                    resolved.append(pid)
            else:
                # UUID or other format, use as-is
                resolved.append(pid)

        return resolved

    def answer_detail_query(
        self,
        product_ids: List[str],
        original_query: str
    ) -> str:
        """
        Answer a detail query about one or more products

        Args:
            product_ids: List of product IDs to get details about
            original_query: User's original question

        Returns:
            Natural language answer as string
        """
        # Step 0: Resolve any short_ids to full product_ids
        product_ids = self._resolve_product_ids(product_ids)

        # Step 1: Fetch all products from SQL
        products = self.sql_client.get_products_by_ids(product_ids)

        if not products:
            return f"I'm sorry, I couldn't find the product(s) {', '.join(product_ids)} in our system."

        # Step 2: Format products as JSON for context
        products_json = self._format_products_as_json(products)

        # Step 3: Generate answer with LLM
        answer = self._generate_answer(
            original_query=original_query,
            products_json=products_json
        )

        # Step 4: Replace vendor contact placeholders with actual info
        answer = self._replace_vendor_placeholders(answer, products)

        return answer

    def _format_products_as_json(self, products: Dict) -> str:
        """
        Format products as JSON string for LLM context.
        Each product is clearly labeled with its product_id.

        Args:
            products: Dict mapping product_id to SQLProduct

        Returns:
            JSON formatted string with all product information
        """
        products_list = []

        for product_id, product in products.items():
            product_dict = {
                "product_id": product.product_id,
                "name": product.prod_name,
                "category": product.category,
                "subcategory": product.subcategory,
                "price": product.price,
                "brand": product.brand,
                "colour": product.colour,
                "description": product.description,
                "dimensions": product.dimensions,
                "size": product.size,
                "stock": product.stock,
                "rating": product.rating,
                "quantity": product.quantity,
                "quantity_unit": product.quantityunit,
                "store_id": product.store,
            }

            # Add other_properties if present
            if product.other_properties:
                product_dict["other_properties"] = product.other_properties

            # Remove None values for cleaner output
            product_dict = {k: v for k, v in product_dict.items() if v is not None}

            products_list.append(product_dict)

        return json.dumps(products_list, indent=2, ensure_ascii=False)

    def _generate_answer(
        self,
        original_query: str,
        products_json: str
    ) -> str:
        """
        Generate natural language answer using LLM

        Args:
            original_query: User's question
            products_json: JSON formatted product information

        Returns:
            Natural language answer
        """
        if self.use_mock:
            return self._mock_answer(original_query, products_json)

        # Build prompt
        prompt = STRONTIUM_DETAIL_ANSWER_PROMPT.format(
            original_query=original_query,
            products_json=products_json
        )

        # Call LLM
        try:
            if hasattr(self.llm_client, 'invoke'):
                # LangChain style
                response = self.llm_client.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback to mock
                return self._mock_answer(original_query, products_json)
        except Exception as e:
            if config.DEBUG:
                print(f"LLM error: {e}")
            return self._mock_answer(original_query, products_json)

    def _mock_answer(self, original_query: str, products_json: str) -> str:
        """Generate a mock answer for testing"""
        try:
            products = json.loads(products_json)
        except json.JSONDecodeError:
            products = []

        answer = "I'm Strontium, and I'd be happy to help!\n\n"

        if len(products) == 1:
            p = products[0]
            answer += f"**Product {p.get('product_id')}** ({p.get('name', 'Unknown')})\n\n"
            answer += f"Based on the available information:\n"
            if p.get('price'):
                answer += f"• Price: ₹{p['price']}\n"
            if p.get('brand'):
                answer += f"• Brand: {p['brand']}\n"
            if p.get('colour'):
                answer += f"• Color: {p['colour']}\n"
            if p.get('description'):
                answer += f"• Description: {p['description']}\n"
            if p.get('size'):
                answer += f"• Size: {p['size']}\n"
            if p.get('stock'):
                answer += f"• In Stock: {p['stock']} units\n"

            # Add vendor contact placeholder for any missing info
            answer += f"\nFor any other details not mentioned above, please contact the vendor: <<VENDOR_CONTACT:{p.get('product_id')}>>"

        else:
            answer += f"Here's information about the {len(products)} products:\n\n"
            for p in products:
                answer += f"**{p.get('product_id')}** - {p.get('name', 'Unknown')}\n"
                if p.get('price'):
                    answer += f"  • Price: ₹{p['price']}\n"
                if p.get('brand'):
                    answer += f"  • Brand: {p['brand']}\n"
                if p.get('colour'):
                    answer += f"  • Color: {p['colour']}\n"
                answer += "\n"

            answer += "For more details about any specific product, please contact the respective vendor."

        return answer

    def _replace_vendor_placeholders(self, answer: str, products: Dict) -> str:
        """
        Replace <<VENDOR_CONTACT:product_id>> placeholders with actual vendor contact info.

        Args:
            answer: LLM generated answer with placeholders
            products: Dict mapping product_id to SQLProduct

        Returns:
            Answer with vendor contact info replacing placeholders
        """
        # Find all placeholders like <<VENDOR_CONTACT:p-123>>
        pattern = r'<<VENDOR_CONTACT:([^>]+)>>'
        matches = re.findall(pattern, answer)

        for product_id in matches:
            product_id = product_id.strip()
            vendor_contact = None

            # Get vendor info for this product
            if product_id in products:
                product = products[product_id]
                store_id = product.store

                # Try to get vendor contact
                if store_id:
                    vendor_info = self.sql_client.get_vendor_by_id(store_id)
                    if vendor_info:
                        vendor_contact = f"{vendor_info.get('name', 'Vendor')} ({vendor_info.get('phone', 'N/A')})"

                # Fallback to store_contact from product itself
                if not vendor_contact and product.store_contact:
                    vendor_contact = product.store_contact

            # Replace placeholder
            placeholder = f'<<VENDOR_CONTACT:{product_id}>>'
            if vendor_contact:
                answer = answer.replace(placeholder, vendor_contact)
            else:
                answer = answer.replace(placeholder, "vendor contact not available")

        return answer
