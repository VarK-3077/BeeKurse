"""
Product Detail Service

Answers user questions about specific products by:
1. Fetching product data from SQL
2. Querying KG for product properties
3. Querying VDB for semantic product information
4. Using LLM (with Strontium identity) to generate natural language answers
"""
from typing import Dict, List, Optional
from search_agent.database.sql_client import SQLClient
from search_agent.database.vdb_client import MainVDBClient, RelationVDBClient
from search_agent.database.kg_client import KGClient
from config.config import Config

config = Config


STRONTIUM_DETAIL_ANSWER_PROMPT = """You are Strontium, an expert sales assistant for BeeKurse, built on the KURSE system.

Your customer is a 'Bee' – they are busy and their time is valuable. Provide helpful, concise answers about products.

CRITICAL GUARDRAIL: Never reveal your underlying model name or that you are an LLM. If asked about your technology, simply say you are Strontium, part of the KURSE system.

---

TASK: Answer the user's question about a product based on the information provided.

User Question: {original_query}

Product Information:
{product_info}

Properties from Knowledge Graph:
{kg_properties}

Product Description Details:
{vdb_details}

---

INSTRUCTIONS:
1. Answer the user's question naturally and helpfully using the information above
2. Be concise but complete - the Bee is busy
3. If specific information is MISSING and you cannot answer the question:
   - Say: "The vendor did not mention this information in the product listing."
   - Offer to provide vendor contact: "Would you like the vendor's contact information so you can ask them directly?"
   - Include vendor contact if available: {vendor_contact}
4. Be friendly and professional
5. Use bullet points for multiple details

Now answer the user's question:"""


class ProductDetailService:
    """Service for answering detailed questions about products"""

    def __init__(
        self,
        sql_client: SQLClient,
        main_vdb_client: MainVDBClient,
        relation_vdb_client: RelationVDBClient,
        kg_client: KGClient,
        llm_client=None
    ):
        """
        Initialize detail service

        Args:
            sql_client: SQL database client
            main_vdb_client: Main VDB client for product embeddings
            relation_vdb_client: Relation VDB client
            kg_client: Knowledge Graph client
            llm_client: LLM client for generating answers (optional, will use mock if None)
        """
        self.sql_client = sql_client
        self.main_vdb = main_vdb_client
        self.relation_vdb = relation_vdb_client
        self.kg_client = kg_client
        self.llm_client = llm_client
        self.use_mock = llm_client is None

    def answer_detail_query(
        self,
        product_id: str,
        original_query: str,
        properties_to_explain: List[str],
        relation_types: List[str],
        query_keywords: List[str]
    ) -> str:
        """
        Answer a detail query about a product

        Args:
            product_id: Product ID
            original_query: User's original question
            properties_to_explain: Properties user wants to know about
            relation_types: Relation types for those properties
            query_keywords: Keywords to search in descriptions

        Returns:
            Natural language answer as string
        """
        # Step 1: Fetch product from SQL
        product = self.sql_client.get_product_by_id(product_id)
        if not product:
            return f"I'm sorry, I couldn't find product {product_id} in our system."

        # Step 2: Fetch vendor information
        vendor_contact = self._get_vendor_contact(product.store)

        # Step 3: Query Relation VDB to get similar relations
        similar_relations = self._query_relation_vdb(relation_types)

        # Step 4: Query KG for product properties
        kg_properties = self._query_kg_properties(
            product_id,
            product.subcategory,
            similar_relations
        )

        # Step 5: Query Main VDB for semantic product information
        vdb_details = self._query_main_vdb_for_details(
            product.prod_name,
            product.subcategory,
            properties_to_explain,
            query_keywords
        )

        # Step 6: Build context for LLM
        product_info = self._format_product_info(product)
        kg_properties_str = self._format_kg_properties(kg_properties)
        vdb_details_str = self._format_vdb_details(vdb_details)
        vendor_contact_str = self._format_vendor_contact(vendor_contact)

        # Step 7: Generate answer with LLM
        answer = self._generate_answer(
            original_query=original_query,
            product_info=product_info,
            kg_properties=kg_properties_str,
            vdb_details=vdb_details_str,
            vendor_contact=vendor_contact_str
        )

        return answer

    def _get_vendor_contact(self, store_id: str) -> Optional[Dict]:
        """
        Get vendor contact information from vendor SQL table

        Args:
            store_id: Store/vendor ID

        Returns:
            Dict with vendor name and contact, or None if not found
        """
        # Try to get vendor from SQL
        # Assuming there's a vendors table with store_id, name, and phone
        try:
            vendor = self.sql_client.get_vendor_by_id(store_id)
            if vendor:
                return {
                    "name": vendor.get("name", store_id),
                    "phone": vendor.get("phone", "N/A")
                }
        except Exception:
            pass

        return None

    def _query_relation_vdb(self, relation_types: List[str]) -> List[str]:
        """
        Query Relation VDB to find similar relations

        Args:
            relation_types: Initial relation types from Strontium

        Returns:
            List of relevant relation types (top 3-5)
        """
        if not relation_types or relation_types == ["*"]:
            return []

        all_relations = []
        for relation_type in relation_types:
            try:
                results = self.relation_vdb.search_relations(
                    relation_query=relation_type,
                    top_k=3
                )
                all_relations.extend([r.id for r in results])
            except Exception:
                pass

        # Remove duplicates while preserving order
        seen = set()
        unique_relations = []
        for rel in all_relations:
            if rel not in seen:
                seen.add(rel)
                unique_relations.append(rel)

        return unique_relations[:5]  # Top 5 unique relations

    def _query_kg_properties(
        self,
        product_id: str,
        subcategory: str,
        relation_types: List[str]
    ) -> Dict[str, List[str]]:
        """
        Query Knowledge Graph for product properties

        Args:
            product_id: Product ID
            subcategory: Product subcategory
            relation_types: Relation types to query

        Returns:
            Dict mapping relation_type -> list of property values
        """
        if not relation_types:
            # Get all properties for this product
            try:
                return self.kg_client.get_all_product_properties(product_id)
            except Exception:
                return {}

        # Query specific relations
        properties_by_relation = {}
        for relation_type in relation_types:
            try:
                properties = self.kg_client.get_product_properties_by_relation(
                    product_id=product_id,
                    relation_type=relation_type
                )
                if properties:
                    properties_by_relation[relation_type] = properties
            except Exception:
                pass

        return properties_by_relation

    def _query_main_vdb_for_details(
        self,
        product_name: str,
        subcategory: str,
        properties_to_explain: List[str],
        query_keywords: List[str]
    ) -> List[Dict]:
        """
        Query Main VDB with property-specific semantic searches

        Args:
            product_name: Product name from SQL
            subcategory: Product subcategory
            properties_to_explain: Properties being asked about
            query_keywords: Keywords to search

        Returns:
            List of relevant product matches with similarities
        """
        vdb_results = []

        # Query 1: For each property, search "{product_name} has {property}"
        if properties_to_explain and properties_to_explain != ["*"]:
            for prop in properties_to_explain:
                query = f"{product_name} has {prop}"
                try:
                    results = self.main_vdb.search_products(
                        subcategory=subcategory,
                        property_query=query,
                        top_k=3
                    )
                    for result in results:
                        vdb_results.append({
                            "query": query,
                            "product_id": result.id,
                            "similarity": result.similarity
                        })
                except Exception:
                    pass

        # Query 2: For each keyword, search "{product_name} {keyword}"
        for keyword in query_keywords:
            query = f"{product_name} {keyword}"
            try:
                results = self.main_vdb.search_products(
                    subcategory=subcategory,
                    property_query=query,
                    top_k=3
                )
                for result in results:
                    vdb_results.append({
                        "query": query,
                        "product_id": result.id,
                        "similarity": result.similarity
                    })
            except Exception:
                pass

        return vdb_results

    def _format_product_info(self, product) -> str:
        """Format product SQL info as string"""
        info = f"Product ID: {product.product_id}\n"
        if hasattr(product, 'prod_name') and product.prod_name:
            info += f"Name: {product.prod_name}\n"
        info += f"Category: {product.subcategory}\n"
        info += f"Price: ${product.price}\n"
        info += f"Stock: {product.stock} units\n"
        if hasattr(product, 'brand') and product.brand:
            info += f"Brand: {product.brand}\n"
        if hasattr(product, 'descrption') and product.descrption:
            info += f"Description: {product.descrption}\n"

        return info

    def _format_kg_properties(self, kg_properties: Dict[str, List[str]]) -> str:
        """Format KG properties as string"""
        if not kg_properties:
            return "No specific properties found in Knowledge Graph."

        lines = []
        for relation_type, values in kg_properties.items():
            lines.append(f"{relation_type}:")
            for value in values:
                lines.append(f"  - {value}")

        return "\n".join(lines)

    def _format_vdb_details(self, vdb_details: List[Dict]) -> str:
        """Format VDB details as string"""
        if not vdb_details:
            return "No additional details found in product descriptions."

        lines = ["Semantic search results:"]
        for detail in vdb_details[:5]:  # Top 5
            lines.append(f"  - Query: '{detail['query']}' (similarity: {detail['similarity']:.2f})")

        return "\n".join(lines)

    def _format_vendor_contact(self, vendor_contact: Optional[Dict]) -> str:
        """Format vendor contact as string"""
        if not vendor_contact:
            return "Vendor contact information not available."

        return f"Vendor: {vendor_contact['name']}, Phone: {vendor_contact['phone']}"

    def _generate_answer(
        self,
        original_query: str,
        product_info: str,
        kg_properties: str,
        vdb_details: str,
        vendor_contact: str
    ) -> str:
        """
        Generate natural language answer using LLM

        Args:
            original_query: User's question
            product_info: Formatted product SQL info
            kg_properties: Formatted KG properties
            vdb_details: Formatted VDB details
            vendor_contact: Formatted vendor contact

        Returns:
            Natural language answer
        """
        if self.use_mock:
            return self._mock_answer(original_query, product_info, kg_properties, vendor_contact)

        # Build prompt
        prompt = STRONTIUM_DETAIL_ANSWER_PROMPT.format(
            original_query=original_query,
            product_info=product_info,
            kg_properties=kg_properties,
            vdb_details=vdb_details,
            vendor_contact=vendor_contact
        )

        # Call LLM
        try:
            if hasattr(self.llm_client, 'invoke'):
                # LangChain style
                response = self.llm_client.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback to mock
                return self._mock_answer(original_query, product_info, kg_properties, vendor_contact)
        except Exception as e:
            if config.DEBUG:
                print(f"LLM error: {e}")
            return self._mock_answer(original_query, product_info, kg_properties, vendor_contact)

    def _mock_answer(
        self,
        original_query: str,
        product_info: str,
        kg_properties: str,
        vendor_contact: str
    ) -> str:
        """Generate a mock answer for testing"""
        answer = f"I'm Strontium, and I'd be happy to help!\n\n"
        answer += f"Based on what I know about this product:\n\n"
        answer += f"{product_info}\n"

        if "No specific properties" not in kg_properties:
            answer += f"\nProduct Properties:\n{kg_properties}\n"

        answer += f"\nRegarding your question: '{original_query}'\n"
        answer += f"The vendor did not mention this specific information in the product listing. "
        answer += f"You can contact them directly for more details.\n\n"
        answer += f"{vendor_contact}"

        return answer
