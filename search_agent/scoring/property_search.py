"""
Property Search (RQ Score) - Sub-Algorithm A

Finds products based on semantically similar properties using:
- Path 1: Main VDB (product embeddings)
- Path 2: Property VDB + Relation VDB + Knowledge Graph
"""
from typing import List, Dict, Tuple
from collections import defaultdict

from search_agent.database.sql_client import SQLClient
from search_agent.database.vdb_client import MainVDBClient, PropertyVDBClient, RelationVDBClient
from search_agent.database.kg_client import KGClient
from search_agent.utils import apply_literal_filters_with_penalties, merge_score_dictionaries, debug_print_scores
from config.config import Config

config = Config


class PropertySearch:
    """Property-based search using VDB and KG"""

    def __init__(
        self,
        sql_client: SQLClient,
        main_vdb_client: MainVDBClient,
        property_vdb_client: PropertyVDBClient,
        relation_vdb_client: RelationVDBClient,
        kg_client: KGClient
    ):
        """
        Initialize property search

        Args:
            sql_client: SQL database client
            main_vdb_client: Main VDB client
            property_vdb_client: Property VDB client
            relation_vdb_client: Relation VDB client
            kg_client: Knowledge Graph client
        """
        self.sql_client = sql_client
        self.main_vdb = main_vdb_client
        self.property_vdb = property_vdb_client
        self.relation_vdb = relation_vdb_client
        self.kg_client = kg_client

    @staticmethod
    def _strip_property_prefix(property_name: str) -> str:
        """
        Strip prefix from Property VDB format to get KG format

        Args:
            property_name: Property in VDB format (e.g., "Color:Red")

        Returns:
            Property value in KG format (e.g., "red")

        Examples:
            "Color:Red" -> "red"
            "Style:Casual" -> "casual"
            "Material:Cotton" -> "cotton"
        """
        if ":" in property_name:
            return property_name.split(":", 1)[1].lower()
        return property_name.lower()

    def search(
        self,
        category: str,
        subcategory: str,
        properties: List[Tuple[str, float, str]],
        literals: List[Tuple[str, str, float, float]]
    ) -> Dict[str, float]:
        """
        Execute property search

        Args:
            category: Product category for strict filtering (e.g., "clothing")
            subcategory: Product subcategory for semantic matching (e.g., "shirt", "polo shirt")
            properties: List of (property_value, importance_weight, relation_type) tuples
            literals: List of (field_name, operator, value, buffer) tuples

        Returns:
            Dictionary mapping product_id to property_score
        """
        if not properties:
            # Fallback: search by subcategory when no explicit properties
            return self._search_main_vdb(
                category, subcategory,
                property_query=subcategory,  # Use subcategory as query
                importance=1.0,  # Default importance
                literals=literals
            )

        # Initialize candidate scores
        candidate_scores = {}

        # Process each property
        for property_value, importance, relation_type in properties:
            # Path 1: Main VDB Search
            main_vdb_scores = self._search_main_vdb(
                category, subcategory, property_value, importance, literals
            )

            # Path 2: Property VDB + KG Search
            property_kg_scores = self._search_property_kg(
                category, subcategory, property_value, importance, literals, relation_type
            )

            # Merge scores from both paths (sum)
            property_scores = merge_score_dictionaries(main_vdb_scores, property_kg_scores)

            # Add to candidate scores
            candidate_scores = merge_score_dictionaries(candidate_scores, property_scores)

            debug_print_scores(property_scores, f"Property '{property_value}' Scores")

        debug_print_scores(candidate_scores, "Total Property Search Scores")

        return candidate_scores

    def _search_main_vdb(
        self,
        category: str,
        subcategory: str,
        property_query: str,
        importance: float,
        literals: List[tuple]
    ) -> Dict[str, float]:
        """
        Path 1: Search Main VDB for products

        Args:
            category: Product category for strict filtering
            subcategory: Product subcategory for semantic matching
            property_query: Property to search
            importance: Importance weight
            literals: Literal constraints

        Returns:
            Dictionary mapping product_id to score
        """
        # Step 1: Query Main VDB
        vdb_results = self.main_vdb.search_products(
            category=category,
            subcategory=subcategory,
            property_query=property_query
        )

        if config.DEBUG:
            print(f"\n[DEBUG] PropertySearch._search_main_vdb():")
            print(f"  Query: '{property_query}' (subcategory: {subcategory})")
            print(f"  Main VDB results: {len(vdb_results)} products")
            if vdb_results:
                print(f"  Sample IDs: {[r.id[:20] + '...' for r in vdb_results[:3]]}")

        if not vdb_results:
            return {}

        # Extract product IDs and similarity scores
        product_ids = [result.id for result in vdb_results]
        similarity_scores = {result.id: result.similarity for result in vdb_results}

        # Step 2: SQL lookup for literal filtering
        product_literal_values = self.sql_client.filter_products_by_literals(
            product_ids, literals
        )

        if config.DEBUG:
            print(f"  After SQL literal filtering: {len(product_literal_values)} products passed")
            if len(product_literal_values) < len(product_ids):
                filtered_out = set(product_ids) - set(product_literal_values.keys())
                print(f"  ⚠️ Filtered out {len(filtered_out)} products by literals")
                print(f"  Literal constraints: {literals}")

        # Step 3: Calculate penalties
        penalties = apply_literal_filters_with_penalties(
            product_literal_values, literals
        )

        # Step 4: Calculate final scores
        scores = {}
        for product_id in product_literal_values.keys():
            vdb_similarity = similarity_scores.get(product_id, 0.0)
            penalty = penalties.get(product_id, 0.0)
            score = (vdb_similarity * importance) + penalty
            scores[product_id] = score

        return scores

    def _search_property_kg(
        self,
        category: str,
        subcategory: str,
        property_query: str,
        importance: float,
        literals: List[tuple],
        relation_type: str
    ) -> Dict[str, float]:
        """
        Path 2: Search Property VDB + Relation VDB + Knowledge Graph

        Args:
            category: Product category label for strict filtering (e.g., "clothing")
            subcategory: Product subcategory (not used in KG queries, only for VDB)
            property_query: Property to search
            importance: Importance weight
            literals: Literal constraints
            relation_type: Relation type to seed Relation VDB search

        Returns:
            Dictionary mapping product_id to score

        Note:
            KG filters by category label (:Product:Clothing)
            Subcategory embeddings are retrieved from SQL for scoring
        """
        # Step 1: Query Property VDB for similar properties
        property_results = self.property_vdb.search_properties(property_query)

        if config.DEBUG:
            print(f"\n[DEBUG] PropertySearch._search_property_kg():")
            print(f"  Query: '{property_query}' (relation: {relation_type})")
            print(f"  Property VDB results: {len(property_results) if property_results else 0} properties")
            if property_results:
                print(f"  Sample properties: {[r.id for r in property_results[:5]]}")

        # Extract property names and similarity scores from VDB results
        property_names_with_prefix = [result.id for result in property_results] if property_results else []
        property_similarity_scores_with_prefix = {result.id: result.similarity for result in property_results} if property_results else {}

        # Step 1.5: Add exact agent property using relation_type
        # E.g., property_query="red" + relation_type="HAS_COLOR" -> "Color:Red"
        if relation_type and ":" not in property_query:
            # Extract property type from relation (HAS_COLOR -> Color)
            property_type = relation_type.replace("HAS_", "").title()
            exact_property_vdb_format = f"{property_type}:{property_query.title()}"

            # Add exact property if not already in results
            if exact_property_vdb_format not in property_names_with_prefix:
                property_names_with_prefix.insert(0, exact_property_vdb_format)
                property_similarity_scores_with_prefix[exact_property_vdb_format] = 1.0  # Perfect match

        if not property_names_with_prefix:
            return {}

        # Step 1.6: Strip prefixes for KG query (Color:Red -> red)
        property_names_for_kg = [self._strip_property_prefix(name) for name in property_names_with_prefix]

        # Create mapping: KG format -> similarity (for later use)
        property_similarity_scores = {
            self._strip_property_prefix(name): property_similarity_scores_with_prefix[name]
            for name in property_names_with_prefix
        }

        # Step 2: Query Relation VDB for relevant relations
        # Use relation_type to seed the similarity search
        relation_query = f"{relation_type} {property_query}"
        relation_results = self.relation_vdb.search_relations(
            relation_query=relation_query,
            top_k=2  # Get top 2 relations
        )

        if not relation_results:
            # If no relations found, try using just the relation type
            relation_results = self.relation_vdb.search_relations(
                relation_query=relation_type,
                top_k=2
            )

        relation_types = [result.id for result in relation_results]

        # Step 3: Query Knowledge Graph
        kg_products = self.kg_client.query_products_by_properties(
            category=category,
            property_names=property_names_for_kg,
            relation_types=relation_types,
            property_similarity_scores=property_similarity_scores
        )

        if config.DEBUG:
            print(f"  KG query results: {len(kg_products) if kg_products else 0} product matches")
            if kg_products:
                unique_products = set(item["product_id"] for item in kg_products)
                print(f"  Unique products from KG: {len(unique_products)}")
                print(f"  Sample IDs: {list(unique_products)[:3]}")

        if not kg_products:
            return {}

        # Aggregate scores by product (max similarity for each product)
        product_max_similarities = defaultdict(float)
        for item in kg_products:
            product_id = item["product_id"]
            similarity = item["similarity"]
            product_max_similarities[product_id] = max(
                product_max_similarities[product_id],
                similarity
            )

        # Get product IDs
        product_ids = list(product_max_similarities.keys())

        # Step 4: SQL lookup for literal filtering
        product_literal_values = self.sql_client.filter_products_by_literals(
            product_ids, literals
        )

        if config.DEBUG:
            print(f"  After SQL literal filtering: {len(product_literal_values)} products passed")
            if len(product_literal_values) < len(product_ids):
                filtered_out = set(product_ids) - set(product_literal_values.keys())
                print(f"  ⚠️ Filtered out {len(filtered_out)} products by literals")
                print(f"  Literal constraints: {literals}")

        # Step 5: Calculate penalties
        penalties = apply_literal_filters_with_penalties(
            product_literal_values, literals
        )

        # Step 6: Calculate final scores
        scores = {}
        for product_id in product_literal_values.keys():
            vdb_similarity = product_max_similarities.get(product_id, 0.0)
            penalty = penalties.get(product_id, 0.0)
            score = (vdb_similarity * importance) + penalty
            scores[product_id] = score

        return scores
