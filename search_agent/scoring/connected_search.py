"""
Connected Search (SQ Bonus Score) - Sub-Algorithm B

Finds products related to user's context (previous product or store)
using Knowledge Graph traversal.
"""
from typing import List, Dict, Tuple, Set, Optional

from search_agent.database.sql_client import SQLClient
from search_agent.database.kg_client import KGClient
from search_agent.utils import debug_print_scores
from config.config import Config

config = Config


class ConnectedSearch:
    """Connected search using Knowledge Graph"""

    def __init__(
        self,
        sql_client: SQLClient,
        kg_client: KGClient
    ):
        """
        Initialize connected search

        Args:
            sql_client: SQL database client
            kg_client: Knowledge Graph client
        """
        self.sql_client = sql_client
        self.kg_client = kg_client

    def search(
        self,
        category: str,
        literals: List[Tuple[str, str, float, float]],
        prev_productid: Optional[str] = None,
        prev_storeid: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Execute connected search

        Args:
            category: Product category filter
            literals: List of (field_name, operator, value, buffer) tuples
            prev_productid: Previous product ID (for product-based recommendations)
            prev_storeid: Previous store ID (for store-based boosting)

        Returns:
            Dictionary mapping product_id to bonus_score
        """
        # Case 1: No context available
        if not prev_productid and not prev_storeid:
            return {}

        connected_scores = {}

        # Case 2: Product-based connected search
        if prev_productid:
            product_scores = self._search_by_product(
                prev_productid, category, literals
            )
            connected_scores.update(product_scores)

        # Case 3: Store-based boosting
        if prev_storeid:
            store_scores = self._search_by_store(
                prev_storeid, category, literals
            )
            # Merge with existing scores (don't overwrite, add)
            for product_id, score in store_scores.items():
                connected_scores[product_id] = connected_scores.get(product_id, 0.0) + score

        debug_print_scores(connected_scores, "Connected Search Scores")

        return connected_scores

    def _search_by_product(
        self,
        source_product_id: str,
        category: str,
        literals: List[tuple]
    ) -> Dict[str, float]:
        """
        Find products connected to source product via KG relations

        Args:
            source_product_id: Source product ID
            category: Target product category
            literals: Literal constraints

        Returns:
            Dictionary mapping product_id to bonus score
        """
        # Step 1: Query KG for connected products
        connected_product_ids = self.kg_client.query_connected_products(
            source_product_id=source_product_id,
            category=category,
            relation_types=config.CONNECTED_SEARCH_RELATIONS
        )

        if not connected_product_ids:
            return {}

        # Step 2: Filter by literal constraints
        product_literal_values = self.sql_client.filter_products_by_literals(
            list(connected_product_ids), literals
        )

        # Step 3: Assign bonus score to all passing products
        scores = {
            product_id: config.CONNECTED_BONUS_SCORE
            for product_id in product_literal_values.keys()
        }

        return scores

    def _search_by_store(
        self,
        store_id: str,
        category: str,
        literals: List[tuple]
    ) -> Dict[str, float]:
        """
        Boost products from a specific store

        Args:
            store_id: Store ID
            category: Product category filter
            literals: Literal constraints

        Returns:
            Dictionary mapping product_id to store bonus score
        """
        # Step 1: Get all products from store with category
        store_products = self.sql_client.get_products_by_store(
            store_id=store_id,
            category=category
        )

        if not store_products:
            return {}

        # Extract product IDs
        product_ids = [product.id for product in store_products]

        # Step 2: Filter by literal constraints
        product_literal_values = self.sql_client.filter_products_by_literals(
            product_ids, literals
        )

        # Step 3: Assign store bonus to all passing products
        scores = {
            product_id: config.STORE_BONUS_SCORE
            for product_id in product_literal_values.keys()
        }

        return scores
