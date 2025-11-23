"""
Score Combiner - Step 3 of Unified Search

Combines property scores and connected scores, filters by threshold,
and ranks products.
"""
from typing import Dict, List, Tuple, Optional
from search_agent.models import ProductScore
from search_agent.utils import merge_score_dictionaries, filter_and_rank_products, debug_print_scores
from config.config import Config

config = Config


class ScoreCombiner:
    """Combines and ranks product scores"""

    def __init__(self, sql_client=None):
        """
        Initialize score combiner

        Args:
            sql_client: SQL client for fetching literal values (needed for rank_by_literal)
        """
        self.sql_client = sql_client

    def combine_and_rank(
        self,
        property_scores: Dict[str, float],
        connected_scores: Dict[str, float],
        subcategory_scores: Dict[str, float] = None,
        min_threshold: float = None
    ) -> List[str]:
        """
        Combine property, connected, and subcategory scores, filter, and rank

        Args:
            property_scores: Dictionary of property-based scores
            connected_scores: Dictionary of connected search bonus scores
            subcategory_scores: Dictionary of subcategory similarity bonus scores
            min_threshold: Minimum score threshold (default from config)

        Returns:
            List of product_ids sorted by final score (highest first)
        """
        # Step 1: Merge scores (additive)
        final_scores = merge_score_dictionaries(property_scores, connected_scores)

        # Merge subcategory scores if provided
        if subcategory_scores:
            final_scores = merge_score_dictionaries(final_scores, subcategory_scores)

        debug_print_scores(property_scores, "Property Scores")
        debug_print_scores(connected_scores, "Connected Scores")
        if subcategory_scores:
            debug_print_scores(subcategory_scores, "Subcategory Scores")
        debug_print_scores(final_scores, "Final Combined Scores")

        # Step 2: Filter and rank
        ranked_product_ids = filter_and_rank_products(
            final_scores,
            min_threshold=min_threshold
        )

        return ranked_product_ids

    def get_detailed_scores(
        self,
        property_scores: Dict[str, float],
        connected_scores: Dict[str, float],
        subcategory_scores: Dict[str, float] = None
    ) -> List[ProductScore]:
        """
        Get detailed score breakdown for each product

        Args:
            property_scores: Dictionary of property-based scores
            connected_scores: Dictionary of connected search bonus scores
            subcategory_scores: Dictionary of subcategory similarity bonus scores

        Returns:
            List of ProductScore objects with breakdown
        """
        # Get all unique product IDs
        all_product_ids = set(property_scores.keys()) | set(connected_scores.keys())
        if subcategory_scores:
            all_product_ids |= set(subcategory_scores.keys())

        # Create ProductScore objects
        product_scores = []
        for product_id in all_product_ids:
            score = ProductScore(
                product_id=product_id,
                property_score=property_scores.get(product_id, 0.0),
                connected_score=connected_scores.get(product_id, 0.0),
                subcategory_score=subcategory_scores.get(product_id, 0.0) if subcategory_scores else 0.0
            )
            score.calculate_final_score()
            product_scores.append(score)

        # Sort by final score
        product_scores.sort(key=lambda x: x.final_score, reverse=True)

        return product_scores

    def rank_by_literal(
        self,
        product_ids: List[str],
        sort_literal: Tuple[str, str]
    ) -> List[str]:
        """
        Re-rank products by literal field value (for superlatives like "cheapest")

        Args:
            product_ids: List of product IDs to re-rank
            sort_literal: Tuple of (field_name, direction)
                         E.g., ("price", "asc") for cheapest, ("price", "desc") for most expensive

        Returns:
            List of product IDs sorted by the literal field
        """
        if not product_ids or not sort_literal or not self.sql_client:
            return product_ids

        field_name, direction = sort_literal

        # Fetch products from SQL to get literal values
        products_dict = self.sql_client.get_products_by_ids(product_ids)

        # Create list of (product_id, field_value) tuples
        product_values = []
        for product_id in product_ids:
            product = products_dict.get(product_id)
            if product:
                field_value = getattr(product, field_name, None)
                if field_value is not None:
                    product_values.append((product_id, field_value))

        # Sort by field value
        reverse = (direction == "desc")  # desc = highest first (reverse=True)
        product_values.sort(key=lambda x: x[1], reverse=reverse)

        # Extract sorted product IDs
        sorted_ids = [product_id for product_id, _ in product_values]

        if config.DEBUG:
            print(f"\n=== Literal Ranking ({field_name} {direction}) ===")
            for i, (product_id, value) in enumerate(product_values[:10], 1):
                print(f"  {i}. {product_id}: {field_name}={value}")

        return sorted_ids
