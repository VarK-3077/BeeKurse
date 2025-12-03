"""
Score Combiner - Step 3 of Unified Search

Combines property scores and connected scores, filters by threshold,
and ranks products.
"""
from typing import Dict, List, Tuple, Optional
from search_agent.models import ProductScore
from search_agent.utils import merge_score_dictionaries, filter_and_rank_products, debug_print_scores
from search_agent.strontium.user_filter import GENDER_FILTER_CATEGORIES
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
        min_threshold: float = None,
        gender_filter: Tuple[str, str] = None
    ) -> Tuple[List[str], Optional[str]]:
        """
        Combine property, connected, and subcategory scores, filter, and rank

        Args:
            property_scores: Dictionary of property-based scores
            connected_scores: Dictionary of connected search bonus scores
            subcategory_scores: Dictionary of subcategory similarity bonus scores
            min_threshold: Minimum score threshold (default from config)
            gender_filter: Optional (user_gender_preference, category) for hard filtering

        Returns:
            Tuple of (List of product_ids sorted by final score, filter_reason or None)
            filter_reason is set if all products were filtered out
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

        # Track filter reasons
        filter_reason = None
        products_before_gender_filter = len(final_scores)

        # Step 1.5: Apply gender hard filter if applicable
        if gender_filter and self.sql_client and final_scores:
            user_gender, category = gender_filter
            # Only apply gender filter for relevant categories
            if category and category.lower() in GENDER_FILTER_CATEGORIES:
                product_ids = list(final_scores.keys())
                filtered_ids = self.sql_client.filter_products_by_gender(
                    product_ids, user_gender
                )
                # Remove filtered-out products
                filtered_out = set(product_ids) - set(filtered_ids)
                for pid in filtered_out:
                    del final_scores[pid]

                if config.DEBUG:
                    print(f"\n=== Gender Filter ({user_gender}) ===")
                    print(f"  Before: {len(product_ids)} products")
                    print(f"  After: {len(filtered_ids)} products")
                    print(f"  Filtered out: {len(filtered_out)} products")

                # If all products were filtered by gender
                if products_before_gender_filter > 0 and len(final_scores) == 0:
                    filter_reason = "gender_filter"

        # Step 2: Filter and rank
        ranked_product_ids = filter_and_rank_products(
            final_scores,
            min_threshold=min_threshold
        )

        # If we had products after gender filter but threshold filtered them all
        if filter_reason is None and products_before_gender_filter > 0 and len(ranked_product_ids) == 0:
            filter_reason = "relevance_threshold"

        return ranked_product_ids, filter_reason

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
        sort_literal: Tuple[str, str],
        relevance_scores: Dict[str, float] = None,
        relevance_weight: float = None
    ) -> List[str]:
        """
        Re-rank products by COMBINED relevance + literal score.
        Relevance has higher priority than price to ensure product type match.

        Args:
            product_ids: List of product IDs to re-rank
            sort_literal: Tuple of (field_name, direction)
                         E.g., ("price", "asc") for cheapest, ("price", "desc") for most expensive
            relevance_scores: Dictionary of product_id -> relevance score
            relevance_weight: Weight for relevance vs literal (default from config or 0.7)

        Returns:
            List of product IDs sorted by combined relevance + literal score
        """
        if not product_ids or not sort_literal or not self.sql_client:
            return product_ids

        # Use config or default weight
        if relevance_weight is None:
            relevance_weight = getattr(config, 'LITERAL_SORT_RELEVANCE_WEIGHT', 0.7)

        field_name, direction = sort_literal

        # Fetch products from SQL to get literal values
        products_dict = self.sql_client.get_products_by_ids(product_ids)

        # Get all literal values for normalization
        literal_values = []
        for product_id in product_ids:
            product = products_dict.get(product_id)
            if product:
                field_value = getattr(product, field_name, None)
                if field_value is not None:
                    literal_values.append((product_id, field_value))

        if not literal_values:
            return product_ids

        # If no relevance scores provided, fall back to pure literal sorting
        if not relevance_scores:
            reverse = (direction == "desc")
            literal_values.sort(key=lambda x: x[1], reverse=reverse)
            sorted_ids = [pid for pid, _ in literal_values]

            if config.DEBUG:
                print(f"\n=== Literal Ranking ({field_name} {direction}) - No relevance scores ===")
                for i, (pid, val) in enumerate(literal_values[:10], 1):
                    print(f"  {i}. {pid}: {field_name}={val}")

            return sorted_ids

        # Normalize literal values for scoring (0-1 range)
        min_val = min(v for _, v in literal_values)
        max_val = max(v for _, v in literal_values)
        val_range = max_val - min_val if max_val != min_val else 1.0

        # Normalize relevance scores
        rel_values = [relevance_scores.get(pid, 0.0) for pid, _ in literal_values]
        max_rel = max(rel_values) if rel_values else 1.0
        max_rel = max_rel if max_rel > 0 else 1.0

        # Calculate combined scores
        combined_scores = []
        for pid, val in literal_values:
            # Normalize literal (0-1)
            if direction == "asc":  # cheapest = higher score
                literal_score = 1 - ((val - min_val) / val_range)
            else:  # most expensive = higher score
                literal_score = (val - min_val) / val_range

            # Normalize relevance (0-1)
            rel_score = relevance_scores.get(pid, 0.0) / max_rel

            # Combined: relevance has higher weight
            combined = (relevance_weight * rel_score) + ((1 - relevance_weight) * literal_score)
            combined_scores.append((pid, combined, val, relevance_scores.get(pid, 0.0)))

        # Sort by combined score (highest first)
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Extract sorted product IDs
        sorted_ids = [pid for pid, _, _, _ in combined_scores]

        if config.DEBUG:
            print(f"\n=== Weighted Literal Ranking ({field_name} {direction}, rel_weight={relevance_weight}) ===")
            for i, (pid, combined, val, rel) in enumerate(combined_scores[:10], 1):
                print(f"  {i}. {pid}: combined={combined:.3f} (rel={rel:.2f}, {field_name}={val})")

        return sorted_ids
