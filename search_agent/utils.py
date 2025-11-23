"""
Utility functions for search orchestrator
"""
from typing import Any, Dict, List
from config.config import Config

config = Config


def calculate_buffer_penalty(
    actual_value: float,
    target_value: float,
    operator: str,
    buffer: float
) -> float:
    """
    Calculate penalty for values within buffer zone

    Args:
        actual_value: Actual product value
        target_value: Target value from literal constraint
        operator: Comparison operator
        buffer: Buffer percentage (e.g., 0.1 = 10%)

    Returns:
        Penalty score (negative value)

    Example:
        Target: price < 20 with 10% buffer (actual limit = 22)
        If price = 21: penalty = -0.05 (50% into buffer)
        If price = 19: penalty = 0.0 (within target)
    """
    # Only applies to numeric comparisons
    if not isinstance(actual_value, (int, float)) or not isinstance(target_value, (int, float)):
        return 0.0

    penalty = 0.0

    if operator == "<":
        # If actual_value > target_value but < target_value * (1 + buffer)
        if actual_value > target_value:
            buffer_limit = target_value * (1 + buffer)
            if actual_value <= buffer_limit:
                # Calculate penalty based on how far into buffer
                buffer_range = buffer_limit - target_value
                overage = actual_value - target_value
                buffer_percentage = overage / buffer_range if buffer_range > 0 else 1.0
                penalty = -config.LINEAR_PENALTY_RATE * buffer_percentage

    elif operator == "<=":
        if actual_value > target_value:
            buffer_limit = target_value * (1 + buffer)
            if actual_value <= buffer_limit:
                buffer_range = buffer_limit - target_value
                overage = actual_value - target_value
                buffer_percentage = overage / buffer_range if buffer_range > 0 else 1.0
                penalty = -config.LINEAR_PENALTY_RATE * buffer_percentage

    elif operator == ">":
        # If actual_value < target_value but > target_value * (1 - buffer)
        if actual_value < target_value:
            buffer_limit = target_value * (1 - buffer)
            if actual_value >= buffer_limit:
                buffer_range = target_value - buffer_limit
                underage = target_value - actual_value
                buffer_percentage = underage / buffer_range if buffer_range > 0 else 1.0
                penalty = -config.LINEAR_PENALTY_RATE * buffer_percentage

    elif operator == ">=":
        if actual_value < target_value:
            buffer_limit = target_value * (1 - buffer)
            if actual_value >= buffer_limit:
                buffer_range = target_value - buffer_limit
                underage = target_value - actual_value
                buffer_percentage = underage / buffer_range if buffer_range > 0 else 1.0
                penalty = -config.LINEAR_PENALTY_RATE * buffer_percentage

    return penalty


def apply_literal_filters_with_penalties(
    product_literal_values: Dict[str, Dict[str, Any]],
    literals: List[tuple]
) -> Dict[str, float]:
    """
    Apply literal filters and calculate penalties for products

    Args:
        product_literal_values: Dictionary from SQL query
            {
                "p-123": {"price": 19.99, "size": "M"},
                "p-456": {"price": 21.00, "size": "L"}
            }
        literals: List of (field_name, operator, value, buffer) tuples

    Returns:
        Dictionary mapping product_id to penalty score
        {
            "p-123": 0.0,     # No penalty (within target)
            "p-456": -0.05    # Small penalty (in buffer zone)
        }
    """
    penalties = {}

    for product_id, literal_values in product_literal_values.items():
        total_penalty = 0.0

        for field_name, operator, target_value, buffer in literals:
            if field_name in literal_values:
                actual_value = literal_values[field_name]
                penalty = calculate_buffer_penalty(
                    actual_value,
                    target_value,
                    operator,
                    buffer
                )
                total_penalty += penalty

        penalties[product_id] = total_penalty

    return penalties


def merge_score_dictionaries(*score_dicts: Dict[str, float]) -> Dict[str, float]:
    """
    Merge multiple score dictionaries by summing scores

    Args:
        *score_dicts: Variable number of score dictionaries

    Returns:
        Merged dictionary with summed scores

    Example:
        dict1 = {"p-123": 2.0, "p-456": 1.5}
        dict2 = {"p-456": 0.5, "p-789": 1.0}
        result = {"p-123": 2.0, "p-456": 2.0, "p-789": 1.0}
    """
    merged = {}

    for score_dict in score_dicts:
        for product_id, score in score_dict.items():
            merged[product_id] = merged.get(product_id, 0.0) + score

    return merged


def filter_and_rank_products(
    product_scores: Dict[str, float],
    min_threshold: float = None
) -> List[str]:
    """
    Filter products by minimum threshold and rank by score

    Args:
        product_scores: Dictionary mapping product_id to score
        min_threshold: Minimum score threshold (default from config)

    Returns:
        List of product_ids sorted by score (highest first)
    """
    threshold = min_threshold if min_threshold is not None else config.MIN_SCORE_THRESHOLD

    # Filter by threshold
    filtered = {
        pid: score
        for pid, score in product_scores.items()
        if score >= threshold
    }

    # Sort by score descending
    ranked = sorted(
        filtered.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return just the product IDs
    return [pid for pid, score in ranked]


def debug_print_scores(
    product_scores: Dict[str, float],
    label: str = "Scores"
):
    """
    Debug utility to print scores

    Args:
        product_scores: Dictionary of scores
        label: Label for the output
    """
    if config.DEBUG:
        print(f"\n=== {label} ===")
        sorted_scores = sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for product_id, score in sorted_scores[:10]:  # Show top 10
            print(f"  {product_id}: {score:.4f}")
        print()
