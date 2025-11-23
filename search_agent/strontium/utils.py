"""
Utility functions for Strontium
"""
from typing import List, Tuple


def extract_product_ids(text: str) -> List[str]:
    """
    Extract product IDs from text (format: p-XXX)

    Args:
        text: Text to search

    Returns:
        List of product IDs found
    """
    import re
    pattern = r'p-[\w-]+'
    matches = re.findall(pattern, text.lower())
    return matches


def normalize_property_name(prop: str) -> str:
    """
    Normalize property name to lowercase with underscores

    Args:
        prop: Property name

    Returns:
        Normalized property name
    """
    return prop.lower().replace(" ", "_").replace("-", "_")


def extract_price_from_text(text: str) -> Tuple[str, float, float]:
    """
    Extract price constraints from text

    Args:
        text: Text containing price info

    Returns:
        Tuple of (operator, value, buffer)
    """
    import re

    # Pattern for "under $30", "less than $30", "< 30"
    under_pattern = r'(?:under|less\s+than|<)\s*\$?(\d+(?:\.\d+)?)'
    match = re.search(under_pattern, text.lower())
    if match:
        price = float(match.group(1))
        return ("<", price, 0.1)

    # Pattern for "over $30", "more than $30", "> 30"
    over_pattern = r'(?:over|more\s+than|>)\s*\$?(\d+(?:\.\d+)?)'
    match = re.search(over_pattern, text.lower())
    if match:
        price = float(match.group(1))
        return (">", price, 0.1)

    # Pattern for "around $30", "about $30"
    around_pattern = r'(?:around|about|~)\s*\$?(\d+(?:\.\d+)?)'
    match = re.search(around_pattern, text.lower())
    if match:
        price = float(match.group(1))
        return ("=", price, 0.2)  # Larger buffer for "around"

    return ("<", 0, 0)  # Default


def is_generic_staple(basetype: str) -> bool:
    """
    Check if product basetype is a generic staple (HQ candidate)

    Args:
        basetype: Product basetype

    Returns:
        True if generic staple
    """
    staples = {
        "tomatoes", "milk", "bread", "eggs", "rice", "pasta",
        "flour", "sugar", "salt", "pepper", "butter", "cheese",
        "onions", "potatoes", "carrots", "apples", "bananas"
    }
    return basetype.lower() in staples
