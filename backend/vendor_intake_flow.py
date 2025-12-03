"""
Vendor intake flow.

Handles:
- Registration gate (vendors must be registered to proceed)
- Session mode locking (add vs update) for a short window
- Missing-field prompting loop (name + price/quantity/stock)
- Similarity checks against existing inventory for additions and updates
- Intake queue logging for downstream SQL/KG/VDB ingestion
- OCR processing for images/documents

The flow is designed to be stateless for the transport layer; all per-vendor
state lives in memory inside the SessionManager.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from config.config import Config
from search_agent.database.sql_client import SQLClient

# -------------------- Demo Mode Config -------------------
DEMO_CONFIG_PATH = Path(__file__).parent.parent / "config" / "demo_config.json"

def load_demo_config():
    """Load demo configuration from JSON file"""
    try:
        with open(DEMO_CONFIG_PATH) as f:
            return json.load(f)
    except Exception as e:
        return {"demo_mode": False}

DEMO_CONFIG = load_demo_config()
# ---------------------------------------------------------

# Optional NVIDIA LLM import for intelligent parsing
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_LLM_AVAILABLE = True
except ImportError:
    NVIDIA_LLM_AVAILABLE = False
    ChatNVIDIA = None


# ==================== Yes/No LLM-Based Intent Detection ====================

# Cache for LLM intent detection client
_intent_llm_client = None

def _get_intent_llm_client():
    """Get or create the LLM client for intent detection."""
    global _intent_llm_client
    if _intent_llm_client is None and NVIDIA_LLM_AVAILABLE:
        try:
            _intent_llm_client = ChatNVIDIA(
                model=Config.NVIDIA_MODEL,
                api_key=Config.NVIDIA_API_KEY,
                temperature=0.0,  # Deterministic for classification
                max_tokens=10,    # We only need YES/NO/UNKNOWN
            )
        except Exception as e:
            print(f"Failed to initialize intent LLM: {e}")
            _intent_llm_client = None
    return _intent_llm_client


# Quick pattern check for obvious cases (to avoid LLM call)
_QUICK_AFFIRMATIVE = {
    "yes", "y", "yep", "yeah", "yea", "ya", "yup", "ok", "okay", "k", "kk",
    "sure", "confirm", "correct", "fine", "good", "great", "done", "proceed",
    "haan", "ha", "theek", "sahi", "ji", "bilkul", "agreed", "accept"
}

_QUICK_NEGATIVE = {
    "no", "n", "nope", "nah", "na", "cancel", "abort", "stop", "nahi", "mat", "reject"
}


def _is_affirmative(text: str) -> bool:
    """
    Check if the text is an affirmative response using LLM inference.
    Falls back to pattern matching if LLM is unavailable.
    """
    normalized = text.lower().strip()
    no_spaces = normalized.replace(" ", "")

    # Quick check for obvious affirmatives (avoid LLM call)
    if normalized in _QUICK_AFFIRMATIVE or no_spaces in _QUICK_AFFIRMATIVE:
        return True

    # Quick check for obvious negatives
    if normalized in _QUICK_NEGATIVE or no_spaces in _QUICK_NEGATIVE:
        return False

    # For short inputs that look like typos of "yes" or "ok"
    if len(no_spaces) <= 4:
        # Typo patterns for "yes"
        if set(no_spaces).issubset({'y', 'e', 's', 'a'}) and len(no_spaces) >= 2:
            return True
        # Typo patterns for "ok"
        if set(no_spaces).issubset({'o', 'k', 'a', 'y'}) and 'o' in no_spaces:
            return True

    # Use LLM for ambiguous/complex responses
    llm = _get_intent_llm_client()
    if llm is None:
        # Fallback: assume not affirmative for safety
        return False

    try:
        prompt = f"""Classify if this message means YES/CONFIRM or NO/REJECT or is UNCLEAR.
Message: "{text}"

Rules:
- If the person is agreeing, confirming, saying yes in any language/style -> output: YES
- If the person is disagreeing, rejecting, saying no in any language/style -> output: NO
- If unclear or unrelated -> output: UNKNOWN

Output only one word: YES, NO, or UNKNOWN"""

        response = llm.invoke(prompt)
        result = response.content.strip().upper()

        return result == "YES"
    except Exception as e:
        print(f"LLM intent detection error: {e}")
        return False


def _is_negative(text: str) -> bool:
    """
    Check if the text is a negative response using LLM inference.
    Falls back to pattern matching if LLM is unavailable.
    """
    normalized = text.lower().strip()
    no_spaces = normalized.replace(" ", "")

    # Quick check for obvious negatives
    if normalized in _QUICK_NEGATIVE or no_spaces in _QUICK_NEGATIVE:
        return True

    # Quick check for obvious affirmatives
    if normalized in _QUICK_AFFIRMATIVE or no_spaces in _QUICK_AFFIRMATIVE:
        return False

    # Use LLM for ambiguous/complex responses
    llm = _get_intent_llm_client()
    if llm is None:
        # Fallback: assume not negative for safety
        return False

    try:
        prompt = f"""Classify if this message means YES/CONFIRM or NO/REJECT or is UNCLEAR.
Message: "{text}"

Rules:
- If the person is agreeing, confirming, saying yes in any language/style -> output: YES
- If the person is disagreeing, rejecting, saying no in any language/style -> output: NO
- If unclear or unrelated -> output: UNKNOWN

Output only one word: YES, NO, or UNKNOWN"""

        response = llm.invoke(prompt)
        result = response.content.strip().upper()

        return result == "NO"
    except Exception as e:
        print(f"LLM intent detection error: {e}")
        return False


# ==================== LLM-Based Product Parser ====================

VENDOR_PRODUCT_PARSER_PROMPT = """You are a product information extraction tool for a vendor inventory system.
Your task: Extract product data from natural language and fit it into the JSON structure below.

====================================================================
IMPORTANT: INPUT MAY CONTAIN MULTIPLE PRODUCTS OR VARIATIONS
====================================================================
- The vendor may describe MULTIPLE PRODUCTS in one message
- Or MULTIPLE VARIATIONS of the same product (different colors, sizes, prices)
- Each variation with different price/color/size = SEPARATE PRODUCT entry
- ALWAYS return an array: {"products": [...]}
- Copy shared properties (brand, material, etc.) to EACH product entry

=== TARGET JSON STRUCTURE ===
{
  "products": [
    {
      "prod_name": "Product Name - Variant",
      "price": 1000,
      "quantity": 1,
      "quantityunit": "unit",
      "stock": 50,
      "description": "GENERATE this from the input message",
      "size": "M or 12x15x30 cm",
      "dimensions": {"length": 12, "width": 15, "height": 30, "unit": "cm"},
      "brand": "Brand Name",
      "colour": "Blue",
      "category": "clothing/electronics/grocery/bags/furniture/other",
      "subcategory": "specific type",
      "rating": 4.5,
      "other_properties": {"material": "cotton", "warranty_years": 2, "origin": "India"}
    }
  ]
}

====================================================================
=== ESSENTIAL FIELDS - MUST EXTRACT FOR EACH PRODUCT ===
====================================================================
These 5 fields are CRITICAL. If ANY is NOT found, set to null (we will ask vendor):

1. prod_name    - Product name (include variant info like color/size in name)
2. price        - Numeric price (REQUIRED - set null if missing)
3. quantity     - Units per price, default 1 (REQUIRED - set null if missing)
4. quantityunit - Unit type: unit, piece, kg, etc. (REQUIRED - set null if missing)
5. stock        - Stock count FOR THIS VARIANT (REQUIRED - set null if missing)

====================================================================
=== DESCRIPTION FIELD - ALWAYS GENERATE ===
====================================================================
ALWAYS create a "description" by summarizing the product from the vendor's message.
Include: key features, materials, special attributes mentioned.

====================================================================
=== OPTIONAL FIELDS - DO NOT FILL IF NOT MENTIONED ===
====================================================================
DO NOT guess. DO NOT make up data. DO NOT include field if not mentioned:
- brand, colour, size, dimensions, category, subcategory, rating, other_properties

====================================================================

=== FEW-SHOT EXAMPLES ===

--- EXAMPLE 1: Multiple Color Variations ---
Input: "I have a floral shirt made by Levi's, 50 red coloured shirts and 50 blue coloured ones in stock. The blue one costs 5000 rupees per piece and the red ones cost 6000 rupees. Made of cotton polymer blend with sustainably sourced materials"
{
  "products": [
    {
      "prod_name": "Levi's Floral Shirt - Red",
      "price": 6000,
      "quantity": 1,
      "quantityunit": "piece",
      "stock": 50,
      "description": "Floral pattern shirt by Levi's in red. Made of cotton polymer blend using sustainably sourced materials.",
      "brand": "Levi's",
      "colour": "red",
      "category": "clothing",
      "subcategory": "shirt",
      "other_properties": {"material": "cotton polymer blend", "pattern": "floral", "sustainable": true}
    },
    {
      "prod_name": "Levi's Floral Shirt - Blue",
      "price": 5000,
      "quantity": 1,
      "quantityunit": "piece",
      "stock": 50,
      "description": "Floral pattern shirt by Levi's in blue. Made of cotton polymer blend using sustainably sourced materials.",
      "brand": "Levi's",
      "colour": "blue",
      "category": "clothing",
      "subcategory": "shirt",
      "other_properties": {"material": "cotton polymer blend", "pattern": "floral", "sustainable": true}
    }
  ]
}

--- EXAMPLE 2: Multiple Size Variations ---
Input: "Nike Air Max shoes, black color, size 9 costs 8000 rs (30 pairs), size 10 costs 8500 rs (25 pairs), size 11 costs 9000 rs (20 pairs). All have air cushion technology"
{
  "products": [
    {
      "prod_name": "Nike Air Max - Black Size 9",
      "price": 8000,
      "quantity": 1,
      "quantityunit": "pair",
      "stock": 30,
      "description": "Nike Air Max shoes in black, size 9. Features air cushion technology for comfort.",
      "brand": "Nike",
      "colour": "black",
      "size": "9",
      "category": "clothing",
      "subcategory": "shoes",
      "other_properties": {"technology": "air cushion"}
    },
    {
      "prod_name": "Nike Air Max - Black Size 10",
      "price": 8500,
      "quantity": 1,
      "quantityunit": "pair",
      "stock": 25,
      "description": "Nike Air Max shoes in black, size 10. Features air cushion technology for comfort.",
      "brand": "Nike",
      "colour": "black",
      "size": "10",
      "category": "clothing",
      "subcategory": "shoes",
      "other_properties": {"technology": "air cushion"}
    },
    {
      "prod_name": "Nike Air Max - Black Size 11",
      "price": 9000,
      "quantity": 1,
      "quantityunit": "pair",
      "stock": 20,
      "description": "Nike Air Max shoes in black, size 11. Features air cushion technology for comfort.",
      "brand": "Nike",
      "colour": "black",
      "size": "11",
      "category": "clothing",
      "subcategory": "shoes",
      "other_properties": {"technology": "air cushion"}
    }
  ]
}

--- EXAMPLE 3: Single Product ---
Input: "skybag premium leather bag, 1000 rupees per bag, 50 in stock, blue colour, 12cm*15cm*30cm"
{
  "products": [
    {
      "prod_name": "Skybag Premium Leather Bag",
      "price": 1000,
      "quantity": 1,
      "quantityunit": "bag",
      "stock": 50,
      "description": "Premium leather bag from Skybag in blue. Dimensions: 12cm x 15cm x 30cm.",
      "brand": "Skybag",
      "colour": "blue",
      "size": "12cm*15cm*30cm",
      "dimensions": {"length": 12, "width": 15, "height": 30, "unit": "cm"},
      "category": "bags",
      "subcategory": "leather bag"
    }
  ]
}

--- EXAMPLE 4: Simple Product (minimal info) ---
Input: "add maggi 10 rupees 1 pack 50 stock"
{
  "products": [
    {
      "prod_name": "Maggi",
      "price": 10,
      "quantity": 1,
      "quantityunit": "pack",
      "stock": 50,
      "description": "Maggi instant noodles."
    }
  ]
}

--- EXAMPLE 5: Missing Essential Fields ---
Input: "rice 50 rupees per kg"
{
  "products": [
    {
      "prod_name": "Rice",
      "price": 50,
      "quantity": 1,
      "quantityunit": "kg",
      "stock": null,
      "description": "Rice sold per kilogram."
    }
  ]
}
NOTE: stock is null because not mentioned - we will ask vendor.

--- EXAMPLE 6: Multiple Different Products ---
Input: "I have Dove soap 50 rs per piece, 100 in stock. Also Lux soap 40 rs, 80 in stock"
{
  "products": [
    {
      "prod_name": "Dove Soap",
      "price": 50,
      "quantity": 1,
      "quantityunit": "piece",
      "stock": 100,
      "description": "Dove soap bar.",
      "brand": "Dove",
      "category": "grocery",
      "subcategory": "soap"
    },
    {
      "prod_name": "Lux Soap",
      "price": 40,
      "quantity": 1,
      "quantityunit": "piece",
      "stock": 80,
      "description": "Lux soap bar.",
      "brand": "Lux",
      "category": "grocery",
      "subcategory": "soap"
    }
  ]
}

====================================================================
Now extract product information from this input:
"{input}"

RULES:
1. Output ONLY valid JSON - no explanations or text outside JSON
2. ALWAYS return {"products": [...]} array format (even for single product)
3. ALWAYS generate "description" from the input message
4. Essential fields (prod_name, price, quantity, quantityunit, stock): set to null if NOT found
5. Optional fields: DO NOT include if not explicitly mentioned
6. Multiple variations (color/size/price) = SEPARATE product entries

====================================================================
NEVER USE EMOJIS IN YOUR OUTPUT. DO NOT INCLUDE ANY EMOJIS.
====================================================================
Output JSON:"""


class VendorProductLLMParser:
    """LLM-based parser for extracting product information from natural language."""

    _instance = None
    _llm_client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if VendorProductLLMParser._llm_client is None and NVIDIA_LLM_AVAILABLE:
            api_key = os.getenv("NVIDIA_API_KEY") or Config.NVIDIA_API_KEY
            if api_key:
                try:
                    VendorProductLLMParser._llm_client = ChatNVIDIA(
                        model=Config.NVIDIA_MODEL,
                        api_key=api_key,
                        temperature=0.1,  # Low temperature for consistent extraction
                        max_tokens=2048,
                    )
                    print("VendorProductLLMParser initialized with NVIDIA LLM")
                except Exception as e:
                    print(f"Failed to initialize NVIDIA LLM: {e}")
                    VendorProductLLMParser._llm_client = None

    def parse(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parse natural language product description using LLM.

        Returns list of product dicts (supports multiple products/variations) or None if parsing fails.
        """
        if VendorProductLLMParser._llm_client is None:
            return None

        try:
            prompt = VENDOR_PRODUCT_PARSER_PROMPT.replace("{input}", text)
            response = VendorProductLLMParser._llm_client.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Clean response
            response_text = response_text.strip()

            # Remove thinking tags if present
            if "</think>" in response_text:
                response_text = response_text.split("</think>", 1)[1].strip()

            # Remove markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Find JSON object
            if not response_text.startswith("{"):
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx+1]

            parsed = json.loads(response_text)

            # Handle array format: {"products": [...]}
            # Fallback to single product wrapped in list for backward compatibility
            if "products" in parsed and isinstance(parsed["products"], list):
                products_list = parsed["products"]
            else:
                # Single product format (backward compatibility)
                products_list = [parsed]

            # Normalize each product
            normalized_products = []
            for product in products_list:
                normalized = self._normalize_product(product)
                if normalized.get("name"):  # Only add if we got a valid name
                    normalized_products.append(normalized)
                    print(f"LLM parsed: {normalized.get('name')} - Rs.{normalized.get('price')}")

            if normalized_products:
                print(f"Total products parsed: {len(normalized_products)}")
                return normalized_products
            return None

        except json.JSONDecodeError as e:
            print(f"LLM JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"LLM parsing error: {e}")
            return None

    def _normalize_product(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single product's field names to match our schema."""
        normalized = {}

        # Required fields
        normalized["name"] = parsed.get("prod_name") or parsed.get("name")
        normalized["price"] = parsed.get("price")
        normalized["quantity"] = parsed.get("quantity", 1)
        normalized["quantityunit"] = parsed.get("quantityunit") or parsed.get("unit", "unit")
        normalized["stock"] = parsed.get("stock")

        # Optional fields - only include if present
        if parsed.get("brand"):
            normalized["brand"] = parsed["brand"]
        if parsed.get("colour") or parsed.get("color"):
            normalized["colour"] = parsed.get("colour") or parsed.get("color")
        if parsed.get("size"):
            normalized["size"] = parsed["size"]
        if parsed.get("dimensions"):
            normalized["dimensions"] = parsed["dimensions"]
        if parsed.get("category"):
            normalized["category"] = parsed["category"]
        if parsed.get("subcategory"):
            normalized["subcategory"] = parsed["subcategory"]
        if parsed.get("description"):
            normalized["description"] = parsed["description"]
        if parsed.get("rating"):
            normalized["rating"] = parsed["rating"]
        if parsed.get("other_properties"):
            normalized["other_properties"] = parsed["other_properties"]

        # Mark as LLM-parsed for confirmation flow
        normalized["parsed_by_llm"] = True

        return normalized


# Singleton instance
_vendor_llm_parser: Optional[VendorProductLLMParser] = None

def get_vendor_llm_parser() -> Optional[VendorProductLLMParser]:
    """Get or create the vendor LLM parser singleton."""
    global _vendor_llm_parser
    if _vendor_llm_parser is None and NVIDIA_LLM_AVAILABLE:
        _vendor_llm_parser = VendorProductLLMParser()
    return _vendor_llm_parser


def _should_use_llm_parser(text: str) -> bool:
    """
    Determine if the input is complex enough to warrant LLM parsing.

    Use LLM for:
    - Long natural language descriptions
    - Sentences with multiple clauses
    - Text mentioning optional fields like color, brand, dimensions
    """
    # Use LLM for longer, more conversational inputs
    word_count = len(text.split())

    # Keywords that suggest complex/optional field extraction needed
    complex_keywords = {
        "colour", "color", "brand", "size", "dimension", "warranty",
        "material", "description", "feature", "cm", "inch", "mm",
        "organic", "premium", "quality", "rating", "star"
    }

    has_complex_keywords = any(kw in text.lower() for kw in complex_keywords)

    # Use LLM if: long text OR has complex keywords OR starts with conversational phrases
    conversational_starts = ["i want", "i have", "i'd like", "please add", "can you add", "this is"]
    is_conversational = any(text.lower().startswith(phrase) for phrase in conversational_starts)

    return (word_count > 15) or has_complex_keywords or is_conversational


# ==================== Pydantic Models for Validation ====================

class ProductItem(BaseModel):
    """Pydantic model for validating a single product item."""
    # REQUIRED fields
    name: Optional[str] = Field(None, description="Product name")
    price: Optional[float] = Field(None, ge=0, description="Price per unit")
    quantity: Optional[int] = Field(None, ge=1, description="Units per price (e.g., ₹10 per 1 unit)")
    quantityunit: Optional[str] = Field(None, description="Unit type (kg, unit, pack, pcs, etc.)")
    stock: Optional[int] = Field(None, ge=0, description="Total stock available")

    # OPTIONAL fields
    category: Optional[str] = Field(None, description="Product category")
    subcategory: Optional[str] = Field(None, description="Product subcategory")
    brand: Optional[str] = Field(None, description="Brand name")
    colour: Optional[str] = Field(None, description="Product colour")
    description: Optional[str] = Field(None, description="Product description")
    size: Optional[str] = Field(None, description="Product size")
    dimensions: Optional[Dict[str, Any]] = Field(None, description="Dimensions dict (length, width, height, weight)")
    imageid: Optional[str] = Field(None, description="Image filename/ID")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Product rating (0-5)")
    other_properties: Optional[Dict[str, Any]] = Field(None, description="Additional flexible properties")

    # Track which fields are missing for prompting
    _missing_fields: List[str] = []

    @field_validator('price', mode='before')
    @classmethod
    def parse_price(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            # Extract number from strings like "₹10", "Rs. 10", "10.50"
            match = re.search(r'(\d+(?:\.\d+)?)', v.replace(',', ''))
            if match:
                return float(match.group(1))
        return None

    @field_validator('quantity', 'stock', mode='before')
    @classmethod
    def parse_int(cls, v):
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, (float, str)):
            try:
                return int(float(str(v).replace(',', '')))
            except (ValueError, TypeError):
                return None
        return None

    @field_validator('name', mode='before')
    @classmethod
    def clean_name(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Clean up name - remove common prefixes
            cleaned = re.sub(r'^(add|update|new)\s+', '', v.strip(), flags=re.IGNORECASE)
            return cleaned.strip() if cleaned.strip() else None
        return None

    def get_missing_required_fields(self) -> List[str]:
        """Return list of missing required fields.
        Required: name, price, quantity, quantityunit, stock
        """
        missing = []
        if not self.name:
            missing.append("name")
        if self.price is None:
            missing.append("price")
        if self.quantity is None:
            missing.append("quantity")
        if not self.quantityunit:
            missing.append("quantityunit")
        if self.stock is None:
            missing.append("stock")
        return missing

    def is_complete(self) -> bool:
        """Check if all required fields are present."""
        return len(self.get_missing_required_fields()) == 0

    def to_payload(self) -> Dict[str, Any]:
        """Convert to payload dict for intake processing."""
        payload = {
            "name": self.name,
            "price": self.price,
            "quantity": self.quantity or 1,
            "quantityunit": self.quantityunit or "unit",
            "stock": self.stock or 0,
            "category": self.category,
            "subcategory": self.subcategory,
            "brand": self.brand,
            "colour": self.colour,
            "description": self.description,
            "size": self.size,
            "dimensions": self.dimensions,
            "imageid": self.imageid,
            "rating": self.rating,
            "other_properties": self.other_properties,
        }
        # Remove None values for optional fields
        return {k: v for k, v in payload.items() if v is not None}


class BulkProductInput(BaseModel):
    """Container for multiple product items."""
    items: List[ProductItem] = Field(default_factory=list)
    raw_text: Optional[str] = None

    @classmethod
    def parse_bulk_text(cls, text: str) -> "BulkProductInput":
        """Parse bulk text input into multiple ProductItems."""
        items = []
        raw_text = text.strip()

        # Try different parsing strategies

        # Strategy 1: JSON array
        if raw_text.startswith('['):
            try:
                data = json.loads(raw_text)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            items.append(ProductItem(**item))
                    return cls(items=items, raw_text=raw_text)
            except json.JSONDecodeError:
                pass

        # Strategy 2: JSON object (single item)
        if raw_text.startswith('{'):
            try:
                data = json.loads(raw_text)
                if isinstance(data, dict):
                    items.append(ProductItem(**data))
                    return cls(items=items, raw_text=raw_text)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Line-by-line parsing
        # Each line could be: "name price qty stock" or "name, price, qty, stock"
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

        # Skip header line if it looks like column names
        if lines and re.match(r'^(name|product|item)\s*[,|\t]', lines[0], re.IGNORECASE):
            lines = lines[1:]

        for line in lines:
            # Skip mode keywords alone
            if line.lower() in {'add', 'update', 'bulk', 'bulk add', 'bulk update'}:
                continue

            item = cls._parse_single_line(line)
            if item and (item.name or item.price is not None):
                items.append(item)

        return cls(items=items, raw_text=raw_text)

    @classmethod
    def _parse_single_line(cls, line: str) -> Optional[ProductItem]:
        """Parse a single line into a ProductItem using smart natural language parsing.

        Handles various formats:
        - "noodles 10 rupees 1 pcs 100 stock"
        - "maggi 10 1 pack 50"
        - "rice, 500, 5, kg, 100"
        """
        if not line.strip():
            return None

        # Use the smart natural language parser
        parsed = _parse_natural_language_input(line, mode=None)

        return ProductItem(
            name=parsed.get("name"),
            price=parsed.get("price"),
            quantity=parsed.get("quantity"),
            quantityunit=parsed.get("quantityunit"),
            stock=parsed.get("stock"),
        )

# OCR imports (lazy loaded to avoid startup cost)
_ocr_initialized = False

config = Config


VITAL_FIELDS = {"name", "price", "quantity", "quantityunit", "stock"}


def _format_preview(payload: Dict[str, Any], missing_fields: List[str] = None) -> str:
    """Format a product preview message for confirmation.

    Args:
        payload: Product data dict with name, price, quantity, quantityunit, stock, etc.
        missing_fields: List of fields still missing (if any)

    Returns:
        Formatted preview string
    """
    lines = ["*Product Preview:*", "━━━━━━━━━━━━━━━━━━━━━━"]

    # Required fields
    name = payload.get("name", "—")
    price = payload.get("price")
    qty = payload.get("quantity", 1)
    unit = payload.get("quantityunit", "unit")
    stock = payload.get("stock", 0)

    lines.append(f"*Name:* {name}")
    if price is not None:
        lines.append(f"*Price:* ₹{price} per {qty} {unit}")
    else:
        lines.append(f"*Price:* —")
    lines.append(f"*Stock:* {stock}")

    # Optional fields (only show if provided)
    if payload.get("brand"):
        lines.append(f"*Brand:* {payload['brand']}")
    if payload.get("category"):
        lines.append(f"*Category:* {payload['category']}")
    if payload.get("subcategory"):
        lines.append(f"*Subcategory:* {payload['subcategory']}")
    if payload.get("colour"):
        lines.append(f"*Colour:* {payload['colour']}")
    if payload.get("size"):
        lines.append(f"*Size:* {payload['size']}")
    if payload.get("dimensions"):
        dims = payload['dimensions']
        if isinstance(dims, dict):
            dim_str = f"{dims.get('length', '?')}x{dims.get('width', '?')}x{dims.get('height', '?')} {dims.get('unit', '')}"
            lines.append(f"*Dimensions:* {dim_str}")
        else:
            lines.append(f"*Dimensions:* {dims}")
    if payload.get("rating"):
        lines.append(f"*Rating:* {payload['rating']}/5")
    if payload.get("description"):
        desc = payload['description'][:100] + "..." if len(payload['description']) > 100 else payload['description']
        lines.append(f"*Description:* {desc}")
    if payload.get("other_properties"):
        props = payload['other_properties']
        if isinstance(props, dict):
            prop_items = [f"{k}: {v}" for k, v in props.items()]
            lines.append(f"*Other:* {', '.join(prop_items[:3])}")  # Show max 3 properties

    # Show if parsed by LLM
    if payload.get("parsed_by_llm"):
        lines.append("")
        lines.append("_Extracted using AI_")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")

    # Show missing fields if any
    if missing_fields:
        lines.append("")
        lines.append(f"*Missing required fields:* {', '.join(missing_fields)}")
        lines.append("Please provide the missing information.")
    else:
        lines.append("")
        lines.append("Is this correct? Reply 'yes' to confirm or provide corrections.")

    return "\n".join(lines)


def _now() -> float:
    return time.time()


def _clean_text(text: str) -> str:
    return (text or "").strip()


def _extract_name(text: str, mode: Optional[str]) -> Optional[str]:
    if not text:
        return None

    cleaned = text.strip()
    if mode in {"add", "update"}:
        cleaned = re.sub(r"^(add|update)\b[:\- ]*", "", cleaned, flags=re.IGNORECASE).strip()

    # Strip trailing fields like price/quantity/stock hints to keep only the product name
    keyword_match = re.search(r"\b(price|cost|mrp|qty|quantity|stock)\b", cleaned, re.IGNORECASE)
    if keyword_match:
        cleaned = cleaned[: keyword_match.start()].strip()

    return cleaned or None


# Unit keywords for flexible parsing
UNIT_KEYWORDS = {
    "kg", "kgs", "kilogram", "kilograms",
    "g", "gm", "gms", "gram", "grams",
    "ml", "millilitre", "milliliter",
    "l", "ltr", "litre", "liter", "litres", "liters",
    "unit", "units",
    "pcs", "pc", "piece", "pieces",
    "pack", "packs", "packet", "packets",
    "box", "boxes",
    "dozen", "doz",
    "pair", "pairs",
    "set", "sets",
    "bottle", "bottles",
    "can", "cans",
    "bag", "bags",
}

# Price keywords for natural language parsing
PRICE_KEYWORDS = {"rupees", "rupee", "rs", "rs.", "inr", "₹", "price", "cost", "mrp"}

# Stock keywords for natural language parsing (DO NOT include qty/quantity - those are for quantity field)
STOCK_KEYWORDS = {"stock", "stocks", "inventory", "available", "total", "instock", "in-stock", "have", "got"}

# Quantity keywords (for "quantity 1" or "qty 5" patterns)
QTY_KEYWORDS = {"qty", "quantity", "per", "each", "x"}

# Words to exclude from product name (stop words for name extraction)
NAME_STOP_WORDS = {"i", "we", "have", "got", "with", "and", "the", "a", "an"}


def _parse_natural_language_input(text: str, mode: Optional[str]) -> Dict[str, Any]:
    """
    Smart natural language parser for product input.

    Handles various formats:
    - "Add noodles 10 rupees 1 pcs 100 stock"
    - "noodles price 10 quantity 1 piece stock 100"
    - "add maggi 10 1 pack 50"
    - "maggi ₹10 per piece, stock: 100"

    Returns dict with parsed values.
    """
    if not text:
        return {"name": None, "price": None, "quantity": None, "quantityunit": None, "stock": None, "parsed_pattern": None}

    cleaned = text.strip()
    if mode in {"add", "update"}:
        cleaned = re.sub(r"^(add|update)\b[\s:\-]*", "", cleaned, flags=re.IGNORECASE).strip()

    result = {
        "name": None,
        "price": None,
        "quantity": None,
        "quantityunit": None,
        "stock": None,
        "parsed_pattern": None,
    }

    # Normalize text: remove extra spaces, handle punctuation
    normalized = re.sub(r'[,;:]', ' ', cleaned)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Tokenize
    tokens = normalized.split()

    # Track which tokens are consumed
    consumed = set()

    # --- Extract price (look for price keywords or currency symbols) ---
    for i, token in enumerate(tokens):
        token_lower = token.lower().rstrip('.')

        # Check for "rs 10" or "price 10" pattern (keyword BEFORE number)
        price_match = re.match(r'^[₹$]?(\d+(?:\.\d+)?)$', token.replace(',', ''))
        if price_match and i > 0 and i-1 not in consumed and tokens[i-1].lower().rstrip('.') in PRICE_KEYWORDS:
            result["price"] = float(price_match.group(1))
            consumed.add(i)
            consumed.add(i-1)
            continue

        # Check for "10 rupees" or "10 rs" pattern (number BEFORE keyword)
        if token_lower in PRICE_KEYWORDS and i > 0 and i-1 not in consumed:
            prev_token = tokens[i-1].replace(',', '')
            if re.match(r'^\d+(?:\.\d+)?$', prev_token):
                result["price"] = float(prev_token)
                consumed.add(i)
                consumed.add(i-1)
                continue

        # Check for "price 10" or "price: 10" or "₹10" pattern
        if token_lower in PRICE_KEYWORDS and i + 1 < len(tokens):
            next_token = tokens[i+1].replace(',', '').lstrip('₹$')
            if re.match(r'^\d+(?:\.\d+)?$', next_token):
                result["price"] = float(next_token)
                consumed.add(i)
                consumed.add(i+1)
                continue

        # Check for standalone ₹10 or $10
        if token.startswith('₹') or token.startswith('$'):
            num_part = token[1:].replace(',', '')
            if re.match(r'^\d+(?:\.\d+)?$', num_part):
                result["price"] = float(num_part)
                consumed.add(i)
                continue

    # --- Extract stock (look for stock keywords) ---
    for i, token in enumerate(tokens):
        if i in consumed:
            continue
        token_lower = token.lower()

        # Check for "100 stock" pattern
        if token_lower in STOCK_KEYWORDS and i > 0 and i-1 not in consumed:
            prev_token = tokens[i-1].replace(',', '')
            if re.match(r'^\d+$', prev_token):
                result["stock"] = int(prev_token)
                consumed.add(i)
                consumed.add(i-1)
                continue

        # Check for "stock 100" or "stock: 100" pattern
        if token_lower in STOCK_KEYWORDS and i + 1 < len(tokens) and i+1 not in consumed:
            next_token = tokens[i+1].replace(',', '')
            if re.match(r'^\d+$', next_token):
                result["stock"] = int(next_token)
                consumed.add(i)
                consumed.add(i+1)
                continue

        # Check for "100 in stock" pattern (number + "in" + stock keyword)
        if token_lower == "in" and i > 0 and i + 1 < len(tokens):
            prev_token = tokens[i-1].replace(',', '')
            next_token_lower = tokens[i+1].lower()
            if re.match(r'^\d+$', prev_token) and next_token_lower in STOCK_KEYWORDS:
                if i-1 not in consumed and i+1 not in consumed:
                    result["stock"] = int(prev_token)
                    consumed.add(i-1)
                    consumed.add(i)
                    consumed.add(i+1)
                    continue

    # --- Extract explicit quantity keywords (e.g., "quantity 5", "qty 3") ---
    for i, token in enumerate(tokens):
        if i in consumed:
            continue
        token_lower = token.lower()

        # Check for "quantity 5" or "qty 3" pattern (keyword BEFORE number)
        if token_lower in QTY_KEYWORDS and token_lower not in {"per", "each", "x"} and i + 1 < len(tokens) and i+1 not in consumed:
            next_token = tokens[i+1].replace(',', '')
            if re.match(r'^\d+$', next_token):
                result["quantity"] = int(next_token)
                consumed.add(i)
                consumed.add(i+1)
                continue

        # Check for "5 qty" or "3 quantity" pattern (number BEFORE keyword)
        if token_lower in QTY_KEYWORDS and token_lower not in {"per", "each", "x"} and i > 0 and i-1 not in consumed:
            prev_token = tokens[i-1].replace(',', '')
            if re.match(r'^\d+$', prev_token):
                result["quantity"] = int(prev_token)
                consumed.add(i)
                consumed.add(i-1)
                continue

    # --- Extract quantity and unit ---
    for i, token in enumerate(tokens):
        if i in consumed:
            continue
        token_lower = token.lower()

        # Check for unit keywords with preceding number: "1 pcs", "5 kg"
        if token_lower in UNIT_KEYWORDS:
            result["quantityunit"] = token_lower
            consumed.add(i)
            # Check if previous token is a number (quantity)
            if i > 0 and i-1 not in consumed:
                prev_token = tokens[i-1].replace(',', '')
                if re.match(r'^\d+$', prev_token):
                    result["quantity"] = int(prev_token)
                    consumed.add(i-1)
            continue

        # Check for "per piece", "per kg" pattern
        if token_lower == "per" and i + 1 < len(tokens):
            next_lower = tokens[i+1].lower()
            if next_lower in UNIT_KEYWORDS:
                result["quantityunit"] = next_lower
                consumed.add(i)
                consumed.add(i+1)
                # Default quantity to 1 for "per unit" style
                if result["quantity"] is None:
                    result["quantity"] = 1
                continue

    # --- Extract name (unconsumed non-numeric tokens at the start) ---
    name_parts = []
    found_number = False
    all_keywords = PRICE_KEYWORDS | STOCK_KEYWORDS | QTY_KEYWORDS | UNIT_KEYWORDS | {"per"}
    for i, token in enumerate(tokens):
        if i in consumed:
            continue
        token_lower = token.lower()

        # Stop collecting name when we hit a stop word like "i", "we", "have"
        if token_lower in NAME_STOP_WORDS:
            break

        # Stop collecting name when we hit a number (unless it's part of product name like "7up")
        if re.match(r'^\d+(?:\.\d+)?$', token.replace(',', '')):
            # Only include number in name if next token is:
            # 1. A clearly alphanumeric suffix (not a keyword)
            # 2. Not a known keyword
            if i + 1 < len(tokens):
                next_token_lower = tokens[i+1].lower()
                # If next token is a keyword, this number is likely a value (price/qty/stock)
                if next_token_lower in all_keywords:
                    found_number = True
                    break
                # If next token starts with letter and is not a keyword, could be product name like "7up"
                if re.match(r'^[a-zA-Z]', tokens[i+1]) and next_token_lower not in all_keywords:
                    name_parts.append(token)
                    consumed.add(i)
                    continue
            found_number = True
            break
        # Skip price/stock/qty keywords
        if token_lower.rstrip('.') in PRICE_KEYWORDS | STOCK_KEYWORDS | QTY_KEYWORDS:
            continue
        name_parts.append(token)
        consumed.add(i)

    if name_parts:
        result["name"] = ' '.join(name_parts)

    # --- Fallback: assign remaining numbers in order (price, qty, stock) ---
    remaining_numbers = []
    for i, token in enumerate(tokens):
        if i in consumed:
            continue
        clean_token = token.replace(',', '')
        if re.match(r'^\d+(?:\.\d+)?$', clean_token):
            remaining_numbers.append((i, clean_token))

    # Assign remaining numbers based on what's still missing
    for idx, (i, num) in enumerate(remaining_numbers):
        if result["price"] is None and idx == 0:
            result["price"] = float(num)
        elif result["quantity"] is None and (idx == 1 or (idx == 0 and result["price"] is not None)):
            result["quantity"] = int(float(num))
        elif result["stock"] is None:
            result["stock"] = int(float(num))

    # Determine parsed pattern
    parts = []
    if result["name"]:
        parts.append("name")
    if result["price"] is not None:
        parts.append("price")
    if result["quantity"] is not None:
        parts.append("qty")
    if result["quantityunit"]:
        parts.append("unit")
    if result["stock"] is not None:
        parts.append("stock")

    if parts:
        result["parsed_pattern"] = "_".join(parts)

    return result


def _parse_space_separated_input(text: str, mode: Optional[str]) -> Dict[str, Any]:
    """
    Parse space/newline separated input - now uses smart natural language parsing.

    Handles various formats:
    - "Add maggi 10 1 pack 50" → name=maggi, price=10, quantity=1, quantityunit=pack, stock=50
    - "Add noodles 10 rupees 1 pcs 100 stock" → name=noodles, price=10, quantity=1, unit=pcs, stock=100
    - "rice 500 5 kg 100" → name=rice, price=500, quantity=5, quantityunit=kg, stock=100

    Returns dict with parsed values and confidence flags.
    """
    # Use the smart natural language parser
    result = _parse_natural_language_input(text, mode)

    # Add raw_numbers for compatibility
    result["raw_numbers"] = []

    return result


def _parse_space_separated_input_legacy(text: str, mode: Optional[str]) -> Dict[str, Any]:
    """
    Legacy parser - kept for reference. Use _parse_space_separated_input instead.
    """
    if not text:
        return {"name": None, "price": None, "quantity": None, "quantityunit": None, "stock": None, "parsed_pattern": None}

    cleaned = text.strip()
    if mode in {"add", "update"}:
        # Strip "add" or "update" at beginning, including any following whitespace/newlines
        cleaned = re.sub(r"^(add|update)\b[\s:\-]*", "", cleaned, flags=re.IGNORECASE).strip()

    # Split by whitespace, newlines, or slashes (for "1/10" format)
    # First normalize: replace / with space (for qty/stock format like "1/10")
    normalized = re.sub(r"(\d+)/(\d+)", r"\1 \2", cleaned)
    parts = normalized.split()

    # Parse: name parts, numbers, and unit keyword
    name_parts = []
    numbers = []
    detected_unit = None
    unit_position = -1

    for i, part in enumerate(parts):
        part_lower = part.lower()
        # Check if it's a unit keyword
        if part_lower in UNIT_KEYWORDS:
            detected_unit = part_lower
            unit_position = i
        # Check if it's a number (price/qty/stock)
        elif re.match(r"^\d+(?:,\d+)*(?:\.\d+)?$", part):
            # Handle numbers with commas like "1,000"
            numbers.append((i, part.replace(",", "")))
        else:
            # If we haven't found any numbers yet, it's part of the name
            if not numbers:
                name_parts.append(part)

    result = {
        "name": " ".join(name_parts) if name_parts else None,
        "price": None,
        "quantity": None,
        "quantityunit": detected_unit,
        "stock": None,
        "parsed_pattern": None,
        "raw_numbers": [n[1] for n in numbers],
    }

    # Assign numbers based on position relative to unit
    # Format: name price qty [unit] stock
    # If unit detected: numbers before unit are price, qty; numbers after unit are stock
    # If no unit: price, qty, stock in order

    if detected_unit and len(numbers) >= 2:
        # Find numbers before and after unit
        nums_before_unit = [(i, n) for i, n in numbers if i < unit_position]
        nums_after_unit = [(i, n) for i, n in numbers if i > unit_position]

        if len(nums_before_unit) >= 1:
            try:
                result["price"] = float(nums_before_unit[0][1])
            except ValueError:
                pass
        if len(nums_before_unit) >= 2:
            try:
                result["quantity"] = int(float(nums_before_unit[1][1]))
            except ValueError:
                pass
        if len(nums_after_unit) >= 1:
            try:
                result["stock"] = int(float(nums_after_unit[0][1]))
            except ValueError:
                pass
    else:
        # No unit or not enough numbers - use sequential assignment
        num_values = [n[1] for n in numbers]
        if len(num_values) >= 1:
            try:
                result["price"] = float(num_values[0])
            except ValueError:
                pass
        if len(num_values) >= 2:
            try:
                result["quantity"] = int(float(num_values[1]))
            except ValueError:
                pass
        if len(num_values) >= 3:
            try:
                result["stock"] = int(float(num_values[2]))
            except ValueError:
                pass

    # Determine parsed pattern for clarification
    if name_parts and numbers:
        num_count = len(numbers)
        if num_count == 1:
            result["parsed_pattern"] = "name_price"
        elif num_count == 2:
            result["parsed_pattern"] = "name_price_qty"
        elif num_count >= 3:
            result["parsed_pattern"] = "name_price_qty_stock"
        if detected_unit:
            result["parsed_pattern"] += "_with_unit"

    return result


def _extract_price(text: str) -> Optional[float]:
    match = re.search(r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d{1,2})?)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_integer(text: str) -> Optional[int]:
    match = re.search(r"(\d+)", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ==================== OCR PROCESSING ====================

def _initialize_ocr(debug: bool = False):
    """Lazy initialization of OCR models."""
    global _ocr_initialized
    if _ocr_initialized:
        return

    try:
        from ocr import OLMOCRProcessor, ReasoningModelProcessor, Config as OCRConfig
        print("Initializing OCR models (this may take a moment)...")
        OLMOCRProcessor.initialize(model_name=OCRConfig.OCR_MODEL, debug=debug)
        ReasoningModelProcessor.initialize(model_name=OCRConfig.REASONING_MODEL, debug=debug)
        _ocr_initialized = True
        print("OCR models initialized successfully")
    except Exception as e:
        print(f"Failed to initialize OCR models: {e}")
        raise


def _process_image_with_ocr(image_path: str, debug: bool = False) -> Dict[str, Any]:
    """
    Process an image with OCR and return structured product data.
    Returns a dict with keys like: prod_name, price, quantity, size, colour, etc.
    """
    try:
        from ocr import process_document
        import tempfile

        output_dir = tempfile.mkdtemp(prefix="ocr_output_")
        output_path = process_document(
            file_path=image_path,
            output_dir=output_dir,
            debug=debug
        )

        # Read the extracted JSON
        with open(output_path, "r", encoding="utf-8") as f:
            product_data = json.load(f)

        print(f"OCR extracted product data: {product_data.get('prod_name', 'Unknown')}")
        return product_data

    except Exception as e:
        print(f"OCR processing failed: {e}")
        return {}


def _ocr_data_to_payload(ocr_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OCR extracted data to intake payload format."""
    payload = {
        "name": ocr_data.get("prod_name") or ocr_data.get("name"),
        "price": None,
        "quantity": None,
        "stock": None,
        "category": ocr_data.get("category"),
        "subcategory": ocr_data.get("subcategory"),
        "brand": ocr_data.get("brand"),
        "colour": ocr_data.get("colour"),
        "size": ocr_data.get("size"),
        "description": ocr_data.get("description") or ocr_data.get("descrption"),
        "dimensions": ocr_data.get("dimensions"),
        "raw": json.dumps(ocr_data),
    }

    # Extract price
    price_str = ocr_data.get("price", "")
    if price_str:
        price_val = _extract_price(str(price_str))
        payload["price"] = price_val

    # Extract quantity/stock
    qty_str = ocr_data.get("quantity", "")
    if qty_str:
        payload["quantity"] = _extract_integer(str(qty_str))

    stock_str = ocr_data.get("stock", "")
    if stock_str:
        if isinstance(stock_str, int):
            payload["stock"] = stock_str
        else:
            payload["stock"] = _extract_integer(str(stock_str))

    return payload


# ==================== Image Save Utility ====================

PRODUCT_IMAGES_DIR = Path(Config.BASE_DIR) / "data" / "databases" / "images"
MOCK_IMAGES_BASE_DIR = Path(Config.BASE_DIR) / "data" / "mock_images"


def _normalize_phone(phone: str) -> str:
    """Normalize phone number for comparison (remove +, -, spaces)."""
    return phone.replace("+", "").replace("-", "").replace(" ", "")


def _get_vendor_username_by_phone(phone: str) -> Optional[str]:
    """
    Look up vendor username from phone number.
    Uses mock_vendor.db in demo mode, main vendor.db otherwise.
    """
    import sqlite3

    if DEMO_CONFIG.get("demo_mode", False):
        db_path = Path(Config.BASE_DIR) / DEMO_CONFIG.get("mock_databases", {}).get("vendor_db", "data/databases/mock/mock_vendor.db")
    else:
        db_path = Path(Config.BASE_DIR) / "data" / "databases" / "sql" / "vendor.db"

    if not db_path.exists():
        print(f"⚠️ Vendor DB not found: {db_path}")
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        normalized_phone = _normalize_phone(phone)

        # Search for vendor by normalized phone
        rows = conn.execute("SELECT username, phone FROM vendors").fetchall()
        for row in rows:
            if _normalize_phone(row["phone"]) == normalized_phone:
                username = row["username"]
                conn.close()
                print(f"📦 Found vendor username '{username}' for phone {phone}")
                return username

        conn.close()
        print(f"⚠️ No vendor found for phone {phone}")
        return None
    except Exception as e:
        print(f"❌ Error looking up vendor: {e}")
        return None


def _save_product_image(source_path: str, product_id: str, vendor_phone: str = None) -> Optional[str]:
    """
    Save a product image to the appropriate directory.

    In demo mode: saves to data/mock_images/{vendor_username}/
    In production: saves to data/databases/images/

    Args:
        source_path: Path to the source image file
        product_id: Product ID to use in the filename
        vendor_phone: Vendor's phone number (used to look up username in demo mode)

    Returns:
        Relative path to saved image or None if saving failed
    """
    import shutil

    if not source_path or not os.path.exists(source_path):
        return None

    try:
        # Determine target directory based on mode
        if DEMO_CONFIG.get("demo_mode", False):
            # Look up vendor username by phone
            vendor_username = _get_vendor_username_by_phone(vendor_phone) if vendor_phone else None
            if vendor_username:
                images_dir = MOCK_IMAGES_BASE_DIR / vendor_username
            else:
                # Fallback to generic whatsapp_vendor folder
                images_dir = MOCK_IMAGES_BASE_DIR / "whatsapp_vendor"
        else:
            images_dir = PRODUCT_IMAGES_DIR

        # Create images directory if not exists
        images_dir.mkdir(parents=True, exist_ok=True)

        # Get file extension from source
        _, ext = os.path.splitext(source_path)
        if not ext:
            ext = ".jpg"

        # Create destination filename
        dest_filename = f"{product_id}{ext}"
        dest_path = images_dir / dest_filename

        # Copy the image
        shutil.copy2(source_path, dest_path)

        # Return relative path from project root
        relative_path = str(dest_path.relative_to(Config.BASE_DIR))
        print(f"Saved product image: {relative_path}")
        return relative_path

    except Exception as e:
        print(f"Failed to save product image: {e}")
        return None


def _is_product_image(image_path: str) -> bool:
    """
    Determine if an image is a product image (to be saved) or
    an inventory document (to be OCR processed).

    Simple heuristic:
    - If the image has minimal text content -> likely a product photo
    - If the image has lots of text -> likely an inventory list/document

    For now, we'll check file size and use OCR text length as indicator.
    """
    try:
        from ocr import preprocess_image_for_ocr, extract_text_from_image, is_minimal_text, OLMOCRProcessor

        # Initialize OCR if needed
        try:
            OLMOCRProcessor.get_model()
        except RuntimeError:
            _initialize_ocr(debug=False)

        # Preprocess and extract text
        preprocessed = preprocess_image_for_ocr(image_path, debug=False)
        text = extract_text_from_image(preprocessed, debug=False)

        # Clean up temp file
        try:
            os.remove(preprocessed)
        except Exception:
            pass

        # If minimal text, it's likely a product image
        return is_minimal_text(text)

    except Exception as e:
        print(f"Error checking image type: {e}")
        # Default to treating as product image if we can't determine
        return True


@dataclass
class SessionState:
    user_id: str
    mode: Optional[str] = None  # "add" or "update"
    session_start: float = field(default_factory=_now)
    locked_until: float = 0.0
    awaiting_missing_fields: bool = False
    missing_fields: List[str] = field(default_factory=list)
    retry_count: int = 0
    last_payload: Dict[str, Any] = field(default_factory=dict)
    pending_confirmation: Optional[Dict[str, Any]] = None
    awaiting_inferred_confirmation: bool = False  # For confirming parsed values
    db_snapshot: Optional[Dict[str, Any]] = None  # Snapshot for rollback on timeout

    # Bulk processing queue
    bulk_queue: List[ProductItem] = field(default_factory=list)
    bulk_current_index: int = 0
    bulk_mode: Optional[str] = None  # "add" or "update" for bulk
    bulk_completed: List[Dict[str, Any]] = field(default_factory=list)  # Successfully processed items

    # Parallel bulk processing (all items sent as separate messages)
    bulk_parallel_mode: bool = False  # True = send all at once, False = sequential
    bulk_pending_items: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # msg_index -> {item, status, temp_changes}
    bulk_confirmed_items: List[str] = field(default_factory=list)  # List of confirmed msg_indices
    bulk_message_count: int = 0  # Total messages sent for tracking

    # Multi-product tracking (for LLM-parsed multiple products)
    multi_product_mode: bool = False
    multi_product_payloads: List[Dict[str, Any]] = field(default_factory=list)
    multi_product_similar: List[List[Tuple]] = field(default_factory=list)  # Similar products per item
    multi_product_missing: List[List[str]] = field(default_factory=list)  # Missing fields per item
    multi_product_statuses: List[str] = field(default_factory=list)  # "pending"/"confirmed"/"skipped"
    product_message_map: Dict[str, int] = field(default_factory=dict)  # wamid -> product_index

    # Pending product image (when image sent without product details)
    pending_image_path: Optional[str] = None

    def reset(self):
        self.mode = None
        self.session_start = _now()
        self.locked_until = 0.0
        self.awaiting_missing_fields = False
        self.missing_fields = []
        self.retry_count = 0
        self.last_payload = {}
        self.pending_confirmation = None
        self.awaiting_inferred_confirmation = False
        self.db_snapshot = None
        self.bulk_queue = []
        self.bulk_current_index = 0
        self.bulk_mode = None
        self.bulk_completed = []
        self.bulk_parallel_mode = False
        self.bulk_pending_items = {}
        self.bulk_confirmed_items = []
        self.bulk_message_count = 0
        # Multi-product tracking
        self.multi_product_mode = False
        self.multi_product_payloads = []
        self.multi_product_similar = []
        self.multi_product_missing = []
        self.multi_product_statuses = []
        self.product_message_map = {}
        # Pending image
        self.pending_image_path = None

    def has_bulk_items(self) -> bool:
        """Check if there are items in the bulk queue to process."""
        return len(self.bulk_queue) > self.bulk_current_index

    def get_current_bulk_item(self) -> Optional[ProductItem]:
        """Get the current item being processed from bulk queue."""
        if self.has_bulk_items():
            return self.bulk_queue[self.bulk_current_index]
        return None

    def advance_bulk_queue(self):
        """Move to the next item in bulk queue."""
        self.bulk_current_index += 1

    def has_pending_bulk_items(self) -> bool:
        """Check if there are pending items in parallel bulk mode."""
        return self.bulk_parallel_mode and len(self.bulk_pending_items) > 0

    def get_pending_item_by_index(self, index: str) -> Optional[Dict[str, Any]]:
        """Get pending item by message index."""
        return self.bulk_pending_items.get(index)

    def all_bulk_items_confirmed(self) -> bool:
        """Check if all parallel bulk items are confirmed."""
        if not self.bulk_parallel_mode:
            return False
        return len(self.bulk_confirmed_items) >= len(self.bulk_pending_items)


class BulkTempStorage:
    """File-backed temporary storage for bulk processing items."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, user_id: str) -> Path:
        """Get temp file path for a user."""
        safe_id = user_id.replace("+", "").replace("-", "").replace(" ", "")
        return self.storage_dir / f"bulk_temp_{safe_id}.json"

    def save(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Save bulk pending items to temp file."""
        try:
            file_path = self._get_file_path(user_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                # Convert ProductItem objects to dicts for JSON serialization
                serializable_data = {
                    "bulk_mode": data.get("bulk_mode"),
                    "bulk_message_count": data.get("bulk_message_count", 0),
                    "bulk_confirmed_items": data.get("bulk_confirmed_items", []),
                    "pending_items": {},
                    "timestamp": _now(),
                }
                for idx, item_data in data.get("bulk_pending_items", {}).items():
                    serializable_data["pending_items"][idx] = {
                        "item": item_data["item"].to_payload() if hasattr(item_data["item"], "to_payload") else item_data["item"],
                        "status": item_data["status"],
                        "temp_changes": item_data["temp_changes"],
                        "original": item_data.get("original", {}),
                    }
                json.dump(serializable_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving bulk temp data: {e}")
            return False

    def load(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load bulk pending items from temp file."""
        try:
            file_path = self._get_file_path(user_id)
            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if data is expired (older than 1 hour)
            if _now() - data.get("timestamp", 0) > 3600:
                self.delete(user_id)
                return None

            # Convert back to proper structure
            result = {
                "bulk_mode": data.get("bulk_mode"),
                "bulk_message_count": data.get("bulk_message_count", 0),
                "bulk_confirmed_items": data.get("bulk_confirmed_items", []),
                "bulk_pending_items": {},
            }

            for idx, item_data in data.get("pending_items", {}).items():
                # Recreate ProductItem from saved dict
                item_dict = item_data.get("item", {})
                product_item = ProductItem(
                    name=item_dict.get("name"),
                    price=item_dict.get("price"),
                    quantity=item_dict.get("quantity"),
                    stock=item_dict.get("stock"),
                    category=item_dict.get("category"),
                    brand=item_dict.get("brand"),
                )
                result["bulk_pending_items"][idx] = {
                    "item": product_item,
                    "status": item_data.get("status", "pending"),
                    "temp_changes": item_data.get("temp_changes", {}),
                    "original": item_data.get("original", {}),
                }

            return result
        except Exception as e:
            print(f"Error loading bulk temp data: {e}")
            return None

    def delete(self, user_id: str) -> bool:
        """Delete temp file for a user."""
        try:
            file_path = self._get_file_path(user_id)
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting bulk temp data: {e}")
            return False


class VendorRegistry:
    """Simple file-backed registry for vendor phone numbers."""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)

    def _load(self) -> Dict[str, List[str]]:
        if not self.registry_path.exists():
            return {"registered": []}
        try:
            with self.registry_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"registered": []}

    def is_registered(self, user_id: str) -> bool:
        data = self._load()
        registered = set(data.get("registered", []))
        # Normalize numbers by stripping +, -, spaces
        normalized = user_id.replace("+", "").replace("-", "").replace(" ", "")
        return normalized in {u.replace("+", "").replace("-", "").replace(" ", "") for u in registered}


class InventoryIntake:
    """Encapsulates inventory addition/update decisions and logging."""

    def __init__(self, sql_client: SQLClient):
        self.sql = sql_client
        self._columns_ensured = False

    def _ensure_columns_exist(self):
        """Ensure all required columns exist in the product_table."""
        if self._columns_ensured:
            return

        conn = self.sql._get_connection()
        cur = conn.cursor()

        # New columns to add if they don't exist
        new_columns = [
            ("quantityunit", "TEXT"),
            ("size", "TEXT"),
            ("imageid", "TEXT"),
            ("rating", "REAL"),
            ("other_properties", "TEXT"),
            ("image_path", "TEXT"),  # Local path to product image
        ]

        for col_name, col_type in new_columns:
            try:
                cur.execute(f"ALTER TABLE product_table ADD COLUMN {col_name} {col_type}")
                conn.commit()
                print(f"Added column: {col_name}")
            except Exception:
                # Column already exists
                pass

        self._columns_ensured = True

    def _generate_product_id(self) -> str:
        """Generate a unique product ID."""
        import uuid
        return f"prod_{uuid.uuid4().hex[:12]}"

    def _generate_short_id(self) -> str:
        """Generate a 4-character short ID for easy reference."""
        import random
        import string
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choices(chars, k=4))

    def find_similar(self, name: str, price: Optional[float] = None, vendor_id: Optional[str] = None, top_k: int = 3) -> List[Tuple[str, str, float, Optional[float]]]:
        """Find similar products by name and optionally price.
        If vendor_id provided, only search that vendor's inventory.
        Returns list of (product_id, prod_name, similarity_score, price)."""
        conn = self.sql._get_connection()
        cur = conn.cursor()

        # Filter by vendor if provided
        if vendor_id:
            # Normalize vendor_id for comparison
            normalized_vendor = vendor_id.replace("+", "").replace("-", "").replace(" ", "")
            cur.execute("SELECT product_id, prod_name, price, store FROM product_table")
            rows = [r for r in cur.fetchall() if r["store"] and r["store"].replace("+", "").replace("-", "").replace(" ", "") == normalized_vendor]
        else:
            cur.execute("SELECT product_id, prod_name, price FROM product_table")
            rows = cur.fetchall()

        scored = []
        for row in rows:
            prod_id, prod_name, prod_price = row["product_id"], row["prod_name"], row["price"]
            if not prod_name:
                continue

            # Calculate name similarity (0-1)
            name_score = _similarity(name, prod_name)

            # If price provided, boost score for matching price
            if price is not None and prod_price is not None:
                # Price similarity: 1.0 if exact match, decreasing as difference grows
                price_diff_ratio = abs(price - prod_price) / max(price, prod_price, 1)
                price_score = max(0, 1 - price_diff_ratio)
                # Combined score: 70% name, 30% price
                combined_score = (name_score * 0.7) + (price_score * 0.3)
            else:
                combined_score = name_score

            scored.append((prod_id, prod_name, combined_score, prod_price))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def query_product(self, name: str, vendor_id: str) -> List[Dict[str, Any]]:
        """Query products by name for a specific vendor.
        Returns list of matching products with full details."""
        conn = self.sql._get_connection()
        cur = conn.cursor()

        # Normalize vendor_id
        normalized_vendor = vendor_id.replace("+", "").replace("-", "").replace(" ", "")

        cur.execute("SELECT product_id, prod_name, price, quantity, stock, short_id, store FROM product_table")
        rows = cur.fetchall()

        results = []
        for row in rows:
            # Filter by vendor
            store = row["store"] or ""
            if store.replace("+", "").replace("-", "").replace(" ", "") != normalized_vendor:
                continue

            prod_name = row["prod_name"]
            if not prod_name:
                continue

            # Check similarity
            score = _similarity(name, prod_name)
            if score > 0.3:  # Threshold for relevance
                results.append({
                    "product_id": row["product_id"],
                    "name": prod_name,
                    "price": row["price"],
                    "quantity": row["quantity"] or 1,
                    "stock": row["stock"] or 0,
                    "short_id": row["short_id"],
                    "similarity": score,
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]

    def get_full_inventory(self, vendor_id: str) -> List[Dict[str, Any]]:
        """Get all products for a specific vendor."""
        conn = self.sql._get_connection()
        cur = conn.cursor()

        # Normalize vendor_id
        normalized_vendor = vendor_id.replace("+", "").replace("-", "").replace(" ", "")

        # Try to select with quantityunit, fall back if column doesn't exist
        try:
            cur.execute("SELECT product_id, prod_name, price, quantity, quantityunit, stock, short_id, store FROM product_table")
        except Exception:
            cur.execute("SELECT product_id, prod_name, price, quantity, stock, short_id, store FROM product_table")

        rows = cur.fetchall()

        results = []
        for row in rows:
            # Filter by vendor
            store = row["store"] or ""
            if store.replace("+", "").replace("-", "").replace(" ", "") != normalized_vendor:
                continue

            prod_name = row["prod_name"]
            if not prod_name:
                continue

            # Safely get quantityunit (may not exist in older schemas)
            quantityunit = None
            try:
                quantityunit = row["quantityunit"]
            except (KeyError, IndexError):
                pass

            results.append({
                "product_id": row["product_id"],
                "name": prod_name,
                "price": row["price"],
                "quantity": row["quantity"] or 1,
                "quantityunit": quantityunit or "unit",
                "stock": row["stock"] or 0,
                "short_id": row["short_id"],
            })

        # Sort by name
        results.sort(key=lambda x: x["name"].lower())
        return results

    def update_product_field(self, product_id: str, field: str, value: Any) -> Tuple[bool, Optional[str]]:
        """Update a single field for a product. Returns (success, error_message)."""
        conn = self.sql._get_connection()
        cur = conn.cursor()

        # Map field names to database columns (supports all columns)
        field_map = {
            # Core required fields
            "price": "price",
            "stock": "stock",
            "quantity": "quantity",
            "qty": "quantity",
            "quantityunit": "quantityunit",
            "unit": "quantityunit",
            "name": "prod_name",
            # Additional fields
            "category": "category",
            "subcategory": "subcategory",
            "brand": "brand",
            "colour": "colour",
            "color": "colour",
            "description": "descrption",  # Note: typo in DB schema
            "desc": "descrption",
            "dimensions": "dimensions",
            "size": "size",
            "imageid": "imageid",
            "image": "imageid",
            "rating": "rating",
            "other_properties": "other_properties",
        }

        db_column = field_map.get(field.lower())
        if not db_column:
            available = ", ".join(sorted(set(field_map.keys())))
            return False, f"Unknown field '{field}'. Available: {available}"

        try:
            cur.execute(f"UPDATE product_table SET {db_column} = ? WHERE product_id = ?", (value, product_id))
            conn.commit()
            return cur.rowcount > 0, None
        except Exception as e:
            print(f"Error updating {field}: {e}")
            return False, str(e)

    def update_product_by_short_id(self, short_id: str, vendor_id: str, field: str, value: Any) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Update a product by short_id. Returns (success, error_message, product_info)."""
        conn = self.sql._get_connection()
        cur = conn.cursor()

        # Normalize vendor_id
        normalized_vendor = vendor_id.replace("+", "").replace("-", "").replace(" ", "")

        # Find product by short_id belonging to this vendor
        cur.execute(
            "SELECT product_id, prod_name, store FROM product_table WHERE UPPER(short_id) = UPPER(?)",
            (short_id,)
        )
        row = cur.fetchone()

        if not row:
            return False, f"Product with ID '{short_id}' not found.", None

        # Verify it belongs to this vendor
        store = row["store"] or ""
        if store.replace("+", "").replace("-", "").replace(" ", "") != normalized_vendor:
            return False, f"Product '{short_id}' doesn't belong to your inventory.", None

        product_info = {"product_id": row["product_id"], "name": row["prod_name"]}

        # Get old value for reporting
        field_map = {
            "price": "price", "stock": "stock", "quantity": "quantity", "qty": "quantity",
            "quantityunit": "quantityunit", "unit": "quantityunit",
            "name": "prod_name", "category": "category", "subcategory": "subcategory",
            "brand": "brand", "colour": "colour", "color": "colour",
            "description": "descrption", "desc": "descrption", "dimensions": "dimensions",
            "size": "size", "imageid": "imageid", "image": "imageid",
            "rating": "rating", "other_properties": "other_properties",
        }
        db_column = field_map.get(field.lower())
        if db_column:
            cur.execute(f"SELECT {db_column} FROM product_table WHERE product_id = ?", (row["product_id"],))
            old_row = cur.fetchone()
            product_info["old_value"] = old_row[db_column] if old_row else None

        success, error = self.update_product_field(row["product_id"], field, value)
        return success, error, product_info

    def add_product(self, user_id: str, payload: Dict[str, Any]) -> str:
        """Add a new product to the product_table. Returns the product_id."""
        # Ensure all required columns exist
        self._ensure_columns_exist()

        conn = self.sql._get_connection()
        cur = conn.cursor()

        # Check and add short_id and subcategoryid columns if needed
        cur.execute("PRAGMA table_info(product_table)")
        existing_cols = {row[1] for row in cur.fetchall()}

        if "short_id" not in existing_cols:
            try:
                cur.execute("ALTER TABLE product_table ADD COLUMN short_id TEXT")
                conn.commit()
            except Exception:
                pass

        if "subcategoryid" not in existing_cols:
            try:
                cur.execute("ALTER TABLE product_table ADD COLUMN subcategoryid TEXT")
                conn.commit()
            except Exception:
                pass

        product_id = self._generate_product_id()
        short_id = self._generate_short_id()

        # Serialize dimensions and other_properties to JSON if they're dicts
        dimensions_val = payload.get("dimensions")
        if isinstance(dimensions_val, dict):
            dimensions_val = json.dumps(dimensions_val)

        other_props_val = payload.get("other_properties")
        if isinstance(other_props_val, dict):
            other_props_val = json.dumps(other_props_val)

        # Save product image if provided
        image_path = None
        if payload.get("image_path"):
            image_path = _save_product_image(payload["image_path"], product_id, vendor_phone=user_id)

        cur.execute("""
            INSERT INTO product_table (
                product_id, prod_name, store, category, subcategory, brand,
                colour, description, dimensions, price, quantity, quantityunit, stock,
                size, imageid, rating, other_properties, short_id, image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            product_id,
            payload.get("name"),
            user_id,  # Store as vendor's user_id
            payload.get("category"),
            payload.get("subcategory"),
            payload.get("brand"),
            payload.get("colour"),
            payload.get("description"),
            dimensions_val,
            payload.get("price"),
            payload.get("quantity") or 1,  # Default to 1
            payload.get("quantityunit") or "unit",  # Default to "unit"
            payload.get("stock"),
            payload.get("size"),
            payload.get("imageid"),
            payload.get("rating"),
            other_props_val,
            short_id,
            image_path,
        ))
        conn.commit()
        print(f"Added product: {payload.get('name')} (ID: {product_id}, Short: {short_id})")
        return product_id

    def update_product(self, product_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing product in the product_table.
        Only STOCK is CUMULATIVE (added to existing value).
        Price and quantity (price per X units) are REPLACED.
        Returns dict with old and new values for reporting."""
        conn = self.sql._get_connection()
        cur = conn.cursor()

        # Get current values first
        cur.execute("SELECT prod_name, price, quantity, stock FROM product_table WHERE product_id = ?", (product_id,))
        row = cur.fetchone()
        old_values = {
            "name": row["prod_name"] if row else None,
            "price": row["price"] if row else None,
            "quantity": row["quantity"] if row else 1,
            "stock": row["stock"] if row else 0,
        }

        # Only STOCK is cumulative (inventory count)
        new_stock = (old_values["stock"] or 0) + (payload.get("stock") or 0)

        # Build update query
        updates = []
        values = []

        # Price is REPLACED (not cumulative)
        if payload.get("price") is not None:
            updates.append("price = ?")
            values.append(payload["price"])

        # Quantity is REPLACED (it's "price per X units", not inventory)
        if payload.get("quantity") is not None:
            updates.append("quantity = ?")
            values.append(payload["quantity"])

        # Only STOCK is cumulative
        if payload.get("stock") is not None:
            updates.append("stock = ?")
            values.append(new_stock)

        # Other fields are replaced
        other_fields = {
            "name": "prod_name",
            "category": "category",
            "subcategory": "subcategory",
            "brand": "brand",
            "colour": "colour",
            "description": "descrption",
            "dimensions": "dimensions",
        }

        for payload_key, db_col in other_fields.items():
            if payload.get(payload_key) is not None:
                updates.append(f"{db_col} = ?")
                values.append(payload[payload_key])

        if updates:
            values.append(product_id)
            query = f"UPDATE product_table SET {', '.join(updates)} WHERE product_id = ?"
            cur.execute(query, values)
            conn.commit()

        new_values = {
            "name": payload.get("name") or old_values["name"],
            "price": payload.get("price") or old_values["price"],
            "quantity": payload.get("quantity") or old_values["quantity"],
            "stock": new_stock,
        }

        print(f"Updated product: {old_values['name']} | Stock: {old_values['stock']} -> {new_stock}")
        return {"old": old_values, "new": new_values}

    def enqueue_intake(self, user_id: str, mode: str, payload: Dict[str, Any], matched_product_id: Optional[str] = None) -> Optional[str]:
        """Persist intake intent and process immediately. Returns product_id for new products."""
        conn = self.sql._get_connection()
        cur = conn.cursor()

        # Create intake_queue table for logging
        table = config.INTAKE_QUEUE_TABLE
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                mode TEXT,
                matched_product_id TEXT,
                payload_json TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Process immediately based on mode
        product_id = None
        if mode == "add":
            product_id = self.add_product(user_id, payload)
            matched_product_id = product_id
        elif mode == "update" and matched_product_id:
            self.update_product(matched_product_id, payload)
            product_id = matched_product_id

        # Log to intake queue with 'completed' status
        cur.execute(
            f"INSERT INTO {table} (user_id, mode, matched_product_id, payload_json, status) VALUES (?, ?, ?, ?, ?)",
            (user_id, mode, matched_product_id, json.dumps(payload), "completed"),
        )
        conn.commit()
        return product_id


class VendorIntakeFlow:
    """Stateful flow manager for vendor intake sessions."""

    def __init__(self, sql_client: Optional[SQLClient] = None, registry: Optional[VendorRegistry] = None):
        # Check demo mode
        is_demo = DEMO_CONFIG.get("demo_mode", False)

        # Use vendor test database if configured, or mock DB in demo mode
        if sql_client is None:
            if is_demo:
                # Demo mode: use mock inventory database
                base_path = Path(__file__).parent.parent
                mock_db_path = str(base_path / DEMO_CONFIG["mock_databases"]["inventory_db"])
                print(f"🎭 Demo mode: Using MOCK database: {mock_db_path}")
                self.sql_client = SQLClient(db_path=mock_db_path)
            elif config.USE_VENDOR_TEST_DB:
                print(f"Using VENDOR TEST database: {config.VENDOR_TEST_DB_PATH}")
                self.sql_client = SQLClient(db_path=config.VENDOR_TEST_DB_PATH)
            else:
                self.sql_client = SQLClient()
        else:
            self.sql_client = sql_client

        self.registry = registry or VendorRegistry(config.VENDOR_REGISTRY_FILE)
        self.sessions: Dict[str, SessionState] = {}
        self.intake = InventoryIntake(self.sql_client)
        self.lock_seconds = max(
            config.VENDOR_SESSION_LOCK_MIN_SECONDS,
            min(config.VENDOR_SESSION_LOCK_SECONDS, config.VENDOR_SESSION_LOCK_MAX_SECONDS),
        )

        # File-based temporary storage for bulk processing
        self.bulk_temp_storage = BulkTempStorage(config.VENDOR_DATA_DIR)

        # Store demo mode flag
        self.demo_mode = is_demo

    # --------------------- Public API ---------------------
    def handle(
        self,
        user_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        reply_context: Optional[Dict[str, str]] = None,
        reply_context_id: Optional[str] = None,
        incoming_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point. Returns a response dict ready for WhatsApp/HTTP.

        Args:
            user_id: Vendor's phone number / user ID
            message: The message text
            attachments: List of attachments (images, documents)
            reply_context: If user replied to a message, contains:
                - 'message_id': The ID of the message they replied to
                - 'item_index': (optional) The bulk item index if replying to bulk message
            reply_context_id: WhatsApp message ID being replied to (for multi-product tracking)
            incoming_message_id: This message's WhatsApp ID
        """
        attachments = attachments or []
        session = self.sessions.get(user_id) or SessionState(user_id=user_id)
        self.sessions[user_id] = session
        lower = message.lower().strip()

        # ===== Multi-Product Mode Handling =====
        if session.multi_product_mode:
            # Handle save/cancel commands
            if lower in {"save", "done", "finish"}:
                return self._save_multi_products(session)
            if lower in {"cancel", "abort"}:
                session.reset()
                return {"messages": [{"type": "text", "text": "Cancelled. No products added.\n\nType 'add' to start again."}]}

            # Determine which product - by context or sequential
            product_index = None
            if reply_context_id:
                product_index = session.product_message_map.get(reply_context_id)

            if product_index is None:
                # Sequential: use first pending product
                for i, status in enumerate(session.multi_product_statuses):
                    if status == "pending":
                        product_index = i
                        break

            if product_index is not None:
                return self._handle_multi_product_reply(session, message, product_index)

        if not self.registry.is_registered(user_id):
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            "You need to register before adding or updating inventory. "
                            f"Sign up here: {config.VENDOR_REGISTRATION_URL}"
                        ),
                    }
                ]
            }

        # ===== Greeting Handler =====
        # Handle "hi", "hello", etc. - show welcome menu
        greetings = {"hi", "hello", "hey", "hii", "hiii", "hola", "namaste", "good morning", "good afternoon", "good evening"}
        if lower in greetings or lower.startswith(("hi ", "hello ", "hey ")):
            session.reset()
            session.mode = "menu"  # Lock to menu mode
            return self._show_welcome_menu()

        # ===== Global Exit/Cancel/Reset Commands =====
        # These work in any state to reset the session
        if lower in {"exit", "cancel", "abort", "reset", "stop", "quit", "start over", "nevermind", "nvm", "close", "bye", "goodbye", "close session", "end session", "end"}:
            session.reset()
            session.mode = "closed"  # Mark session as closed - only greeting can reactivate
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            "Session closed. Goodbye! \n\n"
                            "Send *hi* anytime to start a new session."
                        ),
                    }
                ]
            }

        # ===== Handle Closed Session =====
        # If session is closed, only greetings can reactivate it
        if session.mode == "closed":
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": "Session is closed. Send *hi* to start a new session."
                    }
                ]
            }

        # If lock expired, force re-selection unless user is in the middle of a prompt
        if session.locked_until and _now() > session.locked_until:
            # Check if we're in a pending confirmation state with a snapshot
            pending_action = session.pending_confirmation.get("action") if session.pending_confirmation else None
            if pending_action in {"verify_final", "verify_change"}:
                # Timeout during verification - rollback to previous state
                if session.db_snapshot:
                    restored = self._restore_snapshot(session)
                    product_name = session.pending_confirmation.get("product_name", "product")
                    short_id = session.pending_confirmation.get("short_id", "")
                    session.pending_confirmation = None
                    session.mode = None
                    session.db_snapshot = None

                    if restored:
                        id_info = f" ({short_id})" if short_id else ""
                        return {
                            "messages": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"⏰ Session timed out. Changes to *{product_name}*{id_info} have been rolled back.\n\n"
                                        "The product has been restored to its previous state.\n"
                                        "Please start again when you're ready.\n\n"
                                        "• Type 'inventory' to see your products\n"
                                        "• Type 'add' or 'update' to modify inventory"
                                    ),
                                }
                            ]
                        }
            session.mode = None

        # Try to restore bulk state from temp file if session doesn't have it
        if not session.has_pending_bulk_items() and session.bulk_parallel_mode is False:
            saved_bulk = self.bulk_temp_storage.load(user_id)
            if saved_bulk:
                session.bulk_parallel_mode = True
                session.bulk_mode = saved_bulk.get("bulk_mode")
                session.bulk_pending_items = saved_bulk.get("bulk_pending_items", {})
                session.bulk_confirmed_items = saved_bulk.get("bulk_confirmed_items", [])
                session.bulk_message_count = saved_bulk.get("bulk_message_count", 0)

        # Handle parallel bulk mode (user replied to a specific message)
        if session.has_pending_bulk_items():
            return self._handle_parallel_bulk_reply(session, message, reply_context)

        # Handle pending confirmation for similar products
        if session.pending_confirmation:
            return self._handle_confirmation(session, message)

        # Check if we have bulk items to process (sequential mode)
        if session.has_bulk_items() and not session.bulk_parallel_mode:
            return self._process_next_bulk_item(session, message)

        # Check for bulk input (multiple products at once)
        bulk_result = self._detect_and_handle_bulk(session, message)
        if bulk_result:
            return bulk_result

        # Skip query/change/remove handlers when attachments are present
        # Attachments indicate user is adding/updating a product with an image
        if not attachments:
            # Check for change commands like "change maggi price to 15"
            change_result = self._handle_change_command(user_id, message)
            if change_result:
                return change_result

            # Check for remove commands like "remove 4JTH" or "delete ABCD"
            remove_result = self._handle_remove_command(user_id, message)
            if remove_result:
                return remove_result

            # Check for query patterns like "what is stock of maggi", "inventory", "list"
            query_result = self._handle_query(user_id, message)
            if query_result:
                return query_result

        # Detect explicit mode switches
        detected_mode = self._detect_mode(message)
        if detected_mode:
            session.mode = detected_mode
            session.session_start = _now()
            session.locked_until = session.session_start + self.lock_seconds

        # ===== OCR Processing for Images =====
        ocr_payload = None
        product_image_path = None  # Path to product image to be saved

        # Check if there's a pending image from a previous message
        if session.pending_image_path and not attachments:
            product_image_path = session.pending_image_path
            session.pending_image_path = None  # Clear it after use

        if attachments:
            image_attachments = [a for a in attachments if a.get("type") == "image" and a.get("path")]
            doc_attachments = [a for a in attachments if a.get("type") == "document" and a.get("path")]

            # Check if text provides product info (not just "add" command)
            text_is_minimal = not message.strip() or message.strip().lower() in {"add", "update", "add new", "new"}
            text_has_product_info = bool(message.strip()) and message.strip().lower() not in {"add", "update", "add new", "new"}

            if image_attachments or doc_attachments:
                # Default to 'add' mode if not set and image is sent
                if session.mode not in {"add", "update"}:
                    session.mode = "add"
                    session.session_start = _now()
                    session.locked_until = session.session_start + self.lock_seconds

                attachment_to_process = image_attachments[0] if image_attachments else doc_attachments[0]
                file_path = attachment_to_process.get("path")

                # Case 1: Image + text with product info -> Image is product photo, text is details
                if text_has_product_info and image_attachments:
                    print(f"Product image detected (text provided): {file_path}")
                    product_image_path = file_path
                    # Don't process with OCR, let text extraction handle the details

                # Case 2: Image/doc with minimal text -> Process with OCR
                elif text_is_minimal:
                    try:
                        print(f"Processing attachment with OCR: {file_path}")

                        # Initialize OCR models if not already done
                        _initialize_ocr(debug=False)

                        # Process the image/document with OCR
                        ocr_data = _process_image_with_ocr(file_path, debug=False)

                        if ocr_data and ocr_data.get("prod_name"):
                            ocr_payload = _ocr_data_to_payload(ocr_data)
                            # If OCR extracts a single product, the image might be a product photo
                            # Store the image path for saving
                            if image_attachments:
                                ocr_payload["image_path"] = file_path
                            print(f"OCR extracted: {ocr_payload.get('name')}")
                        else:
                            # OCR couldn't extract product info - maybe it's just a product photo
                            # Ask user to provide text details
                            if image_attachments:
                                # Store in session for next message
                                session.pending_image_path = file_path
                                return {
                                    "messages": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "I see you sent an image. Is this a product photo?\n\n"
                                                "Please provide the product details:\n"
                                                "• Product name\n"
                                                "• Price\n"
                                                "• Stock quantity\n\n"
                                                "Example: 'Red Cotton Shirt Rs 500 stock 50'"
                                            ),
                                        }
                                    ],
                                }
                            else:
                                return {
                                    "messages": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "I couldn't extract product information from this document. "
                                                "Please try sending a clearer image or provide the details as text:\n"
                                                "• Product name\n• Price\n• Quantity/Stock"
                                            ),
                                        }
                                    ]
                                }

                    except Exception as e:
                        print(f"OCR error: {e}")
                        return {
                            "messages": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Sorry, I had trouble processing your image. "
                                        "Please try again or send the product details as text."
                                    ),
                                }
                            ]
                        }
        # ===== End OCR Processing =====

        if session.mode not in {"add", "update"}:
            return self._prompt_mode(session)

        # Run payload extraction and validation
        # If OCR provided data, use it as the base (single product)
        if ocr_payload:
            payload = ocr_payload
            payload["attachments"] = [att.get("type", "image") for att in attachments]
            missing_fields = self._check_missing_fields(payload)
            payloads = [payload]
            all_missing = [missing_fields]
        else:
            # _extract_payload now returns lists for multi-product support
            payloads, all_missing = self._extract_payload(message, attachments, session)

            # If product image was detected (image + text), attach it to the payload(s)
            if product_image_path and payloads:
                for p in payloads:
                    p["image_path"] = product_image_path

        # ===== Multi-Product Flow =====
        if len(payloads) > 1:
            return self._handle_multi_product_intake(session, payloads, all_missing)

        # ===== Single Product Flow (backward compatible) =====
        payload = payloads[0]
        missing_fields = all_missing[0]

        if missing_fields:
            return self._handle_missing_fields(session, missing_fields, payload)

        # All fields present - check for similar products and ask for confirmation
        # Only search this vendor's inventory
        similar = self.intake.find_similar(payload["name"], payload.get("price"), session.user_id) if payload.get("name") else []

        # Always ask for confirmation when values were inferred from space-separated input
        # This helps catch errors and offers existing product suggestions
        if payload.get("parsed_pattern"):
            # Values were inferred - ask for confirmation with suggestions
            session.pending_confirmation = {
                "action": "inferred_add" if session.mode == "add" else "inferred_update",
                "matches": similar,
                "payload": payload,
            }
            return self._format_inferred_confirmation(payload, similar, session.mode)

        # Standard similarity checks for explicit input
        if session.mode == "add" and similar and similar[0][2] >= config.VENDOR_SIMILARITY_THRESHOLD:
            session.pending_confirmation = {
                "action": "similar_add",
                "matches": similar,
                "payload": payload,
            }
            return self._format_similar_prompt(similar, addition=True)

        if session.mode == "update":
            if not similar or similar[0][2] < config.VENDOR_UPDATE_MIN_SIMILARITY:
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                "This product isn't in the inventory. "
                                "Share more details or say 'add as new' to create it."
                            ),
                        }
                    ]
                }
            session.pending_confirmation = {
                "action": "update_match",
                "matches": similar,
                "payload": payload,
            }
            return self._format_similar_prompt(similar, addition=False)

        # No confirmations required; enqueue directly
        return self._complete_intake(session, payload, matched_product_id=None)

    # --------------------- Internal helpers ---------------------
    def _detect_mode(self, text: str) -> Optional[str]:
        lower = text.lower()
        if "update" in lower:
            return "update"
        if "add" in lower or "addition" in lower or "new product" in lower:
            return "add"
        return None

    def _handle_change_command(self, user_id: str, message: str) -> Optional[Dict[str, Any]]:
        """Handle change commands like 'change JXYK price to 15' or 'set maggi stock to 100'."""
        session = self.sessions.get(user_id)
        if not session:
            session = SessionState(user_id=user_id)
            self.sessions[user_id] = session

        lower = message.lower().strip()

        # All supported fields
        fields_pattern = r"(price|stock|quantity|qty|name|category|subcategory|brand|colour|color|description|desc|dimensions|size|rating)"

        # Change command patterns
        change_patterns = [
            # "change [product] [field] to [value]"
            rf"(?:change|set|update)\s+(\S+)\s+{fields_pattern}\s+(?:to|=)\s*(.+)",
            # "[product] [field] to [value]"
            rf"(\S+)\s+{fields_pattern}\s+(?:to|=)\s*(.+)",
            # "change [field] of [product] to [value]"
            rf"(?:change|set|update)\s+{fields_pattern}\s+(?:of|for)\s+(\S+)\s+(?:to|=)\s*(.+)",
        ]

        product_ref = None
        field = None
        value = None

        for i, pattern in enumerate(change_patterns):
            match = re.search(pattern, lower)
            if match:
                if i == 2:  # Pattern with field before product
                    field = match.group(1)
                    product_ref = match.group(2).strip()
                    value = match.group(3).strip()
                else:
                    product_ref = match.group(1).strip()
                    field = match.group(2)
                    value = match.group(3).strip()
                break

        if not product_ref or not field or not value:
            return None

        # Normalize field aliases
        field_aliases = {"qty": "quantity", "color": "colour", "desc": "description"}
        field = field_aliases.get(field, field)

        # Convert value based on field type
        numeric_fields = {"price", "stock", "quantity", "rating"}
        if field in numeric_fields:
            try:
                if field == "price" or field == "rating":
                    value = float(value)
                else:
                    value = int(float(value))
            except ValueError:
                return {
                    "messages": [
                        {"type": "text", "text": f"'{value}' is not a valid number for {field}."}
                    ]
                }

        # Find the product first
        matched_product = None
        product_id = None

        # Try to find by short_id first (4-character ID)
        if len(product_ref) == 4 and product_ref.isalnum():
            conn = self.sql_client._get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT product_id, prod_name, price, quantity, stock, short_id, store FROM product_table WHERE UPPER(short_id) = UPPER(?)",
                (product_ref,)
            )
            row = cur.fetchone()
            if row:
                # Verify ownership
                normalized_vendor = user_id.replace("+", "").replace("-", "").replace(" ", "")
                store = row["store"] or ""
                if store.replace("+", "").replace("-", "").replace(" ", "") == normalized_vendor:
                    matched_product = {
                        "product_id": row["product_id"],
                        "name": row["prod_name"],
                        "price": row["price"],
                        "quantity": row["quantity"] or 1,
                        "stock": row["stock"] or 0,
                        "short_id": row["short_id"],
                    }
                    product_id = row["product_id"]

        # If not found by short_id, try name similarity
        if not matched_product:
            inventory = self.intake.get_full_inventory(user_id)
            for p in inventory:
                if p["short_id"] and p["short_id"].lower() == product_ref.lower():
                    matched_product = p
                    product_id = p["product_id"]
                    break

            if not matched_product:
                best_match = None
                best_score = 0
                for p in inventory:
                    score = _similarity(product_ref, p["name"])
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match = p
                if best_match:
                    matched_product = best_match
                    product_id = best_match["product_id"]

        if not matched_product:
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"Product \"{product_ref}\" not found in your inventory.\n\n"
                            f"Use 'inventory' to see all your products and their IDs."
                        ),
                    }
                ]
            }

        # Save snapshot BEFORE making changes (for rollback on timeout)
        old_value = matched_product.get(field, "N/A")
        session.db_snapshot = {
            "product_id": product_id,
            "was_new_product": False,
            "field": field,
            "old_value": old_value,
            "price": matched_product.get("price"),
            "quantity": matched_product.get("quantity"),
            "stock": matched_product.get("stock"),
        }

        # Update the field
        success, error = self.intake.update_product_field(product_id, field, value)

        if success:
            # Set up confirmation loop
            session.pending_confirmation = {
                "action": "verify_change",
                "product_id": product_id,
                "product_name": matched_product["name"],
                "short_id": matched_product["short_id"],
                "field": field,
                "old_value": old_value,
                "new_value": value,
            }
            session.locked_until = _now() + self.lock_seconds

            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"Updated *{matched_product['name']}* ({matched_product['short_id']}):\n"
                            f"   {field.capitalize()}: {old_value} -> {value}\n\n"
                            f"Is this correct? Reply 'yes' to confirm.\n\n"
                            f"If not, tell me the correction:\n"
                            f"• \"{field} is [correct value]\"\n"
                            f"• Example: {field} is {old_value}"
                        ),
                    }
                ]
            }
        else:
            session.db_snapshot = None  # Clear snapshot on failure
            error_msg = error if error else "Unknown error"
            return {
                "messages": [
                    {"type": "text", "text": f"Failed to update {field}: {error_msg}"}
                ]
            }

    def _handle_remove_command(self, user_id: str, message: str) -> Optional[Dict[str, Any]]:
        """Handle remove commands like 'remove 4JTH' or 'delete ABCD from inventory'."""
        lower = message.lower().strip()

        # Remove command patterns - ONLY match short_id (4 alphanumeric characters)
        remove_patterns = [
            # "remove 4JTH" or "delete ABCD"
            r"(?:remove|delete)\s+([A-Za-z0-9]{4})(?:\s|$)",
            # "remove 4JTH from inventory"
            r"(?:remove|delete)\s+([A-Za-z0-9]{4})\s+(?:from\s+)?(?:inventory|stock|database)",
        ]

        short_id = None
        for pattern in remove_patterns:
            match = re.search(pattern, lower)
            if match:
                short_id = match.group(1).upper()
                break

        if not short_id:
            return None

        # Look up product by short_id
        conn = self.sql_client._get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT product_id, prod_name, price, stock, short_id, store FROM product_table WHERE UPPER(short_id) = ?",
            (short_id,)
        )
        row = cur.fetchone()

        if not row:
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": f"No product found with ID *{short_id}*.\n\nType 'inventory' to see your products."
                    }
                ]
            }

        # Verify ownership
        normalized_vendor = user_id.replace("+", "").replace("-", "").replace(" ", "")
        store = row["store"] or ""
        if store.replace("+", "").replace("-", "").replace(" ", "") != normalized_vendor:
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": f"Product *{short_id}* doesn't belong to your inventory."
                    }
                ]
            }

        product_id = row["product_id"]
        product_name = row["prod_name"]
        price = row["price"]
        stock = row["stock"]

        # Set up confirmation for deletion
        session = self.sessions.get(user_id) or SessionState(user_id=user_id)
        self.sessions[user_id] = session

        session.pending_confirmation = {
            "action": "confirm_delete",
            "product_id": product_id,
            "product_name": product_name,
            "short_id": short_id,
            "price": price,
            "stock": stock,
        }
        session.locked_until = _now() + self.lock_seconds

        return {
            "messages": [
                {
                    "type": "text",
                    "text": (
                        f"Are you sure you want to *permanently delete* this product?\n\n"
                        f"*{product_name}* ({short_id})\n"
                        f"   Price: Rs {price}\n"
                        f"   Stock: {stock}\n\n"
                        f"WARNING: This action cannot be undone!\n\n"
                        f"Reply 'yes' to confirm deletion, or 'no' to cancel."
                    ),
                }
            ]
        }

    def _handle_query(self, user_id: str, message: str) -> Optional[Dict[str, Any]]:
        """Handle inventory query patterns like 'what is stock of maggi' or 'show inventory'."""
        lower = message.lower().strip()

        # IMPORTANT: Don't treat "add X" or "update X" as queries
        # These are product operations, not inventory queries
        if re.match(r"^(add|update)\s+", lower):
            return None

        # Check for "what are we updating" query
        update_query_patterns = [
            r"^what(?:'s| is| are)?\s*(?:we\s+)?(?:updating|adding|changing|pending)",
            r"^(?:show|get|check)\s+(?:pending|current)\s+(?:update|changes?|items?)",
            r"^pending\s*(?:update|changes?|items?)?$",
            r"^status$",
        ]

        is_update_query = any(re.match(p, lower) for p in update_query_patterns)

        if is_update_query:
            session = self._get_session(user_id)

            # Check for pending confirmation
            if session.pending_confirmation:
                payload = session.last_payload or {}
                preview = _format_preview(payload)
                return {"messages": [{"type": "text", "text": preview}]}

            # Check for bulk pending items
            if session.has_pending_bulk_items():
                return self._show_bulk_status(session)

            # Check temp storage
            temp_storage = BulkTempStorage(user_id)
            saved_data = temp_storage.load()
            if saved_data and saved_data.get("pending_items"):
                items = saved_data["pending_items"]
                lines = ["*Pending Updates:*", "━━━━━━━━━━━━━━━━━━━━━━"]
                for idx, item_data in items.items():
                    item = item_data.get("item", {})
                    status = item_data.get("status", "pending")
                    icon = "[OK]" if status == "confirmed" else "[...]"
                    name = item.get("name", "Unknown")
                    price = item.get("price", "—")
                    qty = item.get("quantity", 1)
                    unit = item.get("quantityunit", "unit")
                    stock = item.get("stock", 0)
                    lines.append(f"{icon} {idx}. *{name}* - ₹{price} ({qty} {unit}) | Stock: {stock}")

                lines.append("━━━━━━━━━━━━━━━━━━━━━━")
                lines.append("Reply 'done' to commit or 'cancel' to abort.")
                return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

            # No pending updates
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": "No pending updates.\n\nTo add a product, send:\nadd [name] [price] [qty] [unit] [stock]\n\nExample: add maggi 10 1 pack 50"
                    }
                ]
            }

        # Check for full inventory request first
        # Include fuzzy matching for common misspellings
        def _is_similar_to_inventory(word: str) -> bool:
            """Check if word is similar to 'inventory' (handles typos like 'invemtory')."""
            if len(word) < 5:
                return False
            # Use SequenceMatcher for fuzzy matching
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, word, "inventory").ratio()
            return ratio >= 0.75  # 75% similarity threshold

        inventory_patterns = [
            r"^(?:show|list|get|view|my|all)?\s*(?:my\s*)?(?:full\s*)?inventory$",
            r"^(?:show|list|get|view)\s+(?:all\s+)?(?:my\s+)?products?$",
            r"^(?:all|my)\s+products?$",
            r"^inventory$",
            r"^products?$",
            r"^list$",
        ]

        is_full_inventory = any(re.match(p, lower) for p in inventory_patterns)

        # Fuzzy match for single-word commands that look like "inventory"
        if not is_full_inventory and len(lower.split()) == 1:
            is_full_inventory = _is_similar_to_inventory(lower)

        if is_full_inventory:
            # Return full inventory
            inventory = self.intake.get_full_inventory(user_id)

            if not inventory:
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                "📦 Your inventory is empty.\n\n"
                                "━━━━━━━━━━━━━━━━━━━━━━\n"
                                "➕ *Add Products*\n"
                                "   add [name] [price] [qty] [unit] [stock]\n"
                                "   Example: add Maggi 10 1 pack 50\n\n"
                                "📋 *Bulk Add*\n"
                                "   bulk add\n"
                                "   Maggi 10 1 50\n"
                                "   Chips 20 1 100\n\n"
                                "❌ *Exit*\n"
                                "   Type: exit or close\n"
                                "━━━━━━━━━━━━━━━━━━━━━━"
                            ),
                        }
                    ]
                }

            lines = ["📦 *Your Inventory*", "━━━━━━━━━━━━━━━━━━━━━━"]

            for i, p in enumerate(inventory, start=1):
                lines.append(f"{i}. *{p['name']}* (ID: {p['short_id']})")
                unit = p.get('quantityunit', 'unit') or 'unit'
                lines.append(f"   ₹{p['price']} per {p['quantity']} {unit} | Stock: {p['stock']}")

            lines.append("━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"Total: {len(inventory)} product(s)")
            lines.append("")
            lines.append("*What would you like to do next?*")
            lines.append("")
            lines.append("➕ *Add Products*")
            lines.append("   Type: add [name] [price] [qty] [unit] [stock]")
            lines.append("")
            lines.append("🔄 *Quick Change*")
            lines.append("   change [ID] [field] to [value]")
            lines.append("   Example: change JXYK stock to 50")
            lines.append("")
            lines.append("🗑️ *Remove Product*")
            lines.append("   remove [ID]")
            lines.append("")
            lines.append("❌ *Exit*")
            lines.append("   Type: exit or close")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━")

            return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

        # Query patterns for specific product
        query_patterns = [
            r"(?:what(?:'s| is)?|show|get|check|current)\s*(?:the\s*)?(?:stock|inventory|price|details?)\s*(?:of|for)?\s*(.+)",
            r"(?:stock|inventory|price)\s*(?:of|for)\s*(.+)",
            r"how much\s*(.+)\s*(?:do i have|in stock|left)",
            r"(.+)\s*(?:stock|inventory|price)\??$",
        ]

        product_name = None
        for pattern in query_patterns:
            match = re.search(pattern, lower)
            if match:
                product_name = match.group(1).strip()
                # Clean up common words
                product_name = re.sub(r"\b(please|pls|the|my|i have)\b", "", product_name).strip()
                if product_name:
                    break

        if not product_name:
            return None

        # Query vendor's inventory only
        results = self.intake.query_product(product_name, user_id)

        if results:
            lines = [f"Found in your inventory:"]
            for r in results:
                lines.append(f"")
                lines.append(f"• *{r['name']}* (ID: {r['short_id']})")
                lines.append(f"  Price: ₹{r['price']} per {r['quantity']} unit")
                lines.append(f"  Stock: {r['stock']} units")

            lines.append("")
            lines.append("*Quick Edit:*")
            lines.append("• \"change [name/ID] price to [value]\"")
            lines.append("• \"change [name/ID] stock to [value]\"")

            return {"messages": [{"type": "text", "text": "\n".join(lines)}]}
        else:
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"No product matching \"{product_name}\" found in your inventory.\n\n"
                            "To add it, send: add [name] [price] [qty] [unit] [stock]\n"
                            "Example: add Maggi 10 1 pack 50"
                        ),
                    }
                ]
            }

    def _show_welcome_menu(self) -> Dict[str, Any]:
        """Show the welcome menu with emoji formatting."""
        return {
            "messages": [
                {
                    "type": "text",
                    "text": (
                        "👋 Welcome! What would you like to do?\n\n"
                        "━━━━━━━━━━━━━━━━━━━━━━\n"
                        "📦 *View Inventory*\n"
                        "   Type: inventory\n\n"
                        "➕ *Add Product(s)*\n"
                        "   add [name] [price] [qty] [unit] [stock]\n"
                        "   Example: add Maggi 10 1 pack 50\n\n"
                        "📋 *Bulk Add/Update*\n"
                        "   Send multiple products, one per line:\n"
                        "   bulk add\n"
                        "   Maggi 10 1 50\n"
                        "   Chips 20 1 100\n"
                        "   Biscuits 15 1 75\n\n"
                        "🔄 *Quick Change*\n"
                        "   change [ID] [field] to [value]\n"
                        "   Example: change JXYK stock to 50\n\n"
                        "❌ *Exit*\n"
                        "   Type: exit or close\n"
                        "━━━━━━━━━━━━━━━━━━━━━━"
                    ),
                }
            ]
        }

    def _prompt_mode(self, session: SessionState) -> Dict[str, Any]:
        session.mode = None
        session.pending_confirmation = None
        return self._show_welcome_menu()

    # --------------------- Bulk Processing Methods ---------------------

    def _detect_and_handle_bulk(self, session: SessionState, message: str) -> Optional[Dict[str, Any]]:
        """Detect bulk input and initialize queue for processing."""
        lower = message.lower().strip()

        # Check for explicit bulk commands
        is_bulk_command = lower.startswith(('bulk add', 'bulk update', 'bulk'))

        # Check if message has multiple lines (potential bulk input)
        text_lines = [l.strip() for l in message.strip().split('\n') if l.strip()]
        has_multiple_lines = len(text_lines) > 1

        # Not a bulk input
        if not is_bulk_command and not has_multiple_lines:
            return None

        # Determine mode from command
        if 'update' in lower.split('\n')[0].lower():
            bulk_mode = 'update'
        else:
            bulk_mode = 'add'

        # Parse the bulk input
        bulk_input = BulkProductInput.parse_bulk_text(message)

        if not bulk_input.items:
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            "Couldn't parse any products from your input.\n\n"
                            "Please use this format (one product per line):\n"
                            "━━━━━━━━━━━━━━━━━━━━━━\n"
                            "bulk add\n"
                            "[name] [price] [qty] [unit] [stock]\n"
                            "[name] [price] [qty] [unit] [stock]\n"
                            "━━━━━━━━━━━━━━━━━━━━━━\n"
                            "Example:\n"
                            "bulk add\n"
                            "Maggi 10 1 50\n"
                            "Chips 20 1 100"
                        ),
                    }
                ]
            }

        # Use PARALLEL mode - send all items as separate messages
        session.bulk_parallel_mode = True
        session.bulk_mode = bulk_mode
        session.bulk_pending_items = {}
        session.bulk_confirmed_items = []
        session.bulk_completed = []
        session.bulk_message_count = len(bulk_input.items)
        session.locked_until = _now() + (self.lock_seconds * 3)  # Extended timeout for parallel bulk

        # Build multiple messages - one per item
        messages = []

        # First message: Summary
        summary_lines = [
            f"*Bulk {bulk_mode.capitalize()}*: Found {len(bulk_input.items)} product(s)",
            "━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "I'll send each item as a separate message.",
            "Reply to each message with:",
            "• 'yes' to confirm",
            "• 'no' or corrections like 'price is 20'",
            "",
            "After confirming all items, reply 'done' to save changes.",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]
        messages.append({
            "type": "text",
            "text": "\n".join(summary_lines),
            "bulk_type": "summary",
        })

        # Individual messages for each item
        for idx, item in enumerate(bulk_input.items):
            item_index = str(idx + 1)

            # Store in pending items (temporary - not committed to DB yet)
            session.bulk_pending_items[item_index] = {
                "item": item,
                "status": "pending",  # pending, confirmed, skipped
                "temp_changes": item.to_payload(),  # Temporary changes
                "original": item.to_payload().copy(),  # Original parsed values
            }

            # Build item message
            missing = item.get_missing_required_fields()
            status_icon = "[!]" if missing else ""

            item_lines = [
                f"{status_icon} *Item {item_index}/{len(bulk_input.items)}* ({bulk_mode.upper()})",
                "━━━━━━━━━━━━━━━━━━━━━━",
                f"• Name: {item.name or ' Missing'}",
                f"• Price: ₹{item.price if item.price is not None else ''} per {item.quantity or 1} unit",
                f"• Stock: {item.stock if item.stock is not None else ''}",
                "━━━━━━━━━━━━━━━━━━━━━━",
            ]

            if missing:
                item_lines.append(f"Warning: Missing: {', '.join(missing)}")
                item_lines.append("")
                item_lines.append("Reply with missing info or full details:")
                item_lines.append("[name] [price] [qty] [unit] [stock]")
            else:
                item_lines.append("")
                item_lines.append("Reply to this message:")
                item_lines.append("• 'yes' to confirm")
                item_lines.append("• 'skip' to skip")
                item_lines.append("• Or correction: 'price is 25'")

            messages.append({
                "type": "text",
                "text": "\n".join(item_lines),
                "bulk_type": "item",
                "item_index": item_index,
            })

        # Save to temp file for persistence
        self.bulk_temp_storage.save(session.user_id, {
            "bulk_mode": session.bulk_mode,
            "bulk_message_count": session.bulk_message_count,
            "bulk_confirmed_items": session.bulk_confirmed_items,
            "bulk_pending_items": session.bulk_pending_items,
        })

        return {"messages": messages}

    def _handle_parallel_bulk_reply(
        self,
        session: SessionState,
        message: str,
        reply_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Handle replies in parallel bulk mode."""
        lower = message.lower().strip()

        # Check for global commands
        if lower in {'cancel', 'stop', 'abort'}:
            self.bulk_temp_storage.delete(session.user_id)  # Delete temp file
            session.reset()
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": "Bulk processing cancelled. No changes were saved.\n\nType 'inventory' to see your products.",
                    }
                ]
            }

        # Check for "done" - commit all confirmed items
        if lower in {'done', 'save', 'finish', 'commit'}:
            return self._commit_parallel_bulk(session)

        # Check for "status" - show current status
        if lower in {'status', 'show'}:
            return self._show_parallel_bulk_status(session)

        # Try to determine which item the user is replying to
        item_index = None

        # 1. From reply context (WhatsApp quote/reply)
        if reply_context and reply_context.get('item_index'):
            item_index = reply_context['item_index']

        # 2. From message pattern like "1: yes" or "item 1 yes" or "#1 price is 20"
        if not item_index:
            index_patterns = [
                r'^(\d+)\s*[:\-]?\s*(.+)$',  # "1: yes" or "1 yes"
                r'^#(\d+)\s*(.+)$',  # "#1 yes"
                r'^item\s*(\d+)\s*[:\-]?\s*(.+)$',  # "item 1 yes"
            ]
            for pattern in index_patterns:
                match = re.match(pattern, message, re.IGNORECASE)
                if match:
                    item_index = match.group(1)
                    message = match.group(2).strip()
                    lower = message.lower()
                    break

        # 3. If still no index and only one pending item, use that
        if not item_index:
            pending = [k for k, v in session.bulk_pending_items.items() if v["status"] == "pending"]
            if len(pending) == 1:
                item_index = pending[0]

        if not item_index:
            # Show help
            pending_count = len([v for v in session.bulk_pending_items.values() if v["status"] == "pending"])
            confirmed_count = len(session.bulk_confirmed_items)
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f" Which item are you referring to?\n\n"
                            f"Status: {confirmed_count} confirmed, {pending_count} pending\n\n"
                            f"Reply with item number:\n"
                            f"• \"1: yes\" to confirm item 1\n"
                            f"• \"2: price is 20\" to correct item 2\n"
                            f"• \"3: skip\" to skip item 3\n\n"
                            f"Or:\n"
                            f"• 'done' to save all confirmed items\n"
                            f"• 'status' to see all items\n"
                            f"• 'cancel' to abort"
                        ),
                    }
                ]
            }

        # Get the pending item
        pending_item = session.bulk_pending_items.get(item_index)
        if not pending_item:
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": f"Item {item_index} not found. Use 'status' to see all items.",
                    }
                ]
            }

        # Handle the reply for this specific item
        if _is_affirmative(message):
            # Check if item is complete
            item = pending_item["item"]
            missing = item.get_missing_required_fields()

            if missing:
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                f"Warning: Item {item_index} is missing: {', '.join(missing)}\n\n"
                                f"Please provide the missing values:\n"
                                f"{item_index}: [name] [price] [qty] [unit] [stock]"
                            ),
                        }
                    ]
                }

            # Mark as confirmed (still temporary - not saved to DB yet)
            pending_item["status"] = "confirmed"
            if item_index not in session.bulk_confirmed_items:
                session.bulk_confirmed_items.append(item_index)

            # Save state to temp file
            self._save_bulk_state(session)

            # Check progress
            pending_count = len([v for v in session.bulk_pending_items.values() if v["status"] == "pending"])
            confirmed_count = len(session.bulk_confirmed_items)

            if pending_count == 0:
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                f"Item {item_index} confirmed!\n\n"
                                f"All {confirmed_count} items confirmed!\n\n"
                                f"Reply 'done' to SAVE all changes to database.\n"
                                f"Reply 'cancel' to discard all changes."
                            ),
                        }
                    ]
                }

            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"Item {item_index} confirmed!\n\n"
                            f"Progress: {confirmed_count}/{len(session.bulk_pending_items)} confirmed\n"
                            f"Remaining: {pending_count} item(s)\n\n"
                            f"Continue confirming, or reply 'done' when ready to save."
                        ),
                    }
                ]
            }

        elif lower in {'skip', 'no', 'remove'}:
            pending_item["status"] = "skipped"
            self._save_bulk_state(session)  # Save state to temp file
            pending_count = len([v for v in session.bulk_pending_items.values() if v["status"] == "pending"])

            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f" Item {item_index} skipped.\n\n"
                            f"Remaining: {pending_count} pending item(s)\n"
                            f"Reply 'done' when ready to save confirmed items."
                        ),
                    }
                ]
            }

        else:
            # Try to parse corrections
            corrections = self._parse_bulk_item_correction(message)

            # Also try to parse as full product line
            if not corrections:
                parsed = BulkProductInput._parse_single_line(message)
                if parsed:
                    if parsed.name:
                        corrections["name"] = parsed.name
                    if parsed.price is not None:
                        corrections["price"] = parsed.price
                    if parsed.quantity is not None:
                        corrections["quantity"] = parsed.quantity
                    if parsed.stock is not None:
                        corrections["stock"] = parsed.stock

            if corrections:
                # Apply corrections to temp_changes
                item = pending_item["item"]
                for field, value in corrections.items():
                    if field == "name":
                        item.name = value
                    elif field == "price":
                        item.price = value
                    elif field == "quantity":
                        item.quantity = value
                    elif field == "stock":
                        item.stock = value

                pending_item["temp_changes"] = item.to_payload()
                self._save_bulk_state(session)  # Save state to temp file

                # Show updated item
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                f"Item {item_index} updated!\n\n"
                                f"*Item {item_index}*:\n"
                                f"• Name: {item.name or '?'}\n"
                                f"• Price: ₹{item.price if item.price else '?'} per {item.quantity or 1} unit\n"
                                f"• Stock: {item.stock if item.stock is not None else '?'}\n\n"
                                f"Reply '{item_index}: yes' to confirm, or provide more corrections."
                            ),
                        }
                    ]
                }

            # Couldn't parse
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f" Couldn't understand correction for item {item_index}.\n\n"
                            f"Try:\n"
                            f"• \"{item_index}: yes\" to confirm\n"
                            f"• \"{item_index}: skip\" to skip\n"
                            f"• \"{item_index}: price is 20\"\n"
                            f"• \"{item_index}: stock is 100\""
                        ),
                    }
                ]
            }

    def _save_bulk_state(self, session: SessionState) -> None:
        """Save current bulk state to temp file."""
        self.bulk_temp_storage.save(session.user_id, {
            "bulk_mode": session.bulk_mode,
            "bulk_message_count": session.bulk_message_count,
            "bulk_confirmed_items": session.bulk_confirmed_items,
            "bulk_pending_items": session.bulk_pending_items,
        })

    def _show_parallel_bulk_status(self, session: SessionState) -> Dict[str, Any]:
        """Show status of all parallel bulk items."""
        lines = [
            "*Bulk Processing Status*",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]

        for idx, (item_index, data) in enumerate(session.bulk_pending_items.items(), 1):
            item = data["item"]
            status = data["status"]

            if status == "confirmed":
                icon = "[OK]"
            elif status == "skipped":
                icon = "[SKIP]"
            else:
                icon = "[...]"

            name = item.name or "(no name)"
            price = f"₹{item.price}" if item.price else "?"
            lines.append(f"{icon} {item_index}. {name} - {price}, Stock: {item.stock or '?'}")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━")

        confirmed = len(session.bulk_confirmed_items)
        skipped = len([v for v in session.bulk_pending_items.values() if v["status"] == "skipped"])
        pending = len([v for v in session.bulk_pending_items.values() if v["status"] == "pending"])

        lines.append(f"Confirmed: {confirmed}")
        lines.append(f" Skipped: {skipped}")
        lines.append(f" Pending: {pending}")
        lines.append("")
        lines.append("Reply 'done' to save confirmed items.")

        return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

    def _commit_parallel_bulk(self, session: SessionState) -> Dict[str, Any]:
        """Commit all confirmed items to database."""
        confirmed_items = [
            (idx, data) for idx, data in session.bulk_pending_items.items()
            if data["status"] == "confirmed"
        ]

        if not confirmed_items:
            self.bulk_temp_storage.delete(session.user_id)  # Delete temp file
            session.reset()
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": "No items were confirmed. Nothing saved.\n\nType 'inventory' to see your products.",
                    }
                ]
            }

        successful = []
        failed = []
        mode = session.bulk_mode or "add"

        for item_index, data in confirmed_items:
            item = data["item"]
            payload = data["temp_changes"]

            try:
                # Check for similar products
                if mode == "add" and payload.get("name"):
                    similar = self.intake.find_similar(payload["name"], payload.get("price"), session.user_id)
                    if similar and similar[0][2] >= 0.7:
                        # Update existing
                        self.intake.update_product(similar[0][0], payload)
                        successful.append(f"Updated: {payload['name']}")
                    else:
                        # Add new
                        self.intake.add_product(session.user_id, payload)
                        successful.append(f"Added: {payload['name']}")
                elif mode == "update" and payload.get("name"):
                    similar = self.intake.find_similar(payload["name"], payload.get("price"), session.user_id)
                    if similar and similar[0][2] >= 0.5:
                        self.intake.update_product(similar[0][0], payload)
                        successful.append(f"Updated: {payload['name']}")
                    else:
                        self.intake.add_product(session.user_id, payload)
                        successful.append(f"Added: {payload['name']}")
                else:
                    self.intake.add_product(session.user_id, payload)
                    successful.append(f"Added: {payload['name']}")
            except Exception as e:
                failed.append(f"{payload.get('name', 'Unknown')}: {str(e)}")

        # Delete temp file and reset session
        self.bulk_temp_storage.delete(session.user_id)
        session.reset()

        # Build response
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━",
            "*Bulk Processing Complete*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"Saved: {len(successful)}",
            f"Failed: {len(failed)}",
        ]

        if successful:
            lines.append("")
            lines.append("*Saved:*")
            for s in successful[:5]:
                lines.append(f"  • {s}")
            if len(successful) > 5:
                lines.append(f"  ... and {len(successful) - 5} more")

        if failed:
            lines.append("")
            lines.append("*Failed:*")
            for f in failed[:3]:
                lines.append(f"  • {f}")

        lines.append("")
        lines.append("Type 'inventory' to see your products.")

        return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

    def _process_next_bulk_item(self, session: SessionState, message: str) -> Dict[str, Any]:
        """Process confirmation/correction for current bulk item."""
        lower = message.lower().strip()

        # Check for cancel
        if lower in {'cancel', 'stop', 'abort', 'quit'}:
            completed_count = len(session.bulk_completed)
            session.reset()
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"Bulk processing cancelled.\n"
                            f"{completed_count} item(s) were completed before cancellation.\n\n"
                            "Type 'inventory' to see your products."
                        ),
                    }
                ]
            }

        # Check for skip
        if lower in {'skip', 'next'}:
            session.advance_bulk_queue()
            if session.has_bulk_items():
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": f" Skipped. Moving to next item...\n\n{self._format_bulk_item_confirmation(session)}",
                        }
                    ]
                }
            else:
                return self._complete_bulk_processing(session)

        current_item = session.get_current_bulk_item()
        if not current_item:
            return self._complete_bulk_processing(session)

        # Handle pending confirmation for current bulk item
        if session.pending_confirmation and session.pending_confirmation.get("action") == "verify_bulk_item":
            return self._handle_bulk_item_confirmation(session, message)

        # Check if current item needs missing fields
        missing = current_item.get_missing_required_fields()

        if missing:
            # Try to parse user input as missing field values
            updated = self._fill_missing_bulk_fields(current_item, message)
            if updated:
                missing = current_item.get_missing_required_fields()

        if missing:
            # Still missing fields - ask again
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"Item {session.bulk_current_index + 1}: *{current_item.name or 'Unknown'}*\n\n"
                            f"Still need: {', '.join(missing)}\n\n"
                            f"Please provide: [name] [price] [qty] [unit] [stock]\n"
                            f"Or type 'skip' to skip this item."
                        ),
                    }
                ]
            }

        # Item is complete - ask for confirmation
        session.pending_confirmation = {
            "action": "verify_bulk_item",
            "item_index": session.bulk_current_index,
        }

        return {
            "messages": [
                {
                    "type": "text",
                    "text": self._format_bulk_item_confirmation(session),
                }
            ]
        }

    def _handle_bulk_item_confirmation(self, session: SessionState, message: str) -> Dict[str, Any]:
        """Handle yes/no/correction for bulk item confirmation."""
        lower = message.lower().strip()
        current_item = session.get_current_bulk_item()

        if not current_item:
            session.pending_confirmation = None
            return self._complete_bulk_processing(session)

        # User confirms
        if _is_affirmative(message) or lower in {"done", "good"}:
            # Process this item (add/update)
            payload = current_item.to_payload()
            mode = session.bulk_mode or "add"

            # Check for similar products if adding
            if mode == "add" and payload.get("name"):
                similar = self.intake.find_similar(payload["name"], payload.get("price"), session.user_id)
                if similar and similar[0][2] >= 0.7:
                    # High similarity - switch to update
                    mode = "update"
                    matched_id = similar[0][0]
                else:
                    matched_id = None
            else:
                # For update mode, find matching product
                if payload.get("name"):
                    similar = self.intake.find_similar(payload["name"], payload.get("price"), session.user_id)
                    matched_id = similar[0][0] if similar and similar[0][2] >= 0.5 else None
                else:
                    matched_id = None

            # Process the item
            try:
                if mode == "update" and matched_id:
                    self.intake.update_product(matched_id, payload)
                    result_text = f"Updated: {payload['name']}"
                else:
                    product_id = self.intake.add_product(session.user_id, payload)
                    result_text = f"Added: {payload['name']}"

                session.bulk_completed.append({
                    "name": payload["name"],
                    "mode": mode,
                    "success": True,
                })
            except Exception as e:
                result_text = f"Failed: {payload['name']} - {str(e)}"
                session.bulk_completed.append({
                    "name": payload["name"],
                    "mode": mode,
                    "success": False,
                    "error": str(e),
                })

            # Move to next item
            session.pending_confirmation = None
            session.advance_bulk_queue()

            if session.has_bulk_items():
                next_prompt = self._format_bulk_item_confirmation(session)
                remaining = len(session.bulk_queue) - session.bulk_current_index
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": f"{result_text}\n\n{remaining} item(s) remaining\n\n{next_prompt}",
                        }
                    ]
                }
            else:
                return self._complete_bulk_processing(session, last_result=result_text)

        # User provides correction
        corrections = self._parse_bulk_item_correction(message)

        if corrections:
            # Apply corrections to current item
            if "name" in corrections:
                current_item.name = corrections["name"]
            if "price" in corrections:
                current_item.price = corrections["price"]
            if "quantity" in corrections:
                current_item.quantity = corrections["quantity"]
            if "stock" in corrections:
                current_item.stock = corrections["stock"]

            # Update queue with corrected item
            session.bulk_queue[session.bulk_current_index] = current_item

            # Ask for confirmation again
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"Updated values!\n\n"
                            f"{self._format_bulk_item_confirmation(session)}"
                        ),
                    }
                ]
            }

        # Couldn't parse - show current state and ask again
        return {
            "messages": [
                {
                    "type": "text",
                    "text": (
                        f" I didn't understand that.\n\n"
                        f"{self._format_bulk_item_confirmation(session)}\n\n"
                        f"To correct, use format:\n"
                        f"• \"price is 20\"\n"
                        f"• \"stock is 100\"\n"
                        f"• \"name is New Name\""
                    ),
                }
            ]
        }

    def _format_bulk_item_confirmation(self, session: SessionState) -> str:
        """Format confirmation prompt for current bulk item."""
        current_item = session.get_current_bulk_item()
        if not current_item:
            return "No more items to process."

        idx = session.bulk_current_index + 1
        total = len(session.bulk_queue)
        mode = session.bulk_mode or "add"

        lines = [
            f"*Item {idx}/{total}* ({mode.upper()})",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"• Name: {current_item.name or '?'}",
            f"• Price: ₹{current_item.price if current_item.price is not None else '?'} per {current_item.quantity or 1} unit",
            f"• Stock: {current_item.stock if current_item.stock is not None else '?'}",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]

        missing = current_item.get_missing_required_fields()
        if missing:
            lines.append(f"Warning: Missing: {', '.join(missing)}")
            lines.append(f"Please provide: {', '.join(missing)}")
        else:
            lines.append("")
            lines.append("Is this correct?")
            lines.append("• 'yes' to confirm")
            lines.append("• 'skip' to skip this item")
            lines.append("• Or tell me the correction (e.g., 'price is 25')")

        return "\n".join(lines)

    def _fill_missing_bulk_fields(self, item: ProductItem, message: str) -> bool:
        """Try to fill missing fields from user message. Returns True if any field was filled."""
        filled = False

        # Try to parse as "name price qty stock" format
        parsed = BulkProductInput._parse_single_line(message)
        if parsed:
            if not item.name and parsed.name:
                item.name = parsed.name
                filled = True
            if item.price is None and parsed.price is not None:
                item.price = parsed.price
                filled = True
            if item.quantity is None and parsed.quantity is not None:
                item.quantity = parsed.quantity
                filled = True
            if item.stock is None and parsed.stock is not None:
                item.stock = parsed.stock
                filled = True

        # Also try correction patterns
        corrections = self._parse_bulk_item_correction(message)
        for field, value in corrections.items():
            if field == "name" and not item.name:
                item.name = value
                filled = True
            elif field == "price" and item.price is None:
                item.price = value
                filled = True
            elif field == "quantity" and item.quantity is None:
                item.quantity = value
                filled = True
            elif field == "stock" and item.stock is None:
                item.stock = value
                filled = True

        return filled

    def _parse_bulk_item_correction(self, message: str) -> Dict[str, Any]:
        """Parse correction messages for bulk items."""
        corrections = {}
        lower = message.lower()

        # Name correction
        name_match = re.search(r"name\s*(?:is|=|:)\s*(.+?)(?:\s*[,\n]|$)", message, re.IGNORECASE)
        if name_match:
            corrections["name"] = name_match.group(1).strip()

        # Price correction
        price_match = re.search(r"price\s*(?:is|=|:)?\s*(?:rs\.?|₹)?\s*(\d+(?:\.\d+)?)", lower)
        if price_match:
            corrections["price"] = float(price_match.group(1))

        # Stock correction
        stock_match = re.search(r"stock\s*(?:is|=|:)?\s*(\d+)", lower)
        if stock_match:
            corrections["stock"] = int(stock_match.group(1))

        # Quantity correction
        qty_match = re.search(r"(?:quantity|qty)\s*(?:is|=|:)?\s*(\d+)", lower)
        if qty_match:
            corrections["quantity"] = int(qty_match.group(1))

        return corrections

    def _complete_bulk_processing(self, session: SessionState, last_result: str = "") -> Dict[str, Any]:
        """Complete bulk processing and show summary."""
        completed = session.bulk_completed
        successful = [c for c in completed if c.get("success")]
        failed = [c for c in completed if not c.get("success")]

        lines = []
        if last_result:
            lines.append(last_result)
            lines.append("")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("*Bulk Processing Complete*")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"Successful: {len(successful)}")
        lines.append(f"Failed: {len(failed)}")

        if successful:
            lines.append("")
            lines.append("*Added/Updated:*")
            for item in successful[:5]:  # Show max 5
                lines.append(f"  • {item['name']}")
            if len(successful) > 5:
                lines.append(f"  ... and {len(successful) - 5} more")

        if failed:
            lines.append("")
            lines.append("*Failed:*")
            for item in failed[:3]:
                lines.append(f"  • {item['name']}: {item.get('error', 'Unknown error')}")

        lines.append("")
        lines.append("Type 'inventory' to see your products.")

        # Reset bulk state
        session.bulk_queue = []
        session.bulk_current_index = 0
        session.bulk_mode = None
        session.bulk_completed = []
        session.pending_confirmation = None

        return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

    def _extract_payload(
        self, message: str, attachments: List[Dict[str, Any]], session: SessionState
    ) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        """
        Extract product payload(s) from message.

        Returns:
            Tuple of (list of payloads, list of missing fields per payload)
            For single product: ([payload], [[missing_fields]])
            For multiple products: ([p1, p2, ...], [[missing1], [missing2], ...])
        """
        text = _clean_text(message)
        llm_parsed = None

        # Try LLM parsing for complex/conversational inputs
        if _should_use_llm_parser(text):
            llm_parser = get_vendor_llm_parser()
            if llm_parser:
                llm_parsed = llm_parser.parse(text)  # Now returns List[Dict]
                if llm_parsed:
                    print(f"LLM parsed {len(llm_parsed)} product(s)")

        # If LLM parsed successfully, use its data (now returns list of products)
        if llm_parsed and isinstance(llm_parsed, list) and len(llm_parsed) > 0:
            all_payloads = []
            all_missing = []

            for product in llm_parsed:
                payload = {
                    "name": product.get("name"),
                    "price": product.get("price"),
                    "quantity": product.get("quantity", 1),
                    "quantityunit": product.get("quantityunit") or "unit",
                    "stock": product.get("stock"),
                    "attachments": [att.get("type", "image") for att in attachments],
                    "raw": text,
                    "parsed_by_llm": True,
                }

                # Add optional fields from LLM parsing
                for opt_field in ["brand", "colour", "size", "dimensions", "category", "subcategory", "description", "rating", "other_properties"]:
                    if product.get(opt_field):
                        payload[opt_field] = product[opt_field]

                # Check missing fields for this product
                missing = self._check_missing_fields(payload)
                all_payloads.append(payload)
                all_missing.append(missing)

            return all_payloads, all_missing

        # Fall back to rule-based parsing (single product)
        parsed = _parse_space_separated_input(text, session.mode)

        # Fall back to keyword-based extraction if smart parsing didn't get everything
        if parsed["name"] is None:
            parsed["name"] = _extract_name(text.split("\n")[0] if text else "", session.mode)
        if parsed["price"] is None:
            parsed["price"] = _extract_price(text)
        if parsed["quantity"] is None and parsed["stock"] is None:
            # Try to extract at least one
            extracted_int = _extract_integer(text)
            if extracted_int is not None:
                parsed["quantity"] = extracted_int

        # If we're awaiting missing fields and already have a name, don't overwrite it
        # unless the new input looks like a correction (has price/stock keywords too)
        existing_name = session.last_payload.get("name")
        if existing_name and session.awaiting_missing_fields:
            # Check if the input is just providing missing fields (price, stock, etc.)
            # rather than a completely new product
            text_lower = text.lower()
            has_field_keywords = any(kw in text_lower for kw in ["price", "stock", "rs", "rupee", "₹"])
            has_only_simple_word = len(text.split()) <= 2 and not has_field_keywords

            # If it's just a simple word like "hi", "exit", etc., keep existing name
            if has_only_simple_word and parsed["price"] is None and parsed["stock"] is None:
                parsed["name"] = None  # Will fall back to existing name

        payload = {
            "name": parsed["name"] or session.last_payload.get("name"),
            "price": parsed["price"] if parsed["price"] is not None else session.last_payload.get("price"),
            "quantity": parsed["quantity"] if parsed["quantity"] is not None else session.last_payload.get("quantity"),
            "quantityunit": parsed.get("quantityunit") or session.last_payload.get("quantityunit"),
            "stock": parsed["stock"] if parsed["stock"] is not None else session.last_payload.get("stock"),
            "attachments": [att.get("type", "image") for att in attachments],
            "raw": text,
            "parsed_pattern": parsed.get("parsed_pattern"),
            "image_path": session.last_payload.get("image_path") or session.pending_image_path,  # Preserve image from previous message
        }

        # Clear pending image path after it's been used
        if session.pending_image_path and payload.get("image_path"):
            session.pending_image_path = None

        # Check for missing required fields
        missing = self._check_missing_fields(payload)

        # Return as single-item lists for consistent interface
        return [payload], [missing]

    def _check_missing_fields(self, payload: Dict[str, Any]) -> List[str]:
        """Check for missing required fields in a single payload.

        Only name, price, and stock are truly required.
        Quantity defaults to 1 and unit defaults to 'unit' if not provided.
        """
        missing = []
        if not payload.get("name"):
            missing.append("name")
        if payload.get("price") is None:
            missing.append("price")
        if payload.get("stock") is None:
            missing.append("stock")
        # quantity and quantityunit are optional - will default to 1 unit
        return missing

    def _handle_missing_fields(
        self, session: SessionState, missing_fields: List[str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        session.awaiting_missing_fields = True
        session.missing_fields = missing_fields
        session.retry_count += 1
        session.last_payload.update(payload)

        # Build smart clarification message based on what we have
        has_partial_data = payload.get("name") or payload.get("price") is not None

        if has_partial_data:
            # We have some data - ask for clarification on what's missing
            clarification_parts = []

            if payload.get("name"):
                clarification_parts.append(f"Product: {payload['name']}")
            if payload.get("price") is not None:
                clarification_parts.append(f"Price: ₹{payload['price']}")
            if payload.get("quantity") is not None:
                unit = payload.get('quantityunit', 'unit') or 'unit'
                clarification_parts.append(f"Quantity: {payload['quantity']} {unit}")
            if payload.get("quantityunit") and payload.get("quantity") is None:
                clarification_parts.append(f"Unit: {payload['quantityunit']}")
            if payload.get("stock") is not None:
                clarification_parts.append(f"Stock: {payload['stock']}")

            # What's still needed
            still_needed = []
            if "name" in missing_fields:
                still_needed.append("product name")
            if "price" in missing_fields:
                still_needed.append("price")
            if "quantity" in missing_fields:
                still_needed.append("quantity")
            if "quantityunit" in missing_fields:
                still_needed.append("unit (e.g., kg, pack, unit, pcs)")
            if "stock" in missing_fields:
                still_needed.append("stock")

            if clarification_parts:
                understood = "I understood:\n• " + "\n• ".join(clarification_parts)
            else:
                understood = ""

            if still_needed:
                needed_msg = f"\n\nWarning: *Still need:* {', '.join(still_needed)}."
            else:
                needed_msg = ""

            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"{understood}{needed_msg}\n\n"
                            "Please provide the missing info, or send in format:\n"
                            "[name] [price] [qty] [unit] [stock]\n"
                            "Example: Maggi 10 1 pack 50"
                        ),
                    }
                ]
            }

        # No partial data - ask for everything
        if session.retry_count >= 2:
            session.awaiting_missing_fields = False
            session.retry_count = 0
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            "I couldn't parse that. Please send in this format:\n"
                            "[name] [price] [qty] [unit] [stock]\n"
                            "Example: Maggi 10 1 pack 50\n\n"
                            "Or send with labels:\n"
                            "Name: Maggi\nPrice: 10\nQty: 1\nUnit: pack\nStock: 50"
                        ),
                    }
                ]
            }

        return {
            "messages": [
                {
                    "type": "text",
                    "text": (
                        "Please share product details in format:\n"
                        "[name] [price] [qty] [unit] [stock]\n"
                        "Example: Maggi 10 1 pack 50"
                    ),
                }
            ]
        }

    def _format_inferred_confirmation(
        self, payload: Dict[str, Any], similar: List[Tuple[str, str, float, Optional[float]]], mode: str
    ) -> Dict[str, Any]:
        """Format a confirmation message for inferred/parsed values with similarity suggestions."""
        lines = ["I understood this:"]
        lines.append(f"• Product: {payload.get('name', 'Unknown')}")
        unit = payload.get('quantityunit', 'unit') or 'unit'
        if payload.get("price") is not None and payload.get("quantity") is not None:
            lines.append(f"• Price: ₹{payload['price']} per {payload['quantity']} {unit}")
        elif payload.get("price") is not None:
            lines.append(f"• Price: ₹{payload['price']}")
        if payload.get("stock") is not None:
            lines.append(f"• Stock: {payload['stock']}")

        lines.append("")

        # Check for similar products
        if similar:
            # Filter to reasonably similar items (> 0.3 similarity)
            # Handle both 3-tuple and 4-tuple formats
            relevant_similar = []
            for item in similar:
                if len(item) >= 3 and item[2] > 0.3:
                    relevant_similar.append(item)

            if relevant_similar:
                top_match = relevant_similar[0]
                match_price = top_match[3] if len(top_match) > 3 else None
                match_score = top_match[2]
                if match_score >= 0.7:
                    # High similarity - likely the same product
                    lines.append(f"Found similar product in inventory:")
                    price_str = f" @ ₹{match_price}" if match_price else ""
                    lines.append(f"   *{top_match[1]}*{price_str}")
                    lines.append(f"   Match: {match_score:.0%} (name{' + price' if match_price else ''} similarity)")
                    lines.append("")
                    if mode == "add":
                        lines.append("Options:")
                        lines.append("• Reply 'yes' to confirm & ADD as NEW product")
                        lines.append("• Reply '1' to UPDATE existing instead")
                        lines.append("• Reply 'no' to ADD without updating existing")
                        lines.append("• Reply 'cancel' to start over")
                    else:
                        lines.append("Options:")
                        lines.append("• Reply 'yes' or '1' to update this product")
                        lines.append("• Reply 'no' to cancel and re-enter")
                elif len(relevant_similar) > 0:
                    # Some similarity - show options
                    lines.append("Similar products in inventory:")
                    for idx, item in enumerate(relevant_similar[:3], start=1):
                        item_price = item[3] if len(item) > 3 else None
                        item_score = item[2]
                        price_str = f" @ ₹{item_price}" if item_price else ""
                        lines.append(f"   {idx}. *{item[1]}*{price_str}")
                        lines.append(f"      Match: {item_score:.0%}")
                    lines.append("")
                    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
                    lines.append("ℹ️ Similarity = 70% name + 30% price")
                    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
                    lines.append("")
                    if mode == "add":
                        lines.append("Options:")
                        lines.append("• Reply 'yes' or 'no' to ADD as new product")
                        lines.append("• Reply number (1-3) to UPDATE existing instead")
                        lines.append("• Reply 'cancel' to start over")
                    else:
                        lines.append("Options:")
                        lines.append("• Reply 'yes' to proceed with details above")
                        lines.append("• Reply number (1-3) to use existing product")
                        lines.append("• Reply 'no' to cancel")
            else:
                # No similar products found
                lines.append("This looks like a new product!")
                lines.append("")
                lines.append("Reply 'yes' to add, or 'no' to re-enter details.")
        else:
            lines.append("Reply 'yes' to confirm, or 'no' to re-enter.")

        return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

    def _format_similar_prompt(self, matches: List[Tuple[str, str, float, Optional[float]]], addition: bool) -> Dict[str, Any]:
        intro = "Found something very similar in your inventory." if addition else "Is this the item you want to update?"
        lines = [intro, ""]
        for idx, item in enumerate(matches, start=1):
            pid, name, score = item[0], item[1], item[2]
            price = item[3] if len(item) > 3 else None
            price_str = f" @ ₹{price}" if price else ""
            lines.append(f"{idx}. *{name}*{price_str}")
            lines.append(f"   Match: {score:.0%} (based on name similarity{' + price' if price else ''})")
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("ℹ️ *How similarity works:*")
        lines.append("• Name matching: 70% weight")
        lines.append("• Price matching: 30% weight (if provided)")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        lines.append("Reply with the number to use it, or say 'add new' to continue with your version.")
        return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

    def _handle_confirmation(self, session: SessionState, message: str) -> Dict[str, Any]:
        pending = session.pending_confirmation or {}
        lower = message.lower().strip()
        matches = pending.get("matches", [])
        action = pending.get("action", "")

        # Handle inferred confirmation (from space-separated input parsing)
        if action in {"inferred_add", "inferred_update"}:
            # User confirms with 'yes', 'y', 'ok', 'confirm' (with typo tolerance)
            if _is_affirmative(message):
                payload = pending.get("payload", {})

                # For 'add' mode, do a final similarity check before adding
                if action == "inferred_add" and payload.get("name"):
                    similar = self.intake.find_similar(payload["name"], payload.get("price"), session.user_id)
                    # Check for high similarity (> 0.7 = likely same product)
                    if similar and similar[0][2] >= 0.7:
                        # Ask user if they meant to update instead
                        session.pending_confirmation = {
                            "action": "duplicate_check",
                            "matches": similar,
                            "payload": payload,
                        }
                        top_match = similar[0]
                        return {
                            "messages": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Warning: Found very similar product in inventory:\n"
                                        f"   \"{top_match[1]}\" (ID: {top_match[0]})\n\n"
                                        f"Did you mean to UPDATE this existing product?\n"
                                        f"• Reply 'yes' to UPDATE the existing product\n"
                                        f"• Reply 'no' to ADD as a new product anyway"
                                    ),
                                }
                            ]
                        }

                session.pending_confirmation = None
                return self._complete_intake(session, payload, matched_product_id=None)

            # User says 'no' - behavior depends on mode
            if _is_negative(message):
                payload = pending.get("payload", {})

                # For ADD mode: 'no' means "don't update existing, just add my new product"
                if action == "inferred_add":
                    session.pending_confirmation = None
                    return self._complete_intake(session, payload, matched_product_id=None)

                # For UPDATE mode: 'no' means cancel and re-enter
                else:
                    session.pending_confirmation = None
                    session.last_payload = {}
                    return {
                        "messages": [
                            {
                                "type": "text",
                                "text": (
                                    "No problem! Let's start over.\n"
                                    "Send product details in format:\n"
                                    "[product name] [price] [quantity] [stock]\n"
                                    "Example: Maggi 10 1 pack 50"
                                ),
                            }
                        ]
                    }

            # User explicitly cancels with 'cancel' or 'reset'
            if lower in {"cancel", "reset"}:
                session.pending_confirmation = None
                session.last_payload = {}
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                "No problem! Let's start over.\n"
                                "Send product details in format:\n"
                                "[product name] [price] [quantity] [stock]\n"
                                "Example: Maggi 10 1 pack 50"
                            ),
                        }
                    ]
                }

            # User selects existing product by number
            if matches and lower.isdigit():
                idx = int(lower) - 1
                # Handle 4-element tuples (pid, name, score, price)
                relevant_matches = [m for m in matches if len(m) >= 3 and m[2] > 0.3]
                if 0 <= idx < len(relevant_matches):
                    chosen_id = relevant_matches[idx][0]
                    session.pending_confirmation = None
                    session.mode = "update"  # Switch to update mode for existing product
                    return self._complete_intake(session, pending.get("payload", {}), matched_product_id=chosen_id)

            # Didn't understand - re-prompt
            return self._format_inferred_confirmation(
                pending.get("payload", {}),
                matches,
                "add" if action == "inferred_add" else "update"
            )

        # Handle verification after quick change command
        if action == "verify_change":
            product_id = pending.get("product_id")
            product_name = pending.get("product_name", "product")
            short_id = pending.get("short_id", "")
            changed_field = pending.get("field", "")

            # User confirms it's correct
            if lower in {"yes", "y", "ok", "correct", "right", "confirmed", "done", "good", "perfect"}:
                session.pending_confirmation = None
                session.db_snapshot = None  # Clear snapshot - confirmed, no rollback needed
                session.locked_until = _now() + self.lock_seconds
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                f"Great! *{product_name}* ({short_id}) is confirmed.\n\n"
                                f"What's next?\n"
                                f"• Type 'inventory' to see all products\n"
                                f"• change {short_id} [field] to [value]"
                            ),
                        }
                    ]
                }

            # Try to parse correction for the same or different field
            corrections = self._parse_correction(message)

            # Also try to parse simple "[field] is [value]" pattern for any field
            simple_patterns = [
                r"(price|stock|quantity|qty|name|brand|category)\s*(?:is|=|:)\s*(.+)",
            ]
            for pattern in simple_patterns:
                match = re.search(pattern, lower)
                if match:
                    field_name = match.group(1).strip()
                    field_value = match.group(2).strip()
                    if field_name == "qty":
                        field_name = "quantity"
                    # Convert numeric values
                    if field_name in {"price", "stock", "quantity"}:
                        try:
                            if field_name == "price":
                                corrections[field_name] = float(re.search(r"(\d+(?:\.\d+)?)", field_value).group(1))
                            else:
                                corrections[field_name] = int(re.search(r"(\d+)", field_value).group(1))
                        except (ValueError, AttributeError):
                            pass
                    else:
                        corrections[field_name] = field_value
                    break

            if corrections:
                # Apply corrections
                conn = self.sql_client._get_connection()
                cur = conn.cursor()

                updates = []
                values = []
                correction_msgs = []

                field_to_column = {
                    "stock": "stock",
                    "price": "price",
                    "quantity": "quantity",
                    "name": "prod_name",
                    "brand": "brand",
                    "category": "category",
                }

                for field, value in corrections.items():
                    col = field_to_column.get(field)
                    if col:
                        updates.append(f"{col} = ?")
                        values.append(value)
                        if field == "price":
                            correction_msgs.append(f"Price → ₹{value}")
                        else:
                            correction_msgs.append(f"{field.capitalize()} → {value}")

                if updates:
                    values.append(product_id)
                    query = f"UPDATE product_table SET {', '.join(updates)} WHERE product_id = ?"
                    cur.execute(query, values)
                    conn.commit()

                    # Get updated state
                    new_state = self._get_product_state(product_id)

                    # Keep pending_confirmation active - loop back and ask for confirmation again
                    session.pending_confirmation = {
                        "action": "verify_change",
                        "product_id": product_id,
                        "product_name": new_state.get("name") if new_state else product_name,
                        "short_id": short_id,
                        "field": changed_field,
                    }

                    lines = [
                        f"Corrected! Updated: {', '.join(correction_msgs)}",
                        "",
                        f"Current state of *{new_state.get('name') if new_state else product_name}* ({short_id}):",
                        f"   • Price: ₹{new_state.get('price')} per {new_state.get('quantity')} unit",
                        f"   • Stock: {new_state.get('stock')} units",
                        "",
                        "Is this correct now? Reply 'yes' to confirm.",
                        "",
                        "If not, tell me the correction:",
                        "• \"stock is 20\"",
                        "• \"price is 15\"",
                    ]

                    return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

            # Couldn't parse correction - ask again
            current_state = self._get_product_state(product_id)
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f" I didn't understand that.\n\n"
                            f"Current state of *{product_name}* ({short_id}):\n"
                            f"• Price: ₹{current_state.get('price') if current_state else '?'}\n"
                            f"• Stock: {current_state.get('stock') if current_state else '?'}\n\n"
                            "Reply 'yes' if correct, or tell me the correction:\n"
                            "• \"stock is 20\"\n"
                            "• \"price is 15\""
                        ),
                    }
                ]
            }

        # Handle final verification (after add/update, verify the result)
        if action == "verify_final":
            product_id = pending.get("product_id")
            product_name = pending.get("product_name", "product")

            # User confirms it's correct
            if lower in {"yes", "y", "ok", "correct", "right", "confirmed", "done", "good", "perfect"}:
                session.pending_confirmation = None
                session.db_snapshot = None  # Clear snapshot - confirmed, no rollback needed
                session.locked_until = _now() + self.lock_seconds
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                f"Great! \"{product_name}\" is confirmed.\n\n"
                                "Ready for next item! Send in format:\n"
                                "add/update [name] [price] [qty] [unit] [stock]\n\n"
                                "Example: add Chips 20 1 pack 100"
                            ),
                        }
                    ]
                }

            # User says no or provides correction
            corrections = self._parse_correction(message)

            if corrections:
                # Apply corrections directly (not cumulative - these are absolute values)
                conn = self.sql_client._get_connection()
                cur = conn.cursor()

                updates = []
                values = []
                correction_msgs = []

                if "stock" in corrections:
                    updates.append("stock = ?")
                    values.append(corrections["stock"])
                    correction_msgs.append(f"Stock → {corrections['stock']}")

                if "price" in corrections:
                    updates.append("price = ?")
                    values.append(corrections["price"])
                    correction_msgs.append(f"Price → ₹{corrections['price']}")

                if "quantity" in corrections:
                    updates.append("quantity = ?")
                    values.append(corrections["quantity"])
                    correction_msgs.append(f"Quantity → {corrections['quantity']}")

                if updates:
                    values.append(product_id)
                    query = f"UPDATE product_table SET {', '.join(updates)} WHERE product_id = ?"
                    cur.execute(query, values)
                    conn.commit()

                    # Get updated state
                    new_state = self._get_product_state(product_id)

                    # Keep pending_confirmation active - loop back and ask for confirmation again
                    # This creates a confirmation loop until user says "yes" or timeout
                    session.pending_confirmation = {
                        "action": "verify_final",
                        "product_id": product_id,
                        "product_name": new_state.get("name") if new_state else product_name,
                    }

                    lines = [
                        f"Corrected! Updated: {', '.join(correction_msgs)}",
                        "",
                        f"New state of \"{new_state.get('name') if new_state else product_name}\":",
                        f"   • Price: ₹{new_state.get('price')} per {new_state.get('quantity')} unit",
                        f"   • Stock: {new_state.get('stock')} units",
                        "",
                        "Is this correct now? Reply 'yes' to confirm.",
                        "",
                        "If not, tell me the correction:",
                        "• \"stock is 20\"",
                        "• \"price is 15\"",
                    ]

                    return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

            # Couldn't parse correction - ask again
            current_state = self._get_product_state(product_id)
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f" I didn't understand that.\n\n"
                            f"Current state:\n"
                            f"• Price: ₹{current_state.get('price') if current_state else '?'}\n"
                            f"• Stock: {current_state.get('stock') if current_state else '?'}\n\n"
                            "Reply 'yes' if correct, or use this format:\n"
                            "━━━━━━━━━━━━━━━━━━━━━━\n"
                            "• \"stock is 20\"\n"
                            "• \"price is 15\"\n"
                            "━━━━━━━━━━━━━━━━━━━━━━"
                        ),
                    }
                ]
            }

        # Handle duplicate check (when user confirms 'yes' but similar product found)
        if action == "duplicate_check":
            payload = pending.get("payload", {})
            matches = pending.get("matches", [])

            # User says 'yes' - update the existing product
            if lower in {"yes", "y", "ok", "confirm", "yep", "yeah", "sure", "update"}:
                if matches:
                    matched_id = matches[0][0]  # Use the top match
                    session.pending_confirmation = None
                    session.mode = "update"
                    return self._complete_intake(session, payload, matched_product_id=matched_id)

            # User says 'no' - add as new product anyway
            if lower in {"no", "n", "add", "new", "add new"}:
                session.pending_confirmation = None
                session.mode = "add"
                return self._complete_intake(session, payload, matched_product_id=None)

            # Didn't understand - re-prompt
            top_match = matches[0] if matches else ("", "Unknown", 0)
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"Please reply:\n"
                            f"• 'yes' to UPDATE \"{top_match[1]}\"\n"
                            f"• 'no' to ADD as new product"
                        ),
                    }
                ]
            }

        # Handle product deletion confirmation
        if action == "confirm_delete":
            product_id = pending.get("product_id")
            product_name = pending.get("product_name", "product")
            short_id = pending.get("short_id", "")

            # User confirms deletion
            if lower in {"yes", "y", "ok", "confirm", "delete", "remove"}:
                try:
                    conn = self.sql_client._get_connection()
                    cur = conn.cursor()
                    cur.execute("DELETE FROM product_table WHERE product_id = ?", (product_id,))
                    conn.commit()

                    session.pending_confirmation = None
                    return {
                        "messages": [
                            {
                                "type": "text",
                                "text": (
                                    f"*{product_name}* ({short_id}) has been deleted.\n\n"
                                    f"Type 'inventory' to see your remaining products."
                                ),
                            }
                        ]
                    }
                except Exception as e:
                    print(f"Delete error: {e}")
                    session.pending_confirmation = None
                    return {
                        "messages": [
                            {"type": "text", "text": f"Failed to delete product: {e}"}
                        ]
                    }

            # User cancels deletion
            if lower in {"no", "n", "cancel", "abort", "stop", "keep"}:
                session.pending_confirmation = None
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                f"Cancelled. *{product_name}* ({short_id}) was NOT deleted.\n\n"
                                f"Type 'inventory' to see your products."
                            ),
                        }
                    ]
                }

            # Didn't understand - re-prompt
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            f"Please reply:\n"
                            f"• 'yes' to DELETE *{product_name}* ({short_id})\n"
                            f"• 'no' to KEEP the product"
                        ),
                    }
                ]
            }

        # Handle multi-product add confirmation
        if action == "multi_add":
            payloads = pending.get("payloads", [])

            # User confirms with 'yes' - add all products
            if lower in {"yes", "y", "ok", "confirm", "yep", "yeah", "sure", "add all"}:
                return self._complete_multi_intake(session, payloads)

            # User cancels
            if lower in {"cancel", "no", "n", "abort", "stop"}:
                session.pending_confirmation = None
                session.last_payload = {}
                return {
                    "messages": [
                        {
                            "type": "text",
                            "text": (
                                "Cancelled. No products were added.\n\n"
                                "Ready for next action! Send product details or say 'add' or 'update'."
                            ),
                        }
                    ]
                }

            # Try to parse per-product corrections like "Product 1 price is 500"
            product_correction_pattern = r"product\s*(\d+)\s+(price|stock|quantity|name)\s*(?:is|=|:)?\s*(.+)"
            correction_match = re.search(product_correction_pattern, lower)

            if correction_match:
                product_idx = int(correction_match.group(1)) - 1
                field = correction_match.group(2).strip()
                value = correction_match.group(3).strip()

                if 0 <= product_idx < len(payloads):
                    # Parse value based on field type
                    if field in {"price", "stock", "quantity"}:
                        try:
                            num_match = re.search(r"(\d+(?:\.\d+)?)", value)
                            if num_match:
                                if field == "price":
                                    payloads[product_idx][field] = float(num_match.group(1))
                                else:
                                    payloads[product_idx][field] = int(num_match.group(1))
                        except ValueError:
                            pass
                    else:
                        payloads[product_idx][field] = value

                    # Update pending confirmation with corrected payloads
                    session.pending_confirmation = {
                        "action": "multi_add",
                        "payloads": payloads,
                    }

                    # Re-show preview with updated values
                    return self._format_preview_multi(payloads)

            # Didn't understand - re-prompt
            return {
                "messages": [
                    {
                        "type": "text",
                        "text": (
                            " I didn't understand that.\n\n"
                            "Reply 'yes' to add all products, or specify corrections like:\n"
                            "• \"Product 1 price is 500\"\n"
                            "• \"Product 2 stock is 100\"\n"
                            "• \"cancel\" to abort"
                        ),
                    }
                ]
            }

        # Original confirmation handling for similar_add, update_match
        if lower in {"add new", "add as new", "new"}:
            session.pending_confirmation = None
            return self._complete_intake(session, pending.get("payload", {}), matched_product_id=None)

        # Accept number or product_id
        chosen_id = None
        if matches:
            if lower.isdigit():
                idx = int(lower) - 1
                if 0 <= idx < len(matches):
                    chosen_id = matches[idx][0]
            else:
                for pid, _name, _score in [(m[0], m[1], m[2]) for m in matches]:
                    if pid.lower() in lower:
                        chosen_id = pid
                        break

        if chosen_id:
            session.pending_confirmation = None
            # Switch mode to update if user selected an existing item during addition
            if pending.get("action") == "similar_add":
                session.mode = "update"
            return self._complete_intake(session, pending.get("payload", {}), matched_product_id=chosen_id)

        # Fallback prompt
        return self._format_similar_prompt(matches, addition=pending.get("action") == "similar_add")

    def _complete_intake(
        self, session: SessionState, payload: Dict[str, Any], matched_product_id: Optional[str]
    ) -> Dict[str, Any]:
        # Save snapshot BEFORE making changes for rollback on timeout
        is_update = matched_product_id is not None
        if is_update:
            # Updating existing product - save its current state
            old_state = self._get_product_state(matched_product_id)
            session.db_snapshot = {
                "product_id": matched_product_id,
                "was_new_product": False,
                "price": old_state.get("price") if old_state else None,
                "quantity": old_state.get("quantity") if old_state else None,
                "stock": old_state.get("stock") if old_state else None,
            }
        else:
            # Adding new product - will save product_id after creation
            session.db_snapshot = {"was_new_product": True}

        start = _now()
        product_id = self.intake.enqueue_intake(
            user_id=session.user_id,
            mode=session.mode or "add",
            payload=payload,
            matched_product_id=matched_product_id,
        )
        ingestion_time = _now() - start

        # For new products, save the product_id in snapshot
        if not is_update and product_id:
            session.db_snapshot["product_id"] = product_id

        session.awaiting_missing_fields = False
        session.retry_count = 0

        # Get current product state from database for verification
        final_product_id = matched_product_id or product_id
        current_state = self._get_product_state(final_product_id)

        if current_state:
            # Ask vendor to verify the final state
            session.pending_confirmation = {
                "action": "verify_final",
                "product_id": final_product_id,
                "product_name": current_state.get("name"),
            }

            lines = [
                f"{'Updated' if matched_product_id else 'Added'} successfully!",
                "",
                f"Current state of \"{current_state.get('name')}\":",
                f"   • Price: ₹{current_state.get('price')} per {current_state.get('quantity')} unit",
                f"   • Stock: {current_state.get('stock')} units",
                "",
                "Is this correct? Reply 'yes' to confirm.",
                "",
                "If not, tell me the correction in this format:",
                "━━━━━━━━━━━━━━━━━━━━━━",
                "• \"stock is 20\" or \"total stock 20\"",
                "• \"price is 15\" or \"price should be 15\"",
                "• \"quantity is 2\" or \"per 2 units\"",
                "━━━━━━━━━━━━━━━━━━━━━━",
            ]

            return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

        # Fallback if we can't get product state
        session.pending_confirmation = None
        session.locked_until = _now() + self.lock_seconds
        return {
            "messages": [
                {"type": "text", "text": f"{'Updated' if matched_product_id else 'Added'} successfully!"},
                {"type": "text", "text": "Choose next action: add or update."},
            ]
        }

    def _get_product_state(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a product from database."""
        if not product_id:
            return None
        try:
            conn = self.sql_client._get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT prod_name, price, quantity, stock FROM product_table WHERE product_id = ?",
                (product_id,)
            )
            row = cur.fetchone()
            if row:
                return {
                    "name": row["prod_name"],
                    "price": row["price"],
                    "quantity": row["quantity"] or 1,
                    "stock": row["stock"] or 0,
                }
        except Exception as e:
            print(f"Error getting product state: {e}")
        return None

    # ==================== Multi-Product Handling ====================

    def _handle_multi_product_intake(
        self,
        session: SessionState,
        payloads: List[Dict[str, Any]],
        all_missing: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Handle intake flow for multiple products - sends SEPARATE message per product
        with similarity checking for each.
        """
        # Initialize multi-product tracking state
        session.multi_product_mode = True
        session.multi_product_payloads = payloads
        session.multi_product_missing = all_missing
        session.multi_product_statuses = ["pending"] * len(payloads)
        session.multi_product_similar = []

        messages = []

        # Summary message first
        messages.append({
            "type": "text",
            "text": f"*Found {len(payloads)} product(s) to add*\n\nSending each separately for review...",
            "requires_reply_tracking": False
        })

        # Per-product messages with similarity check
        for idx, (payload, missing) in enumerate(zip(payloads, all_missing)):
            # Check for similar products in this vendor's inventory
            similar = []
            if payload.get("name"):
                similar = self.intake.find_similar(
                    payload["name"],
                    payload.get("price"),
                    session.user_id
                )
            session.multi_product_similar.append(similar)

            # Format per-product message
            product_msg = self._format_single_product_message(
                idx=idx,
                total=len(payloads),
                payload=payload,
                missing=missing,
                similar=similar
            )

            messages.append({
                "type": "text",
                "text": product_msg,
                "product_index": idx,
                "requires_reply_tracking": True
            })

        # Final instruction message
        messages.append({
            "type": "text",
            "text": "━━━━━━━━━━━━━━━━━━━━━━\nReply in order: 'yes', 'skip', or corrections.\nType 'save' when done, 'cancel' to abort.",
            "requires_reply_tracking": False
        })

        return {"messages": messages, "multi_product_tracking": True}

    def _format_single_product_message(
        self,
        idx: int,
        total: int,
        payload: Dict[str, Any],
        missing: List[str],
        similar: List[Tuple]
    ) -> str:
        """Format a single product message with similarity info and missing fields."""
        name = payload.get("name") or f"Product {idx + 1}"
        price = payload.get("price")
        qty = payload.get("quantity", 1)
        unit = payload.get("quantityunit", "unit")
        stock = payload.get("stock")

        status_icon = "[!]" if missing else ""

        lines = [
            f"{status_icon} *Product {idx + 1}/{total}*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"*{name}*",
            f"Price: {'Rs.' + str(price) if price else 'MISSING'} per {qty} {unit}",
            f"Stock: {stock if stock is not None else 'MISSING'}",
        ]

        # Optional fields
        if payload.get("brand"):
            lines.append(f"Brand: {payload['brand']}")
        if payload.get("colour"):
            lines.append(f"Colour: {payload['colour']}")
        if payload.get("description"):
            desc = payload['description'][:50] + "..." if len(payload['description']) > 50 else payload['description']
            lines.append(f"{desc}")

        # Missing fields warning
        if missing:
            lines.extend(["", f"Warning: *Missing:* {', '.join(missing)}"])

        # Similar products section
        if similar:
            relevant = [s for s in similar if len(s) >= 3 and s[2] >= 0.5][:3]
            if relevant:
                lines.extend(["", "*Similar in your inventory:*"])
                for i, match in enumerate(relevant, 1):
                    pid, pname, score = match[0], match[1], match[2]
                    pprice = match[3] if len(match) > 3 else None
                    price_str = f" (₹{pprice})" if pprice else ""
                    lines.append(f"   {i}. {pname}{price_str} - {int(score*100)}% match")

                if relevant[0][2] >= 0.8:
                    lines.extend([
                        "",
                        "Warning: *This may be a DUPLICATE!*",
                        "Reply 'update' to update existing,",
                        "or 'add new' to add anyway."
                    ])

        # Action instructions
        lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ])

        if missing:
            lines.append("Provide missing info, e.g., 'stock 50'")
        else:
            lines.append("'yes' to confirm, 'skip' to skip")

        return "\n".join(lines)

    def _handle_multi_product_reply(
        self,
        session: SessionState,
        message: str,
        product_index: int
    ) -> Dict[str, Any]:
        """Handle a reply to a specific product in multi-product mode."""
        lower = message.lower().strip()

        if product_index >= len(session.multi_product_payloads):
            return {"messages": [{"type": "text", "text": "Invalid product reference."}]}

        payload = session.multi_product_payloads[product_index]
        missing = session.multi_product_missing[product_index]
        similar = session.multi_product_similar[product_index] if product_index < len(session.multi_product_similar) else []
        name = payload.get("name") or f"Product {product_index + 1}"

        # Handle confirmation
        if lower in {"yes", "y", "ok", "confirm"}:
            if missing:
                return {
                    "messages": [{
                        "type": "text",
                        "text": f"Warning: Product {product_index + 1} ({name}) is missing: {', '.join(missing)}\n\nPlease provide the missing values first."
                    }]
                }
            session.multi_product_statuses[product_index] = "confirmed"
            return self._multi_product_status_update(session, product_index, "confirmed")

        # Handle skip
        if lower in {"skip", "no"}:
            session.multi_product_statuses[product_index] = "skipped"
            return self._multi_product_status_update(session, product_index, "skipped")

        # Handle "update" for similar product
        if lower == "update" and similar:
            matched_id = similar[0][0]
            payload["_update_existing_id"] = matched_id
            session.multi_product_statuses[product_index] = "confirmed"
            return self._multi_product_status_update(session, product_index, "confirmed (as update)")

        # Handle "add new" to ignore similarity warning
        if lower == "add new":
            payload["_force_new"] = True
            session.multi_product_statuses[product_index] = "confirmed"
            return self._multi_product_status_update(session, product_index, "confirmed")

        # Handle corrections
        corrections = self._parse_correction(message)
        if corrections:
            # Apply corrections to payload
            for field, value in corrections.items():
                payload[field] = value

            # Re-check missing fields
            new_missing = self._check_missing_fields(payload)
            session.multi_product_missing[product_index] = new_missing

            if new_missing:
                return {
                    "messages": [{
                        "type": "text",
                        "text": f"Updated Product {product_index + 1}!\n\nStill missing: {', '.join(new_missing)}"
                    }]
                }
            return {
                "messages": [{
                    "type": "text",
                    "text": f"Updated Product {product_index + 1} ({name})!\n\nReply 'yes' to confirm or provide more corrections."
                }]
            }

        # Didn't understand
        return {
            "messages": [{
                "type": "text",
                "text": (
                    f" Didn't understand that for Product {product_index + 1} ({name}).\n\n"
                    "Reply with:\n"
                    "• 'yes' to confirm\n"
                    "• 'skip' to skip\n"
                    "• Correction like 'stock 50' or 'price 100'"
                )
            }]
        }

    def _multi_product_status_update(
        self,
        session: SessionState,
        idx: int,
        action: str
    ) -> Dict[str, Any]:
        """Check if all products are processed and provide status update."""
        statuses = session.multi_product_statuses
        pending = statuses.count("pending")
        confirmed = statuses.count("confirmed")
        skipped = statuses.count("skipped")
        name = session.multi_product_payloads[idx].get("name") or f"Product {idx + 1}"

        icon = "" if "skipped" in action else ""

        if pending == 0:
            # All processed - prompt to save
            return {
                "messages": [{
                    "type": "text",
                    "text": (
                        f"{icon} Product {idx + 1} ({name}) {action}!\n\n"
                        f"*All products processed!*\n"
                        f"   Confirmed: {confirmed}\n"
                        f"   Skipped: {skipped}\n\n"
                        f"Type 'save' to add confirmed products.\n"
                        f"Type 'cancel' to discard all."
                    )
                }]
            }

        return {
            "messages": [{
                "type": "text",
                "text": (
                    f"{icon} Product {idx + 1} ({name}) {action}!\n\n"
                    f"Progress: {confirmed + skipped}/{len(statuses)} done, {pending} remaining."
                )
            }]
        }

    def _save_multi_products(self, session: SessionState) -> Dict[str, Any]:
        """Save all confirmed products to database."""
        added = []
        updated = []
        failed = []

        for idx, (payload, status) in enumerate(zip(
            session.multi_product_payloads,
            session.multi_product_statuses
        )):
            if status != "confirmed":
                continue

            name = payload.get("name") or f"Product {idx + 1}"

            try:
                # Check if this is an update or new add
                matched_id = payload.pop("_update_existing_id", None)
                payload.pop("_force_new", None)

                product_id = self.intake.enqueue_intake(
                    user_id=session.user_id,
                    mode="update" if matched_id else "add",
                    payload=payload,
                    matched_product_id=matched_id
                )

                if product_id:
                    if matched_id:
                        updated.append({"name": name, "id": product_id})
                    else:
                        added.append({"name": name, "id": product_id})
                else:
                    failed.append({"name": name, "error": "Unknown error"})
            except Exception as e:
                failed.append({"name": name, "error": str(e)})

        # Reset session
        session.reset()

        # Build response
        lines = []
        if added:
            lines.append(f"*Added {len(added)} product(s):*")
            for p in added:
                lines.append(f"   • {p['name']}")

        if updated:
            lines.append(f"\n*Updated {len(updated)} product(s):*")
            for p in updated:
                lines.append(f"   • {p['name']}")

        if failed:
            lines.append(f"\n*Failed {len(failed)} product(s):*")
            for p in failed:
                lines.append(f"   • {p['name']}: {p['error']}")

        if not added and not updated and not failed:
            lines.append("ℹ️ No products were confirmed for saving.")

        lines.append("\n\nType 'inventory' to see all your products.")

        return {"messages": [{"type": "text", "text": "\n".join(lines)}]}

    def _restore_snapshot(self, session: SessionState) -> bool:
        """Restore product state from db_snapshot. Returns True if restored."""
        snapshot = session.db_snapshot
        if not snapshot:
            return False

        product_id = snapshot.get("product_id")
        if not product_id:
            return False

        try:
            conn = self.sql_client._get_connection()
            cur = conn.cursor()

            # Check if product exists (was it newly added or just updated?)
            if snapshot.get("was_new_product"):
                # Product was newly added - delete it to rollback
                cur.execute("DELETE FROM product_table WHERE product_id = ?", (product_id,))
                print(f"Rollback: Deleted newly added product {product_id}")
            else:
                # Product existed before - restore original values
                cur.execute(
                    """UPDATE product_table
                       SET price = ?, quantity = ?, stock = ?
                       WHERE product_id = ?""",
                    (
                        snapshot.get("price"),
                        snapshot.get("quantity"),
                        snapshot.get("stock"),
                        product_id,
                    )
                )
                print(f"Rollback: Restored {product_id} to previous state")

            conn.commit()
            session.db_snapshot = None
            return True
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False

    def _parse_correction(self, message: str) -> Dict[str, Any]:
        """Parse correction messages like 'no, stock is 20' or 'price should be 15'."""
        corrections = {}
        lower = message.lower()

        # Stock corrections: "stock is 20", "stock should be 20", "total stock 20", "only 20 stock"
        stock_patterns = [
            r"stock\s*(?:is|should be|=|:)?\s*(\d+)",
            r"total\s*stock\s*(?:is)?\s*(\d+)",
            r"only\s*(\d+)\s*(?:stock|units|items)",
            r"(\d+)\s*(?:stock|units|items)\s*(?:only|left|available)",
            r"(\d+)\s+in\s+stock",  # "100 in stock"
        ]
        for pattern in stock_patterns:
            match = re.search(pattern, lower)
            if match:
                corrections["stock"] = int(match.group(1))
                break

        # Price corrections: "price is 15", "price should be 15", "rs 15", "₹15"
        price_patterns = [
            r"price\s*(?:is|should be|=|:)?\s*(?:rs\.?|₹)?\s*(\d+(?:\.\d+)?)",
            r"(?:rs\.?|₹)\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:rs|rupees)",
        ]
        for pattern in price_patterns:
            match = re.search(pattern, lower)
            if match:
                corrections["price"] = float(match.group(1))
                break

        # Quantity corrections: "quantity is 2", "per 2 units"
        qty_patterns = [
            r"(?:quantity|qty)\s*(?:is|should be|=|:)?\s*(\d+)",
            r"per\s*(\d+)\s*unit",
        ]
        for pattern in qty_patterns:
            match = re.search(pattern, lower)
            if match:
                corrections["quantity"] = int(match.group(1))
                break

        return corrections