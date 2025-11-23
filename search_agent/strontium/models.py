"""
Pydantic data models for Strontium Curator Agent
"""
from typing import List, Tuple, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# LLM Output Models (Step 1: Parsing)
# ============================================================================

class ProductRequest(BaseModel):
    """Single product request from LLM parsing"""
    product_category: str = Field(..., description="High-level category (e.g., 'clothing', 'electronics', 'furniture', 'grocery', 'other')")
    product_subcategory: str = Field(..., description="Product subcategory (e.g., 'shirt', 'pillow', 'polo shirt')")
    properties: List[Tuple[str, float, str]] = Field(
        default_factory=list,
        description="List of (property_value, importance_weight, relation_type). E.g., ('red', 1.5, 'HAS_COLOUR')"
    )
    literals: List[Tuple[str, str, float, float]] = Field(
        default_factory=list,
        description="List of (field, operator, value, buffer). E.g., ('price', '<', 30.0, 0.1)"
    )
    prev_products: List[Tuple[str, List[str]]] = Field(
        default_factory=list,
        description="List of (product_id, [liked_properties]). Empty list [] = fetch all"
    )
    is_hq: bool = Field(
        default=False,
        description="High-quality/hurry query flag. True if user says 'my usual'"
    )
    sort_literal: Optional[Tuple[str, str]] = Field(
        default=None,
        description="Sort by literal field. Tuple of (field_name, direction). E.g., ('price', 'asc') for cheapest, ('price', 'desc') for most expensive"
    )


class SearchQueryOutput(BaseModel):
    """LLM output for search queries"""
    query_type: Literal["search"] = "search"
    products: List[ProductRequest] = Field(
        ...,
        description="List of products requested in the query"
    )


class DetailQueryOutput(BaseModel):
    """LLM output for detail/information queries"""
    query_type: Literal["detail"] = "detail"
    original_query: str = Field(
        ...,
        description="Original user query for downstream LLM to answer"
    )
    product_id: str = Field(..., description="Product ID to get details about")
    properties_to_explain: List[str] = Field(
        ...,
        description="Properties user wants to know. ['*'] for all properties"
    )
    relation_types: List[str] = Field(
        default_factory=list,
        description="Relation types for the properties being queried. E.g., ['HAS_MATERIAL', 'HAS_CARE_INSTRUCTIONS']"
    )
    query_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords to search in product descriptions. E.g., ['dry cleaning', 'wash', 'care']"
    )


class ChatQueryOutput(BaseModel):
    """LLM output for general chat/greetings"""
    query_type: Literal["chat"] = "chat"
    message: str = Field(..., description="The user's message")


# Union type for LLM output
LLMOutput = SearchQueryOutput | DetailQueryOutput | ChatQueryOutput


# ============================================================================
# User Context Models (Step 2: Enrichment)
# ============================================================================

class PurchaseRecord(BaseModel):
    """Single purchase from user history"""
    user_id: str
    product_id: str
    subcategory: str
    store: str
    purchase_date: datetime
    price: float
    rating: Optional[float] = None


class UserProfile(BaseModel):
    """User profile with preferences"""
    user_id: str

    # Preferences
    style_preferences: List[str] = Field(
        default_factory=list,
        description="E.g., ['casual', 'minimalist', 'modern']"
    )
    favorite_brands: Dict[str, float] = Field(
        default_factory=dict,
        description="Brand -> affinity score (0.0-1.0)"
    )
    size_preferences: Dict[str, str] = Field(
        default_factory=dict,
        description="Subcategory -> size. E.g., {'shirt': 'M', 'shoes': '10'}"
    )

    # Behavioral flags
    eco_conscious: bool = False
    organic_preference: float = Field(
        default=0.0,
        description="Preference for organic products (0.0-1.0)"
    )
    budget_patterns: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Subcategory -> (min_price, max_price)"
    )


class EnrichedProductRequest(ProductRequest):
    """Product request enriched with user context"""
    prev_productid: Optional[str] = None  # For HQ fast path
    prev_storeid: Optional[str] = None    # For store preference


class EnrichedSearchQuery(BaseModel):
    """Enriched search query ready for formatting"""
    query_type: Literal["search"] = "search"
    products: List[EnrichedProductRequest]


# ============================================================================
# Output Models (Step 3: Formatting)
# ============================================================================

class FormattedProductQuery(BaseModel):
    """Single product query formatted for SearchOrchestrator"""
    product_query: str = Field(..., description="Original query text")
    product_category: str = Field(..., description="High-level category (e.g., 'clothing', 'electronics')")
    product_subcategory: str = Field(..., description="Specific subcategory (e.g., 'polo shirt', 'smartwatch')")
    properties: List[Tuple[str, float, str]] = Field(
        ...,
        description="List of (value, weight, relation_type). E.g., ('red', 1.5, 'HAS_COLOUR')"
    )
    literals: List[Tuple[str, str, float, float]]
    is_hq: bool
    prev_productid: Optional[str] = None
    prev_storeid: Optional[str] = None
    prev_products: List[Tuple[str, List[str]]] = Field(default_factory=list)
    sort_literal: Optional[Tuple[str, str]] = None


class FormattedSearchOutput(BaseModel):
    """Final output for search queries"""
    query_type: Literal["search"] = "search"
    products: List[FormattedProductQuery]


class FormattedDetailOutput(BaseModel):
    """Final output for detail queries"""
    query_type: Literal["detail"] = "detail"
    original_query: str
    product_id: str
    properties_to_explain: List[str]
    relation_types: List[str] = Field(default_factory=list)
    query_keywords: List[str] = Field(default_factory=list)


class FormattedChatOutput(BaseModel):
    """Final output for chat queries"""
    query_type: Literal["chat"] = "chat"
    message: str


# Union type for final output
StrontiumOutput = FormattedSearchOutput | FormattedDetailOutput | FormattedChatOutput


# ============================================================================
# Cache Models
# ============================================================================

class CachedLLMResponse(BaseModel):
    """Cached LLM parsing result"""
    query_hash: str
    parsed_output: Dict[str, Any]  # LLMOutput as dict
    timestamp: datetime
    ttl: int = Field(default=3600, description="Time to live in seconds")


class CachedKGProperties(BaseModel):
    """Cached product properties from Knowledge Graph"""
    product_id: str
    properties: Dict[str, List[str]] = Field(
        ...,
        description="Relation type -> property values. E.g., {'HAS_COLOR': ['Color:Red']}"
    )
    timestamp: datetime
    ttl: int = Field(default=3600, description="Time to live in seconds")
