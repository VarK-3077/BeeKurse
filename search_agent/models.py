"""
Pydantic models for unified search input/output
"""
from typing import List, Tuple, Optional, Literal
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Unified input format for all search queries"""

    product_query: str = Field(..., description="Natural language search query")
    product_category: str = Field(..., description="High-level category - strict filter (e.g., 'clothing', 'electronics')")
    product_subcategory: str = Field(..., description="Specific subcategory (e.g., 'shirt', 'polo shirt', 'watch')")

    properties: List[Tuple[str, float, str]] = Field(
        default_factory=list,
        description="List of (property_value, importance_weight, relation_type) tuples"
    )

    literals: List[Tuple[str, str, float, float]] = Field(
        default_factory=list,
        description="List of (literal_name, operator, value, buffer) tuples"
    )

    is_hq: bool = Field(
        default=False,
        description="True triggers 'Hurry Query' exact match logic"
    )

    prev_productid: Optional[str] = Field(
        default=None,
        description="Used for HQ exact match and SQ connected search"
    )

    prev_storeid: Optional[str] = Field(
        default=None,
        description="Store context for boosting store-specific products"
    )

    prev_products: List[Tuple[str, List[str]]] = Field(
        default_factory=list,
        description="List of (product_id, properties_list) for similarity queries"
    )

    sort_literal: Optional[Tuple[str, str]] = Field(
        default=None,
        description="Sort results by literal field. Tuple of (field_name, direction). E.g., ('price', 'asc') for cheapest"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "product_query": "red shirt under 20",
                "product_category": "clothing",
                "product_subcategory": "shirt",
                "properties": [
                    ["red", 1.5, "HAS_COLOUR"],
                    ["casual", 1.0, "HAS_STYLE"]
                ],
                "literals": [
                    ["price", "<", 20.0, 0.1],
                    ["size", "=", "M", 0.0]
                ],
                "is_hq": False,
                "prev_productid": "p-9876",
                "prev_storeid": "s-1234",
                "prev_products": [
                    ["p-123", ["color", "style"]]
                ]
            }
        }


class ProductScore(BaseModel):
    """Internal model for product scoring"""

    product_id: str
    property_score: float = 0.0
    connected_score: float = 0.0
    subcategory_score: float = 0.0
    final_score: float = 0.0

    def calculate_final_score(self) -> float:
        """Calculate final score as sum of property, connected, and subcategory scores"""
        self.final_score = self.property_score + self.connected_score + self.subcategory_score
        return self.final_score


class SearchResult(BaseModel):
    """Output format: ranked list of product IDs"""

    product_ids: List[str] = Field(
        ...,
        description="List of product IDs sorted by relevance (highest first)"
    )

    no_relevant_results: bool = Field(
        default=False,
        description="True if no products matched the query criteria after filtering"
    )

    filter_reason: Optional[str] = Field(
        default=None,
        description="Reason for no results: 'gender_filter', 'relevance_threshold', or 'no_matches'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "product_ids": ["p-456", "p-123", "p-789"],
                "no_relevant_results": False,
                "filter_reason": None
            }
        }


class VDBResult(BaseModel):
    """Vector DB search result"""

    id: str
    similarity: float
    metadata: dict = Field(default_factory=dict)


class SQLProduct(BaseModel):
    """SQL database product record"""

    # ACTUAL PRODUCTION SCHEMA
    product_id: str
    prod_name: str
    store: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    colour: Optional[str] = None
    description: Optional[str] = None
    dimensions: Optional[str] = None
    imageid: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    quantityunit: Optional[str] = None
    rating: Optional[float] = None
    size: Optional[str] = None
    stock: Optional[int] = None
    store_contact: Optional[str] = None
    store_location: Optional[dict] = None
    other_properties: Optional[dict] = None


class KGNode(BaseModel):
    """Knowledge Graph node"""

    id: str
    type: str
    properties: dict = Field(default_factory=dict)


class KGRelation(BaseModel):
    """Knowledge Graph relation"""

    source_id: str
    target_id: str
    relation_type: str
    properties: dict = Field(default_factory=dict)
