"""
User context management and query enrichment for Strontium
"""
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from .models import (
    UserProfile,
    PurchaseRecord,
    SearchQueryOutput,
    EnrichedSearchQuery,
    EnrichedProductRequest
)


class UserContextManager:
    """
    Manages user profiles and purchase history
    Enriches queries with user context
    """

    def __init__(self, mock_data_dir: Optional[str] = None, kg_client=None):
        """
        Initialize user context manager

        Args:
            mock_data_dir: Directory containing mock user data JSON files
            kg_client: Knowledge graph client for fetching product properties
        """
        self.mock_data_dir = mock_data_dir or "mock_data"
        self.kg_client = kg_client
        self._profiles_cache: Dict[str, UserProfile] = {}
        self._history_cache: Dict[str, List[PurchaseRecord]] = {}
        self._load_mock_data()

    def _load_mock_data(self):
        """Load mock user profiles and purchase history"""
        base_path = Path(self.mock_data_dir)

        # Load profiles
        profiles_path = base_path / "user_profiles.json"
        if profiles_path.exists():
            with open(profiles_path, 'r') as f:
                profiles_data = json.load(f)
                for profile_data in profiles_data:
                    profile = UserProfile(**profile_data)
                    self._profiles_cache[profile.user_id] = profile

        # Load purchase history
        history_path = base_path / "purchase_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history_data = json.load(f)
                for record in history_data:
                    # Convert date string to datetime
                    if isinstance(record.get('purchase_date'), str):
                        record['purchase_date'] = datetime.fromisoformat(record['purchase_date'])
                    purchase = PurchaseRecord(**record)
                    if purchase.user_id not in self._history_cache:
                        self._history_cache[purchase.user_id] = []
                    self._history_cache[purchase.user_id].append(purchase)

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self._profiles_cache.get(user_id)

    def get_purchase_history(self, user_id: str) -> List[PurchaseRecord]:
        """Get purchase history for user"""
        return self._history_cache.get(user_id, [])

    def enrich_query(
        self,
        parsed_query: SearchQueryOutput,
        user_id: str,
        kg_cache=None
    ) -> EnrichedSearchQuery:
        """
        Enrich search query with user context

        Args:
            parsed_query: LLM parsed query
            user_id: User ID
            kg_cache: Optional KG property cache

        Returns:
            Enriched query with user context
        """
        user_profile = self.get_user_profile(user_id)
        purchase_history = self.get_purchase_history(user_id)

        enriched_products = []

        for product in parsed_query.products:
            enriched = self._enrich_product(
                product,
                user_profile,
                purchase_history,
                kg_cache
            )
            enriched_products.append(enriched)

        return EnrichedSearchQuery(products=enriched_products)

    def _enrich_product(
        self,
        product: 'ProductRequest',
        user_profile: Optional[UserProfile],
        purchase_history: List[PurchaseRecord],
        kg_cache
    ) -> EnrichedProductRequest:
        """
        Enrich a single product request with user context

        Args:
            product: Product request from LLM
            user_profile: User profile (if available)
            purchase_history: Purchase history
            kg_cache: KG property cache

        Returns:
            Enriched product request
        """
        # Convert to enriched model
        enriched = EnrichedProductRequest(
            product_category=product.product_category,
            product_subcategory=product.product_subcategory,
            properties=list(product.properties),  # Will modify
            literals=list(product.literals),
            prev_products=list(product.prev_products),
            is_hq=product.is_hq,
            prev_productid=None,
            prev_storeid=None,
            sort_literal=product.sort_literal
        )

        # Step 1: Check if user regularly purchases this product type
        is_regular = self._is_regularly_purchased(product.product_subcategory, purchase_history)

        # If is_hq already set (LLM detected "my usual") OR user regularly buys this
        if enriched.is_hq or is_regular:
            enriched.is_hq = True
            enriched.prev_productid = self._get_most_recent_purchase_id(
                product.product_subcategory,
                purchase_history
            )
            enriched.prev_storeid = self._get_preferred_store(
                product.product_subcategory,
                purchase_history
            )

        # Step 2: Add implicit user preferences (if profile exists)
        if user_profile:
            implicit_properties = self._get_implicit_properties(
                product.product_subcategory,
                user_profile
            )
            enriched.properties = self._merge_properties(
                enriched.properties,
                implicit_properties
            )

        # Step 3: Populate prev_products if empty list (fetch from KG)
        if kg_cache and enriched.prev_products:
            enriched_prev_products = []
            for prev_id, prev_props in enriched.prev_products:
                if not prev_props:  # Empty list = fetch all
                    if kg_cache:
                        all_props = kg_cache.get_product_properties(prev_id)
                        enriched_prev_products.append((prev_id, all_props))
                    else:
                        # If no cache, keep empty (will be fetched later)
                        enriched_prev_products.append((prev_id, []))
                else:
                    enriched_prev_products.append((prev_id, prev_props))
            enriched.prev_products = enriched_prev_products

        return enriched

    def _is_regularly_purchased(
        self,
        subcategory: str,
        purchase_history: List[PurchaseRecord]
    ) -> bool:
        """
        Check if user regularly buys this product type

        Criteria:
        - Purchased 3+ times
        - At least once in last 60 days
        - Consistent product (same or very similar)
        """
        purchases = [p for p in purchase_history if p.subcategory == subcategory]

        if len(purchases) < 3:
            return False

        # Check recent purchases
        sixty_days_ago = datetime.now() - timedelta(days=60)
        recent = [p for p in purchases if p.purchase_date > sixty_days_ago]
        if not recent:
            return False

        # Check if same product ID appears multiple times
        product_counts = Counter([p.product_id for p in purchases])
        most_common = product_counts.most_common(1)[0]

        if most_common[1] >= 2:  # Bought same product 2+ times
            return True

        return False

    def _get_most_recent_purchase_id(
        self,
        subcategory: str,
        purchase_history: List[PurchaseRecord]
    ) -> Optional[str]:
        """Get the most recently purchased product ID for this subcategory"""
        purchases = [p for p in purchase_history if p.subcategory == subcategory]
        if not purchases:
            return None
        purchases.sort(key=lambda p: p.purchase_date, reverse=True)
        return purchases[0].product_id

    def _get_preferred_store(
        self,
        subcategory: str,
        purchase_history: List[PurchaseRecord]
    ) -> Optional[str]:
        """Get the preferred store for this subcategory"""
        purchases = [p for p in purchase_history if p.subcategory == subcategory]
        if not purchases:
            return None

        # Get most common store
        store_counts = Counter([p.store for p in purchases])
        most_common_store = store_counts.most_common(1)[0][0]
        return most_common_store

    def _get_implicit_properties(
        self,
        subcategory: str,
        user_profile: UserProfile
    ) -> List[Tuple[str, float, str]]:
        """
        Get implicit properties from user profile (with relation types)

        Args:
            subcategory: Product subcategory
            user_profile: User profile

        Returns:
            List of (property_value, weight, relation_type) tuples
        """
        implicit = []

        # Size preferences
        if subcategory in user_profile.size_preferences:
            size = user_profile.size_preferences[subcategory]
            implicit.append((f"size_{size}", 0.8, "HAS_SIZE"))

        # Brand preferences
        for brand, affinity in user_profile.favorite_brands.items():
            if affinity > 0.7:  # Strong preference
                implicit.append((f"brand_{brand}", 0.9 * affinity, "HAS_BRAND"))

        # Style preferences
        for style in user_profile.style_preferences:
            implicit.append((f"style_{style}", 0.7, "HAS_STYLE"))

        # Eco-conscious
        if user_profile.eco_conscious:
            implicit.append(("eco_friendly", 0.8, "HAS_FEATURE"))
            if user_profile.organic_preference > 0.5:
                implicit.append(("organic", 0.7 * user_profile.organic_preference, "HAS_FEATURE"))

        return implicit

    def _merge_properties(
        self,
        llm_properties: List[Tuple[str, float, str]],
        implicit_properties: List[Tuple[str, float, str]]
    ) -> List[Tuple[str, float, str]]:
        """
        Merge LLM properties with implicit user properties (with relation types)

        LLM properties have "majority opinion" - their weights are primary
        Implicit properties either boost existing or add new ones

        Args:
            llm_properties: Properties from LLM parsing (value, weight, relation_type)
            implicit_properties: Properties from user profile (value, weight, relation_type)

        Returns:
            Merged property list (value, weight, relation_type)
        """
        # Convert to dict for easier manipulation: {value: (weight, relation_type)}
        merged = {prop: (weight, relation) for prop, weight, relation in llm_properties}

        for prop, weight, relation in implicit_properties:
            if prop in merged:
                # Boost existing weight (LLM has majority, boost 20%)
                existing_weight, existing_relation = merged[prop]
                merged[prop] = (existing_weight * 1.2, existing_relation)  # Keep LLM's relation type
            else:
                # Add new property with lower weight
                merged[prop] = (weight, relation)

        # Convert back to list of 3-tuples
        return [(prop, weight, relation) for prop, (weight, relation) in merged.items()]
