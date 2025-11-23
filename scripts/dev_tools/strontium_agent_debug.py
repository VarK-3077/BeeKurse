"""
Strontium Curator Agent - DEBUG VERSION
Shows JSON outputs at each pipeline stage for testing
"""
import json
from typing import Optional, Union

from search_agent.strontium.models import StrontiumOutput
from search_agent.strontium.llm_parser import LLMParser
from search_agent.strontium.user_context import UserContextManager
from search_agent.strontium.formatters import QueryFormatter
from search_agent.strontium.caching import CacheManager


class StrontiumAgentDebug:
    """
    DEBUG version of Strontium agent
    Prints JSON outputs at each stage for testing and debugging
    """

    def __init__(
        self,
        llm_client=None,
        kg_client=None,
        mock_data_dir: Optional[str] = None,
        enable_caching: bool = True,
        llm_cache_ttl: int = 3600,
        kg_cache_ttl: int = 3600,
        use_nvidia: bool = False,
        nvidia_api_key: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Strontium agent (Debug version)

        Args:
            llm_client: LLM client for query parsing (if None and use_nvidia=False, uses mock)
            kg_client: Knowledge graph client for property fetching
            mock_data_dir: Directory with mock user data JSON files
            enable_caching: Whether to enable caching
            llm_cache_ttl: LLM cache TTL in seconds
            kg_cache_ttl: KG cache TTL in seconds
            use_nvidia: If True, use NVIDIA API for LLM
            nvidia_api_key: NVIDIA API key (uses env var if not provided)
            verbose: If True, print detailed debug output
        """
        # Initialize components
        self.llm_parser = LLMParser(
            llm_client=llm_client,
            use_nvidia=use_nvidia,
            nvidia_api_key=nvidia_api_key
        )
        self.user_context_manager = UserContextManager(
            mock_data_dir=mock_data_dir,
            kg_client=kg_client
        )
        self.formatter = QueryFormatter()
        self.verbose = verbose

        # Initialize caching
        self.enable_caching = enable_caching
        if enable_caching:
            self.cache_manager = CacheManager(
                kg_client=kg_client,
                llm_ttl=llm_cache_ttl,
                kg_ttl=kg_cache_ttl
            )
        else:
            self.cache_manager = None

    def process_query(
        self,
        raw_query: str,
        user_id: str
    ) -> StrontiumOutput:
        """
        Process a raw natural language query (DEBUG VERSION)
        Prints JSON at each stage

        Args:
            raw_query: Natural language user query
            user_id: User ID for personalization

        Returns:
            Formatted output (search or detail query)
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("STRONTIUM DEBUG - QUERY PROCESSING")
            print("=" * 70)
            print(f"\n[INPUT]")
            print(f"  Query: {raw_query}")
            print(f"  User: {user_id}")
            print()

        # Step 1: Parse with LLM (check cache first)
        if self.verbose:
            print("=" * 70)
            print("STEP 1: LLM PARSING")
            print("=" * 70)

        if self.enable_caching:
            cached_parse = self.cache_manager.llm_cache.get(raw_query)
            if cached_parse:
                if self.verbose:
                    print("✓ Cache hit!")
                # Reconstruct parsed object from cached dict
                from .models import SearchQueryOutput, DetailQueryOutput
                if cached_parse['query_type'] == 'search':
                    parsed = SearchQueryOutput(**cached_parse)
                else:
                    parsed = DetailQueryOutput(**cached_parse)
            else:
                if self.verbose:
                    print(f"✗ Cache miss - calling LLM ({self.llm_parser.client_type} mode)")
                # Parse and cache
                parsed = self.llm_parser.parse(raw_query)
                self.cache_manager.llm_cache.set(raw_query, parsed.model_dump())
        else:
            if self.verbose:
                print(f"Calling LLM ({self.llm_parser.client_type} mode)...")
            parsed = self.llm_parser.parse(raw_query)

        if self.verbose:
            print("\n[LLM OUTPUT]")
            print(json.dumps(parsed.model_dump(), indent=2))
            print()

        # Step 2: Route based on query type
        if parsed.query_type == "detail":
            if self.verbose:
                print("=" * 70)
                print("STEP 2: DETAIL QUERY - SKIPPING ENRICHMENT")
                print("=" * 70)
                print("Detail queries don't need user context enrichment")
                print()

            result = self.formatter.format(parsed)

            if self.verbose:
                print("=" * 70)
                print("FINAL OUTPUT")
                print("=" * 70)
                print(json.dumps(self.formatter.to_dict(result), indent=2))
                print()

            return result

        # Step 2: Enrich with user context (for search queries)
        if self.verbose:
            print("=" * 70)
            print("STEP 2: USER CONTEXT ENRICHMENT")
            print("=" * 70)

        # Get user profile and history
        user_profile = self.user_context_manager.get_user_profile(user_id)
        purchase_history = self.user_context_manager.get_purchase_history(user_id)

        if self.verbose:
            print("\n[USER PROFILE]")
            if user_profile:
                print(json.dumps(user_profile.model_dump(), indent=2))
            else:
                print("No user profile found")

            print("\n[PURCHASE HISTORY]")
            if purchase_history:
                print(f"Total purchases: {len(purchase_history)}")
                for i, purchase in enumerate(purchase_history[:3], 1):
                    print(f"  {i}. {purchase.subcategory} ({purchase.product_id}) - {purchase.purchase_date}")
                if len(purchase_history) > 3:
                    print(f"  ... and {len(purchase_history) - 3} more")
            else:
                print("No purchase history")
            print()

        kg_cache = self.cache_manager.kg_cache if self.enable_caching else None
        enriched = self.user_context_manager.enrich_query(
            parsed,
            user_id,
            kg_cache=kg_cache
        )

        if self.verbose:
            print("[ENRICHED OUTPUT]")
            print(json.dumps(enriched.model_dump(), indent=2))
            print()

            # Show what was added
            print("[ENRICHMENT DETAILS]")
            for i, product in enumerate(enriched.products):
                print(f"  Product {i+1}: {product.subcategory}")

                # Compare with original
                original_props = set(p[0] for p in parsed.products[i].properties)
                enriched_props = set(p[0] for p in product.properties)
                added_props = enriched_props - original_props

                if added_props:
                    print(f"    ✓ Added properties: {', '.join(added_props)}")

                if product.is_hq:
                    print(f"    ✓ HQ enabled")
                    print(f"      - prev_productid: {product.prev_productid}")
                    print(f"      - prev_storeid: {product.prev_storeid}")
            print()

        # Step 3: Format to final JSON
        if self.verbose:
            print("=" * 70)
            print("STEP 3: FORMATTING")
            print("=" * 70)

        result = self.formatter.format(enriched, original_query=raw_query)

        if self.verbose:
            print("\n[FINAL OUTPUT - Ready for SearchOrchestrator]")
            print(json.dumps(self.formatter.to_dict(result), indent=2))
            print("\n" + "=" * 70)
            print()

        return result

    def process_query_to_dict(
        self,
        raw_query: str,
        user_id: str
    ) -> dict:
        """
        Process query and return as dictionary

        Args:
            raw_query: Natural language user query
            user_id: User ID

        Returns:
            Dictionary output
        """
        output = self.process_query(raw_query, user_id)
        return self.formatter.to_dict(output)

    def process_query_to_json(
        self,
        raw_query: str,
        user_id: str
    ) -> str:
        """
        Process query and return as JSON string

        Args:
            raw_query: Natural language user query
            user_id: User ID

        Returns:
            JSON string output
        """
        output = self.process_query(raw_query, user_id)
        return self.formatter.to_json(output)

    def clear_caches(self):
        """Clear all caches"""
        if self.cache_manager:
            self.cache_manager.clear_all()
