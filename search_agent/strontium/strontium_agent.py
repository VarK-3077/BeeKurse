"""
Strontium Curator Agent - Main Orchestrator
Coordinates query parsing, context enrichment, and formatting
"""
from typing import Optional, Union

from .models import StrontiumOutput
from .llm_parser import LLMParser
from .user_context import UserContextManager
from .formatters import QueryFormatter
from .caching import CacheManager


class StrontiumAgent:
    """
    Main Strontium curator agent
    Transforms natural language queries into structured JSON for SearchOrchestrator
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
        nvidia_api_key: Optional[str] = None
    ):
        """
        Initialize Strontium agent

        Args:
            llm_client: LLM client for query parsing (if None and use_nvidia=False, uses mock)
            kg_client: Knowledge graph client for property fetching
            mock_data_dir: Directory with mock user data JSON files
            enable_caching: Whether to enable caching
            llm_cache_ttl: LLM cache TTL in seconds
            kg_cache_ttl: KG cache TTL in seconds
            use_nvidia: If True, use NVIDIA API for LLM
            nvidia_api_key: NVIDIA API key (uses env var if not provided)
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
        Process a raw natural language query

        This is the main entry point for Strontium.
        Pipeline:
        1. Parse query with LLM (with caching)
        2. Enrich with user context (for search queries)
        3. Format to JSON output

        Args:
            raw_query: Natural language user query
            user_id: User ID for personalization

        Returns:
            Formatted output (search or detail query)
        """
        # Step 1: Parse with LLM (check cache first)
        if self.enable_caching:
            cached_parse = self.cache_manager.llm_cache.get(raw_query)
            if cached_parse:
                # Reconstruct parsed object from cached dict
                from .models import SearchQueryOutput, DetailQueryOutput, ChatQueryOutput, CartActionOutput, CartViewOutput
                if cached_parse['query_type'] == 'search':
                    parsed = SearchQueryOutput(**cached_parse)
                elif cached_parse['query_type'] == 'detail':
                    parsed = DetailQueryOutput(**cached_parse)
                elif cached_parse['query_type'] == 'chat':
                    parsed = ChatQueryOutput(**cached_parse)
                elif cached_parse['query_type'] == 'cart_action':
                    parsed = CartActionOutput(**cached_parse)
                elif cached_parse['query_type'] == 'cart_view':
                    parsed = CartViewOutput(**cached_parse)
            else:
                # Parse and cache
                parsed = self.llm_parser.parse(raw_query)
                self.cache_manager.llm_cache.set(raw_query, parsed.model_dump())
        else:
            parsed = self.llm_parser.parse(raw_query)

        # Step 2: Route based on query type
        if parsed.query_type in ("detail", "chat", "cart_action", "cart_view"):
            # These queries skip enrichment (no products to enrich)
            return self.formatter.format(parsed)

        # Step 3: Enrich with user context (for search queries only)
        kg_cache = self.cache_manager.kg_cache if self.enable_caching else None
        enriched = self.user_context_manager.enrich_query(
            parsed,
            user_id,
            kg_cache=kg_cache
        )

        # Step 4: Format to final JSON
        return self.formatter.format(enriched, original_query=raw_query)

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


# Convenience function for simple usage
def process_query(
    query: str,
    user_id: str = "default_user",
    llm_client=None,
    kg_client=None,
    mock_data_dir: Optional[str] = None,
    use_nvidia: bool = False,
    nvidia_api_key: Optional[str] = None
) -> dict:
    """
    Convenience function to process a query

    Args:
        query: Natural language query
        user_id: User ID (default: "default_user")
        llm_client: Optional LLM client
        kg_client: Optional KG client
        mock_data_dir: Optional mock data directory
        use_nvidia: If True, use NVIDIA API
        nvidia_api_key: NVIDIA API key

    Returns:
        Dictionary output
    """
    agent = StrontiumAgent(
        llm_client=llm_client,
        kg_client=kg_client,
        mock_data_dir=mock_data_dir,
        use_nvidia=use_nvidia,
        nvidia_api_key=nvidia_api_key
    )
    return agent.process_query_to_dict(query, user_id)
