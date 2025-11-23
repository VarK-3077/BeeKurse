"""
Unified Search Orchestrator for Kurse Ecommerce

Main orchestrator that coordinates:
1. HQ Fast Path (Hurry Query)
2. Property Search (RQ)
3. Connected Search (SQ)
4. Score Combination & Ranking
"""
import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from search_agent.models import SearchQuery, SearchResult, ProductScore
from search_agent.database.sql_client import SQLClient
from search_agent.database.vdb_client import MainVDBClient, PropertyVDBClient, RelationVDBClient
from search_agent.database.kg_client import KGClient
from search_agent.scoring.property_search import PropertySearch
from search_agent.scoring.connected_search import ConnectedSearch
from search_agent.scoring.subcategory_scorer import SubcategoryScorer
from search_agent.scoring.score_combiner import ScoreCombiner
from search_agent.orchestrator.detail_service import ProductDetailService
from search_agent.orchestrator.chat_handler import ChatHandler
from config.config import Config
import numpy as np

config = Config


class SearchOrchestrator:
    """Main search orchestrator"""

    def __init__(self):
        """Initialize orchestrator and database clients"""
        # Initialize database clients
        self.sql_client = SQLClient()
        self.main_vdb = MainVDBClient()
        self.property_vdb = PropertyVDBClient()
        self.relation_vdb = RelationVDBClient()
        self.kg_client = KGClient()

        # Initialize search algorithms
        self.property_search = PropertySearch(
            sql_client=self.sql_client,
            main_vdb_client=self.main_vdb,
            property_vdb_client=self.property_vdb,
            relation_vdb_client=self.relation_vdb,
            kg_client=self.kg_client
        )

        self.connected_search = ConnectedSearch(
            sql_client=self.sql_client,
            kg_client=self.kg_client
        )

        self.subcategory_scorer = SubcategoryScorer(
            sql_client=self.sql_client,
            main_vdb_client=self.main_vdb
        )

        self.score_combiner = ScoreCombiner(sql_client=self.sql_client)

        # Initialize detail service
        self.detail_service = ProductDetailService(
            sql_client=self.sql_client,
            main_vdb_client=self.main_vdb,
            relation_vdb_client=self.relation_vdb,
            kg_client=self.kg_client,
            llm_client=None  # Will use mock for now
        )

        # Initialize chat handler
        self.chat_handler = ChatHandler()

    def search(self, query: SearchQuery) -> SearchResult:
        """
        Execute unified search

        Args:
            query: SearchQuery object with all search parameters

        Returns:
            SearchResult with ranked product IDs
        """
        # STEP 0: Embed target subcategory in parallel with HQ check
        target_subcat_embedding = None
        if config.ENABLE_SUBCATEGORY_SCORING:
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Embed subcategory in parallel
                embed_future = executor.submit(
                    self.subcategory_scorer.embed_subcategory,
                    query.product_subcategory
                )

                # HQ Fast Path check
                hq_future = None
                if query.is_hq and query.prev_productid:
                    hq_future = executor.submit(self._hurry_query_fast_path, query)

                # Get results
                target_subcat_embedding = embed_future.result()
                if hq_future:
                    hq_result = hq_future.result()
                    if hq_result:
                        # Found and in stock - return immediately
                        return SearchResult(product_ids=hq_result)
        else:
            # STEP 1: HQ Fast Path (if applicable)
            if query.is_hq and query.prev_productid:
                hq_result = self._hurry_query_fast_path(query)
                if hq_result:
                    # Found and in stock - return immediately
                    return SearchResult(product_ids=hq_result)

        # STEP 2: Parallel Property Search (RQ) + Connected Search (SQ)
        property_scores, connected_scores = self._parallel_retrieval(query)

        # STEP 3: Subcategory Scoring (if enabled)
        subcategory_scores = {}
        if config.ENABLE_SUBCATEGORY_SCORING and property_scores:
            # Get all product IDs from property search
            product_ids = list(property_scores.keys())

            # Score by subcategory similarity
            subcategory_scores = self.subcategory_scorer.score_products(
                product_ids=product_ids,
                target_subcategory=query.product_subcategory,
                target_embedding=target_subcat_embedding
            )

        # STEP 4: Combine, Score & Rank
        ranked_product_ids = self.score_combiner.combine_and_rank(
            property_scores=property_scores,
            connected_scores=connected_scores,
            subcategory_scores=subcategory_scores if subcategory_scores else None
        )

        # STEP 5: Re-rank by literal field if sort_literal is specified (for superlatives)
        if query.sort_literal:
            ranked_product_ids = self.score_combiner.rank_by_literal(
                product_ids=ranked_product_ids,
                sort_literal=query.sort_literal
            )

        return SearchResult(product_ids=ranked_product_ids)

    def search_strontium(self, strontium_output: dict) -> List[SearchResult]:
        """
        Execute search with Strontium output format

        Args:
            strontium_output: Strontium JSON with structure:
                {
                    "query_type": "search",
                    "products": [
                        {
                            "product_query": "...",
                            "product_basetype": "...",
                            "properties": [[value, weight, relation_type], ...],
                            "literals": [[field, op, value, buffer], ...],
                            "is_hq": bool,
                            "prev_productid": str or None,
                            "prev_storeid": str or None,
                            "prev_products": [[product_id, [properties]], ...]
                        },
                        ...
                    ]
                }

        Returns:
            List of SearchResult objects (one per product query)

        Raises:
            ValueError: If query_type is not "search"
        """
        if strontium_output.get("query_type") != "search":
            raise ValueError(
                f"Invalid query_type: {strontium_output.get('query_type')}. "
                "Use 'search' for product searches."
            )

        results = []
        for product_query_dict in strontium_output.get("products", []):
            # Create SearchQuery from Strontium product format
            query = SearchQuery(**product_query_dict)

            # Execute search
            result = self.search(query)
            results.append(result)

        return results

    def answer_detail_query(self, detail_output: dict) -> str:
        """
        Answer a detail query about a product

        Args:
            detail_output: Strontium detail query output with structure:
                {
                    "query_type": "detail",
                    "original_query": "...",
                    "product_id": "...",
                    "properties_to_explain": [...],
                    "relation_types": [...],
                    "query_keywords": [...]
                }

        Returns:
            Natural language answer as string

        Raises:
            ValueError: If query_type is not "detail"
        """
        if detail_output.get("query_type") != "detail":
            raise ValueError(
                f"Invalid query_type: {detail_output.get('query_type')}. "
                "Use 'detail' for product detail queries."
            )

        return self.detail_service.answer_detail_query(
            product_id=detail_output["product_id"],
            original_query=detail_output["original_query"],
            properties_to_explain=detail_output.get("properties_to_explain", ["*"]),
            relation_types=detail_output.get("relation_types", []),
            query_keywords=detail_output.get("query_keywords", [])
        )

    def handle_chat(self, chat_output: dict) -> str:
        """
        Handle a chat query (greetings, general conversation)

        Args:
            chat_output: Strontium chat query output with structure:
                {
                    "query_type": "chat",
                    "message": "..."
                }

        Returns:
            Friendly chat response as string

        Raises:
            ValueError: If query_type is not "chat"
        """
        return self.chat_handler.handle_chat_output(chat_output)

    def _hurry_query_fast_path(self, query: SearchQuery) -> Optional[List[str]]:
        """
        Step 1: HQ Fast Path for exact product lookup

        Args:
            query: SearchQuery with is_hq=True and prev_productid

        Returns:
            List with single product_id if found and in stock, None otherwise
        """
        if config.DEBUG:
            print(f"\n=== HQ Fast Path: {query.prev_productid} ===")

        # Direct SQL lookup
        product = self.sql_client.get_product_by_id(
            product_id=query.prev_productid,
            store_id=query.prev_storeid  # Optional store filter
        )

        if product:
            if config.DEBUG:
                print(f"  Found product: {product.product_id}, stock: {product.stock}")

            # Check stock
            if product.stock > 0:
                if config.DEBUG:
                    print("  ✓ In stock - returning immediately")
                return [product.product_id]
            else:
                if config.DEBUG:
                    print("  ✗ Out of stock - falling through to full search")
        else:
            if config.DEBUG:
                print("  ✗ Product not found - falling through to full search")

        return None

    def _parallel_retrieval(self, query: SearchQuery) -> tuple:
        """
        Step 2: Execute Property Search and Connected Search in parallel

        Args:
            query: SearchQuery object

        Returns:
            Tuple of (property_scores, connected_scores)
        """
        if config.DEBUG:
            print("\n=== Parallel Retrieval (RQ + SQ) ===")

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            property_future = executor.submit(
                self._execute_property_search, query
            )
            connected_future = executor.submit(
                self._execute_connected_search, query
            )

            # Wait for results
            property_scores = property_future.result()
            connected_scores = connected_future.result()

        return property_scores, connected_scores

    def _execute_property_search(self, query: SearchQuery) -> dict:
        """Execute property search (RQ)"""
        if config.DEBUG:
            print("  → Executing Property Search (RQ)...")

        return self.property_search.search(
            category=query.product_category,
            subcategory=query.product_subcategory,
            properties=query.properties,
            literals=query.literals
        )

    def _execute_connected_search(self, query: SearchQuery) -> dict:
        """Execute connected search (SQ)"""
        # Determine if we should run connected search
        should_run_sq = (
            query.prev_productid is not None or
            query.prev_storeid is not None
        )

        if not should_run_sq:
            if config.DEBUG:
                print("  → Skipping Connected Search (no context)")
            return {}

        if config.DEBUG:
            print("  → Executing Connected Search (SQ)...")

        return self.connected_search.search(
            category=query.product_category,
            literals=query.literals,
            prev_productid=query.prev_productid,
            prev_storeid=query.prev_storeid
        )

    def get_detailed_results(self, query: SearchQuery) -> List[ProductScore]:
        """
        Get detailed results with score breakdown

        Args:
            query: SearchQuery object

        Returns:
            List of ProductScore objects with detailed scoring
        """
        # Execute search steps
        if query.is_hq and query.prev_productid:
            hq_result = self._hurry_query_fast_path(query)
            if hq_result:
                # Return HQ result as ProductScore
                return [
                    ProductScore(
                        product_id=hq_result[0],
                        property_score=0.0,
                        connected_score=0.0,
                        final_score=999.0  # Indicate HQ match
                    )
                ]

        # Parallel retrieval
        property_scores, connected_scores = self._parallel_retrieval(query)

        # Get detailed scores
        return self.score_combiner.get_detailed_scores(
            property_scores=property_scores,
            connected_scores=connected_scores
        )

    def close(self):
        """Close all database connections"""
        self.sql_client.close()
        self.kg_client.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience function for single searches
def search(query_dict: dict) -> List[str]:
    """
    Execute a single search query

    Args:
        query_dict: Dictionary matching SearchQuery schema

    Returns:
        List of product IDs

    Example:
        >>> result = search({
        ...     "product_query": "red shirt under 20",
        ...     "product_basetype": "shirt",
        ...     "properties": [["color: 'red'", 1.5]],
        ...     "literals": [["price", "<", 20.0, 0.1]],
        ...     "is_hq": False
        ... })
    """
    query = SearchQuery(**query_dict)

    with SearchOrchestrator() as orchestrator:
        result = orchestrator.search(query)
        return result.product_ids
