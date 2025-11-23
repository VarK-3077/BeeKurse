"""
Output formatters for Strontium
Converts enriched queries to JSON format for SearchOrchestrator
"""
from typing import Union
from .models import (
    EnrichedSearchQuery,
    DetailQueryOutput,
    ChatQueryOutput,
    FormattedSearchOutput,
    FormattedDetailOutput,
    FormattedChatOutput,
    FormattedProductQuery,
    StrontiumOutput
)
class QueryFormatter:
    """
    Formats enriched queries into final JSON output
    """

    def format(
        self,
        query: Union[EnrichedSearchQuery, DetailQueryOutput, ChatQueryOutput],
        original_query: str = ""
    ) -> StrontiumOutput:
        """
        Format query to final JSON output

        Args:
            query: Enriched search query, detail query, or chat query
            original_query: Original user query text (for search queries)

        Returns:
            Formatted output ready for SearchOrchestrator or chat handler
        """
        if isinstance(query, EnrichedSearchQuery):
            return self._format_search(query, original_query)
        elif isinstance(query, DetailQueryOutput):
            return self._format_detail(query)
        elif isinstance(query, ChatQueryOutput):
            return self._format_chat(query)
        else:
            raise ValueError(f"Unknown query type: {type(query)}")

    def _format_search(
        self,
        query: EnrichedSearchQuery,
        original_query: str
    ) -> FormattedSearchOutput:
        """
        Format search query

        Args:
            query: Enriched search query
            original_query: Original user query text

        Returns:
            Formatted search output
        """
        formatted_products = []

        for product in query.products:
            # Category is now provided directly by LLM (no mapping needed)
            formatted = FormattedProductQuery(
                product_query=original_query,
                product_category=product.product_category,
                product_subcategory=product.product_subcategory,
                properties=product.properties,
                literals=product.literals,
                is_hq=product.is_hq,
                prev_productid=product.prev_productid,
                prev_storeid=product.prev_storeid,
                prev_products=product.prev_products,
                sort_literal=product.sort_literal
            )
            formatted_products.append(formatted)

        return FormattedSearchOutput(products=formatted_products)

    def _format_detail(
        self,
        query: DetailQueryOutput
    ) -> FormattedDetailOutput:
        """
        Format detail query

        Args:
            query: Detail query output

        Returns:
            Formatted detail output
        """
        return FormattedDetailOutput(
            query_type="detail",
            original_query=query.original_query,
            product_id=query.product_id,
            properties_to_explain=query.properties_to_explain,
            relation_types=query.relation_types,
            query_keywords=query.query_keywords
        )

    def _format_chat(
        self,
        query: ChatQueryOutput
    ) -> FormattedChatOutput:
        """
        Format chat query

        Args:
            query: Chat query output

        Returns:
            Formatted chat output
        """
        return FormattedChatOutput(
            query_type="chat",
            message=query.message
        )

    def to_dict(self, output: StrontiumOutput) -> dict:
        """
        Convert output to dictionary for JSON serialization

        Args:
            output: Formatted output

        Returns:
            Dictionary representation
        """
        return output.model_dump()

    def to_json(self, output: StrontiumOutput) -> str:
        """
        Convert output to JSON string

        Args:
            output: Formatted output

        Returns:
            JSON string
        """
        return output.model_dump_json(indent=2)
