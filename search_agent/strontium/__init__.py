"""
Strontium Curator Agent
An expert sales assistant for BeeKurse, built on KURSE
(Knowledge Utilization, Retrieval, and Summarization Engine)

Transforms natural language queries into structured JSON for SearchOrchestrator
"""

from .strontium_agent import StrontiumAgent, process_query
from .models import (
    StrontiumOutput,
    FormattedSearchOutput,
    FormattedDetailOutput,
    FormattedProductQuery
)

__version__ = "1.0.0"

__all__ = [
    "StrontiumAgent",
    "process_query",
    "StrontiumOutput",
    "FormattedSearchOutput",
    "FormattedDetailOutput",
    "FormattedProductQuery"
]
