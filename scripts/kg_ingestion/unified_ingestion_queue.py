#!/usr/bin/env python3
"""
Unified Triplet Ingestion Queue

This module provides a shared queue system for both process_jsons and extract_relations
to output triplets to a unified Knowledge Graph ingestion worker.

Architecture:
    process_jsons (parallel) → Unified Queue → KG Ingestion Worker → Memgraph + Property VDB
    extract_relations (parallel) → Unified Queue → KG Ingestion Worker → Memgraph + Property VDB
"""

import asyncio
import os
from typing import Dict, Any, Optional
from memgraph_utils import MemgraphConnection, write_triplet_to_memgraph
from init_property_vdb import add_property_to_vdb
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings


# Property VDB Configuration
PROPERTY_VDB_PATH = "../property_vdb"
PROPERTY_COLLECTION_NAME = "product_properties"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_property_type_from_relation(relation_name: str) -> str:
    """
    Extract property type from relation name.
    Example: HAS_COLOR -> color, HAS_BRAND -> brand, SOLD_BY -> store

    Args:
        relation_name: Relation name (e.g., HAS_COLOR)

    Returns:
        Property type (e.g., color)
    """
    # Special case for SOLD_BY
    if relation_name == "SOLD_BY":
        return "store"

    # For HAS_X patterns, extract X and convert to lowercase
    if relation_name.startswith("HAS_"):
        return relation_name[4:].lower()

    # Default: use last word in lowercase
    return relation_name.split('_')[-1].lower()


class UnifiedIngestionQueue:
    """
    Unified queue for triplet ingestion from multiple sources.
    """

    def __init__(self, use_property_vdb: bool = True):
        """
        Initialize the unified ingestion queue.

        Args:
            use_property_vdb: Whether to populate property VDB
        """
        self.queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.mg_conn = None
        self.property_vdb_collection = None
        self.property_embeddings = None
        self.use_property_vdb = use_property_vdb
        self.worker_task = None

        # Statistics
        self.stats = {
            "total_triplets": 0,
            "attribute_triplets": 0,
            "description_triplets": 0,
            "property_vdb_entries": 0,
            "errors": 0
        }

    async def initialize(self):
        """Initialize Memgraph connection and Property VDB."""
        print("="*80)
        print("UNIFIED INGESTION QUEUE - INITIALIZATION")
        print("="*80)

        # Initialize Memgraph
        print("Connecting to Memgraph...")
        self.mg_conn = MemgraphConnection()
        await self.mg_conn.connect()
        print("✓ Memgraph connected")

        # Initialize Property VDB if requested
        if self.use_property_vdb:
            print("Initializing Property VDB...")
            try:
                os.makedirs(PROPERTY_VDB_PATH, exist_ok=True)

                # Initialize embedding function
                self.property_embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )

                # Initialize ChromaDB client
                client = chromadb.PersistentClient(
                    path=PROPERTY_VDB_PATH,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True # <-- FIX: Changed to True to match init_property_vdb.py
                    )
                )

                # Get or create collection
                try:
                    self.property_vdb_collection = client.get_collection(name=PROPERTY_COLLECTION_NAME)
                    print(f"✓ Using existing Property VDB collection")
                except:
                    self.property_vdb_collection = client.create_collection(
                        name=PROPERTY_COLLECTION_NAME,
                        metadata={"description": "Property value embeddings"}
                    )
                    print(f"✓ Created new Property VDB collection")

            except Exception as e:
                print(f"⚠ Warning: Could not initialize Property VDB: {e}")
                self.property_vdb_collection = None
                self.property_embeddings = None

        print("="*80)

    async def start_worker(self):
        """Start the KG ingestion worker."""
        self.worker_task = asyncio.create_task(self._kg_ingestion_worker())
        print("[QUEUE] KG ingestion worker started")

    async def _kg_ingestion_worker(self):
        """Worker that pulls triplets from queue and writes to Memgraph + Property VDB."""
        # Check if dry-run mode is enabled
        dry_run = os.getenv("DEBUG_DRY_RUN") == "true"

        if dry_run:
            print("[WORKER] KG ingestion worker running in DRY-RUN mode (no database writes)")
        else:
            print("[WORKER] KG ingestion worker running...")

        async with await self.mg_conn.get_session() as session:
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    # Wait for triplet with timeout
                    try:
                        triplet_data = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue

                    triplet = triplet_data["triplet"]
                    source_type = triplet_data.get("source_type", "unknown")

                    # Write triplet to Memgraph (or simulate in dry-run mode)
                    if dry_run:
                        # DRY-RUN MODE: Just simulate success
                        success = True
                    else:
                        success = await write_triplet_to_memgraph(
                            session,
                            triplet,
                            match_product_node=True
                        )

                    if success:
                        self.stats["total_triplets"] += 1

                        if source_type == "attribute":
                            self.stats["attribute_triplets"] += 1
                        elif source_type == "description":
                            self.stats["description_triplets"] += 1

                        # Add to Property VDB if enabled (skip in dry-run mode)
                        if self.property_vdb_collection and self.property_embeddings and not dry_run:
                            try:
                                relation_type = triplet["relation"]["type"]
                                property_value = triplet["target"]["name"]
                                property_type = extract_property_type_from_relation(relation_type)

                                add_property_to_vdb(
                                    self.property_vdb_collection,
                                    self.property_embeddings,
                                    property_type=property_type,
                                    property_value=property_value,
                                    metadata={
                                        "product_id": triplet["source"].get("metadata", {}).get("product_id", ""),
                                        "relation": relation_type,
                                        "source_type": source_type
                                    }
                                )
                                self.stats["property_vdb_entries"] += 1
                            except Exception as e:
                                print(f"[WORKER] Warning: Could not add to Property VDB: {e}")

                    else:
                        self.stats["errors"] += 1

                    # Mark task as done
                    self.queue.task_done()

                except Exception as e:
                    print(f"[WORKER] Error processing triplet: {e}")
                    self.stats["errors"] += 1
                    self.queue.task_done()

        print("[WORKER] KG ingestion worker finished")

    async def add_triplet(self, triplet: Dict[str, Any], source_type: str = "unknown"):
        """
        Add a triplet to the ingestion queue.

        Args:
            triplet: Triplet dictionary (source, relation, target)
            source_type: Type of source ("attribute" or "description")
        """
        await self.queue.put({
            "triplet": triplet,
            "source_type": source_type
        })

    async def wait_for_completion(self):
        """Wait for all triplets in queue to be processed."""
        await self.queue.join()

    async def stop(self):
        """Signal the worker to stop and wait for it to finish."""
        print("\n[QUEUE] Stopping KG ingestion worker...")
        self.stop_event.set()

        if self.worker_task:
            await self.worker_task

        print("[QUEUE] Printing final statistics...")
        self.print_stats()

    async def cleanup(self):
        """Clean up resources."""
        if self.mg_conn:
            await self.mg_conn.close()
        print("[QUEUE] Cleanup complete")

    def print_stats(self):
        """Print ingestion statistics."""
        print("\n" + "="*80)
        print("UNIFIED INGESTION QUEUE - FINAL STATISTICS")
        print("="*80)
        print(f"Total triplets ingested:     {self.stats['total_triplets']}")
        print(f"  - Attribute triplets:      {self.stats['attribute_triplets']}")
        print(f"  - Description triplets:    {self.stats['description_triplets']}")
        print(f"Property VDB entries added:  {self.stats['property_vdb_entries']}")
        print(f"Errors:                      {self.stats['errors']}")
        print("="*80)


# Global queue instance
_global_queue: Optional[UnifiedIngestionQueue] = None


async def get_global_queue() -> UnifiedIngestionQueue:
    """Get or create the global ingestion queue."""
    global _global_queue
    if _global_queue is None:
        _global_queue = UnifiedIngestionQueue()
        await _global_queue.initialize()
        await _global_queue.start_worker()
    return _global_queue


async def shutdown_global_queue():
    """Shutdown the global ingestion queue."""
    global _global_queue
    if _global_queue is not None:
        await _global_queue.wait_for_completion()
        await _global_queue.stop()
        await _global_queue.cleanup()
        _global_queue = None