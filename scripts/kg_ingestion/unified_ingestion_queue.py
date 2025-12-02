#!/usr/bin/env python3
"""
Unified Triplet Ingestion Queue

This module provides a shared queue system for both process_jsons and extract_relations
to output triplets to a unified Knowledge Graph ingestion worker.

Architecture:
    process_jsons (parallel) → Unified Queue → KG Ingestion Worker → Memgraph
    extract_relations (parallel) → Unified Queue → KG Ingestion Worker → Memgraph

Note: Property VDB population is handled separately in extract_relation_ingestion.py
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from memgraph_utils import MemgraphConnection, write_triplet_to_memgraph


class UnifiedIngestionQueue:
    """
    Unified queue for triplet ingestion from multiple sources.
    """

    def __init__(self):
        """Initialize the unified ingestion queue."""
        self.queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.mg_conn = None
        self.worker_task = None

        # Store all triplets for JSON backup
        self.all_triplets: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            "total_triplets": 0,
            "attribute_triplets": 0,
            "description_triplets": 0,
            "errors": 0
        }

    async def initialize(self):
        """Initialize Memgraph connection."""
        print("="*80)
        print("UNIFIED INGESTION QUEUE - INITIALIZATION")
        print("="*80)

        # Initialize Memgraph
        print("Connecting to Memgraph...")
        self.mg_conn = MemgraphConnection()
        await self.mg_conn.connect()
        print("Memgraph connected")

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
        triplet_data = {
            "triplet": triplet,
            "source_type": source_type
        }
        # Store for JSON backup
        self.all_triplets.append(triplet_data)
        await self.queue.put(triplet_data)

    async def wait_for_completion(self):
        """Wait for all triplets in queue to be processed."""
        await self.queue.join()

    async def stop(self):
        """Signal the worker to stop and wait for it to finish."""
        print("\n[QUEUE] Stopping KG ingestion worker...")
        self.stop_event.set()

        if self.worker_task:
            await self.worker_task

        # Save triplets to JSON backup
        self.save_triplets_to_json()

        print("[QUEUE] Printing final statistics...")
        self.print_stats()

    def save_triplets_to_json(self):
        """Save all triplets to a JSON file for backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"triplets_backup_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": timestamp,
                    "total_count": len(self.all_triplets),
                    "triplets": self.all_triplets
                }, f, indent=2, ensure_ascii=False)
            print(f"[QUEUE] Saved {len(self.all_triplets)} triplets to {filename}")
        except Exception as e:
            print(f"[QUEUE] Error saving triplets to JSON: {e}")

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