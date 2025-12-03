#!/usr/bin/env python3
"""
Knowledge Graph Pipeline - Main Orchestrator

Runs the complete end-to-end pipeline:
1. Initializes VDBs (custom_relations_vdb, property_vdb)
2. Starts the Unified Ingestion Queue
3. Processes product JSONs (attributes) -> sends to Queue
4. Extracts relations from descriptions -> sends to Queue
5. Shuts down queue, finalizing all writes to Memgraph
6. Exports to deployment package

Usage:
    python run_pipeline.py <input_directory>

    Example:
    python run_pipeline.py ./Fashion_Products
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify NVIDIA API key
if not (os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEY_1")):
    print("ERROR: NVIDIA_API_KEY or NVIDIA_API_KEY_1 not found in environment!")
    print("Please set keys in .env file (e.g., NVIDIA_API_KEY_1=your_key)")
    sys.exit(1)

# Import pipeline modules
from init_property_vdb import init_property_vdb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Use the global queue helpers
from unified_ingestion_queue import get_global_queue, shutdown_global_queue
from process_product_jsons import process_directory as process_jsons_directory
# Import the correct function from the AI script
from extract_relation_ingestion import extract_relations_from_directory
from export_and_package import create_deployment_package
from progress_tracker import print_progress_summary, mark_phase_complete

# Custom Relations VDB Configuration
CUSTOM_RELATIONS_VDB_PATH = os.path.join(os.path.dirname(__file__), "custom_relations_vdb")
CUSTOM_RELATIONS_COLLECTION_NAME = "ecommerce_relations"


async def initialize_vdbs():
    """Load existing VDBs (non-destructive)."""
    print("\n" + "="*80)
    print("STEP 1: LOADING VECTOR DATABASES")
    print("="*80)

    # Load existing Custom Relations VDB (do NOT reinitialize from seeds)
    print("\n[1/2] Loading Custom Relations VDB...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        custom_vdb = Chroma(
            persist_directory=CUSTOM_RELATIONS_VDB_PATH,
            embedding_function=embeddings,
            collection_name=CUSTOM_RELATIONS_COLLECTION_NAME
        )
        count = custom_vdb._collection.count()
        print(f"Custom Relations VDB ready ({count} relations)")
    except Exception as e:
        print(f"Error loading Custom Relations VDB: {e}")
        raise

    # Initialize Property VDB (non-destructive, uses get_or_create)
    print("\n[2/2] Loading Property VDB...")
    try:
        property_vdb, property_embeddings = init_property_vdb()
        print("Property VDB ready")
    except Exception as e:
        print(f"Error loading Property VDB: {e}")
        raise

    print("\nAll VDBs loaded successfully")
    print("="*80)


async def run_pipeline(input_directory: str, skip_export: bool = False):
    """
    Run the complete knowledge graph pipeline.

    Args:
        input_directory: Directory containing product JSON files
        skip_export: If True, skip the export step
    """
    # Validate input directory
    if not os.path.isdir(input_directory):
        print(f"ERROR: '{input_directory}' is not a valid directory")
        sys.exit(1)

    input_path = os.path.abspath(input_directory)
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH PIPELINE - STARTING")
    print("="*80)
    print(f"Input Directory: {input_path}")
    print(f"Number of JSON files: {len([f for f in os.listdir(input_path) if f.endswith('.json')])}")
    print("="*80)

    # Show progress from previous run (if any)
    print_progress_summary()

    try:
        # Step 1: Initialize VDBs
        await initialize_vdbs()

        # Step 2: Initialize Unified Ingestion Queue
        print("\n" + "="*80)
        print("STEP 2: STARTING UNIFIED INGESTION QUEUE")
        print("="*80)

        # Get the global queue instance. This also starts its worker.
        ingestion_queue = await get_global_queue()
        print("✓ Ingestion queue ready")
        print("="*80)

        # Step 3: Process Product JSONs (Attributes)
        print("\n" + "="*80)
        print("STEP 3: PROCESSING PRODUCT ATTRIBUTES")
        print("="*80)

        # Pass the queue to Step 3
        # This fixes the concurrency errors by sending all writes to the queue
        product_info_list = await process_jsons_directory(
            input_path,
            use_property_vdb=True, # This is now ignored, queue handles VDB
            queue=ingestion_queue
        )
        print(f"✓ Queued {len(product_info_list)} products for attribute ingestion")
        mark_phase_complete("attributes_complete")
        print("="*80)

        # Step 4: Extract Relations from Descriptions
        print("\n" + "="*80)
        print("STEP 4: EXTRACTING RELATIONS FROM DESCRIPTIONS")
        print("="*80)

        # Pass the queue to Step 4
        # This fixes the "Zero Ingestion" bug by sending data to the queue
        total_triplets = await extract_relations_from_directory(
            input_path,
            queue=ingestion_queue
        )
        print(f"✓ Queued {total_triplets} relation triplets for ingestion")
        mark_phase_complete("relations_complete")
        print("="*80)

        # Step 5: Wait for queue to finish and stop
        print("\n" + "="*80)
        print("STEP 5: FINALIZING INGESTION")
        print("="*80)

        # Shut down the global queue, which waits for all items
        # and prints final stats.
        await shutdown_global_queue()
        print("="*80)

        # Step 6: Export to deployment package
        if not skip_export:
            print("\n" + "="*80)
            print("STEP 6: CREATING DEPLOYMENT PACKAGE")
            print("="*80)

            output_zip = "kg_deployment_package.zip"
            success = await create_deployment_package(output_zip)

            if success:
                print(f"\n✓ Deployment package created: {os.path.abspath(output_zip)}")
            else:
                print("\n⚠ Deployment package creation had issues")

            print("="*80)

        # Pipeline complete
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH PIPELINE - COMPLETE")
        print("="*80)
        print("✓ All data ingested to Memgraph")
        print("✓ Property VDB populated")
        print("✓ Custom Relations VDB updated")
        if not skip_export:
            print(f"✓ Deployment package: {output_zip}")
        print("="*80)
        print("\nYou can now:")
        print("1. Query Memgraph: MATCH (n) RETURN count(n);")
        print("2. Check VDBs in: custom_relations_vdb/ and property_vdb/")
        if not skip_export:
            print(f"3. Deploy using: {output_zip}")
        print("="*80)

    except Exception as e:
        print(f"\n✗ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# [DELETED]
# The local extract_relations_from_directory function was removed.
# The script now correctly imports and uses the one from
# extract_relation_ingestion.py


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the complete Knowledge Graph ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py ./Fashion_Products
  python run_pipeline.py ../data/products --skip-export
        """
    )

    parser.add_argument(
        "input_directory",
        help="Directory containing product JSON files"
    )

    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip creating the deployment package ZIP"
    )

    args = parser.parse_args()

    # Run pipeline
    asyncio.run(run_pipeline(args.input_directory, args.skip_export))


if __name__ == "__main__":
    main()