"""
Process Product JSON Files and Ingest into Memgraph Knowledge Graph

This script:
1. Reads product JSON files from a directory
2. For each product, extracts attributes (brand, colour, store, etc.)
3. [MODIFIED] Sends product/attribute triplets to the UnifiedIngestionQueue
4. Returns product info for subsequent description relation extraction
"""

import json
import os
import glob
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from memgraph_utils import MemgraphConnection, create_product_node, write_triplet_to_memgraph
from init_property_vdb import add_property_to_vdb
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
# Import the queue for type hinting
from unified_ingestion_queue import UnifiedIngestionQueue

# Property VDB Configuration
PROPERTY_VDB_PATH = "../property_vdb"
PROPERTY_COLLECTION_NAME = "product_properties"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Attribute to relation name mapping (removed size - goes to SQL DB)
ATTRIBUTE_TO_RELATION = {
    "brand": "HAS_BRAND",
    "colour": "HAS_COLOR",
    "store": "SOLD_BY"
}

# Mapping of attribute names to node types
ATTR_TYPE_MAPPING = {
    "brand": "manufacturer",
    "store": "seller",
    "colour": "colour"
}


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


def format_triplet(
    product_id: str,
    product_name: str,
    document_id: str,
    relation_type: str,
    target_name: str,
    target_type: str,
    metadata: Dict[str, Any],
    category: str  # <-- ADDED category
) -> Dict[str, Any]:
    """
    Format a triplet in the standardized format matching extract_relation_ingestion_debug.py.

    Args:
        product_id: Product ID (node name)
        product_name: Product name (in properties)
        document_id: Document ID
        relation_type: Relation type (HAS_BRAND, etc.)
        target_name: Target node name
        target_type: Target node type
        metadata: Additional metadata
        category: Product category (for node label)

    Returns:
        Formatted triplet dict
    """
    # Sanitize category for valid Cypher identifier
    sanitized_category = category.replace(' ', '_').replace('-', '_')

    return {
        "source": {
            "name": product_id,
            "type": sanitized_category, # <-- Use category label for consistency
            "metadata": {
                "product_id": product_id,
                "document_id": document_id,
                **metadata
            },
            "properties": {
                # [MODIFIED] Add all product data as properties for node creation
                "prod_name": product_name,
                "product_id": product_id,
                "document_id": document_id
            }
        },
        "relation": {
            "type": relation_type,
            "metadata": {
                "source": document_id
            }
        },
        "target": {
            "name": target_name,
            "type": target_type,
            "properties": {}
        }
    }


def extract_product_data(
    product_data: Dict[str, Any],
    metadata: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract product node data and attribute triplets from product JSON.

    Args:
        product_data: Product metadata
        metadata: Document metadata

    Returns:
        Tuple of (product_node_data, attribute_triplets)
    """
    # Handle both old and new field names
    product_id = product_data['product_id']
    product_name = product_data.get('prod_name') or product_data.get('product_name', '')
    category = product_data.get('category', 'Product')
    document_id = metadata.get('document_id', product_id)  # Use product_id if no document_id

    # --- Product Node Data ---
    # Only store prod_name and product_id as properties
    product_node_data = {
        "product_id": product_id,
        "prod_name": product_name,
        "category": category,  # Used for dynamic node label
        "document_id": document_id
    }

    # --- Attribute Triplets ---
    attribute_triplets = []

    for attr_name, relation_name in ATTRIBUTE_TO_RELATION.items():
        if attr_name in product_data and product_data[attr_name]:
            value = product_data[attr_name]
            node_type = ATTR_TYPE_MAPPING.get(attr_name, attr_name)

            triplet = format_triplet(
                product_id=product_id,
                product_name=product_name,
                document_id=document_id,
                relation_type=relation_name,
                target_name=str(value),
                target_type=node_type,
                metadata=metadata,
                category=category  # <-- Pass category
            )
            attribute_triplets.append(triplet)

    return product_node_data, attribute_triplets


async def process_product_json_file(
    filepath: str,
    session,
    property_vdb_collection=None,
    property_embeddings=None,
    queue: Optional[UnifiedIngestionQueue] = None  # <-- Add queue parameter
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Process a single product JSON file and write to Memgraph.

    Args:
        filepath: Path to JSON file
        session: Memgraph session
        property_vdb_collection: (Ignored)
        property_embeddings: (Ignored)
        queue: The UnifiedIngestionQueue instance

    Returns:
        Tuple of (products_count, triplets_count, product_info_list)
    """
    # Check if dry-run mode is enabled
    dry_run = os.getenv("DEBUG_DRY_RUN") == "true"

    print(f"\nProcessing: {os.path.basename(filepath)}")
    if dry_run:
        print("  [DRY-RUN MODE: Simulating operations]")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle JSON array format: [{...}] -> {...}
    if isinstance(data, list):
        if len(data) == 0:
            print(f"  ⚠ Skipping empty array in {os.path.basename(filepath)}")
            return 0, 0, []
        data = data[0]  # Extract first object from array

    products_count = 0
    triplets_count = 0
    product_info_list = []

    # Handle both old nested structure (chunks array) and new flat structure
    if 'chunks' in data:
        # Old format with chunks array
        items_to_process = [(chunk['product_data'], chunk.get('metadata', {}))
                           for chunk in data.get('chunks', [])
                           if 'product_data' in chunk]
    else:
        # New flat format - single product per file
        items_to_process = [(data, {})]

    for product_data, metadata in items_to_process:
        # Extract product and attribute data
        product_node_data, attribute_triplets = extract_product_data(
            product_data, metadata
        )

        # [MODIFIED] Node creation is now handled by the queue worker's MERGE logic
        # We just count the product.
        products_count += 1

        # [MODIFIED] Send triplets to the queue instead of writing directly
        for triplet in attribute_triplets:
            if queue:
                # Send to the unified queue
                await queue.add_triplet(triplet, source_type="attribute")
                triplets_count += 1
            else:
                # Fallback if no queue is provided (e.g., testing this file alone)
                print(f"  [WARN] No queue provided for {os.path.basename(filepath)}, skipping triplet ingestion.")
                if dry_run:
                    triplets_count += 1

        # Store product info for description extraction
        # Handle both 'description' and 'descrption' (typo in schema)
        description = product_data.get('description') or product_data.get('descrption', '')

        product_info_list.append({
            "product_id": product_node_data['product_id'],
            "product_name": product_node_data['prod_name'],
            "document_id": product_node_data['document_id'],
            "description": description
        })

        print(f"  ✓ Product: {product_node_data['prod_name']} ({len(attribute_triplets)} attributes queued)")

    return products_count, triplets_count, product_info_list


async def process_directory(
    directory_path: str,
    use_property_vdb: bool = True,
    queue: Optional[UnifiedIngestionQueue] = None  # <-- Add queue parameter
) -> List[Dict[str, Any]]:
    """
    Process all product JSON files in a directory and write to Memgraph.

    Args:
        directory_path: Path to directory containing product JSON files
        use_property_vdb: (Ignored) VDB logic is handled by queue
        queue: The UnifiedIngestionQueue instance

    Returns:
        List of product info dicts for description extraction
    """
    print("="*80)
    print("PRODUCT JSON PROCESSOR - MEMGRAPH INGESTION")
    print("="*80)
    print(f"Directory: {directory_path}\n")

    # Find all JSON files
    pattern = os.path.join(directory_path, "*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print(f"\n⚠ No JSON files found in: {directory_path}")
        return []

    print(f"Found {len(json_files)} JSON files\n")

    # [DELETED] VDB Initialization block removed.
    # This fixes the "different settings" warning.
    # The UnifiedIngestionQueue worker now handles all VDB writes.
    property_vdb_collection = None
    property_embeddings = None

    # Connect to Memgraph (still needed for the session context)
    mg_conn = MemgraphConnection()
    await mg_conn.connect()

    total_products = 0
    total_triplets = 0
    all_product_info = []

    try:
        # Process all files in parallel (with concurrency limit)
        MAX_CONCURRENT_FILES = 10  # Process up to 10 files at once
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)

        async def process_with_limit(filepath):
            """Process a single file with semaphore limit"""
            async with semaphore:
                # We still create a session, but it's not used
                # if the queue is active. This could be optimized,
                # but it's safer to leave for now.
                async with await mg_conn.get_session() as session:
                    return await process_product_json_file(
                        filepath,
                        session,
                        property_vdb_collection,
                        property_embeddings,
                        queue=queue  # <-- Pass the queue
                    )

        # Launch all file processing tasks in parallel
        print(f"Processing files with max {MAX_CONCURRENT_FILES} concurrent tasks...\n")
        results = await asyncio.gather(
            *[process_with_limit(filepath) for filepath in json_files],
            return_exceptions=True
        )

        # Aggregate results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"✗ Error processing {json_files[i]}: {result}")
                continue

            products, triplets, product_info = result
            total_products += products
            total_triplets += triplets
            all_product_info.extend(product_info)

        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Total products processed: {total_products}")
        print(f"Total attribute triplets queued: {total_triplets}")
        print(f"Products ready for description extraction: {len(all_product_info)}")
        print("="*80)

    finally:
        await mg_conn.close()

    return all_product_info


async def main():
    """Main entry point."""
    import sys
    
    # Get the queue for standalone testing
    from unified_ingestion_queue import get_global_queue, shutdown_global_queue
    queue = await get_global_queue()

    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."  # Current directory

    product_info_list = await process_directory(directory, queue=queue)

    # Print sample
    if product_info_list:
        print("\n" + "="*80)
        print("SAMPLE - First Product Info for Description Extraction")
        print("="*80)
        sample = product_info_list[0]
        print(f"Product ID: {sample['product_id']}")
        print(f"Product Name: {sample['product_name']}")
        print(f"Document ID: {sample['document_id']}")
        print(f"Description: {sample['description'][:150]}...")
        print("="*80)

    print("Shutting down queue...")
    await shutdown_global_queue()
    print("Standalone test complete.")

    return product_info_list


if __name__ == "__main__":
    asyncio.run(main())