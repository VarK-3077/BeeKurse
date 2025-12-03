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
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
# Import the queue for type hinting
from unified_ingestion_queue import UnifiedIngestionQueue
# Import progress tracker for incremental processing
from progress_tracker import get_completed_attribute_files, mark_attribute_file_complete

# Property VDB Configuration
PROPERTY_VDB_PATH = "../property_vdb"
PROPERTY_COLLECTION_NAME = "product_properties"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Attribute to relation name mapping (removed size - goes to SQL DB)
ATTRIBUTE_TO_RELATION = {
    "brand": "HAS_BRAND",
    "colour": "HAS_COLOR",
    "store": "SOLD_BY",
    "gender": "FOR_GENDER"
}

# Mapping of attribute names to node types
ATTR_TYPE_MAPPING = {
    "brand": "manufacturer",
    "store": "seller",
    "colour": "colour",
    "gender": "gender"
}

# Category-specific attribute mappings for other_properties
# Format: field_name: (relation_type, node_type, condition_fn, transform_fn)
# - condition_fn: If provided, only add triplet if condition_fn(value) is True
# - transform_fn: If provided, transform value before adding
CATEGORY_ATTRIBUTE_MAPPINGS = {
    "electronics": {
        "warranty_years": ("HAS_WARRANTY", "warranty", lambda v: v and v > 0, lambda v: f"{v} years"),
        "energy_rating": ("HAS_ENERGY_RATING", "energy_rating", lambda v: v and v != "Not Rated", None),
    },
    "fashion": {
        "gender": ("FOR_GENDER", "gender", None, None),
        "season": ("HAS_SEASONALITY", "season", None, None),
        "usage": ("HAS_OCCASION_TYPE", "occasion", None, None),
        "occasion": ("HAS_OCCASION_TYPE", "occasion", None, None),
        "fabric": ("HAS_MATERIAL", "material", None, None),
        "fit": ("HAS_FIT", "fit", None, None),
        "neck": ("HAS_NECK_TYPE", "neck_type", None, None),
        "pattern": ("HAS_PATTERN", "pattern", None, None),
        "print_pattern": ("HAS_PATTERN", "pattern", None, None),
        "sleeve_length": ("HAS_SLEEVE_LENGTH", "sleeve_length", None, None),
    },
    "grocery": {
        "organic": ("IS_ORGANIC", "organic", lambda v: v is True, lambda v: "Yes"),
        "locally_sourced": ("IS_LOCALLY_SOURCED", "locally_sourced", lambda v: v is True, lambda v: "Yes"),
        "ripeness": ("HAS_RIPENESS", "ripeness", None, None),
        "fat_content": ("HAS_FAT_CONTENT", "fat_content", None, None),
    }
}

# New relations that need to be added to custom_relations_vdb
NEW_RELATIONS_TO_ADD = {
    "HAS_ENERGY_RATING": {
        "description": "Energy efficiency rating of electronic products",
        "category": "electronics"
    },
    "IS_ORGANIC": {
        "description": "Whether the product is organically produced",
        "category": "food"
    },
    "IS_LOCALLY_SOURCED": {
        "description": "Whether the product is locally sourced",
        "category": "food"
    },
    "HAS_RIPENESS": {
        "description": "Ripeness level of produce",
        "category": "food"
    },
    "HAS_FAT_CONTENT": {
        "description": "Fat content level of food products",
        "category": "food"
    },
    "HAS_NECK_TYPE": {
        "description": "Neckline type of clothing",
        "category": "fashion"
    },
    "HAS_SLEEVE_LENGTH": {
        "description": "Sleeve length of clothing",
        "category": "fashion"
    }
}


def ensure_relations_in_vdb(vdb_path: str = "./custom_relations_vdb"):
    """
    Ensure new category-specific relations exist in the custom_relations_vdb.
    Only adds relations that don't already exist (similarity < 0.98).

    Args:
        vdb_path: Path to the custom relations VDB
    """
    print("\nChecking custom_relations_vdb for new relations...")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load existing VDB
    vdb = Chroma(
        persist_directory=vdb_path,
        embedding_function=embeddings,
        collection_name="ecommerce_relations"
    )

    added_count = 0
    for relation_name, relation_data in NEW_RELATIONS_TO_ADD.items():
        # Check if relation already exists (similarity > 0.98 means exact match)
        results = vdb.similarity_search_with_relevance_scores(relation_name, k=1)

        if results and results[0][1] > 0.98:
            print(f"  [skip] {relation_name} already exists")
            continue

        # Add to VDB
        doc = Document(
            page_content=relation_name,
            metadata={
                "relation": relation_name,
                "description": relation_data["description"],
                "category": relation_data["category"],
                "type": "seed_relation"
            }
        )
        vdb.add_documents([doc])
        print(f"  [added] {relation_name}")
        added_count += 1

    print(f"Added {added_count} new relations to custom_relations_vdb\n")
    return added_count


def detect_category_from_filename(filepath: str) -> str:
    """
    Detect product category from filename prefix.

    Args:
        filepath: Path to the JSON file

    Returns:
        Category string: "electronics", "fashion", "grocery", or "unknown"
    """
    if not filepath:
        return "unknown"
    basename = os.path.basename(filepath).lower()
    if basename.startswith("electronics_"):
        return "electronics"
    elif basename.startswith("fashion_"):
        return "fashion"
    elif basename.startswith("grocery_"):
        return "grocery"
    return "unknown"


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
    metadata: Dict[str, Any],
    filepath: str = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract product node data and attribute triplets from product JSON.

    Args:
        product_data: Product metadata
        metadata: Document metadata
        filepath: Path to source file (used to detect category)

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

    # Extract top-level attributes
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
                category=category
            )
            attribute_triplets.append(triplet)

    # --- Extract other_properties based on category ---
    category_type = detect_category_from_filename(filepath)
    other_props = product_data.get("other_properties", {})
    category_mappings = CATEGORY_ATTRIBUTE_MAPPINGS.get(category_type, {})

    for field, (relation_name, node_type, condition_fn, transform_fn) in category_mappings.items():
        value = other_props.get(field)
        if value is None:
            continue

        # Check condition (e.g., skip organic=false, skip "Not Rated")
        if condition_fn and not condition_fn(value):
            continue

        # Transform value if needed (e.g., 3 -> "3 years", True -> "Yes")
        final_value = transform_fn(value) if transform_fn else str(value)

        triplet = format_triplet(
            product_id=product_id,
            product_name=product_name,
            document_id=document_id,
            relation_type=relation_name,
            target_name=final_value,
            target_type=node_type,
            metadata=metadata,
            category=category
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
            product_data, metadata, filepath=filepath
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

    # Load already completed files for resumability
    completed_files = get_completed_attribute_files()
    files_to_process = [f for f in json_files if os.path.basename(f) not in completed_files]

    print(f"Found {len(json_files)} JSON files total")
    print(f"Already processed: {len(completed_files)} files")
    print(f"Remaining to process: {len(files_to_process)} files\n")

    # Ensure new category-specific relations are in custom_relations_vdb
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_vdb_path = os.path.join(script_dir, "custom_relations_vdb")
    ensure_relations_in_vdb(custom_vdb_path)

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
        # Process files sequentially so we can track progress for resumability
        print(f"Processing files sequentially for progress tracking...\n")

        for filepath in files_to_process:
            try:
                async with await mg_conn.get_session() as session:
                    result = await process_product_json_file(
                        filepath,
                        session,
                        property_vdb_collection,
                        property_embeddings,
                        queue=queue
                    )

                products, triplets, product_info = result
                total_products += products
                total_triplets += triplets
                all_product_info.extend(product_info)

                # Mark file as complete for resumability
                mark_attribute_file_complete(os.path.basename(filepath))

            except Exception as e:
                print(f"✗ Error processing {filepath}: {e}")
                continue

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