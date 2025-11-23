"""
Initialize Custom Relations VDB for Ecommerce Products

This script:
1. Loads seed relations from seed_relations.json
2. Creates a Chroma vector database collection
3. Adds seed relations as documents
4. Verifies the VDB is working correctly
"""

import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configuration
SEED_RELATIONS_FILE = "./seed_relations.json"
CUSTOM_RELATIONS_VDB_PATH = "./custom_relations_vdb"
COLLECTION_NAME = "ecommerce_relations"

def load_seed_relations(filepath):
    """Load seed relations from JSON file."""
    print(f"Loading seed relations from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Seed relations file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"  ✓ Loaded {data['metadata']['total_relations']} seed relations")
    print(f"  ✓ Categories: {', '.join(data['metadata']['categories_analyzed'])}")

    return data['seed_relations']


def create_vdb(seed_relations, vdb_path, collection_name, embeddings=None):
    """Create and populate the custom relations VDB."""
    print(f"\nInitializing Chroma VDB at: {vdb_path}")

    # Initialize embeddings if not provided
    if embeddings is None:
        print("  Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("  ✓ Embedding model loaded")
    else:
        print("  Using provided embedding model...")

    # Prepare documents from seed relations
    print("\n  Preparing seed documents...")
    documents = []

    for relation_name, relation_data in seed_relations.items():
        # Skip comment keys
        if relation_name.startswith("_comment"):
            continue

        # Create document content
        # Store JUST the relation name as primary content for better matching
        # The description and examples go in metadata for reference
        content = relation_name

        # Create document
        doc = Document(
            page_content=content,
            metadata={
                "relation": relation_name,
                "description": relation_data['description'],
                "category": relation_data.get('category', 'unknown'),
                "examples": json.dumps(relation_data.get('examples', [])),
                "type": "seed_relation"
            }
        )
        documents.append(doc)

    print(f"  ✓ Prepared {len(documents)} seed documents")

    # Create Chroma collection
    print("\n  Creating Chroma collection...")
    vdb = Chroma(
        persist_directory=vdb_path,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    # Add documents
    print("  Adding seed relations to VDB...")
    vdb.add_documents(documents)
    print(f"  ✓ Added {len(documents)} relations to VDB")

    return vdb


def verify_vdb(vdb):
    """Verify the VDB is working correctly with test queries."""
    print("\n" + "="*80)
    print("VERIFICATION: Testing VDB with sample queries")
    print("="*80)

    test_queries = [
        ("refresh rate", "HAS_REFRESH_RATE"),
        ("price", "HAS_PRICE"),
        ("color", "HAS_COLOR"),
        ("material", "HAS_MATERIAL"),
        ("diet type", "HAS_DIET_TYPE"),
        ("storage capacity", "HAS_STORAGE"),
        ("fabric composition", "HAS_MATERIAL_COMPOSITION"),
        ("assembly needed", "HAS_ASSEMBLY_REQUIRED")
    ]

    all_passed = True

    for query, expected_relation in test_queries:
        results = vdb.similarity_search(query, k=3)

        if results:
            top_result = results[0].metadata.get('relation', 'UNKNOWN')
            found = top_result == expected_relation

            status = "✓" if found else "✗"
            print(f"\n{status} Query: '{query}'")
            print(f"  Expected: {expected_relation}")
            print(f"  Got:      {top_result}")

            if not found:
                print(f"  Top 3 matches:")
                for i, r in enumerate(results[:3], 1):
                    print(f"    {i}. {r.metadata.get('relation', 'UNKNOWN')}")
                all_passed = False
        else:
            print(f"\n✗ Query: '{query}' - No results!")
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - VDB is ready!")
    else:
        print("⚠ SOME TESTS FAILED - Review results above")
    print("="*80)

    return all_passed


def init_custom_relations_vdb():
    """
    Initialize custom relations VDB for import by pipeline.

    Returns:
        tuple: (vdb, embeddings) - ChromaDB instance and embedding function
    """
    # Load seed relations
    seed_relations = load_seed_relations(SEED_RELATIONS_FILE)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create VDB (this also adds the seed relations)
    vdb = create_vdb(seed_relations, CUSTOM_RELATIONS_VDB_PATH, COLLECTION_NAME, embeddings)

    return vdb, embeddings


def main():
    """Main initialization workflow."""
    print("="*80)
    print("CUSTOM RELATIONS VDB INITIALIZATION")
    print("="*80 + "\n")

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")

    try:
        # Step 1: Load seed relations
        seed_relations = load_seed_relations(SEED_RELATIONS_FILE)

        # Step 2: Create VDB
        vdb = create_vdb(seed_relations, CUSTOM_RELATIONS_VDB_PATH, COLLECTION_NAME)

        # Step 3: Verify VDB
        verify_vdb(vdb)

        # Step 4: Summary
        print("\n" + "="*80)
        print("INITIALIZATION COMPLETE")
        print("="*80)
        print(f"VDB Location: {os.path.abspath(CUSTOM_RELATIONS_VDB_PATH)}")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Seed Relations: {len([k for k in seed_relations.keys() if not k.startswith('_comment')])}")
        print("\nNext steps:")
        print("  1. Test VDB with your own queries")
        print("  2. Integrate into extract_relations pipeline")
        print("  3. Process ecommerce products")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
