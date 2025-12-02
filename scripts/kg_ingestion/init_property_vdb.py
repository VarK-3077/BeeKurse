#!/usr/bin/env python3
"""
Initialize Property Value Vector Database

This script creates a ChromaDB vector database to store property value embeddings
for use in the curator algorithm and similarity search.

Property Format: {property_type}:{value}
Examples:
  - color:red
  - brand:LoomNest
  - store:Aara Boutique

Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings.
"""

import os
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
PROPERTY_VDB_PATH = "../property_vdb"
COLLECTION_NAME = "product_properties"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def init_property_vdb(vdb_path: str = PROPERTY_VDB_PATH):
    """
    Initialize the property value vector database.

    Args:
        vdb_path: Path to store the ChromaDB database

    Returns:
        ChromaDB collection object
    """
    print("="*80)
    print("PROPERTY VALUE VDB INITIALIZATION")
    print("="*80)
    print(f"VDB Path: {vdb_path}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print("="*80)

    # Create directory if it doesn't exist
    os.makedirs(vdb_path, exist_ok=True)

    # Initialize embedding function
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✓ Embedding model loaded")

    # Initialize ChromaDB client
    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(
        path=vdb_path,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Get or create collection (non-destructive)
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Property value embeddings for curator algorithm"}
        )
        count = collection.count()
        print(f"Collection '{COLLECTION_NAME}' ready ({count} existing embeddings)")

    except Exception as e:
        print(f"Error accessing collection: {e}")
        raise

    print("\n" + "="*80)
    print("INITIALIZATION COMPLETE")
    print("="*80)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Location: {os.path.abspath(vdb_path)}")
    print("="*80)
    print("\nThe property VDB is ready to receive property value embeddings.")
    print("Property values will be added during product processing.")
    print("="*80)

    return collection, embeddings


def add_property_to_vdb(
    collection,
    embeddings,
    property_type: str,
    property_value: str,
    metadata: dict = None
):
    """
    Add a property value to the VDB.

    Args:
        collection: ChromaDB collection
        embeddings: Embedding function
        property_type: Type of property (e.g., "color", "brand", "store")
        property_value: Value of property (e.g., "red", "LoomNest")
        metadata: Optional additional metadata

    Returns:
        True if successful, False otherwise
    """
    try:
        # Format: {property_type}:{value}
        property_string = f"{property_type}:{property_value}"

        # Generate embedding
        embedding = embeddings.embed_query(property_string)

        # Prepare metadata
        meta = {
            "property_type": property_type,
            "property_value": property_value,
            "property_string": property_string
        }
        if metadata:
            meta.update(metadata)

        # Add to collection
        collection.add(
            embeddings=[embedding],
            documents=[property_string],
            metadatas=[meta],
            ids=[f"{property_type}_{property_value}_{hash(property_string)}"]
        )

        return True

    except Exception as e:
        print(f"Error adding property to VDB: {e}")
        return False


def query_similar_properties(
    collection,
    embeddings,
    property_string: str,
    top_k: int = 10
):
    """
    Find similar properties in the VDB.

    Args:
        collection: ChromaDB collection
        embeddings: Embedding function
        property_string: Property to search for (format: "type:value")
        top_k: Number of similar properties to return

    Returns:
        List of similar properties with distances
    """
    try:
        # Generate embedding for query
        query_embedding = embeddings.embed_query(property_string)

        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results

    except Exception as e:
        print(f"Error querying VDB: {e}")
        return None


def main():
    """Main entry point for testing."""
    import sys

    # Initialize VDB
    collection, embeddings = init_property_vdb()

    # Add some test properties if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\n" + "="*80)
        print("TESTING - Adding Sample Properties")
        print("="*80)

        test_properties = [
            ("color", "red"),
            ("color", "blue"),
            ("color", "black"),
            ("brand", "Nike"),
            ("brand", "Adidas"),
            ("brand", "LoomNest"),
            ("store", "Aara Boutique"),
            ("store", "Amazon"),
        ]

        for prop_type, prop_value in test_properties:
            success = add_property_to_vdb(collection, embeddings, prop_type, prop_value)
            if success:
                print(f"  ✓ Added: {prop_type}:{prop_value}")

        # Test query
        print("\n" + "="*80)
        print("TESTING - Querying Similar Properties")
        print("="*80)
        print("Query: color:crimson")

        results = query_similar_properties(collection, embeddings, "color:crimson", top_k=3)
        if results:
            print("\nTop 3 similar properties:")
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                print(f"  {i+1}. {doc} (distance: {distance:.4f})")

        print("="*80)


if __name__ == "__main__":
    main()
