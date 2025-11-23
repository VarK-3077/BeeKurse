"""
Vector Database Client using Chroma and Qdrant
"""
from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
import requests
import os

from search_agent.models import VDBResult
from config.config import Config

config = Config


# Global shared embedding model instance (singleton pattern)
_SHARED_EMBEDDING_MODEL: Optional[HuggingFaceEmbeddings] = None
_SHARED_MODEL_NAME: Optional[str] = None


def get_shared_embedding_model(model_name: str = None) -> HuggingFaceEmbeddings:
    """
    Get or create shared embedding model instance.

    This singleton pattern ensures the heavy embedding model is only loaded once
    and reused across all VDB clients, reducing initialization time from ~16s to ~6s.

    Args:
        model_name: HuggingFace model name (defaults to config.EMBEDDING_MODEL)

    Returns:
        Shared HuggingFaceEmbeddings instance
    """
    global _SHARED_EMBEDDING_MODEL, _SHARED_MODEL_NAME

    target_model = model_name or config.EMBEDDING_MODEL

    # Create model if doesn't exist or model name changed
    if _SHARED_EMBEDDING_MODEL is None or _SHARED_MODEL_NAME != target_model:
        if config.DEBUG:
            print(f"[VDB] Loading embedding model: {target_model}...")
        _SHARED_EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=target_model)
        _SHARED_MODEL_NAME = target_model
        if config.DEBUG:
            print(f"[VDB] Embedding model loaded successfully")

    return _SHARED_EMBEDDING_MODEL


class ChromaEmbeddingAdapter(EmbeddingFunction):
    """Adapter to make HuggingFaceEmbeddings compatible with ChromaDB"""

    def __init__(self, hf_embeddings: HuggingFaceEmbeddings):
        self._embeddings = hf_embeddings
        self._model_name = hf_embeddings.model_name

    def __call__(self, input: Documents) -> List[List[float]]:
        """Generate embeddings for documents"""
        return self._embeddings.embed_documents(input)


class VDBClient:
    """Client for Vector Database operations using Chroma"""

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        embedding_model: str = None
    ):
        """
        Initialize VDB client

        Args:
            db_path: Path to Chroma database directory
            collection_name: Name of the collection
            embedding_model: HuggingFace embedding model name
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL

        # Ensure directory exists
        Path(db_path).mkdir(parents=True, exist_ok=True)

        # Get shared embedding model (singleton pattern for performance)
        hf_embeddings = get_shared_embedding_model(self.embedding_model_name)
        self.embedding_function = hf_embeddings  # Keep for embed_query access

        # Wrap for ChromaDB compatibility
        chroma_embeddings = ChromaEmbeddingAdapter(hf_embeddings)

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection - handle existing collections
        try:
            # Try to get existing collection first (without embedding function to avoid conflicts)
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            # Collection doesn't exist, create it with embedding function
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=chroma_embeddings
            )

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Dict] = None
    ) -> List[VDBResult]:
        """
        Search vector database

        Args:
            query: Search query string
            top_k: Number of results to return
            where: Optional metadata filter (e.g., {"basetype": "shirt"})

        Returns:
            List of VDBResult objects
        """
        # Perform query
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where
        )

        # Parse results
        vdb_results = []
        if results and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                vdb_result = VDBResult(
                    id=results['ids'][0][i],
                    similarity=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                vdb_results.append(vdb_result)

        return vdb_results

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None
    ):
        """
        Add documents to vector database

        Args:
            ids: List of document IDs
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
        """
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def delete(self, ids: List[str]):
        """Delete documents by IDs"""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Get count of documents in collection"""
        return self.collection.count()


class MainVDBClient:
    """Client for Main Product VDB using Qdrant with NVIDIA nvclip embeddings"""

    def __init__(self):
        """Initialize Qdrant client for main product VDB"""
        self.db_path = config.MAIN_VDB_PATH
        self.collection_name = config.MAIN_VDB_COLLECTION

        # NVIDIA API configuration for nvclip
        self.nvidia_api_key = config.NVIDIA_API_KEY
        self.nvidia_api_url = "https://integrate.api.nvidia.com/v1/embeddings"

        if not self.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY not found in config. Required for nvclip embeddings.")

        # Initialize Qdrant client
        self.client = QdrantClient(path=self.db_path)

        if config.DEBUG:
            print(f"[MainVDB] Initialized Qdrant client at {self.db_path}")
            print(f"[MainVDB] Collection: {self.collection_name}")
            print(f"[MainVDB] Using NVIDIA nvclip for embeddings")

    def _get_nvclip_embedding(self, text: str) -> List[float]:
        """
        Get embedding from NVIDIA nvclip API

        Args:
            text: Text to embed

        Returns:
            List of embedding values (1024-dimensional)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.nvidia_api_key}"
        }

        payload = {
            "input": [text],
            "model": "nvidia/nvclip",
            "encoding_format": "float"
        }

        try:
            response = requests.post(self.nvidia_api_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            # Extract embedding from response
            if result.get("data") and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            else:
                raise ValueError(f"Unexpected response format from NVIDIA API: {result}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get embedding from NVIDIA nvclip API: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to parse embedding from NVIDIA API response: {e}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Dict] = None
    ) -> List[VDBResult]:
        """
        Search vector database using text vector

        Args:
            query: Search query string
            top_k: Number of results to return
            where: Optional metadata filter (e.g., {"category": "clothing"})

        Returns:
            List of VDBResult objects
        """
        # Generate query embedding using NVIDIA nvclip
        query_vector = self._get_nvclip_embedding(query)

        # Build filter if provided
        query_filter = None
        if where:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                for key, value in where.items()
            ]
            query_filter = Filter(must=conditions)

        # Query using text vector (Qdrant's query method for named vectors)
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="text",  # Specify which named vector to use
            query_filter=query_filter,
            limit=top_k
        ).points

        # Convert to VDBResult
        vdb_results = []
        for result in search_results:
            vdb_result = VDBResult(
                id=result.payload.get('product_id', str(result.id)),  # Use payload product_id (with PID- prefix)
                similarity=result.score,
                metadata=result.payload or {}
            )
            vdb_results.append(vdb_result)

        return vdb_results

    def search_products(
        self,
        category: str,
        subcategory: str,
        property_query: str,
        top_k: int = None
    ) -> List[VDBResult]:
        """
        Search for products with specific property

        Args:
            category: Product category for strict filtering (e.g., "clothing")
                     Use "other" to skip category filtering
            subcategory: Product subcategory for semantic matching (e.g., "shirt", "polo shirt")
            property_query: Property to search (e.g., "red color")
            top_k: Number of results

        Returns:
            List of VDBResult with product_id and similarity scores
        """
        # Construct query: include subcategory for semantic matching
        query = f"{subcategory} {property_query}"

        # Filter by category in metadata (strict filter)
        # Skip filter if category is "other" (allows searching across all categories)
        where_filter = {"category": category} if category != "other" else None

        if config.DEBUG:
            print(f"\n[DEBUG] Main VDB search_products():")
            print(f"  Query: '{query}'")
            print(f"  Category filter: {where_filter}")
            print(f"  Top K: {top_k or config.MAIN_VDB_TOP_K}")

        # Search
        results = self.search(
            query=query,
            top_k=top_k or config.MAIN_VDB_TOP_K,
            where=where_filter
        )

        if config.DEBUG:
            print(f"  Results returned: {len(results)}")
            if not results and where_filter:
                print(f"  ⚠️ Category filter '{category}' returned 0 results!")
            elif results:
                print(f"  Sample results:")
                for i, r in enumerate(results[:3]):
                    print(f"    {i+1}. ID: {r.id[:40]}..., Similarity: {r.similarity:.3f}")
                    if r.metadata:
                        print(f"       Category: {r.metadata.get('category', 'N/A')}, Name: {r.metadata.get('prod_name', 'N/A')[:50]}")

        return results

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for arbitrary text using NVIDIA nvclip

        Args:
            text: Text to embed

        Returns:
            List of embedding values (1024-dimensional)
        """
        return self._get_nvclip_embedding(text)

    def count(self) -> int:
        """Get count of documents in collection"""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count


class PropertyVDBClient(VDBClient):
    """Client for Property VDB"""

    def __init__(self):
        super().__init__(
            db_path=config.PROPERTY_VDB_PATH,
            collection_name=config.PROPERTY_VDB_COLLECTION
        )

    def search_properties(
        self,
        property_query: str,
        top_k: int = None
    ) -> List[VDBResult]:
        """
        Search for similar properties

        Args:
            property_query: Property to search (e.g., "red")
            top_k: Number of results

        Returns:
            List of VDBResult with property names (e.g., "Color:Red") and similarity scores
        """
        return self.search(
            query=property_query,
            top_k=top_k or config.PROPERTY_VDB_TOP_K
        )


class RelationVDBClient(VDBClient):
    """Client for Relation VDB"""

    def __init__(self):
        super().__init__(
            db_path=config.RELATION_VDB_PATH,
            collection_name=config.RELATION_VDB_COLLECTION
        )

    def search_relations(
        self,
        relation_query: str,
        top_k: int = None
    ) -> List[VDBResult]:
        """
        Search for similar relations

        Args:
            relation_query: Relation to search (e.g., "has color")
            top_k: Number of results

        Returns:
            List of VDBResult with relation names (e.g., "HAS_COLOR") and similarity scores
        """
        return self.search(
            query=relation_query,
            top_k=top_k or config.RELATION_VDB_TOP_K
        )
