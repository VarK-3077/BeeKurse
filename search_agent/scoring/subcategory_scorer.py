"""
Subcategory Similarity Scorer

Scores products based on subcategory similarity using embeddings.
Provides additive bonus scores for products with similar subcategories.
"""
import numpy as np
from typing import List, Dict, Optional

from search_agent.database.sql_client import SQLClient
from search_agent.database.vdb_client import MainVDBClient
from config.config import Config

config = Config


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

    # Calculate cosine similarity
    similarity = np.dot(vec1_norm, vec2_norm)

    # Clip to [0, 1] range
    return max(0.0, min(1.0, float(similarity)))


class SubcategoryScorer:
    """
    Scores products based on subcategory similarity.

    Uses pre-computed subcategory embeddings to calculate similarity
    between target subcategory and product subcategories, providing
    an additive bonus score.
    """

    def __init__(
        self,
        sql_client: SQLClient,
        main_vdb_client: MainVDBClient
    ):
        """
        Initialize subcategory scorer

        Args:
            sql_client: SQL database client for embedding retrieval
            main_vdb_client: VDB client for embedding generation
        """
        self.sql_client = sql_client
        self.main_vdb_client = main_vdb_client

    def score_products(
        self,
        product_ids: List[str],
        target_subcategory: str,
        target_embedding: Optional[np.ndarray] = None,
        threshold: float = None,
        max_bonus: float = None
    ) -> Dict[str, float]:
        """
        Score products by subcategory similarity

        Args:
            product_ids: List of product IDs to score
            target_subcategory: Target subcategory text (e.g., "polo shirt")
            target_embedding: Pre-computed embedding for target (optional, for performance)
            threshold: Minimum similarity to award bonus (default: from config)
            max_bonus: Maximum bonus score at similarity=1.0 (default: from config)

        Returns:
            Dictionary mapping product_id to subcategory bonus score
            {
                "p-001": 0.32,  # similarity=0.8 * max_bonus=0.4
                "p-004": 0.24,  # similarity=0.6 * max_bonus=0.4
            }
        """
        if not product_ids:
            return {}

        # Check if scoring is enabled
        if not config.ENABLE_SUBCATEGORY_SCORING:
            return {pid: 0.0 for pid in product_ids}

        # Use config defaults if not specified
        threshold = threshold if threshold is not None else config.SUBCATEGORY_THRESHOLD
        max_bonus = max_bonus if max_bonus is not None else config.SUBCATEGORY_MAX_BONUS

        # Generate target embedding if not provided
        if target_embedding is None:
            target_embedding_list = self.main_vdb_client.embed_text(target_subcategory)
            target_embedding = np.array(target_embedding_list)

        # Get product embeddings from SQL
        product_embeddings = self.sql_client.get_embeddings(product_ids)

        # Calculate similarity scores
        scores = {}
        for product_id in product_ids:
            # Check if embedding exists for this product
            if product_id not in product_embeddings:
                # No embedding stored - skip or score as 0
                scores[product_id] = 0.0
                continue

            product_emb = product_embeddings[product_id]

            # Calculate cosine similarity
            similarity = cosine_similarity(target_embedding, product_emb)

            # Apply threshold and calculate bonus
            if similarity >= threshold:
                # Scale bonus by similarity (linear)
                bonus = similarity * max_bonus
                scores[product_id] = bonus
            else:
                scores[product_id] = 0.0

        return scores

    def embed_subcategory(self, subcategory: str) -> np.ndarray:
        """
        Generate embedding for a subcategory

        Args:
            subcategory: Subcategory text

        Returns:
            Numpy array of embedding values
        """
        embedding_list = self.main_vdb_client.embed_text(subcategory)
        return np.array(embedding_list)
