"""
Configuration for Kurse Ecommerce Search Orchestrator - Test & Debug Version
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    def load_dotenv():
        return None

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for search orchestrator"""

    # ===== Base Directory =====

    BASE_DIR = Path(__file__).parent.parent.absolute()  # Project root (parent of config/)

    # ===== Database Connections =====

    # SQL Database
    SQL_DB_PATH: str = os.getenv(
        "SQL_DB_PATH",
        str(BASE_DIR / "data" / "databases" / "sql" / "inventory.db")
    )

    # Memgraph (Knowledge Graph)
    MEMGRAPH_URI: str = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
    MEMGRAPH_USER: str = os.getenv("MEMGRAPH_USER", "")
    MEMGRAPH_PASS: str = os.getenv("MEMGRAPH_PASS", "")

    # Vector Database Paths
    MAIN_VDB_PATH: str = os.getenv(
        "MAIN_VDB_PATH",
        str(BASE_DIR / "data" / "databases" / "vector_db" / "Main Vector_Database")
    )

    PROPERTY_VDB_PATH: str = os.getenv(
        "PROPERTY_VDB_PATH",
        str(BASE_DIR / "data" / "databases" / "vector_db" / "property_vdb")
    )

    RELATION_VDB_PATH: str = os.getenv(
        "RELATION_VDB_PATH",
        str(BASE_DIR / "data" / "databases" / "vector_db" / "relation_vdb")
    )

    # VDB Collection Names
    MAIN_VDB_COLLECTION: str = "my_multimodal_product_collection"  # Qdrant collection
    PROPERTY_VDB_COLLECTION: str = "ecommerce_properties"
    RELATION_VDB_COLLECTION: str = "ecommerce_relations"

    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ===== LLM Configuration =====

    # NVIDIA API Configuration
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY")
    NVIDIA_MODEL: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    USE_NVIDIA_LLM: bool = os.getenv("USE_NVIDIA_LLM", "True").lower() == "true"

    # ===== User Context Configuration =====

    # Directory containing user profiles and purchase history
    USER_CONTEXT_DATA_DIR: str = os.getenv(
        "USER_CONTEXT_DATA_DIR",
        str(BASE_DIR / "data" / "user_data")
    )

    # User profile and purchase history file paths
    USER_PROFILES_FILE: str = os.getenv(
        "USER_PROFILES_FILE",
        str(BASE_DIR / "data" / "user_data" / "user_profiles.json")
    )

    PURCHASE_HISTORY_FILE: str = os.getenv(
        "PURCHASE_HISTORY_FILE",
        str(BASE_DIR / "data" / "user_data" / "purchase_history.json")
    )

    # ===== Caching Configuration =====

    # Enable caching for LLM responses and KG queries
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "True").lower() == "true"

    # Cache TTL (time to live) in seconds
    LLM_CACHE_TTL: int = int(os.getenv("LLM_CACHE_TTL", "3600"))  # 1 hour
    KG_CACHE_TTL: int = int(os.getenv("KG_CACHE_TTL", "3600"))    # 1 hour

    # ===== Scoring Parameters =====

    # Connected Search Bonus Scores
    CONNECTED_BONUS_SCORE: float = 0.5  # Bonus for products connected via KG
    STORE_BONUS_SCORE: float = 0.2      # Bonus for products from same store

    # Buffer Penalty
    LINEAR_PENALTY_RATE: float = 0.1    # Penalty per 10% buffer violation

    # Minimum Score Threshold
    MIN_SCORE_THRESHOLD: float = 0.0    # Filter products below this score

    # Subcategory Scoring Parameters
    ENABLE_SUBCATEGORY_SCORING: bool = True  # Enable subcategory similarity scoring
    SUBCATEGORY_THRESHOLD: float = 0.5       # Minimum similarity for bonus
    SUBCATEGORY_MAX_BONUS: float = 0.4       # Maximum bonus score (at similarity=1.0)


    # ===== VDB Query Parameters =====

    # Top-K limits for VDB queries
    PROPERTY_VDB_TOP_K: int = 5         # Top similar properties
    RELATION_VDB_TOP_K: int = 3         # Top similar relations
    MAIN_VDB_TOP_K: int = 20            # Top similar products from Main VDB

    # ===== KG Query Parameters =====

    # Relation types for connected search (traversal)
    CONNECTED_SEARCH_RELATIONS: list = [
        "ALSO_BOUGHT",
        "PART_OF_SET",
        "SIMILAR_TO",
        "COMPLEMENTS",
        "HAS_COLOR",
        "HAS_STYLE",
        "HAS_MATERIAL",
        "HAS_BRAND",
        "SUITABLE_FOR"
    ]

    # ===== Literal Operators =====

    SUPPORTED_OPERATORS: dict = {
        "<": "less than",
        "<=": "less than or equal",
        ">": "greater than",
        ">=": "greater than or equal",
        "=": "equal",
        "!=": "not equal"
    }

    # ===== Async Settings =====

    ASYNC_TIMEOUT: int = 30  # Timeout for async operations (seconds)

    # ===== Debug Settings =====

    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ===== Vendor / Nunchi Flow Settings =====

    # Registration
    VENDOR_REGISTRATION_URL: str = os.getenv(
        "VENDOR_REGISTRATION_URL", "https://vendor.example.com/register"
    )
    VENDOR_REGISTRY_FILE: str = os.getenv(
        "VENDOR_REGISTRY_FILE",
        str(BASE_DIR / "data" / "vendor_registry.json")
    )

    # Session handling
    VENDOR_SESSION_LOCK_SECONDS: int = int(
        os.getenv("VENDOR_SESSION_LOCK_SECONDS", "45")
    )
    VENDOR_SESSION_LOCK_MAX_SECONDS: int = int(
        os.getenv("VENDOR_SESSION_LOCK_MAX_SECONDS", "60")
    )
    VENDOR_SESSION_LOCK_MIN_SECONDS: int = int(
        os.getenv("VENDOR_SESSION_LOCK_MIN_SECONDS", "30")
    )

    # Similarity thresholds
    VENDOR_SIMILARITY_THRESHOLD: float = float(
        os.getenv("VENDOR_SIMILARITY_THRESHOLD", "0.82")
    )
    VENDOR_UPDATE_MIN_SIMILARITY: float = float(
        os.getenv("VENDOR_UPDATE_MIN_SIMILARITY", "0.65")
    )

    # Intake queue
    INTAKE_QUEUE_TABLE: str = os.getenv("INTAKE_QUEUE_TABLE", "intake_queue")


# Singleton instance
config = Config()