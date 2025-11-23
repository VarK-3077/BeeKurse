"""
Caching layer for Strontium
Provides LLM response caching and KG property caching
"""
import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class LLMCache:
    """
    Cache for LLM parsing responses
    Reduces latency and cost by caching common queries
    """

    def __init__(self, ttl: int = 3600):
        """
        Initialize LLM cache

        Args:
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached LLM response for query

        Args:
            query: User query string

        Returns:
            Cached parsed output dict, or None if not found/expired
        """
        cache_key = self._hash_query(query)

        if cache_key not in self._cache:
            return None

        cached_entry = self._cache[cache_key]

        # Check if expired
        if self._is_expired(cached_entry['timestamp']):
            del self._cache[cache_key]
            return None

        return cached_entry['parsed_output']

    def set(self, query: str, parsed_output: Dict[str, Any]):
        """
        Cache LLM response

        Args:
            query: User query string
            parsed_output: Parsed LLM output as dict
        """
        cache_key = self._hash_query(query)

        self._cache[cache_key] = {
            'parsed_output': parsed_output,
            'timestamp': datetime.now()
        }

    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()

    def _hash_query(self, query: str) -> str:
        """Hash query for cache key"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - timestamp > timedelta(seconds=self.ttl)


class KGPropertyCache:
    """
    Cache for Knowledge Graph product properties
    Reduces KG query load for frequently accessed products
    """

    def __init__(self, kg_client=None, ttl: int = 3600):
        """
        Initialize KG property cache

        Args:
            kg_client: Knowledge graph client for fetching properties
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.kg_client = kg_client
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_product_properties(self, product_id: str) -> List[str]:
        """
        Get all properties for a product (with caching)

        Args:
            product_id: Product ID

        Returns:
            List of property names
        """
        # Check cache first
        if product_id in self._cache:
            cached_entry = self._cache[product_id]

            # Check if expired
            if not self._is_expired(cached_entry['timestamp']):
                return cached_entry['properties']
            else:
                del self._cache[product_id]

        # Cache miss - fetch from KG
        if self.kg_client:
            properties = self._fetch_from_kg(product_id)
        else:
            # No KG client - return empty list
            properties = []

        # Cache the result
        self._cache[product_id] = {
            'properties': properties,
            'timestamp': datetime.now()
        }

        return properties

    def _fetch_from_kg(self, product_id: str) -> List[str]:
        """
        Fetch product properties from Knowledge Graph

        Args:
            product_id: Product ID

        Returns:
            List of property names
        """
        if not self.kg_client:
            return []

        # Query KG for all properties
        query = """
        MATCH (p:Product {id: $product_id})-[r]->(prop:Property)
        RETURN type(r) as relation, prop.name as property
        """

        try:
            result = self.kg_client.execute_read(query, {"product_id": product_id})

            # Extract property names
            properties = []
            for record in result:
                prop_name = record.get('property')
                if prop_name:
                    # Extract just the value (e.g., "Color:Red" -> "red")
                    if ':' in prop_name:
                        value = prop_name.split(':', 1)[1].lower()
                        properties.append(value)
                    else:
                        properties.append(prop_name.lower())

            return properties

        except Exception as e:
            print(f"Error fetching properties for {product_id}: {e}")
            return []

    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()

    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - timestamp > timedelta(seconds=self.ttl)


class CacheManager:
    """
    Unified cache manager for Strontium
    Manages both LLM and KG caches
    """

    def __init__(self, kg_client=None, llm_ttl: int = 3600, kg_ttl: int = 3600):
        """
        Initialize cache manager

        Args:
            kg_client: Knowledge graph client
            llm_ttl: LLM cache TTL in seconds
            kg_ttl: KG cache TTL in seconds
        """
        self.llm_cache = LLMCache(ttl=llm_ttl)
        self.kg_cache = KGPropertyCache(kg_client=kg_client, ttl=kg_ttl)

    def clear_all(self):
        """Clear all caches"""
        self.llm_cache.clear()
        self.kg_cache.clear()
