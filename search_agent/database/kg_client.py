"""
Knowledge Graph Client using Memgraph
"""
from typing import List, Dict, Optional, Set
from neo4j import GraphDatabase, AsyncGraphDatabase
import asyncio

from search_agent.models import KGNode, KGRelation
from config.config import Config

config = Config


class KGClient:
    """Client for Knowledge Graph operations using Memgraph"""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Initialize KG client

        Args:
            uri: Memgraph URI
            user: Username
            password: Password
        """
        self.uri = uri or config.MEMGRAPH_URI
        self.user = user or config.MEMGRAPH_USER
        self.password = password or config.MEMGRAPH_PASS

        # Initialize driver
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password) if self.user else None
        )

    def close(self):
        """Close driver connection"""
        if self.driver:
            self.driver.close()

    def query_products_by_properties(
        self,
        category: str,
        property_names: List[str],
        relation_types: List[str],
        property_similarity_scores: Dict[str, float]
    ) -> List[Dict]:
        """
        Query products that have specific properties via specific relations

        Args:
            category: Product category label for strict filtering (e.g., "clothing")
            property_names: List of property names (e.g., ["Color:Red", "Style:Casual"])
            relation_types: List of relation types (e.g., ["HAS_COLOR", "HAS_STYLE"])
            property_similarity_scores: Mapping of property_name to VDB similarity score

        Returns:
            List of dictionaries with product_id and similarity score
            [
                {"product_id": "p-123", "property": "Color:Red", "similarity": 0.95},
                ...
            ]

        Note:
            Category is matched against node labels (e.g., :Product:Clothing)
            Subcategory embeddings are retrieved from SQL, not KG
        """
        with self.driver.session() as session:
            # Build Cypher query
            # MATCH (p:Product)-[r]->(prop:Property)
            # WHERE $category IN labels(p)  -- Filter by category label
            #   AND type(r) IN $relation_types
            #   AND prop.name IN $property_names  -- Now expects ["red", "blue", ...] not ["Color:Red", ...]
            # RETURN p.id, prop.name, prop.type

            query = """
            MATCH (p)-[r]->(prop:Property)
            WHERE $category IN labels(p)
              AND type(r) IN $relation_types
              AND prop.name IN $property_names
            RETURN p.product_id AS product_id, prop.name AS property_name, prop.type AS property_type
            """

            result = session.run(
                query,
                category=category,
                relation_types=relation_types,
                property_names=property_names
            )

            # Parse results and attach similarity scores
            products = []
            for record in result:
                product_id = record["product_id"]
                property_name = record["property_name"]
                property_type = record["property_type"]
                similarity = property_similarity_scores.get(property_name, 0.0)

                products.append({
                    "product_id": product_id,
                    "property": property_name,
                    "property_type": property_type,
                    "similarity": similarity
                })

            return products

    def query_connected_products(
        self,
        source_product_id: str,
        category: str,
        relation_types: List[str] = None
    ) -> Set[str]:
        """
        Query products connected to a source product via specific relations

        Args:
            source_product_id: Source product ID
            category: Target product category label for strict filtering (e.g., "clothing")
            relation_types: List of relation types to traverse (if None, use all)

        Returns:
            Set of connected product IDs

        Note:
            Category is matched against node labels (e.g., :Product:Clothing)
        """
        with self.driver.session() as session:
            if relation_types:
                # Use specific relation types
                query = """
                MATCH (source:Product {id: $source_id})-[r]->(target:Product)
                WHERE $category IN labels(target)
                  AND type(r) IN $relation_types
                RETURN DISTINCT target.id AS product_id
                """
                result = session.run(
                    query,
                    source_id=source_product_id,
                    category=category,
                    relation_types=relation_types
                )
            else:
                # Use all relation types
                query = """
                MATCH (source:Product {id: $source_id})-[r]->(target:Product)
                WHERE $category IN labels(target)
                RETURN DISTINCT target.id AS product_id
                """
                result = session.run(
                    query,
                    source_id=source_product_id,
                    category=category
                )

            # Return set of product IDs
            return {record["product_id"] for record in result}

    def query_product_connection_count(
        self,
        product_ids: List[str]
    ) -> Dict[str, int]:
        """
        Count the number of connections (degree) for each product

        Args:
            product_ids: List of product IDs

        Returns:
            Dictionary mapping product_id to connection count
            {"p-123": 5, "p-456": 3}
        """
        with self.driver.session() as session:
            query = """
            MATCH (p)-[r]->()
            WHERE p.product_id IN $product_ids
            RETURN p.product_id AS product_id, count(r) AS connection_count
            """

            result = session.run(query, product_ids=product_ids)

            return {
                record["product_id"]: record["connection_count"]
                for record in result
            }

    def get_all_properties_for_product(
        self,
        product_id: str
    ) -> List[Dict]:
        """
        Get all properties connected to a product

        Args:
            product_id: Product ID

        Returns:
            List of property dictionaries
            [
                {"property_name": "Color:Red", "relation": "HAS_COLOR"},
                ...
            ]
        """
        with self.driver.session() as session:
            query = """
            MATCH (p:Product {id: $product_id})-[r]->(prop:Property)
            RETURN prop.name AS property_name, type(r) AS relation_type
            """

            result = session.run(query, product_id=product_id)

            return [
                {
                    "property_name": record["property_name"],
                    "relation": record["relation_type"]
                }
                for record in result
            ]

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
