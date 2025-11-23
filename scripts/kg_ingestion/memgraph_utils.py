#!/usr/bin/env python3
"""
Shared utilities for Memgraph knowledge graph ingestion.
Provides connection management and write operations for nodes and relationships.
"""

import os
from typing import Dict, List, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
import asyncio


# Memgraph connection configuration
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "")
MEMGRAPH_PASS = os.getenv("MEMGRAPH_PASS", "")


class MemgraphConnection:
    """Manages connection to Memgraph database."""

    def __init__(self):
        self.driver: Optional[AsyncDriver] = None

    async def connect(self):
        """Establish connection to Memgraph."""
        if self.driver is None:
            self.driver = AsyncGraphDatabase.driver(
                MEMGRAPH_URI,
                auth=(MEMGRAPH_USER, MEMGRAPH_PASS) if MEMGRAPH_USER else None
            )
        return self.driver

    async def close(self):
        """Close the connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None

    async def get_session(self) -> AsyncSession:
        """Get a new session."""
        if not self.driver:
            await self.connect()
        return self.driver.session()


async def create_or_merge_node(
    session: AsyncSession,
    name: str,
    node_type: str,
    metadata: Dict[str, Any],
    properties: Dict[str, Any],
    merge_on_name: bool = True
) -> Dict[str, Any]:
    """
    Create or merge a node in Memgraph.

    Args:
        session: Memgraph session
        name: Node name (unique identifier)
        node_type: Node type/label
        metadata: Metadata dict
        properties: Properties dict
        merge_on_name: If True, use MERGE (avoid duplicates), else CREATE

    Returns:
        Dictionary with node info
    """
    # Combine all properties for the node
    all_props = {
        "name": name,
        **metadata,
        **properties
    }
    
    # Sanitize node_type for Cypher by wrapping in backticks
    # This is safer than the string replacement done in other scripts
    safe_node_type = f"`{node_type}`"

    if merge_on_name:
        # MERGE on name to avoid duplicates
        query = f"""
        MERGE (n:{safe_node_type} {{name: $name}})
        ON CREATE SET n += $props
        ON MATCH SET n += $props
        RETURN n
        """
    else:
        # CREATE new node
        query = f"""
        CREATE (n:{safe_node_type} $props)
        RETURN n
        """

    result = await session.run(query, name=name, props=all_props)
    record = await result.single()

    return {
        "name": name,
        "type": node_type,
        "created": True
    }


async def create_relationship(
    session: AsyncSession,
    source_name: str,
    relation_type: str,
    target_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create or merge a relationship between two nodes.

    Args:
        session: Memgraph session
        source_name: Source node name
        relation_type: Relationship type
        target_name: Target node name
        metadata: Optional relationship metadata

    Returns:
        Dictionary with relationship info
    """
    metadata = metadata or {}
    
    # [FIX] Wrap the relation_type in backticks (`)
    # This allows for relation types with spaces, like "HAS_DEVICE WARRANTY"
    safe_relation_type = f"`{relation_type}`"

    # MERGE relationship to avoid duplicates
    query = f"""
    MATCH (s {{name: $source_name}})
    MATCH (t {{name: $target_name}})
    MERGE (s)-[r:{safe_relation_type}]->(t)
    ON CREATE SET r += $metadata
    RETURN r
    """

    try:
        result = await session.run(
            query,
            source_name=source_name,
            target_name=target_name,
            metadata=metadata
        )
        await result.consume()

        return {
            "source": source_name,
            "relation": relation_type,
            "target": target_name,
            "created": True
        }
    except Exception as e:
        print(f"Error creating relationship: {e}")
        return {
            "source": source_name,
            "relation": relation_type,
            "target": target_name,
            "created": False,
            "error": str(e)
        }


async def write_triplet_to_memgraph(
    session: AsyncSession,
    triplet: Dict[str, Any],
    match_product_node: bool = False
) -> bool:
    """
    Write a complete triplet (source-relation-target) to Memgraph.

    Args:
        session: Memgraph session
        triplet: Triplet dict with source, relation, target
        match_product_node: If True, MATCH product node instead of MERGE
                           (assumes it already exists from process_json)

    Returns:
        True if successful, False otherwise
    """
    try:
        source = triplet["source"]
        relation = triplet["relation"]
        target = triplet["target"]

        # Create/merge source node
        if match_product_node and source["type"] in ["product", "Clothing", "Electronics", "Furniture"]:
            # Product node should already exist, just MATCH it
            # We must use the backticked label to match
            safe_node_type = f"`{source['type']}`"
            query = f"""
            MATCH (s:{safe_node_type} {{name: $name}})
            SET s += $props
            RETURN s
            """
            await session.run(
                query,
                name=source["name"],
                props={**source.get("properties", {})}
            )
        else:
            # MERGE other nodes
            await create_or_merge_node(
                session,
                name=source["name"],
                node_type=source["type"],
                metadata=source.get("metadata", {}),
                properties=source.get("properties", {})
            )

        # Create/merge target node
        await create_or_merge_node(
            session,
            name=target["name"],
            node_type=target["type"],
            metadata=target.get("metadata", {}),
            properties=target.get("properties", {})
        )

        # Create relationship
        await create_relationship(
            session,
            source_name=source["name"],
            relation_type=relation["type"],
            target_name=target["name"],
            metadata=relation.get("metadata", {})
        )

        return True

    except Exception as e:
        print(f"Error writing triplet: {e}")
        return False


async def batch_write_triplets(
    session: AsyncSession,
    triplets: List[Dict[str, Any]],
    match_product_node: bool = False
) -> int:
    """
    Write multiple triplets to Memgraph in batch.

    Args:
        session: Memgraph session
        triplets: List of triplet dicts
        match_product_node: If True, MATCH product nodes instead of MERGE

    Returns:
        Number of successfully written triplets
    """
    success_count = 0

    for triplet in triplets:
        if await write_triplet_to_memgraph(session, triplet, match_product_node):
            success_count += 1

    return success_count


async def create_product_node(
    session: AsyncSession,
    product_id: str,
    prod_name: str,
    category: str
) -> Dict[str, Any]:
    """
    Create a product node with category-based label.

    Args:
        session: Memgraph session
        product_id: Product ID (unique identifier, stored as 'name' property)
        prod_name: Product name
        category: Category (used as node label, e.g., :Clothing, :Electronics)

    Returns:
        Dictionary with product node info
    """
    # Only store product_id and prod_name as properties
    all_props = {
        "name": product_id,
        "prod_name": prod_name
    }

    # Use category as node label for better filtering and querying
    # Sanitize category to ensure it's a valid Cypher identifier
    sanitized_category = category.replace(' ', '_').replace('-', '_')
    # Wrap in backticks for safety
    safe_node_type = f"`{sanitized_category}`"


    query = f"""
    MERGE (p:{safe_node_type} {{name: $product_id}})
    ON CREATE SET p += $props
    ON MATCH SET p += $props
    RETURN p
    """

    result = await session.run(query, product_id=product_id, props=all_props)
    await result.consume()

    return {
        "name": product_id,
        "type": sanitized_category,
        "prod_name": prod_name
    }