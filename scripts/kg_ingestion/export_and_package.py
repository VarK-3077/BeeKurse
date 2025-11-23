#!/usr/bin/env python3
"""
Export and Package Knowledge Graph for Cloud Deployment

This script:
1. Exports Memgraph data to Cypher query format
2. Packages custom_relations_vdb directory
3. Packages property_vdb directory
4. Creates a compressed zip file with all artifacts

The zip file can be given to your teammate to host on cloud.
"""

import os
import asyncio
import zipfile
import shutil
from datetime import datetime
from pathlib import Path
from memgraph_utils import MemgraphConnection

# Configuration
EXPORT_DIR = "../export"
CUSTOM_RELATIONS_VDB_PATH = "../custom_relations_vdb"
PROPERTY_VDB_PATH = "../property_vdb"
OUTPUT_ZIP_NAME = "kg_deployment_package.zip"


async def export_memgraph_to_cypher(output_file: str):
    """
    Export all Memgraph data to Cypher queries.

    Args:
        output_file: Path to output Cypher file

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print("EXPORTING MEMGRAPH DATA")
    print("="*80)

    try:
        # Connect to Memgraph
        mg_conn = MemgraphConnection()
        await mg_conn.connect()

        cypher_queries = []

        async with await mg_conn.get_session() as session:
            # Export all nodes
            print("Exporting nodes...")
            node_query = """
            MATCH (n)
            RETURN labels(n) as labels, properties(n) as props
            """
            result = await session.run(node_query)
            records = await result.data()

            node_count = 0
            for record in records:
                labels = record['labels']
                props = record['props']

                if labels and props:
                    label = labels[0]  # Take first label
                    # Create MERGE query for node
                    props_str = ", ".join([f"{k}: {repr(v)}" for k, v in props.items()])
                    cypher_queries.append(
                        f"MERGE (n:{label} {{{props_str}}});"
                    )
                    node_count += 1

            print(f"✓ Exported {node_count} nodes")

            # Export all relationships
            print("Exporting relationships...")
            rel_query = """
            MATCH (s)-[r]->(t)
            RETURN labels(s) as source_labels, properties(s) as source_props,
                   type(r) as rel_type, properties(r) as rel_props,
                   labels(t) as target_labels, properties(t) as target_props
            """
            result = await session.run(rel_query)
            records = await result.data()

            rel_count = 0
            for record in records:
                source_label = record['source_labels'][0] if record['source_labels'] else "Node"
                target_label = record['target_labels'][0] if record['target_labels'] else "Node"
                rel_type = record['rel_type']

                source_props = record['source_props']
                target_props = record['target_props']
                rel_props = record['rel_props']

                # Get name property for matching
                source_name = source_props.get('name', '')
                target_name = target_props.get('name', '')

                if source_name and target_name:
                    # Create MERGE query for relationship
                    rel_props_str = ""
                    if rel_props:
                        props_items = ", ".join([f"{k}: {repr(v)}" for k, v in rel_props.items()])
                        rel_props_str = f" {{{props_items}}}"

                    cypher_queries.append(
                        f"MATCH (s:{source_label} {{name: {repr(source_name)}}}), "
                        f"(t:{target_label} {{name: {repr(target_name)}}}) "
                        f"MERGE (s)-[r:{rel_type}{rel_props_str}]->(t);"
                    )
                    rel_count += 1

            print(f"✓ Exported {rel_count} relationships")

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("// Memgraph Knowledge Graph Export\n")
            f.write(f"// Generated: {datetime.now().isoformat()}\n")
            f.write(f"// Total Nodes: {node_count}\n")
            f.write(f"// Total Relationships: {rel_count}\n\n")

            for query in cypher_queries:
                f.write(query + "\n")

        await mg_conn.close()

        print(f"✓ Cypher export saved to: {output_file}")
        print("="*80)
        return True

    except Exception as e:
        print(f"✗ Error exporting Memgraph: {e}")
        print("="*80)
        return False


def package_directory(dir_path: str, zip_file: zipfile.ZipFile, arcname: str):
    """
    Add a directory to zip file.

    Args:
        dir_path: Path to directory to package
        zip_file: ZipFile object
        arcname: Name in archive

    Returns:
        Number of files added
    """
    if not os.path.exists(dir_path):
        print(f"⚠ Directory not found: {dir_path}")
        return 0

    file_count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Calculate archive name preserving directory structure
            rel_path = os.path.relpath(file_path, os.path.dirname(dir_path))
            zip_file.write(file_path, arcname=rel_path)
            file_count += 1

    return file_count


async def create_deployment_package(output_zip: str):
    """
    Create deployment package with all artifacts.

    Args:
        output_zip: Path to output zip file

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print("CREATING DEPLOYMENT PACKAGE")
    print("="*80)

    try:
        # Create export directory
        os.makedirs(EXPORT_DIR, exist_ok=True)

        # Export Memgraph data
        cypher_file = os.path.join(EXPORT_DIR, "memgraph_export.cypher")
        success = await export_memgraph_to_cypher(cypher_file)

        if not success:
            print("⚠ Memgraph export failed, but continuing with VDB packaging...")

        # Create zip file
        print("\nPackaging artifacts...")
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add Memgraph export
            if os.path.exists(cypher_file):
                zipf.write(cypher_file, arcname="memgraph_export.cypher")
                print(f"✓ Added Memgraph export")

            # Add custom_relations_vdb
            print("Packaging custom_relations_vdb...")
            if os.path.exists(CUSTOM_RELATIONS_VDB_PATH):
                file_count = package_directory(
                    CUSTOM_RELATIONS_VDB_PATH,
                    zipf,
                    "custom_relations_vdb"
                )
                print(f"✓ Added custom_relations_vdb ({file_count} files)")
            else:
                print(f"⚠ custom_relations_vdb not found at: {CUSTOM_RELATIONS_VDB_PATH}")

            # Add property_vdb
            print("Packaging property_vdb...")
            if os.path.exists(PROPERTY_VDB_PATH):
                file_count = package_directory(
                    PROPERTY_VDB_PATH,
                    zipf,
                    "property_vdb"
                )
                print(f"✓ Added property_vdb ({file_count} files)")
            else:
                print(f"⚠ property_vdb not found at: {PROPERTY_VDB_PATH}")

            # Add README
            readme_content = f"""# Knowledge Graph Deployment Package

Generated: {datetime.now().isoformat()}

## Contents

1. **memgraph_export.cypher** - Cypher queries to recreate the knowledge graph
2. **custom_relations_vdb/** - ChromaDB vector database with relation types
3. **property_vdb/** - ChromaDB vector database with property value embeddings

## Deployment Instructions

### 1. Import Memgraph Data

```bash
# Option A: Using mgconsole (Memgraph CLI)
cat memgraph_export.cypher | mgconsole --host <host> --port <port>

# Option B: Using Python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://<host>:<port>", auth=("<user>", "<password>"))
with driver.session() as session:
    with open("memgraph_export.cypher", "r") as f:
        queries = f.read().split(';')
        for query in queries:
            if query.strip():
                session.run(query)
```

### 2. Deploy Vector Databases

Copy the VDB directories to your application server:

```bash
# Copy to your application directory
cp -r custom_relations_vdb /path/to/app/
cp -r property_vdb /path/to/app/
```

Update your application configuration to point to these paths.

### 3. Verify Deployment

```cypher
# Check node count
MATCH (n) RETURN count(n);

# Check relationship count
MATCH ()-[r]->() RETURN count(r);

# Sample query
MATCH (p:Clothing)-[:HAS_BRAND]->(b)
RETURN p.prod_name, b.name
LIMIT 10;
```

## Graph Schema

### Node Labels
- Category-based labels (e.g., `:Clothing`, `:Electronics`, `:Furniture`)
  - Properties: `name` (product_id), `prod_name`

- `:manufacturer` - Brand/manufacturer nodes
  - Properties: `name`

- `:colour` - Color attribute nodes
  - Properties: `name`

- `:seller` - Store/seller nodes
  - Properties: `name`

- `:property` - Generic property values (from descriptions)
  - Properties: `name`

### Relationships
- `HAS_BRAND` - Product to manufacturer
- `HAS_COLOR` - Product to colour
- `SOLD_BY` - Product to seller
- Dynamic relations from descriptions (e.g., `HAS_PROCESSOR`, `HAS_MATERIAL`, etc.)

## Vector Databases

### custom_relations_vdb
- Contains embeddings of relation types
- Used for relation standardization during extraction
- Model: sentence-transformers/all-MiniLM-L6-v2

### property_vdb
- Contains embeddings of property values (format: "type:value")
- Examples: "color:red", "brand:LoomNest", "store:Aara Boutique"
- Used for curator algorithm and similarity search
- Model: sentence-transformers/all-MiniLM-L6-v2

## Support

For issues or questions, contact the data engineering team.
"""

            zipf.writestr("README.md", readme_content)
            print("✓ Added README.md")

        print("\n" + "="*80)
        print("PACKAGING COMPLETE")
        print("="*80)
        print(f"Package saved to: {os.path.abspath(output_zip)}")
        print(f"Package size: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")
        print("="*80)
        print("\nYou can now send this zip file to your teammate for cloud deployment!")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n✗ Error creating package: {e}")
        print("="*80)
        return False


async def main():
    """Main entry point."""
    import sys

    # Get output filename
    if len(sys.argv) > 1:
        output_zip = sys.argv[1]
    else:
        output_zip = OUTPUT_ZIP_NAME

    # Ensure .zip extension
    if not output_zip.endswith('.zip'):
        output_zip += '.zip'

    print("="*80)
    print("KNOWLEDGE GRAPH EXPORT & PACKAGING TOOL")
    print("="*80)
    print(f"Output: {output_zip}\n")

    success = await create_deployment_package(output_zip)

    if success:
        print("\n✓ SUCCESS - Deployment package ready!")
        return 0
    else:
        print("\n✗ FAILED - Package creation incomplete")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
