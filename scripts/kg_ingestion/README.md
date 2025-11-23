# E-Commerce Knowledge Graph Ingestion Pipeline

## Overview

This pipeline processes e-commerce product JSON files and creates a knowledge graph with:
- Product nodes with category-based labels
- Attribute relationships (brand, color, store)
- Description-based relationships extracted via multi-agent AI pipeline
- Vector databases for property values and relation standardization

## Architecture

```
Input: JSON Files (full_list/)
    ↓
Step 1: Initialize VDBs (custom_relations_vdb, property_vdb)
    ↓
Step 2: Start Unified Ingestion Queue
    ↓
Step 3: Process Product Attributes → Memgraph + Property VDB
    ↓
Step 4: Extract Relations from Descriptions (AI Multi-Agent)
    ├── Agent 1 (Extractor): Extract raw triplets from chunks
    ├── Agent 2 (Standardizer): RAG + VDB relation matching
    └── Agent 3 (Verifier): Validate against source text
    ↓
Output: Memgraph KG + Property VDB + Custom Relations VDB
```

## Requirements

### System Requirements
- **Python**: 3.10+
- **Memgraph**: Running on localhost:7687
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ free space

### Python Dependencies

```bash
pip install neo4j chromadb langchain-nvidia-ai-endpoints \
            langchain-huggingface langchain-chroma langchain-core \
            python-dotenv pydantic
```

## Installation & Setup

### 1. Install Memgraph

**Ubuntu/Debian:**
```bash
# Add Memgraph repository
curl https://download.memgraph.com/memgraph/v2.14.1/ubuntu-22.04/memgraph_2.14.1-1_amd64.deb -o memgraph.deb
sudo dpkg -i memgraph.deb

# Start Memgraph
sudo systemctl start memgraph
sudo systemctl enable memgraph
```

**Verify Memgraph is running:**
```bash
ps aux | grep memgraph
```

### 2. Configure Environment

Edit `.env` file and add your API keys:

```bash
# NVIDIA API Key for LLM access (required)
NVIDIA_API_KEY=your_nvidia_api_key_here

# Memgraph Database Configuration
MEMGRAPH_URI=bolt://localhost:7687
MEMGRAPH_USER=
MEMGRAPH_PASS=
```

**Get NVIDIA API Key:** https://build.nvidia.com/

### 3. Verify Dataset

```bash
ls full_list/*.json | wc -l
# Should show: 462
```

## Usage

### Basic Usage

Process all JSON files:

```bash
python run_pipeline.py ./full_list
```

### Skip Export (Faster for Testing)

```bash
python run_pipeline.py ./full_list --skip-export
```

### Test with Smaller Batch

```bash
mkdir -p test_batch
ls full_list/*.json | head -5 | xargs -I {} cp {} test_batch/
python run_pipeline.py ./test_batch --skip-export
```

### Dry-Run Mode (No Database Writes)

```bash
export DEBUG_DRY_RUN=true
python run_pipeline.py ./test_batch --skip-export
unset DEBUG_DRY_RUN
```

## Performance

- **10 files**: ~10-15 minutes (with API rate limiting)
- **462 files**: ~4-8 hours (depends on API tier)

**Note**: Free NVIDIA API tier has strict rate limits (429 errors). Upgrade for better performance.

## Output

### Query Memgraph

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687')
with driver.session() as session:
    result = session.run('MATCH (n) RETURN count(n) as count')
    print(f"Nodes: {result.single()['count']}")
```

### Files Created

- `property_vdb/` - Property embeddings VDB
- `custom_relations_vdb/` - Relations VDB
- `output/description_triplets.jsonl` - Extracted triplets
- `kg_deployment_package.zip` - Deployment package

## Troubleshooting

### "Too Many Requests" (429 Errors)

- NVIDIA API rate limiting
- Solution: Wait and retry, or upgrade API tier

### Memgraph Not Running

```bash
sudo systemctl start memgraph
nc -zv localhost 7687
```

## File Structure

```
.
├── run_pipeline.py                    # Main orchestrator
├── process_product_jsons.py           # Attribute extraction
├── extract_relation_ingestion.py      # Relation extraction (AI)
├── unified_ingestion_queue.py         # Shared ingestion queue
├── memgraph_utils.py                  # Memgraph operations
├── init_property_vdb.py               # Property VDB initialization
├── init_custom_relations_vdb.py       # Custom relations VDB init
├── export_and_package.py              # Deployment packaging
├── seed_relations.json                # 86 seed relations
├── .env                               # Environment configuration
├── full_list/                         # 462 product JSON files
└── README.md                          # This file
```

## License

Internal use only.
