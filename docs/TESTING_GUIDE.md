# Testing Guide - Mock Databases & Interactive Testing

This guide explains how to use the new mock database system and interactive testing tools.

## Quick Start

### 1. Setup Mock Databases

First, create all mock databases:

```bash
cd "test & debug/Databases"
python setup_all.py
```

This will create:
- **SQL Database**: `inventory_mock.db` with 10 sample products
- **Vector Databases**: 3 Chroma collections (main, property, relation)
- **Knowledge Graph**: Memgraph with products and relationships

**Note**: Make sure Memgraph is running before setup:
```bash
docker run -p 7687:7687 -p 7444:7444 memgraph/memgraph
```

### 2. Run Interactive Tests

```bash
cd "test & debug"
python interactive_test_full.py
```

This starts an interactive session where you can:
- Type natural language queries
- See detailed time measurements at each stage
- Switch between different users
- View parsed outputs and search results

## Database Schema

### SQL Database Structure

```sql
CREATE TABLE products (
    pdt_id TEXT PRIMARY KEY,
    pdt_name TEXT NOT NULL,
    subcategory TEXT,
    vendor_name TEXT,
    price REAL,
    ratings REAL,
    size TEXT,
    dimensions TEXT,
    colour TEXT,
    description TEXT,
    brand TEXT,
    subcategory_embeddings BLOB,
    image_id TEXT,
    stock INTEGER
)
```

**Sample Products**:
- p-001: Red Cotton T-Shirt ($25.99)
- p-002: Blue Denim Jeans ($45.99)
- p-003: Black Leather Jacket ($120.00)
- p-004: White Running Shoes ($65.00)
- p-005: Green Polo Shirt ($35.99)
- ... and 5 more

### Vector Database Metadata

**Main VDB** (Product embeddings):
```json
{
  "productid": "p-001",
  "category": "clothing"
}
```

Only two fields: `productid` and `category`

### Knowledge Graph Structure

**Product Nodes** (labeled by category):
```cypher
(:clothing {
  product_id: "p-001",
  product_name: "Red Cotton T-Shirt",
  category: "clothing"
})

(:footwear {
  product_id: "p-004",
  product_name: "White Running Shoes",
  category: "footwear"
})
```

**Property Nodes**:
```cypher
(:Property {name: "Color:Red"})
(:Property {name: "Material:Cotton"})
(:Property {name: "Style:Casual"})
```

**Relationships**:
```cypher
(product)-[:HAS_COLOUR]->(property)
(product)-[:HAS_MATERIAL]->(property)
(product)-[:HAS_STYLE]->(property)
```

## Mock User Data

Located in `mock_user_data/`:

### user-001 (Alice Johnson)
- Preferences: casual, athletic styles
- Brands: ComfortWear, RunFast
- Purchase history: 3 items (jeans, sneakers, hoodie)

### user-002 (Bob Smith)
- Preferences: formal, business casual
- Brands: ExecutiveStyle, ClassicFit
- Purchase history: 3 items (trousers, blazer, polo shirt)

### default_user (Test User)
- No preferences or purchase history
- Good for testing basic queries

## Interactive Test Modes

### 1. Interactive Mode (Default)

```bash
python interactive_test_full.py
```

**Usage**:
```
[default_user] > Red cotton shirt under $30
# Processes query with timing breakdown

[default_user] > user:user-001
# Switch to user-001

[user-001] > I want casual clothing
# User context will be applied
```

### 2. Preset Tests Mode

```bash
python interactive_test_full.py --preset
```

Runs 5 predefined test queries covering:
- Basic search with constraints
- Similarity search
- User context enrichment
- Multi-property search
- Detail queries

### 3. Single Query Mode

```bash
python interactive_test_full.py --query "red shirt" --user user-001
```

Run a single query and exit.

### 4. NVIDIA API Mode

```bash
export NVIDIA_API_KEY=your-key
python interactive_test_full.py --nvidia
```

Use real NVIDIA LLM instead of mock responses.

## Time Measurements

The test script provides detailed timing for each stage:

```
TIMING BREAKDOWN
================================================================================
  stage1_parse        :  0.234s ( 15.6%)
  stage2_search       :  1.267s ( 84.4%)
  ──────────────────────────────────────
  TOTAL               :  1.501s
```

**Stages**:
1. **stage1_parse**: Strontium parsing (LLM + user context enrichment)
2. **stage2_search**: Orchestrator execution (VDB + KG + SQL + scoring)
3. **stage2_detail**: Detail service (for detail queries)

## Test Query Examples

### Basic Search
```
Red cotton shirt under $30
Blue jeans size 32
Shoes with rating above 4
```

### Property Search
```
Black leather items
Casual cotton clothing
Formal wear from ExecutiveStyle
```

### Similarity Search
```
Shoes similar to p-004
Items like the blue jeans
Similar to what I bought last month
```

### User Context
```
I want casual clothing
# With user-001: Will prefer casual/athletic styles

Show me formal wear
# With user-002: Will prefer formal/business styles
```

### Detail Queries
```
What is the price of p-001?
Tell me about product p-004
What material is p-003 made of?
```

## Rebuilding Databases

To recreate databases with different data:

1. Edit the `MOCK_PRODUCTS` list in:
   - `Databases/setup_mock_sql.py`
   - `Databases/setup_mock_vdb.py`
   - `Databases/setup_mock_kg.py`

2. Run setup again:
   ```bash
   python Databases/setup_all.py
   ```

## Individual Database Setup

You can also setup databases individually:

```bash
# SQL only
python Databases/setup_mock_sql.py

# VDB only
python Databases/setup_mock_vdb.py

# KG only (requires Memgraph running)
python Databases/setup_mock_kg.py
```

## Troubleshooting

### Missing Databases
```
⚠ Missing databases! Run setup script first
```
**Fix**: Run `python Databases/setup_all.py`

### Memgraph Connection Failed
```
✗ KG Client failed to initialize
```
**Fix**: Start Memgraph with Docker:
```bash
docker run -p 7687:7687 -p 7444:7444 memgraph/memgraph
```

### No Search Results
- Check that VDB collections have data
- Verify SQL database has products
- Check KG has relationships

### Import Errors
- Make sure you're in the `test & debug` directory
- Check that parent directory is accessible

## Database Locations

```
test & debug/
├── Databases/
│   ├── inventory_mock.db          # SQL database
│   ├── main_vdb_mock/             # Main VDB collection
│   ├── property_vdb_mock/         # Property VDB collection
│   ├── relation_vdb_mock/         # Relation VDB collection
│   ├── setup_all.py               # Master setup script
│   ├── setup_mock_sql.py          # SQL setup
│   ├── setup_mock_vdb.py          # VDB setup
│   └── setup_mock_kg.py           # KG setup
├── mock_user_data/
│   ├── user-001.json              # Alice's profile
│   ├── user-002.json              # Bob's profile
│   └── default_user.json          # Test user
└── interactive_test_full.py       # Main test script
```

## Next Steps

1. **Setup databases**: `python Databases/setup_all.py`
2. **Start testing**: `python interactive_test_full.py`
3. **Try different users**: `user:user-001` in interactive mode
4. **Test preset queries**: `python interactive_test_full.py --preset`
5. **Measure performance**: Check timing breakdowns for optimization

## Performance Benchmarks

With mock data (10 products):
- **Parsing**: ~0.2-0.5s (mock LLM)
- **Search**: ~0.5-1.5s (includes VDB, KG, SQL queries)
- **Total**: ~1-2s per query

With NVIDIA API:
- **Parsing**: ~1-3s (real LLM call)
- **Search**: Same as above
- **Total**: ~2-4s per query

Note: Times will increase with larger datasets.
