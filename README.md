# Test & Debug - Interactive Testing Suite

This directory contains all testing, debugging, and example files for The Whole Story pipeline.

## 🎯 Quick Start

### Interactive Testing (Recommended!)

Run the interactive test suite:
```bash
python interactive_test.py
```

This provides a menu-driven interface to test:
1. **Strontium Parsing Only** - See how queries are parsed to JSON
2. **Search Pipeline** - Test product search with rankings
3. **Detail Query System** - Ask questions about products
4. **Chat Handler** - Have conversations with Strontium
5. **Complete Pipeline** - Auto-routes any query type

### Features

- 🎨 **Color-coded output** for easy reading
- 🔄 **REPL-style interface** - test multiple queries without restarting
- 📊 **JSON pretty-printing** for parsed results
- ✅ **Success/error indicators** for quick feedback
- 🔍 **Detailed debugging** with stack traces when errors occur

---

## 📁 Directory Contents

### Interactive Testing
- `interactive_test.py` - **Main testing interface** (start here!)

### Unit Tests
- `tests/test_orchestrator.py` - Orchestrator tests
- `tests/test_strontium.py` - Strontium parsing tests
- `tests/test_strontium_debug.py` - Debug tests for Strontium
- `tests/test_strontium_nvidia.py` - NVIDIA API tests
- `tests/setup_databases.py` - Database initialization

### Examples
- `examples/example_usage.py` - Basic usage examples
- `examples/example_strontium.py` - Strontium parsing examples
- `examples/example_debug_queries.py` - Debug query examples

### Documentation
- `docs/` - Additional documentation (if any)

---

## 🧪 Testing Modes

### Mode 1: Strontium Parsing Only

Tests the natural language → JSON conversion:

```
Enter query: Red cotton shirt under $30

>>> Parsing: 'Red cotton shirt under $30'

>>> Parsed Result:
{
  "query_type": "search",
  "products": [{
    "product_query": "Red cotton shirt under $30",
    "product_basetype": "shirt",
    "properties": [
      ["red", 1.5, "HAS_COLOUR"],
      ["cotton", 1.2, "HAS_MATERIAL"]
    ],
    "literals": [["price", "<", 30.0, 0.1]]
  }]
}

✓ Query Type: SEARCH
ℹ Products requested: 1
```

**Use this to:**
- Verify Strontium correctly extracts properties
- Check relation types are accurate
- Debug parsing issues
- Test new query patterns

---

### Mode 2: Search Pipeline

Tests the complete search flow from parsing → ranking:

```
Enter search query: Red cotton shirt under $30

>>> Processing: 'Red cotton shirt under $30'
ℹ Parsed successfully

>>> Executing search...

>>> Search Results:
✓ Found 12 products
  Top 5: ['p-001', 'p-005', 'p-004', 'p-007', 'p-002']
```

**Use this to:**
- Test end-to-end search
- Verify product rankings
- Debug search algorithm issues
- Test with different query patterns

---

### Mode 3: Detail Query System

Tests product question answering:

```
Enter detail query: What material is p-456 made of?

>>> Processing: 'What material is p-456 made of?'
ℹ Parsed successfully
ℹ Product: p-456
ℹ Properties: ['material']
ℹ Keywords: ['material', 'fabric', 'made of']

>>> Generating answer...

>>> Answer:
I'm Strontium, and I'd be happy to help!

Based on what I know about this product:

Product ID: p-456
Name: Blue Formal Shirt
Category: shirt
Price: $19.99
Brand: BrandB

The vendor did not mention this specific information in the product listing.
You can contact them directly for more details.

Vendor: Store s-1234, Phone: N/A
```

**Use this to:**
- Test detail query answering
- Verify multi-source retrieval (SQL + KG + VDB)
- Check vendor fallback works
- Debug LLM answer generation

---

### Mode 4: Chat Handler

Tests conversational responses:

```
You: Hello!

Strontium: Hello! I'm Strontium, your curator at BeeKurse. How can I help you find what you need today?

You: What can you do?

Strontium: I can help you in several ways:

• Find Products: Tell me what you're looking for (e.g., "red cotton shirt under $30")
• Product Details: Ask me about specific products (e.g., "What material is p-456 made of?")
• Comparisons: Help you find items similar to ones you like
• Recommendations: Suggest products based on your preferences

Just tell me what you need, and I'll curate the perfect options for you!
```

**Use this to:**
- Test chat responses
- Verify Strontium identity is consistent
- Test different conversation patterns
- Debug chat routing

---

### Mode 5: Complete Pipeline (Auto-route)

Automatically detects query type and routes appropriately:

```
Enter query: Red shirt under $30
ℹ Query type: SEARCH
✓ Found 8 products

Enter query: What material is p-001 made of?
ℹ Query type: DETAIL
Answer: [natural language answer]

Enter query: Hello!
ℹ Query type: CHAT
Response: Hello! I'm Strontium...
```

**Use this to:**
- Test the complete system
- Verify routing works correctly
- Simulate real user interactions
- Integration testing

---

## 🔍 Debugging Tips

### Viewing Parsed JSON

In Mode 1 (Strontium Parsing), you'll see the complete JSON output with:
- Query type classification
- Extracted properties with weights and relation types
- Literal constraints with buffers
- Relation types and keywords (for detail queries)

### Common Issues

**Issue: Parsing fails**
- Check if query is ambiguous
- Verify property values are clear
- Test in Mode 1 to see exact JSON output

**Issue: No search results**
- Check if database is populated
- Verify literals aren't too restrictive
- Look at parsed JSON to confirm extraction

**Issue: Detail query returns "vendor contact"**
- Expected behavior when info is missing
- Check if KG has properties for that product
- Verify VDB has relevant embeddings

**Issue: Chat not recognized**
- Add more greeting patterns if needed
- Check `chat_handler.py` response templates

---

## 📝 Running Unit Tests

From the parent directory:

```bash
# Run orchestrator tests
python -m pytest "test & debug/tests/test_orchestrator.py" -v

# Run Strontium tests
python -m pytest "test & debug/tests/test_strontium.py" -v

# Run all tests
python -m pytest "test & debug/tests/" -v
```

Or from this directory:

```bash
cd "test & debug"
python tests/test_orchestrator.py
python tests/test_strontium.py
```

---

## 🛠️ Setup

### Database Setup

Before testing, ensure databases are initialized:

```bash
cd "test & debug/tests"
python setup_databases.py
```

This creates:
- SQLite database with sample products
- Chroma vector databases (Main, Property, Relation)
- Memgraph knowledge graph (if running)

### Configuration

All configuration is in the parent `config.py`:
- Database paths
- VDB top K values
- Score thresholds
- Debug mode

---

## 💡 Examples

### Test a New Query Pattern

```python
# In Mode 1 (Strontium Parsing)
Enter query: ergonomic memory foam pillow for neck pain under $50

# Verify it extracts:
# - basetype: "pillow"
# - properties: [["ergonomic", weight, "HAS_FEATURE"],
#                ["memory foam", weight, "HAS_MATERIAL"],
#                ["neck pain", weight, "HAS_FEATURE"]]
# - literals: [["price", "<", 50.0, buffer]]
```

### Test Detail Query Extraction

```python
# In Mode 1
Enter query: Does p-789 need special care instructions?

# Verify it extracts:
# - query_type: "detail"
# - product_id: "p-789"
# - properties_to_explain: ["care_instructions"]
# - relation_types: ["HAS_CARE_INSTRUCTIONS"]
# - query_keywords: ["care", "instructions", "special care"]
```

### Test Multi-Property Search

```python
# In Mode 2 (Search Pipeline)
Enter search query: red cotton casual shirt under $30 size M

# Verify:
# - Multiple properties extracted
# - Multiple literals (price + size)
# - Reasonable product rankings
```

---

## 🎓 Learning the System

**Recommended Testing Path:**

1. **Start with Mode 1** - Understand how Strontium parses queries
2. **Move to Mode 5** - See the complete pipeline in action
3. **Deep dive with Modes 2-4** - Debug specific components
4. **Experiment freely** - Try edge cases and unusual queries

---

## 📊 Interactive Test Features

### Color Coding

- 🟦 Blue (ℹ) - Info messages
- 🟩 Green (✓) - Success messages
- 🟥 Red (✗) - Error messages
- 🟨 Yellow - JSON output
- 🟪 Purple - Headers
- 🔵 Cyan - Section titles

### Keyboard Shortcuts

- `Ctrl+C` - Exit current mode, return to menu
- `quit`, `exit`, or `q` - Exit current mode gracefully
- `Ctrl+D` (Unix) or `Ctrl+Z` (Windows) - EOF, same as Ctrl+C

---

## 🔗 Related Documentation

See parent directory `README.md` for:
- Complete pipeline architecture
- Algorithm explanations
- Database schema
- Production usage

---

**Happy Testing! 🚀**

*Remember: This is a safe testing environment. Break things, try weird queries, and learn how the system works!*
