"""
Full End-to-End Interactive Testing Script
Tests complete search orchestrator pipeline with time measurements at each stage
"""
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Use actual databases from config (no environment overrides needed)
# The config.py already points to the correct database paths:
#   - Databases/sql/inventory.db
#   - Databases/vector_db/Main Vector_Database
#   - Databases/vector_db/property_vdb
#   - Databases/vector_db/relation_vdb

# Import production code
from search_agent.strontium.strontium_agent import StrontiumAgent
from search_agent.orchestrator.orchestrator import SearchOrchestrator
from search_agent.orchestrator.detail_service import ProductDetailService
from search_agent.orchestrator.chat_handler import ChatHandler
from search_agent.database.sql_client import SQLClient
from search_agent.database.vdb_client import MainVDBClient, PropertyVDBClient, RelationVDBClient
from search_agent.database.kg_client import KGClient
from scripts.database_operations.sql_extract import fetch_products_by_ids


class TimedTestRunner:
    """Test runner with detailed time measurements"""

    def __init__(self, use_nvidia=False):
        """Initialize test runner with mock databases"""

        print("\n" + "=" * 80)
        print(" " * 25 + "INTERACTIVE TEST RUNNER")
        print(" " * 20 + "with Time Measurements")
        print("=" * 80)

        # Import config to get actual database paths
        from config.config import Config

        # Use actual database paths from config
        self.sql_path = Config.SQL_DB_PATH
        self.main_vdb_path = Config.MAIN_VDB_PATH
        self.property_vdb_path = Config.PROPERTY_VDB_PATH
        self.relation_vdb_path = Config.RELATION_VDB_PATH
        self.mock_user_data_dir = Config.USER_CONTEXT_DATA_DIR

        # Check if databases exist
        self._verify_databases()

        # Initialize components with timing
        print("\n" + "=" * 80)
        print("INITIALIZING COMPONENTS")
        print("=" * 80)

        start = time.time()

        # Initialize Search Orchestrator FIRST (it creates all DB clients)
        print("\n[1/3] Initializing Search Orchestrator...")
        orch_start = time.time()

        # SearchOrchestrator creates its own clients from config
        self.orchestrator = SearchOrchestrator()
        print(f"  ✓ Search Orchestrator initialized ({time.time() - orch_start:.3f}s)")

        # Use the clients from orchestrator (avoid double initialization)
        self.sql_client = self.orchestrator.sql_client
        self.main_vdb_client = self.orchestrator.main_vdb
        self.property_vdb_client = self.orchestrator.property_vdb
        self.relation_vdb_client = self.orchestrator.relation_vdb
        self.kg_client = self.orchestrator.kg_client
        print(f"  ✓ Reusing database clients from orchestrator")

        # Initialize Strontium Agent
        print("\n[2/3] Initializing Strontium Agent...")
        strontium_start = time.time()

        self.strontium = StrontiumAgent(
            llm_client=None,
            kg_client=self.kg_client,
            mock_data_dir=self.mock_user_data_dir,
            use_nvidia=use_nvidia,
            enable_caching=False  # Disable caching for testing
        )
        print(f"  ✓ Strontium Agent initialized ({time.time() - strontium_start:.3f}s)")
        print(f"  Mode: {'NVIDIA API' if use_nvidia else 'Mock LLM'}")

        # Initialize Detail Service
        print("\n[3/3] Initializing Detail Service and Chat Handler...")
        detail_start = time.time()

        self.detail_service = ProductDetailService(
            sql_client=self.sql_client
        )
        print(f"  ✓ Detail Service initialized ({time.time() - detail_start:.3f}s)")

        # Initialize Chat Handler
        chat_start = time.time()
        self.chat_handler = ChatHandler()
        print(f"  ✓ Chat Handler initialized ({time.time() - chat_start:.3f}s)")

        total_init_time = time.time() - start
        print(f"\n✓ All components initialized in {total_init_time:.3f}s")

        self.timings = {}

    def _verify_databases(self):
        """Verify mock databases exist"""
        print("\n[Verifying Mock Databases]")

        checks = [
            ("SQL Database", Path(self.sql_path)),
            ("Main VDB", Path(self.main_vdb_path)),
            ("Property VDB", Path(self.property_vdb_path)),
            ("Relation VDB", Path(self.relation_vdb_path)),
            ("User Data", Path(self.mock_user_data_dir))
        ]

        all_exist = True
        for name, path in checks:
            if path.exists():
                print(f"  ✓ {name}: {path}")
            else:
                print(f"  ✗ {name}: NOT FOUND at {path}")
                all_exist = False

        if not all_exist:
            print("\n⚠ Missing databases! Run setup script first:")
            print(f"  python '{Path(__file__).parent}/Databases/setup_all.py'")
            sys.exit(1)

        print("  ✓ All databases found")

    def _format_whatsapp_search_response(self, product_ids: List[str]) -> Dict[str, Any]:
        """Format search response exactly as /process endpoint would send to WhatsApp"""
        if not product_ids:
            return {"text": "😔 Sorry, no products matched your search.", "images": []}

        products = fetch_products_by_ids(product_ids)
        top_products = [products[pid] for pid in product_ids[:4] if pid in products]

        # Build text (same as strontium_api.py format_search_response)
        lines = [f"🔍 Found {len(product_ids)} product(s):\n"]
        for i, p in enumerate(top_products, 1):
            name = p["prod_name"]
            if len(name) > 40:
                name = name[:40] + "..."
            lines.extend([
                f"{i}. *{name}*",
                f"   ₹{p['price'] or 'N/A'} | ⭐{p['rating'] or 'N/A'}",
                f"   Store: {p['store'] or 'Unknown'}",
                f"   ID: {p['short_id']}\n"
            ])
        if len(product_ids) > 4:
            lines.append(f"_... and {len(product_ids) - 4} more results_")
        lines.extend(["\n💡 Ask about a product using its ID", 'Example: _"details of A3F1"_'])

        return {
            "text": "\n".join(lines),
            "images": [{"url": p["image_url"], "caption": f"{p['prod_name']} (ID: {p['short_id']})"} for p in top_products],
            "all_product_ids": product_ids  # Full list of ALL product IDs found
        }

    def _format_whatsapp_detail_response(self, parsed_dict: Dict, answer: str) -> Dict[str, Any]:
        """Format detail response exactly as /process endpoint would send to WhatsApp"""
        product_ids = parsed_dict.get("product_ids", [])
        if not product_ids:
            return {"messages": [{"type": "text", "text": "❌ No product IDs found."}]}

        # Resolve short_ids (4-char codes like "44QM") to full product_ids
        import re
        short_id_pattern = re.compile(r'^[A-Z0-9]{4}$', re.IGNORECASE)
        resolved_ids = []
        for pid in product_ids:
            if short_id_pattern.match(pid):
                full_id = self.sql_client.resolve_short_id(pid)
                resolved_ids.append(full_id if full_id else pid)
            else:
                resolved_ids.append(pid)
        product_ids = resolved_ids

        products = fetch_products_by_ids(product_ids)

        if not products:
            return {"messages": [{"type": "text", "text": f"❌ No products found for IDs {product_ids}"}]}

        # For single product, show image; for multiple, just show text
        messages = []
        if len(product_ids) == 1 and product_ids[0] in products:
            product = products[product_ids[0]]
            messages.append({"type": "image", "url": product["image_url"]})

        # Just use the natural LLM response - no technical headers
        messages.append({"type": "text", "text": answer})

        return {
            "messages": messages,
            "product_ids": product_ids
        }

    def _format_whatsapp_chat_response(self, response: str) -> Dict[str, Any]:
        """Format chat response exactly as /process endpoint would send to WhatsApp"""
        return {"reply": response}

    def run_search_query(self, query: str, user_id: str = "default_user") -> Dict[str, Any]:
        """Run a search query with detailed timing"""

        print("\n" + "=" * 80)
        print("PROCESSING SEARCH QUERY")
        print("=" * 80)
        print(f"\nQuery: '{query}'")
        print(f"User:  {user_id}")

        self.timings = {}
        total_start = time.time()

        # Stage 1: Parse with Strontium
        print("\n" + "-" * 80)
        print("STAGE 1: Natural Language Parsing (Strontium)")
        print("-" * 80)

        stage1_start = time.time()
        parsed = self.strontium.process_query(query, user_id)
        stage1_time = time.time() - stage1_start

        self.timings['stage1_parse'] = stage1_time

        print(f"\n[Parsing Result] ({stage1_time:.3f}s)")
        parsed_dict = self.strontium.formatter.to_dict(parsed)
        print(json.dumps(parsed_dict, indent=2))

        # Debug: Show parsed products and properties
        from config.config import Config
        if Config.DEBUG and parsed_dict["query_type"] == "search":
            print(f"\n[DEBUG] Parsed query breakdown:")
            for i, product in enumerate(parsed_dict.get("products", []), 1):
                print(f"  Product {i}:")
                print(f"    Category: {product.get('product_category')}")
                print(f"    Subcategory: {product.get('product_subcategory')}")
                print(f"    Properties: {product.get('properties', [])}")
                print(f"    Literals: {product.get('literals', [])}")

        # Check query type
        if parsed_dict["query_type"] == "detail":
            return self._run_detail_query(parsed_dict, total_start)
        elif parsed_dict["query_type"] == "chat":
            return self._run_chat_query(parsed_dict, total_start)

        # Stage 2: Execute Search
        print("\n" + "-" * 80)
        print("STAGE 2: Search Execution (Orchestrator)")
        print("-" * 80)

        if Config.DEBUG:
            print(f"\n[DEBUG] Executing orchestrator search...")
            print(f"  Products to search: {len(parsed_dict.get('products', []))}")

        stage2_start = time.time()
        search_results = self.orchestrator.search_strontium(parsed_dict)
        stage2_time = time.time() - stage2_start

        self.timings['stage2_search'] = stage2_time

        # Extract product IDs from all SearchResult objects
        all_product_ids = []
        for search_result in search_results:
            all_product_ids.extend(search_result.product_ids)

        print(f"\n[Search Results] ({stage2_time:.3f}s)")
        print(f"  Found {len(all_product_ids)} products")

        if Config.DEBUG:
            print(f"\n[DEBUG] Search results breakdown:")
            print(f"  Total SearchResult objects: {len(search_results)}")
            for i, sr in enumerate(search_results, 1):
                print(f"  SearchResult {i}: {len(sr.product_ids)} products")
            if not all_product_ids:
                print(f"  ⚠️ No products found in final results!")
            else:
                print(f"  Sample product IDs: {all_product_ids[:5]}")

        # Fetch actual product details from SQL
        if all_product_ids:
            products_dict = self.sql_client.get_products_by_ids(all_product_ids)

            print("\n  Top 5 Results:")
            for i, product_id in enumerate(all_product_ids[:5], 1):
                product = products_dict.get(product_id)
                if product:
                    print(f"    {i}. {product.prod_name} - {product.price}")
                    print(f"       ID: {product.product_id} | Rating: {product.rating} | Stock: {product.stock}")
                else:
                    print(f"    {i}. Product {product_id} (details not found)")

        # Stage 3: Show WhatsApp Response (Strontium /process endpoint output)
        print("\n" + "-" * 80)
        print("STAGE 3: WhatsApp Response (Strontium /process endpoint output)")
        print("-" * 80)

        whatsapp_response = self._format_whatsapp_search_response(all_product_ids)
        print("\n[Final JSON to WhatsApp Bot]")
        print(json.dumps(whatsapp_response, indent=2, ensure_ascii=False))

        # Total time
        total_time = time.time() - total_start
        self.timings['total'] = total_time

        self._print_timing_summary()

        return {
            "query_type": "search",
            "parsed": parsed_dict,
            "product_ids": all_product_ids,
            "products": products_dict if all_product_ids else {},
            "whatsapp_response": whatsapp_response,
            "timings": self.timings
        }

    def _run_detail_query(self, parsed_dict: Dict, total_start: float) -> Dict[str, Any]:
        """Run detail query"""

        print("\n" + "-" * 80)
        print("STAGE 2: Detail Query Execution")
        print("-" * 80)

        stage2_start = time.time()
        # Use orchestrator.answer_detail_query like strontium_api.py does
        answer = self.orchestrator.answer_detail_query(parsed_dict)
        stage2_time = time.time() - stage2_start

        self.timings['stage2_detail'] = stage2_time

        print(f"\n[Detail Answer] ({stage2_time:.3f}s)")
        print(answer)

        # Stage 3: Show WhatsApp Response (Strontium /process endpoint output)
        print("\n" + "-" * 80)
        print("STAGE 3: WhatsApp Response (Strontium /process endpoint output)")
        print("-" * 80)

        whatsapp_response = self._format_whatsapp_detail_response(parsed_dict, answer)
        print("\n[Final JSON to WhatsApp Bot]")
        print(json.dumps(whatsapp_response, indent=2, ensure_ascii=False))

        total_time = time.time() - total_start
        self.timings['total'] = total_time

        self._print_timing_summary()

        return {
            "query_type": "detail",
            "parsed": parsed_dict,
            "answer": answer,
            "whatsapp_response": whatsapp_response,
            "timings": self.timings
        }

    def _run_chat_query(self, parsed_dict: Dict, total_start: float) -> Dict[str, Any]:
        """Run chat query"""

        print("\n" + "-" * 80)
        print("STAGE 2: Chat Query Handling")
        print("-" * 80)

        print("\n[Chat Query Detected]")
        print(f"Message: {parsed_dict.get('message', '')}")

        # Generate chat response
        stage2_start = time.time()
        response = self.chat_handler.handle_chat_output(parsed_dict)
        stage2_time = time.time() - stage2_start

        self.timings['stage2_chat'] = stage2_time

        print(f"\n[Chat Response] ({stage2_time:.3f}s)")
        print(f"\n{response}\n")

        # Stage 3: Show WhatsApp Response (Strontium /process endpoint output)
        print("\n" + "-" * 80)
        print("STAGE 3: WhatsApp Response (Strontium /process endpoint output)")
        print("-" * 80)

        whatsapp_response = self._format_whatsapp_chat_response(response)
        print("\n[Final JSON to WhatsApp Bot]")
        print(json.dumps(whatsapp_response, indent=2, ensure_ascii=False))

        total_time = time.time() - total_start
        self.timings['total'] = total_time

        self._print_timing_summary()

        return {
            "query_type": "chat",
            "parsed": parsed_dict,
            "response": response,
            "whatsapp_response": whatsapp_response,
            "timings": self.timings
        }

    def _print_timing_summary(self):
        """Print timing breakdown"""

        print("\n" + "=" * 80)
        print("TIMING BREAKDOWN")
        print("=" * 80)

        for stage, duration in self.timings.items():
            if stage != 'total':
                percentage = (duration / self.timings['total']) * 100
                print(f"  {stage:20s}: {duration:6.3f}s ({percentage:5.1f}%)")

        print(f"  {'─' * 40}")
        print(f"  {'TOTAL':20s}: {self.timings['total']:6.3f}s")

    def interactive_mode(self):
        """Interactive query testing mode"""

        print("\n" + "=" * 80)
        print(" " * 25 + "INTERACTIVE MODE")
        print("=" * 80)
        print("\nCommands:")
        print("  - Enter a query to test")
        print("  - 'user:ID' to change user (e.g., 'user:user-001')")
        print("  - 'quit' or 'exit' to quit")
        print("=" * 80)

        current_user = "default_user"

        while True:
            try:
                query = input(f"\n[{current_user}] > ").strip()

                if not query:
                    continue

                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                elif query.lower().startswith('user:'):
                    current_user = query.split(':', 1)[1].strip()
                    print(f"✓ Changed user to: {current_user}")
                    continue

                # Process query
                self.run_search_query(query, current_user)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")
                import traceback
                traceback.print_exc()

    def run_preset_tests(self):
        """Run preset test queries"""

        print("\n" + "=" * 80)
        print(" " * 25 + "PRESET TEST QUERIES")
        print("=" * 80)

        test_cases = [
            ("Red cotton shirt under $30", "default_user", "Basic search with property and price constraint"),
            ("Shoes similar to p-004", "default_user", "Similarity search"),
            ("I want casual clothing", "user-001", "User context enrichment"),
            ("Show me black leather items", "default_user", "Multi-property search"),
            ("What is the price of p-001?", "default_user", "Detail query"),
            ("Hello!", "default_user", "Chat query - greeting"),
            ("What can you do?", "default_user", "Chat query - help"),
        ]

        for i, (query, user_id, description) in enumerate(test_cases, 1):
            print(f"\n{'▓' * 80}")
            print(f"▓  TEST {i}/{len(test_cases)}: {description}")
            print(f"{'▓' * 80}")

            try:
                self.run_search_query(query, user_id)
                print(f"\n✓ Test {i} completed")
            except Exception as e:
                print(f"\n✗ Test {i} failed: {e}")
                import traceback
                traceback.print_exc()

            if i < len(test_cases):
                input("\nPress Enter to continue...")


def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(description="Interactive Test Runner")
    parser.add_argument('--no-nvidia', action='store_true', help="Disable NVIDIA API (use mock LLM instead)")
    parser.add_argument('--preset', action='store_true', help="Run preset tests")
    parser.add_argument('--query', type=str, help="Run single query")
    parser.add_argument('--user', type=str, default="default_user", help="User ID")

    args = parser.parse_args()

    # Initialize runner (NVIDIA enabled by default, use --no-nvidia to disable)
    runner = TimedTestRunner(use_nvidia=not args.no_nvidia)

    if args.query:
        # Run single query
        runner.run_search_query(args.query, args.user)

    elif args.preset:
        # Run preset tests
        runner.run_preset_tests()

    else:
        # Interactive mode
        runner.interactive_mode()


if __name__ == "__main__":
    main()
