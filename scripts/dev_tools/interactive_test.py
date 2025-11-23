"""
Interactive Testing Script for The Whole Story Pipeline

Test Strontium parsing, search orchestration, detail queries, and chat
in an interactive REPL-style interface.
"""
import sys
from pathlib import Path
import json

# Import local modules
from search_agent.strontium.strontium_agent import StrontiumAgent
from config.config import Config
from search_agent.orchestrator.orchestrator import SearchOrchestrator

config = Config


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def initialize_strontium_agent():
    """Initialize Strontium agent with config settings"""
    return StrontiumAgent(
        mock_data_dir=config.USER_CONTEXT_DATA_DIR,
        enable_caching=config.ENABLE_CACHING,
        llm_cache_ttl=config.LLM_CACHE_TTL,
        kg_cache_ttl=config.KG_CACHE_TTL,
        use_nvidia=config.USE_NVIDIA_LLM,
        nvidia_api_key=config.NVIDIA_API_KEY
    )


def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")


def print_section(title):
    """Print section title"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}>>> {title}{Colors.END}")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_json(data, indent=2):
    """Print formatted JSON"""
    print(f"{Colors.YELLOW}{json.dumps(data, indent=indent, default=str)}{Colors.END}")


def test_strontium_parsing():
    """Interactive Strontium parsing test"""
    print_header("STRONTIUM PARSING - Interactive Test")

    print_info("This mode tests the Strontium query parser")
    print_info("Enter natural language queries and see the structured JSON output\n")

    # Initialize Strontium
    print_section("Initializing Strontium Agent...")
    agent = initialize_strontium_agent()
    print_success("Strontium initialized\n")

    print(f"{Colors.BOLD}Example queries to try:{Colors.END}")
    print("  • Red cotton shirt under $30")
    print("  • What material is p-456 made of?")
    print("  • Does p-789 require dry cleaning?")
    print("  • Shoes like p-123 but cheaper")
    print("  • Hello!")
    print(f"\n{Colors.BOLD}Type 'quit' or 'exit' to return to main menu{Colors.END}\n")

    while True:
        try:
            # Get user input
            query = input(f"{Colors.BOLD}Enter query:{Colors.END} ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                break

            # Parse query
            print_section(f"Parsing: '{query}'")

            try:
                result = agent.process_query_to_dict(query, user_id="test_user")

                # Print result
                print_section("Parsed Result:")
                print_json(result)

                # Identify query type
                query_type = result.get("query_type", "unknown")
                if query_type == "search":
                    print_success(f"Query Type: SEARCH")
                    num_products = len(result.get("products", []))
                    print_info(f"Products requested: {num_products}")
                elif query_type == "detail":
                    print_success(f"Query Type: DETAIL")
                    print_info(f"Product: {result.get('product_id')}")
                    print_info(f"Properties: {result.get('properties_to_explain')}")
                elif query_type == "chat":
                    print_success(f"Query Type: CHAT")
                    print_info(f"Message: {result.get('message')}")

                print()

            except Exception as e:
                print_error(f"Parsing failed: {e}")
                import traceback
                traceback.print_exc()
                print()

        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            print("\n")
            break


def test_search_pipeline():
    """Interactive search pipeline test"""
    print_header("SEARCH PIPELINE - Interactive Test")

    print_info("This mode tests the complete search pipeline")
    print_info("Enter search queries and see ranked product results\n")

    # Initialize
    print_section("Initializing Components...")
    agent = initialize_strontium_agent()
    orchestrator = SearchOrchestrator()
    print_success("Components initialized\n")

    print(f"{Colors.BOLD}Example search queries:{Colors.END}")
    print("  • Red cotton shirt under $30")
    print("  • Casual watch for men")
    print("  • Memory foam pillow for neck pain")
    print(f"\n{Colors.BOLD}Type 'quit' or 'exit' to return to main menu{Colors.END}\n")

    while True:
        try:
            query = input(f"{Colors.BOLD}Enter search query:{Colors.END} ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                break

            print_section(f"Processing: '{query}'")

            try:
                # Parse
                parsed = agent.process_query_to_dict(query, user_id="test_user")

                if parsed.get("query_type") != "search":
                    print_error(f"Not a search query (type: {parsed.get('query_type')})")
                    continue

                print_info("Parsed successfully")

                # Execute search
                print_section("Executing search...")
                results = orchestrator.search_strontium(parsed)

                # Display results
                print_section("Search Results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{Colors.BOLD}Result {i}:{Colors.END}")
                    if result.product_ids:
                        print_success(f"Found {len(result.product_ids)} products")
                        print(f"  Top 5: {result.product_ids[:5]}")
                    else:
                        print_error("No products found")

                print()

            except Exception as e:
                print_error(f"Search failed: {e}")
                import traceback
                traceback.print_exc()
                print()

        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            print("\n")
            break


def test_detail_query():
    """Interactive detail query test"""
    print_header("DETAIL QUERY - Interactive Test")

    print_info("This mode tests the detail query answering system")
    print_info("Ask questions about products and get natural language answers\n")

    # Initialize
    print_section("Initializing Components...")
    agent = initialize_strontium_agent()
    orchestrator = SearchOrchestrator()
    print_success("Components initialized\n")

    print(f"{Colors.BOLD}Example detail queries:{Colors.END}")
    print("  • What material is p-456 made of?")
    print("  • Does p-789 require dry cleaning?")
    print("  • Tell me more about p-001")
    print(f"\n{Colors.BOLD}Type 'quit' or 'exit' to return to main menu{Colors.END}\n")

    while True:
        try:
            query = input(f"{Colors.BOLD}Enter detail query:{Colors.END} ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                break

            print_section(f"Processing: '{query}'")

            try:
                # Parse
                parsed = agent.process_query_to_dict(query, user_id="test_user")

                if parsed.get("query_type") != "detail":
                    print_error(f"Not a detail query (type: {parsed.get('query_type')})")
                    continue

                print_info("Parsed successfully")
                print_info(f"Product: {parsed.get('product_id')}")
                print_info(f"Properties: {parsed.get('properties_to_explain')}")
                print_info(f"Keywords: {parsed.get('query_keywords')}")

                # Get answer
                print_section("Generating answer...")
                answer = orchestrator.answer_detail_query(parsed)

                # Display answer
                print_section("Answer:")
                print(f"{Colors.GREEN}{answer}{Colors.END}")
                print()

            except Exception as e:
                print_error(f"Detail query failed: {e}")
                import traceback
                traceback.print_exc()
                print()

        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            print("\n")
            break


def test_chat():
    """Interactive chat test"""
    print_header("CHAT HANDLER - Interactive Test")

    print_info("This mode tests the chat conversation system")
    print_info("Have a conversation with Strontium!\n")

    # Initialize
    print_section("Initializing Components...")
    agent = initialize_strontium_agent()
    orchestrator = SearchOrchestrator()
    print_success("Components initialized\n")

    print(f"{Colors.BOLD}Try saying:{Colors.END}")
    print("  • Hello!")
    print("  • How are you?")
    print("  • Who are you?")
    print("  • What can you do?")
    print(f"\n{Colors.BOLD}Type 'quit' or 'exit' to return to main menu{Colors.END}\n")

    while True:
        try:
            message = input(f"{Colors.BOLD}You:{Colors.END} ").strip()

            if not message:
                continue

            if message.lower() in ['quit', 'exit', 'q']:
                break

            try:
                # Parse
                parsed = agent.process_query_to_dict(message, user_id="test_user")

                if parsed.get("query_type") != "chat":
                    print_error(f"Not a chat query (type: {parsed.get('query_type')})")
                    continue

                # Get response
                response = orchestrator.handle_chat(parsed)

                # Display response
                print(f"{Colors.CYAN}Strontium:{Colors.END} {response}\n")

            except Exception as e:
                print_error(f"Chat failed: {e}")
                print()

        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            print("\n")
            break


def test_complete_pipeline():
    """Test any query type automatically"""
    print_header("COMPLETE PIPELINE - Interactive Test")

    print_info("This mode automatically routes queries to the appropriate handler")
    print_info("Enter any query (search, detail, or chat) and see the result\n")

    # Initialize
    print_section("Initializing Components...")
    agent = initialize_strontium_agent()
    orchestrator = SearchOrchestrator()
    print_success("Components initialized\n")

    print(f"{Colors.BOLD}Try any query:{Colors.END}")
    print("  • Red cotton shirt under $30 (search)")
    print("  • What material is p-456 made of? (detail)")
    print("  • Hello! (chat)")
    print(f"\n{Colors.BOLD}Type 'quit' or 'exit' to return to main menu{Colors.END}\n")

    while True:
        try:
            query = input(f"{Colors.BOLD}Enter query:{Colors.END} ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                break

            print_section(f"Processing: '{query}'")

            try:
                # Parse
                parsed = agent.process_query_to_dict(query, user_id="test_user")
                query_type = parsed.get("query_type")

                print_info(f"Query type: {query_type.upper()}")

                # Route based on type
                if query_type == "search":
                    results = orchestrator.search_strontium(parsed)
                    print_section("Search Results:")
                    for i, result in enumerate(results, 1):
                        if result.product_ids:
                            print_success(f"Found {len(result.product_ids)} products")
                            print(f"  Top 5: {result.product_ids[:5]}")
                        else:
                            print_error("No products found")

                elif query_type == "detail":
                    answer = orchestrator.answer_detail_query(parsed)
                    print_section("Answer:")
                    print(f"{Colors.GREEN}{answer}{Colors.END}")

                elif query_type == "chat":
                    response = orchestrator.handle_chat(parsed)
                    print_section("Response:")
                    print(f"{Colors.CYAN}{response}{Colors.END}")

                print()

            except Exception as e:
                print_error(f"Processing failed: {e}")
                import traceback
                traceback.print_exc()
                print()

        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            print("\n")
            break


def show_menu():
    """Show main menu"""
    print_header("THE WHOLE STORY - Interactive Testing Suite")

    print(f"{Colors.BOLD}Select a test mode:{Colors.END}\n")
    print("  1. Strontium Parsing Only")
    print("  2. Search Pipeline")
    print("  3. Detail Query System")
    print("  4. Chat Handler")
    print("  5. Complete Pipeline (Auto-route)")
    print("  6. Exit")
    print()


def main():
    """Main interactive test loop"""
    while True:
        try:
            show_menu()
            choice = input(f"{Colors.BOLD}Enter choice (1-6):{Colors.END} ").strip()

            if choice == '1':
                test_strontium_parsing()
            elif choice == '2':
                test_search_pipeline()
            elif choice == '3':
                test_detail_query()
            elif choice == '4':
                test_chat()
            elif choice == '5':
                test_complete_pipeline()
            elif choice == '6':
                print_success("Goodbye!")
                break
            else:
                print_error("Invalid choice. Please enter 1-6.")

        except KeyboardInterrupt:
            print("\n")
            print_success("Goodbye!")
            break
        except EOFError:
            print("\n")
            print_success("Goodbye!")
            break


if __name__ == "__main__":
    main()
