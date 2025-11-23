"""
Predefined Debug Queries
Automatically runs test queries and shows complete JSON trace
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from strontium_agent_debug import StrontiumAgentDebug


def run_all_examples(use_nvidia=False):
    """Run all example queries and show outputs"""

    print("\n" + "=" * 80)
    print(" " * 20 + "STRONTIUM DEBUG - EXAMPLE QUERIES")
    print("=" * 80)
    print()
    print(f"LLM Mode: {'NVIDIA API' if use_nvidia else 'Mock Parser'}")
    print(f"Mock Data: mock_data/ (3 users with profiles & purchase history)")
    print()

    # Initialize agent
    agent = StrontiumAgentDebug(
        mock_data_dir="mock_data",
        use_nvidia=use_nvidia,
        verbose=True
    )

    examples = [
        {
            "name": "Example 1: Simple Search with Literals",
            "query": "Red cotton shirt under $30",
            "user": "default_user",
            "description": "Tests: property extraction, literal constraints, user context"
        },
        {
            "name": "Example 2: Property Inference",
            "query": "I want a pillow for neck pain",
            "user": "default_user",
            "description": "Tests: LLM inference (neck pain → orthopedic, firm support)"
        },
        {
            "name": "Example 3: Similarity Query",
            "query": "Shoes similar to p-123 but cheaper",
            "user": "default_user",
            "description": "Tests: prev_products extraction, similarity search"
        },
        {
            "name": "Example 4: HQ Detection (Explicit)",
            "query": "I need my usual tomatoes",
            "user": "user-001",
            "description": "Tests: HQ detection ('my usual'), purchase history lookup"
        },
        {
            "name": "Example 5: HQ Detection (Implicit)",
            "query": "Get me some tomatoes",
            "user": "user-001",
            "description": "Tests: HQ inferred from purchase history (3+ regular purchases)"
        },
        {
            "name": "Example 6: Detail Query (Specific)",
            "query": "What material is product p-456 made of?",
            "user": "default_user",
            "description": "Tests: Detail query classification, property extraction"
        },
        {
            "name": "Example 7: Detail Query (General)",
            "query": "Tell me more about p-789",
            "user": "default_user",
            "description": "Tests: General detail query (properties_to_explain: ['*'])"
        },
        {
            "name": "Example 8: User Context Enrichment",
            "query": "I want a shirt",
            "user": "user-001",
            "description": "Tests: User preferences added (eco-conscious, brands, size, style)"
        },
        {
            "name": "Example 9: Multiple Constraints",
            "query": "Casual blue jeans under $50 with rating above 4",
            "user": "default_user",
            "description": "Tests: Multiple properties + multiple literals"
        },
        {
            "name": "Example 10: Size Literal",
            "query": "Red sneakers size 10",
            "user": "default_user",
            "description": "Tests: Size as literal constraint"
        },
    ]

    for i, example in enumerate(examples, 1):
        print("\n" + "▓" * 80)
        print(f"▓  {example['name']}")
        print("▓" * 80)
        print(f"\nDescription: {example['description']}")
        print()

        try:
            agent.process_query_to_dict(example["query"], example["user"])
            print(f"\n✓ Example {i} completed successfully\n")

        except Exception as e:
            print(f"\n✗ Example {i} failed: {e}\n")
            import traceback
            traceback.print_exc()

        # Separator
        print("\n" + "─" * 80 + "\n")

        # Pause between examples (except last one)
        if i < len(examples):
            input("Press Enter to continue to next example...")
            print("\n" * 2)

    print("\n" + "=" * 80)
    print(" " * 25 + "ALL EXAMPLES COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys

    # Check for --nvidia flag
    use_nvidia = '--nvidia' in sys.argv

    if use_nvidia:
        print("\n⚠ WARNING: Using NVIDIA API - this will make real API calls")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(0)

    run_all_examples(use_nvidia=use_nvidia)
