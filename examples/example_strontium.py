"""
Example usage of Strontium Curator Agent

This demonstrates how to use Strontium to transform natural language queries
into structured JSON for the SearchOrchestrator.
"""

from strontium import StrontiumAgent, process_query
import json


def main():
    print("=" * 70)
    print("STRONTIUM CURATOR AGENT - EXAMPLES")
    print("=" * 70)

    # Initialize agent
    agent = StrontiumAgent(mock_data_dir="mock_data")

    # Example 1: Simple search query
    print("\n\nExample 1: Simple Search Query")
    print("-" * 70)
    query = "Red cotton shirt under $30"
    result = agent.process_query_to_dict(query, "default_user")
    print(f"Input: {query}")
    print(f"Output:\n{json.dumps(result, indent=2)}")

    # Example 2: HQ query (my usual)
    print("\n\nExample 2: HQ Query - Explicit")
    print("-" * 70)
    query = "I need my usual tomatoes"
    result = agent.process_query_to_dict(query, "user-001")
    print(f"Input: {query}")
    print(f"Output:\n{json.dumps(result, indent=2)}")

    # Example 3: Property inference (neck pain pillow)
    print("\n\nExample 3: Property Inference")
    print("-" * 70)
    query = "pillow for neck pain"
    result = agent.process_query_to_dict(query, "default_user")
    print(f"Input: {query}")
    print(f"Output:\n{json.dumps(result, indent=2)}")

    # Example 4: Similarity query
    print("\n\nExample 4: Similarity Query")
    print("-" * 70)
    query = "Shoes like p-123 but cheaper"
    result = agent.process_query_to_dict(query, "default_user")
    print(f"Input: {query}")
    print(f"Output:\n{json.dumps(result, indent=2)}")

    # Example 5: Detail query
    print("\n\nExample 5: Detail Query")
    print("-" * 70)
    query = "What material is product p-456 made of?"
    result = agent.process_query_to_dict(query, "default_user")
    print(f"Input: {query}")
    print(f"Output:\n{json.dumps(result, indent=2)}")

    # Example 6: Using convenience function
    print("\n\nExample 6: Convenience Function")
    print("-" * 70)
    result = process_query(
        "Red shirt under $25",
        user_id="default_user",
        mock_data_dir="mock_data"
    )
    print("Input: Red shirt under $25")
    print(f"Output:\n{json.dumps(result, indent=2)}")

    # Example 7: User preference enrichment
    print("\n\nExample 7: User Preference Enrichment")
    print("-" * 70)
    print("User: user-001 (eco-conscious, likes Nike, size M)")
    query = "I want a shirt"
    result = agent.process_query_to_dict(query, "user-001")
    print(f"Input: {query}")
    print(f"Output:\n{json.dumps(result, indent=2)}")
    print("Note: Properties include eco_friendly, organic, brand_Nike, size_M, style preferences")

    print("\n" + "=" * 70)
    print("END OF EXAMPLES")
    print("=" * 70)


if __name__ == "__main__":
    main()
