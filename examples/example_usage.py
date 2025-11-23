"""
Example usage of Kurse Ecommerce Search Orchestrator
"""
from orchestrator import SearchOrchestrator, search
from models import SearchQuery, SQLProduct
from database.sql_client import SQLClient


def example_1_basic_search():
    """Example 1: Basic property search"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Property Search")
    print("="*60)

    query_dict = {
        "product_query": "red shirt under 20",
        "product_basetype": "shirt",
        "properties": [
            ["red color", 1.5],
            ["casual style", 1.0]
        ],
        "literals": [
            ["price", "<", 20.0, 0.1]  # 10% buffer = accept up to $22
        ],
        "is_hq": False
    }

    result = search(query_dict)
    print(f"\nResult: {result}")


def example_2_hurry_query():
    """Example 2: Hurry Query (fast re-purchase)"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Hurry Query (Fast Re-Purchase)")
    print("="*60)

    query_dict = {
        "product_query": "buy same shirt again",
        "product_basetype": "shirt",
        "properties": [],
        "literals": [],
        "is_hq": True,              # Enable HQ fast path
        "prev_productid": "p-001",  # Previous purchase
        "prev_storeid": "s-1234"
    }

    result = search(query_dict)
    print(f"\nResult: {result}")
    print("(Returns immediately if product in stock)")


def example_3_connected_search():
    """Example 3: Connected search with recommendations"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Connected Search (Recommendations)")
    print("="*60)

    query_dict = {
        "product_query": "similar casual shirts",
        "product_basetype": "shirt",
        "properties": [
            ["casual", 1.0]
        ],
        "literals": [
            ["price", "<", 25.0, 0.1]
        ],
        "is_hq": False,
        "prev_productid": "p-001"  # Products related to p-001 get +0.5 bonus
    }

    result = search(query_dict)
    print(f"\nResult: {result}")
    print("(Products connected to p-001 via KG get bonus score)")


def example_4_store_context():
    """Example 4: Store-based boosting"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Store-Based Boosting")
    print("="*60)

    query_dict = {
        "product_query": "casual shirt",
        "product_basetype": "shirt",
        "properties": [
            ["casual", 1.0]
        ],
        "literals": [
            ["price", "<", 30.0, 0.1]
        ],
        "is_hq": False,
        "prev_storeid": "s-1234"  # Products from s-1234 get +0.2 bonus
    }

    result = search(query_dict)
    print(f"\nResult: {result}")
    print("(Products from store s-1234 get bonus score)")


def example_5_detailed_results():
    """Example 5: Get detailed score breakdown"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Detailed Score Breakdown")
    print("="*60)

    query = SearchQuery(
        product_query="red casual shirt",
        product_basetype="shirt",
        properties=[
            ["red", 1.5],
            ["casual", 1.0]
        ],
        literals=[
            ["price", "<", 20.0, 0.1]
        ],
        is_hq=False,
        prev_productid="p-001"
    )

    with SearchOrchestrator() as orchestrator:
        detailed_results = orchestrator.get_detailed_results(query)

        print("\nTop 5 Results with Score Breakdown:")
        print("-" * 60)
        for i, score in enumerate(detailed_results[:5], 1):
            print(f"{i}. Product ID: {score.product_id}")
            print(f"   Property Score:  {score.property_score:.3f}")
            print(f"   Connected Score: {score.connected_score:.3f}")
            print(f"   Final Score:     {score.final_score:.3f}")
            print()


def example_6_multiple_literals():
    """Example 6: Multiple literal constraints"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Multiple Literal Constraints")
    print("="*60)

    query_dict = {
        "product_query": "medium casual shirt under 20",
        "product_basetype": "shirt",
        "properties": [
            ["casual", 1.2]
        ],
        "literals": [
            ["price", "<", 20.0, 0.1],   # price < $20 with 10% buffer
            ["size", "=", "M", 0.0],      # exact size match
            ["stock", ">", 5, 0.0]        # at least 5 in stock
        ],
        "is_hq": False
    }

    result = search(query_dict)
    print(f"\nResult: {result}")
    print("(Filtered by price, size, and stock)")


def example_7_api_integration():
    """Example 7: Integration with REST API (pseudo-code)"""
    print("\n" + "="*60)
    print("EXAMPLE 7: REST API Integration (Pseudo-Code)")
    print("="*60)

    api_payload = """
    POST /api/search
    Content-Type: application/json

    {
      "product_query": "red shirt under 20",
      "product_basetype": "shirt",
      "properties": [
        ["red color", 1.5],
        ["casual style", 1.0]
      ],
      "literals": [
        ["price", "<", 20.0, 0.1]
      ],
      "is_hq": false,
      "prev_productid": null,
      "prev_storeid": "s-1234"
    }
    """

    print(api_payload)

    api_handler_code = """
    # Flask/FastAPI handler
    @app.post("/api/search")
    def search_products(query: SearchQuery):
        orchestrator = SearchOrchestrator()
        result = orchestrator.search(query)
        orchestrator.close()
        return result
    """

    print("\nHandler Code:")
    print(api_handler_code)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("KURSE ECOMMERCE SEARCH ORCHESTRATOR - USAGE EXAMPLES")
    print("="*60)

    # Run examples
    try:
        example_1_basic_search()
        example_2_hurry_query()
        example_3_connected_search()
        example_4_store_context()
        example_5_detailed_results()
        example_6_multiple_literals()
        example_7_api_integration()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED")
        print("="*60)
        print("\nNote: Some examples may return empty results if VDB/KG")
        print("are not populated. See README.md for setup instructions.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
