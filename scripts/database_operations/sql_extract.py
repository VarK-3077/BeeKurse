import sqlite3
import json
from typing import List, Dict, Any

DB_PATH = "data/databases/sql/inventory.db"
PRODUCT_TABLE_NAME = "product_table"

DUMMY_IMG_BASE = "https://my-img-bucket-123.s3.ap-south-1.amazonaws.com"


def fetch_products_by_ids(
    product_ids: List[str],
    db_path: str = DB_PATH,
    table_name: str = PRODUCT_TABLE_NAME
) -> Dict[str, Dict[str, Any]]:
    """
    Given a list of product_id values, fetch the matching rows
    and return a dictionary keyed by product_id.

    Each product dict includes:
        - product_id
        - short_id
        - prod_name
        - price
        - rating
        - store
        - image_url  (dummy for now)
        - raw: full DB row
    """

    if not product_ids:
        return {}

    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Prepare query
    placeholders = ", ".join("?" for _ in product_ids)
    query = f"""
        SELECT * FROM {table_name}
        WHERE product_id IN ({placeholders});
    """

    cursor.execute(query, product_ids)

    # Extract column names
    col_names = [desc[0] for desc in cursor.description]

    # Build response
    results: Dict[str, Dict[str, Any]] = {}

    for raw_row in cursor.fetchall():
        row_dict = {}
        for col, val in zip(col_names, raw_row):
            # Decode JSON-like strings
            if isinstance(val, str) and val.strip().startswith("[") and val.strip().endswith("]"):
                try:
                    row_dict[col] = json.loads(val)
                except:
                    row_dict[col] = val
            else:
                row_dict[col] = val

        pid = row_dict.get("product_id")
        if not pid:
            continue

        # ---- Image URL mapping (dummy for now) ----
        # image_url = f"{DUMMY_IMG_BASE}/{pid}.jpg"
        image_id = row_dict.get("imageid")
        image_url = f"https://my-img-bucket-123.s3.ap-south-1.amazonaws.com/{image_id}"

        print(f"DEBUG: {image_url}")
        # image_url = "https://plus.unsplash.com/premium_photo-1762541871245-ffaf4ab5da47?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHwyfHx8ZW58MHx8fHx8"

        # ---- Clean minimal object for WhatsApp backend ----
        results[pid] = {
            "product_id": pid,
            "short_id": row_dict.get("short_id"),
            "prod_name": row_dict.get("prod_name"),
            "price": row_dict.get("price"),
            "rating": row_dict.get("rating"),
            "store": row_dict.get("store"),
            "image_url": image_url,
            "raw": row_dict,     # full DB row in case you need anything else
        }

    conn.close()
    return results
