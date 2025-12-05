import os
import sqlite3
import json
from typing import List, Dict, Any

from config.config import config

# Use vendor test DB if configured, otherwise main inventory
DB_PATH = config.VENDOR_TEST_DB_PATH if config.USE_VENDOR_TEST_DB else config.SQL_DB_PATH
PRODUCT_TABLE_NAME = "product_table"

# Load S3 base URL from environment
S3_IMAGE_BASE_URL = os.getenv("S3_IMAGE_BASE_URL")
if not S3_IMAGE_BASE_URL:
    raise ValueError("S3_IMAGE_BASE_URL environment variable is required")

def fetch_products_by_ids(
    product_ids: List[str],
    db_path: str = DB_PATH,
    table_name: str = PRODUCT_TABLE_NAME
) -> Dict[str, Dict[str, Any]]:
    """
    Given a list of product_id values, fetch the matching rows
    and return a dictionary keyed by product_id.
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
            if isinstance(val, str) and val.strip().startswith(("{", "[")):
                try:
                    row_dict[col] = json.loads(val)
                except:
                    row_dict[col] = val
            else:
                row_dict[col] = val
        pid = row_dict.get("product_id")
        if not pid:
            continue
        # ---- Image URL mapping ----
        image_id = row_dict.get("imageid")  # Already includes "images/..." path
        image_url = f"{S3_IMAGE_BASE_URL}/{image_id}" if image_id else ""
        # ---- Complete product object ----
        results[pid] = {
            "product_id": pid,
            "short_id": row_dict.get("short_id"),
            "prod_name": row_dict.get("prod_name"),
            "price": row_dict.get("price"),
            "rating": row_dict.get("rating"),
            "store": row_dict.get("store"),
            "store_contact": row_dict.get("store_contact"),
            "store_location": row_dict.get("store_location"),
            "brand": row_dict.get("brand"),
            "colour": row_dict.get("colour"),
            "description": row_dict.get("description"),
            "category": row_dict.get("category"),
            "subcategory": row_dict.get("subcategory"),
            "stock": row_dict.get("stock"),
            "quantity": row_dict.get("quantity"),
            "quantityunit": row_dict.get("quantityunit"),
            "size": row_dict.get("size"),
            "dimensions": row_dict.get("dimensions"),
            "other_properties": row_dict.get("other_properties"),
            "imageid": image_id,
            "image_url": image_url,
        }
    conn.close()
    return results