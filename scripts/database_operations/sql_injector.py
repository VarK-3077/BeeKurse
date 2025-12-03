import os
import json
import sqlite3
import random
import string
from typing import Any, Dict, Union, Optional
import csv

from sentence_transformers import SentenceTransformer

# Lazy global model
_ST_MODEL = None
DB_PATH = "data/databases/sql/inventory.db"
PRODUCT_TABLE_NAME = "product_table"
CONTACTS_TABLE_NAME = "store_contacts"


# [CHANGE 1] Added 'gender' to the required columns list
REQUIRED_FIELDS = [
    'brand', 'category', 'colour', 'descrption', 'dimensions', 'imageid',
    'price', 'prod_name', 'product_id', 'quantity', 'qunatityunit',
    'rating', 'size', 'stock', 'store', 'subcategory', 'subcategoryid', 
    'short_id', 'gender'
]


def generate_short_id(length: int = 4) -> str:
    """Generate a unique short ID (e.g., A1B2) for WhatsApp references."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

def add_subcategory_embedding_and_save(
    product_json: Union[Dict[str, Any], str],
    db_path: str = DB_PATH,
    table_name: str = "product_table",
) -> Dict[str, Any]:
    """
    1. Take a product JSON.
    2. Extract 'gender' from 'otherproperties' if it exists.
    3. Recompute 'subcategoryid'.
    4. Insert into SQLite.
    """

    # ---- Parse input JSON ----
    if isinstance(product_json, str):
        raw = json.loads(product_json)
    else:
        raw = dict(product_json)

    subcat = raw.get("subcategory")
    if not subcat:
        raise ValueError("Input JSON must contain a non-empty 'subcategory' field")

    # [CHANGE 2] Extract gender from otherproperties if explicitly missing at top level
    if "gender" not in raw:
        other_props = raw.get("otherproperties")
        
        # Parse otherproperties if it is a JSON string
        if isinstance(other_props, str):
            try:
                other_props = json.loads(other_props)
            except:
                other_props = {}
        
        # If it's a dict, try to get gender
        if isinstance(other_props, dict):
            raw["gender"] = other_props.get("gender")

    # ---- Generate short_id if not provided ----
    if not raw.get("short_id"):
        raw["short_id"] = generate_short_id()

    # ---- Load / reuse SentenceTransformer model ----
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embedding and overwrite any existing subcategoryid
    emb = _ST_MODEL.encode([subcat], convert_to_numpy=True)[0]
    raw["subcategoryid"] = emb.tolist()

    # ---- Normalize to exactly REQUIRED_FIELDS ----
    # If something missing, we put None; if extra keys exist, we drop them.
    row = {field: raw.get(field) for field in REQUIRED_FIELDS}

    # ---- Connect to SQLite ----
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # ---- Create table if it doesn't exist ----
        # All columns as TEXT (SQLite is type-flexible; JSON strings are fine).
        cols_def = ", ".join(f'"{c}" TEXT' for c in REQUIRED_FIELDS)
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols_def});'
        cursor.execute(create_sql)

        # ---- Insert row ----
        cols = REQUIRED_FIELDS
        col_names_sql = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("?" for _ in cols)

        insert_sql = f"""
            INSERT INTO "{table_name}" ({col_names_sql})
            VALUES ({placeholders});
        """

        values = []
        for c in cols:
            v = row[c]
            # serialize complex types (subcategoryid is a list)
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            elif v is not None and not isinstance(v, (str, int, float)):
                v = str(v)
            values.append(v)

        cursor.execute(insert_sql, values)
        conn.commit()
    finally:
        conn.close()

     # return the normalized row (with fresh subcategoryid)
    return row


def load_store_contacts_to_db(
    csv_path: str,
    db_path: str = DB_PATH, 
    table_name: str = "store_contacts"
):
    """
    Reads the CSV with columns: store, contact_number
    Inserts into SQLite table (auto-created if not exists).
    """

    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            store TEXT,
            contact_number TEXT
        );
    """)

    # Read CSV + insert rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(row["store"], row["contact_number"]) for row in reader]

    cursor.executemany(
        f"INSERT INTO {table_name} (store, contact_number) VALUES (?, ?)",
        rows
    )

    conn.commit()
    conn.close()

    print(f"Inserted {len(rows)} rows into '{table_name}' table.")