import os
import json
import sqlite3
import random
import string
from typing import Any, Dict, Union, Optional, Tuple
import csv

from sentence_transformers import SentenceTransformer

# Lazy global model
_ST_MODEL = None
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "data/databases/sql/inventory.db"
PRODUCT_TABLE_NAME = "product_table"
CONTACTS_TABLE_NAME = "store_contacts"

# Canonical schema for product_table (order matters for inserts)
DB_COLUMNS: Dict[str, str] = {
    "product_id": "TEXT PRIMARY KEY",
    "prod_name": "TEXT",
    "store": "TEXT",
    "store_contact": "TEXT",
    "store_location": "TEXT",
    "category": "TEXT",
    "subcategory": "TEXT",
    "subcategoryid": "TEXT",
    "brand": "TEXT",
    "colour": "TEXT",
    "description": "TEXT",
    "dimensions": "TEXT",
    "imageid": "TEXT",
    "price": "REAL",
    "quantity": "INTEGER",
    "quantityunit": "TEXT",
    "rating": "REAL",
    "size": "TEXT",
    "stock": "INTEGER",
    "other_properties": "TEXT",
    "short_id": "TEXT",
    # actually in json format
    "other_properties": "TEXT",
}

# Aliases we accept from legacy JSONs
FIELD_ALIASES = {
    "descrption": "description",
    "qunatityunit": "quantityunit",
    "otherproperties": "other_properties",
}


def generate_short_id(length: int = 4) -> str:
    """Generate a unique short ID (e.g., A1B2) for WhatsApp references."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

def _load_product(product_json: Union[Dict[str, Any], str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Accept a dict, JSON string, or filesystem path and return a product dict
    and the originating path (if any).
    """
    source_path: Optional[str] = None

    if isinstance(product_json, str):
        if os.path.isfile(product_json):
            source_path = product_json
            with open(product_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = json.loads(product_json)
    else:
        raw = dict(product_json)

    # Normalize legacy keys
    for old, new in FIELD_ALIASES.items():
        if old in raw and new not in raw:
            raw[new] = raw.pop(old)

    return raw, source_path


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return None


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _ensure_table(cursor: sqlite3.Cursor, table_name: str):
    """Create product_table (or add missing columns) to match DB_COLUMNS."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    exists = cursor.fetchone() is not None

    if not exists:
        cols_def = []
        for col, col_type in DB_COLUMNS.items():
            cols_def.append(f'"{col}" {col_type}')
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(cols_def)});'
        cursor.execute(create_sql)
        return

    cursor.execute(f'PRAGMA table_info("{table_name}")')
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col, col_type in DB_COLUMNS.items():
        if col not in existing_cols:
            cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {col_type}')


def add_subcategory_embedding_and_save(
    product_json: Union[Dict[str, Any], str],
    db_path: str = DB_PATH,
    table_name: str = PRODUCT_TABLE_NAME,
    save_json_embedding: bool = True,
    json_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    1. Take a product JSON (dict, JSON string, or path to a JSON file).
    2. Compute subcategory embedding.
    3. Persist embedding back into the JSON (if we have a path).
    4. Insert into SQLite with normalized columns.
    """

    raw, source_path = _load_product(product_json)

    subcat = raw.get("subcategory")
    if not subcat:
        raise ValueError("Input JSON must contain a non-empty 'subcategory' field")

    # Fill optional fields to keep the schema consistent
    raw.setdefault("store_contact", None)
    raw.setdefault("store_location", None)
    raw.setdefault("other_properties", None)
    raw.setdefault("description", None)
    raw.setdefault("quantityunit", None)
    raw.setdefault("store", None)
    raw.setdefault("short_id", None)

    # Generate a short id only if missing
    if not raw.get("short_id"):
        raw["short_id"] = generate_short_id()

    # ---- Load / reuse SentenceTransformer model ----
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(MODEL_NAME)

    # Compute embedding and overwrite any existing subcategoryid
    emb = _ST_MODEL.encode([subcat], convert_to_numpy=True)[0]
    raw["subcategoryid"] = emb.tolist()

    # Optionally write the embedding back to the JSON file
    target_json_path = json_output_path or source_path
    if save_json_embedding and target_json_path:
        with open(target_json_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=4, ensure_ascii=False)

    # Prepare row for DB insertion with normalized/parsed values
    db_row: Dict[str, Any] = {}
    for col in DB_COLUMNS.keys():
        value = raw.get(col)
        if col == "price":
            value = _parse_float(value)
        elif col in {"quantity", "stock"}:
            value = _parse_int(value)
        elif col == "rating":
            value = _parse_float(value)

        db_row[col] = value

    # ---- Connect to SQLite ----
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        _ensure_table(cursor, table_name)

        cols = list(DB_COLUMNS.keys())
        col_names_sql = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("?" for _ in cols)

        insert_sql = f"""
            INSERT OR REPLACE INTO "{table_name}" ({col_names_sql})
            VALUES ({placeholders});
        """

        values = []
        for c in cols:
            v = db_row.get(c)
            # serialize complex types (subcategoryid/store_location/other_properties/dimensions)
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
    return raw


def embed_and_inject_directory(
    dir_path: str,
    db_path: str = DB_PATH,
    table_name: str = PRODUCT_TABLE_NAME,
    save_json_embedding: bool = True,
) -> Dict[str, Any]:
    """
    Walk a directory tree, process every .json file with add_subcategory_embedding_and_save,
    and return a summary dict.
    """
    processed = 0
    errors = []

    for root, _dirs, files in os.walk(dir_path):
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                add_subcategory_embedding_and_save(
                    fpath,
                    db_path=db_path,
                    table_name=table_name,
                    save_json_embedding=save_json_embedding,
                )
                processed += 1
            except Exception as exc:  # Keep going on errors, but record them
                errors.append({"file": fpath, "error": str(exc)})

    summary = {"processed": processed, "errors": errors}
    print(f"Processed {processed} JSON files in '{dir_path}'. Errors: {len(errors)}")
    if errors:
        for err in errors:
            print(f" - {err['file']}: {err['error']}")

    return summary


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
 