"""
SQL Database Client for inventory and product data
"""
import sqlite3
import numpy as np
import pickle
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from search_agent.models import SQLProduct
from config.config import Config

config = Config

# Gender mapping: user preference -> acceptable product genders
GENDER_MAPPING = {
    "male": ["Men", "Boys", "Unisex"],
    "female": ["Women", "Girls", "Unisex"],
    "both": None,  # No filtering
}


class SQLClient:
    """Client for SQL database operations"""

    def __init__(self, db_path: str = None):
        """
        Initialize SQL client

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or config.SQL_DB_PATH
        self._connection = None
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Ensure database and schema exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Create products table if not exists - ACTUAL PRODUCTION SCHEMA
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS product_table (
                product_id TEXT PRIMARY KEY,
                prod_name TEXT NOT NULL,
                store TEXT,
                category TEXT,
                subcategory TEXT,
                brand TEXT,
                colour TEXT,
                description TEXT,
                dimensions TEXT,
                imageid TEXT,
                price REAL,
                quantity INTEGER,
                quantityunit TEXT,
                rating REAL,
                size TEXT,
                stock INTEGER,
                store_contact TEXT,
                store_location TEXT,
                other_properties TEXT
            )
        """)

        # Create product_embeddings table for subcategory embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS product_embeddings (
                product_id TEXT PRIMARY KEY,
                subcategory TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (product_id) REFERENCES product_table(product_id)
            )
        """)

        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON product_table(category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_subcategory ON product_table(subcategory)
        """)

        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection"""
        if self._connection is None:
            # Use check_same_thread=False to allow multi-threaded access
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None

    def get_product_by_id(self, product_id: str, store_id: str = None) -> Optional[SQLProduct]:
        """
        Get product by ID and optionally store_id

        Args:
            product_id: Product ID
            store_id: Optional store ID filter

        Returns:
            SQLProduct or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if store_id:
            cursor.execute(
                "SELECT * FROM product_table WHERE product_id = ? AND store = ?",
                (product_id, store_id)
            )
        else:
            cursor.execute(
                "SELECT * FROM product_table WHERE product_id = ?",
                (product_id,)
            )

        row = cursor.fetchone()
        if row:
            # Convert row to a mutable dictionary
            product_data = dict(row)
            
            # Deserialize JSON strings back to dicts
            if product_data.get('store_location') and isinstance(product_data['store_location'], str):
                product_data['store_location'] = json.loads(product_data['store_location'])
            if product_data.get('other_properties') and isinstance(product_data['other_properties'], str):
                product_data['other_properties'] = json.loads(product_data['other_properties'])
                
            return SQLProduct(**product_data)
        return None

    def get_products_by_ids(self, product_ids: List[str]) -> Dict[str, SQLProduct]:
        """
        Get multiple products by IDs

        Args:
            product_ids: List of product IDs

        Returns:
            Dictionary mapping product_id to SQLProduct
        """
        if not product_ids:
            return {}

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(product_ids))
        cursor.execute(
            f"SELECT * FROM product_table WHERE product_id IN ({placeholders})",
            product_ids
        )

        results = {}
        for row in cursor.fetchall():
            # Convert row to a mutable dictionary
            product_data = dict(row)
            
            # Deserialize JSON strings back to dicts
            if product_data.get('store_location') and isinstance(product_data['store_location'], str):
                product_data['store_location'] = json.loads(product_data['store_location'])
            if product_data.get('other_properties') and isinstance(product_data['other_properties'], str):
                product_data['other_properties'] = json.loads(product_data['other_properties'])
            
            product = SQLProduct(**product_data)
            results[product.product_id] = product

        return results

    def get_products_by_store(self, store_id: str, category: str = None, subcategory: str = None) -> List[SQLProduct]:
        """
        Get all products from a store, optionally filtered by category and/or subcategory

        Args:
            store_id: Store ID
            category: Optional product category filter
            subcategory: Optional product subcategory filter

        Returns:
            List of SQLProduct
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM product_table WHERE store = ?"
        params = [store_id]

        if category:
            query += " AND category = ?"
            params.append(category)

        if subcategory:
            query += " AND subcategory = ?"
            params.append(subcategory)

        cursor.execute(query, params)

        product_list = []
        for row in cursor.fetchall():
            # Convert row to a mutable dictionary
            product_data = dict(row)
            
            # Deserialize JSON strings back to dicts
            if product_data.get('store_location') and isinstance(product_data['store_location'], str):
                product_data['store_location'] = json.loads(product_data['store_location'])
            if product_data.get('other_properties') and isinstance(product_data['other_properties'], str):
                product_data['other_properties'] = json.loads(product_data['other_properties'])
                
            product_list.append(SQLProduct(**product_data))
        return product_list

    def filter_products_by_literals(
        self,
        product_ids: List[str],
        literals: List[tuple]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Filter products by literal constraints and return values

        Args:
            product_ids: List of product IDs to filter
            literals: List of (field_name, operator, value, buffer) tuples

        Returns:
            Dictionary mapping product_id to literal values
            {
                "p-123": {"price": 19.99, "size": "M"},
                "p-456": {"price": 15.00, "size": "L"}
            }
        """
        if not product_ids:
            return {}

        # Get all products
        products = self.get_products_by_ids(product_ids)

        result = {}
        for product_id, product in products.items():
            # Check all literal constraints
            passes_all = True
            literal_values = {}

            for field_name, operator, value, buffer in literals:
                # Get field value from product
                product_value = getattr(product, field_name, None)

                if product_value is None:
                    passes_all = False
                    break

                # Store the value
                literal_values[field_name] = product_value

                # Apply operator with buffer
                if not self._check_operator(product_value, operator, value, buffer):
                    passes_all = False
                    break

            if passes_all:
                result[product_id] = literal_values

        return result

    def filter_products_by_gender(
        self,
        product_ids: List[str],
        user_gender_preference: str
    ) -> List[str]:
        """
        Filter products by gender from other_properties JSON.

        Args:
            product_ids: List of product IDs to filter
            user_gender_preference: User's gender preference ("male", "female", "both")

        Returns:
            List of product IDs that match the gender preference.
            Products without gender (e.g., electronics, grocery) pass through.
        """
        if not product_ids:
            return []

        # "both" or unknown means no filtering
        acceptable_genders = GENDER_MAPPING.get(user_gender_preference)
        if acceptable_genders is None:
            return product_ids

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(product_ids))

        # Build gender conditions for acceptable values
        gender_conditions = " OR ".join(
            f"json_extract(other_properties, '$.gender') = ?"
            for _ in acceptable_genders
        )

        # Products without gender in other_properties pass through
        # (e.g., Electronics, Grocery don't have gender)
        query = f"""
            SELECT product_id FROM product_table
            WHERE product_id IN ({placeholders})
            AND (
                other_properties IS NULL
                OR json_extract(other_properties, '$.gender') IS NULL
                OR {gender_conditions}
            )
        """

        params = list(product_ids) + acceptable_genders
        cursor.execute(query, params)

        return [row[0] for row in cursor.fetchall()]

    def _check_operator(self, product_value: Any, operator: str, target_value: Any, buffer: float) -> bool:
        """
        Check if product_value satisfies operator constraint with buffer

        Args:
            product_value: Actual value from product
            operator: Comparison operator
            target_value: Target value to compare against
            buffer: Buffer tolerance (e.g., 0.1 = 10%)

        Returns:
            True if constraint satisfied (within buffer), False otherwise
        """
        # For numeric comparisons, apply buffer
        if isinstance(target_value, (int, float)) and isinstance(product_value, (int, float)):
            buffered_value = target_value * (1 + buffer) if operator in ["<", "<="] else target_value * (1 - buffer)

            if operator == "<":
                return product_value < buffered_value
            elif operator == "<=":
                return product_value <= buffered_value
            elif operator == ">":
                return product_value > buffered_value
            elif operator == ">=":
                return product_value >= buffered_value
            elif operator == "=":
                # For equality, allow small tolerance
                tolerance = abs(target_value * buffer)
                return abs(product_value - target_value) <= tolerance
            elif operator == "!=":
                tolerance = abs(target_value * buffer)
                return abs(product_value - target_value) > tolerance

        # For string/exact comparisons
        else:
            if operator == "=":
                return product_value == target_value
            elif operator == "!=":
                return product_value != target_value

        return False

    def insert_product(self, product: SQLProduct):
        """Insert a new product"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Handle price conversion from string (e.g., "105,272") to float
        parsed_price = None
        if product.price is not None:
            try:
                # Remove commas and convert to float
                parsed_price = float(str(product.price).replace(",", ""))
            except (ValueError, TypeError):
                # Log or handle error if price cannot be parsed
                parsed_price = None # Or raise an error

        # Serialize dict fields to JSON strings
        store_location_json = json.dumps(product.store_location) if product.store_location is not None else None
        other_properties_json = json.dumps(product.other_properties) if product.other_properties is not None else None

        cursor.execute(
            """
            INSERT OR REPLACE INTO product_table
            (product_id, prod_name, store, category, subcategory, brand, colour, description, dimensions, imageid, price, quantity, quantityunit, rating, size, stock, store_contact, store_location, other_properties)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                product.product_id,
                product.prod_name,
                product.store,
                product.category,
                product.subcategory,
                product.brand,
                product.colour,
                product.description,
                product.dimensions,
                product.imageid,
                parsed_price,  # Use parsed price
                product.quantity,
                product.quantityunit, # Corrected field name
                product.rating,
                product.size,
                product.stock,
                product.store_contact, # New field
                store_location_json, # New field, JSON string
                other_properties_json # New field, JSON string
            )
        )

        conn.commit()

    def store_embedding(self, product_id: str, subcategory: str, embedding: np.ndarray):
        """
        Store subcategory embedding for a product

        Args:
            product_id: Product ID
            subcategory: Subcategory text
            embedding: Numpy array of embedding values
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Serialize embedding to BLOB using pickle
        embedding_blob = pickle.dumps(embedding)

        cursor.execute(
            """
            INSERT OR REPLACE INTO product_embeddings
            (product_id, subcategory, embedding)
            VALUES (?, ?, ?)
            """,
            (product_id, subcategory, embedding_blob)
        )

        conn.commit()

    def get_embedding(self, product_id: str) -> Optional[np.ndarray]:
        """
        Get subcategory embedding for a product

        Args:
            product_id: Product ID

        Returns:
            Numpy array of embedding values, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT embedding FROM product_embeddings WHERE product_id = ?",
            (product_id,)
        )

        row = cursor.fetchone()
        if row:
            return pickle.loads(row['embedding'])
        return None

    def get_embeddings(self, product_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Get subcategory embeddings for multiple products

        Args:
            product_ids: List of product IDs

        Returns:
            Dictionary mapping product_id to embedding numpy array
        """
        if not product_ids:
            return {}

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(product_ids))
        cursor.execute(
            f"SELECT product_id, embedding FROM product_embeddings WHERE product_id IN ({placeholders})",
            product_ids
        )

        results = {}
        for row in cursor.fetchall():
            product_id = row['product_id']
            embedding = pickle.loads(row['embedding'])
            results[product_id] = embedding

        return results

    def get_product_subcategories(self, product_ids: List[str]) -> Dict[str, str]:
        """
        Get subcategories for multiple products

        Args:
            product_ids: List of product IDs

        Returns:
            Dictionary mapping product_id to subcategory string
        """
        if not product_ids:
            return {}

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(product_ids))
        cursor.execute(
            f"SELECT product_id, subcategory FROM product_table WHERE product_id IN ({placeholders})",
            product_ids
        )

        results = {}
        for row in cursor.fetchall():
            results[row['product_id']] = row['subcategory']

        return results

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
