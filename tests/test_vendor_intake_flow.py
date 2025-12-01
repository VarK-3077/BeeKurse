import json
import os
import tempfile
import unittest

try:
    from backend.vendor_intake_flow import VendorIntakeFlow, VendorRegistry
    from search_agent.database.sql_client import SQLClient
    SKIP_REASON = None
except ModuleNotFoundError as exc:  # pragma: no cover - handled as skip when deps missing
    VendorIntakeFlow = VendorRegistry = SQLClient = None  # type: ignore
    SKIP_REASON = f"missing dependency: {exc}"


@unittest.skipIf(SKIP_REASON is not None, "Vendor intake dependencies missing")
class VendorIntakeFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "inventory.db")
        self.registry_path = os.path.join(self.temp_dir.name, "registry.json")

        # Seed registry with one registered user
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump({"registered": ["+123"]}, f)

        self.sql_client = SQLClient(db_path=self.db_path)

        # Seed one product for similarity checks
        conn = self.sql_client._get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO product_table (product_id, prod_name) VALUES (?, ?)",
            ("p-001", "Red Cotton Shirt"),
        )
        conn.commit()

        registry = VendorRegistry(self.registry_path)
        self.flow = VendorIntakeFlow(sql_client=self.sql_client, registry=registry)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_unregistered_user_gets_registration_link(self):
        resp = self.flow.handle("999", "hello")
        self.assertIn("register", resp["messages"][0]["text"].lower())

    def test_missing_fields_prompt(self):
        resp = self.flow.handle("+123", "add new item")
        self.assertIn("missing", resp["messages"][0]["text"].lower())

    def test_similarity_prompt_for_add(self):
        resp = self.flow.handle("+123", "add Red Cotton Shirt price 199 quantity 5")
        text = resp["messages"][0]["text"].lower()
        self.assertIn("similar", text)
        # Confirm adding as new should complete intake
        resp2 = self.flow.handle("+123", "add new")
        combined = "\n".join(m["text"] for m in resp2["messages"])
        self.assertIn("input processing done", combined.lower())


if __name__ == "__main__":
    unittest.main()
