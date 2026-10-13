import csv
import tempfile
import unittest
from pathlib import Path

import pytest


class AppEndpointTests(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _client(self, client):
        self.client = client

    def _make_csv(self, rows):
        tmp_dir = tempfile.TemporaryDirectory()
        csv_path = Path(tmp_dir.name) / "sample.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(tmp_dir.cleanup)
        return str(csv_path)

    def test_root_serves_frontend(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Easy-ML", response.data)

    def test_get_models_endpoint(self):
        response = self.client.get("/get_models")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("MLClassification", payload)
        self.assertIn("MLRegression", payload)

    def test_settings_info_endpoint(self):
        response = self.client.get("/settings/info")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("data_dir", payload)
        self.assertIn("version", payload)

    def test_load_data_and_class_distribution(self):
        csv_path = self._make_csv(
            [
                {"feature": 1, "target": "a"},
                {"feature": 2, "target": "a"},
                {"feature": 3, "target": "b"},
            ]
        )

        load_response = self.client.post("/load_data", json={"path": csv_path})
        self.assertEqual(load_response.status_code, 200)
        saved_path = load_response.get_json()["saved_path"]

        distribution_response = self.client.post(
            "/class_distribution",
            json={"dataset_path": saved_path, "target_column": "target"},
        )
        self.assertEqual(distribution_response.status_code, 200)
        payload = distribution_response.get_json()
        self.assertEqual(payload["num_classes"], 2)
        self.assertEqual(payload["total_samples"], 3)

    def test_nlp_models_endpoint(self):
        response = self.client.get("/nlp/get_models")
        self.assertEqual(response.status_code, 200)
        self.assertIn("NLPTextClassification", response.get_json())


if __name__ == "__main__":
    unittest.main()
