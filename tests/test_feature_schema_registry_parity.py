from __future__ import annotations

import json
import unittest
from pathlib import Path

from aba_deep_learning.feature_schemas import list_feature_schemas


class FeatureSchemaRegistryParityTests(unittest.TestCase):
    def test_json_registry_matches_python_registry(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "contracts" / "feature_schemas_v1.json").read_text(
                encoding="utf-8"
            )
        )
        json_schemas = {item["schema_id"]: item for item in payload["schemas"]}
        python_schemas = {item.schema_id: item for item in list_feature_schemas()}

        self.assertEqual(set(json_schemas), set(python_schemas))
        for schema_id, schema in python_schemas.items():
            item = json_schemas[schema_id]
            self.assertEqual(item["width"], schema.width)
            self.assertEqual(item["version"], schema.version)
            self.assertEqual(
                item["compatibility_group"],
                schema.compatibility_group,
            )
            self.assertEqual(set(item["aliases"]), set(schema.aliases))
            self.assertEqual(
                item.get("reference_producer_sha256"),
                schema.reference_producer_sha256,
            )

    def test_ranked_schema_is_bound_to_verified_watcher(self) -> None:
        ranked = next(
            schema
            for schema in list_feature_schemas()
            if schema.schema_id == "ranked_explicit_v3_72"
        )
        self.assertEqual(
            ranked.reference_producer_sha256,
            "c695c08783c53b8cf7c7a9741ed6dd433a3ffb75d9f22af0"
            "b109f380e19dbebd",
        )


if __name__ == "__main__":
    unittest.main()
