from __future__ import annotations

import unittest
from pathlib import Path

from scripts.validate_evidence import validate


class EvidenceTests(unittest.TestCase):
    def test_committed_ranked_evidence_is_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        summary = validate(root)
        self.assertEqual(summary["ranked_feature_width"], 72)
        self.assertEqual(summary["ranked_steps"], 2806)
        self.assertTrue(summary["training_eligible"])
        self.assertFalse(summary["release_eligible"])
        self.assertFalse(summary["executor_integration_allowed"])


if __name__ == "__main__":
    unittest.main()
