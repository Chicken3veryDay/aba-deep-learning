from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "runtime_luau" / "collector_bundle.client.luau"


class CollectorSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = COLLECTOR.read_text(encoding="utf-8")

    def test_live_verified_version_is_present(self) -> None:
        self.assertIn('local VERSION = "0.2.1"', self.source)
        self.assertIn("local FEATURE_COUNT = 64", self.source)

    def test_export_uses_explicit_newline_byte(self) -> None:
        self.assertIn("table.concat(lines, string.char(10))", self.source)
        self.assertNotIn('table.concat(lines, "\n")', self.source)

    def test_public_collector_api_is_present(self) -> None:
        for method in (
            "function Collector:Start",
            "function Collector:Stop",
            "function Collector:Validate",
            "function Collector:Export",
            "function Collector:GetSummary",
            "function Collector:RecordActionRequest",
            "function Collector:RecordExecutorResult",
            "function Collector:Confirm",
        ):
            self.assertIn(method, self.source)

    def test_runtime_remains_observation_only(self) -> None:
        forbidden = (
            ":FireServer(",
            "VirtualInputManager",
            "mouse1click",
            "keypress(",
            "keyrelease(",
            "mouse1press",
            "mouse1release",
        )
        for token in forbidden:
            self.assertNotIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
