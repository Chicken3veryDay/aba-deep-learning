from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


class RankedWatcherBootstrapSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.source_path = (
            cls.root
            / "runtime_luau"
            / "RankedWatcherBootstrap.runtime.luau"
        )
        cls.manifest_path = (
            cls.root
            / "runtime_luau"
            / "RankedWatcherBootstrap.manifest.json"
        )
        cls.source_bytes = cls.source_path.read_bytes()
        cls.source = cls.source_bytes.decode("utf-8")
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_bootstrap_matches_manifest(self) -> None:
        self.assertEqual(len(self.source_bytes), self.manifest["bytes"])
        self.assertEqual(
            hashlib.sha256(self.source_bytes).hexdigest(),
            self.manifest["sha256"],
        )
        self.assertEqual(self.manifest["version"], "0.1.0")
        self.assertFalse(self.manifest["executor_integration_allowed"])

    def test_bootstrap_is_commit_and_hash_pinned(self) -> None:
        pinned = self.manifest["pinned_watcher"]
        required = (
            pinned["commit"],
            pinned["sha256"],
            str(pinned["bytes"]),
            pinned["version"],
            "application/vnd.github.raw+json",
            "crypt.hash",
            "loadstring",
            "writefile(LOCAL_SOURCE_PATH, source)",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)
        self.assertNotIn("?ref=main", self.source)

    def test_bootstrap_has_no_gameplay_control_surface(self) -> None:
        forbidden = (
            ":FireServer(",
            ":InvokeServer(",
            "VirtualInputManager",
            "mouse1click",
            "mousemoverel",
            "keypress(",
            "keyrelease(",
            "ActionExecutor:",
            'Instance.new("ScreenGui")',
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_bootstrap_verifies_passive_receipt(self) -> None:
        required = (
            "manifest.executor_integration_allowed ~= false",
            "manifest.passive_invariants.creates_ui ~= false",
            "manifest.passive_invariants.executes_remote_calls ~= false",
            "manifest.passive_invariants.simulates_input ~= false",
            "status.passive ~= true",
            "status.executor_integration_allowed ~= false",
            "bootstrap_receipt.json",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
