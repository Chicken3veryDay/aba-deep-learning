from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


class RankedWatcherSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.source_path = (
            cls.root
            / "runtime_luau"
            / "RankedPassiveWatcherV2.runtime.luau"
        )
        cls.manifest_path = (
            cls.root
            / "runtime_luau"
            / "RankedPassiveWatcherV2.manifest.json"
        )
        cls.source_bytes = cls.source_path.read_bytes()
        cls.source = cls.source_bytes.decode("utf-8")
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_source_matches_manifest(self) -> None:
        digest = hashlib.sha256(self.source_bytes).hexdigest()
        self.assertEqual(len(self.source_bytes), self.manifest["bytes"])
        self.assertEqual(digest, self.manifest["sha256"])
        self.assertEqual(
            self.manifest["version"],
            "0.2.3-ranked-full-passive",
        )
        self.assertFalse(self.manifest["executor_integration_allowed"])

    def test_source_contains_exact_feature_contract(self) -> None:
        match = re.search(
            r"local FEATURE_NAMES = \{(?P<body>.*?)\n\}",
            self.source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        names = re.findall(r'^\s+"([^"]+)",\s*$', match.group("body"), re.MULTILINE)
        self.assertEqual(len(names), 72)
        self.assertEqual(len(set(names)), 72)
        self.assertIn('assert(#values == 72', self.source)
        self.assertIn('FEATURE_SCHEMA_ID = "ranked_explicit_v3_72"', self.source)

    def test_source_has_no_gameplay_control_surface(self) -> None:
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

    def test_source_is_fail_closed_and_teleport_persistent(self) -> None:
        required = (
            "controllerContamination",
            "disableKnownControllers",
            'self:Stop("controller_contamination"',
            "queue_on_teleport or queueonteleport",
            "OPPONENT_LOSS_GRACE_MS",
            "RESULT_SEAL_DELAY_MS",
            "executor_integration_allowed = false",
            "G.ABA_RANKED_PASSIVE_WATCHER_V2_BOOT_LOCK",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
