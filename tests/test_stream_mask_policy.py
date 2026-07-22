from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aba_deep_learning.contracts import ContractError
from aba_deep_learning.feature_schemas import RANKED_EXPLICIT_SCHEMA_ID
from aba_deep_learning.stream import read_episode_stream


def _records(*, action_mask: object = None) -> list[dict[str, object]]:
    episode_id = "passive-ranked-fixture"
    step: dict[str, object] = {
        "observation": {
            "schema_version": "1.0.0",
            "episode_id": episode_id,
            "step_index": 0,
            "timestamp_ms": 1000,
            "dt_ms": 0,
            "self_state": {},
            "target_state": {},
            "relative": {},
            "combat": {},
            "cooldowns": {},
            "network": {},
            "history": {},
            "feature_vector": [0.0] * 72,
        },
        "action_request": None,
        "executor_result": {"status": "unobserved"},
        "confirmations": [],
        "raw_events": [],
        "rewards": {},
    }
    if action_mask is not None:
        step["action_mask"] = action_mask

    return [
        {
            "record_type": "header",
            "schema_version": "1.0.0",
            "header": {
                "episode_id": episode_id,
                "stream_format": "aba_episode_jsonl_v1",
                "feature_schema_id": RANKED_EXPLICIT_SCHEMA_ID,
            },
        },
        {
            "record_type": "step",
            "schema_version": "1.0.0",
            "step": step,
        },
        {
            "record_type": "terminal",
            "schema_version": "1.0.0",
            "terminal": {"reason": "fixture_complete", "duration_ms": 0},
        },
    ]


def _write(records: list[dict[str, object]], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


class StreamMaskPolicyTests(unittest.TestCase):
    def test_maskless_stream_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passive.jsonl"
            _write(_records(), path)
            with self.assertRaises(ContractError):
                read_episode_stream(path)

    def test_maskless_stream_loads_when_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passive.jsonl"
            _write(_records(), path)
            episode = read_episode_stream(path, require_action_masks=False)
            self.assertEqual(episode["header"]["feature_width"], 72)
            self.assertEqual(len(episode["steps"]), 1)

    def test_present_invalid_mask_is_never_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-mask.jsonl"
            _write(_records(action_mask={"commands": {}}), path)
            with self.assertRaises(ContractError):
                read_episode_stream(path, require_action_masks=False)


if __name__ == "__main__":
    unittest.main()
