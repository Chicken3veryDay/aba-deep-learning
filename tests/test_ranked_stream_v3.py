from __future__ import annotations

import io
import json
import unittest

from aba_deep_learning.dataset import DatasetConfig, build_dataset
from aba_deep_learning.feature_schemas import RANKED_EXPLICIT_SCHEMA_ID
from aba_deep_learning.ranked_stream_v3 import (
    RANKED_STREAM_FORMAT_V3,
    RankedStreamV3Error,
    read_ranked_v3_stream,
)
from aba_deep_learning.stream import read_episode_stream


WATCHER_VERSION = "0.2.3-ranked-full-passive"


def action_label(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "move_x": 0,
        "move_z": 0,
        "sprint": 0,
        "block_held": False,
        "transform_held": False,
        "m1": False,
        "dodge": False,
        "jump": False,
        "block_start": False,
        "block_stop": False,
        "move_slot": 0,
        "source": "UserInputService",
    }
    value.update(updates)
    return value


def observation(index: int, *, target_health: float = 100) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "episode_id": "ranked-v3-fixture",
        "step_index": index,
        "timestamp_ms": 1000 + index * 67,
        "dt_ms": 0 if index == 0 else 67,
        "self_state": {"health": 100, "max_health": 100},
        "target_state": {
            "health": target_health,
            "max_health": 100,
            "present": True,
        },
        "relative": {"distance": 5},
        "combat": {},
        "cooldowns": {"source": "hud_snapshot", "hud": []},
        "network": {"ping_ms": 40},
        "history": {
            "held": {"w": index == 1, "shift": index == 1},
            "raw_input_events": [],
        },
        "camera": {},
        "nearby_players": [],
        "feature_schema_id": RANKED_EXPLICIT_SCHEMA_ID,
        "feature_vector": [0.0] * 72,
    }


def step(
    index: int,
    *,
    label: dict[str, object] | None = None,
    target_health: float = 100,
    damage_dealt: float = 0,
) -> dict[str, object]:
    raw_events = []
    if label and label.get("m1"):
        raw_events.append({
            "timestamp_ms": 1000 + index * 67 - 1,
            "phase": "began",
            "name": "m1",
            "processed": False,
        })
    return {
        "record_type": "step",
        "schema_version": "3.0.0",
        "step": {
            "observation": observation(index, target_health=target_health),
            "action_label": label or action_label(),
            "executor_result": {
                "status": "not_applicable_passive_watcher",
                "source": WATCHER_VERSION,
            },
            "confirmations": [],
            "raw_events": raw_events,
            "rewards": {
                "damage_dealt": damage_dealt,
                "damage_received": 0,
            },
        },
    }


def records() -> list[dict[str, object]]:
    return [
        {
            "record_type": "header",
            "schema_version": "3.0.0",
            "header": {
                "episode_id": "ranked-v3-fixture",
                "stream_format": RANKED_STREAM_FORMAT_V3,
                "schema_version": "3.0.0",
                "feature_schema_id": RANKED_EXPLICIT_SCHEMA_ID,
                "feature_width": 72,
                "feature_names": [f"f{index}" for index in range(72)],
                "watcher_version": WATCHER_VERSION,
                "place_id": 1458767429,
                "place_version": 1,
                "job_id": "job-v3",
                "player": "Self",
                "opponent": "Enemy",
                "started_at_ms": 1000,
                "sample_hz": 15,
                "observation_only": True,
                "input_source": "UserInputService",
                "executor_integration_allowed": False,
            },
        },
        step(0),
        step(
            1,
            label=action_label(move_z=1, sprint=1, m1=True),
        ),
        step(
            2,
            label=action_label(move_x=1, dodge=True, move_slot=3),
            target_health=80,
            damage_dealt=20,
        ),
        {
            "record_type": "terminal",
            "schema_version": "3.0.0",
            "terminal": {
                "reason": "match_result",
                "match_result": "victory",
                "ended_at_ms": 1200,
                "duration_ms": 200,
                "steps": 3,
                "contaminated": False,
                "watcher_version": WATCHER_VERSION,
                "executor_integration_allowed": False,
            },
        },
    ]


def jsonl(value: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(row) for row in value) + "\n"


class RankedStreamV3Tests(unittest.TestCase):
    def test_standard_loader_auto_detects_v3_and_shifts_actions(self) -> None:
        episode = read_episode_stream(io.StringIO(jsonl(records())))
        self.assertEqual(
            episode["header"]["feature_schema_id"],
            RANKED_EXPLICIT_SCHEMA_ID,
        )
        self.assertEqual(episode["header"]["action_shift"], 1)
        self.assertEqual(len(episode["steps"]), 3)
        first = episode["steps"][0]
        self.assertEqual(first["action_label"]["move_z"], 1)
        self.assertEqual(first["action_label"]["sprint"], 1)
        self.assertTrue(first["action_label"]["m1"])
        self.assertEqual(first["action_label"]["label_space"], "camera_relative_input_v1")
        self.assertIsNone(episode["steps"][-1]["action_label"])

    def test_direct_v3_loader_preserves_rewards_and_terminal(self) -> None:
        episode = read_ranked_v3_stream(io.StringIO(jsonl(records())))
        self.assertEqual(episode["steps"][2]["rewards"]["damage_dealt"], 20)
        self.assertEqual(episode["terminal"]["match_result"], "victory")
        self.assertFalse(episode["header"]["executor_integration_allowed"])
        self.assertEqual(
            episode["header"]["independence_group_id"],
            "job-v3",
        )

    def test_contaminated_terminal_is_rejected(self) -> None:
        value = records()
        value[-1]["terminal"]["contaminated"] = True
        with self.assertRaises(RankedStreamV3Error):
            read_ranked_v3_stream(io.StringIO(jsonl(value)))

    def test_non_passive_executor_result_is_rejected(self) -> None:
        value = records()
        value[1]["step"]["executor_result"]["status"] = "accepted"
        with self.assertRaises(RankedStreamV3Error):
            read_ranked_v3_stream(io.StringIO(jsonl(value)))

    def test_bad_feature_width_is_rejected(self) -> None:
        value = records()
        value[1]["step"]["observation"]["feature_vector"].pop()
        with self.assertRaises(RankedStreamV3Error):
            read_ranked_v3_stream(io.StringIO(jsonl(value)))

    def test_v3_episode_builds_behavior_dataset_without_masks(self) -> None:
        episode = read_episode_stream(io.StringIO(jsonl(records())))
        config = DatasetConfig(
            window_size=2,
            stride=1,
            min_segment_steps=2,
            min_action_coverage=0.5,
            feature_schema_id=RANKED_EXPLICIT_SCHEMA_ID,
            require_action_masks=False,
        )
        dataset = build_dataset(
            [episode],
            config,
            task="behavior_cloning",
        )
        self.assertEqual(dataset["statistics"]["episodes_accepted"], 1)
        self.assertEqual(dataset["feature_schema"]["width"], 72)
        self.assertGreater(dataset["statistics"]["segments_total"], 0)


if __name__ == "__main__":
    unittest.main()
