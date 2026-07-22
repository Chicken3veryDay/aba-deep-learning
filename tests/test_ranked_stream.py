from __future__ import annotations

import io
import json
import unittest

from aba_deep_learning.dataset import DatasetConfig, build_dataset
from aba_deep_learning.feature_schemas import RANKED_EXPLICIT_SCHEMA_ID
from aba_deep_learning.ranked_stream import (
    RANKED_STREAM_FORMAT,
    RankedStreamError,
    action_label_from_ranked_step,
    read_ranked_stream,
)
from aba_deep_learning.stream import read_episode_stream


def held(**updates: int) -> dict[str, int]:
    value = {
        key: 0
        for key in (
            "w", "a", "s", "d", "shift", "f", "q", "space",
            "one", "two", "three", "four", "g", "mouse1", "mouse2",
        )
    }
    value.update(updates)
    return value


def model(health: float = 100) -> dict:
    return {
        "name": "Target",
        "health": health,
        "max_health": 100,
        "health_ratio": health / 100,
        "position": [0, 0, 5],
        "velocity": [0, 0, 0],
        "look": [0, 0, -1],
        "move_direction": [0, 0, 0],
        "markers": {},
        "animations": [],
        "grounded": True,
        "state": "Running",
    }


def step(
    index: int,
    *,
    self_health: float = 100,
    target_health: float = 100,
    held_state: dict | None = None,
    events: list[dict] | None = None,
) -> dict:
    return {
        "record_type": "step",
        "schema_version": "1.0.0",
        "session_id": "session-1",
        "step_index": index,
        "timestamp_ms": 1000 + index * 67,
        "dt_ms": 0 if index == 0 else 67,
        "feature_schema": "ranked_feature_vector_v3_explicit",
        "feature_vector": [0.0] * 72,
        "held": held_state or held(),
        "input_events": events or [],
        "self": model(self_health),
        "target": {
            "player_name": "Enemy",
            "user_id": 2,
            "distance": 5,
            "model": model(target_health),
        },
        "camera": {},
        "nearby_players": [],
        "ping_ms": 40,
        "passive_invariants": {
            "controller_inactive": True,
            "no_input_simulation": True,
            "no_remote_calls": True,
            "no_ui": True,
        },
    }


def records() -> list[dict]:
    return [
        {
            "record_type": "header",
            "schema_version": "1.0.0",
            "header": {
                "stream_format": RANKED_STREAM_FORMAT,
                "session_id": "session-1",
                "feature_schema": "ranked_feature_vector_v3_explicit",
                "feature_width": 72,
                "watcher_version": "0.2.2",
                "sample_hz": 15,
                "feature_names": [f"f{index}" for index in range(72)],
            },
        },
        {
            "record_type": "event",
            "schema_version": "1.0.0",
            "session_id": "session-1",
            "timestamp_ms": 999,
            "event_type": "watcher_started",
            "payload": {},
        },
        step(0),
        step(
            1,
            held_state=held(w=1, shift=1),
            events=[
                {"key": "w", "phase": "began"},
                {"key": "shift", "phase": "began"},
                {"key": "mouse1", "phase": "began"},
            ],
        ),
        step(
            2,
            self_health=90,
            target_health=80,
            held_state=held(d=1, f=1),
            events=[
                {"key": "q", "phase": "began"},
                {"key": "space", "phase": "began"},
                {"key": "three", "phase": "began"},
                {"key": "f", "phase": "began"},
            ],
        ),
        {
            "record_type": "terminal",
            "schema_version": "1.0.0",
            "session_id": "session-1",
            "timestamp_ms": 1200,
            "terminal": {
                "reason": "match_result",
                "duration_ms": 200,
                "steps": 3,
                "passive_invariants": {
                    "controller_inactive": True,
                    "no_input_simulation": True,
                    "no_remote_calls": True,
                    "no_ui": True,
                },
            },
        },
    ]


def jsonl(value: list[dict]) -> str:
    return "\n".join(json.dumps(row) for row in value) + "\n"


class RankedStreamTests(unittest.TestCase):
    def test_action_label_maps_held_and_events(self) -> None:
        label = action_label_from_ranked_step(records()[4])
        self.assertEqual(label["move_x"], 1)
        self.assertEqual(label["move_z"], 0)
        self.assertTrue(label["dodge"])
        self.assertTrue(label["jump"])
        self.assertTrue(label["block_start"])
        self.assertEqual(label["move_slot"], 3)

    def test_conversion_preserves_schema_and_shifts_actions(self) -> None:
        episode = read_ranked_stream(io.StringIO(jsonl(records())))
        self.assertEqual(
            episode["header"]["feature_schema_id"],
            RANKED_EXPLICIT_SCHEMA_ID,
        )
        self.assertEqual(len(episode["steps"]), 3)
        first = episode["steps"][0]
        self.assertEqual(first["action_label"]["move_z"], 1)
        self.assertTrue(first["action_label"]["m1"])
        self.assertIsNone(episode["steps"][-1]["action_label"])

    def test_standard_loader_auto_detects_ranked_stream(self) -> None:
        episode = read_episode_stream(io.StringIO(jsonl(records())))
        self.assertEqual(
            episode["header"]["feature_schema_id"],
            RANKED_EXPLICIT_SCHEMA_ID,
        )
        self.assertEqual(episode["header"]["action_shift"], 1)

    def test_damage_is_derived_from_health_deltas(self) -> None:
        episode = read_ranked_stream(io.StringIO(jsonl(records())))
        final = episode["steps"][2]
        self.assertEqual(final["rewards"]["damage_received"], 10)
        self.assertEqual(final["rewards"]["damage_dealt"], 20)
        kinds = {item["kind"] for item in final["confirmations"]}
        self.assertEqual(kinds, {"damage_received", "damage_dealt"})

    def test_lifecycle_events_are_preserved(self) -> None:
        episode = read_ranked_stream(io.StringIO(jsonl(records())))
        source_events = episode["terminal"]["source_events"]
        self.assertEqual(source_events[0]["event_type"], "watcher_started")

    def test_passive_violation_is_rejected(self) -> None:
        value = records()
        value[2]["passive_invariants"]["controller_inactive"] = False
        with self.assertRaises(RankedStreamError):
            read_ranked_stream(io.StringIO(jsonl(value)))

    def test_duplicate_step_is_rejected(self) -> None:
        value = records()
        value.insert(4, dict(value[2]))
        with self.assertRaises(RankedStreamError):
            read_ranked_stream(io.StringIO(jsonl(value)))

    def test_bad_width_is_rejected(self) -> None:
        value = records()
        value[2]["feature_vector"].pop()
        with self.assertRaises(RankedStreamError):
            read_ranked_stream(io.StringIO(jsonl(value)))

    def test_ranked_episode_builds_behavior_dataset_without_masks(self) -> None:
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
