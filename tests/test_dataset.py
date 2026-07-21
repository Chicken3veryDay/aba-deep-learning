from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aba_deep_learning.dataset import (
    DatasetConfig,
    assess_episode,
    build_dataset,
    inspect_segment,
    segment_episode,
    split_name,
    write_dataset,
)


def mask() -> dict:
    return {
        "schema_version": "1.0.0",
        "commands": {
            "none": True,
            "m1": True,
            "block_start": True,
            "block_stop": False,
            "dodge": True,
            "jump": True,
            "move": True,
            "transform": False,
            "reset_spacing": True,
        },
        "move_slots": [True, True, True, True],
        "reasons": {},
        "confidence": 0.9,
    }


def request() -> dict:
    return {
        "schema_version": "1.0.0",
        "intent": "approach",
        "move_x": 0.0,
        "move_z": 1.0,
        "sprint": 1.0,
        "facing_mode": "face_target",
        "command": "none",
        "move_slot": None,
        "delay_ms": 0,
        "hold_ms": 0,
        "confidence": 0.8,
        "policy_id": "fixture",
        "model_version": "fixture",
    }


def episode(
    episode_id: str,
    *,
    steps: int = 80,
    labels: bool = True,
    gap_after: int | None = None,
    target_switch_after: int | None = None,
) -> dict:
    rows = []
    timestamp = 1000
    for index in range(steps):
        if gap_after is not None and index == gap_after:
            timestamp += 1000
        elif index:
            timestamp += 67
        target_name = (
            "Dummy B"
            if target_switch_after is not None and index >= target_switch_after
            else "Dummy A"
        )
        rows.append(
            {
                "observation": {
                    "schema_version": "1.0.0",
                    "episode_id": episode_id,
                    "step_index": index,
                    "timestamp_ms": timestamp,
                    "dt_ms": 0 if index == 0 else 67,
                    "self_state": {"name": "Self"},
                    "target_state": {"name": target_name},
                    "relative": {},
                    "combat": {},
                    "cooldowns": {},
                    "network": {},
                    "history": {},
                    "feature_vector": [index / max(steps, 1)] + [0.0] * 63,
                },
                "action_mask": mask(),
                "action_request": request() if labels else None,
                "executor_result": {"status": "unobserved"},
                "confirmations": [],
                "rewards": {
                    "damage_dealt": 1.0 if index == steps - 1 else 0.0,
                    "damage_received": 0.0,
                    "invalid_request": 0.0,
                },
                "raw_events": [],
            }
        )
    return {
        "schema_version": "1.0.0",
        "header": {
            "episode_id": episode_id,
            "stream_format": "aba_episode_jsonl_v1",
            "place_id": 1,
            "job_id": f"job-{episode_id}",
            "character": "Fixture",
            "target": "Dummy A",
            "collector_version": "0.2.1",
        },
        "steps": rows,
        "terminal": {"reason": "fixture", "duration_ms": max(0, timestamp - 1000)},
    }


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DatasetConfig(
            window_size=16,
            stride=8,
            min_segment_steps=8,
            max_gap_ms=250,
            min_quality_score=0.7,
            min_action_coverage=0.5,
        )

    def test_observation_task_accepts_unlabeled_episode(self) -> None:
        report = assess_episode(episode("unlabeled", labels=False), self.config, task="observation")
        self.assertTrue(report["accepted"])
        self.assertEqual(report["action_coverage"], 0.0)

    def test_behavior_cloning_rejects_unlabeled_episode(self) -> None:
        report = assess_episode(episode("unlabeled", labels=False), self.config, task="behavior_cloning")
        self.assertFalse(report["accepted"])
        self.assertIn("action_coverage_below_threshold", report["reasons"])

    def test_invalid_feature_vector_is_rejected(self) -> None:
        value = episode("bad-vector")
        value["steps"][4]["observation"]["feature_vector"].pop()
        report = assess_episode(value, self.config)
        self.assertFalse(report["accepted"])
        self.assertIn("invalid_feature_shape", report["reasons"])

    def test_segments_do_not_cross_large_timing_gap(self) -> None:
        value = episode("gap", steps=48, gap_after=24)
        segments = segment_episode(value, self.config)
        for segment in segments:
            timestamps = segment["timestamps_ms"]
            self.assertLessEqual(
                max(right - left for left, right in zip(timestamps, timestamps[1:])),
                self.config.max_gap_ms,
            )

    def test_segments_do_not_cross_target_switch(self) -> None:
        value = episode("switch", steps=48, target_switch_after=24)
        segments = segment_episode(value, self.config)
        self.assertGreaterEqual(len(segments), 4)
        for segment in segments:
            self.assertLessEqual(segment["length"], self.config.window_size)

    def test_episode_segments_stay_in_one_split(self) -> None:
        value = episode("same-episode")
        dataset = build_dataset([value], self.config)
        populated = [name for name, segments in dataset["splits"].items() if segments]
        self.assertEqual(len(populated), 1)
        self.assertEqual(populated[0], split_name("same-episode", self.config))

    def test_split_is_deterministic(self) -> None:
        self.assertEqual(split_name("stable-id", self.config), split_name("stable-id", self.config))

    def test_dataset_statistics_have_frozen_width(self) -> None:
        dataset = build_dataset([episode("one"), episode("two")], self.config, task="behavior_cloning")
        stats = dataset["statistics"]["feature_statistics"]
        self.assertEqual(len(stats["mean"]), 64)
        self.assertEqual(len(stats["std"]), 64)
        self.assertGreater(stats["count"], 0)

    def test_write_dataset_produces_manifest_and_jsonl(self) -> None:
        dataset = build_dataset([episode("write")], self.config)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_dataset(dataset, directory)
            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_version"], "1.0.0")
            self.assertEqual(
                sum(manifest["statistics"]["segments_by_split"].values()),
                manifest["statistics"]["segments_total"],
            )
            for name in ("train", "validation", "test"):
                self.assertTrue(Path(paths[name]).exists())

    def test_inspector_summarizes_segment(self) -> None:
        segment = segment_episode(episode("inspect", steps=16), self.config)[0]
        summary = inspect_segment(segment)
        self.assertEqual(summary["length"], 16)
        self.assertEqual(summary["labeled_steps"], 16)
        self.assertEqual(summary["damage_dealt"], 1.0)


if __name__ == "__main__":
    unittest.main()
