from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from aba_deep_learning.contracts import ContractError
from aba_deep_learning.dataset import (
    DatasetConfig,
    assess_episode,
    build_dataset,
    build_partitioned_datasets,
    inspect_segment,
    segment_episode,
    split_name,
    write_dataset,
)
from aba_deep_learning.feature_schemas import (
    HUMAN_CAMERA_SCHEMA_ID,
    LEGACY_SCHEMA_ID,
    RANKED_EXPLICIT_SCHEMA_ID,
    AmbiguousFeatureSchemaError,
    IncompatibleFeatureSchemaError,
    get_feature_schema,
    resolve_episode_schema,
    schemas_compatible,
)
from aba_deep_learning.stream import read_episode_stream


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
    schema_id: str | None = None,
    width: int = 64,
    action_masks: bool = True,
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
        observation = {
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
            "feature_vector": [index / max(steps, 1)] + [0.0] * (width - 1),
        }
        if schema_id:
            observation["feature_schema_id"] = schema_id
        row = {
            "observation": observation,
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
        if action_masks:
            row["action_mask"] = mask()
        rows.append(row)

    header = {
        "episode_id": episode_id,
        "stream_format": "aba_episode_jsonl_v1",
        "place_id": 1,
        "job_id": f"job-{episode_id}",
        "character": "Fixture",
        "target": "Dummy A",
        "collector_version": "fixture",
    }
    if schema_id:
        header["feature_schema_id"] = schema_id
        header["feature_width"] = width
    return {
        "schema_version": "1.0.0",
        "header": header,
        "steps": rows,
        "terminal": {"reason": "fixture", "duration_ms": max(0, timestamp - 1000)},
    }


def to_jsonl(value: dict) -> str:
    records = [
        {
            "record_type": "header",
            "schema_version": "1.0.0",
            "header": value["header"],
        },
        *[
            {
                "record_type": "step",
                "schema_version": "1.0.0",
                "step": step,
            }
            for step in value["steps"]
        ],
        {
            "record_type": "terminal",
            "schema_version": "1.0.0",
            "terminal": value["terminal"],
        },
    ]
    return "\n".join(json.dumps(record) for record in records) + "\n"


class SchemaTests(unittest.TestCase):
    def test_alias_resolution(self) -> None:
        schema = get_feature_schema("ranked_feature_vector_v3_explicit")
        self.assertEqual(schema.schema_id, RANKED_EXPLICIT_SCHEMA_ID)
        self.assertEqual(schema.width, 72)

    def test_legacy_64_is_inferred(self) -> None:
        schema = resolve_episode_schema(episode("legacy"))
        self.assertEqual(schema.schema_id, LEGACY_SCHEMA_ID)

    def test_undeclared_72_is_ambiguous(self) -> None:
        with self.assertRaises(AmbiguousFeatureSchemaError):
            resolve_episode_schema(episode("ambiguous", width=72))

    def test_human_and_ranked_72_are_incompatible(self) -> None:
        self.assertFalse(
            schemas_compatible(HUMAN_CAMERA_SCHEMA_ID, RANKED_EXPLICIT_SCHEMA_ID)
        )


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
        report = assess_episode(
            episode("unlabeled", labels=False),
            self.config,
            task="observation",
        )
        self.assertTrue(report["accepted"])
        self.assertEqual(report["action_coverage"], 0.0)

    def test_behavior_cloning_rejects_unlabeled_episode(self) -> None:
        report = assess_episode(
            episode("unlabeled", labels=False),
            self.config,
            task="behavior_cloning",
        )
        self.assertFalse(report["accepted"])
        self.assertIn("action_coverage_below_threshold", report["reasons"])

    def test_invalid_feature_vector_is_rejected(self) -> None:
        value = episode("bad-vector")
        value["steps"][4]["observation"]["feature_vector"].pop()
        report = assess_episode(value, self.config)
        self.assertFalse(report["accepted"])
        self.assertIn("invalid_feature_shape", report["reasons"])

    def test_explicit_ranked_72_is_accepted(self) -> None:
        value = episode(
            "ranked",
            schema_id=RANKED_EXPLICIT_SCHEMA_ID,
            width=72,
        )
        report = assess_episode(value, self.config)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["feature_width"], 72)

    def test_schema_mismatch_is_rejected(self) -> None:
        value = episode(
            "ranked",
            schema_id=RANKED_EXPLICIT_SCHEMA_ID,
            width=72,
        )
        config = DatasetConfig(feature_schema_id=HUMAN_CAMERA_SCHEMA_ID)
        report = assess_episode(value, config)
        self.assertFalse(report["accepted"])
        self.assertIn("feature_schema_mismatch", report["reasons"])

    def test_segments_do_not_cross_large_timing_gap(self) -> None:
        value = episode("gap", steps=48, gap_after=24)
        segments = segment_episode(value, self.config)
        for segment in segments:
            timestamps = segment["timestamps_ms"]
            self.assertLessEqual(
                max(
                    right - left
                    for left, right in zip(timestamps, timestamps[1:])
                ),
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
        populated = [
            name
            for name, segments in dataset["splits"].items()
            if segments
        ]
        self.assertEqual(len(populated), 1)
        self.assertEqual(populated[0], split_name("same-episode", self.config))

    def test_dataset_statistics_follow_schema_width(self) -> None:
        value = episode(
            "ranked",
            schema_id=RANKED_EXPLICIT_SCHEMA_ID,
            width=72,
        )
        dataset = build_dataset(
            [value],
            self.config,
            task="behavior_cloning",
        )
        stats = dataset["statistics"]["feature_statistics"]
        self.assertEqual(stats["width"], 72)
        self.assertEqual(len(stats["mean"]), 72)

    def test_mixed_schemas_are_refused(self) -> None:
        values = [
            episode("legacy"),
            episode(
                "ranked",
                schema_id=RANKED_EXPLICIT_SCHEMA_ID,
                width=72,
            ),
        ]
        with self.assertRaises(IncompatibleFeatureSchemaError):
            build_dataset(values, self.config)

    def test_partitioned_builder_keeps_schemas_separate(self) -> None:
        values = [
            episode("legacy"),
            episode(
                "ranked",
                schema_id=RANKED_EXPLICIT_SCHEMA_ID,
                width=72,
            ),
        ]
        datasets = build_partitioned_datasets(values, self.config)
        self.assertEqual(
            set(datasets),
            {LEGACY_SCHEMA_ID, RANKED_EXPLICIT_SCHEMA_ID},
        )
        self.assertEqual(
            datasets[LEGACY_SCHEMA_ID]["feature_schema"]["width"],
            64,
        )
        self.assertEqual(
            datasets[RANKED_EXPLICIT_SCHEMA_ID]["feature_schema"]["width"],
            72,
        )

    def test_missing_masks_can_be_allowed_for_passive_human_data(self) -> None:
        value = episode(
            "passive",
            schema_id=RANKED_EXPLICIT_SCHEMA_ID,
            width=72,
            action_masks=False,
        )
        config = DatasetConfig(
            require_action_masks=False,
            min_action_coverage=0.5,
        )
        report = assess_episode(value, config, task="behavior_cloning")
        self.assertTrue(report["accepted"])
        self.assertEqual(report["raw_mask_integrity"], 0.0)

    def test_write_dataset_produces_schema_manifest(self) -> None:
        dataset = build_dataset([episode("write")], self.config)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_dataset(dataset, directory)
            manifest = json.loads(
                Path(paths["manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["dataset_version"], "2.0.0")
            self.assertEqual(
                manifest["feature_schema"]["schema_id"],
                LEGACY_SCHEMA_ID,
            )

    def test_inspector_summarizes_segment(self) -> None:
        segment = segment_episode(
            episode("inspect", steps=16),
            self.config,
        )[0]
        summary = inspect_segment(segment)
        self.assertEqual(summary["length"], 16)
        self.assertEqual(summary["labeled_steps"], 16)
        self.assertEqual(summary["damage_dealt"], 1.0)

    def test_stream_accepts_declared_72_schema(self) -> None:
        value = episode(
            "ranked-stream",
            schema_id=RANKED_EXPLICIT_SCHEMA_ID,
            width=72,
        )
        rebuilt = read_episode_stream(io.StringIO(to_jsonl(value)))
        self.assertEqual(
            rebuilt["header"]["feature_schema_id"],
            RANKED_EXPLICIT_SCHEMA_ID,
        )
        self.assertEqual(
            len(rebuilt["steps"][0]["observation"]["feature_vector"]),
            72,
        )

    def test_stream_rejects_mismatched_declared_width(self) -> None:
        value = episode(
            "mismatch-stream",
            schema_id=RANKED_EXPLICIT_SCHEMA_ID,
            width=72,
        )
        value["steps"][0]["observation"]["feature_vector"].pop()
        with self.assertRaises(ContractError):
            read_episode_stream(io.StringIO(to_jsonl(value)))


if __name__ == "__main__":
    unittest.main()
