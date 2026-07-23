from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from aba_deep_learning.ranked_release import (
    ACTION_NAMES, FEATURE_NAMES, RankedReleaseError, assign_splits, build_release,
    extract_action_label, inventory_corpus, normalize_feature_vector,
    parse_recording, validate_split_isolation,
)


def header(episode: str, job: str, *, alias: bool = False) -> dict:
    return {"record_type": "header", "header": {
        "episode_id": episode, "job_id": job, "started_at_ms": 1000,
        "watcher_version": "test-passive", "stream_format": "aba_ranked_passive_jsonl_v2",
        "feature_schema": "ranked_feature_vector_v3_explicit" if alias else "ranked_explicit_v3_72",
        "feature_width": 72, "feature_names": list(FEATURE_NAMES),
    }}


def step(index: int, *, keyed: bool = False, contaminated: bool = False) -> dict:
    vector: list[float] | dict[str, float] = [index / 100] * 72
    if keyed:
        vector = {str(i + 1): value for i, value in enumerate(vector)}
    return {"record_type": "step", "step": {
        "step_index": index, "timestamp_ms": 1000 + index * 67,
        "feature_vector": vector,
        "passive_invariants": {"no_ui": not contaminated, "no_remote_calls": True, "no_input_simulation": True, "controller_guard": True},
        "held": {"W": index % 2 == 1, "MouseButton1": index == 2},
        "self_state": {"health": 100 - index}, "target_state": {"health": 100 - index * 2},
    }}


def terminal(count: int) -> dict:
    return {"record_type": "terminal", "terminal": {
        "reason": "match_complete", "steps": count, "duration_ms": (count - 1) * 67,
        "ended_at_ms": 1000 + (count - 1) * 67,
    }}


def write_episode(path: Path, episode: str, job: str, *, keyed: bool = False, terminal_record: bool = True, contaminated: bool = False, alias: bool = False) -> None:
    records = [header(episode, job, alias=alias)] + [step(i, keyed=keyed, contaminated=contaminated and i == 1) for i in range(4)]
    if terminal_record:
        records.append(terminal(4))
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


class RankedReleaseV1Tests(unittest.TestCase):
    def test_array_and_numeric_keyed_object_are_equivalent(self) -> None:
        vector = [index / 100 for index in range(72)]
        array, _ = normalize_feature_vector(vector)
        mapped, encoding = normalize_feature_vector({str(index + 1): value for index, value in enumerate(vector)})
        self.assertEqual(array, mapped)
        self.assertEqual(encoding, "numeric_key_object_1_72")

    def test_numeric_keyed_object_requires_every_index(self) -> None:
        with self.assertRaises(RankedReleaseError):
            normalize_feature_vector({str(index): 0 for index in range(1, 72)})

    def test_nonfinite_features_are_rejected(self) -> None:
        vector = [0.0] * 72
        vector[4] = math.inf
        with self.assertRaises(RankedReleaseError):
            normalize_feature_vector(vector)

    def test_malformed_json_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(header("e", "j")) + "\n{broken\n", encoding="utf-8")
            parsed = parse_recording(path)
            self.assertEqual(parsed.inventory.parse_error_lines, [2])
            self.assertIn("parse_errors", parsed.inventory.rejection_or_quarantine_reasons)

    def test_terminal_reconstruction_is_derived_and_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"
            write_episode(path, "e", "j", terminal_record=False)
            raw = path.read_bytes()
            parsed = parse_recording(path)
            self.assertTrue(parsed.inventory.terminal_reconstructed)
            self.assertTrue(parsed.inventory.partial)
            self.assertEqual(path.read_bytes(), raw)

    def test_contamination_is_rejected_but_status_ui_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clean = Path(directory) / "clean.jsonl"
            dirty = Path(directory) / "dirty.jsonl"
            write_episode(clean, "clean", "clean-job")
            write_episode(dirty, "dirty", "dirty-job", contaminated=True)
            self.assertNotIn("passive_invariant_failure", parse_recording(clean).inventory.rejection_or_quarantine_reasons)
            self.assertIn("passive_invariant_failure", parse_recording(dirty).inventory.rejection_or_quarantine_reasons)

    def test_authoritative_held_label_extraction(self) -> None:
        move_x, move_z, actions, available, source = extract_action_label({"held": {"W": True, "A": True, "Q": True, "One": True}})
        self.assertEqual((move_x, move_z), (-1.0, 1.0))
        self.assertTrue(actions["dodge"] and actions["move_slot_1"])
        self.assertTrue(available)
        self.assertEqual(source, "held")

    def test_duplicate_clean_selection_and_split_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(11):
                write_episode(root / f"episode-{index}.jsonl", f"e{index}", f"j{index}", keyed=index % 2 == 0, alias=index == 0)
            write_episode(root / "episode-0.clean.jsonl", "e0", "j0", alias=True)
            recordings, groups = inventory_corpus([root], relative_to=root)
            self.assertEqual(len(groups), 1)
            self.assertTrue(groups[0]["chosen"].endswith(".clean.jsonl"))
            splits = assign_splits(recordings)
            self.assertEqual((len(splits["train"]), len(splits["validation"]), len(splits["test"])), (7, 2, 2))
            validate_split_isolation(recordings, splits)
            with self.assertRaises(RankedReleaseError):
                validate_split_isolation(recordings, {"train": [splits["train"][0]], "validation": [splits["train"][0]], "test": [splits["test"][0]]})

    def test_release_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            root.mkdir()
            for index in range(11):
                write_episode(root / f"episode-{index}.jsonl", f"e{index}", f"j{index}")
            first = build_release([root], Path(directory) / "out1", relative_to=root)
            second = build_release([root], Path(directory) / "out2", relative_to=root)
            self.assertEqual(first["splits"], second["splits"])
            self.assertEqual(first["total_admitted_steps"], 33)
            self.assertEqual(set(ACTION_NAMES), set(first["episodes"][0]["action_coverage"]))

if __name__ == "__main__":
    unittest.main()
