from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .feature_schemas import (
    LEGACY_SCHEMA_ID,
    AmbiguousFeatureSchemaError,
    FeatureSchema,
    FeatureSchemaError,
    IncompatibleFeatureSchemaError,
    UnknownFeatureSchemaError,
    get_feature_schema,
    resolve_episode_schema,
)
from .stream import read_episode_stream

DATASET_VERSION = "2.0.0"
TASKS = frozenset({"observation", "behavior_cloning"})


@dataclass(frozen=True)
class DatasetConfig:
    window_size: int = 32
    stride: int = 16
    min_segment_steps: int = 8
    max_gap_ms: int = 250
    min_quality_score: float = 0.70
    min_action_coverage: float = 0.50
    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    split_seed: str = "aba-step3-v1"
    feature_schema_id: str | None = None
    allow_legacy_64_inference: bool = True
    require_action_masks: bool = True

    def validate(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.min_segment_steps < 1:
            raise ValueError("min_segment_steps must be positive")
        if self.min_segment_steps > self.window_size:
            raise ValueError("min_segment_steps cannot exceed window_size")
        if self.max_gap_ms < 1:
            raise ValueError("max_gap_ms must be positive")
        if not 0 <= self.min_quality_score <= 1:
            raise ValueError("min_quality_score must be in [0, 1]")
        if not 0 <= self.min_action_coverage <= 1:
            raise ValueError("min_action_coverage must be in [0, 1]")
        ratios = self.train_ratio + self.validation_ratio + self.test_ratio
        if not math.isclose(ratios, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("split ratios must sum to 1")
        if self.feature_schema_id is not None:
            get_feature_schema(self.feature_schema_id)


def load_episodes(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [read_episode_stream(Path(path)) for path in paths]


def discover_episode_files(root: str | Path) -> list[Path]:
    path = Path(root)
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*.jsonl")
        if candidate.is_file()
    )


def _step_timestamp(step: Mapping[str, Any]) -> int:
    return int(step["observation"]["timestamp_ms"])


def _step_target(step: Mapping[str, Any]) -> str:
    target = step["observation"].get("target_state", {})
    name = target.get("name")
    return str(name) if name else ""


def _is_valid_mask(mask: Any) -> bool:
    if not isinstance(mask, Mapping):
        return False
    commands = mask.get("commands")
    slots = mask.get("move_slots")
    confidence = mask.get("confidence")
    return (
        isinstance(commands, Mapping)
        and len(commands) >= 9
        and isinstance(slots, Sequence)
        and not isinstance(slots, (str, bytes))
        and len(slots) == 4
        and all(isinstance(item, bool) for item in slots)
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        and 0 <= float(confidence) <= 1
    )


def _action_label(step: Mapping[str, Any]) -> Any:
    request = step.get("action_request")
    if request is not None:
        return request
    for key in ("action_label", "input_label", "labels"):
        value = step.get(key)
        if value is not None:
            return value
    return None


def _schema_failure_reason(exc: FeatureSchemaError) -> str:
    if isinstance(exc, AmbiguousFeatureSchemaError):
        return "ambiguous_feature_schema"
    if isinstance(exc, IncompatibleFeatureSchemaError):
        return "incompatible_feature_schema"
    if isinstance(exc, UnknownFeatureSchemaError):
        return "unknown_feature_schema"
    return "feature_schema_error"


def _resolve_assessment_schema(
    episode: Mapping[str, Any],
    config: DatasetConfig,
) -> tuple[FeatureSchema | None, list[str]]:
    reasons: list[str] = []
    try:
        schema = resolve_episode_schema(
            episode,
            allow_legacy_64_inference=config.allow_legacy_64_inference,
        )
    except FeatureSchemaError as exc:
        return None, [_schema_failure_reason(exc)]

    if config.feature_schema_id is not None:
        expected = get_feature_schema(config.feature_schema_id)
        if schema.compatibility_group != expected.compatibility_group:
            reasons.append("feature_schema_mismatch")
        schema = expected
    return schema, reasons


def assess_episode(
    episode: Mapping[str, Any],
    config: DatasetConfig | None = None,
    *,
    task: str = "observation",
) -> dict[str, Any]:
    config = config or DatasetConfig()
    config.validate()
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task!r}")

    schema, reasons = _resolve_assessment_schema(episode, config)
    expected_width = schema.width if schema is not None else None

    steps = episode.get("steps", [])
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        steps = []

    total = len(steps)
    feature_shape_ok = 0
    finite_ok = 0
    bounded_ok = 0
    mask_ok = 0
    action_labels = 0
    timestamp_violations = 0
    gap_count = 0
    target_switches = 0
    previous_timestamp: int | None = None
    previous_target: str | None = None

    for step in steps:
        observation = step.get("observation", {}) if isinstance(step, Mapping) else {}
        vector = observation.get("feature_vector", [])
        shape_ok = (
            expected_width is not None
            and isinstance(vector, Sequence)
            and not isinstance(vector, (str, bytes))
            and len(vector) == expected_width
        )
        if shape_ok:
            feature_shape_ok += 1
            finite = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in vector
            )
            if finite:
                finite_ok += 1
                if all(-1.000001 <= float(value) <= 1.000001 for value in vector):
                    bounded_ok += 1

        if isinstance(step, Mapping):
            if _is_valid_mask(step.get("action_mask")):
                mask_ok += 1
            if _action_label(step) is not None:
                action_labels += 1

        try:
            timestamp = _step_timestamp(step)
        except (KeyError, TypeError, ValueError):
            timestamp_violations += 1
            timestamp = previous_timestamp if previous_timestamp is not None else 0

        if previous_timestamp is not None:
            delta = timestamp - previous_timestamp
            if delta <= 0:
                timestamp_violations += 1
            elif delta > config.max_gap_ms:
                gap_count += 1
        previous_timestamp = timestamp

        try:
            target_name = _step_target(step)
        except (KeyError, TypeError, ValueError):
            target_name = ""
        if (
            previous_target is not None
            and target_name
            and previous_target
            and target_name != previous_target
        ):
            target_switches += 1
        if target_name:
            previous_target = target_name

    denominator = max(total, 1)
    feature_integrity = min(feature_shape_ok, finite_ok, bounded_ok) / denominator
    raw_mask_integrity = mask_ok / denominator
    mask_integrity = raw_mask_integrity if config.require_action_masks else 1.0
    action_coverage = action_labels / denominator
    timing_integrity = max(
        0.0,
        1.0 - (timestamp_violations + gap_count) / max(total - 1, 1),
    )
    continuity = max(
        0.0,
        1.0 - (gap_count + target_switches) / max(total - 1, 1),
    )

    if task == "behavior_cloning":
        score = (
            0.35 * feature_integrity
            + 0.15 * timing_integrity
            + 0.15 * mask_integrity
            + 0.15 * continuity
            + 0.20 * action_coverage
        )
    else:
        score = (
            0.45 * feature_integrity
            + 0.20 * timing_integrity
            + 0.15 * mask_integrity
            + 0.20 * continuity
        )

    if total < config.min_segment_steps:
        reasons.append("too_few_steps")
    if feature_shape_ok != total:
        reasons.append("invalid_feature_shape")
    if finite_ok != total:
        reasons.append("non_finite_features")
    if bounded_ok != total:
        reasons.append("out_of_range_features")
    if config.require_action_masks and mask_ok != total:
        reasons.append("invalid_action_masks")
    if timestamp_violations:
        reasons.append("non_monotonic_or_missing_timestamps")
    if task == "behavior_cloning" and action_coverage < config.min_action_coverage:
        reasons.append("action_coverage_below_threshold")
    if score < config.min_quality_score:
        reasons.append("quality_score_below_threshold")

    header = episode.get("header", {}) if isinstance(episode, Mapping) else {}
    episode_id = str(header.get("episode_id") or "")
    if not episode_id:
        reasons.append("missing_episode_id")

    reasons = list(dict.fromkeys(reasons))
    return {
        "episode_id": episode_id,
        "task": task,
        "accepted": not reasons,
        "score": round(score, 6),
        "reasons": reasons,
        "steps": total,
        "feature_schema_id": schema.schema_id if schema is not None else None,
        "feature_width": schema.width if schema is not None else None,
        "compatibility_group": (
            schema.compatibility_group if schema is not None else None
        ),
        "feature_integrity": round(feature_integrity, 6),
        "timing_integrity": round(timing_integrity, 6),
        "mask_integrity": round(mask_integrity, 6),
        "raw_mask_integrity": round(raw_mask_integrity, 6),
        "continuity": round(continuity, 6),
        "action_coverage": round(action_coverage, 6),
        "gap_count": gap_count,
        "target_switches": target_switches,
        "timestamp_violations": timestamp_violations,
    }


def _contiguous_runs(
    steps: Sequence[Mapping[str, Any]],
    *,
    max_gap_ms: int,
) -> list[list[Mapping[str, Any]]]:
    runs: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    previous_timestamp: int | None = None
    previous_target: str | None = None

    for step in steps:
        timestamp = _step_timestamp(step)
        target = _step_target(step)
        split = False
        if previous_timestamp is not None and timestamp - previous_timestamp > max_gap_ms:
            split = True
        if previous_target and target and target != previous_target:
            split = True
        if split and current:
            runs.append(current)
            current = []
        current.append(step)
        previous_timestamp = timestamp
        if target:
            previous_target = target

    if current:
        runs.append(current)
    return runs


def _window_ranges(length: int, config: DatasetConfig) -> list[tuple[int, int]]:
    if length < config.min_segment_steps:
        return []
    if length <= config.window_size:
        return [(0, length)]

    ranges: list[tuple[int, int]] = []
    start = 0
    while start + config.window_size <= length:
        ranges.append((start, start + config.window_size))
        start += config.stride

    tail = (max(0, length - config.window_size), length)
    if tail not in ranges:
        ranges.append(tail)
    return ranges


def _segment_id(
    episode_id: str,
    feature_schema_id: str,
    run_index: int,
    start: int,
    end: int,
) -> str:
    raw = (
        f"{feature_schema_id}:{episode_id}:{run_index}:{start}:{end}"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def segment_episode(
    episode: Mapping[str, Any],
    config: DatasetConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or DatasetConfig()
    config.validate()
    schema = resolve_episode_schema(
        episode,
        allow_legacy_64_inference=config.allow_legacy_64_inference,
    )
    if config.feature_schema_id is not None:
        expected = get_feature_schema(config.feature_schema_id)
        if schema.compatibility_group != expected.compatibility_group:
            raise IncompatibleFeatureSchemaError(
                f"episode schema {schema.schema_id} does not match "
                f"requested schema {expected.schema_id}"
            )
        schema = expected

    header = episode.get("header", {})
    episode_id = str(header.get("episode_id") or "")
    steps = episode.get("steps", [])
    if not episode_id or not isinstance(steps, Sequence):
        return []

    segments: list[dict[str, Any]] = []
    for run_index, run in enumerate(
        _contiguous_runs(steps, max_gap_ms=config.max_gap_ms)
    ):
        for start, end in _window_ranges(len(run), config):
            selected = run[start:end]
            first = selected[0]["observation"]
            last = selected[-1]["observation"]
            segments.append(
                {
                    "dataset_version": DATASET_VERSION,
                    "feature_schema_id": schema.schema_id,
                    "feature_width": schema.width,
                    "segment_id": _segment_id(
                        episode_id,
                        schema.schema_id,
                        run_index,
                        start,
                        end,
                    ),
                    "episode_id": episode_id,
                    "run_index": run_index,
                    "start_step_index": int(first["step_index"]),
                    "end_step_index": int(last["step_index"]),
                    "start_timestamp_ms": int(first["timestamp_ms"]),
                    "end_timestamp_ms": int(last["timestamp_ms"]),
                    "length": len(selected),
                    "metadata": {
                        "place_id": header.get("place_id"),
                        "job_id": header.get("job_id"),
                        "player": header.get("player"),
                        "character": header.get("character"),
                        "target": header.get("target"),
                        "collector_version": header.get("collector_version"),
                        "feature_schema_id": schema.schema_id,
                    },
                    "feature_vectors": [
                        list(step["observation"]["feature_vector"])
                        for step in selected
                    ],
                    "action_masks": [
                        step.get("action_mask")
                        for step in selected
                    ],
                    "action_requests": [
                        step.get("action_request")
                        for step in selected
                    ],
                    "action_labels": [
                        _action_label(step)
                        for step in selected
                    ],
                    "executor_results": [
                        step.get("executor_result", {})
                        for step in selected
                    ],
                    "rewards": [
                        step.get("rewards", {})
                        for step in selected
                    ],
                    "timestamps_ms": [
                        int(step["observation"]["timestamp_ms"])
                        for step in selected
                    ],
                    "step_indices": [
                        int(step["observation"]["step_index"])
                        for step in selected
                    ],
                }
            )
    return segments


def split_name(episode_id: str, config: DatasetConfig | None = None) -> str:
    config = config or DatasetConfig()
    config.validate()
    digest = hashlib.sha256(
        f"{config.split_seed}:{episode_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < config.train_ratio:
        return "train"
    if bucket < config.train_ratio + config.validation_ratio:
        return "validation"
    return "test"


class _FeatureStats:
    def __init__(self, width: int) -> None:
        if width < 1:
            raise ValueError("feature-stat width must be positive")
        self.width = width
        self.count = 0
        self.mean = [0.0] * width
        self.m2 = [0.0] * width
        self.minimum = [math.inf] * width
        self.maximum = [-math.inf] * width

    def update(self, vector: Sequence[float]) -> None:
        if len(vector) != self.width:
            raise ValueError(f"expected feature width {self.width}")
        self.count += 1
        for index, raw in enumerate(vector):
            value = float(raw)
            delta = value - self.mean[index]
            self.mean[index] += delta / self.count
            delta2 = value - self.mean[index]
            self.m2[index] += delta * delta2
            self.minimum[index] = min(self.minimum[index], value)
            self.maximum[index] = max(self.maximum[index], value)

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "width": self.width,
                "mean": [0.0] * self.width,
                "std": [0.0] * self.width,
                "min": [0.0] * self.width,
                "max": [0.0] * self.width,
            }
        variance = [value / max(self.count - 1, 1) for value in self.m2]
        return {
            "count": self.count,
            "width": self.width,
            "mean": [round(value, 8) for value in self.mean],
            "std": [round(math.sqrt(value), 8) for value in variance],
            "min": [round(value, 8) for value in self.minimum],
            "max": [round(value, 8) for value in self.maximum],
        }


def build_dataset(
    episodes: Iterable[Mapping[str, Any]],
    config: DatasetConfig | None = None,
    *,
    task: str = "observation",
) -> dict[str, Any]:
    config = config or DatasetConfig()
    config.validate()
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task!r}")

    episode_list = list(episodes)
    reports: list[dict[str, Any]] = []
    accepted: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    accepted_schema_ids: set[str] = set()

    for episode in episode_list:
        report = assess_episode(episode, config, task=task)
        reports.append(report)
        if report["accepted"]:
            accepted.append((episode, report))
            accepted_schema_ids.add(str(report["feature_schema_id"]))

    if len(accepted_schema_ids) > 1:
        raise IncompatibleFeatureSchemaError(
            "multiple incompatible feature schemas were accepted: "
            + ", ".join(sorted(accepted_schema_ids))
            + "; use build_partitioned_datasets() or set "
              "DatasetConfig.feature_schema_id"
        )

    schema: FeatureSchema
    if accepted_schema_ids:
        schema = get_feature_schema(next(iter(accepted_schema_ids)))
    elif config.feature_schema_id is not None:
        schema = get_feature_schema(config.feature_schema_id)
    else:
        schema = get_feature_schema(LEGACY_SCHEMA_ID)

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    feature_stats = _FeatureStats(schema.width)
    total_steps = 0
    labeled_steps = 0

    for episode, report in accepted:
        total_steps += int(report["steps"])
        episode_segments = segment_episode(
            episode,
            replace(config, feature_schema_id=schema.schema_id),
        )
        destination = split_name(report["episode_id"], config)
        splits[destination].extend(episode_segments)
        for segment in episode_segments:
            for vector in segment["feature_vectors"]:
                feature_stats.update(vector)
            labeled_steps += sum(
                label is not None
                for label in segment["action_labels"]
            )

    all_segment_ids = [
        segment["segment_id"]
        for split_segments in splits.values()
        for segment in split_segments
    ]
    if len(all_segment_ids) != len(set(all_segment_ids)):
        raise ValueError("duplicate segment IDs detected")

    accepted_count = len(accepted)
    return {
        "dataset_version": DATASET_VERSION,
        "task": task,
        "feature_schema": {
            "schema_id": schema.schema_id,
            "version": schema.version,
            "width": schema.width,
            "compatibility_group": schema.compatibility_group,
        },
        "config": asdict(config),
        "quality_reports": reports,
        "splits": splits,
        "statistics": {
            "episodes_total": len(reports),
            "episodes_accepted": accepted_count,
            "episodes_rejected": len(reports) - accepted_count,
            "steps_in_accepted_episodes": total_steps,
            "segments_total": sum(len(value) for value in splits.values()),
            "segments_by_split": {
                name: len(value)
                for name, value in splits.items()
            },
            "labeled_segment_steps": labeled_steps,
            "feature_statistics": feature_stats.result(),
        },
    }


def build_partitioned_datasets(
    episodes: Iterable[Mapping[str, Any]],
    config: DatasetConfig | None = None,
    *,
    task: str = "observation",
) -> dict[str, dict[str, Any]]:
    config = config or DatasetConfig()
    config.validate()
    episode_list = list(episodes)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    rejected: list[Mapping[str, Any]] = []

    for episode in episode_list:
        try:
            schema = resolve_episode_schema(
                episode,
                allow_legacy_64_inference=config.allow_legacy_64_inference,
            )
        except FeatureSchemaError:
            rejected.append(episode)
            continue
        groups.setdefault(schema.schema_id, []).append(episode)

    result: dict[str, dict[str, Any]] = {}
    for schema_id, schema_episodes in sorted(groups.items()):
        schema_config = replace(config, feature_schema_id=schema_id)
        result[schema_id] = build_dataset(
            schema_episodes,
            schema_config,
            task=task,
        )

    if rejected:
        rejection_config = replace(config, feature_schema_id=None)
        result["_unresolved"] = {
            "dataset_version": DATASET_VERSION,
            "task": task,
            "feature_schema": None,
            "config": asdict(rejection_config),
            "quality_reports": [
                assess_episode(episode, rejection_config, task=task)
                for episode in rejected
            ],
            "splits": {"train": [], "validation": [], "test": []},
            "statistics": {
                "episodes_total": len(rejected),
                "episodes_accepted": 0,
                "episodes_rejected": len(rejected),
                "steps_in_accepted_episodes": 0,
                "segments_total": 0,
                "segments_by_split": {
                    "train": 0,
                    "validation": 0,
                    "test": 0,
                },
                "labeled_segment_steps": 0,
                "feature_statistics": None,
            },
        }
    return result


def write_dataset(
    dataset: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    manifest = {
        key: value
        for key, value in dataset.items()
        if key != "splits"
    }
    manifest["split_files"] = {}
    for split_name_value in ("train", "validation", "test"):
        path = root / f"{split_name_value}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for segment in dataset["splits"][split_name_value]:
                handle.write(
                    json.dumps(
                        segment,
                        separators=(",", ":"),
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                handle.write("\n")
        paths[split_name_value] = str(path)
        manifest["split_files"][split_name_value] = path.name

    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = str(manifest_path)
    return paths


def write_partitioned_datasets(
    datasets: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, dict[str, str]]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, str]] = {}
    for schema_id, dataset in datasets.items():
        destination = root / schema_id
        written[schema_id] = write_dataset(dataset, destination)
    return written


def inspect_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    labels = segment.get(
        "action_labels",
        segment.get("action_requests", []),
    )
    rewards = segment.get("rewards", [])
    return {
        "segment_id": segment["segment_id"],
        "episode_id": segment["episode_id"],
        "feature_schema_id": segment.get("feature_schema_id"),
        "feature_width": segment.get("feature_width"),
        "length": segment["length"],
        "time_span_ms": (
            int(segment["end_timestamp_ms"])
            - int(segment["start_timestamp_ms"])
        ),
        "labeled_steps": sum(label is not None for label in labels),
        "damage_dealt": sum(
            float(item.get("damage_dealt", 0))
            for item in rewards
        ),
        "damage_received": sum(
            float(item.get("damage_received", 0))
            for item in rewards
        ),
        "first_step_index": segment["start_step_index"],
        "last_step_index": segment["end_step_index"],
    }
