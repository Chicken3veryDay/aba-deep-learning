from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from .contracts import ContractError, validate_observation
from .feature_schemas import (
    RANKED_EXPLICIT_SCHEMA_ID,
    FeatureSchemaError,
    get_feature_schema,
)
from .ranked_stream import RANKED_LABEL_SPACE

RANKED_STREAM_FORMAT_V3 = "aba_ranked_passive_jsonl_v3"
RANKED_RECORD_SCHEMA_VERSION_V3 = "3.0.0"
CANONICAL_SCHEMA_VERSION = "1.0.0"


class RankedStreamV3Error(ValueError):
    pass


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RankedStreamV3Error(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, Mapping) and not value:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RankedStreamV3Error(f"{path} must be an array")
    return value


def _finite_number(
    value: Any,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RankedStreamV3Error(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RankedStreamV3Error(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise RankedStreamV3Error(f"{path} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise RankedStreamV3Error(f"{path} must be <= {maximum}")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise RankedStreamV3Error(f"{path} must be boolean")
    return value


def _parse_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RankedStreamV3Error(
                f"invalid ranked v3 JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise RankedStreamV3Error(
                f"ranked v3 line {line_number} must contain an object"
            )
        records.append(value)
    return records


def _validate_action_label(value: Any, path: str) -> dict[str, Any]:
    label = dict(_mapping(value, path))
    for key in ("move_x", "move_z"):
        number = _finite_number(label.get(key), f"{path}.{key}", -1, 1)
        if number not in (-1.0, 0.0, 1.0):
            raise RankedStreamV3Error(
                f"{path}.{key} must be one of -1, 0, or 1"
            )
    _finite_number(label.get("sprint"), f"{path}.sprint", 0, 1)
    for key in (
        "block_held",
        "transform_held",
        "m1",
        "dodge",
        "jump",
        "block_start",
        "block_stop",
    ):
        _boolean(label.get(key), f"{path}.{key}")
    slot = int(_finite_number(label.get("move_slot"), f"{path}.move_slot", 0, 4))
    if slot not in (0, 1, 2, 3, 4):
        raise RankedStreamV3Error(f"{path}.move_slot must be in 0..4")
    if label.get("source") != "UserInputService":
        raise RankedStreamV3Error(
            f"{path}.source must equal 'UserInputService'"
        )
    return label


def _canonical_action_label(
    source_step: Mapping[str, Any],
) -> dict[str, Any]:
    label = _validate_action_label(
        source_step.get("action_label"),
        "step.action_label",
    )
    observation = _mapping(source_step.get("observation"), "step.observation")
    history = _mapping(observation.get("history"), "observation.history")
    held = history.get("held", {})
    if not isinstance(held, Mapping):
        raise RankedStreamV3Error("observation.history.held must be an object")
    events = _sequence(
        source_step.get("raw_events", []),
        "step.raw_events",
    )
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "label_space": RANKED_LABEL_SPACE,
        **label,
        "held": dict(held),
        "input_events": [
            dict(event)
            for event in events
            if isinstance(event, Mapping)
        ],
    }


def validate_ranked_v3_records(
    records: Iterable[Any],
) -> dict[str, Any]:
    header: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    steps: list[Mapping[str, Any]] = []
    episode_id: str | None = None
    previous_step_index: int | None = None
    previous_timestamp: int | None = None
    schema = get_feature_schema(RANKED_EXPLICIT_SCHEMA_ID)

    for line_number, raw in enumerate(records, start=1):
        record = _mapping(raw, f"record[{line_number}]")
        if record.get("schema_version") != RANKED_RECORD_SCHEMA_VERSION_V3:
            raise RankedStreamV3Error(
                f"record[{line_number}] schema_version mismatch"
            )
        record_type = record.get("record_type")

        if record_type == "header":
            if header is not None or steps or terminal is not None:
                raise RankedStreamV3Error("ranked v3 header must be first and unique")
            candidate = _mapping(record.get("header"), "header")
            if candidate.get("stream_format") != RANKED_STREAM_FORMAT_V3:
                raise RankedStreamV3Error("unsupported ranked v3 stream format")
            if candidate.get("schema_version") != RANKED_RECORD_SCHEMA_VERSION_V3:
                raise RankedStreamV3Error("ranked v3 header schema mismatch")
            try:
                declared_schema = get_feature_schema(
                    str(candidate.get("feature_schema_id"))
                )
            except FeatureSchemaError as exc:
                raise RankedStreamV3Error(str(exc)) from exc
            if declared_schema.schema_id != schema.schema_id:
                raise RankedStreamV3Error(
                    f"ranked v3 stream must use {schema.schema_id}"
                )
            if int(candidate.get("feature_width", -1)) != schema.width:
                raise RankedStreamV3Error("ranked v3 header feature width mismatch")
            names = _sequence(candidate.get("feature_names"), "header.feature_names")
            if len(names) != schema.width or len(set(names)) != schema.width:
                raise RankedStreamV3Error(
                    "ranked v3 feature names must contain 72 unique entries"
                )
            raw_episode_id = candidate.get("episode_id")
            if not isinstance(raw_episode_id, str) or not raw_episode_id:
                raise RankedStreamV3Error("ranked v3 header episode_id is required")
            if candidate.get("observation_only") is not True:
                raise RankedStreamV3Error("ranked v3 header must be observation-only")
            if candidate.get("executor_integration_allowed") is not False:
                raise RankedStreamV3Error(
                    "ranked v3 header must prohibit executor integration"
                )
            if candidate.get("input_source") != "UserInputService":
                raise RankedStreamV3Error(
                    "ranked v3 header input_source must be UserInputService"
                )
            episode_id = raw_episode_id
            header = candidate

        elif record_type == "step":
            if header is None or terminal is not None:
                raise RankedStreamV3Error("ranked v3 step is out of order")
            candidate = _mapping(record.get("step"), "step")
            observation = _mapping(candidate.get("observation"), "step.observation")
            if observation.get("episode_id") != episode_id:
                raise RankedStreamV3Error("ranked v3 step episode mismatch")
            try:
                validate_observation(
                    observation,
                    expected_feature_width=schema.width,
                )
            except ContractError as exc:
                raise RankedStreamV3Error(str(exc)) from exc
            if observation.get("feature_schema_id") != schema.schema_id:
                raise RankedStreamV3Error("ranked v3 observation schema mismatch")
            step_index = int(
                _finite_number(
                    observation.get("step_index"),
                    "observation.step_index",
                    0,
                )
            )
            if previous_step_index is not None and step_index <= previous_step_index:
                raise RankedStreamV3Error(
                    "ranked v3 step indices must be strictly increasing"
                )
            timestamp = int(
                _finite_number(
                    observation.get("timestamp_ms"),
                    "observation.timestamp_ms",
                    0,
                )
            )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise RankedStreamV3Error(
                    "ranked v3 timestamps must be strictly increasing"
                )
            vector = _sequence(
                observation.get("feature_vector"),
                "observation.feature_vector",
            )
            for index, item in enumerate(vector):
                number = _finite_number(
                    item,
                    f"observation.feature_vector[{index}]",
                )
                if not -1.000001 <= number <= 1.000001:
                    raise RankedStreamV3Error(
                        f"observation.feature_vector[{index}] is outside [-1, 1]"
                    )
            _validate_action_label(candidate.get("action_label"), "step.action_label")
            executor = _mapping(candidate.get("executor_result"), "step.executor_result")
            if executor.get("status") != "not_applicable_passive_watcher":
                raise RankedStreamV3Error(
                    "ranked v3 executor_result must remain passive"
                )
            if executor.get("source") != header.get("watcher_version"):
                raise RankedStreamV3Error(
                    "ranked v3 executor source must match watcher version"
                )
            _sequence(candidate.get("confirmations", []), "step.confirmations")
            _sequence(candidate.get("raw_events", []), "step.raw_events")
            rewards = _mapping(candidate.get("rewards"), "step.rewards")
            _finite_number(rewards.get("damage_dealt", 0), "rewards.damage_dealt", 0)
            _finite_number(
                rewards.get("damage_received", 0),
                "rewards.damage_received",
                0,
            )
            previous_step_index = step_index
            previous_timestamp = timestamp
            steps.append(candidate)

        elif record_type == "terminal":
            if header is None or terminal is not None:
                raise RankedStreamV3Error("ranked v3 terminal is out of order")
            candidate = _mapping(record.get("terminal"), "terminal")
            if candidate.get("executor_integration_allowed") is not False:
                raise RankedStreamV3Error(
                    "ranked v3 terminal must prohibit executor integration"
                )
            if candidate.get("contaminated") is True:
                raise RankedStreamV3Error("ranked v3 episode is contaminated")
            if candidate.get("reason") == "controller_contamination":
                raise RankedStreamV3Error("ranked v3 controller contamination")
            terminal = candidate
        else:
            raise RankedStreamV3Error(
                f"unsupported ranked v3 record_type: {record_type!r}"
            )

    if header is None:
        raise RankedStreamV3Error("ranked v3 stream is missing a header")
    if terminal is None:
        raise RankedStreamV3Error("ranked v3 stream is missing a terminal")
    if not steps:
        raise RankedStreamV3Error("ranked v3 stream contains no steps")
    if int(terminal.get("steps", len(steps))) != len(steps):
        raise RankedStreamV3Error("ranked v3 terminal step count mismatch")

    return {
        "header": dict(header),
        "steps": [dict(step) for step in steps],
        "terminal": dict(terminal),
    }


def convert_ranked_v3_records(
    records: Iterable[Any],
    *,
    action_shift: int = 1,
) -> dict[str, Any]:
    if action_shift < 0:
        raise ValueError("action_shift must be non-negative")
    ranked = validate_ranked_v3_records(records)
    source_header = ranked["header"]
    source_steps = ranked["steps"]
    episode_id = str(source_header["episode_id"])
    schema = get_feature_schema(RANKED_EXPLICIT_SCHEMA_ID)

    canonical_steps: list[dict[str, Any]] = []
    for index, raw in enumerate(source_steps):
        observation = dict(_mapping(raw.get("observation"), "step.observation"))
        observation["feature_schema_id"] = schema.schema_id
        label_index = index + action_shift
        action_label = (
            _canonical_action_label(source_steps[label_index])
            if label_index < len(source_steps)
            else None
        )
        confirmations = [
            dict(item)
            for item in _sequence(
                raw.get("confirmations", []),
                "step.confirmations",
            )
            if isinstance(item, Mapping)
        ]
        raw_events = [
            dict(item)
            for item in _sequence(raw.get("raw_events", []), "step.raw_events")
            if isinstance(item, Mapping)
        ]
        source_rewards = _mapping(raw.get("rewards"), "step.rewards")
        canonical_steps.append({
            "observation": observation,
            "action_label": action_label,
            "executor_result": {
                "status": "passive_observation",
                "accepted": None,
                "source": RANKED_STREAM_FORMAT_V3,
            },
            "confirmations": confirmations,
            "rewards": {
                "damage_dealt": float(source_rewards.get("damage_dealt", 0)),
                "damage_received": float(
                    source_rewards.get("damage_received", 0)
                ),
                "invalid_request": 0.0,
            },
            "raw_events": raw_events,
            "source_record": {
                "watcher_version": source_header.get("watcher_version"),
                "input_source": source_header.get("input_source"),
                "observation_only": True,
                "executor_integration_allowed": False,
            },
        })

    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "header": {
            "episode_id": episode_id,
            "stream_format": RANKED_STREAM_FORMAT_V3,
            "source_stream_format": RANKED_STREAM_FORMAT_V3,
            "feature_schema_id": schema.schema_id,
            "feature_width": schema.width,
            "place_id": source_header.get("place_id"),
            "place_version": source_header.get("place_version"),
            "job_id": source_header.get("job_id"),
            "independence_group_id": source_header.get("job_id") or episode_id,
            "collector_version": source_header.get("watcher_version"),
            "watcher_version": source_header.get("watcher_version"),
            "sample_hz": source_header.get("sample_hz"),
            "started_at_ms": source_header.get("started_at_ms"),
            "player": source_header.get("player"),
            "opponent": source_header.get("opponent"),
            "observation_only": True,
            "executor_integration_allowed": False,
            "action_label_space": RANKED_LABEL_SPACE,
            "action_shift": action_shift,
            "feature_names": source_header.get("feature_names", []),
        },
        "steps": canonical_steps,
        "terminal": dict(ranked["terminal"]),
    }


def read_ranked_v3_stream(
    source: str | Path | TextIO,
    *,
    action_shift: int = 1,
) -> dict[str, Any]:
    if hasattr(source, "read"):
        records = _parse_lines(source)
    else:
        with Path(source).open("r", encoding="utf-8") as handle:
            records = _parse_lines(handle)
    return convert_ranked_v3_records(records, action_shift=action_shift)
