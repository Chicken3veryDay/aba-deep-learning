from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from .feature_schemas import (
    RANKED_EXPLICIT_SCHEMA_ID,
    FeatureSchemaError,
    get_feature_schema,
)

RANKED_STREAM_FORMAT = "aba_ranked_passive_jsonl_v2"
RANKED_LABEL_SPACE = "camera_relative_input_v1"
SCHEMA_VERSION = "1.0.0"


class RankedStreamError(ValueError):
    pass


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RankedStreamError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RankedStreamError(f"{path} must be an array")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RankedStreamError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RankedStreamError(f"{path} must be finite")
    return result


def parse_ranked_jsonl(lines: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RankedStreamError(
                f"invalid ranked JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise RankedStreamError(
                f"ranked line {line_number} must contain an object"
            )
        records.append(value)
    return records


def _passive_invariants_ok(value: Any) -> bool:
    invariants = _mapping(value, "passive_invariants")
    return (
        invariants.get("controller_inactive") is True
        and invariants.get("no_input_simulation") is True
        and invariants.get("no_remote_calls") is True
        and invariants.get("no_ui") is True
    )


def _event_began(events: Sequence[Any], key: str) -> bool:
    return any(
        isinstance(event, Mapping)
        and event.get("key") == key
        and event.get("phase") == "began"
        for event in events
    )


def _event_ended(events: Sequence[Any], key: str) -> bool:
    return any(
        isinstance(event, Mapping)
        and event.get("key") == key
        and event.get("phase") == "ended"
        for event in events
    )


def _held_value(held: Mapping[str, Any], key: str) -> int:
    value = held.get(key, 0)
    if value is True:
        return 1
    if value is False or value is None:
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1 if float(value) > 0.5 else 0
    raise RankedStreamError(f"held.{key} must be boolean or numeric")


def action_label_from_ranked_step(step: Mapping[str, Any]) -> dict[str, Any]:
    held = _mapping(step.get("held"), "step.held")
    events = _sequence(step.get("input_events", []), "step.input_events")
    slot = 0
    for name, number in (
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
    ):
        if _event_began(events, name):
            slot = number
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "label_space": RANKED_LABEL_SPACE,
        "move_x": _held_value(held, "d") - _held_value(held, "a"),
        "move_z": _held_value(held, "w") - _held_value(held, "s"),
        "sprint": _held_value(held, "shift"),
        "block_held": _held_value(held, "f"),
        "m1_held": _held_value(held, "mouse1"),
        "m2_held": _held_value(held, "mouse2"),
        "transform_held": _held_value(held, "g"),
        "m1": _event_began(events, "mouse1"),
        "dodge": _event_began(events, "q"),
        "jump": _event_began(events, "space"),
        "block_start": _event_began(events, "f"),
        "block_stop": _event_ended(events, "f"),
        "move_slot": slot,
        "held": dict(held),
        "input_events": [
            dict(event)
            for event in events
            if isinstance(event, Mapping)
        ],
    }


def validate_ranked_records(records: Iterable[Any]) -> dict[str, Any]:
    header: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    steps: list[Mapping[str, Any]] = []
    events: list[Mapping[str, Any]] = []
    session_id: str | None = None
    previous_step_index: int | None = None
    previous_timestamp: int | None = None
    schema = get_feature_schema(RANKED_EXPLICIT_SCHEMA_ID)

    for line_number, raw in enumerate(records, start=1):
        record = _mapping(raw, f"record[{line_number}]")
        if record.get("schema_version") != SCHEMA_VERSION:
            raise RankedStreamError(
                f"record[{line_number}] schema_version mismatch"
            )
        record_type = record.get("record_type")
        if record_type == "header":
            if header is not None or steps or events or terminal is not None:
                raise RankedStreamError("ranked header must be first and unique")
            candidate = _mapping(record.get("header"), "header")
            if candidate.get("stream_format") != RANKED_STREAM_FORMAT:
                raise RankedStreamError("unsupported ranked stream format")
            try:
                declared_schema = get_feature_schema(
                    str(candidate.get("feature_schema"))
                )
            except FeatureSchemaError as exc:
                raise RankedStreamError(str(exc)) from exc
            if declared_schema.schema_id != schema.schema_id:
                raise RankedStreamError(
                    f"ranked stream must use {schema.schema_id}"
                )
            if int(candidate.get("feature_width", -1)) != schema.width:
                raise RankedStreamError("ranked header feature width mismatch")
            raw_session_id = candidate.get("session_id")
            if not isinstance(raw_session_id, str) or not raw_session_id:
                raise RankedStreamError("ranked header session_id is required")
            session_id = raw_session_id
            header = candidate
        elif record_type == "event":
            if header is None or terminal is not None:
                raise RankedStreamError("ranked event is out of order")
            if record.get("session_id") != session_id:
                raise RankedStreamError("ranked event session mismatch")
            _finite_number(record.get("timestamp_ms"), "event.timestamp_ms")
            events.append(record)
        elif record_type == "step":
            if header is None or terminal is not None:
                raise RankedStreamError("ranked step is out of order")
            if record.get("session_id") != session_id:
                raise RankedStreamError("ranked step session mismatch")
            step_index = int(
                _finite_number(record.get("step_index"), "step.step_index")
            )
            if (
                previous_step_index is not None
                and step_index <= previous_step_index
            ):
                raise RankedStreamError(
                    "ranked step indices must be strictly increasing"
                )
            timestamp = int(
                _finite_number(record.get("timestamp_ms"), "step.timestamp_ms")
            )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise RankedStreamError(
                    "ranked timestamps must be strictly increasing"
                )
            vector = _sequence(
                record.get("feature_vector"),
                "step.feature_vector",
            )
            if len(vector) != schema.width:
                raise RankedStreamError(
                    f"ranked feature_vector must contain {schema.width} values"
                )
            for index, value in enumerate(vector):
                number = _finite_number(
                    value,
                    f"step.feature_vector[{index}]",
                )
                if not -1.000001 <= number <= 1.000001:
                    raise RankedStreamError(
                        f"step.feature_vector[{index}] is outside [-1, 1]"
                    )
            if not _passive_invariants_ok(record.get("passive_invariants")):
                raise RankedStreamError(
                    f"passive invariant violation at step {step_index}"
                )
            _mapping(record.get("self"), "step.self")
            target = record.get("target")
            if target is not None:
                _mapping(target, "step.target")
            _mapping(record.get("camera"), "step.camera")
            _mapping(record.get("held"), "step.held")
            _sequence(record.get("input_events", []), "step.input_events")
            previous_step_index = step_index
            previous_timestamp = timestamp
            steps.append(record)
        elif record_type == "terminal":
            if header is None or terminal is not None:
                raise RankedStreamError("ranked terminal is out of order")
            if record.get("session_id") != session_id:
                raise RankedStreamError("ranked terminal session mismatch")
            candidate = _mapping(record.get("terminal"), "terminal")
            if not _passive_invariants_ok(candidate.get("passive_invariants")):
                raise RankedStreamError("terminal passive invariants failed")
            terminal = candidate
        else:
            raise RankedStreamError(
                f"unsupported ranked record_type: {record_type!r}"
            )

    if header is None:
        raise RankedStreamError("ranked stream is missing a header")
    if terminal is None:
        raise RankedStreamError("ranked stream is missing a terminal")
    if not steps:
        raise RankedStreamError("ranked stream contains no steps")
    return {
        "header": dict(header),
        "steps": [dict(step) for step in steps],
        "events": [dict(event) for event in events],
        "terminal": dict(terminal),
    }


def _health(state: Any) -> float | None:
    if not isinstance(state, Mapping):
        return None
    value = state.get("health")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _target_model(step: Mapping[str, Any]) -> Mapping[str, Any]:
    target = step.get("target")
    if not isinstance(target, Mapping):
        return {}
    model = target.get("model")
    return model if isinstance(model, Mapping) else {}


def convert_ranked_records(
    records: Iterable[Any],
    *,
    action_shift: int = 1,
) -> dict[str, Any]:
    if action_shift < 0:
        raise ValueError("action_shift must be non-negative")
    ranked = validate_ranked_records(records)
    source_header = ranked["header"]
    source_steps = ranked["steps"]
    session_id = str(source_header["session_id"])
    schema = get_feature_schema(RANKED_EXPLICIT_SCHEMA_ID)

    canonical_steps: list[dict[str, Any]] = []
    previous_self_health: float | None = None
    previous_target_health: float | None = None

    for index, raw in enumerate(source_steps):
        self_state = dict(_mapping(raw.get("self"), "step.self"))
        raw_target = raw.get("target")
        target_wrapper = (
            dict(raw_target)
            if isinstance(raw_target, Mapping)
            else {}
        )
        target_state = dict(_target_model(raw))
        target_state.setdefault(
            "name",
            target_wrapper.get("player_name")
            or target_wrapper.get("name")
            or "",
        )
        target_state["player_name"] = target_wrapper.get("player_name")
        target_state["user_id"] = target_wrapper.get("user_id")
        target_state["distance"] = target_wrapper.get("distance")

        self_health = _health(self_state)
        target_health = _health(target_state)
        damage_received = (
            max(0.0, previous_self_health - self_health)
            if previous_self_health is not None and self_health is not None
            else 0.0
        )
        damage_dealt = (
            max(0.0, previous_target_health - target_health)
            if previous_target_health is not None and target_health is not None
            else 0.0
        )
        previous_self_health = self_health
        previous_target_health = target_health

        label_index = index + action_shift
        action_label = (
            action_label_from_ranked_step(source_steps[label_index])
            if label_index < len(source_steps)
            else None
        )
        confirmations: list[dict[str, Any]] = []
        if damage_received > 0:
            confirmations.append({
                "kind": "damage_received",
                "amount": damage_received,
                "source": "ranked_health_delta",
                "confidence": 1.0,
            })
        if damage_dealt > 0:
            confirmations.append({
                "kind": "damage_dealt",
                "amount": damage_dealt,
                "source": "ranked_target_health_delta",
                "confidence": 1.0,
            })

        camera = dict(_mapping(raw.get("camera"), "step.camera"))
        held = dict(_mapping(raw.get("held"), "step.held"))
        input_events = [
            dict(event)
            for event in _sequence(
                raw.get("input_events", []),
                "step.input_events",
            )
            if isinstance(event, Mapping)
        ]
        canonical_steps.append({
            "observation": {
                "schema_version": SCHEMA_VERSION,
                "feature_schema_id": schema.schema_id,
                "episode_id": session_id,
                "step_index": int(raw["step_index"]),
                "timestamp_ms": int(raw["timestamp_ms"]),
                "dt_ms": max(
                    0,
                    min(1000, int(round(float(raw.get("dt_ms", 0))))),
                ),
                "self_state": self_state,
                "target_state": target_state,
                "relative": {
                    "distance": target_wrapper.get("distance"),
                    "target_present": bool(target_wrapper),
                },
                "combat": {
                    "self_markers": self_state.get("markers", {}),
                    "target_markers": target_state.get("markers", {}),
                },
                "cooldowns": {
                    "hud_text": raw.get("hud_text", []),
                    "hud_filter_version": raw.get("hud_filter_version"),
                },
                "network": {"ping_ms": raw.get("ping_ms", 0)},
                "history": {
                    "held": held,
                    "camera": camera,
                    "nearby_players": raw.get("nearby_players", []),
                    "monotonic_s": raw.get("monotonic_s"),
                },
                "feature_vector": list(raw["feature_vector"]),
            },
            "action_label": action_label,
            "executor_result": {
                "status": "passive_observation",
                "accepted": None,
                "source": RANKED_STREAM_FORMAT,
            },
            "confirmations": confirmations,
            "rewards": {
                "damage_dealt": damage_dealt,
                "damage_received": damage_received,
                "invalid_request": 0.0,
            },
            "raw_events": input_events,
            "source_record": {
                "job_id": raw.get("job_id"),
                "place_id": raw.get("place_id"),
                "passive_invariants": dict(raw["passive_invariants"]),
            },
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "header": {
            "episode_id": session_id,
            "stream_format": RANKED_STREAM_FORMAT,
            "source_stream_format": RANKED_STREAM_FORMAT,
            "feature_schema_id": schema.schema_id,
            "feature_width": schema.width,
            "place_id": source_header.get("place_id"),
            "job_id": source_header.get("job_id"),
            "collector_version": source_header.get("watcher_version"),
            "watcher_version": source_header.get("watcher_version"),
            "watcher_source_sha256": source_header.get(
                "watcher_source_sha256"
            ),
            "source_sha256": source_header.get("source_sha256"),
            "sample_hz": source_header.get("sample_hz"),
            "started_at_ms": source_header.get("started_at_ms"),
            "mode": source_header.get("mode"),
            "controller_guard": source_header.get("controller_guard"),
            "action_label_space": RANKED_LABEL_SPACE,
            "action_shift": action_shift,
            "feature_names": source_header.get("feature_names", []),
        },
        "steps": canonical_steps,
        "terminal": {
            **ranked["terminal"],
            "source_events": ranked["events"],
        },
    }


def read_ranked_stream(
    source: str | Path | TextIO,
    *,
    action_shift: int = 1,
) -> dict[str, Any]:
    if hasattr(source, "read"):
        records = parse_ranked_jsonl(source)
    else:
        with Path(source).open("r", encoding="utf-8") as handle:
            records = parse_ranked_jsonl(handle)
    return convert_ranked_records(records, action_shift=action_shift)
