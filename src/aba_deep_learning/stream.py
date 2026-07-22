from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

from .contracts import ContractError, SCHEMA_VERSION, validate_step
from .feature_schemas import (
    LEGACY_SCHEMA_ID,
    FeatureSchemaError,
    get_feature_schema,
)

STREAM_FORMAT = "aba_episode_jsonl_v1"


def parse_jsonl(lines: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"invalid JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ContractError(f"line {line_number} must contain an object")
        records.append(value)
    return records


def _header_schema(header: Mapping[str, Any]) -> tuple[str, int]:
    for key in (
        "feature_schema_id",
        "feature_schema",
        "feature_vector_schema",
        "observation_schema_id",
    ):
        value = header.get(key)
        if isinstance(value, str) and value.strip():
            try:
                schema = get_feature_schema(value)
            except FeatureSchemaError as exc:
                raise ContractError(str(exc)) from exc
            return schema.schema_id, schema.width
    schema = get_feature_schema(LEGACY_SCHEMA_ID)
    return schema.schema_id, schema.width


def validate_stream_records(
    records: Iterable[Any],
    *,
    require_action_masks: bool = True,
) -> dict[str, Any]:
    header: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    steps: list[Mapping[str, Any]] = []
    previous_index: int | None = None
    feature_schema_id = LEGACY_SCHEMA_ID
    feature_width = 64

    for line_number, raw in enumerate(records, start=1):
        if not isinstance(raw, Mapping):
            raise ContractError(f"record[{line_number}] must be an object")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ContractError(f"record[{line_number}] schema mismatch")

        record_type = raw.get("record_type")
        if record_type == "header":
            if header is not None or steps or terminal is not None:
                raise ContractError("header must be first and unique")
            candidate = raw.get("header")
            if not isinstance(candidate, Mapping):
                raise ContractError("header must be an object")
            if candidate.get("stream_format") != STREAM_FORMAT:
                raise ContractError("unsupported stream format")
            if (
                not isinstance(candidate.get("episode_id"), str)
                or not candidate["episode_id"]
            ):
                raise ContractError("header.episode_id is required")
            feature_schema_id, feature_width = _header_schema(candidate)
            header = dict(candidate)
            header.setdefault("feature_schema_id", feature_schema_id)
            header.setdefault("feature_width", feature_width)
        elif record_type == "step":
            if header is None or terminal is not None:
                raise ContractError("step is out of order")
            candidate = raw.get("step")
            if not isinstance(candidate, Mapping):
                raise ContractError("step must be an object")
            previous_index = validate_step(
                candidate,
                previous_index,
                expected_feature_width=feature_width,
                require_action_mask=require_action_masks,
            )
            if candidate["observation"]["episode_id"] != header["episode_id"]:
                raise ContractError("step episode_id does not match header")
            steps.append(candidate)
        elif record_type == "terminal":
            if header is None or terminal is not None:
                raise ContractError("terminal is out of order")
            candidate = raw.get("terminal")
            if not isinstance(candidate, Mapping):
                raise ContractError("terminal must be an object")
            terminal = candidate
        else:
            raise ContractError(f"unsupported record_type: {record_type!r}")

    if header is None:
        raise ContractError("stream is missing a header")
    if terminal is None:
        raise ContractError("stream is missing a terminal")

    return {
        "schema_version": SCHEMA_VERSION,
        "header": dict(header),
        "steps": [dict(step) for step in steps],
        "terminal": dict(terminal),
    }


def _stream_format(records: list[dict[str, Any]]) -> str | None:
    if not records:
        return None
    first = records[0]
    if first.get("record_type") != "header":
        return None
    header = first.get("header")
    if not isinstance(header, Mapping):
        return None
    value = header.get("stream_format")
    return value if isinstance(value, str) else None


def read_episode_stream(
    source: str | Path | TextIO,
    *,
    ranked_action_shift: int = 1,
    require_action_masks: bool = True,
) -> dict[str, Any]:
    if hasattr(source, "read"):
        records = parse_jsonl(source)
    else:
        with Path(source).open("r", encoding="utf-8") as handle:
            records = parse_jsonl(handle)

    stream_format = _stream_format(records)
    if stream_format == STREAM_FORMAT:
        return validate_stream_records(
            records,
            require_action_masks=require_action_masks,
        )

    from .ranked_stream import (  # imported lazily to avoid a module cycle
        RANKED_STREAM_FORMAT,
        convert_ranked_records,
    )

    if stream_format == RANKED_STREAM_FORMAT:
        return convert_ranked_records(
            records,
            action_shift=ranked_action_shift,
        )

    from .ranked_stream_v3 import (
        RANKED_STREAM_FORMAT_V3,
        convert_ranked_v3_records,
    )

    if stream_format == RANKED_STREAM_FORMAT_V3:
        return convert_ranked_v3_records(
            records,
            action_shift=ranked_action_shift,
        )
    raise ContractError(f"unsupported stream format: {stream_format!r}")


def summarize_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    steps = episode.get("steps", [])
    terminal = episode.get("terminal", {})
    return {
        "episode_id": episode["header"]["episode_id"],
        "feature_schema_id": episode["header"].get(
            "feature_schema_id",
            LEGACY_SCHEMA_ID,
        ),
        "steps": len(steps),
        "duration_ms": int(terminal.get("duration_ms", 0)),
        "damage_dealt": sum(
            float(step["rewards"].get("damage_dealt", 0))
            for step in steps
        ),
        "damage_received": sum(
            float(step["rewards"].get("damage_received", 0))
            for step in steps
        ),
        "requests": sum(
            step.get("action_request") is not None
            or step.get("action_label") is not None
            for step in steps
        ),
        "confirmations": sum(
            len(step.get("confirmations", []))
            for step in steps
        ),
        "reason": terminal.get("reason"),
    }
