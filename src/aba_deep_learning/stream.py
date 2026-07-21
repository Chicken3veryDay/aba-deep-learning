from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

from .contracts import ContractError, SCHEMA_VERSION, validate_step

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
            raise ContractError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"line {line_number} must contain an object")
        records.append(value)
    return records


def validate_stream_records(records: Iterable[Any]) -> dict[str, Any]:
    header: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    steps: list[Mapping[str, Any]] = []
    previous_index: int | None = None

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
            if not isinstance(candidate.get("episode_id"), str) or not candidate["episode_id"]:
                raise ContractError("header.episode_id is required")
            header = candidate
        elif record_type == "step":
            if header is None or terminal is not None:
                raise ContractError("step is out of order")
            candidate = raw.get("step")
            if not isinstance(candidate, Mapping):
                raise ContractError("step must be an object")
            previous_index = validate_step(candidate, previous_index)
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


def read_episode_stream(source: str | Path | TextIO) -> dict[str, Any]:
    if hasattr(source, "read"):
        return validate_stream_records(parse_jsonl(source))
    with Path(source).open("r", encoding="utf-8") as handle:
        return validate_stream_records(parse_jsonl(handle))


def summarize_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    steps = episode.get("steps", [])
    terminal = episode.get("terminal", {})
    return {
        "episode_id": episode["header"]["episode_id"],
        "steps": len(steps),
        "duration_ms": int(terminal.get("duration_ms", 0)),
        "damage_dealt": sum(float(step["rewards"].get("damage_dealt", 0)) for step in steps),
        "damage_received": sum(float(step["rewards"].get("damage_received", 0)) for step in steps),
        "requests": sum(step.get("action_request") is not None for step in steps),
        "confirmations": sum(len(step.get("confirmations", [])) for step in steps),
        "reason": terminal.get("reason"),
    }
