from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ranked_schema import (
    FEATURE_NAMES, SCHEMA_ALIASES, SCHEMA_ID, ACTION_NAMES, CanonicalStep,
    FileInventory, ParsedRecording, RankedReleaseError, extract_action_label,
    normalize_feature_vector,
)


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    step = record.get("step")
    return step if record.get("record_type") == "step" and isinstance(step, Mapping) else record


def _observation(step: Mapping[str, Any]) -> Mapping[str, Any]:
    value = step.get("observation")
    return value if isinstance(value, Mapping) else step


def _passive_failure(step: Mapping[str, Any], observation: Mapping[str, Any]) -> str | None:
    invariants = step.get("passive_invariants", observation.get("passive_invariants"))
    if isinstance(invariants, Mapping):
        for key, value in invariants.items():
            if (str(key).startswith("no_") or key == "controller_guard") and value is not True:
                return f"passive_invariant_{key}"
    if step.get("controller_active") is True or step.get("contaminated") is True:
        return "controller_or_contaminated_flag"
    executor = step.get("executor_result")
    if isinstance(executor, Mapping) and executor.get("status") not in (None, "not_applicable_passive_watcher"):
        return "non_passive_executor_result"
    return None


def parse_recording(path: str | Path, *, relative_to: str | Path | None = None, max_gap_ms: int = 1000) -> ParsedRecording:
    source = Path(path)
    raw = source.read_bytes()
    relative = str(source.relative_to(relative_to)) if relative_to else str(source)
    inv = FileInventory(relative.replace("\\", "/"), len(raw), hashlib.sha256(raw).hexdigest())
    header: dict[str, Any] = {}
    terminal: dict[str, Any] | None = None
    steps: list[CanonicalStep] = []
    last_lifecycle: Mapping[str, Any] | None = None
    opponent: str | None = None
    last_index: int | None = None
    last_timestamp: int | None = None
    seen: set[int] = set()
    previous_health: dict[str, float | None] = {"self": None, "target": None}

    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        inv.line_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            inv.parse_error_count += 1
            inv.parse_error_lines.append(line_number)
            continue
        if not isinstance(record, Mapping):
            inv.parse_error_count += 1
            inv.parse_error_lines.append(line_number)
            continue
        kind = record.get("record_type")
        if kind == "header":
            inv.header_count += 1
            value = record.get("header", record)
            if isinstance(value, Mapping):
                header = dict(value)
            continue
        if kind == "terminal":
            inv.terminal_record_count += 1
            value = record.get("terminal", record)
            terminal = dict(value) if isinstance(value, Mapping) else None
            continue
        if kind == "lifecycle":
            inv.lifecycle_record_count += 1
            last_lifecycle = record
            payload = record.get("payload")
            if record.get("kind") == "recording_started" and isinstance(payload, Mapping) and payload.get("opponent"):
                opponent = opponent or str(payload["opponent"])
            continue
        if kind == "event":
            inv.event_count += 1
            payload = record.get("payload")
            if record.get("event_type") == "target_changed" and isinstance(payload, Mapping) and payload.get("to"):
                opponent = opponent or str(payload["to"])
            continue
        if kind != "step":
            continue
        inv.step_count += 1
        step = _payload(record)
        observation = _observation(step)
        try:
            vector, encoding = normalize_feature_vector(observation.get("feature_vector", step.get("feature_vector")))
        except RankedReleaseError:
            vector, encoding = [], "invalid_or_missing"
        inv.vector_encoding_counts[encoding] = inv.vector_encoding_counts.get(encoding, 0) + 1
        inv.nonfinite_feature_count += sum(not math.isfinite(value) for value in vector)
        try:
            index = int(observation.get("step_index", step.get("step_index", step.get("index"))))
        except (TypeError, ValueError):
            index = -1
            inv.step_index_errors += 1
        try:
            timestamp = int(observation.get("timestamp_ms", step.get("timestamp_ms")))
        except (TypeError, ValueError):
            timestamp = -1
            inv.timestamp_errors += 1
        if index in seen:
            inv.duplicate_step_indices += 1
        seen.add(index)
        if last_index is not None and index <= last_index:
            inv.step_index_errors += 1
        if last_timestamp is not None:
            if timestamp < last_timestamp:
                inv.timestamp_errors += 1
            elif timestamp - last_timestamp > max_gap_ms:
                inv.unreasonable_timestamp_jumps += 1
        last_index, last_timestamp = index, timestamp
        failure = _passive_failure(step, observation)
        if failure:
            inv.contamination_state = "contaminated"
            inv.rejection_or_quarantine_reasons.append(failure)
        move_x, move_z, actions, label_available, label_source = extract_action_label(step)
        for name, active in actions.items():
            if active:
                inv.action_counts[name] += 1
        for state_name, state in (
            ("self", observation.get("self_state", step.get("self_state", step.get("self")))),
            ("target", observation.get("target_state", step.get("target_state", step.get("target")))),
        ):
            if isinstance(state, Mapping) and isinstance(state.get("health"), (int, float)):
                health = float(state["health"])
                previous = previous_health[state_name]
                if previous is not None and health != previous:
                    inv.health_changes += 1
                    if health < previous:
                        inv.damage_events += 1
                previous_health[state_name] = health
        raw_opponent = step.get("target_player", observation.get("target_player"))
        if raw_opponent:
            opponent = opponent or str(raw_opponent)
        steps.append(CanonicalStep(index, timestamp, vector, move_x, move_z, actions, label_available, label_source))

    inv.watcher_version = str(header.get("watcher_version")) if header.get("watcher_version") else None
    inv.stream_format = str(header.get("stream_format")) if header.get("stream_format") else None
    declared = header.get("feature_schema_id", header.get("feature_schema"))
    inv.schema_name = str(declared) if declared else None
    inv.schema_version = str(header.get("schema_version")) if header.get("schema_version") else None
    inv.feature_width = 72 if steps and all(len(step.feature_vector) == 72 for step in steps) else None
    inv.episode_id = str(header.get("episode_id", header.get("session_id", ""))) or None
    inv.roblox_job_id = str(header.get("job_id", "")) or None
    inv.opponent = str(header.get("opponent")) if header.get("opponent") else opponent
    inv.start_timestamp = int(header["started_at_ms"]) if header.get("started_at_ms") is not None else (steps[0].timestamp_ms if steps else None)
    terminal_end = terminal.get("ended_at_ms", terminal.get("end_timestamp_ms")) if terminal else None
    lifecycle_end = last_lifecycle.get("timestamp_ms") if last_lifecycle else None
    inv.end_timestamp = int(terminal_end or lifecycle_end or (steps[-1].timestamp_ms if steps else 0)) or None
    inv.duration_ms = int(terminal["duration_ms"]) if terminal and terminal.get("duration_ms") is not None else ((inv.end_timestamp - inv.start_timestamp) if inv.end_timestamp and inv.start_timestamp else None)
    inv.terminal_reason = str(terminal.get("reason")) if terminal and terminal.get("reason") else None
    if terminal is None and steps and not inv.timestamp_errors:
        inv.terminal_reconstructed = True
        inv.partial = True
        inv.terminal_reason = "reconstructed_player_removing_eof" if last_lifecycle and last_lifecycle.get("kind") == "player_removing" else "reconstructed_monotonic_eof"
    elif inv.terminal_reason in {"bounded_dry_run", "duplicate_instance_cleanup"}:
        inv.partial = True

    names = header.get("feature_names")
    if inv.schema_name in SCHEMA_ALIASES:
        inv.canonical_schema = SCHEMA_ID
        inv.schema_resolution = "declared_canonical" if inv.schema_name == SCHEMA_ID else "verified_alias"
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)) and tuple(names) != FEATURE_NAMES:
            inv.rejection_or_quarantine_reasons.append("feature_name_order_mismatch")
    elif inv.schema_name is None and isinstance(names, Sequence) and tuple(names) == FEATURE_NAMES:
        inv.canonical_schema = SCHEMA_ID
        inv.schema_resolution = "exact_feature_name_inference"

    reasons = inv.rejection_or_quarantine_reasons
    if inv.parse_error_count:
        reasons.append("parse_errors")
    if inv.header_count != 1:
        reasons.append(f"header_count_{inv.header_count}")
    if not steps:
        reasons.append("no_steps")
    if any(len(step.feature_vector) != 72 for step in steps):
        reasons.append("feature_vector_not_exact_72")
    if inv.nonfinite_feature_count:
        reasons.append("nonfinite_features")
    if inv.step_index_errors or inv.duplicate_step_indices:
        reasons.append("step_order_invalid")
    if inv.timestamp_errors:
        reasons.append("timestamp_order_invalid")
    if inv.contamination_state != "clean":
        reasons.append("passive_invariant_failure")
    if not inv.roblox_job_id:
        reasons.append("missing_job_id")
    if inv.canonical_schema != SCHEMA_ID:
        reasons.append("incompatible_or_unresolved_schema")
    inv.rejection_or_quarantine_reasons = list(dict.fromkeys(reasons))
    inv.admission_status = "candidate" if not inv.rejection_or_quarantine_reasons else "quarantined"
    digest = hashlib.sha256()
    for item in steps:
        digest.update(f"{item.step_index}|{item.timestamp_ms}|".encode())
        digest.update(",".join(f"{value:.9g}" for value in item.feature_vector).encode())
        digest.update(json.dumps({"move_x": item.move_x, "move_z": item.move_z, **item.actions}, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    inv.ordered_step_hash = digest.hexdigest()
    return ParsedRecording(inv, header, terminal, steps)


def discover_recordings(roots: Iterable[str | Path]) -> list[Path]:
    found: set[Path] = set()
    for value in roots:
        root = Path(value)
        if root.is_file() and root.suffix == ".jsonl":
            found.add(root.resolve())
        elif root.is_dir():
            found.update(path.resolve() for path in root.rglob("*.jsonl") if path.is_file())
    return sorted(found)


def resolve_duplicates(recordings: Sequence[ParsedRecording]) -> list[dict[str, Any]]:
    jobs: dict[str, list[ParsedRecording]] = defaultdict(list)
    for recording in recordings:
        key = recording.inventory.roblox_job_id or f"missing:{recording.inventory.relative_path}"
        jobs[key].append(recording)
    groups: list[dict[str, Any]] = []
    for job_id, members in sorted(jobs.items()):
        if len(members) < 2:
            continue
        chosen = max(members, key=lambda item: (
            int(item.inventory.relative_path.endswith(".clean.jsonl")),
            int(not item.inventory.rejection_or_quarantine_reasons),
            int(not item.inventory.partial), item.inventory.step_count,
        ))
        paths = sorted(item.inventory.relative_path for item in members)
        for item in members:
            item.inventory.duplicate_candidates = paths
            if item is not chosen:
                item.inventory.admission_status = "quarantined"
                item.inventory.rejection_or_quarantine_reasons.append(f"duplicate_of_{chosen.inventory.relative_path}")
        groups.append({
            "group_id": f"job:{job_id}", "job_id": job_id, "files": paths,
            "chosen": chosen.inventory.relative_path,
            "reason": "same Roblox job; prefer validated clean representation, then complete longest capture",
            "information_lost": False,
        })
    return groups


def inventory_corpus(roots: Iterable[str | Path], *, relative_to: str | Path | None = None) -> tuple[list[ParsedRecording], list[dict[str, Any]]]:
    recordings = [parse_recording(path, relative_to=relative_to) for path in discover_recordings(roots)]
    groups = resolve_duplicates(recordings)
    chosen = {group["chosen"] for group in groups}
    grouped = {path for group in groups for path in group["files"]}
    for recording in recordings:
        inv = recording.inventory
        if inv.relative_path in grouped and inv.relative_path not in chosen:
            continue
        if not inv.rejection_or_quarantine_reasons:
            inv.admission_status = "admitted"
    return recordings, groups


def assign_splits(recordings: Sequence[ParsedRecording], *, seed: int = 20260722, counts: tuple[int, int, int] = (7, 2, 2)) -> dict[str, list[str]]:
    admitted = [item for item in recordings if item.inventory.admission_status == "admitted"]
    if len(admitted) != sum(counts):
        raise RankedReleaseError(f"split counts require {sum(counts)} episodes, found {len(admitted)}")
    dry_runs = [item for item in admitted if item.inventory.terminal_reason == "bounded_dry_run"]
    others = [item for item in admitted if item not in dry_runs]
    others.sort(key=lambda item: int(hashlib.sha256(f"{seed}|{item.inventory.roblox_job_id}".encode()).hexdigest()[:8], 16))
    ordered = dry_runs + others
    train_count, validation_count, _ = counts
    partitions = {"train": ordered[:train_count], "validation": ordered[train_count:train_count + validation_count], "test": ordered[train_count + validation_count:]}
    splits: dict[str, list[str]] = {}
    for name, members in partitions.items():
        splits[name] = []
        for member in members:
            member.inventory.split_assignment = name
            splits[name].append(str(member.inventory.roblox_job_id))
    validate_split_isolation(recordings, splits)
    return splits


def validate_split_isolation(recordings: Sequence[ParsedRecording], splits: Mapping[str, Sequence[str]]) -> None:
    lookup = {item.inventory.roblox_job_id: item for item in recordings if item.inventory.admission_status == "admitted"}
    seen_jobs: set[str] = set()
    seen_episodes: set[str] = set()
    seen_hashes: set[str] = set()
    for split, jobs in splits.items():
        if not jobs:
            raise RankedReleaseError(f"{split} split is empty")
        for job in jobs:
            if job in seen_jobs:
                raise RankedReleaseError(f"job leakage: {job}")
            item = lookup.get(job)
            if item is None:
                raise RankedReleaseError(f"unknown admitted job: {job}")
            episode = str(item.inventory.episode_id)
            source_hash = item.inventory.sha256
            if episode in seen_episodes:
                raise RankedReleaseError(f"episode leakage: {episode}")
            if source_hash in seen_hashes:
                raise RankedReleaseError(f"source-hash leakage: {source_hash}")
            seen_jobs.add(job)
            seen_episodes.add(episode)
            seen_hashes.add(source_hash)
