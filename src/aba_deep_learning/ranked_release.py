from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .ranked_inventory import (
    assign_splits, discover_recordings, inventory_corpus, parse_recording,
    resolve_duplicates, validate_split_isolation,
)
from .ranked_schema import (
    FEATURE_NAMES, SCHEMA_ALIASES, SCHEMA_ID, SCHEMA_VERSION, ACTION_NAMES,
    CanonicalStep, FileInventory, ParsedRecording, RankedReleaseError,
    extract_action_label, normalize_feature_vector,
)

__all__ = [
    "ACTION_NAMES", "FEATURE_NAMES", "SCHEMA_ID", "CanonicalStep",
    "FileInventory", "ParsedRecording", "RankedReleaseError",
    "normalize_feature_vector", "extract_action_label", "parse_recording",
    "discover_recordings", "resolve_duplicates", "inventory_corpus",
    "assign_splits", "validate_split_isolation", "build_release",
]


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(recordings: list[ParsedRecording]) -> str:
    lines = ["# Ranked recording inventory", "", "| File | Job | Steps | Schema | Terminal | Verdict |", "|---|---|---:|---|---|---|"]
    for recording in recordings:
        inv = recording.inventory
        reason = ", ".join(inv.rejection_or_quarantine_reasons) or "validated"
        lines.append(f"| `{inv.relative_path}` | `{inv.roblox_job_id or ''}` | {inv.step_count} | `{inv.canonical_schema or inv.schema_name or 'unresolved'}` | `{inv.terminal_reason or 'missing'}` | {inv.admission_status}: {reason} |")
    return "\n".join(lines) + "\n"


def build_release(
    roots: Iterable[str | Path], output: str | Path, *,
    relative_to: str | Path | None = None, split_seed: int = 20260722,
    split_counts: tuple[int, int, int] = (7, 2, 2),
) -> dict[str, Any]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    recordings, duplicates = inventory_corpus(roots, relative_to=relative_to)
    splits = assign_splits(recordings, seed=split_seed, counts=split_counts)
    admitted = [item for item in recordings if item.inventory.admission_status == "admitted"]
    quarantined = [item for item in recordings if item.inventory.admission_status != "admitted"]
    episodes: list[dict[str, Any]] = []
    for recording in admitted:
        inv = recording.inventory
        episodes.append({
            "canonical_episode_id": inv.episode_id,
            "roblox_job_id": inv.roblox_job_id,
            "source_file_hashes": [inv.sha256],
            "canonical_file_hash": inv.ordered_step_hash,
            "source_path": inv.relative_path,
            "opponent": inv.opponent,
            "watcher_version": inv.watcher_version,
            "schema": SCHEMA_ID,
            "original_declared_schema": inv.schema_name,
            "source_steps": len(recording.steps),
            "steps": max(0, len(recording.steps) - 1),
            "duration_ms": inv.duration_ms,
            "action_coverage": inv.action_counts,
            "terminal_status": "partial" if inv.partial else "complete",
            "terminal_reason": inv.terminal_reason,
            "contamination_verdict": "clean_passive",
            "derivation_history": (
                (["terminal_reconstructed_at_last_observed_timestamp"] if inv.terminal_reconstructed else [])
                + (["numeric_key_object_to_ordered_array"] if inv.vector_encoding_counts.get("numeric_key_object_1_72") else [])
            ),
            "split_assignment": inv.split_assignment,
        })
    manifest = {
        "dataset_id": "ranked_explicit_v3_72/v1",
        "dataset_version": "1.0.0",
        "canonical_schema": SCHEMA_ID,
        "feature_width": 72,
        "split_seed": split_seed,
        "split_algorithm": "bounded dry-run forced to train, remaining jobs ordered by SHA-256(seed|job_id)",
        "splits": splits,
        "admitted_episode_count": len(admitted),
        "quarantined_file_count": len(quarantined),
        "duplicate_group_count": len(duplicates),
        "total_admitted_steps": sum(max(0, len(item.steps) - 1) for item in admitted),
        "total_admitted_duration_ms": sum(item.inventory.duration_ms or 0 for item in admitted),
        "episodes": episodes,
    }
    _dump(destination / "manifest.json", manifest)
    _dump(destination / "admission-report.json", {"files": [asdict(item.inventory) for item in recordings], "duplicate_groups": duplicates})
    _dump(destination / "duplicate-groups.json", duplicates)
    _dump(destination / "quarantine-manifest.json", {"files": [asdict(item.inventory) for item in quarantined]})
    _dump(destination / "schema.json", {
        "schema_id": SCHEMA_ID, "version": SCHEMA_VERSION, "width": 72,
        "feature_names": list(FEATURE_NAMES), "aliases": sorted(SCHEMA_ALIASES - {SCHEMA_ID}),
        "numeric_key_object_migration": {
            "accepted_keys": [str(index) for index in range(1, 73)],
            "requires_all_indices_exactly_once": True,
            "output_order": "numeric ascending 1..72",
        },
    })
    _dump(destination / "schema-provenance.json", {
        "canonical_schema": SCHEMA_ID,
        "alias_equivalence": "feature names and order must exactly match schema.json",
        "original_declared_schema_preserved": True,
        "producer_reference_sha256": "d52b1795114dc287f1f91f33c487c96ad5f44f4694694b9202b0e8b75feff63f",
    })
    for split in ("train", "validation", "test"):
        _dump(destination / f"{split}.json", {
            "job_ids": splits[split],
            "episode_ids": [item["canonical_episode_id"] for item in episodes if item["split_assignment"] == split],
        })
    (destination / "admission-report.md").write_text(_markdown(recordings), encoding="utf-8")
    (destination / "manifest.md").write_text(
        "# Ranked explicit v3 dataset v1\n\n"
        f"- Admitted episodes: {len(admitted)}\n"
        f"- Quarantined files: {len(quarantined)}\n"
        f"- Aligned steps: {manifest['total_admitted_steps']}\n"
        f"- Duration: {manifest['total_admitted_duration_ms'] / 1000:.3f} seconds\n"
        f"- Split seed: {split_seed}\n",
        encoding="utf-8",
    )
    checksums = [f"{_sha256(path)}  {path.name}" for path in sorted(destination.iterdir()) if path.is_file() and path.name != "checksums.sha256"]
    (destination / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return {**manifest, "manifest_sha256": _sha256(destination / "manifest.json"), "duplicate_groups": duplicates}
