from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from aba_deep_learning.dataset import DatasetConfig, assess_episode
from aba_deep_learning.feature_schemas import RANKED_EXPLICIT_SCHEMA_ID
from aba_deep_learning.stream import read_episode_stream, summarize_episode


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_counts(episode: dict[str, Any]) -> dict[str, int]:
    counts = {
        "labeled_steps": 0,
        "moving_steps": 0,
        "sprint_steps": 0,
        "m1": 0,
        "dodge": 0,
        "jump": 0,
        "block_start": 0,
        "block_stop": 0,
        "move1": 0,
        "move2": 0,
        "move3": 0,
        "move4": 0,
        "transform_held_steps": 0,
    }
    for step in episode.get("steps", []):
        label = step.get("action_label")
        if not isinstance(label, dict):
            continue
        counts["labeled_steps"] += 1
        if label.get("move_x") or label.get("move_z"):
            counts["moving_steps"] += 1
        if label.get("sprint"):
            counts["sprint_steps"] += 1
        for key in ("m1", "dodge", "jump", "block_start", "block_stop"):
            if label.get(key):
                counts[key] += 1
        slot = int(label.get("move_slot") or 0)
        if 1 <= slot <= 4:
            counts[f"move{slot}"] += 1
        if label.get("transform_held"):
            counts["transform_held_steps"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize an ABA ranked passive JSONL stream"
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--action-shift", type=int, default=1)
    parser.add_argument("--min-quality-score", type=float, default=0.70)
    args = parser.parse_args()

    actual_hash = sha256(args.path)
    if args.expected_sha256 and actual_hash != args.expected_sha256.lower():
        raise SystemExit(
            f"SHA-256 mismatch: expected {args.expected_sha256}, got {actual_hash}"
        )

    episode = read_episode_stream(
        args.path,
        ranked_action_shift=args.action_shift,
    )
    config = DatasetConfig(
        min_quality_score=args.min_quality_score,
        min_action_coverage=0.50,
        feature_schema_id=RANKED_EXPLICIT_SCHEMA_ID,
        require_action_masks=False,
    )
    observation_report = assess_episode(
        episode,
        config,
        task="observation",
    )
    behavior_report = assess_episode(
        episode,
        config,
        task="behavior_cloning",
    )
    summary = summarize_episode(episode)
    terminal = episode.get("terminal", {})
    source_events = terminal.get("source_events", [])

    print(
        json.dumps(
            {
                "path": str(args.path),
                "sha256": actual_hash,
                "summary": summary,
                "action_shift": args.action_shift,
                "action_counts": action_counts(episode),
                "source_event_count": len(source_events),
                "terminal_reason": terminal.get("reason"),
                "terminal_steps": terminal.get("steps"),
                "observation_admission": observation_report,
                "behavior_cloning_admission": behavior_report,
                "executor_integration_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
