from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aba_deep_learning.dataset import (  # noqa: E402
    DatasetConfig,
    build_dataset,
    discover_episode_files,
    load_episodes,
    write_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic ABA learning datasets")
    parser.add_argument("inputs", nargs="+", help="Episode JSONL files or directories")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument(
        "--task",
        choices=("observation", "behavior_cloning"),
        default="observation",
    )
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--min-segment-steps", type=int, default=8)
    parser.add_argument("--max-gap-ms", type=int, default=250)
    parser.add_argument("--min-quality-score", type=float, default=0.70)
    parser.add_argument("--min-action-coverage", type=float, default=0.50)
    parser.add_argument("--split-seed", default="aba-step3-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for value in args.inputs:
        paths.extend(discover_episode_files(value))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No .jsonl episode files found")

    config = DatasetConfig(
        window_size=args.window_size,
        stride=args.stride,
        min_segment_steps=args.min_segment_steps,
        max_gap_ms=args.max_gap_ms,
        min_quality_score=args.min_quality_score,
        min_action_coverage=args.min_action_coverage,
        split_seed=args.split_seed,
    )
    episodes = load_episodes(paths)
    dataset = build_dataset(episodes, config, task=args.task)
    written = write_dataset(dataset, args.output)
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in paths],
                "output": written,
                "statistics": dataset["statistics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
