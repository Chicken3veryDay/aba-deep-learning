from __future__ import annotations

import argparse
import json
from pathlib import Path

from aba_deep_learning.dataset import discover_episode_files, load_episodes
from aba_deep_learning.episode_splits import (
    EpisodeSplitConfig,
    plan_episode_splits,
    validate_episode_split_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan deterministic episode-level ABA train, validation, and test "
            "splits without leaking duplicate match jobs"
        )
    )
    parser.add_argument("inputs", nargs="+", help="Episode JSONL files or directories")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", default="aba-episode-split-v1")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every schema has independent train, validation, and test groups",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for value in args.inputs:
        paths.extend(discover_episode_files(value))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No .jsonl episode files found")

    episodes = load_episodes(paths)
    config = EpisodeSplitConfig(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        require_complete_splits=args.require_complete,
    )
    plan = plan_episode_splits(episodes, config)
    validate_episode_split_plan(plan)
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
