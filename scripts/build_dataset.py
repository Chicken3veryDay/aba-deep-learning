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
    build_partitioned_datasets,
    discover_episode_files,
    write_dataset,
    write_partitioned_datasets,
)
from aba_deep_learning.feature_schemas import list_feature_schemas  # noqa: E402
from aba_deep_learning.stream import read_episode_stream  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic, schema-safe ABA learning datasets"
    )
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
    parser.add_argument(
        "--feature-schema",
        choices=tuple(schema.schema_id for schema in list_feature_schemas()),
        help="Require one canonical feature schema",
    )
    parser.add_argument(
        "--partition-by-schema",
        action="store_true",
        help="Write one isolated dataset per feature schema",
    )
    parser.add_argument(
        "--allow-missing-action-masks",
        action="store_true",
        help="Do not reject passive-human streams solely for missing action masks",
    )
    parser.add_argument(
        "--no-legacy-64-inference",
        action="store_true",
        help="Require an explicit feature schema even for 64-feature streams",
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

    config = DatasetConfig(
        window_size=args.window_size,
        stride=args.stride,
        min_segment_steps=args.min_segment_steps,
        max_gap_ms=args.max_gap_ms,
        min_quality_score=args.min_quality_score,
        min_action_coverage=args.min_action_coverage,
        split_seed=args.split_seed,
        feature_schema_id=args.feature_schema,
        allow_legacy_64_inference=not args.no_legacy_64_inference,
        require_action_masks=not args.allow_missing_action_masks,
    )
    episodes = [
        read_episode_stream(
            path,
            require_action_masks=config.require_action_masks,
        )
        for path in paths
    ]

    if args.partition_by_schema:
        datasets = build_partitioned_datasets(
            episodes,
            config,
            task=args.task,
        )
        written = write_partitioned_datasets(datasets, args.output)
        statistics = {
            schema_id: dataset["statistics"]
            for schema_id, dataset in datasets.items()
        }
    else:
        dataset = build_dataset(episodes, config, task=args.task)
        written = write_dataset(dataset, args.output)
        statistics = dataset["statistics"]

    print(
        json.dumps(
            {
                "inputs": [str(path) for path in paths],
                "output": written,
                "statistics": statistics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
