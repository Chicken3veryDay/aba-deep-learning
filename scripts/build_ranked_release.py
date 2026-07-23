from __future__ import annotations

import argparse
import json
from pathlib import Path

from aba_deep_learning.imitation import recordings_to_arrays, save_episode_cache
from aba_deep_learning.ranked_release import assign_splits, build_release, inventory_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ranked recordings and build a leakage-safe v1 release")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-to", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument("--cache", type=Path, help="optional compressed NumPy training cache; raw JSONL is not copied")
    args = parser.parse_args()
    report = build_release(args.roots, args.output, relative_to=args.relative_to, split_seed=args.split_seed)
    if args.cache:
        recordings, _ = inventory_corpus(args.roots, relative_to=args.relative_to)
        assign_splits(recordings, seed=args.split_seed)
        save_episode_cache(recordings_to_arrays(recordings), args.cache)
        report["training_cache"] = str(args.cache)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
