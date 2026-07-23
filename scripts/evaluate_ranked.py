from __future__ import annotations

import argparse
import json
from pathlib import Path

from aba_deep_learning.evaluation import evaluate_model
from aba_deep_learning.imitation import load_episode_cache, load_policy_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune thresholds on validation and evaluate once on untouched test episodes")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    episodes = load_episode_cache(args.cache)
    validation = [episode for episode in episodes if episode.split == "validation"]
    test = [episode for episode in episodes if episode.split == "test"]
    model = load_policy_checkpoint(args.checkpoint, device=args.device)
    report = evaluate_model(model, validation, test, args.output, device=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
