from __future__ import annotations

import argparse
import json
from pathlib import Path

from aba_deep_learning.imitation import FrameMLPPolicy, GRUPolicy, TrainConfig, load_episode_cache, train_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the ranked imitation baseline or GRU")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("baseline", "gru"), required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    episodes = load_episode_cache(args.cache)
    train = [episode for episode in episodes if episode.split == "train"]
    validation = [episode for episode in episodes if episode.split == "validation"]
    config = TrainConfig(seed=args.seed, hidden_size=args.hidden_size, sequence_length=args.sequence_length, stride=args.stride, batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate)
    model = FrameMLPPolicy(hidden_size=args.hidden_size) if args.model == "baseline" else GRUPolicy(hidden_size=args.hidden_size)
    report = train_policy(model, train, validation, args.output, config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
