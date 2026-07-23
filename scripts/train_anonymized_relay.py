from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import lz4.block
import numpy as np
import torch

from aba_deep_learning.evaluation import evaluate_predictions, predict_episodes, runtime_metrics, tune_thresholds
from aba_deep_learning.imitation import (
    ACTION_NAMES,
    EpisodeArrays,
    FrameMLPPolicy,
    GRUPolicy,
    TrainConfig,
    load_policy_checkpoint,
    train_policy,
)

MAGIC = b"ABAAN16\0"
ROW_BYTES = 154
SPLITS = {0: "train", 1: "validation", 2: "test"}


def decode_cache(path: Path) -> list[EpisodeArrays]:
    payload = path.read_bytes()
    if len(payload) < 4:
        raise ValueError("relay cache is truncated")
    raw_size = struct.unpack_from("<I", payload, 0)[0]
    raw = lz4.block.decompress(payload[4:], uncompressed_size=raw_size)
    if len(raw) != raw_size:
        raise ValueError(f"relay cache size mismatch: {len(raw)} != {raw_size}")
    if raw[:8] != MAGIC:
        raise ValueError("relay cache magic mismatch")
    offset = 8
    episode_count = struct.unpack_from("<H", raw, offset)[0]
    offset += 2
    episodes: list[EpisodeArrays] = []
    for episode_index in range(episode_count):
        split_code, steps = struct.unpack_from("<BI", raw, offset)
        offset += 5
        if split_code not in SPLITS:
            raise ValueError(f"invalid split code {split_code}")
        expected = steps * ROW_BYTES
        if offset + expected > len(raw):
            raise ValueError(f"episode {episode_index} is truncated")
        features = np.empty((steps, 72), dtype=np.float32)
        movement = np.empty((steps, 2), dtype=np.float32)
        actions = np.empty((steps, len(ACTION_NAMES)), dtype=np.float32)
        timestamps = np.empty(steps, dtype=np.int64)
        label_mask = np.empty(steps, dtype=np.float32)
        elapsed = 0
        for row in range(steps):
            delta_ms = struct.unpack_from("<H", raw, offset)[0]
            offset += 2
            elapsed += int(delta_ms)
            timestamps[row] = elapsed
            quantized_features = np.frombuffer(raw, dtype="<i2", count=72, offset=offset)
            features[row] = quantized_features.astype(np.float32) / 32767.0
            offset += 144
            move_x, move_z, action_mask, available_mask = struct.unpack_from("<hhHH", raw, offset)
            offset += 8
            movement[row] = (move_x / 32767.0, move_z / 32767.0)
            actions[row] = [float(bool(action_mask & (1 << bit))) for bit in range(len(ACTION_NAMES))]
            label_mask[row] = float(available_mask != 0)
        episodes.append(
            EpisodeArrays(
                episode_id=f"anonymous_episode_{episode_index:02d}",
                job_id=f"anonymous_job_{episode_index:02d}",
                split=SPLITS[split_code],
                features=features,
                movement=movement,
                actions=actions,
                timestamps_ms=timestamps,
                label_mask=label_mask,
            )
        )
    if offset != len(raw):
        raise ValueError(f"relay cache has {len(raw) - offset} unexpected trailing bytes")
    return episodes


def validation_report(model: torch.nn.Module, episodes: list[EpisodeArrays]) -> tuple[dict, dict[str, float], float]:
    predictions = predict_episodes(model, episodes, device="cpu")
    probabilities = np.concatenate([item["action_probability"] for item in predictions])
    targets = np.concatenate([item["action_target"] for item in predictions])
    thresholds = tune_thresholds(probabilities, targets)
    metrics = evaluate_predictions(predictions, thresholds)
    mean_action_f1 = float(np.mean([metrics["actions"][name]["f1"] for name in ACTION_NAMES]))
    score = mean_action_f1 - metrics["movement"]["mean_absolute_error"]
    return metrics, thresholds, score


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the first baseline and GRU from an anonymized int16 relay cache")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = decode_cache(args.cache)
    train = [episode for episode in episodes if episode.split == "train"]
    validation = [episode for episode in episodes if episode.split == "validation"]
    test = [episode for episode in episodes if episode.split == "test"]
    if (len(train), len(validation), len(test)) != (7, 2, 2):
        raise ValueError(f"unexpected split sizes: {(len(train), len(validation), len(test))}")
    if sum(len(episode.features) for episode in episodes) != 26465:
        raise ValueError("unexpected aligned step count")

    specifications = {
        "baseline": (
            FrameMLPPolicy(hidden_size=64),
            TrainConfig(seed=args.seed, hidden_size=64, sequence_length=64, stride=32, batch_size=64, epochs=15, patience=4, learning_rate=1e-3),
        ),
        "gru": (
            GRUPolicy(hidden_size=64),
            TrainConfig(seed=args.seed, hidden_size=64, sequence_length=64, stride=32, batch_size=64, epochs=20, patience=5, learning_rate=1e-3),
        ),
    }
    candidates: dict[str, dict] = {}
    loaded: dict[str, torch.nn.Module] = {}
    for name, (model, config) in specifications.items():
        model_dir = args.output / name
        training = train_policy(model, train, validation, model_dir, config)
        selected = load_policy_checkpoint(model_dir / "best.pt", device="cpu")
        loaded[name] = selected
        validation_metrics, thresholds, selection_score = validation_report(selected, validation)
        candidates[name] = {
            "training": training,
            "validation": validation_metrics,
            "validation_thresholds": thresholds,
            "selection_score": selection_score,
        }

    selected_name = max(candidates, key=lambda name: candidates[name]["selection_score"])
    selected_model = loaded[selected_name]
    selected_thresholds = candidates[selected_name]["validation_thresholds"]
    untouched_predictions = predict_episodes(selected_model, test, device="cpu")
    untouched_test = evaluate_predictions(untouched_predictions, selected_thresholds)
    runtime = runtime_metrics(selected_model, device="cpu")
    checkpoint = args.output / selected_name / "best.pt"
    summary = {
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "transport": {
            "format": "anonymous signed int16, scale 32767, LZ4 block",
            "cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
            "contains_identifiers": False,
            "feature_width": 72,
            "aligned_steps": 26465,
            "split_episode_counts": {"train": 7, "validation": 2, "test": 2},
            "maximum_absolute_quantization_error": 1.0 / 32767.0,
        },
        "models": candidates,
        "selected_model": selected_name,
        "selection_rule": "highest validation mean per-action F1 minus validation movement MAE",
        "untouched_test_evaluations": 1,
        "untouched_test": untouched_test,
        "runtime": runtime,
        "selected_checkpoint": str(checkpoint.relative_to(args.output)),
        "selected_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "shadow_mode_ready": False,
        "shadow_mode_reason": "offline evidence only; prediction-only shadow integration requires explicit review and remains disabled by default",
    }
    (args.output / "model-comparison-and-test.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
