from __future__ import annotations

import json
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .imitation import ACTION_NAMES, EpisodeArrays


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    target, predicted = target.astype(bool), predicted.astype(bool)
    tp = int(np.logical_and(target, predicted).sum())
    tn = int(np.logical_and(~target, ~predicted).sum())
    fp = int(np.logical_and(~target, predicted).sum())
    fn = int(np.logical_and(target, ~predicted).sum())
    precision, recall = _divide(tp, tp + fp), _divide(tp, tp + fn)
    return {
        "precision": precision, "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
        "support": int(target.sum()),
        "false_positive_rate": _divide(fp, fp + tn),
        "false_negative_rate": _divide(fn, fn + tp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def tune_thresholds(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for index, name in enumerate(ACTION_NAMES):
        best_threshold, best_score = 0.5, -1.0
        for threshold in np.linspace(0.05, 0.95, 37):
            score = binary_metrics(targets[:, index], probabilities[:, index] >= threshold)["f1"]
            if score > best_score + 1e-12:
                best_threshold, best_score = float(threshold), score
        thresholds[name] = best_threshold
    return thresholds


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = values.astype(bool)
    starts = np.flatnonzero(np.logical_and(values, np.r_[True, ~values[:-1]]))
    ends = np.flatnonzero(np.logical_and(values, np.r_[~values[1:], True])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _temporal(target: np.ndarray, predicted: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    target_runs, predicted_runs = _runs(target), _runs(predicted)
    onset: list[float] = []
    duration: list[float] = []
    for start, end in target_runs:
        if not predicted_runs:
            continue
        nearest = min(predicted_runs, key=lambda run: abs(run[0] - start))
        onset.append(abs(float(timestamps[nearest[0]] - timestamps[start])))
        target_duration = float(timestamps[min(end - 1, len(timestamps) - 1)] - timestamps[start])
        predicted_duration = float(timestamps[min(nearest[1] - 1, len(timestamps) - 1)] - timestamps[nearest[0]])
        duration.append(abs(predicted_duration - target_duration))
    transitions = int(np.not_equal(predicted[1:], predicted[:-1]).sum()) if len(predicted) > 1 else 0
    return {
        "action_onset_timing_error_ms": float(np.mean(onset)) if onset else 0.0,
        "action_duration_error_ms": float(np.mean(duration)) if duration else 0.0,
        "prediction_flicker_per_active_frame": transitions / max(int(predicted.sum()), 1),
        "repeated_action_instability": _divide(transitions, max(len(predicted) - 1, 1)),
    }


@torch.no_grad()
def predict_episodes(model: nn.Module, episodes: Sequence[EpisodeArrays], *, device: str | torch.device = "cpu") -> list[dict[str, Any]]:
    model = model.to(device).eval()
    result: list[dict[str, Any]] = []
    for episode in episodes:
        features = torch.from_numpy(episode.features).float().unsqueeze(0).to(device)
        output, hidden = model(features)
        result.append({
            "episode_id": episode.episode_id, "job_id": episode.job_id,
            "features": episode.features, "movement_target": episode.movement,
            "action_target": episode.actions, "timestamps_ms": episode.timestamps_ms,
            "movement_prediction": output["movement"].squeeze(0).cpu().numpy(),
            "action_probability": torch.sigmoid(output["action_logits"]).squeeze(0).cpu().numpy(),
            "hidden_norm": float(hidden.norm().cpu()) if hidden is not None else 0.0,
        })
    return result


def _movement(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    target_norm = np.linalg.norm(target, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    cosine = np.sum(target * predicted, axis=1) / np.maximum(target_norm * predicted_norm, 1e-8)
    return {
        "mean_absolute_error": float(np.abs(target - predicted).mean()),
        "directional_accuracy": float(np.all(np.sign(target) == np.sign(predicted), axis=1).mean()),
        "movement_no_movement_accuracy": float(((target_norm >= 0.25) == (predicted_norm >= 0.25)).mean()),
        "cosine_similarity": float(cosine.mean()),
    }


def _contexts(features: np.ndarray, movement_target: np.ndarray, movement_prediction: np.ndarray, action_target: np.ndarray, action_prediction: np.ndarray) -> dict[str, Any]:
    masks = {
        "distance_close": features[:, 2] <= 0.03,
        "distance_mid": np.logical_and(features[:, 2] > 0.03, features[:, 2] <= 0.10),
        "distance_far": features[:, 2] > 0.10,
        "self_health_low": features[:, 0] < 0.33,
        "self_health_mid": np.logical_and(features[:, 0] >= 0.33, features[:, 0] < 0.67),
        "self_health_high": features[:, 0] >= 0.67,
        "opponent_health_low": features[:, 1] < 0.33,
        "opponent_health_high": features[:, 1] >= 0.67,
        "self_grounded": features[:, 20] >= 0.5,
        "self_airborne": features[:, 22] >= 0.5,
        "facing_target": features[:, 12] >= 0.5,
        "self_blocking": features[:, 24] >= 0.5,
        "self_hitstun": features[:, 26] >= 0.5,
        "damage_received_window": features[:, 50] >= 0.5,
        "damage_dealt_window": features[:, 51] >= 0.5,
    }
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        support = int(mask.sum())
        result[name] = {
            "support": support,
            "movement_mae": float(np.abs(movement_target[mask] - movement_prediction[mask]).mean()) if support else None,
            "micro_action_f1": binary_metrics(action_target[mask].reshape(-1), action_prediction[mask].reshape(-1))["f1"] if support else None,
        }
    result["cooldown_availability"] = {"available": False, "reason": "the verified 72-feature contract contains no cooldown-availability feature"}
    return result


def evaluate_predictions(predictions: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    movement_target = np.concatenate([item["movement_target"] for item in predictions])
    movement_prediction = np.concatenate([item["movement_prediction"] for item in predictions])
    action_target = np.concatenate([item["action_target"] for item in predictions])
    probabilities = np.concatenate([item["action_probability"] for item in predictions])
    features = np.concatenate([item["features"] for item in predictions])
    action_prediction = np.column_stack([probabilities[:, index] >= thresholds[name] for index, name in enumerate(ACTION_NAMES)])
    action_metrics: dict[str, Any] = {}
    temporal: dict[str, list[dict[str, float]]] = defaultdict(list)
    for index, name in enumerate(ACTION_NAMES):
        action_metrics[name] = {**binary_metrics(action_target[:, index], action_prediction[:, index]), "calibrated_threshold": float(thresholds[name])}
        for item in predictions:
            temporal[name].append(_temporal(item["action_target"][:, index], item["action_probability"][:, index] >= thresholds[name], item["timestamps_ms"]))
    temporal_summary = {name: {key: float(np.mean([episode[key] for episode in episodes])) for key in episodes[0]} for name, episodes in temporal.items() if episodes}
    slot_count = action_prediction[:, 5:9].sum(axis=1)
    invalid_attack_state = np.logical_or(features[:, 26] >= 0.5, features[:, 22] >= 0.5)
    any_attack = np.logical_or(action_prediction[:, 1], action_prediction[:, 5:9].any(axis=1))
    first, later = [], []
    for item in predictions:
        count = min(5, len(item["movement_target"]))
        first.extend(np.abs(item["movement_target"][:count] - item["movement_prediction"][:count]).reshape(-1))
        later.extend(np.abs(item["movement_target"][count:] - item["movement_prediction"][count:]).reshape(-1))
    return {
        "movement": _movement(movement_target, movement_prediction),
        "actions": action_metrics,
        "temporal": temporal_summary,
        "sequence": {
            "hidden_state_norm_by_episode": {item["episode_id"]: item["hidden_norm"] for item in predictions},
            "boundary_movement_mae_first_5": float(np.mean(first)) if first else 0.0,
            "steady_state_movement_mae": float(np.mean(later)) if later else 0.0,
        },
        "combat_context": _contexts(features, movement_target, movement_prediction, action_target, action_prediction),
        "safety_legality": {
            "impossible_simultaneous_move_slots": int((slot_count > 1).sum()),
            "attack_predictions_during_invalid_state": int(np.logical_and(invalid_attack_state, any_attack).sum()),
            "excessive_action_rate_predictions": int((action_prediction.sum(axis=1) > 3).sum()),
            "false_m1_rate": binary_metrics(action_target[:, 1], action_prediction[:, 1])["false_positive_rate"],
            "false_dodge_rate": binary_metrics(action_target[:, 3], action_prediction[:, 3])["false_positive_rate"],
            "false_move_slot_rate": binary_metrics(action_target[:, 5:9].reshape(-1), action_prediction[:, 5:9].reshape(-1))["false_positive_rate"],
            "cooldown_violation_count": None,
            "cooldown_violation_reason": "cooldown availability is not encoded in ranked_explicit_v3_72",
        },
    }


@torch.no_grad()
def runtime_metrics(model: nn.Module, *, iterations: int = 500, device: str | torch.device = "cpu") -> dict[str, Any]:
    model = model.to(device).eval()
    sample = torch.zeros(1, 1, 72, device=device)
    for _ in range(20):
        model(sample)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    tracemalloc.start()
    started = time.perf_counter()
    for _ in range(iterations):
        model(sample)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "mean_inference_latency_ms": elapsed * 1000 / iterations,
        "python_peak_allocated_bytes": peak,
        "polling_loop_10hz_compatible": elapsed / iterations < 0.1,
        "polling_loop_15hz_compatible": elapsed / iterations < (1 / 15),
    }


def evaluate_model(model: nn.Module, validation_episodes: Sequence[EpisodeArrays], test_episodes: Sequence[EpisodeArrays], output: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    validation_predictions = predict_episodes(model, validation_episodes, device=device)
    thresholds = tune_thresholds(
        np.concatenate([item["action_probability"] for item in validation_predictions]),
        np.concatenate([item["action_target"] for item in validation_predictions]),
    )
    report = {
        "thresholds_selected_on": "validation_only",
        "thresholds": thresholds,
        "validation": evaluate_predictions(validation_predictions, thresholds),
        "untouched_test": evaluate_predictions(predict_episodes(model, test_episodes, device=device), thresholds),
        "runtime": runtime_metrics(model, device=device),
    }
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
