from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aba_deep_learning.evaluation import evaluate_model, tune_thresholds
from aba_deep_learning.imitation import (
    ACTION_NAMES, EpisodeArrays, FrameMLPPolicy, GRUPolicy, SequenceDataset,
    TrainConfig, load_policy_checkpoint, parameter_count, train_policy,
)


def episodes() -> list[EpisodeArrays]:
    rng = np.random.default_rng(4)
    result = []
    for index, split in enumerate(("train", "train", "validation", "test")):
        length = 48
        features = rng.normal(0, 0.2, (length, 72)).clip(-1, 1).astype(np.float32)
        movement = np.sign(features[:, :2]).astype(np.float32)
        actions = np.zeros((length, len(ACTION_NAMES)), dtype=np.float32)
        actions[:, 0] = features[:, 2] > 0
        actions[:, 1] = features[:, 3] > 0.25
        result.append(EpisodeArrays(f"e{index}", f"j{index}", split, features, movement, actions, np.arange(length, dtype=np.int64) * 67, np.ones(length, dtype=np.float32)))
    return result


class RankedImitationV1Tests(unittest.TestCase):
    def test_sequence_windows_never_cross_episode_boundaries(self) -> None:
        data = episodes()[:2]
        dataset = SequenceDataset(data, sequence_length=16, stride=8)
        for episode_index, start, end in dataset.index:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(data[episode_index].features))

    def test_baseline_and_gru_forward_shapes(self) -> None:
        value = torch.zeros(2, 8, 72)
        for model in (FrameMLPPolicy(hidden_size=32), GRUPolicy(hidden_size=32)):
            outputs, _ = model(value)
            self.assertEqual(outputs["movement"].shape, (2, 8, 2))
            self.assertEqual(outputs["action_logits"].shape, (2, 8, 9))
            self.assertGreater(parameter_count(model), 0)

    def test_checkpoint_loading_and_evaluation_reproducibility(self) -> None:
        data = episodes()
        with tempfile.TemporaryDirectory() as directory:
            config = TrainConfig(seed=9, hidden_size=16, sequence_length=16, stride=16, batch_size=4, epochs=2, patience=2)
            train_policy(FrameMLPPolicy(hidden_size=16), data[:2], [data[2]], directory, config)
            loaded = load_policy_checkpoint(Path(directory) / "best.pt")
            first = evaluate_model(loaded, [data[2]], [data[3]], Path(directory) / "eval1.json")
            second = evaluate_model(loaded, [data[2]], [data[3]], Path(directory) / "eval2.json")
            self.assertEqual(first["thresholds"], second["thresholds"])
            self.assertEqual(first["untouched_test"]["actions"], second["untouched_test"]["actions"])

    def test_threshold_tuning_covers_every_action_head(self) -> None:
        probability = np.tile(np.linspace(0, 1, 20)[:, None], (1, 9))
        thresholds = tune_thresholds(probability, probability > 0.7)
        self.assertEqual(set(thresholds), set(ACTION_NAMES))

if __name__ == "__main__":
    unittest.main()
