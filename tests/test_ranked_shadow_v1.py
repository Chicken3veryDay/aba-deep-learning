from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aba_deep_learning.imitation import ACTION_NAMES, FrameMLPPolicy
from aba_deep_learning.shadow_mode import (
    CONTROLLER_ENABLED, INPUT_EXECUTION_ENABLED, PREDICTION_ONLY,
    REMOTE_DISPATCH_ENABLED, PredictionOnlyShadowRunner, ShadowConfig,
    ShadowModeSafetyError,
)


class RankedShadowV1Tests(unittest.TestCase):
    def thresholds(self) -> dict[str, float]:
        return {name: 0.5 for name in ACTION_NAMES}

    def test_hard_gates_are_prediction_only(self) -> None:
        self.assertTrue(PREDICTION_ONLY)
        self.assertFalse(INPUT_EXECUTION_ENABLED)
        self.assertFalse(REMOTE_DISPATCH_ENABLED)
        self.assertFalse(CONTROLLER_ENABLED)

    def test_shadow_mode_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ShadowModeSafetyError):
                PredictionOnlyShadowRunner(FrameMLPPolicy(hidden_size=8), ShadowConfig("m", "h", self.thresholds()), Path(directory) / "out.jsonl")

    def test_prediction_record_asserts_no_input_or_remote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = PredictionOnlyShadowRunner(
                FrameMLPPolicy(hidden_size=8),
                ShadowConfig("m", "h", self.thresholds(), enabled=True),
                Path(directory) / "out.jsonl",
            )
            result = runner.predict({"feature_vector": [0.0] * 72, "timestamp_ms": 1, "step_index": 0})
            self.assertTrue(result["prediction_only"])
            self.assertFalse(result["input_sent"])
            self.assertFalse(result["remote_fired"])
            self.assertFalse(result["character_state_changed"])
            self.assertFalse(result["controller_enabled"])

if __name__ == "__main__":
    unittest.main()
