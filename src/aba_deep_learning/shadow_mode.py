from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .imitation import ACTION_NAMES
from .ranked_schema import normalize_feature_vector

PREDICTION_ONLY = True
INPUT_EXECUTION_ENABLED = False
REMOTE_DISPATCH_ENABLED = False
CONTROLLER_ENABLED = False

class ShadowModeSafetyError(RuntimeError):
    pass

@dataclass(frozen=True)
class ShadowConfig:
    model_version: str
    dataset_manifest_sha256: str
    thresholds: dict[str, float]
    enabled: bool = False
    prediction_only: bool = True

    def validate(self) -> None:
        if not self.enabled:
            raise ShadowModeSafetyError("shadow mode is disabled by default; explicit --enable-shadow is required")
        if not self.prediction_only:
            raise ShadowModeSafetyError("shadow mode must remain prediction-only")
        if INPUT_EXECUTION_ENABLED or REMOTE_DISPATCH_ENABLED or CONTROLLER_ENABLED:
            raise ShadowModeSafetyError("a gameplay-control gate is enabled")
        if set(self.thresholds) != set(ACTION_NAMES):
            raise ShadowModeSafetyError("thresholds must cover exactly the nine action heads")

class PredictionOnlyShadowRunner:
    """Consumes live watcher observations and writes predictions. It exposes no input or remote API."""

    def __init__(self, model: nn.Module, config: ShadowConfig, output: str | Path, *, device: str | torch.device = "cpu") -> None:
        config.validate()
        self.model = model.to(device).eval()
        self.config = config
        self.output = Path(output)
        self.device = torch.device(device)
        self.hidden_state: torch.Tensor | None = None
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True
        self.hidden_state = None

    @torch.no_grad()
    def predict(self, observation_record: Mapping[str, Any]) -> dict[str, Any]:
        if self.stopped:
            raise ShadowModeSafetyError("shadow runner is stopped")
        observation = observation_record.get("observation", observation_record)
        if not isinstance(observation, Mapping):
            raise ValueError("observation record must be an object")
        vector, encoding = normalize_feature_vector(observation.get("feature_vector"))
        features = torch.tensor(vector, dtype=torch.float32, device=self.device).reshape(1, 1, 72)
        output, self.hidden_state = self.model(features, self.hidden_state)
        movement = output["movement"][0, -1].cpu().tolist()
        probability = torch.sigmoid(output["action_logits"][0, -1]).cpu().tolist()
        return {
            "record_type": "shadow_prediction",
            "prediction_only": True,
            "input_sent": False,
            "remote_fired": False,
            "character_state_changed": False,
            "controller_enabled": False,
            "timestamp_ms": int(time.time() * 1000),
            "source_timestamp_ms": observation.get("timestamp_ms"),
            "source_step_index": observation.get("step_index"),
            "feature_encoding": encoding,
            "model_version": self.config.model_version,
            "dataset_manifest_sha256": self.config.dataset_manifest_sha256,
            "movement": {"left_right": movement[0], "forward_backward": movement[1]},
            "action_probabilities": dict(zip(ACTION_NAMES, probability, strict=True)),
            "action_predictions": {name: bool(probability[index] >= self.config.thresholds[name]) for index, name in enumerate(ACTION_NAMES)},
        }

    def run(self, records: Iterable[Mapping[str, Any]]) -> int:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.output.open("a", encoding="utf-8") as stream:
            for record in records:
                if self.stopped:
                    break
                stream.write(json.dumps(self.predict(record), sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                count += 1
        return count


def read_jsonl(path: str | Path, *, follow: bool = False, poll_seconds: float = 0.1) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value.get("step", value) if value.get("record_type") == "step" else value
            elif not follow:
                break
            else:
                time.sleep(poll_seconds)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
