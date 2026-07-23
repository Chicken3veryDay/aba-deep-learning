from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

ACTION_NAMES = ("sprint", "m1", "block", "dodge", "jump", "move_slot_1", "move_slot_2", "move_slot_3", "move_slot_4")

@dataclass(frozen=True)
class EpisodeArrays:
    episode_id: str
    job_id: str
    split: str
    features: np.ndarray
    movement: np.ndarray
    actions: np.ndarray
    timestamps_ms: np.ndarray
    label_mask: np.ndarray

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20260722
    sequence_length: int = 64
    stride: int = 16
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    patience: int = 6
    movement_loss_weight: float = 1.0
    action_loss_weight: float = 1.0


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class SequenceDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, episodes: Sequence[EpisodeArrays], *, sequence_length: int, stride: int, full_episodes: bool = False) -> None:
        if sequence_length < 1 or stride < 1:
            raise ValueError("sequence_length and stride must be positive")
        self.episodes = list(episodes)
        self.sequence_length = sequence_length
        self.index: list[tuple[int, int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            length = int(episode.features.shape[0])
            if not length:
                continue
            if full_episodes or length <= sequence_length:
                self.index.append((episode_index, 0, length))
                continue
            for start in range(0, length - sequence_length + 1, stride):
                self.index.append((episode_index, start, start + sequence_length))
            tail = (episode_index, length - sequence_length, length)
            if tail not in self.index:
                self.index.append(tail)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        episode_index, start, end = self.index[index]
        episode = self.episodes[episode_index]
        return {
            "features": torch.from_numpy(episode.features[start:end]).float(),
            "movement": torch.from_numpy(episode.movement[start:end]).float(),
            "actions": torch.from_numpy(episode.actions[start:end]).float(),
            "label_mask": torch.from_numpy(episode.label_mask[start:end]).float(),
            "timestamps_ms": torch.from_numpy(episode.timestamps_ms[start:end]).long(),
        }


def collate_sequences(batch: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    maximum = max(item["features"].shape[0] for item in batch)
    batch_size = len(batch)
    features = torch.zeros(batch_size, maximum, 72)
    movement = torch.zeros(batch_size, maximum, 2)
    actions = torch.zeros(batch_size, maximum, len(ACTION_NAMES))
    label_mask = torch.zeros(batch_size, maximum, 1)
    valid = torch.zeros(batch_size, maximum, 1)
    timestamps = torch.zeros(batch_size, maximum, dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["features"].shape[0]
        features[row, :length] = item["features"]
        movement[row, :length] = item["movement"]
        actions[row, :length] = item["actions"]
        label_mask[row, :length] = item["label_mask"].reshape(-1, 1)
        valid[row, :length] = 1
        timestamps[row, :length] = item["timestamps_ms"]
    return {"features": features, "movement": movement, "actions": actions, "label_mask": label_mask, "valid": valid, "timestamps_ms": timestamps}


class PolicyHeads(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.movement = nn.Linear(input_size, 2)
        self.actions = nn.Linear(input_size, len(ACTION_NAMES))

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        return {"movement": torch.tanh(self.movement(hidden)), "action_logits": self.actions(hidden)}


class FrameMLPPolicy(nn.Module):
    def __init__(self, feature_width: int = 72, hidden_size: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_width, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.heads = PolicyHeads(hidden_size)

    def forward(self, features: Tensor, hidden_state: Tensor | None = None) -> tuple[dict[str, Tensor], None]:
        return self.heads(self.encoder(features)), None


class GRUPolicy(nn.Module):
    def __init__(self, feature_width: int = 72, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_width)
        self.gru = nn.GRU(feature_width, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.heads = PolicyHeads(hidden_size)

    def forward(self, features: Tensor, hidden_state: Tensor | None = None) -> tuple[dict[str, Tensor], Tensor]:
        output, hidden = self.gru(self.input_norm(features), hidden_state)
        return self.heads(output), hidden


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def positive_weights(episodes: Sequence[EpisodeArrays]) -> Tensor:
    actions = np.concatenate([episode.actions for episode in episodes], axis=0)
    positives = actions.sum(axis=0)
    negatives = actions.shape[0] - positives
    return torch.tensor(np.clip(negatives / np.maximum(positives, 1), 1.0, 100.0), dtype=torch.float32)


def policy_loss(outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor], *, pos_weight: Tensor, movement_weight: float = 1.0, action_weight: float = 1.0) -> tuple[Tensor, dict[str, float]]:
    mask = batch["valid"] * batch["label_mask"]
    denominator = mask.sum().clamp_min(1.0)
    movement_loss = (torch.abs(outputs["movement"] - batch["movement"]) * mask).sum() / (denominator * 2)
    binary = nn.functional.binary_cross_entropy_with_logits(outputs["action_logits"], batch["actions"], reduction="none", pos_weight=pos_weight.to(outputs["action_logits"].device))
    action_loss = (binary * mask).sum() / (denominator * len(ACTION_NAMES))
    total = movement_weight * movement_loss + action_weight * action_loss
    return total, {"movement_l1": float(movement_loss.detach()), "action_bce": float(action_loss.detach()), "total": float(total.detach())}


@torch.no_grad()
def _validation_loss(model: nn.Module, loader: DataLoader, device: torch.device, pos_weight: Tensor, config: TrainConfig) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output, _ = model(batch["features"])
        loss, _ = policy_loss(output, batch, pos_weight=pos_weight, movement_weight=config.movement_loss_weight, action_weight=config.action_loss_weight)
        total += float(loss) * batch["features"].shape[0]
        count += batch["features"].shape[0]
    return total / max(count, 1)


def train_policy(model: nn.Module, train_episodes: Sequence[EpisodeArrays], validation_episodes: Sequence[EpisodeArrays], output_dir: str | Path, config: TrainConfig) -> dict[str, Any]:
    set_deterministic(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(SequenceDataset(train_episodes, sequence_length=config.sequence_length, stride=config.stride), batch_size=config.batch_size, shuffle=True, collate_fn=collate_sequences, generator=generator)
    validation_loader = DataLoader(SequenceDataset(validation_episodes, sequence_length=config.sequence_length, stride=config.sequence_length), batch_size=config.batch_size, shuffle=False, collate_fn=collate_sequences)
    pos_weight = positive_weights(train_episodes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = output / "best.pt"
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch["features"])
            loss, _ = policy_loss(prediction, batch, pos_weight=pos_weight, movement_weight=config.movement_loss_weight, action_weight=config.action_loss_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * batch["features"].shape[0]
            count += batch["features"].shape[0]
        train_loss = total / max(count, 1)
        validation_loss = _validation_loss(model, validation_loader, device, pos_weight, config)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best - 1e-6:
            best, best_epoch, stale = validation_loss, epoch + 1, 0
            torch.save({
                "model_state": model.state_dict(), "model_class": model.__class__.__name__,
                "model_config": {"feature_width": 72, "hidden_size": config.hidden_size, "num_layers": config.num_layers, "dropout": config.dropout},
                "train_config": asdict(config), "positive_weights": pos_weight.tolist(),
                "epoch": best_epoch, "validation_loss": best,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= config.patience:
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    report = {
        "model_class": model.__class__.__name__, "parameter_count": parameter_count(model), "device": str(device),
        "best_epoch": best_epoch, "best_validation_loss": best, "training_time_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "history": history, "config": asdict(config), "positive_weights": pos_weight.tolist(),
    }
    (output / "training-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_policy_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    if checkpoint["model_class"] == "FrameMLPPolicy":
        model: nn.Module = FrameMLPPolicy(config["feature_width"], config["hidden_size"])
    elif checkpoint["model_class"] == "GRUPolicy":
        model = GRUPolicy(**config)
    else:
        raise ValueError(f"unsupported model class: {checkpoint['model_class']}")
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def recordings_to_arrays(recordings: Sequence[Any]) -> list[EpisodeArrays]:
    result: list[EpisodeArrays] = []
    for recording in recordings:
        inv = recording.inventory
        if inv.admission_status != "admitted" or not inv.split_assignment or len(recording.steps) < 2:
            continue
        source = recording.steps
        labels = source[1:]
        result.append(EpisodeArrays(
            str(inv.episode_id), str(inv.roblox_job_id), str(inv.split_assignment),
            np.asarray([step.feature_vector for step in source[:-1]], dtype=np.float32),
            np.asarray([[step.move_x, step.move_z] for step in labels], dtype=np.float32),
            np.asarray([[float(step.actions[name]) for name in ACTION_NAMES] for step in labels], dtype=np.float32),
            np.asarray([step.timestamp_ms for step in source[:-1]], dtype=np.int64),
            np.asarray([float(step.label_available) for step in labels], dtype=np.float32),
        ))
    return result


def save_episode_cache(episodes: Sequence[EpisodeArrays], path: str | Path) -> None:
    payload: dict[str, Any] = {"metadata": np.asarray(json.dumps([{"episode_id": item.episode_id, "job_id": item.job_id, "split": item.split} for item in episodes]))}
    for index, item in enumerate(episodes):
        prefix = f"episode_{index}"
        payload.update({
            f"{prefix}_features": item.features, f"{prefix}_movement": item.movement,
            f"{prefix}_actions": item.actions, f"{prefix}_timestamps": item.timestamps_ms,
            f"{prefix}_label_mask": item.label_mask,
        })
    np.savez_compressed(path, **payload)


def load_episode_cache(path: str | Path) -> list[EpisodeArrays]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        result = []
        for index, item in enumerate(metadata):
            prefix = f"episode_{index}"
            result.append(EpisodeArrays(item["episode_id"], item["job_id"], item["split"], payload[f"{prefix}_features"], payload[f"{prefix}_movement"], payload[f"{prefix}_actions"], payload[f"{prefix}_timestamps"], payload[f"{prefix}_label_mask"]))
        return result
