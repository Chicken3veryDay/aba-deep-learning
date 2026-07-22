from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .feature_schemas import FeatureSchemaError, resolve_episode_schema

SPLIT_PLAN_VERSION = "1.0.0"
SPLIT_NAMES = ("train", "validation", "test")


class EpisodeSplitError(ValueError):
    pass


@dataclass(frozen=True)
class EpisodeSplitConfig:
    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: str = "aba-episode-split-v1"
    require_complete_splits: bool = False

    def validate(self) -> None:
        ratios = self.train_ratio + self.validation_ratio + self.test_ratio
        if not math.isclose(ratios, 1.0, rel_tol=0, abs_tol=1e-9):
            raise EpisodeSplitError("split ratios must sum to 1")
        if min(self.train_ratio, self.validation_ratio, self.test_ratio) < 0:
            raise EpisodeSplitError("split ratios cannot be negative")
        if not self.seed:
            raise EpisodeSplitError("split seed cannot be empty")


def _episode_id(episode: Mapping[str, Any]) -> str:
    header = episode.get("header", {})
    value = header.get("episode_id") if isinstance(header, Mapping) else None
    if not isinstance(value, str) or not value:
        raise EpisodeSplitError("episode header is missing episode_id")
    return value


def _independence_group(episode: Mapping[str, Any]) -> str:
    header = episode.get("header", {})
    if not isinstance(header, Mapping):
        return _episode_id(episode)
    for key in (
        "independence_group_id",
        "source_match_id",
        "match_id",
        "job_id",
    ):
        value = header.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return f"episode_id:{_episode_id(episode)}"


def _steps(episode: Mapping[str, Any]) -> int:
    value = episode.get("steps", [])
    return (
        len(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else 0
    )


def _duration_ms(episode: Mapping[str, Any]) -> int:
    terminal = episode.get("terminal", {})
    if not isinstance(terminal, Mapping):
        return 0
    value = terminal.get("duration_ms", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _stable_order_key(seed: str, schema_id: str, group_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:{schema_id}:{group_id}".encode("utf-8")
    ).hexdigest()


def _allocate_counts(
    group_count: int,
    config: EpisodeSplitConfig,
) -> dict[str, int]:
    if group_count <= 0:
        return {name: 0 for name in SPLIT_NAMES}
    if group_count == 1:
        return {"train": 1, "validation": 0, "test": 0}
    if group_count == 2:
        return {"train": 1, "validation": 1, "test": 0}

    counts = {"train": 1, "validation": 1, "test": 1}
    remaining = group_count - 3
    if remaining == 0:
        return counts

    ratios = {
        "train": config.train_ratio,
        "validation": config.validation_ratio,
        "test": config.test_ratio,
    }
    raw = {
        name: remaining * ratio
        for name, ratio in ratios.items()
    }
    floors = {
        name: int(math.floor(value))
        for name, value in raw.items()
    }
    for name, value in floors.items():
        counts[name] += value

    left = remaining - sum(floors.values())
    ranked = sorted(
        SPLIT_NAMES,
        key=lambda name: (
            raw[name] - floors[name],
            ratios[name],
            name == "train",
        ),
        reverse=True,
    )
    for index in range(left):
        counts[ranked[index % len(ranked)]] += 1
    return counts


def _schema_plan(
    schema_id: str,
    episodes: Sequence[Mapping[str, Any]],
    config: EpisodeSplitConfig,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        groups.setdefault(
            _independence_group(episode),
            [],
        ).append(episode)

    ordered_groups = sorted(
        groups,
        key=lambda group_id: _stable_order_key(
            config.seed,
            schema_id,
            group_id,
        ),
    )
    counts = _allocate_counts(len(ordered_groups), config)
    assignments: dict[str, str] = {}
    group_assignments: dict[str, str] = {}
    cursor = 0
    for split_name in SPLIT_NAMES:
        for group_id in ordered_groups[
            cursor:cursor + counts[split_name]
        ]:
            group_assignments[group_id] = split_name
            for episode in groups[group_id]:
                assignments[_episode_id(episode)] = split_name
        cursor += counts[split_name]

    warnings: list[str] = []
    duplicate_groups = {
        group_id: [_episode_id(episode) for episode in values]
        for group_id, values in groups.items()
        if len(values) > 1
    }
    if duplicate_groups:
        warnings.append("non_independent_episode_groups_collapsed")
    missing = [name for name in SPLIT_NAMES if counts[name] == 0]
    if missing:
        warnings.append("missing_splits:" + ",".join(missing))

    split_steps = {name: 0 for name in SPLIT_NAMES}
    split_duration_ms = {name: 0 for name in SPLIT_NAMES}
    split_episode_counts = {name: 0 for name in SPLIT_NAMES}
    for episode in episodes:
        destination = assignments[_episode_id(episode)]
        split_steps[destination] += _steps(episode)
        split_duration_ms[destination] += _duration_ms(episode)
        split_episode_counts[destination] += 1

    complete = all(counts[name] > 0 for name in SPLIT_NAMES)
    if config.require_complete_splits and not complete:
        raise EpisodeSplitError(
            f"schema {schema_id} has {len(groups)} independent groups; "
            "at least three are required for train, validation, and test"
        )

    return {
        "schema_id": schema_id,
        "complete": complete,
        "episodes": len(episodes),
        "independent_groups": len(groups),
        "group_counts_by_split": counts,
        "episode_counts_by_split": split_episode_counts,
        "steps_by_split": split_steps,
        "duration_ms_by_split": split_duration_ms,
        "assignments": assignments,
        "group_assignments": group_assignments,
        "duplicate_groups": duplicate_groups,
        "warnings": warnings,
    }


def plan_episode_splits(
    episodes: Iterable[Mapping[str, Any]],
    config: EpisodeSplitConfig | None = None,
) -> dict[str, Any]:
    config = config or EpisodeSplitConfig()
    config.validate()
    episode_list = list(episodes)
    partitions: dict[str, list[Mapping[str, Any]]] = {}
    unresolved: list[dict[str, str]] = []

    for episode in episode_list:
        try:
            schema = resolve_episode_schema(episode)
            partitions.setdefault(schema.schema_id, []).append(episode)
        except FeatureSchemaError as exc:
            try:
                episode_id = _episode_id(episode)
            except EpisodeSplitError:
                episode_id = ""
            unresolved.append({
                "episode_id": episode_id,
                "reason": str(exc),
            })

    plans = {
        schema_id: _schema_plan(schema_id, values, config)
        for schema_id, values in sorted(partitions.items())
    }
    complete = (
        bool(plans)
        and all(plan["complete"] for plan in plans.values())
        and not unresolved
    )

    if config.require_complete_splits and unresolved:
        raise EpisodeSplitError(
            "unresolved episode schemas prevent complete split planning"
        )

    return {
        "split_plan_version": SPLIT_PLAN_VERSION,
        "config": asdict(config),
        "complete": complete,
        "schema_partitions": plans,
        "unresolved": unresolved,
        "executor_integration_allowed": False,
    }


def validate_episode_split_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("split_plan_version") != SPLIT_PLAN_VERSION:
        raise EpisodeSplitError("unsupported split plan version")
    partitions = plan.get("schema_partitions")
    if not isinstance(partitions, Mapping):
        raise EpisodeSplitError("split plan is missing schema partitions")

    all_episode_ids: set[str] = set()
    for schema_id, raw_partition in partitions.items():
        if not isinstance(raw_partition, Mapping):
            raise EpisodeSplitError(
                f"schema partition {schema_id} must be an object"
            )
        assignments = raw_partition.get("assignments")
        groups = raw_partition.get("group_assignments")
        if not isinstance(assignments, Mapping) or not isinstance(groups, Mapping):
            raise EpisodeSplitError(
                f"schema partition {schema_id} is missing assignments"
            )
        for episode_id, split_name in assignments.items():
            if split_name not in SPLIT_NAMES:
                raise EpisodeSplitError(
                    f"episode {episode_id} has invalid split {split_name!r}"
                )
            if episode_id in all_episode_ids:
                raise EpisodeSplitError(
                    "episode appears in multiple schema partitions: "
                    f"{episode_id}"
                )
            all_episode_ids.add(str(episode_id))
        for group_id, split_name in groups.items():
            if split_name not in SPLIT_NAMES:
                raise EpisodeSplitError(
                    f"group {group_id} has invalid split {split_name!r}"
                )
        duplicate_groups = raw_partition.get("duplicate_groups", {})
        if isinstance(duplicate_groups, Mapping):
            for group_id, episode_ids in duplicate_groups.items():
                expected = groups[group_id]
                actual = {
                    assignments[episode_id]
                    for episode_id in episode_ids
                }
                if actual != {expected}:
                    raise EpisodeSplitError(
                        "independence group leaked across splits: "
                        f"{group_id}"
                    )
