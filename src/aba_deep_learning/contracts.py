from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .feature_schemas import (
    FeatureSchemaError,
    get_feature_schema,
)

SCHEMA_VERSION = "1.0.0"
FEATURE_VECTOR_LENGTH = 64

ACTION_INTENTS = frozenset({
    "idle", "approach", "orbit", "retreat", "defend",
    "pressure", "punish", "bait", "disengage",
})
ACTION_COMMANDS = frozenset({
    "none", "m1", "block_start", "block_stop", "dodge",
    "jump", "move", "transform", "reset_spacing",
})
FACING_MODES = frozenset({
    "preserve", "face_target", "face_movement", "camera",
})


class ContractError(ValueError):
    pass


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{path} must be a non-empty string")
    return value


def _integer(
    value: Any,
    path: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{path} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ContractError(f"{path} is outside the allowed range")
    return value


def _number(
    value: Any,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise ContractError(f"{path} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ContractError(f"{path} must be <= {maximum}")
    return result


def _version(obj: Mapping[str, Any], path: str) -> None:
    if obj.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"{path}.schema_version must equal {SCHEMA_VERSION!r}")


def _declared_observation_width(obj: Mapping[str, Any]) -> int | None:
    for key in ("feature_schema_id", "feature_schema", "feature_vector_schema"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            try:
                return get_feature_schema(value).width
            except FeatureSchemaError as exc:
                raise ContractError(str(exc)) from exc
    return None


def validate_observation(
    value: Any,
    *,
    expected_feature_width: int | None = FEATURE_VECTOR_LENGTH,
) -> None:
    obj = _mapping(value, "observation")
    _version(obj, "observation")
    _string(obj.get("episode_id"), "observation.episode_id")
    _integer(obj.get("step_index"), "observation.step_index")
    _integer(obj.get("timestamp_ms"), "observation.timestamp_ms")
    _integer(obj.get("dt_ms"), "observation.dt_ms", 0, 1000)

    for name in (
        "self_state", "target_state", "relative", "combat",
        "cooldowns", "network", "history",
    ):
        _mapping(obj.get(name), f"observation.{name}")

    vector = obj.get("feature_vector")
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise ContractError("observation.feature_vector must be an array")

    declared_width = _declared_observation_width(obj)
    width = expected_feature_width
    if declared_width is not None:
        if width is not None and width != declared_width:
            raise ContractError(
                f"observation feature schema expects width {declared_width}, "
                f"stream expects {width}"
            )
        width = declared_width
    if width is None:
        raise ContractError(
            "expected_feature_width is required when the observation does not "
            "declare a feature schema"
        )
    if len(vector) != width:
        raise ContractError(f"feature_vector must contain {width} values")
    for index, item in enumerate(vector):
        _number(item, f"observation.feature_vector[{index}]")


def validate_action_request(value: Any) -> None:
    obj = _mapping(value, "action")
    _version(obj, "action")
    if obj.get("intent") not in ACTION_INTENTS:
        raise ContractError("action.intent is invalid")
    if obj.get("command") not in ACTION_COMMANDS:
        raise ContractError("action.command is invalid")
    if obj.get("facing_mode") not in FACING_MODES:
        raise ContractError("action.facing_mode is invalid")

    _number(obj.get("move_x"), "action.move_x", -1, 1)
    _number(obj.get("move_z"), "action.move_z", -1, 1)
    _number(obj.get("sprint"), "action.sprint", 0, 1)
    _integer(obj.get("delay_ms"), "action.delay_ms", 0, 1000)
    _integer(obj.get("hold_ms"), "action.hold_ms", 0, 3000)
    _number(obj.get("confidence"), "action.confidence", 0, 1)
    _string(obj.get("policy_id"), "action.policy_id")
    _string(obj.get("model_version"), "action.model_version")

    move_slot = obj.get("move_slot")
    if obj.get("command") == "move":
        _integer(move_slot, "action.move_slot", 1, 4)
    elif move_slot is not None:
        raise ContractError("move_slot must be null unless command='move'")


def validate_action_mask(value: Any) -> None:
    obj = _mapping(value, "mask")
    _version(obj, "mask")
    commands = _mapping(obj.get("commands"), "mask.commands")
    for command in ACTION_COMMANDS:
        if not isinstance(commands.get(command), bool):
            raise ContractError(f"mask.commands.{command} must be boolean")
    slots = obj.get("move_slots")
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise ContractError("mask.move_slots must be an array")
    if len(slots) != 4 or any(not isinstance(item, bool) for item in slots):
        raise ContractError("mask.move_slots must contain four booleans")
    _mapping(obj.get("reasons"), "mask.reasons")
    _number(obj.get("confidence"), "mask.confidence", 0, 1)


def validate_step(
    value: Any,
    previous_index: int | None = None,
    *,
    expected_feature_width: int | None = FEATURE_VECTOR_LENGTH,
) -> int:
    step = _mapping(value, "step")
    observation = step.get("observation")
    validate_observation(
        observation,
        expected_feature_width=expected_feature_width,
    )
    index = int(observation["step_index"])
    if previous_index is not None and index <= previous_index:
        raise ContractError("step indices must be strictly increasing")
    validate_action_mask(step.get("action_mask"))
    if step.get("action_request") is not None:
        validate_action_request(step["action_request"])
    _mapping(step.get("executor_result"), "step.executor_result")
    confirmations = step.get("confirmations")
    raw_events = step.get("raw_events")
    if not isinstance(confirmations, Sequence) or isinstance(confirmations, (str, bytes)):
        raise ContractError("step.confirmations must be an array")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise ContractError("step.raw_events must be an array")
    rewards = _mapping(step.get("rewards"), "step.rewards")
    for key, reward in rewards.items():
        _number(reward, f"step.rewards.{key}")
    return index
