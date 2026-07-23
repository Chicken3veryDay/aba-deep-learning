from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_ID = "ranked_explicit_v3_72"
SCHEMA_VERSION = "3.0.0"
SCHEMA_ALIASES = frozenset({SCHEMA_ID, "ranked_feature_vector_v3_explicit", "ranked_explicit_v3", "feature_vector_v3_explicit"})
FEATURE_NAMES = (
    "self_health_ratio", "target_health_ratio", "distance_norm_200", "relative_height_norm_50",
    "self_velocity_x_norm_100", "self_velocity_y_norm_100", "self_velocity_z_norm_100",
    "target_velocity_x_norm_100", "target_velocity_y_norm_100", "target_velocity_z_norm_100",
    "self_move_world_x", "self_move_world_z", "self_facing_dot_target", "self_facing_cross_target",
    "target_facing_dot_self", "target_facing_cross_self", "relative_direction_local_x",
    "relative_direction_local_z", "relative_velocity_local_x_norm_100", "relative_velocity_local_z_norm_100",
    "self_grounded", "target_grounded", "self_airborne", "target_airborne", "self_blocking_marker",
    "target_blocking_marker", "self_stunned_marker", "target_stunned_marker", "self_iframes_marker",
    "target_iframes_marker", "self_attacking_marker", "target_attacking_marker", "self_sprinting_estimate",
    "target_moving", "in_m1_range_6", "in_close_move_range_20", "target_on_screen", "target_screen_x",
    "target_screen_y", "camera_dot_self_forward", "camera_cross_self_forward", "camera_dot_target_direction",
    "camera_cross_target_direction", "camera_pitch", "camera_distance_to_self_norm_20", "ping_norm_250",
    "dt_norm_100ms", "self_health_delta_norm_20", "target_health_delta_norm_20", "self_took_damage",
    "target_took_damage", "self_state_running", "self_state_jumping", "self_state_freefall",
    "target_state_running", "target_state_jumping", "target_state_freefall", "self_animation_count_norm_12",
    "target_animation_count_norm_12", "self_walkspeed_norm_50", "target_walkspeed_norm_50",
    "self_jump_power_norm_100", "target_jump_power_norm_100", "camera_fov_norm_120",
    "camera_dot_self_forward_v2", "camera_cross_self_forward_v2", "camera_dot_target_direction_v2",
    "camera_cross_target_direction_v2", "camera_pitch_v2", "target_screen_x_v2", "target_screen_y_v2",
    "target_on_screen_v2",
)
ACTION_NAMES = ("sprint", "m1", "block", "dodge", "jump", "move_slot_1", "move_slot_2", "move_slot_3", "move_slot_4")

class RankedReleaseError(ValueError):
    pass

@dataclass
class CanonicalStep:
    step_index: int
    timestamp_ms: int
    feature_vector: list[float]
    move_x: float
    move_z: float
    actions: dict[str, bool]
    label_available: bool
    label_source: str

@dataclass
class FileInventory:
    relative_path: str
    byte_size: int
    sha256: str
    line_count: int = 0
    parse_error_count: int = 0
    parse_error_lines: list[int] = field(default_factory=list)
    header_count: int = 0
    step_count: int = 0
    lifecycle_record_count: int = 0
    event_count: int = 0
    terminal_record_count: int = 0
    watcher_version: str | None = None
    stream_format: str | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    feature_width: int | None = None
    episode_id: str | None = None
    roblox_job_id: str | None = None
    opponent: str | None = None
    start_timestamp: int | None = None
    end_timestamp: int | None = None
    duration_ms: int | None = None
    contamination_state: str = "clean"
    terminal_reason: str | None = None
    terminal_reconstructed: bool = False
    partial: bool = False
    action_counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in ACTION_NAMES})
    held_input_counts: dict[str, int] = field(default_factory=dict)
    damage_events: int = 0
    health_changes: int = 0
    duplicate_candidates: list[str] = field(default_factory=list)
    admission_status: str = "pending"
    rejection_or_quarantine_reasons: list[str] = field(default_factory=list)
    step_index_errors: int = 0
    timestamp_errors: int = 0
    duplicate_step_indices: int = 0
    unreasonable_timestamp_jumps: int = 0
    nonfinite_feature_count: int = 0
    vector_encoding_counts: dict[str, int] = field(default_factory=dict)
    canonical_schema: str | None = None
    schema_resolution: str = "unresolved"
    ordered_step_hash: str | None = None
    split_assignment: str | None = None

@dataclass
class ParsedRecording:
    inventory: FileInventory
    header: dict[str, Any]
    terminal: dict[str, Any] | None
    steps: list[CanonicalStep]


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RankedReleaseError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RankedReleaseError(f"{path} must be finite")
    return result


def normalize_feature_vector(value: Any) -> tuple[list[float], str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 72:
            raise RankedReleaseError(f"feature vector width is {len(value)}, expected 72")
        return [_finite(item, f"feature_vector[{index}]") for index, item in enumerate(value)], "json_array"
    if isinstance(value, Mapping):
        result: list[float | None] = [None] * 72
        for raw_key, raw_value in value.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise RankedReleaseError("feature object has a nonnumeric key") from exc
            if raw_key != key and str(key) != str(raw_key):
                raise RankedReleaseError(f"feature object key is not canonical: {raw_key!r}")
            if not 1 <= key <= 72 or result[key - 1] is not None:
                raise RankedReleaseError(f"invalid or duplicate feature object key: {key}")
            result[key - 1] = _finite(raw_value, f"feature_vector[{key}]")
        missing = [index + 1 for index, item in enumerate(result) if item is None]
        if missing:
            raise RankedReleaseError(f"feature object missing indices: {missing}")
        return [float(item) for item in result], "numeric_key_object_1_72"
    raise RankedReleaseError("feature vector must be an array or object")


def _truth(value: Any) -> bool:
    return value is True or value == 1 or value == "1"


def _held(mapping: Mapping[str, Any], *names: str) -> bool:
    return any(_truth(mapping.get(name)) for name in names)


def extract_action_label(step: Mapping[str, Any]) -> tuple[float, float, dict[str, bool], bool, str]:
    direct = step.get("action_label")
    if isinstance(direct, Mapping):
        slot = int(direct.get("move_slot", 0) or 0)
        actions = {
            "sprint": _truth(direct.get("sprint")), "m1": _truth(direct.get("m1")),
            "block": _truth(direct.get("block_held")) or _truth(direct.get("block_start")),
            "dodge": _truth(direct.get("dodge")), "jump": _truth(direct.get("jump")),
            **{f"move_slot_{index}": slot == index for index in range(1, 5)},
        }
        return float(direct.get("move_x", 0) or 0), float(direct.get("move_z", 0) or 0), actions, True, "action_label"
    input_record = step.get("input")
    held = input_record.get("held") if isinstance(input_record, Mapping) else step.get("held")
    if not isinstance(held, Mapping):
        return 0.0, 0.0, {name: False for name in ACTION_NAMES}, False, "missing"
    movement = (
        float(int(_held(held, "d", "D")) - int(_held(held, "a", "A"))),
        float(int(_held(held, "w", "W")) - int(_held(held, "s", "S"))),
    )
    actions = {
        "sprint": _held(held, "shift", "LeftShift", "RightShift"),
        "m1": _held(held, "mouse1", "MouseButton1"),
        "block": _held(held, "f", "F", "mouse2", "MouseButton2"),
        "dodge": _held(held, "q", "Q"), "jump": _held(held, "space", "Space"),
        "move_slot_1": _held(held, "one", "One"), "move_slot_2": _held(held, "two", "Two"),
        "move_slot_3": _held(held, "three", "Three"), "move_slot_4": _held(held, "four", "Four"),
    }
    return movement[0], movement[1], actions, True, "held"
