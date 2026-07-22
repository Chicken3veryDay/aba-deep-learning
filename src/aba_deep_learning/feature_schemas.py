from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

LEGACY_SCHEMA_ID = "observation_v1_64"
HUMAN_CAMERA_SCHEMA_ID = "human_camera_v2_72"
RANKED_EXPLICIT_SCHEMA_ID = "ranked_explicit_v3_72"


class FeatureSchemaError(ValueError):
    """Base error for feature-schema resolution failures."""


class UnknownFeatureSchemaError(FeatureSchemaError):
    pass


class AmbiguousFeatureSchemaError(FeatureSchemaError):
    pass


class IncompatibleFeatureSchemaError(FeatureSchemaError):
    pass


@dataclass(frozen=True)
class FeatureSchema:
    schema_id: str
    width: int
    version: str
    compatibility_group: str
    description: str
    aliases: tuple[str, ...] = ()
    reference_producer_sha256: str | None = None

    def matches(self, value: str) -> bool:
        normalized = value.strip().lower()
        return normalized == self.schema_id.lower() or normalized in {
            alias.lower() for alias in self.aliases
        }


_SCHEMAS = (
    FeatureSchema(
        schema_id=LEGACY_SCHEMA_ID,
        width=64,
        version="1.0.0",
        compatibility_group=LEGACY_SCHEMA_ID,
        description="Original normalized 64-feature observation vector.",
        aliases=(
            "observation_v1",
            "feature_vector_v1",
            "legacy_64",
        ),
    ),
    FeatureSchema(
        schema_id=HUMAN_CAMERA_SCHEMA_ID,
        width=72,
        version="2.0.0",
        compatibility_group=HUMAN_CAMERA_SCHEMA_ID,
        description=(
            "Human demonstration vector: opaque legacy 64-vector plus eight "
            "camera-alignment features."
        ),
        aliases=(
            "feature_vector_v2",
            "human_feature_vector_v2",
            "camera_feature_vector_v2",
            "human_camera_v2",
        ),
    ),
    FeatureSchema(
        schema_id=RANKED_EXPLICIT_SCHEMA_ID,
        width=72,
        version="3.0.0",
        compatibility_group=RANKED_EXPLICIT_SCHEMA_ID,
        description=(
            "Explicit ranked watcher vector with named self, opponent, camera, "
            "input, HUD, network, and lifecycle features."
        ),
        aliases=(
            "ranked_feature_vector_v3_explicit",
            "ranked_explicit_v3",
            "feature_vector_v3_explicit",
        ),
        reference_producer_sha256=(
            "c695c08783c53b8cf7c7a9741ed6dd433a3ffb75d9f22af0"
            "b109f380e19dbebd"
        ),
    ),
)

FEATURE_SCHEMAS = {schema.schema_id: schema for schema in _SCHEMAS}
_ALIAS_INDEX = {
    alias.lower(): schema
    for schema in _SCHEMAS
    for alias in (schema.schema_id, *schema.aliases)
}

_HEADER_KEYS = (
    "feature_schema_id",
    "feature_schema",
    "feature_vector_schema",
    "observation_schema_id",
)
_OBSERVATION_KEYS = ("feature_schema_id", "feature_schema", "feature_vector_schema")


def list_feature_schemas() -> tuple[FeatureSchema, ...]:
    return _SCHEMAS


def get_feature_schema(schema_id_or_alias: str) -> FeatureSchema:
    if not isinstance(schema_id_or_alias, str) or not schema_id_or_alias.strip():
        raise UnknownFeatureSchemaError("feature schema id must be a non-empty string")
    schema = _ALIAS_INDEX.get(schema_id_or_alias.strip().lower())
    if schema is None:
        raise UnknownFeatureSchemaError(
            f"unknown feature schema: {schema_id_or_alias!r}"
        )
    return schema


def feature_width(episode: Mapping[str, Any]) -> int | None:
    steps = episode.get("steps", [])
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return None
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        observation = step.get("observation")
        if not isinstance(observation, Mapping):
            continue
        vector = observation.get("feature_vector")
        if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)):
            return len(vector)
    return None


def _declared_schema_value(episode: Mapping[str, Any]) -> str | None:
    header = episode.get("header")
    if isinstance(header, Mapping):
        for key in _HEADER_KEYS:
            value = header.get(key)
            if isinstance(value, str) and value.strip():
                return value

    steps = episode.get("steps", [])
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            observation = step.get("observation")
            if not isinstance(observation, Mapping):
                continue
            for key in _OBSERVATION_KEYS:
                value = observation.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            break
    return None


def resolve_episode_schema(
    episode: Mapping[str, Any],
    *,
    allow_legacy_64_inference: bool = True,
) -> FeatureSchema:
    declared = _declared_schema_value(episode)
    width = feature_width(episode)

    if declared is not None:
        schema = get_feature_schema(declared)
        if width is not None and width != schema.width:
            raise IncompatibleFeatureSchemaError(
                f"declared schema {schema.schema_id!r} expects width "
                f"{schema.width}, observed {width}"
            )
        return schema

    if width == 64 and allow_legacy_64_inference:
        return FEATURE_SCHEMAS[LEGACY_SCHEMA_ID]

    if width == 72:
        raise AmbiguousFeatureSchemaError(
            "72-feature episode has no feature_schema_id; human_camera_v2_72 "
            "and ranked_explicit_v3_72 are intentionally incompatible"
        )

    if width is None:
        raise UnknownFeatureSchemaError("episode contains no feature vectors")

    raise UnknownFeatureSchemaError(
        f"no registered feature schema for inferred width {width}"
    )


def schemas_compatible(left: str | FeatureSchema, right: str | FeatureSchema) -> bool:
    left_schema = get_feature_schema(left) if isinstance(left, str) else left
    right_schema = get_feature_schema(right) if isinstance(right, str) else right
    return left_schema.compatibility_group == right_schema.compatibility_group


def require_compatible_schemas(
    schemas: Sequence[str | FeatureSchema],
) -> FeatureSchema:
    if not schemas:
        raise IncompatibleFeatureSchemaError("no feature schemas supplied")
    resolved = [
        get_feature_schema(value) if isinstance(value, str) else value
        for value in schemas
    ]
    first = resolved[0]
    incompatible = [
        schema.schema_id
        for schema in resolved[1:]
        if schema.compatibility_group != first.compatibility_group
    ]
    if incompatible:
        raise IncompatibleFeatureSchemaError(
            "incompatible feature schemas: "
            + ", ".join([first.schema_id, *incompatible])
        )
    return first
