from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(root: Path) -> dict[str, Any]:
    registry = load_json(root / "contracts" / "feature_schemas_v1.json")
    ranked_manifest = load_json(
        root / "datasets" / "metadata" / "ranked_explicit_v3_v1.json"
    )
    ranked_package = load_json(
        root / "datasets" / "metadata"
        / "ranked_match_1784755145671.package.json"
    )
    human_manifest = load_json(
        root / "datasets" / "metadata" / "human_behavior_cloning_v1.json"
    )
    corpus_package = load_json(
        root / "datasets" / "metadata" / "human_corpus_v2.package.json"
    )
    closure = load_json(root / "evidence" / "ranked-capture-v2-closure.json")
    rejection = load_json(
        root / "evidence" / "ranked-temporal-mlp-v0-rejection.json"
    )
    release_gate = load_json(
        root / "evidence" / "human-policy-release-gate-v2.json"
    )

    schemas = registry.get("schemas")
    require(isinstance(schemas, list) and schemas, "schema registry is empty")
    schema_ids = [item.get("schema_id") for item in schemas]
    require(
        len(schema_ids) == len(set(schema_ids)),
        "schema registry contains duplicate IDs",
    )
    aliases: set[str] = set()
    for schema in schemas:
        require(
            isinstance(schema.get("width"), int) and schema["width"] > 0,
            f"invalid width for {schema.get('schema_id')}",
        )
        for alias in schema.get("aliases", []):
            require(alias not in aliases, f"duplicate schema alias: {alias}")
            aliases.add(alias)

    ranked_schema = next(
        schema
        for schema in schemas
        if schema["schema_id"] == "ranked_explicit_v3_72"
    )
    human_schema = next(
        schema
        for schema in schemas
        if schema["schema_id"] == "human_camera_v2_72"
    )
    require(ranked_schema["width"] == 72, "ranked schema width must be 72")
    require(human_schema["width"] == 72, "human schema width must be 72")
    require(
        ranked_schema["compatibility_group"]
        != human_schema["compatibility_group"],
        "human and ranked 72-feature schemas must remain incompatible",
    )
    require(
        ranked_manifest["feature_schema"] in ranked_schema["aliases"],
        "ranked manifest schema alias is not registered",
    )
    require(
        human_manifest["feature_schema"] in human_schema["aliases"],
        "human manifest schema alias is not registered",
    )
    require(
        ranked_manifest["feature_width"] == 72,
        "ranked manifest width mismatch",
    )
    require(
        human_manifest["feature_width"] == 72,
        "human manifest width mismatch",
    )
    require(
        ranked_package["feature_width"] == 72,
        "ranked package width mismatch",
    )
    require(
        ranked_manifest["source_sha256"]
        == ranked_package["canonical_clean_sha256"],
        "manifest and package clean hashes disagree",
    )
    require(
        SHA256_RE.fullmatch(
            ranked_package["canonical_clean_sha256"]
        )
        is not None,
        "canonical clean hash is not SHA-256",
    )
    require(
        closure["canonical_episode"] == ranked_manifest["episode_id"],
        "closure episode does not match manifest",
    )
    require(
        closure["dataset"]["steps"] == ranked_package["steps"],
        "closure and package step counts disagree",
    )
    require(
        ranked_package["training_eligible"] is True,
        "ranked package must remain training eligible",
    )
    require(
        ranked_package["release_eligible"] is False,
        "ranked package must not be release eligible",
    )
    require(
        closure["release"]["executor_integration_allowed"] is False,
        "closure must prohibit executor integration",
    )
    require(
        rejection["status"] == "REJECTED_DISTRIBUTION_SHIFT",
        "ranked model rejection status changed",
    )
    prohibited = rejection.get("prohibited", {})
    for key in (
        "release_selection",
        "executor_integration",
        "production_runtime",
    ):
        require(
            prohibited.get(key) is True,
            f"missing model prohibition: {key}",
        )

    partitions = corpus_package.get("schema_partitions", {})
    require(
        set(partitions)
        == {"feature_vector_v2", "ranked_feature_vector_v3_explicit"},
        "human corpus schema partitions changed",
    )
    require(
        corpus_package["compatibility"]["direct_merge_allowed"] is False,
        "human corpus must prohibit direct schema merge",
    )
    require(
        corpus_package["release_eligible"] is False,
        "human corpus must not be release eligible",
    )
    require(
        corpus_package["executor_integration_allowed"] is False,
        "human corpus must prohibit executor integration",
    )
    require(
        release_gate["release_eligible"] is False,
        "human release gate must remain closed",
    )
    require(
        release_gate["executor_integration_allowed"] is False,
        "human release gate must prohibit executor integration",
    )
    require(
        release_gate["schema_partitions"]
        == {
            "feature_vector_v2": 1,
            "ranked_feature_vector_v3_explicit": 1,
        },
        "release gate schema counts changed",
    )
    blocking_ids = {
        item["id"]
        for item in release_gate.get("blocking_checks", [])
    }
    for required_id in (
        "same_schema_independent_human_episodes",
        "schema_partition_count",
        "movement_vector_accuracy",
        "m1_precision",
        "block_start_examples_corpus",
        "move_slot_4_examples_corpus",
    ):
        require(
            required_id in blocking_ids,
            f"release gate lost blocker: {required_id}",
        )

    return {
        "schemas": len(schemas),
        "schema_partitions": len(partitions),
        "ranked_episode": ranked_manifest["episode_id"],
        "ranked_steps": ranked_package["steps"],
        "ranked_feature_width": ranked_package["feature_width"],
        "human_steps": human_manifest["totals"]["steps"],
        "clean_sha256": ranked_package["canonical_clean_sha256"],
        "training_eligible": ranked_package["training_eligible"],
        "release_eligible": release_gate["release_eligible"],
        "model_status": rejection["status"],
        "executor_integration_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate committed ABA evidence and release prohibitions"
    )
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
