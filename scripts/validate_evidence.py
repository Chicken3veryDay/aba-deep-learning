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
    manifest = load_json(
        root / "datasets" / "metadata" / "ranked_explicit_v3_v1.json"
    )
    package = load_json(
        root / "datasets" / "metadata"
        / "ranked_match_1784755145671.package.json"
    )
    closure = load_json(root / "evidence" / "ranked-capture-v2-closure.json")
    rejection = load_json(
        root / "evidence" / "ranked-temporal-mlp-v0-rejection.json"
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
    require(ranked_schema["width"] == 72, "ranked schema width must be 72")
    require(
        manifest["feature_schema"] in ranked_schema["aliases"],
        "ranked manifest schema alias is not registered",
    )
    require(manifest["feature_width"] == 72, "ranked manifest width mismatch")
    require(package["feature_width"] == 72, "ranked package width mismatch")
    require(
        manifest["source_sha256"] == package["canonical_clean_sha256"],
        "manifest and package clean hashes disagree",
    )
    require(
        SHA256_RE.fullmatch(package["canonical_clean_sha256"]) is not None,
        "canonical clean hash is not SHA-256",
    )
    require(
        closure["canonical_episode"] == manifest["episode_id"],
        "closure episode does not match manifest",
    )
    require(
        closure["dataset"]["steps"] == package["steps"],
        "closure and package step counts disagree",
    )
    require(
        package["training_eligible"] is True,
        "ranked package must remain training eligible",
    )
    require(
        package["release_eligible"] is False,
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
    for key in ("release_selection", "executor_integration", "production_runtime"):
        require(prohibited.get(key) is True, f"missing model prohibition: {key}")

    return {
        "schemas": len(schemas),
        "ranked_episode": manifest["episode_id"],
        "ranked_steps": package["steps"],
        "ranked_feature_width": package["feature_width"],
        "clean_sha256": package["canonical_clean_sha256"],
        "training_eligible": package["training_eligible"],
        "release_eligible": package["release_eligible"],
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
