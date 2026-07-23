from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path) -> dict:
    manifest = load(root / "manifest.json")
    schema = load(root / "schema.json")
    duplicates = load(root / "duplicate-groups.json")
    quarantine = load(root / "quarantine-manifest.json")
    assert schema["schema_id"] == "ranked_explicit_v3_72"
    assert schema["width"] == 72
    assert len(schema["feature_names"]) == len(set(schema["feature_names"])) == 72
    assert manifest["admitted_episode_count"] == len(manifest["episodes"]) == 11
    assert manifest["quarantined_file_count"] == len(quarantine["files"]) == 3
    assert manifest["duplicate_group_count"] == len(duplicates) == 1
    assert manifest["total_admitted_steps"] == sum(episode["steps"] for episode in manifest["episodes"])
    assert manifest["total_admitted_duration_ms"] == sum(episode["duration_ms"] for episode in manifest["episodes"])
    all_jobs: list[str] = []
    all_episodes: list[str] = []
    all_hashes: list[str] = []
    for split in ("train", "validation", "test"):
        document = load(root / f"{split}.json")
        assert document["job_ids"] == manifest["splits"][split]
        assert document["job_ids"]
        all_jobs.extend(document["job_ids"])
        all_episodes.extend(document["episode_ids"])
    assert len(all_jobs) == len(set(all_jobs)) == 11
    assert len(all_episodes) == len(set(all_episodes)) == 11
    for episode in manifest["episodes"]:
        assert episode["roblox_job_id"] in manifest["splits"][episode["split_assignment"]]
        assert episode["schema"] == schema["schema_id"]
        assert episode["contamination_verdict"] == "clean_passive"
        for digest in episode["source_file_hashes"]:
            assert SHA256.fullmatch(digest)
            all_hashes.append(digest)
    assert len(all_hashes) == len(set(all_hashes))
    checksums = {}
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    for name, expected in checksums.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, name
    return {
        "status": "valid",
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "episodes": 11,
        "steps": manifest["total_admitted_steps"],
        "duration_ms": manifest["total_admitted_duration_ms"],
        "executor_integration_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
