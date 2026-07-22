# Feature schema registry

The project contains multiple feature vectors that happen to share similar widths but do not share meanings or ordering. Width alone is never a compatibility guarantee.

## Registered schemas

| Canonical schema | Width | Purpose |
|---|---:|---|
| `observation_v1_64` | 64 | Original normalized observation vector used by the authoritative collector. |
| `human_camera_v2_72` | 72 | Legacy 64-vector plus eight camera-alignment fields from the human demonstration recorder. |
| `ranked_explicit_v3_72` | 72 | Explicit ranked-watcher vector with named self, opponent, camera, input, HUD, network, and lifecycle fields. |

`human_camera_v2_72` and `ranked_explicit_v3_72` are intentionally incompatible. They must not be concatenated, normalized together, or placed in the same train/validation/test files unless a separately tested lossless adapter is introduced.

## Admission rules

1. Declared schema IDs and aliases are canonicalized through `feature_schemas.py`.
2. An undeclared 64-feature stream may be treated as `observation_v1_64` for legacy compatibility.
3. An undeclared 72-feature stream is rejected as ambiguous.
4. A declared schema whose vector width does not match is rejected.
5. `build_dataset()` refuses accepted episodes from multiple compatibility groups.
6. `build_partitioned_datasets()` writes isolated datasets per schema.
7. Segment IDs include the schema ID so equal episode ranges from incompatible schemas cannot collide.

## CLI

Build one required schema:

```bash
python scripts/build_dataset.py episodes/ \
  --output datasets/ranked-v3 \
  --task behavior_cloning \
  --feature-schema ranked_explicit_v3_72 \
  --allow-missing-action-masks
```

Partition a mixed corpus without merging it:

```bash
python scripts/build_dataset.py episodes/ \
  --output datasets/by-schema \
  --task observation \
  --partition-by-schema
```

The manifest for each partition records its canonical schema ID, version, width, and compatibility group.
