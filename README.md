# ABA Deep Learning Agent

Fresh hybrid learning system for an authorized ABA developer-testing environment.

The learned policy requests high-level combat actions. A deterministic runtime remains responsible for legality, cooldowns, stun/recovery gating, input cleanup, confirmations, and lifecycle recovery. Learned-policy executor integration remains disabled until independent episode-level evaluation passes explicit release gates.

## Current state

### Contracts and collection

- versioned observations, action requests, masks, executor results, confirmations, and terminal records
- observation-only live collector
- human demonstration recorder with controller-contamination abort
- passive ranked watcher with teleport persistence, periodic flushes, singleton locking, HUD-noise filtering, and automatic match-result sealing
- no watcher UI, input simulation, gameplay control, or remote calls

### Schema-safe dataset pipeline

The repository now recognizes three incompatible feature contracts:

| Schema | Width | Status |
|---|---:|---|
| `observation_v1_64` | 64 | Legacy authoritative collector |
| `human_camera_v2_72` | 72 | Human recorder, legacy 64 plus camera fields |
| `ranked_explicit_v3_72` | 72 | Explicit ranked watcher |

Two 72-feature schemas are not assumed compatible merely because their widths match.

The dataset builder now:

- canonicalizes declared schema IDs and aliases
- rejects undeclared 72-feature streams as ambiguous
- validates declared width against the registry
- refuses mixed compatibility groups in `build_dataset()`
- supports isolated per-schema output through `build_partitioned_datasets()`
- records schema ID, version, width, and compatibility group in manifests and segments
- includes the schema ID in segment hashes
- supports passive human labels without requiring an action mask when explicitly configured

See `docs/FEATURE_SCHEMAS.md`.

## Ranked evidence

Canonical clean ranked episode metadata:

- episode: `ranked-watch-v2-1784755145671`
- duration: 187.501 seconds
- steps: 2,806
- feature schema: `ranked_explicit_v3_72`
- feature width: 72
- parse errors: 0
- clean source SHA-256: `e0117462db8b9075033e2413984ea3552737e2c83f3a227df7608be35b0ab03c`
- duplicate watcher stream: quarantined

The multi-megabyte raw episode is intentionally not committed. Its immutable package and evidence live under:

```text
datasets/metadata/ranked_match_1784755145671.package.json
datasets/metadata/ranked_explicit_v3_v1.json
evidence/ranked-capture-v2-closure.json
```

## Model release status

The single-match ranked temporal baseline was rejected for distribution shift:

- untouched test exact movement accuracy: 10.4%
- test M1 precision: 1.6%
- test block F1: 1.9%

It is prohibited from executor integration or production selection. See `evidence/ranked-temporal-mlp-v0-rejection.json`.

The next valid model milestone requires at least two additional independent `ranked_explicit_v3_72` episodes and episode-level train, validation, and test separation.

## Build a dataset

One schema only:

```bash
python scripts/build_dataset.py episodes/ \
  --output datasets/ranked-v3 \
  --task behavior_cloning \
  --feature-schema ranked_explicit_v3_72 \
  --allow-missing-action-masks
```

Safely partition a mixed corpus:

```bash
python scripts/build_dataset.py episodes/ \
  --output datasets/by-schema \
  --task observation \
  --partition-by-schema
```

## Validation

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The schema-safe v0.4.0 rewrite passes legacy 64-feature contract tests plus explicit 72-feature stream, ambiguity, partitioning, and mixed-schema refusal regressions.
