# ABA Deep Learning Agent

Fresh hybrid learning system for an authorized ABA developer-testing environment.

The learned policy will request high-level combat actions. A deterministic runtime remains responsible for legality, cooldowns, stun/recovery gating, input cleanup, and confirmations.

## Current phase

### Step 1: contracts

- versioned observation contract
- frozen 64-value feature vector
- high-level action requests
- legal action masks
- append-only episode records

### Step 2: authoritative collector

**Complete and live-verified.**

- observation-only runtime
- synchronized health, movement, blocking, combo, marker, animation, cooldown, ping, and input signals
- optional teacher or policy action requests
- deterministic JSONL export
- no remote dispatch or combat execution
- live feature-shape, timeline, mask, export, and no-action proof

### Step 3: dataset pipeline

**Implemented.**

- recursive JSONL episode discovery and ingestion
- separate observation and behavior-cloning quality gates
- finite, bounded, fixed-width feature validation
- monotonic timing, gap, mask, target-continuity, and label-coverage checks
- deterministic segmentation that never crosses timing gaps or target changes
- episode-grouped train, validation, and test splits
- deterministic split seed and stable segment IDs
- per-feature mean, standard deviation, minimum, and maximum
- explicit rejection reasons in the dataset manifest
- dependency-free CLI and regression coverage

## Repository boundary

```text
live game state
  -> collector
  -> ObservationV1 + ActionMaskV1
  -> optional teacher/policy request
  -> executor result and confirmations
  -> JSONL episode
  -> quality gates and segmentation
  -> leakage-safe train / validation / test artifacts
```

## Live usage

Deploy:

```text
runtime_luau/collector_bundle.client.luau
```

Then run:

```lua
local collector = getgenv().ABA_NEURAL_COLLECTOR
collector:Start({ sample_hz = 15, max_steps = 9000 })

-- Play manually or run an observed teacher controller.

local result = collector:Export()
print(result.path, result.steps, result.bytes)
```

## Step 2 live proof

Canonical source commit `44a2175e13f843d0795e23ababf39eadda303546` was loaded directly into `devplacetesting10`.

- collector version: `0.2.1`
- schema: `1.0.0`
- target: `Punish Dummy`
- samples: `20` at `15 Hz`
- feature count: `64` on every step
- indices: `0..19`, strictly increasing
- action-mask confidence: `0.82`
- V3 controller enabled: `false`
- collector-caused position delta: `0` studs
- JSONL records: `22`
- JSONL bytes: `76,232`
- JSONL SHA-256: `20cc974f9b50726504317feb74b62d45ac7deb5ed47a2ac5aae44a893d097dee`
- decode and reconstruction result: valid

See `evidence/step2-live-proof-2026-07-21.json`.

## Build a dataset

Observation or representation-learning data:

```bash
python scripts/build_dataset.py episodes/ \
  --output datasets/step3-observation \
  --task observation
```

Behavior-cloning data:

```bash
python scripts/build_dataset.py episodes/ \
  --output datasets/step3-bc \
  --task behavior_cloning \
  --min-action-coverage 0.75
```

See `docs/DATASET.md` for quality scoring, segmentation, and split rules.

## Validation

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Next phase

Step 4 is a compact recurrent baseline: sequence dataloading, masked multi-head behavior cloning, outcome prediction auxiliaries, deterministic baselines, offline evaluation, and model export. Live policy execution remains disabled until offline metrics pass explicit gates.
