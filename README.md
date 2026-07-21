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

## Repository boundary

```text
live game state
  -> collector
  -> ObservationV1 + ActionMaskV1
  -> optional teacher/policy request
  -> executor result and confirmations
  -> JSONL episode
  -> offline validation and dataset pipeline
```

## Live usage

Deploy:

```text
runtime_luau/collector_bundle.client.luau
```

The MCP can execute the file from a host path, accept inline source on updated installations, or load a pinned public GitHub commit in an authorized test client.

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

## Validation

```bash
python -m unittest discover -s tests -v
```

## Next phase

Step 3 is the dataset pipeline: ingestion, normalization, episode segmentation, quality gates, train/validation/test splitting, dataset statistics, and replay inspection.
