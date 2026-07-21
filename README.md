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

- observation-only runtime
- synchronized health, movement, blocking, combo, marker, animation, cooldown, ping, and input signals
- optional legacy-controller decision observation
- deterministic JSONL export
- no remote dispatch or combat execution

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

After the Roblox MCP exposes inline `execute-file` transfer, deploy:

```text
runtime_luau/collector_bundle.client.luau
```

Then run:

```lua
local collector = getgenv().ABA_NEURAL_COLLECTOR
collector:Start({ sample_hz = 15, max_steps = 9000 })

-- Play manually or run an observed teacher controller.

local result = collector:Export()
print(result.path, result.steps, result.sha256)
```

## Validation

```bash
python -m unittest discover -s tests -v
```

The next phase begins only after a live episode passes feature-shape, timeline, mask, signal, export, and no-action verification.
