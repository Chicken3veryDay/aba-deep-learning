# Step 3 dataset pipeline

The dataset pipeline converts validated episode JSONL streams into deterministic training artifacts without adding third-party dependencies.

## Tasks

### Observation

Use for representation learning, outcome prediction, state modeling, and replay analysis. Action labels are optional.

### Behavior cloning

Use for supervised policy training. Episodes must satisfy the configured minimum action-request coverage.

## Quality gates

Each episode receives a structured report covering:

- exact 64-value feature shape
- finite and bounded feature values
- monotonic timestamps
- excessive sampling gaps
- action-mask structure and confidence
- target continuity
- action-label coverage
- aggregate quality score

Rejected episodes remain listed in `manifest.json` with explicit reasons. They are never silently discarded.

## Segmentation

Episodes are first divided into contiguous runs. A new run starts when:

- the timestamp gap exceeds `max_gap_ms`
- the observed target identity changes

Runs are then converted into overlapping fixed-size windows. Short valid tails are retained without allowing any window to cross a gap or target switch.

## Leakage-safe splitting

The split is selected by hashing:

```text
split_seed + episode_id
```

All windows from the same episode therefore remain together in train, validation, or test. Re-running with the same seed produces the same assignment.

## Outputs

```text
output/
├── manifest.json
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

Each split line contains one segment with:

- feature vectors
- action masks
- optional action requests
- executor results
- reward components
- timestamps and original step indices
- source episode metadata

The manifest records configuration, quality reports, split counts, label coverage, and per-feature mean, standard deviation, minimum, and maximum.

## CLI

```bash
python scripts/build_dataset.py episodes/ \
  --output datasets/step3-observation \
  --task observation
```

Behavior cloning example:

```bash
python scripts/build_dataset.py episodes/ \
  --output datasets/step3-bc \
  --task behavior_cloning \
  --min-action-coverage 0.75
```

## Default configuration

| Setting | Default |
|---|---:|
| Window size | 32 steps |
| Stride | 16 steps |
| Minimum segment | 8 steps |
| Maximum gap | 250 ms |
| Minimum quality score | 0.70 |
| Minimum BC label coverage | 0.50 |
| Train / validation / test | 80% / 10% / 10% |

The next phase should train a compact recurrent baseline against these artifacts and compare it to deterministic and no-op baselines before any live policy execution is enabled.
