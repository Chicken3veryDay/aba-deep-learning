# Ranked dataset closure and first multi-match model

The canonical release is `datasets/ranked_explicit_v3_72/v1/`. Raw executor JSONL is not committed; manifests, source hashes, derivation history, deterministic import tooling, split assignments, model reports, and checkpoints are committed.

## Reproduction

```bash
python -m pip install -e ".[ml]"
python scripts/build_ranked_release.py aba_deep_learning/ranked_watch aba_deep_learning/ranked_watch_v2 --output datasets/ranked_explicit_v3_72/v1 --relative-to . --cache ranked-v1.npz
python scripts/train_ranked.py --cache ranked-v1.npz --output runs/baseline --model baseline --seed 20260722
python scripts/train_ranked.py --cache ranked-v1.npz --output runs/gru --model gru --seed 20260722
python scripts/evaluate_ranked.py --cache ranked-v1.npz --checkpoint runs/gru/best.pt --output runs/gru/evaluation.json
python scripts/validate_ranked_release.py datasets/ranked_explicit_v3_72/v1
```

Shadow mode stays disabled by default and has no input, remote, or character-state mutation interface.
