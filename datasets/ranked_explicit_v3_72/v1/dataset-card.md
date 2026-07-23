# Dataset card: ranked_explicit_v3_72/v1

## Purpose

Passive human-ranked imitation learning for offline training and prediction-only shadow evaluation. The release must not be used to issue gameplay input or fire remotes.

## Composition

Eleven independent Roblox jobs, 26,465 causally aligned observation/action pairs, and 2,063.183 seconds of admitted capture. Seven jobs are training data, two are validation-only, and two are untouched test data.

## Validation and derivation

Every admitted stream parsed without error, contained exactly 72 finite features per step, had monotonic step indices and timestamps, and passed passive-capture invariants. Missing terminal records were reconstructed only at the last observed timestamp and remain labeled partial. Eight streams used a JSON object keyed `1` through `72`; the importer requires every numeric index exactly once and creates an ordered derived vector without modifying the raw source.

## Schema

`ranked_feature_vector_v3_explicit` is accepted as an alias only after exact feature-name and order comparison against `schema.json`. The old v0.1 stream is quarantined and is not coerced.

## Limitations

The corpus is small, action distributions are highly imbalanced, several episodes are partial, and the verified feature contract does not expose authoritative cooldown availability. Frame-level accuracy alone is not a release criterion.

## Safety boundary

Autonomous gameplay is disabled. Models may run offline or through the prediction-only JSONL shadow runner, which has no input, remote, or character-state mutation API.
