# Ranked imitation model v1

- Selected model: `baseline`
- Dataset manifest SHA-256: `affb81582c503b76b84780a4d1ed4cc87868d378b1bdd1be7f716d23e17adad8`
- Selected checkpoint SHA-256: `93bde1ac36c5339a52464604d48529686f606bf2459ebdaec8724b1f3bf06c0c`
- Test evaluations: 1
- Test movement MAE: 0.357753
- Test movement/no-movement accuracy: 0.745640
- Mean inference latency: 0.196764 ms

## Untouched-test action metrics

| Action | Precision | Recall | F1 | Support | FPR | FNR | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| block | 0.446942 | 0.821488 | 0.578917 | 605 | 0.196989 | 0.178512 | 0.525 |
| dodge | 0.112290 | 0.472119 | 0.181429 | 269 | 0.290341 | 0.527881 | 0.675 |
| jump | 0.399803 | 0.636792 | 0.491207 | 636 | 0.196700 | 0.363208 | 0.700 |
| m1 | 0.370651 | 0.758514 | 0.497967 | 646 | 0.270042 | 0.241486 | 0.650 |
| move_slot_1 | 0.011222 | 0.931034 | 0.022177 | 29 | 0.643321 | 0.068966 | 0.325 |
| move_slot_2 | 0.005960 | 0.823529 | 0.011834 | 17 | 0.629380 | 0.176471 | 0.425 |
| move_slot_3 | 0.009254 | 0.857143 | 0.018311 | 21 | 0.519968 | 0.142857 | 0.325 |
| move_slot_4 | 0.000000 | 0.000000 | 0.000000 | 15 | 0.018588 | 1.000000 | 0.750 |
| sprint | 0.894260 | 0.994401 | 0.941676 | 2679 | 0.300573 | 0.005599 | 0.175 |

## Verdict

Prediction-only shadow integration is implemented but **not ready to enable**. The test split shows excessive move-slot false positives, impossible simultaneous slot predictions, and invalid-state attack predictions. Autonomous execution remains absent and prohibited.
