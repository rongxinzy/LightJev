# LightJev versus Laya: workflow-held-out comparison (lightjev_objective_ab/rlcd)

Four folds, 4 LightJev seeds per fold. Every held-out case is scored once per model/seed; LightJev mean metrics are the arithmetic mean of runs, not a probability ensemble.

Exploratory: prior work inspected these workflow families and public-test results. The official public test split was not used here. LightJev inputs are run at 640 tokens, above its released 256-token training window. Case-bootstrap intervals condition on the 4 selected training seeds and do not estimate seed uncertainty.

| Model | Accuracy | Target CE ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| LightJev mean across seeds | 0.466 | 1.162 | 0.230 | 0.079 |
| laya_base | 0.361 | 1.570 | 0.419 | 0.269 |
| laya_rlcd | 0.453 | 1.160 | 0.228 | 0.083 |
| laya_soft_ce | 0.447 | 1.166 | 0.226 | 0.074 |

## Paired case-cluster bootstrap

Intervals resample cases within each held-out workflow (10,000 draws by default); the five questions on one state stay clustered.

| Comparison | Difference | 95% CI | P(LightJev better) |
|---|---:|---:|---:|
| lightjev_vs_laya_base | +0.104 | [+0.088, +0.120] | 1.000 |
| lightjev_vs_laya_rlcd | +0.013 | [-0.002, +0.027] | 0.950 |
| lightjev_vs_laya_soft_ce | +0.018 | [+0.003, +0.034] | 0.990 |

## Seed and case bootstrap

This resamples both observed LightJev training seeds and held-out case clusters. Laya's own seed uncertainty is not sampled.

| Comparison | Difference | 95% CI | P(LightJev better) |
|---|---:|---:|---:|
| lightjev_vs_laya_base | +0.104 | [+0.082, +0.125] | 1.000 |
| lightjev_vs_laya_rlcd | +0.013 | [-0.009, +0.032] | 0.879 |
| lightjev_vs_laya_soft_ce | +0.018 | [-0.003, +0.039] | 0.948 |

## Accuracy by held-out workflow

| Workflow | LightJev (seed mean) | Laya RLCD | Laya soft-CE |
|---|---:|---:|---:|
| agent_trace_observability | 0.426 | 0.443 | 0.445 |
| customer_service | 0.436 | 0.539 | 0.495 |
| invoice_processing | 0.490 | 0.434 | 0.475 |
| security_incidents | 0.509 | 0.396 | 0.374 |

## Accuracy by training seed

| Seed | Macro accuracy |
|---:|---:|
| 31 | 0.476 |
| 47 | 0.441 |
| 73 | 0.472 |
| 97 | 0.473 |

Full distribution metrics and per-seed reports are in `comparison.json`.
