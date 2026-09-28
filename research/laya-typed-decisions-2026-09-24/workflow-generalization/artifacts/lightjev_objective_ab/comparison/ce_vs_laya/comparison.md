# LightJev versus Laya: workflow-held-out comparison (lightjev_objective_ab/ce)

Four folds, 4 LightJev seeds per fold. Every held-out case is scored once per model/seed; LightJev mean metrics are the arithmetic mean of runs, not a probability ensemble.

Exploratory: prior work inspected these workflow families and public-test results. The official public test split was not used here. LightJev inputs are run at 640 tokens, above its released 256-token training window. Case-bootstrap intervals condition on the 4 selected training seeds and do not estimate seed uncertainty.

| Model | Accuracy | Target CE ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| LightJev mean across seeds | 0.462 | 1.155 | 0.227 | 0.080 |
| laya_base | 0.361 | 1.570 | 0.419 | 0.269 |
| laya_rlcd | 0.453 | 1.160 | 0.228 | 0.083 |
| laya_soft_ce | 0.447 | 1.166 | 0.226 | 0.074 |

## Paired case-cluster bootstrap

Intervals resample cases within each held-out workflow (10,000 draws by default); the five questions on one state stay clustered.

| Comparison | Difference | 95% CI | P(LightJev better) |
|---|---:|---:|---:|
| lightjev_vs_laya_base | +0.101 | [+0.084, +0.117] | 1.000 |
| lightjev_vs_laya_rlcd | +0.009 | [-0.006, +0.024] | 0.879 |
| lightjev_vs_laya_soft_ce | +0.015 | [-0.001, +0.030] | 0.970 |

## Seed and case bootstrap

This resamples both observed LightJev training seeds and held-out case clusters. Laya's own seed uncertainty is not sampled.

| Comparison | Difference | 95% CI | P(LightJev better) |
|---|---:|---:|---:|
| lightjev_vs_laya_base | +0.101 | [+0.083, +0.118] | 1.000 |
| lightjev_vs_laya_rlcd | +0.009 | [-0.008, +0.025] | 0.855 |
| lightjev_vs_laya_soft_ce | +0.015 | [-0.002, +0.031] | 0.954 |

## Accuracy by held-out workflow

| Workflow | LightJev (seed mean) | Laya RLCD | Laya soft-CE |
|---|---:|---:|---:|
| agent_trace_observability | 0.418 | 0.443 | 0.445 |
| customer_service | 0.426 | 0.539 | 0.495 |
| invoice_processing | 0.498 | 0.434 | 0.475 |
| security_incidents | 0.505 | 0.396 | 0.374 |

## Accuracy by training seed

| Seed | Macro accuracy |
|---:|---:|
| 31 | 0.454 |
| 47 | 0.466 |
| 73 | 0.469 |
| 97 | 0.458 |

Full distribution metrics and per-seed reports are in `comparison.json`.
