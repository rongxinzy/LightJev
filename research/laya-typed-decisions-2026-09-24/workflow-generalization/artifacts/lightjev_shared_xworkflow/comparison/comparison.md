# LightJev versus Laya: workflow-held-out comparison (lightjev_shared_xworkflow)

Four folds, 4 LightJev seeds per fold. Every held-out case is scored once per model/seed; LightJev mean metrics are the arithmetic mean of runs, not a probability ensemble.

Exploratory: prior work inspected these workflow families and public-test results. The official public test split was not used here. LightJev inputs are run at 640 tokens, above its released 256-token training window. Case-bootstrap intervals condition on the 4 selected training seeds and do not estimate seed uncertainty.

| Model | Accuracy | Target CE ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| LightJev mean across seeds | 0.457 | 1.152 | 0.226 | 0.085 |
| Independent-candidate LightJev | 0.437 | 1.170 | 0.236 | 0.095 |
| laya_base | 0.361 | 1.570 | 0.419 | 0.269 |
| laya_rlcd | 0.453 | 1.160 | 0.228 | 0.083 |
| laya_soft_ce | 0.447 | 1.166 | 0.226 | 0.074 |

## Paired case-cluster bootstrap

Intervals resample cases within each held-out workflow (10,000 draws by default); the five questions on one state stay clustered.

| Comparison | Difference | 95% CI | P(LightJev better) |
|---|---:|---:|---:|
| lightjev_vs_laya_base | +0.096 | [+0.081, +0.111] | 1.000 |
| lightjev_vs_laya_rlcd | +0.004 | [-0.010, +0.019] | 0.718 |
| lightjev_vs_laya_soft_ce | +0.010 | [-0.005, +0.025] | 0.909 |
| shared_context_vs_independent_candidate | +0.034 | [+0.025, +0.043] | 1.000 |

## Seed and case bootstrap

This resamples both observed LightJev training seeds and held-out case clusters. Laya's own seed uncertainty is not sampled.

| Comparison | Difference | 95% CI | P(LightJev better) |
|---|---:|---:|---:|
| lightjev_vs_laya_base | +0.096 | [+0.070, +0.123] | 1.000 |
| lightjev_vs_laya_rlcd | +0.004 | [-0.021, +0.031] | 0.613 |
| lightjev_vs_laya_soft_ce | +0.010 | [-0.016, +0.037] | 0.760 |

## Accuracy by held-out workflow

| Workflow | LightJev (seed mean) | Laya RLCD | Laya soft-CE |
|---|---:|---:|---:|
| agent_trace_observability | 0.403 | 0.443 | 0.445 |
| customer_service | 0.446 | 0.539 | 0.495 |
| invoice_processing | 0.477 | 0.434 | 0.475 |
| security_incidents | 0.503 | 0.396 | 0.374 |

## Accuracy by training seed

| Seed | Macro accuracy |
|---:|---:|
| 31 | 0.450 |
| 47 | 0.492 |
| 73 | 0.458 |
| 97 | 0.429 |

Full distribution metrics and per-seed reports are in `comparison.json`.
