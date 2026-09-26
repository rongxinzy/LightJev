# LightJev versus Laya: workflow-held-out comparison

Four folds, two LightJev seeds per fold. Every held-out case is scored once per model/seed; two-seed LightJev metrics are the arithmetic mean of runs, not a probability ensemble.

Exploratory: prior work inspected these workflow families and public-test results. The official public test split was not used here. LightJev inputs are run at 640 tokens, above its released 256-token training window.

| Model | Accuracy |
|---|---:|
| LightJev mean across seeds | 0.437 |
| laya_base | 0.361 |
| laya_rlcd | 0.453 |
| laya_soft_ce | 0.447 |

## Paired case-cluster bootstrap

Intervals resample cases within each held-out workflow (10,000 draws by default); the five questions on one state stay clustered.

| Comparison | Difference | 95% CI | P(LightJev better) |
|---|---:|---:|---:|
| lightjev_vs_laya_base | +0.076 | [+0.059, +0.093] | 1.000 |
| lightjev_vs_laya_rlcd | -0.016 | [-0.031, -0.001] | 0.021 |
| lightjev_vs_laya_soft_ce | -0.010 | [-0.026, +0.006] | 0.117 |

## Accuracy by held-out workflow

| Workflow | LightJev (2-seed mean) | Laya RLCD | Laya soft-CE |
|---|---:|---:|---:|
| agent_trace_observability | 0.379 | 0.443 | 0.445 |
| customer_service | 0.483 | 0.539 | 0.495 |
| invoice_processing | 0.387 | 0.434 | 0.475 |
| security_incidents | 0.500 | 0.396 | 0.374 |

Full distribution metrics and per-seed reports are in `comparison.json`.
