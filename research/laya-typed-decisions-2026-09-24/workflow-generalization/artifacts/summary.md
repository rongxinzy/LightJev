# Workflow-held-out generalization results

Four-fold leave-one-workflow-out evaluation on the pinned public train split. Each row below tests all 300 cases from one workflow excluded from training, development, and calibration.

Exploratory result: this workflow family and dataset have been inspected in prior runs. The official public test split was not used in this experiment.

Accuracy and probability metrics are weighted by decisions across all 1,200 held-out cases (6,000 decisions); per-workflow rows show the held-out fold scores.

| Model | Accuracy | Soft acc. | KL | Brier | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| laya_base_calibrated | 0.361 | 0.330 | 0.596 | 0.331 | 1.344 | 0.174 |
| laya_base_raw | 0.361 | 0.329 | 0.821 | 0.419 | 1.570 | 0.265 |
| rlcd_calibrated | 0.453 | 0.356 | 0.408 | 0.226 | 1.156 | 0.025 |
| rlcd_raw | 0.453 | 0.357 | 0.412 | 0.228 | 1.160 | 0.023 |
| soft_ce_calibrated | 0.447 | 0.357 | 0.416 | 0.226 | 1.164 | 0.035 |
| soft_ce_raw | 0.447 | 0.357 | 0.417 | 0.226 | 1.166 | 0.034 |
| lightjev_zero_shot | 0.357 | 0.323 | 0.495 | 0.269 | 1.243 | 0.072 |

## Accuracy by held-out workflow

| Model | agent_trace_observability | customer_service | invoice_processing | security_incidents |
|---|---:|---:|---:|---:|
| laya_base_calibrated | 0.345 | 0.401 | 0.369 | 0.329 |
| laya_base_raw | 0.345 | 0.401 | 0.369 | 0.329 |
| rlcd_calibrated | 0.443 | 0.539 | 0.434 | 0.396 |
| rlcd_raw | 0.443 | 0.539 | 0.434 | 0.396 |
| soft_ce_calibrated | 0.445 | 0.495 | 0.475 | 0.374 |
| soft_ce_raw | 0.445 | 0.495 | 0.475 | 0.374 |
| lightjev_zero_shot | 0.267 | 0.403 | 0.308 | 0.449 |

## Accuracy and distribution metrics by question type

Full per-type metrics are in `summary.json` under `workflow_macro_metrics.*.by_type_metrics`.

