# Paired CE versus RLCD objective comparison

Four held-out workflows and 4 matched seeds. Each seed scores the same 300 case clusters per fold.
Case-only intervals condition on the observed seeds. Seed-and-case intervals resample matched training seeds and case clusters within workflow.

| Arm | Accuracy | Soft accuracy | Target CE ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| CE | 0.4617 | 0.3562 | 1.1545 | 0.2274 | 0.0796 |
| CE + RLCD | 0.4655 | 0.3570 | 1.1624 | 0.2305 | 0.0787 |

RLCD minus CE accuracy: +0.0037 (+0.37 percentage points).
Case bootstrap 95% CI: [-0.0042, +0.0117]; P(RLCD > CE)=0.821.
Seed-and-case bootstrap 95% CI: [-0.0174, +0.0235]; P(RLCD > CE)=0.652.

| Workflow | CE accuracy | CE+RLCD accuracy | Difference |
|---|---:|---:|---:|
| agent_trace_observability | 0.4177 | 0.4263 | +0.0087 |
| customer_service | 0.4260 | 0.4363 | +0.0103 |
| invoice_processing | 0.4983 | 0.4898 | -0.0085 |
| security_incidents | 0.5050 | 0.5095 | +0.0045 |

## Macro accuracy by training seed

| Seed | CE | CE + RLCD |
|---:|---:|---:|
| 31 | 0.4538 | 0.4763 |
| 47 | 0.4657 | 0.4405 |
| 73 | 0.4693 | 0.4722 |
| 97 | 0.4582 | 0.4730 |

All metrics are measured on the same held-out decisions after each arm's source-only per-type temperature fit.
