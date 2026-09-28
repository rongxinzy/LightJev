# LightJev Typed Decisions v0.2 results

LightJev-0.6B-typed-decisions is a shared-context candidate-scoring model initialized from Qwen3-0.6B. This note records one final evaluation on the official public 400-case test partition of `LocalLLaMA/typed-decisions` (2,000 decisions). It is exploratory evidence, not a leaderboard submission or a claim of superiority over Laya.

## Training and selection

The release uses 960 training cases (4,800 decisions), 120 development cases (600 decisions), and a separate 120-case calibration partition (600 decisions), all drawn from the dataset's official 1,200-case training partition. Four training seeds (31, 47, 73, 97) each ran 300 updates. The checkpoint was selected by minimum development soft cross-entropy; seed 47 won with 0.83362. The public test partition was not used for checkpoint selection. The training procedure and split hashes are recorded in the Hugging Face model manifest.

The model uses shared-context listwise candidate scoring. Inference averages candidate probabilities across four deterministic candidate-order permutations and realigns them to the input order. Temperature scaling was fit on the separate calibration partition by question type. Applying those fitted temperatures slightly worsened held-out test cross-entropy, Brier score, and ECE, so the published inference defaults to identity temperatures (raw probabilities); fitted temperatures remain in the manifest for inspection.

## Test results

| Metric | Result |
|---|---:|
| Hard-label accuracy | 78.60% |
| Soft accuracy, all question types | 0.5178 |
| Target cross-entropy | 0.8497 |
| Brier score | 0.04492 |
| 15-bin ECE | 0.1578 |
| Choice accuracy | 73.33% |
| Boolean accuracy | 87.17% |
| Ordinal-score accuracy | 76.13% |

Workflow accuracy was 71.4% for agent trace observability, 80.4% for customer service, 84.8% for invoice processing, and 77.8% for security incidents. Detailed machine-readable raw and calibrated reports and predictions are included with the model release.

## Limits and comparison

This result uses one selected seed on a synthetic benchmark with four workflow families. Earlier experiments had already inspected aggregate results from these workflow families, so the evaluation was not preregistered. It does not establish real-world business decision quality or robust generalization to new domains. The reported accuracy is not directly comparable with Laya's published number: prompt formatting, sample accounting, and evaluation protocol differ. It should not be described as beating Laya or as an official leaderboard result.

The checkpoint is a finite-candidate scorer, not a free-form chat model. It requires LightJev's Python inference path; `vllm serve` and standard `AutoModelForCausalLM.generate()` are not supported or validated.

Model: [rongxinzy/LightJev-0.6B-typed-decisions](https://huggingface.co/rongxinzy/LightJev-0.6B-typed-decisions). Code: [LightJev v0.2.0](https://github.com/rongxinzy/LightJev/releases/tag/v0.2.0).
