# Leave-one-workflow-out validation

This experiment estimates cross-workflow transfer for the four synthetic workflow families in the pinned `LocalLLaMA/typed-decisions` training split. It does **not** use the public test split. Because earlier exploratory work inspected aggregate test results from these same workflow families, treat this as exploratory evidence rather than a preregistered benchmark result.

For each fold, one entire workflow (300 cases, 1,500 typed decisions) is held out. The other three workflows each contribute 272 training, 14 development, and 14 calibration cases. Cases are split as units, so the five questions sharing one state never cross partitions. This gives 4,080 training decisions (divisible across eight DDP ranks), 210 development decisions, and 210 calibration decisions per fold. Four held-out folds cover the entire 1,200-case public training split once.

Each fold evaluates:

- The pinned Laya checkpoint with its published calibration.
- The same Laya checkpoint fine-tuned with RLCD + soft cross-entropy.
- A matched pure soft-cross-entropy fine-tuning control.
- LightJev-0.6B zero-shot as an additional generalist baseline (full inputs up to 640 tokens; this extends beyond its 256-token training window).

The fine-tuned models are calibrated only on the three source workflows' calibration cases. The completed run includes raw and calibrated predictions for held-out cases, per-workflow/per-question-type metrics, proper-score metrics, and confidence calibration. Specialist results from the earlier official public test are not used as the workflow-held-out test set here.

Pinned references: dataset revision `c76749ec58bd8c3d2ea706b31c333a9059c38f90`; Laya checkpoint revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (SHA256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`). The released LightJev checkpoint revision is `b3d9a281885e3c315b12a22d3602fc627cdba3dd`.

## Results

The complete aggregate report is [`artifacts/summary.md`](artifacts/summary.md), with machine-readable metrics in [`artifacts/summary.json`](artifacts/summary.json). All seven model arms have raw held-out prediction distributions under `artifacts/results/`; fold manifests and selected-checkpoint metadata are included for provenance. Trained weights are retained on the training host and are not committed.

Weighted across all 6,000 held-out decisions, accuracy is 45.3% for RLCD + soft-CE, 44.7% for the matched soft-CE control, 36.1% for the pinned Laya base, and 35.7% for LightJev zero-shot. RLCD leads the control overall by 0.6 percentage points, but the result varies by workflow: RLCD wins on customer service and security incidents, while soft-CE wins narrowly on agent trace observability and more clearly on invoice processing. This small margin does not establish a robust RLCD advantage. Both fine-tuning arms exceed the Laya base overall; LightJev is strongest only on the held-out security workflow among these folds.

The experiment used eight RTX 4060 Ti GPUs with BF16 DDP, effective batch 64, and four epochs per fold and training arm. Peak reported memory was about 9.35 GiB per GPU. This is an exploratory four-workflow result on the public training split, not an official leaderboard score or evidence of transfer to arbitrary real-world workflows. The LightJev comparison also evaluates inputs up to 640 tokens despite its 256-token training window, so it is an out-of-window baseline.

`prepare_workflow_folds.py` creates the splits from the official train parquet. `run_all.sh` contains the eight-GPU execution sequence; set `EXPERIMENT_ROOT`, `LAYA_BASE_DIR`, `LIGHTJEV_CHECKPOINT`, `TRAIN_PARQUET`, and `PYTHON_ENV` to match the training host before running it.
