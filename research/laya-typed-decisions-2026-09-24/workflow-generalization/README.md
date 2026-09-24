# Leave-one-workflow-out validation

This experiment estimates cross-workflow transfer for the four synthetic workflow families in the pinned `LocalLLaMA/typed-decisions` training split. It does **not** use the public test split. Because earlier exploratory work inspected aggregate test results from these same workflow families, treat this as exploratory evidence rather than a preregistered benchmark result.

For each fold, one entire workflow (300 cases, 1,500 typed decisions) is held out. The other three workflows each contribute 272 training, 14 development, and 14 calibration cases. Cases are split as units, so the five questions sharing one state never cross partitions. This gives 4,080 training decisions (divisible across eight DDP ranks), 210 development decisions, and 210 calibration decisions per fold. Four held-out folds cover the entire 1,200-case public training split once.

Each fold evaluates:

- The pinned Laya checkpoint with its published calibration.
- The same Laya checkpoint fine-tuned with RLCD + soft cross-entropy.
- A matched pure soft-cross-entropy fine-tuning control.
- LightJev-0.6B zero-shot as an additional generalist baseline (full inputs up to 640 tokens; this extends beyond its 256-token training window).

The fine-tuned models are calibrated only on the three source workflows' calibration cases. Reports will include raw and calibrated predictions for held-out cases, per-workflow/per-question-type metrics, proper-score metrics, and confidence calibration. Specialist results from the earlier official public test are not used as the workflow-held-out test set here.

Pinned references: dataset revision `c76749ec58bd8c3d2ea706b31c333a9059c38f90`; Laya checkpoint revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (SHA256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`). The released LightJev checkpoint revision is `b3d9a281885e3c315b12a22d3602fc627cdba3dd`.

`prepare_workflow_folds.py` creates the splits from the official train parquet. `run_all.sh` contains the eight-GPU execution sequence; set `EXPERIMENT_ROOT`, `LAYA_BASE_DIR`, `LIGHTJEV_CHECKPOINT`, `TRAIN_PARQUET`, and `PYTHON_ENV` to match the training host before running it. Weights remain in the training environment and are not committed; code, fold manifests, logs, prediction distributions, and aggregate reports are intended to be published after validation completes.
