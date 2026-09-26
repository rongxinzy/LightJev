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

## LightJev fine-tuned on the same held-out folds

To compare the Qwen3-0.6B LightJev candidate scorer directly with the Laya fine-tune arms, we trained two LightJev seeds (31 and 47) on each of the same four workflow folds. Each run started from the released LightJev checkpoint, used soft-target cross-entropy, BF16 autocast with FP32 master weights, 256 optimizer steps, an effective batch of 64, cosine decay with warmup, and dev soft cross-entropy checkpoint selection. This corresponds to about four passes over the 4,080 training decisions by sample count (random sampling with replacement). The eight runs were independent single-GPU jobs on eight RTX 4060 Ti GPUs. Each used about 11.2 GiB peak GPU memory.

Per-type temperature scaling was fit only on the fold's 210 source-workflow calibration questions. The held-out workflow's 1,500 decisions were used only for final evaluation. We used a 640-token non-truncating input limit to preserve these cases; the released LightJev checkpoint was originally trained to 256 tokens, so this fine-tune continues it on longer inputs. Treat this as a context-extension treatment, not a clean comparison of the original 256-token model.

The two-seed mean accuracy is **0.4373**, compared with 0.4530 for the Laya RLCD arm, 0.4472 for the Laya soft-CE arm, and 0.3612 for the Laya base. LightJev beats the Laya base by 7.61 percentage points in this protocol. Against Laya RLCD, the paired case-cluster bootstrap difference is −1.58 points (95% interval −3.11 to −0.05); against Laya soft-CE it is −0.99 points (95% interval −2.59 to +0.61). These intervals resample cases within each workflow and are conditional on these two LightJev seeds; two seeds are not enough to estimate training-seed uncertainty reliably.

LightJev wins on the held-out `security_incidents` workflow (0.500 across seeds, versus 0.396 for Laya RLCD and 0.374 for Laya soft-CE), but trails Laya on the other three workflows. Seed-level overall accuracy is 0.4423 (seed 31) and 0.4322 (seed 47), showing nontrivial run variation. The current evidence therefore supports “LightJev fine-tuning nearly matches Laya soft-CE and beats the Laya base on these held-out folds,” not “LightJev surpasses Laya overall.”

The detailed paired comparison is [`artifacts/lightjev_xworkflow/comparison/comparison.md`](artifacts/lightjev_xworkflow/comparison/comparison.md); machine-readable bootstrap and metric output is [`comparison.json`](artifacts/lightjev_xworkflow/comparison/comparison.json). All raw/calibrated predictions and per-seed reports are under [`artifacts/results/lightjev_xworkflow/`](artifacts/results/lightjev_xworkflow/). Training manifests, logs, and selected-weight SHA256 hashes are under [`artifacts/lightjev_xworkflow/`](artifacts/lightjev_xworkflow/). The two per-seed predictions are repeated evaluations of the same cases, not an ensemble; their mean is not a deployable combined model.

The next model change should target the main structural difference: LightJev scores each candidate as a separate full encoder input, while Laya places option markers in a shared sequence and compares candidates jointly. Before another RLCD sweep, build a shared-context LightJev head (encode state/question once, score all candidate descriptions against that representation), then rerun the same folds and seeds. Keep cross-workflow data augmentation and any RLCD changes as separate ablations so a gain can be attributed to the architecture.

Reproduction entry points for this LightJev comparison are `prepare_lightjev_records.py`, `train_lightjev_fold.py`, `run_lightjev_all.sh`, `evaluate_lightjev_calibrated.py`, `evaluate_lightjev_all.sh`, and `compare_lightjev_to_laya.py`. The runners default to the agc8f-6 experiment paths used here; set `STUDY_ROOT`, `LIGHTJEV_CHECKPOINT`, `PYTHON_ENV`, and `LIGHTJEV_SRC` to override them. Training outputs are intentionally not committed as full model weights; hashes, selected-step manifests, and logs are included.
