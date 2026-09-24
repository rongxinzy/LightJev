# Laya typed-decisions experiments

This study reproduces the Laya RLCD fine-tuning idea and runs a soft-CE control on the public [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) benchmark. It reserves full cases from the train split for development and calibration, so questions sharing one state cannot cross partitions. The public 400-case test split is excluded from training and checkpoint selection.

The official dataset README says to score the public test split with full probability distributions and post the metrics plus mode (`specialist` or `generalist`) in a dataset Discussion. It is a public community leaderboard rather than a hidden-test contest. As of 2026-09-24, the table lists generalist meraGPT Decider 1 at 0.768 and TypeSafe Jev at 0.727; the Laya card reports 0.766 as a specialist. Specialist and generalist scores are explicitly not comparable.

## Files

- `manifest.json`: pinned model, data, source revisions and SHA256 values.
- `prepare_data.py`: parquet validation, workflow-stratified case splits, Laya tokenization and overlap checks.
- `train_rlcd_ddp.py`: BF16 DDP RLCD + soft cross-entropy; set `--rl-weight 0` for the pure soft-CE control. Both runs use case-held-out dev selection and separate calibration.
- `evaluate_benchmark.py`: Laya Agent evaluation plus full-distribution metrics, per-workflow/per-type scores, labels and predictions.
- `evaluate_lightjev.py`: LightJev's generalist zero-shot evaluation; an explicit 640-token cap preserves the entire benchmark input without truncation, although that exceeds the released checkpoint's 256-token training window.
- `data/`: pinned source cases and workflow-stratified train/dev/calibration/test splits.
- `runs/`: selected-checkpoint metadata, calibration and training logs. The two safetensors checkpoints (~804 MB each) are not included in this GitHub branch; their SHA256 values are recorded in the selected metadata files.
- `results/`: full test-set prediction distributions and JSON metric reports.

RLCD follows the Laya notebook: four centered Gaussian logit perturbations, proper-score rewards (log, spherical, ordinal RPS), group-centered normalized advantage, Gaussian policy-gradient loss and soft CE. The primary RLCD checkpoint was frozen by dev NLL before test evaluation. A pure-CE control was added after seeing that test result; its score is exploratory, not an independent benchmark claim.

See [REPORT.zh-CN.md](REPORT.zh-CN.md) for the benchmark rules, training details, results and caveats. The soft-CE control was started after inspecting the first run's public-test result, so its test score is exploratory and should not be treated as an independent model-selection result or a clean leaderboard claim. Model weights have not been published to Hugging Face.
