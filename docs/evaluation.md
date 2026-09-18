# Evaluation and calibration

Evaluate saved predictions against records matched by unique ID and exact candidate order. Missing, duplicate, or extra predictions are rejected.

All labeled sets report mean target cross-entropy and squared L2. If every target is one-hot, evaluation additionally reports accuracy, vector Brier, ten equal-width top-label reliability bins and ECE, and selective risk at several top-probability thresholds. Empty selected cohorts have null error rate, not zero error. Logarithms floor probabilities at 1e-12; distributions themselves are not clipped to inflate confidence.

Top-label ECE compares the maximum candidate probability to prediction correctness. It differs from binary positive-class ECE. Small data, bin choice, domain changes, and selection bias limit its interpretation. Keep full distributions and inspect reliability bins instead of reporting only one scalar.

`calibrate` searches a bounded logarithmic temperature grid using held-out calibration targets. It rescales saved probabilities; exact zero support remains zero. Generate those input predictions at temperature 1, fit once, then apply the result to a separate untouched test set. A probability rounded to zero has lost information that this tool cannot recover. A fitted temperature is not a calibration guarantee or a universal deployment threshold.

v0.1 checks train/dev group separation at training time. It cannot determine whether a user has passed test data to the calibration command. Dataset owners must enforce source-group separation and freeze test sets before experimenting.

## Release evidence

The offline tiny-model demo is a lifecycle test with synthetic, template-shared data. It supports a claim that code can train, serialize, reload, predict, and compute metrics. It does not support claims about language understanding, real-world calibration, beating Jev, or throughput improvement.

The real Qwen3-0.6B CE/Brier experiment is complete; [results-v0.1.md](results-v0.1.md) records both arms and weak task families. Production metrics are not claimed. Its frozen protocol is in [training-release.md](training-release.md). `scripts/evaluate_release.py` evaluates the selected checkpoint, fits temperature only on calibration, and writes raw and temperature-scaled results for dev, calibration, test, and OOD. It reports hard-label and exact soft-distribution subsets separately, plus task-family summaries. Hard-label accuracy and reliability concern deterministic outcomes; conditional-distribution CE and squared L2 concern matching a known random process. Do not interpret soft-target argmax as an observed correct outcome.

The upstream OOD split is a synthetic generator holdout, not evidence of arbitrary real-world distribution shift. This run uses one seed and no matched NanoJev checkpoint comparison; neither superiority nor universal calibration follows from it. Publish the dataset provenance, pinned backbone revision, seed, selected checkpoint, candidate counts, input lengths, device, and measurements with the results.

The published arm remains CE as chosen by development CE, although Brier is better on some final test metrics. The selected CE model's fitted temperature 0.8187307530779818 worsened held-out cross-entropy and top-label ECE; raw probabilities remain the default. A calibration fit is an experiment, not an automatic improvement.
