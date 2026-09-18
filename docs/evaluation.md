# Evaluation and calibration

Evaluate saved predictions against records matched by unique ID and exact candidate order. Missing, duplicate, or extra predictions are rejected.

All labeled sets report mean target cross-entropy and squared L2. If every target is one-hot, evaluation additionally reports accuracy, vector Brier, ten equal-width top-label reliability bins and ECE, and selective risk at several top-probability thresholds. Empty selected cohorts have null error rate, not zero error. Logarithms floor probabilities at 1e-12; distributions themselves are not clipped to inflate confidence.

Top-label ECE compares the maximum candidate probability to prediction correctness. It differs from binary positive-class ECE. Small data, bin choice, domain changes, and selection bias limit its interpretation. Keep full distributions and inspect reliability bins instead of reporting only one scalar.

`calibrate` searches a bounded logarithmic temperature grid using held-out calibration targets. It rescales saved probabilities; exact zero support remains zero. Generate those input predictions at temperature 1, fit once, then apply the result to a separate untouched test set. A probability rounded to zero has lost information that this tool cannot recover. A fitted temperature is not a calibration guarantee or a universal deployment threshold.

v0.1 checks train/dev group separation at training time. It cannot determine whether a user has passed test data to the calibration command. Dataset owners must enforce source-group separation and freeze test sets before experimenting.

## Release evidence

The offline tiny-model demo is a lifecycle test with synthetic, template-shared data. It supports a claim that code can train, serialize, reload, predict, and compute metrics. It does not support claims about language understanding, real-world calibration, beating Jev, or throughput improvement.

Pretrained-backbone and production metrics remain future work. Publish the dataset provenance, pinned backbone revision, seed, selected checkpoint, candidate counts, input lengths, device, and end-to-end measurements with any later result.
