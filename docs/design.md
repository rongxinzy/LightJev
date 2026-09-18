# Model and training design

For question x and candidate description c_i, the backbone encodes the complete text path. Take the final active token representation h_i and compute:

```
z_i = Linear(LayerNorm(h_i))
p_i = softmax(z)_i
```

One head is shared across candidates and questions. It does not use the language-model vocabulary head. Current candidate encodings are independent: there is no learned set attention or cross-field constraint solver. Probability normalization introduces competition but does not give the backbone the other candidate descriptions. This is a deliberately small baseline with room for candidate-set interaction research.

In evaluation mode, permuting Choice candidates permutes their scores, except for numerical effects; exact ties select the first maximum. Score order has explicit ordinal meaning and should not be randomized without remapping the labels and output scale.

Full backbone and head parameters are updated with question-mean losses:

- CE: `-sum_i target_i * log_softmax(z)_i`.
- Brier: `sum_i (softmax(z)_i - target_i)^2`.

For an observed one-hot target, vector Brier is twice the scalar Bernoulli Brier in the binary case. For a soft target, the same expression measures target distribution L2; it does not establish event calibration.

Train/dev source groups and IDs cannot overlap. The dev objective selects checkpoints. Optimizer state is not a resumable training artifact in v0.1; saved weights are for inference and analysis. The framework does not automatically launch jobs, call teachers, or use remote machines.

Each candidate currently repeats context computation. A single batched Python forward is not O(1) compute, shared-prefix attention, or a measured latency win. Shared-prefix inference should be accepted only after both numerical-equivalence and real end-to-end benchmarks.

## Relationship to prior work

Jev motivates typed probabilistic decisions. NanoJev shows a trainable decision-head approach with broader experiments. LightJev starts with independently written, small modules for data validation, training, inference, and metrics. It does not import their implementation, weight files, or evaluation data.

An RL-based extension is future work. Direct CE/Brier are explicit controls; introducing a sampled reward is not by itself evidence of better calibration.
