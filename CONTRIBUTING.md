# Contributing

Use a branch and submit a pull request. Install `.[dev]`, run `python -m pytest -q`, and build with `python -m build`. Tests and the tiny demo must run offline after dependencies are installed.

For model changes, preserve complete-candidate normalization and add meaningful checks for gradients, candidate order, serialization, and input leakage. For benchmark claims, provide fixed data provenance, held-out splits, full probabilities, a pinned model revision, and device/timing details. Do not commit credentials, private datasets, or checkpoints.
