# Independent preflight review

An independent read-only reviewer checked the protocol, data generator, oracle, CE trainer, and diagnostic evaluator before the pilot results were inspected.

Findings addressed:

- A mean-NLL-only tiny-fit gate could pass a model with one wrong example. The gate now requires every selected-checkpoint prediction to be correct, all three primitives to be represented, and NLL below 0.15. Early stopping checks the selected checkpoint as well. Exact checkpoint round-trip validation requires identical decisions and maximum logit difference <= 1e-5.
- The declared effective batch is enforced at 64 for the pilot. Starting weight, runtime code, encoder configuration and tokenizer JSON hashes are checked.
- Diagnostic prediction rows are retained. Candidate permutation is labeled as a semantic-equivalence control; missing-state and missing-criteria modes are labeled as corrupted-input controls.

Independent checks found agreement between a separate predicate implementation and 2,304 temporary train/development/calibration-family records. Real-tokenizer plus fake-logit checks verified canonical remapping for choice, noul and score, and confirmed missing-input controls rebuild the actual input tokens. DDP rank slicing, accumulation scaling and synchronization were inspected statically. No test-family samples or model predictions were used by this reviewer.

Local validation: 21 mainline tests including tokenizer integration; 52 existing package tests. GPU-host input parity: 2,048 prepared training/development sequences exactly matched freshly generated native Laya sequences. Real distributed training and reload results are recorded in run artifacts, not inferred from this code review.
