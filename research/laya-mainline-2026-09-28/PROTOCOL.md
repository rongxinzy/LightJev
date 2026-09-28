# LightJev: Laya mainline, controlled rule-following pilot

Status: protocol fixed before training. This replaces Qwen recipe search as the research mainline; existing releases and historical results remain available.

## Question and scope

Does CE adaptation of the ordinary Laya checkpoint improve **joint correctness on rule and fact counterfactuals**, including two rule families absent from adaptation training?

This first pilot is a synthetic mechanism diagnostic. It is not a business-workflow benchmark, a multilingual claim, or evidence of beating Laya generally. Different industry names do not count as independent families. No official typed-decisions public test data enters this study.

## Pinned starting point

- Weights: `convaiinnovations/laya`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
- Weight SHA256: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
- Runtime source: `NandhaKishorM/laya`, commit `23a17522aa4942da6cce53a995a275760320b691`.
- Ordinary English checkpoint only; no automatic router or typed-decisions specialist.
- Native question format and candidate markers; raw logits for CE and primary evaluation.
- 512 total tokens / 192 head tokens. Reject any sample that would be truncated, including individual option truncation. Validate exact native token sequence against an untruncated reference.
- Upstream is Apache-2.0. Derived releases must retain attribution and license; the initial checkpoint's pretraining data coverage is not fully independently established.

## Data and partitions

Four training rule families: numeric threshold, category membership, conjunction, ordinal bands. Two development-only families: disjunction and veto. Two locked test-only families: exclusive-or and weighted ordinal bands.

Each group contains a base example, a policy change with a changed gold answer, a fact change with a changed gold answer, and an irrelevant-field change with the same answer. Every answer is recomputed by an executable oracle. All variants stay together. These are held-out **post-training rule generators**, not a claim that the pretrained model has never encountered the corresponding concepts.

- Train: 96 groups per training family (1,536 decisions).
- Development, seen families: 16 groups per training family (256 decisions).
- Development, new families: 32 groups per development-only family (256 decisions).
- Calibration: 12 groups per train/development family (288 decisions), separate from optimization and checkpoint selection.
- Locked test: 48 groups per test-only family (384 decisions). Do not run during the pilot.

The generator seed, per-file hashes, family assignments, group IDs, label counts and code hashes must be recorded before execution. Raw states and exact rendered input/label associations are checked for split overlap. Contrast groups—not decision rows—are the smallest independent evaluation unit. There are only two held-out development mechanisms, so this pilot cannot estimate broad domain generalization.

## Required gates

1. Independent boundary examples verify oracle predicates and ordinal boundaries.
2. Generated counterfactuals are re-evaluated; both changes really flip answers, irrelevant changes do not.
3. Native tokenizer validation preserves all instructions, criteria and facts. Marker count/order and target order match for choice, noul and score.
4. Untrained Laya establishes raw development results before training.
5. A tiny fit-and-reload smoke test verifies gradients and checkpoint equivalence before the pilot.

## Fixed first experiment

One CE seed (31), 128 optimizer updates, effective batch 64; full-parameter FP32 master weights/Adam states with BF16 CUDA autocast. Encoder LR 1e-5, head LR 5e-5, 8-step linear warmup followed by cosine decay, weight decay 0.01, gradient norm 1.0. No RLCD and no ordinal reward. Keep score candidates in canonical semantic order.

Evaluate at step 0 and every 32 updates. Select solely by mean of per-family raw NLL over the six development families; retain the untouched base when training does not improve this criterion. Save only the current selected weights. The fixed budget is a pilot, not an optimal training prescription.

Report seen and new-family accuracy, macro NLL, Brier, ECE, policy-pair both-correct rate, fact-pair both-correct rate, irrelevant-perturbation semantic consistency and both-correct rate, and per-family prediction distributions. Do not count answer changes alone as success. Run candidate-order and input-ablation diagnostics on development only. Temperature fitting is separate and cannot change hard argmax accuracy.

## Continue / stop rules

Promising enough for a second seed: new-family average policy/fact both-correct rate improves by at least 5 percentage points over base, while seen-family accuracy falls by no more than 2 points and irrelevant-perturbation consistency falls by no more than 2 points in either development split. These are resource-allocation thresholds, not statistical significance claims.

If the gate fails, inspect saved errors and token snapshots; do not automatically extend steps or launch a seed grid. Record a structural explanation or a specific unresolved hypothesis before the next experiment. The locked test is enabled only after the method and checkpoint selection are frozen in a separate selection record; any later use for tuning retires its final-test status.

## Subsequent scope

Successful mechanism diagnostics justify collecting diverse, independently specified workflows and rules. Preserve the old four-workflow evaluation as auxiliary regression only. Product or leaderboard claims require new task-family evaluation, provenance review, repeated training and latency/calibration measurements.
