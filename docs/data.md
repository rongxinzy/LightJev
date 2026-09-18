# Data contract

One JSON object per line:

```json
{"id":"ticket-1","group_id":"customer-thread-1","state":"The same order was charged twice.","question":"Which team should handle this ticket?","kind":"choice","candidates":["Billing: charges and refunds","Technical: product faults","Other or insufficient information"],"target":[1,0,0]}
```

- `id`: unique record identifier.
- `group_id`: original source group (document, thread, scene, patient, episode). Train/dev overlap is rejected. The producer must also keep calibration/test isolated; renamed IDs do not prevent semantic leakage.
- `state`, `question`: nonempty strings. Structured input should be serialized intentionally by the caller.
- `kind`: `choice`, `boolean`, or `score`.
- `candidates`: 2–255 unique, nonempty descriptions. Boolean requires exactly `["false", "true"]`. Score descriptions are ordered low to high.
- `target`: optional at inference; required for training/evaluation. Finite nonnegative probabilities, same length as candidates, summing to 1. A tolerance of 1e-6 permits floating-point export rounding, which is normalized on load.

A one-hot target can represent an observed label; a soft target can represent a teacher forecast, an exact simulator distribution, or annotated uncertainty. These are not interchangeable evidence. Keep their provenance in your dataset documentation. v0.1 does not automatically verify how a target was obtained.

IDs, groups, and target values never enter model input. Each candidate is encoded with the same question and state. Long input is rejected, never silently truncated. Training uses complete candidate sets; do not split one question across separately normalized losses.

Candidate descriptions should include meaning rather than opaque IDs. Include an explicit other/unknown option when the candidate set is not exhaustive. This creates a possible abstention outcome; it does not guarantee the model will select it correctly.

## Frozen Qwen training data

The first pretrained experiment uses [C-Tianyu / TianyuCodings NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data), revision `87061eb91e8fc687e9b046454afdcc5551e3eff7`, file `stage1/all.jsonl`. The 4,967,698-byte source has SHA256 `7294765b80e751fc5aee7aba906b28a8ea80d6f147253d3f6c3e4f491de0b2d7`.

All 2,312 source records were checked for `metadata.license=CC0-1.0` and `metadata.source=self_authored_programmatic`. This is a per-record declaration; the upstream repository has no blanket license declaration. Attribute the original data to C-Tianyu / TianyuCodings, retain that provenance, and identify our modification as one-question-per-record conversion and target-kind filtering. LightJev's Apache-2.0 code license does not replace the data declaration.

`python scripts/prepare_stage1.py` verifies the pinned bytes, preserves source split/group assignments, and whitelists inputs, descriptions and programmatic `gold_probs`. All `teacher` content is excluded. The raw download goes into ignored `runs/stage1-source`; it contains upstream teacher fields and must not be included in a release. Only converted data in `runs/stage1-data` is a release input.

Accepted targets are `deterministic_truth` (4,264 questions) and `programmatic_conditional_distribution` (1,080 questions). The latter are exact bag-draw probabilities, not teacher forecasts. The fixed metadata policy excludes 1,160 `optimal_action_policy` questions and 432 questions missing a target-kind declaration. No evaluation results were used to choose these exclusions. The manifest records every excluded ID.

| Split | Questions |
|---|---:|
| train | 3,200 |
| dev | 424 |
| calibration | 424 |
| test | 848 |
| ood | 448 |

Training families are smart-home rules (600), catalog lookup (480), known-chance probability (600), grid-navigation state judgments (800), and tic-tac-toe state judgments (720). The training primitives are Choice (560), Boolean (1,320), and Score (1,320). Navigation/action-policy questions were excluded; remaining game questions must not be described as trained action planning.

The audit found zero cross-split source groups or state IDs, zero conflicting normalized model inputs, and zero cross-split duplicate model inputs. Before evaluation, the deduplication policy fixed split precedence as train/dev/calibration/test/ood: conflicting targets fail; identical inputs would be removed from later splits and logged. None required removal. All Boolean questions had no extra criteria; their original instructions are preserved.

Converted manifest SHA256: `d22a3481048d79730f86d208cd29bde03d42f15ae1a01f89bdfee312dabb53ca`.
Converted train SHA256: `2c91b4869398373ac7575fcdd0159df28a79d2a01bdb39a3a84c19e8c15778ad`.
The manifest includes all split hashes and counts. These synthetic tasks support a bounded research experiment, not a claim of broad business-domain competence.
