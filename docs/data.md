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
