# Qwen3-0.6B research training protocol

Status: the first real-backbone CE/Brier comparison is in progress. This document specifies the reproducible protocol; it does not claim completed training, released model weights, or measured capability. The scope is synthetic rules, lookup, game-state judgments, and exact random-draw probabilities.

## Inputs and preparation

Use the current LightJev source checkout and install `python -m pip install -e '.[dev]'` in a dedicated environment. Record the source commit, Python/package versions and GPU environment alongside results. The backbone is `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`.

```bash
python scripts/prepare_stage1.py
```

The preparation script downloads pinned NanoJev-Data stage1, verifies its SHA256, audits every record's CC0-1.0/programmatic declaration, and preserves existing split groups. It exports only programmatic gold targets, never Jev teacher probabilities. [Data provenance, hashes, exclusions and counts](data.md) are part of this protocol. Keep raw `runs/stage1-source` out of released artifacts. All runs must use the same frozen converted data.

## Two training arms

Run each arm from the same pretrained revision and seed, using a new output directory. Assign an available GPU through the process environment when using a shared host; do not modify other services.

```bash
lightjev train \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --train runs/stage1-data/train.jsonl --dev runs/stage1-data/dev.jsonl \
  --output-dir runs/qwen-stage1-ce \
  --loss ce --head-steps 20 --steps 300 \
  --batch-size 8 --grad-accum-steps 2 --seed 17 \
  --lr 2e-5 --head-lr 2e-4 --head-warmup-lr 1e-3 \
  --eval-every 50 --selection-metric ce \
  --max-length 256 --precision bf16 --gradient-checkpointing --device cuda
```

Repeat the identical command with `--loss brier` and `--output-dir runs/qwen-stage1-brier`. These are independent starts, not sequential fine-tuning. Both arms use effective batches of 16 questions per optimizer update, 20 head-only updates followed by 300 full-model updates. Head warmup learning rate is 1e-3; full-stage backbone/head learning rates are 2e-5/2e-4. Both select their saved checkpoint by development cross-entropy, checked at the configured evaluation intervals. This shared selection rule avoids choosing each arm using its final test outcome.

The maximum encoded candidate sequence observed in the prepared dataset was 137 tokens under the pinned Qwen tokenizer and current prompt formatting; the limit is 256. Inputs exceeding the configured limit are errors, not silently truncated. Changing tokenizer or prompt formatting requires checking lengths again.

## Evaluation and temperature fitting

```bash
python scripts/evaluate_release.py \
  --checkpoint runs/qwen-stage1-ce \
  --data-dir runs/stage1-data --output-dir runs/qwen-stage1-ce-evaluation \
  --device cuda --batch-size 8
```

Repeat for the independently trained Brier checkpoint and a fresh evaluation directory. The evaluator loads the selected checkpoint, runs float32 inference, fits temperature on the calibration split only, and saves raw and temperature-scaled predictions and metrics. Report test and OOD without further fitting or selecting recipes using those results. Development metrics selected the checkpoint; calibration metrics selected temperature; neither is an untouched final test.

Report hard-label accuracy, Brier, reliability and selective risk separately from exact soft-distribution cross-entropy and squared L2. Include results by family, raw and calibrated distributions, the selected training step, artifact hashes and environment versions. Keep unsuccessful outcomes too. This single-seed experiment cannot establish general superiority over NanoJev, Jev, or other baselines.

## Release contents

Once verified, a release should contain the complete LightJev checkpoint and tokenizer, model/base revision and license references, exact training configuration, data transformation and attribution, environment records, and complete evaluation reports. State the custom LightJev loading API: these scoring weights are not an ordinary autoregressive chat checkpoint. Do not include credentials, source teacher payloads, or private host data. A model card should distinguish completed measurements from planned work and avoid treating softmax normalization or fitted temperature as a calibration guarantee.
