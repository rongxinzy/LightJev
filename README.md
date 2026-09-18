# LightJev

**Train small language backbones to make typed decisions.**

[![CI](https://github.com/rongxinzy/LightJev/actions/workflows/ci.yml/badge.svg)](https://github.com/rongxinzy/LightJev/actions/workflows/ci.yml)
[简体中文](README.zh-CN.md) · [Data format](docs/data.md) · [Design](docs/design.md) · [Evaluation](docs/evaluation.md)

LightJev turns a text backbone into a finite-candidate decision model. Supply context, a question, and candidate descriptions; receive a normalized distribution and a selected value. Train the backbone and a small scoring head with cross-entropy or Brier loss.

**v0.1 is a research toolkit release.** It includes an executable offline training demo, not a broadly trained decision checkpoint. The demo verifies the training and inference pipeline; it is not a capability benchmark.

## Quick start — no model download

```bash
git clone https://github.com/rongxinzy/LightJev.git
cd LightJev
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
lightjev demo --output-dir runs/demo --steps 8
```

The demo creates a tiny random BERT backbone and a local tokenizer, trains a decision head and backbone on synthetic sensor records, saves the selected checkpoint, reloads it, and evaluates held-out records. It makes no Hub or paid API calls after dependencies are installed. The sensor templates overlap across splits: this is a plumbing check, not evidence of generalization.

Outputs include `runs/demo/report.json`, `predictions.json`, and a reusable checkpoint directory. Run with a fresh output directory to preserve previous results.

## What is implemented

| Component | v0.1 |
|---|---|
| Choice / Boolean / ordinal Score | Dynamic candidate descriptions, 2–255 candidates |
| Model | Text backbone + shared LayerNorm / scalar decision head |
| Training | Full-parameter CE or vector Brier, question-level averaging |
| Data checks | Finite normalized targets, no train/dev group or ID overlap, no silent truncation |
| Checkpoints | Local configuration, tokenizer, safe tensor weights, training provenance |
| Evaluation | Target cross-entropy/L2; hard-label accuracy, Brier, reliability bins, selective risk |
| Calibration | Optional temperature fit on a separate calibration set |
| Offline demo | Random tiny backbone; train → save → reload → predict → evaluate |

There is no autoregressive answer generation. Each candidate currently carries its **complete context through the backbone**. Batching candidates reduces serial orchestration, but does not share prefix computation or guarantee any speedup over another engine. Boolean uses two candidate paths. Candidate probabilities are model scores, not a guarantee of correctness.

## Train a pretrained backbone

Prepare train/dev files using the [JSONL contract](docs/data.md). Use source-group-separated splits; teacher probability targets and observed outcomes have different meanings.

```bash
lightjev train \
  --model Qwen/Qwen3-0.6B \
  --train data/train.jsonl --dev data/dev.jsonl \
  --output-dir runs/qwen-ce \
  --steps 200 --batch-size 2 --lr 2e-5 \
  --loss ce --max-length 512 --device cuda
```

Pass `--revision <commit>` to pin a Hub backbone. This command downloads that backbone and performs full-parameter training. GPU memory depends on candidate count and input length; the CLI defaults to CPU and does not schedule jobs on remote hosts. The Qwen command is an integration recipe, not a published LightJev Qwen training result.

For a direct probability-loss control, use a fresh output directory with `--loss brier`, holding the dataset, seed, steps, and backbone fixed. No RL implementation or proprietary RLCD equivalence is claimed.

## Predict, evaluate, calibrate

```bash
lightjev predict --checkpoint runs/qwen-ce \
  --input data/test.jsonl --output runs/test-predictions.json --device cuda
lightjev evaluate --input data/test.jsonl \
  --predictions runs/test-predictions.json --output runs/test-metrics.json
```

Inference records may omit `target`. Prediction output includes candidate order and the selected index, so labels can be audited without parsing generated text. Score is the probability-weighted **ordinal index**, not a numerical value extracted from a candidate string.

For temperature fitting, generate predictions at temperature 1 on a separate calibration split:

```bash
lightjev predict --checkpoint runs/qwen-ce \
  --input data/calibration.jsonl --output runs/calibration-predictions.json
lightjev calibrate --input data/calibration.jsonl \
  --predictions runs/calibration-predictions.json --output runs/temperature.json
```

Use the reported temperature with `lightjev predict --temperature <value>` on the untouched test split. Calibration is an empirical property on a distribution, not a property conferred by calling softmax. [Metric definitions and limitations](docs/evaluation.md).

## Development

```bash
python -m pytest -q
python -m build
```

CI runs on CPU with Hub access disabled for tests and the demo. It uploads a demo report for inspection.

## Roadmap

- Domain datasets and pretrained-backbone measurements with multiple seeds.
- Candidate-set interaction and equivalent shared-prefix inference.
- Explicit observed-event targets and optional proper-reward experiments.
- Batched serving and end-to-end latency/memory comparisons at matched quality.

## Related work and license

Inspired by [TypeSafe AI's Jev](https://typesafe.ai/) and [NanoJev](https://github.com/TianyuCodings/NanoJev). LightJev is an independent implementation and is not affiliated with TypeSafe AI. We do not claim to reproduce its unpublished architecture or RLCD training recipe. No NanoJev weights, datasets, benchmark numbers, or implementation files are redistributed here.

Apache-2.0 for LightJev code. Pretrained backbones and external datasets retain their respective licenses. See [NOTICE](NOTICE).
