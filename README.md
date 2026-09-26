# LightJev

**Train small language backbones to make typed decisions.**

[![CI](https://github.com/rongxinzy/LightJev/actions/workflows/ci.yml/badge.svg)](https://github.com/rongxinzy/LightJev/actions/workflows/ci.yml)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-LightJev--0.6B-yellow)](https://huggingface.co/rongxinzy/LightJev-0.6B-v0.1)
[简体中文](README.zh-CN.md) · [Data format](docs/data.md) · [Design](docs/design.md) · [Evaluation](docs/evaluation.md)

LightJev uses **Qwen3-0.6B as its standard pretrained backbone** and turns it into a finite-candidate decision model. Supply context, a question, and candidate descriptions; receive a normalized distribution and a selected value. Train the backbone and a small scoring head with cross-entropy or Brier loss.

## Release progress

**2026-09-18:** Completed Qwen3-0.6B full-parameter CE/Brier training and published the first model checkpoint on Hugging Face. The accompanying [v0.1.1 code release](https://github.com/rongxinzy/LightJev/releases/tag/v0.1.1) passed 46 tests and CI. Public model files were checked against release hashes; anonymous weight-file access was verified.

**[Download LightJev-0.6B-v0.1](https://huggingface.co/rongxinzy/LightJev-0.6B-v0.1)** — trained Qwen3-0.6B scoring weights, tokenizer, frozen gold-only data, training histories, and both CE/Brier evaluation reports. The published CE checkpoint was selected at step 250 using development CE before held-out results were inspected.

Raw hard-label accuracy is **79.57% on test (656 questions)** and **72.44% on source-defined OOD (352)**. Exact soft-distribution squared L2 is **0.003132 / 0.003964** respectively. These are synthetic-task, single-seed results. Catalog lookup and smart-home rules are stronger than grid and tic-tac-toe judgments; this is not a broadly capable business decision model. [Full results and limitations](docs/results-v0.1.md).

**2026-09-24 — Typed Decisions transfer study:** LightJev-0.6B scored 0.396 accuracy zero-shot, below the dataset prior (0.470). A Laya specialist soft-CE run scored 0.781, but specialist and generalist results are not comparable, and this control was initiated after inspecting the RLCD test result; it is exploratory, not a leaderboard claim. The [reproducible report and artifacts](research/laya-typed-decisions-2026-09-24/REPORT.zh-CN.md) include splits, code and full prediction distributions. The ~804 MB checkpoints are not published.

**Workflow transfer validation is complete:** four leave-one-workflow-out folds compare Laya RLCD, a matched soft-CE control, the pinned Laya checkpoint, and LightJev zero-shot. Across 6,000 held-out decisions, accuracy is 45.3% for RLCD, 44.7% for soft-CE, 36.1% for Laya base, and 35.7% for LightJev. The [protocol, reproducibility scripts, and full results](research/laya-typed-decisions-2026-09-24/workflow-generalization/README.md) exclude the official public test split. Results are exploratory; earlier work inspected aggregate scores from these workflow families.
**LightJev cross-workflow fine-tuning:** two seeds across the same four held-out workflow folds average 43.7% accuracy, versus 45.3% for Laya RLCD and 44.7% for Laya soft-CE; LightJev reaches 50.0% on held-out security incidents and beats the Laya base overall. The paired case-clustered analysis and per-seed outputs are in the [workflow study](research/laya-typed-decisions-2026-09-24/workflow-generalization/README.md). This is exploratory and does not establish an overall win over Laya fine-tuning.

## Use the trained checkpoint

### Installation and first run

Use Python 3.10 or newer (Python 3.12 was used for validation):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'git+https://github.com/rongxinzy/LightJev.git@v0.1.1' huggingface_hub
```

On Windows, activate with `.venv\Scripts\activate`. CPU inference works without CUDA. For NVIDIA GPU inference, install a CUDA-enabled PyTorch build compatible with your driver; check `python -c "import torch; print(torch.cuda.is_available())"` before selecting `device="cuda"`.

The public checkpoint requires no HF token. First download includes about **2.38 GB of FP32 weights**, plus tokenizer and metadata; later calls reuse the Hugging Face disk cache. Runtime memory exceeds the weight-file size; no minimum RAM/VRAM requirement has been benchmarked. Save the Python example below as `example.py`, then run `python example.py`.

```python
from huggingface_hub import snapshot_download
from lightjev.inference import predict

checkpoint = snapshot_download("rongxinzy/LightJev-0.6B-v0.1")
records = [{
    "id": "example-1", "group_id": "example",
    "state": "Home log: user wants the dining room fan set to on; access=yes; occupants=2; clock=20:00.",
    "question": "Select exactly the room and device named in the request. Ignore authorization and occupancy for this question.",
    "kind": "choice", "candidates": ["balcony speaker", "dining room fan"],
}]
print(predict(checkpoint, records, device="cpu"))
```

This example is the first question from the frozen test split, shown in its original task format; it is not a new generalization test. A separate hand-written cooling-rule probe failed, as recorded in [limitations](docs/results-v0.1.md#additional-manual-probe).

This is a **custom LightJev scoring checkpoint**, not an ordinary chat model or an `AutoModelForCausalLM` checkpoint. Use `device="cuda"` for GPU inference. Prediction defaults to raw probabilities; temperature scaling is optional and did not improve the selected CE model's held-out CE/ECE. Inputs over 256 tokens per candidate fail explicitly. [Reproduce training](docs/training-release.md) · [Data attribution](docs/data.md).

### Input, output and multiple questions

Every input record needs nonempty string fields `id`, `group_id`, `state`, `question`, plus `kind` and `candidates`. `target` is optional for inference; omit it for your own requests. Candidates must be 2–255 unique, nonempty strings.

| `kind` | Candidates | Additional output |
|---|---|---|
| `choice` | Candidate descriptions | — |
| `boolean` | Exactly `["false", "true"]` in this order | — |
| `score` | Descriptions ordered from low to high | `expectation`: probability-weighted ordinal index |

`predict()` returns a list in input order. The example's verified CPU result selects `dining room fan`, with probabilities approximately `[0.000000076, 0.99999988]`; small numerical differences across devices are expected. Each result contains `id`, `kind`, `candidates`, `probabilities`, `selected`, zero-based `selected_index`, and `temperature`.

Pass several records in one `predict(checkpoint, records, device="cuda")` call to share model loading. **Each call loads the model again, and records inside a call are processed sequentially**; this is not a persistent server or a cross-question GPU batching API. The download cache avoids downloading again, not loading weights into memory again. A long-running service needs a wrapper that retains `load_checkpoint()`'s model/tokenizer and implements the same encoding, scoring and per-question softmax.

The 256-token limit applies to the **entire formatted input for each candidate**, including state, question, candidate and prompt text. Shorten oversized inputs explicitly. Default `temperature=1` is unchanged; the released fitted temperature worsened the selected model's held-out CE/ECE.

### Command-line inference with JSONL

After the Python example has defined `checkpoint` and `records`, save them locally:

```python
import json
from pathlib import Path

Path("checkpoint-path.txt").write_text(checkpoint, encoding="utf-8")
Path("input.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
    encoding="utf-8",
)
```

On a POSIX shell:

```bash
lightjev predict --checkpoint "$(cat checkpoint-path.txt)" \
  --input input.jsonl --output predictions.json --device cpu
```

Use `--device cuda` for a compatible GPU. Input is one JSON object per line with unique IDs; output is a JSON array. For offline inference, use the previously downloaded local checkpoint path; the loader requires `manifest.json`, `model.safetensors`, `backbone/` and `tokenizer/`, and does not fetch missing files.

**Backend status:** the published interface uses PyTorch/Transformers. Direct `vllm serve` loading is not implemented or validated. Do not load these scoring weights through `AutoModelForCausalLM` or a chat-completions API.

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

The default backbone is pinned to `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`. Pass `--revision <commit>` to override it; other model IDs and local paths are not assigned this Qwen commit. This command downloads that backbone and performs full-parameter training. GPU memory depends on candidate count and input length; the CLI defaults to CPU and does not schedule jobs on remote hosts. The Qwen command is a training recipe, not a published capability result.

For a direct probability-loss control, use a fresh output directory with `--loss brier`, holding the dataset, seed, steps, and backbone fixed. No RL implementation or proprietary RLCD equivalence is claimed.

### Verify the real Qwen backbone

```bash
python scripts/verify_qwen.py --output-dir runs/qwen-smoke --device cpu
```

This opt-in integration check downloads the pinned Qwen weights, performs one full-parameter training update, and saves/reloads a decision checkpoint. It uses six small hand-written examples and is not a quality or calibration benchmark. Unlike the tiny offline demo, it requires substantial memory and a first-time model download.

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

Inspired by [TypeSafe AI's Jev](https://typesafe.ai/) and [NanoJev](https://github.com/TianyuCodings/NanoJev). LightJev is an independent implementation and is not affiliated with TypeSafe AI. We do not claim to reproduce its unpublished architecture or RLCD training recipe. The Qwen training experiment uses a converted subset of [C-Tianyu/NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data), with every source row declaring CC0-1.0. Only programmatic gold targets are retained; Jev teacher outputs are excluded. NanoJev weights and implementation files are not used. Our independent implementation does not imply independently authored training data. See [data attribution and conversion](docs/data.md).

Apache-2.0 for LightJev code. Pretrained backbones and external datasets retain their respective licenses. See [NOTICE](NOTICE).
