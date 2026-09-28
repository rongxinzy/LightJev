# LightJev Laya mainline

This research line adapts the ordinary English Laya checkpoint to follow supplied decision rules. See the frozen [protocol](PROTOCOL.md), [runtime lock](runtime_lock.json), and [data manifest](data/manifest.json). Existing Qwen releases remain supported by the existing package; the scripts here form a separate research pipeline.

**Pilot complete:** eight-GPU tiny-fit and checkpoint-reload gates passed; the fixed 128-update CE pilot selected step 32. On two development rule families absent from adaptation training, accuracy improved from **53.13% to 83.20%**, and counterfactual pair both-correct rate from **10.16% to 66.41%**. However, NLL worsened from **1.385 to 2.311** and every remaining error in that split had confidence above 99%. These are single-seed synthetic development results, not a final generalization claim. The 384 locked-test decisions remain unevaluated. [Full results, negative findings and next steps](RESULTS.zh-CN.md).

## What changes

- Ordinary `convaiinnovations/laya`, not the benchmark-specific specialist.
- Full native criteria for Choice, NOUL and ordered Score; reject silent truncation.
- Four training rule generators, two development-only generators, two locked test generators.
- Each group includes a base, a rule counterfactual, a fact counterfactual and a label-preserving irrelevant change.
- Exact programmatic labels, CE training, raw probability metrics, and untouched-base checkpoint selection as a fallback.
- Pair both-correct rates, per-family metrics, option-order sensitivity and missing-input controls.

These are synthetic rule-generator splits, not independent real-world industries. Initial model pretraining coverage is not fully known. The inherited Laya escalation head is frozen and is not a validated abstention mechanism. This pilot does not fit temperatures; calibration data is reserved for a subsequent independently reported calibration step.

## Reproduce

The validated environments have Python 3.12 (local tests) / 3.10 (GPU training), PyTorch 2.14.0, Transformers 5.17.0, Safetensors, NumPy and pytest for tests. CUDA training uses BF16 autocast with FP32 parameters and optimizer state. The existing source dependency pins are in `runtime_lock.json`; the runner checks their hashes.

Create a Python environment and install the listed dependencies with an appropriate CUDA PyTorch build. Obtain the upstream repository and fixed snapshot:

```bash
git clone https://github.com/NandhaKishorM/laya.git vendor-laya
git -C vendor-laya checkout 23a17522aa4942da6cce53a995a275760320b691
hf download convaiinnovations/laya \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 \
  --include model.safetensors rl_agent_config.json 'encoder/*' 'tokenizer/*' \
  --local-dir laya-base
export PYTHONPATH="$PWD/vendor-laya"
export LAYA_BASE_DIR="$PWD/laya-base"
```

From the repository root, run the semantic and pinned-tokenizer tests:

```bash
python -m pytest research/laya-mainline-2026-09-28/test_mainline.py -q
```

CI runs the dependency-light checks offline; the tokenizer integration check is skipped unless `LAYA_BASE_DIR` is set. Local preflight runs included that integration check. The checked-in data was validated against the pinned tokenizer. Regenerate into a fresh directory if needed:

```bash
python research/laya-mainline-2026-09-28/data.py \
  --model-dir "$LAYA_BASE_DIR" --output /absolute/path/to/new-data
```

Run the tiny fit before the pilot; both require a new output directory:

```bash
torchrun --standalone --nproc_per_node=8 research/laya-mainline-2026-09-28/train.py \
  --base "$LAYA_BASE_DIR" --data research/laya-mainline-2026-09-28/data \
  --output /absolute/path/to/smoke --smoke --steps 64 \
  --micro-batch 1 --grad-accum 1 --eval-every 8 --encoder-lr 3e-5 --head-lr 3e-4

# Only proceed after smoke/complete.json exists and reports success.
torchrun --standalone --nproc_per_node=8 research/laya-mainline-2026-09-28/train.py \
  --base "$LAYA_BASE_DIR" --data research/laya-mainline-2026-09-28/data \
  --output /absolute/path/to/ce-seed31
```

`selection.json` identifies the chosen checkpoint, which can be the original base if no checkpoint improves development NLL. Evaluation reads only the two development splits:

```bash
python research/laya-mainline-2026-09-28/evaluate.py \
  --model /absolute/path/to/chosen-checkpoint \
  --data research/laya-mainline-2026-09-28/data \
  --output /absolute/path/to/new-development-report --diagnostics
```

The remote [runner](run_pilot.sh) provides the same sequence with environment-configurable locations. It checks that all eight GPUs are free, fails on an unsuccessful tiny fit, and stops after one pilot and its development diagnostics. No automatic extra training is scheduled after a negative result. Apply the continuation gate with `report_pilot.py --artifacts /absolute/path/to/artifacts`; the included `artifacts/` directory retains metadata and predictions, without weight files.

The exported selected checkpoint also passed native SDK parity checks on nine development examples through `verify_native.py`. It can be loaded with `laya.load('/absolute/path/to/selected', device='cuda')`; use the supplied native question schema. Its raw probabilities are uncalibrated and its inherited `action.act_probability` is not validated for escalation.

## Attribution

Laya architecture/runtime and starting weights are from Convai Innovations / NandhaKishorM under Apache-2.0. The original repository and checkpoint are identified in the lock file. This directory does not vendor their code or weights; derivative model publication must carry the applicable license, notices, source lineage and measured changes. Rule data and pilot code are new LightJev research artifacts under the repository license.
