#!/usr/bin/env bash
set -euo pipefail

ROOT="${RELEASE_ROOT:-/root/lightjev-release-v0.2-20260928}"
CODE="${CODE_DIR:-$ROOT/code}"
PYTHON="${PYTHON_ENV:-/root/lightjev-laya-study-20260924/.venv/bin/python}"
PACKAGE_SRC="${LIGHTJEV_SRC:-$ROOT/src}"
MODEL_CARD="${MODEL_CARD:-$ROOT/model-card/README.md}"
EXPORT="$ROOT/export"

if [[ ! -s "$ROOT/selection.json" ]]; then
  echo "missing completed training selection: $ROOT/selection.json" >&2
  exit 2
fi
if [[ -d "$EXPORT" ]] && compgen -G "$EXPORT/*" >/dev/null; then
  echo "refusing nonempty export directory: $EXPORT" >&2
  exit 2
fi
seed="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_seed"])' "$ROOT/selection.json")"
SOURCE="$ROOT/runs/seed${seed}"
mkdir -p "$EXPORT/training" "$EXPORT/evaluation"
cp -a "$SOURCE/backbone" "$SOURCE/tokenizer" "$EXPORT/"
cp "$SOURCE/model.safetensors" "$EXPORT/model.safetensors"
cp "$SOURCE/manifest.json" "$EXPORT/manifest.json"
cp "$SOURCE/train_log.jsonl" "$EXPORT/training/train_log.jsonl"
cp "$ROOT/selection.json" "$EXPORT/training/selection.json"
cp "$MODEL_CARD" "$EXPORT/README.md"
cp "${LIGHTJEV_REPO:-/root/lightjev-release-v0.2-20260928/repo}/LICENSE" "$EXPORT/LICENSE"
cat "${LIGHTJEV_REPO:-/root/lightjev-release-v0.2-20260928/repo}/NOTICE" \
    "$CODE/NOTICE-typed-decisions-v0.2.txt" > "$EXPORT/NOTICE"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PACKAGE_SRC:$CODE" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" "$CODE/evaluate_lightjev_shared_calibrated.py" \
    --checkpoint "$EXPORT" \
    --calibration "$ROOT/records/records_calibration.jsonl" \
    --test "$ROOT/records/records_test.jsonl" \
    --output-dir "$EXPORT/evaluation" --batch-size 8 --permutations 4

cp "$EXPORT/evaluation/evaluation.json" "$EXPORT/temperature.json"
"$PYTHON" - "$EXPORT/manifest.json" "$ROOT/selection.json" "$EXPORT/evaluation/evaluation.json" "$ROOT/records" <<'PY'
import hashlib, json, pathlib, sys
manifest_path, selection_path, evaluation_path = sys.argv[1:4]
records_dir = pathlib.Path(sys.argv[4])
manifest = json.load(open(manifest_path))
selection = json.load(open(selection_path))
evaluation = json.load(open(evaluation_path))
split_hashes = {}
for split in ("train", "dev", "calibration", "test"):
    path = records_dir / f"records_{split}.jsonl"
    split_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest.update({
    "release_id": "LightJev-0.6B-typed-decisions",
    "release_version": "0.2.0",
    "release_training_seed": selection["selected_seed"],
    "seed_selection": selection,
    "temperatures_by_kind": {kind: 1.0 for kind in evaluation["temperatures"]},
    "fitted_temperatures_by_kind": evaluation["temperatures"],
    "calibration_split": "120 cases held out from the official training split",
    "evaluation_split": "official 400-case test split; evaluated only after checkpoint selection",
    "data_provenance": {
        "dataset": "LocalLLaMA/typed-decisions",
        "dataset_revision": "c76749ec58bd8c3d2ea706b31c333a9059c38f90",
        "dataset_license": "Apache-2.0",
        "records_jsonl_sha256": split_hashes,
    },
})
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY
(cd "$EXPORT" && sha256sum model.safetensors manifest.json > SHA256SUMS)
echo "EXPORT_READY seed=$seed directory=$EXPORT"
