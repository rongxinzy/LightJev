#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${TASK_ROOT:-/root/lightjev-laya-mainline-20260928}
BASE_DIR=${LAYA_BASE_DIR:-/root/lightjev-laya-study-20260924/model}
ENV_DIR=${LAYA_ENV_DIR:-/root/lightjev-laya-study-20260924/.venv}
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${TASK_ROOT}/vendor:${TASK_ROOT}/code"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 USE_TF=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TASK_ROOT}/code"
mkdir -p "${TASK_ROOT}/logs" "${TASK_ROOT}/runs" "${TASK_ROOT}/results"
python - <<'PY'
import subprocess
rows = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
if len(rows.splitlines()) != 8 or any(int(x) > 1024 for x in rows.splitlines()):
    raise SystemExit("Expected eight free GPUs; refusing to overlap another resident workload.")
PY
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --base "${BASE_DIR}" --data "${TASK_ROOT}/data" --output "${TASK_ROOT}/runs/smoke" \
  --smoke --steps 64 --micro-batch 1 --grad-accum 1 --eval-every 8 \
  --encoder-lr 3e-5 --head-lr 3e-4 2>&1 | tee "${TASK_ROOT}/logs/smoke.log"
python - "${TASK_ROOT}/runs/smoke/complete.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
assert value["status"] == "complete" and value["reload_max_logit_delta"] <= 1e-5
PY
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --base "${BASE_DIR}" --data "${TASK_ROOT}/data" --output "${TASK_ROOT}/runs/ce_seed31" \
  2>&1 | tee "${TASK_ROOT}/logs/ce_seed31.log"
SELECTED=$(python - "${TASK_ROOT}/runs/ce_seed31/selection.json" "${TASK_ROOT}" <<'PY'
import json, pathlib, sys
value = json.load(open(sys.argv[1]))
print(pathlib.Path(sys.argv[2]) / "runs/ce_seed31/selected" if value["step"] else value["checkpoint"])
PY
)
python evaluate.py --model "${BASE_DIR}" --data "${TASK_ROOT}/data" \
  --output "${TASK_ROOT}/results/base_diagnostics" --diagnostics \
  2>&1 | tee "${TASK_ROOT}/logs/base_diagnostics.log"
python evaluate.py --model "${SELECTED}" --data "${TASK_ROOT}/data" \
  --output "${TASK_ROOT}/results/selected_diagnostics" --diagnostics \
  2>&1 | tee "${TASK_ROOT}/logs/selected_diagnostics.log"
echo 'PILOT_COMPLETE: development diagnostics only; locked test remains unopened.'
