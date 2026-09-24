#!/usr/bin/env bash
set -euo pipefail

ROOT=${EXPERIMENT_ROOT:-/root/lightjev-laya-study-20260924/typed-decisions/workflow-generalization-20260924}
BASE=${LAYA_BASE_DIR:-/root/lightjev-laya-study-20260924/model}
LIGHTJEV=${LIGHTJEV_CHECKPOINT:-/root/lightjev-laya-study-20260924/lightjev-checkpoint}
DATASET_REV=c76749ec58bd8c3d2ea706b31c333a9059c38f90
BASE_REV=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
TRAIN_PARQUET=${TRAIN_PARQUET:-/root/lightjev-laya-study-20260924/typed-decisions/${DATASET_REV}/all/train.parquet}
VENV=${PYTHON_ENV:-/root/lightjev-laya-study-20260924/.venv}
CODE="${ROOT}/code"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${CODE}:/root/lightjev-laya-study-20260924/LightJev/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

cd "${CODE}"
mkdir -p "${ROOT}/data" "${ROOT}/runs" "${ROOT}/results" "${ROOT}/logs"
python prepare_workflow_folds.py \
  --train-parquet "${TRAIN_PARQUET}" --model-dir "${BASE}" \
  --output-dir "${ROOT}/data" --seed 20260925

workflows=(agent_trace_observability customer_service invoice_processing security_incidents)
fold_index=0
for workflow in "${workflows[@]}"; do
  fold="${ROOT}/data/${workflow}"
  results="${ROOT}/results/${workflow}"
  mkdir -p "${results}"
  fold_seed=$((20261001 + fold_index))
  echo "=== fold ${workflow} ==="

  torchrun --standalone --nnodes=1 --nproc_per_node=8 evaluate_benchmark.py \
    --model-dir "${BASE}" --cases "${fold}/cases_test.jsonl" \
    --output-dir "${results}" --label "laya_base_calibrated" --mode generalist
  torchrun --standalone --nnodes=1 --nproc_per_node=8 evaluate_benchmark.py \
    --model-dir "${BASE}" --cases "${fold}/cases_test.jsonl" \
    --output-dir "${results}" --label "laya_base_raw" --mode generalist --uncalibrated
  torchrun --standalone --nnodes=1 --nproc_per_node=8 evaluate_lightjev.py \
    --checkpoint "${LIGHTJEV}" --cases "${fold}/cases_test.jsonl" \
    --output-dir "${results}" --label "lightjev_zero_shot" --max-length 640

  for arm in rlcd soft_ce; do
    out="${ROOT}/runs/${workflow}/${arm}"
    weight=1.0
    [[ "${arm}" == soft_ce ]] && weight=0.0
    if [[ ! -s "${out}/model.safetensors" ]]; then
      torchrun --standalone --nnodes=1 --nproc_per_node=8 train_rlcd_ddp.py \
        --model-dir "${BASE}" --data-dir "${fold}" --output-dir "${out}" \
        --base-revision "${BASE_REV}" --dataset-revision "${DATASET_REV}" \
        --seed "${fold_seed}" --epochs 4 --rl-weight "${weight}" --ce-weight 1.0 \
        --micro-batch 2 --grad-accum 4 --max-len 1024 --head-max-len 256 \
        2>&1 | tee "${ROOT}/logs/${workflow}_${arm}.log"
    fi
    torchrun --standalone --nnodes=1 --nproc_per_node=8 evaluate_benchmark.py \
      --model-dir "${out}" --cases "${fold}/cases_test.jsonl" \
      --output-dir "${results}" --label "${arm}_calibrated" --mode generalist
    torchrun --standalone --nnodes=1 --nproc_per_node=8 evaluate_benchmark.py \
      --model-dir "${out}" --cases "${fold}/cases_test.jsonl" \
      --output-dir "${results}" --label "${arm}_raw" --mode generalist --uncalibrated
  done
  fold_index=$((fold_index + 1))
done

python summarize_folds.py --results-dir "${ROOT}/results" --output "${ROOT}/summary.json"
echo "ALL_FOLDS_COMPLETE ${ROOT}"
