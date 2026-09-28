#!/usr/bin/env bash
set -euo pipefail

ROOT="${RELEASE_ROOT:-/root/lightjev-release-v0.2-20260928}"
CODE="${CODE_DIR:-$ROOT/code}"
CHECKPOINT="${LIGHTJEV_CHECKPOINT:-/root/lightjev-laya-study-20260924/lightjev-checkpoint}"
PYTHON="${PYTHON_ENV:-/root/lightjev-laya-study-20260924/.venv/bin/python}"
PACKAGE_SRC="${LIGHTJEV_SRC:-/root/lightjev-release-v0.2-20260928/src}"
SEEDS=(31 47 73 97)
GPUS=(0 1 2 3)

mkdir -p "$ROOT/runs" "$ROOT/logs"
pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  output="$ROOT/runs/seed${seed}"
  if [[ -d "$output" ]] && compgen -G "$output/*" >/dev/null; then
    echo "refusing nonempty output: $output" >&2
    exit 2
  fi
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" PYTHONPATH="$PACKAGE_SRC:$CODE" \
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON" "$CODE/train_lightjev_shared_fold.py" \
      --checkpoint "$CHECKPOINT" \
      --train "$ROOT/records/records_train.jsonl" \
      --dev "$ROOT/records/records_dev.jsonl" \
      --output "$output" --seed "$seed" --steps 300 --batch-size 4 \
      --grad-accum 16 --max-length 640 --eval-every 60 \
      --encoder-lr 2.5e-5 --head-lr 1e-4 --warmup-steps 15 \
      --objective ce --gradient-checkpointing \
      > "$ROOT/logs/train_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "TRAIN_DONE seed=${SEEDS[$i]}"
  else
    echo "TRAIN_FAILED seed=${SEEDS[$i]}" >&2
    status=1
  fi
done
(( status == 0 )) || exit "$status"

"$PYTHON" "$CODE/write_selection.py" "$ROOT/runs" "$ROOT/selection.json"
