#!/usr/bin/env bash
set -u
STUDY_ROOT="${STUDY_ROOT:-/root/lightjev-laya-study-20260924/typed-decisions/workflow-generalization-20260924}"
CHECKPOINT="${LIGHTJEV_CHECKPOINT:-/root/lightjev-laya-study-20260924/lightjev-checkpoint}"
PYTHON="${PYTHON_ENV:-/root/lightjev-laya-study-20260924/.venv/bin/python}"
PACKAGE_SRC="${LIGHTJEV_SRC:-/root/lightjev-laya-study-20260924/LightJev/src}"
CODE="$STUDY_ROOT/code"
mkdir -p "$STUDY_ROOT/runs/lightjev_xworkflow" "$STUDY_ROOT/logs/lightjev_xworkflow"
declare -a pids=()
declare -a names=()
folds=(agent_trace_observability customer_service invoice_processing security_incidents)
seeds=(31 47)
gpu=0
for fold in "${folds[@]}"; do
  for seed in "${seeds[@]}"; do
    name="${fold}_seed${seed}"
    output="$STUDY_ROOT/runs/lightjev_xworkflow/$fold/seed${seed}"
    mkdir -p "$(dirname "$output")"
    if [[ -d "$output" ]] && compgen -G "$output/*" >/dev/null; then
      echo "refusing nonempty output $output" >&2; exit 2
    fi
    echo "START $name gpu=$gpu $(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PACKAGE_SRC:$CODE" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
      "$PYTHON" "$CODE/train_lightjev_fold.py" \
      --checkpoint "$CHECKPOINT" \
      --train "$STUDY_ROOT/lightjev_records/$fold/records_train.jsonl" \
      --dev "$STUDY_ROOT/lightjev_records/$fold/records_dev.jsonl" \
      --output "$output" --seed "$seed" --steps 256 --batch-size 2 --grad-accum 32 \
      --max-length 640 --eval-every 64 --encoder-lr 2.5e-5 --head-lr 1e-4 \
      --warmup-steps 12 --gradient-checkpointing \
      > "$STUDY_ROOT/logs/lightjev_xworkflow/$name.log" 2>&1 &
    pids+=("$!"); names+=("$name"); gpu=$((gpu + 1))
  done
done
printf '%s\n' "${pids[@]}" > "$STUDY_ROOT/lightjev_training.pids"
status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "DONE ${names[$index]} $(date -u +%FT%TZ)"
  else
    code=$?; echo "FAILED ${names[$index]} exit=$code $(date -u +%FT%TZ)"; status=1
  fi
done
exit "$status"
