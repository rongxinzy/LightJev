#!/usr/bin/env bash
set -u
STUDY_ROOT="${STUDY_ROOT:-/root/lightjev-laya-study-20260924/typed-decisions/workflow-generalization-20260924}"
PYTHON="${PYTHON_ENV:-/root/lightjev-laya-study-20260924/.venv/bin/python}"
PACKAGE_SRC="${LIGHTJEV_SRC:-/root/lightjev-laya-study-20260924/LightJev/src}"
CODE="$STUDY_ROOT/code"
mkdir -p "$STUDY_ROOT/results/lightjev_xworkflow" "$STUDY_ROOT/logs/lightjev_xworkflow_eval"
declare -a pids=()
declare -a names=()
folds=(agent_trace_observability customer_service invoice_processing security_incidents)
seeds=(31 47)
gpu=0
for fold in "${folds[@]}"; do
  for seed in "${seeds[@]}"; do
    name="${fold}_seed${seed}"
    checkpoint="$STUDY_ROOT/runs/lightjev_xworkflow/$fold/seed${seed}"
    output="$STUDY_ROOT/results/lightjev_xworkflow/$fold/seed${seed}"
    mkdir -p "$(dirname "$output")"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PACKAGE_SRC:$CODE" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
      "$PYTHON" "$CODE/evaluate_lightjev_calibrated.py" \
      --checkpoint "$checkpoint" \
      --calibration "$STUDY_ROOT/lightjev_records/$fold/records_calibration.jsonl" \
      --test "$STUDY_ROOT/lightjev_records/$fold/records_test.jsonl" \
      --output-dir "$output" --max-length 640 --batch-size 2 \
      > "$STUDY_ROOT/logs/lightjev_xworkflow_eval/$name.log" 2>&1 &
    pids+=("$!"); names+=("$name"); gpu=$((gpu + 1))
  done
done
printf '%s\n' "${pids[@]}" > "$STUDY_ROOT/lightjev_eval.pids"
status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "DONE ${names[$index]} $(date -u +%FT%TZ)"
  else
    code=$?; echo "FAILED ${names[$index]} exit=$code $(date -u +%FT%TZ)"; status=1
  fi
done
exit "$status"
