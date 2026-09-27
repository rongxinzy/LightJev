#!/usr/bin/env bash
set -u
STUDY_ROOT="${STUDY_ROOT:-/root/lightjev-laya-study-20260924/typed-decisions/workflow-generalization-20260924}"
CHECKPOINT="${LIGHTJEV_CHECKPOINT:-/root/lightjev-laya-study-20260924/lightjev-checkpoint}"
PYTHON="${PYTHON_ENV:-/root/lightjev-laya-study-20260924/.venv/bin/python}"
PACKAGE_SRC="${LIGHTJEV_SRC:-/root/lightjev-laya-study-20260924/LightJev/src}"
CODE="$STUDY_ROOT/code"
folds=(agent_trace_observability customer_service invoice_processing security_incidents)
read -r -a seeds <<< "${LIGHTJEV_SEEDS:-31 47}"
mkdir -p "$STUDY_ROOT/runs/lightjev_shared_xworkflow" "$STUDY_ROOT/logs/lightjev_shared_xworkflow"
declare -a pids=() names=(); gpu=0; status=0
for fold in "${folds[@]}"; do
  for seed in "${seeds[@]}"; do
    name="${fold}_seed${seed}"; output="$STUDY_ROOT/runs/lightjev_shared_xworkflow/$fold/seed${seed}"
    mkdir -p "$(dirname "$output")"
    if [[ -d "$output" ]] && compgen -G "$output/*" >/dev/null; then echo "refusing nonempty output $output" >&2; exit 2; fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PACKAGE_SRC:$CODE" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
      "$PYTHON" "$CODE/train_lightjev_shared_fold.py" --checkpoint "$CHECKPOINT" \
      --train "$STUDY_ROOT/lightjev_records/$fold/records_train.jsonl" \
      --dev "$STUDY_ROOT/lightjev_records/$fold/records_dev.jsonl" --output "$output" \
      --seed "$seed" --steps 256 --batch-size 2 --grad-accum 32 --max-length 640 \
      --eval-every 64 --encoder-lr 2.5e-5 --head-lr 1e-4 --warmup-steps 12 --gradient-checkpointing \
      > "$STUDY_ROOT/logs/lightjev_shared_xworkflow/train_$name.log" 2>&1 &
    pids+=("$!"); names+=("$name"); gpu=$((gpu + 1))
  done
done
printf '%s\n' "${pids[@]}" > "$STUDY_ROOT/shared_training.pids"
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "TRAIN_DONE ${names[$i]} $(date -u +%FT%TZ)"; else echo "TRAIN_FAILED ${names[$i]}"; status=1; fi
done
[[ "$status" == 0 ]] || exit "$status"
declare -a eval_pids=(); gpu=0
for fold in "${folds[@]}"; do
  for seed in "${seeds[@]}"; do
    name="${fold}_seed${seed}"; checkpoint="$STUDY_ROOT/runs/lightjev_shared_xworkflow/$fold/seed${seed}"
    output="$STUDY_ROOT/results/lightjev_shared_xworkflow/$fold/seed${seed}"
    mkdir -p "$output"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PACKAGE_SRC:$CODE" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
      "$PYTHON" "$CODE/evaluate_lightjev_shared_calibrated.py" --checkpoint "$checkpoint" \
      --calibration "$STUDY_ROOT/lightjev_records/$fold/records_calibration.jsonl" \
      --test "$STUDY_ROOT/lightjev_records/$fold/records_test.jsonl" --output-dir "$output" \
      --batch-size 8 --permutations 4 \
      > "$STUDY_ROOT/logs/lightjev_shared_xworkflow/eval_$name.log" 2>&1 &
    eval_pids+=("$!"); gpu=$((gpu + 1))
  done
done
printf '%s\n' "${eval_pids[@]}" > "$STUDY_ROOT/shared_eval.pids"
for i in "${!eval_pids[@]}"; do
  if wait "${eval_pids[$i]}"; then echo "EVAL_DONE ${names[$i]} $(date -u +%FT%TZ)"; else echo "EVAL_FAILED ${names[$i]}"; status=1; fi
done
exit "$status"
