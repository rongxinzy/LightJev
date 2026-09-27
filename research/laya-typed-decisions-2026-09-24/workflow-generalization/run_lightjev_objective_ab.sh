#!/usr/bin/env bash
set -euo pipefail

STUDY_ROOT="${STUDY_ROOT:-/root/lightjev-laya-study-20260924/typed-decisions/workflow-generalization-20260924}"
CHECKPOINT="${LIGHTJEV_CHECKPOINT:-/root/lightjev-laya-study-20260924/lightjev-checkpoint}"
PYTHON="${PYTHON_ENV:-/root/lightjev-laya-study-20260924/.venv/bin/python}"
PACKAGE_SRC="${LIGHTJEV_SRC:-/root/lightjev-laya-study-20260924/LightJev/src}"
CODE="${CODE_DIR:-$STUDY_ROOT/code}"
folds=(agent_trace_observability customer_service invoice_processing security_incidents)
read -r -a seeds <<< "${LIGHTJEV_SEEDS:-31 47 73 97}"
objectives=(ce rlcd)
tasks=()
for objective in "${objectives[@]}"; do
  for fold in "${folds[@]}"; do
    for seed in "${seeds[@]}"; do tasks+=("$objective $fold $seed"); done
  done
done
mkdir -p "$STUDY_ROOT/runs/lightjev_objective_ab" "$STUDY_ROOT/results/lightjev_objective_ab" "$STUDY_ROOT/logs/lightjev_objective_ab"

run_batches() {
  local phase="$1"; shift
  local -a pids=() names=()
  local offset=0 gpu objective fold seed name input output
  while (( offset < ${#tasks[@]} )); do
    pids=(); names=(); gpu=0
    while (( gpu < 8 && offset < ${#tasks[@]} )); do
      read -r objective fold seed <<< "${tasks[$offset]}"
      name="${objective}_${fold}_seed${seed}"
      if [[ "$phase" == train ]]; then
        output="$STUDY_ROOT/runs/lightjev_objective_ab/$objective/$fold/seed${seed}"
        if [[ -d "$output" ]] && compgen -G "$output/*" >/dev/null; then
          echo "refusing nonempty output $output" >&2; return 2
        fi
        mkdir -p "$(dirname "$output")"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PACKAGE_SRC:$CODE" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
          "$PYTHON" "$CODE/train_lightjev_shared_fold.py" --checkpoint "$CHECKPOINT" \
          --train "$STUDY_ROOT/lightjev_records/$fold/records_train.jsonl" \
          --dev "$STUDY_ROOT/lightjev_records/$fold/records_dev.jsonl" --output "$output" \
          --seed "$seed" --steps 256 --batch-size 2 --grad-accum 32 --max-length 640 \
          --eval-every 64 --encoder-lr 2.5e-5 --head-lr 1e-4 --warmup-steps 12 \
          --objective "$objective" --rlcd-group-size 4 --rlcd-sigma-start 0.4 \
          --rlcd-sigma-end 0.1 --rlcd-ce-weight 1.0 --gradient-checkpointing \
          > "$STUDY_ROOT/logs/lightjev_objective_ab/train_$name.log" 2>&1 &
      else
        input="$STUDY_ROOT/runs/lightjev_objective_ab/$objective/$fold/seed${seed}"
        output="$STUDY_ROOT/results/lightjev_objective_ab/$objective/$fold/seed${seed}"
        mkdir -p "$output"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PACKAGE_SRC:$CODE" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
          "$PYTHON" "$CODE/evaluate_lightjev_shared_calibrated.py" --checkpoint "$input" \
          --calibration "$STUDY_ROOT/lightjev_records/$fold/records_calibration.jsonl" \
          --test "$STUDY_ROOT/lightjev_records/$fold/records_test.jsonl" --output-dir "$output" \
          --batch-size 8 --permutations 4 \
          > "$STUDY_ROOT/logs/lightjev_objective_ab/eval_$name.log" 2>&1 &
      fi
      pids+=("$!"); names+=("$name"); gpu=$((gpu + 1)); offset=$((offset + 1))
    done
    local status=0
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then echo "${phase^^}_DONE ${names[$i]}"; else echo "${phase^^}_FAILED ${names[$i]}"; status=1; fi
    done
    (( status == 0 )) || return "$status"
  done
}

run_batches train
run_batches eval
echo "OBJECTIVE_AB_COMPLETE jobs=${#tasks[@]} gpu_count=8 seeds=${seeds[*]}"
