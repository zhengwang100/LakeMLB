#!/bin/bash
# Parallel grid search (memory-efficient) for tabular neural network models.

set -euo pipefail

################ Config ################

PYTHON="/home/pfy/.conda/envs/lake/bin/python"
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

MODELS=("fttransformer" "tabtransformer" "excelformer" "saint" "tromptnet")

# Parallelism (adjust by GPU memory: ~20GB per task)
NUM_TASKS=2
TASK_DELAY=5

# Grid search space
GRID_HIDDEN="32,64"
GRID_LAYERS="2,3"
GRID_LR="1e-3,1e-4"
GRID_WD="1e-4,1e-3"
GRID_BS="128"
GRID_EPOCHS=200
GRID_PATIENCE=20
GRAD_ACCUM=2

# Final training
FINAL_EPOCHS=200
FINAL_PATIENCE=20
NUM_RUNS=10
SEED=42

########################################

BASELINE_DIR="$(cd "$(dirname "$0")/../baseline" && pwd)"
OUT_DIR="${BASELINE_DIR}/../results/grid_search_parallel"
LOG_DIR="${BASELINE_DIR}/../results/logs_parallel"
mkdir -p "$OUT_DIR" "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/run_parallel_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== Parallel Grid Search (Memory-Efficient) ==="
echo "GPU: $GPU_ID | Tasks: $NUM_TASKS | Models: ${MODELS[*]}"
echo "Grid: hidden=$GRID_HIDDEN layers=$GRID_LAYERS lr=$GRID_LR wd=$GRID_WD bs=$GRID_BS"
echo "Grid epochs=$GRID_EPOCHS patience=$GRID_PATIENCE | GradAccum=$GRAD_ACCUM"
echo "Final: epochs=$FINAL_EPOCHS runs=$NUM_RUNS seed=$SEED"
echo ""

START=$(date +%s)

# Run a single grid search task
run_task() {
    local model=$1 tid=$2
    local log="${LOG_DIR}/${model}_task_${tid}.log"
    echo "  [Task $((tid+1))/$NUM_TASKS] Starting..."
    $PYTHON "${BASELINE_DIR}/tnns_test.py" \
        --model "$model" --grid \
        --task_id "$tid" --num_tasks "$NUM_TASKS" \
        --grid_hidden "$GRID_HIDDEN" --grid_layers "$GRID_LAYERS" \
        --grid_lr "$GRID_LR" --grid_wd "$GRID_WD" --grid_bs "$GRID_BS" \
        --grid_epochs "$GRID_EPOCHS" --grid_patience "$GRID_PATIENCE" \
        --grid_output_dir "$OUT_DIR" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --device 0 --seed "$SEED" \
        > "$log" 2>&1 &
    echo "    Log: $log"
}

# Merge partial results and run final training
merge_and_train() {
    local model=$1
    local log="${LOG_DIR}/${model}_merge.log"
    echo "Merging $model results + final training..."
    $PYTHON "${BASELINE_DIR}/tnns_test.py" \
        --model "$model" --merge_results \
        --grid_output_dir "$OUT_DIR" \
        --epochs "$FINAL_EPOCHS" --patience "$FINAL_PATIENCE" \
        --num_runs "$NUM_RUNS" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --seed "$SEED" --device 0 \
        2>&1 | tee "$log"
}

# Main loop
TOTAL=${#MODELS[@]}
IDX=0

for model in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    echo "=== [$IDX/$TOTAL] $model ==="

    # Launch parallel tasks
    PIDS=()
    for ((i=0; i<NUM_TASKS; i++)); do
        run_task "$model" $i
        PIDS+=($!)
        [ $i -lt $((NUM_TASKS-1)) ] && sleep $TASK_DELAY
    done

    echo "Waiting for $model tasks..."

    # Wait and check results
    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || { echo "  [Task $((i+1))] FAILED"; FAILED=$((FAILED+1)); }
    done
    [ $FAILED -gt 0 ] && echo "WARNING: $model had $FAILED failed task(s)"

    merge_and_train "$model"
    echo "[OK] $model done."
    echo ""
done

# Summary
END=$(date +%s)
ELAPSED=$((END - START))
printf "\n=== All Done === (Elapsed: %dh %dm %ds)\n" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Results: $OUT_DIR"
echo "Logs:    $LOG_DIR"
for model in "${MODELS[@]}"; do
    echo "  - ${model}: ${OUT_DIR}/${model}_grid_merged.json"
done
