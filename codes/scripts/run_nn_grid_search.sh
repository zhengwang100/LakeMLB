#!/bin/bash
# Grid search + repeated evaluation for tabular neural network baselines.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "$SCRIPT_DIR/../baseline" && pwd)"
RESULTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/results"

PYTHON="${PYTHON:-python}"
GPU_ID=1
DATASET="mstraffic"
TABLE_IDX=0
DATA_TAG=""
MODELS=("fttransformer" "tabtransformer" "excelformer" "saint" "tromptnet")

# Parallelism. Increase NUM_TASKS only when GPU memory allows it.
NUM_TASKS=2
TASK_DELAY=5

# Grid search space from the experiment spec.
GRID_HIDDEN="32,64,128"
GRID_LAYERS="2,3,4"
GRID_LR="1e-3,5e-4,1e-4"
GRID_WD="1e-4,5e-4,1e-3"
GRID_BS="256"
GRID_EPOCHS=500
GRID_PATIENCE=10
GRAD_ACCUM=1

# Final repeated training with best grid config.
FINAL_EPOCHS=500
FINAL_PATIENCE=10
NUM_RUNS=10
SEED=42

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --dataset NAME       LakeMLB dataset name. Default: mstraffic.
  --table_idx N        Table index inside --dataset. Default: 0.
  --data_tag NAME      Output directory tag. Auto-inferred when omitted.
  --gpu N              Physical GPU id exposed via CUDA_VISIBLE_DEVICES. Default: $GPU_ID.
  --num_tasks N        Number of parallel grid-search shards. Default: $NUM_TASKS.
  --num_runs N         Repeated runs after grid search. Default: $NUM_RUNS.
  --seed N             Base random seed. Default: $SEED.
  --models LIST        Comma-separated models. Default: fttransformer,tabtransformer,excelformer,saint,tromptnet.
  --grid_hidden LIST   Hidden dimensions for grid search. Default: $GRID_HIDDEN.
  --grid_layers LIST   Layer counts for grid search. Default: $GRID_LAYERS.
  --grid_lr LIST       Learning rates for grid search. Default: $GRID_LR.
  --grid_wd LIST       Weight decays for grid search. Default: $GRID_WD.
  --grid_bs LIST       Batch sizes for grid search. Default: $GRID_BS.
  --grid_epochs N      Maximum epochs per grid configuration. Default: $GRID_EPOCHS.
  --grid_patience N    Grid-search early-stopping patience. Default: $GRID_PATIENCE.
  --final_epochs N     Maximum epochs for each final run. Default: $FINAL_EPOCHS.
  --final_patience N   Final-training early-stopping patience. Default: $FINAL_PATIENCE.
  -h, --help           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"; shift 2 ;;
        --table_idx)
            TABLE_IDX="$2"; shift 2 ;;
        --data_tag)
            DATA_TAG="$2"; shift 2 ;;
        --gpu)
            GPU_ID="$2"; shift 2 ;;
        --num_tasks)
            NUM_TASKS="$2"; shift 2 ;;
        --num_runs)
            NUM_RUNS="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --models)
            IFS=',' read -r -a MODELS <<< "$2"; shift 2 ;;
        --grid_hidden)
            GRID_HIDDEN="$2"; shift 2 ;;
        --grid_layers)
            GRID_LAYERS="$2"; shift 2 ;;
        --grid_lr)
            GRID_LR="$2"; shift 2 ;;
        --grid_wd)
            GRID_WD="$2"; shift 2 ;;
        --grid_bs)
            GRID_BS="$2"; shift 2 ;;
        --grid_epochs)
            GRID_EPOCHS="$2"; shift 2 ;;
        --grid_patience)
            GRID_PATIENCE="$2"; shift 2 ;;
        --final_epochs)
            FINAL_EPOCHS="$2"; shift 2 ;;
        --final_patience)
            FINAL_PATIENCE="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

if [[ -z "$DATA_TAG" ]]; then
    case "${DATASET}:${TABLE_IDX}" in
        mstraffic:0) DATA_TAG="mstraffic_maryland" ;;
        mstraffic:1) DATA_TAG="mstraffic_seattle" ;;
        mstraffic:2) DATA_TAG="mstraffic_da" ;;
        mstraffic:3) DATA_TAG="mstraffic_fa" ;;
        ncbuilding:0) DATA_TAG="ncbuilding_newyork" ;;
        ncbuilding:1) DATA_TAG="ncbuilding_chicago" ;;
        ncbuilding:2) DATA_TAG="ncbuilding_da" ;;
        ncbuilding:3) DATA_TAG="ncbuilding_fa" ;;
        nctaxi:0) DATA_TAG="nctaxi_newyork_taxi" ;;
        nctaxi:1) DATA_TAG="nctaxi_chicago_taxi" ;;
        nctaxi:2) DATA_TAG="nctaxi_da" ;;
        nctaxi:3) DATA_TAG="nctaxi_fa" ;;
        dsmusic:0) DATA_TAG="dsmusic_discogs" ;;
        dsmusic:1) DATA_TAG="dsmusic_spotify" ;;
        dsmusic:2) DATA_TAG="dsmusic_da" ;;
        dsmusic:3) DATA_TAG="dsmusic_fa" ;;
        agbooks:0) DATA_TAG="agbooks_amazon" ;;
        agbooks:1) DATA_TAG="agbooks_goodreads" ;;
        agbooks:2) DATA_TAG="agbooks_da" ;;
        agbooks:3) DATA_TAG="agbooks_fa" ;;
        nnstocks:0) DATA_TAG="nnstocks_nnlist" ;;
        nnstocks:1) DATA_TAG="nnstocks_nnwiki" ;;
        nnstocks:2) DATA_TAG="nnstocks_da" ;;
        nnstocks:3) DATA_TAG="nnstocks_fa" ;;
        *) DATA_TAG="${DATASET}_table${TABLE_IDX}" ;;
    esac
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

GRID_ROOT="$RESULTS_ROOT/grid_search/tnns/$DATA_TAG"
RESULTS_DIR="$RESULTS_ROOT/tnns/$DATA_TAG"
LOG_DIR="$RESULTS_ROOT/logs/tnns/$DATA_TAG"
ARTIFACT_DIR="$RESULTS_ROOT/artifacts/tnns"
mkdir -p "$GRID_ROOT" "$RESULTS_DIR" "$LOG_DIR" "$ARTIFACT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/run_tnns_grid_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== TNNS Grid Search + ${NUM_RUNS} Runs ==="
echo "Dataset   : $DATASET[$TABLE_IDX] ($DATA_TAG)"
echo "GPU       : physical cuda:$GPU_ID exposed as cuda:0"
echo "Tasks     : $NUM_TASKS"
echo "Models    : ${MODELS[*]}"
echo "Grid      : hidden=$GRID_HIDDEN layers=$GRID_LAYERS lr=$GRID_LR wd=$GRID_WD bs=$GRID_BS"
echo "Grid train: epochs=$GRID_EPOCHS patience=$GRID_PATIENCE grad_accum=$GRAD_ACCUM"
echo "Final     : epochs=$FINAL_EPOCHS patience=$FINAL_PATIENCE runs=$NUM_RUNS seed=$SEED"
echo "Grid dir  : $GRID_ROOT"
echo "Results   : $RESULTS_DIR"
echo "Logs      : $LOG_DIR"
echo "Artifacts : $ARTIFACT_DIR/$DATA_TAG"
echo "Main log  : $MAIN_LOG"
echo ""

START=$(date +%s)

run_task() {
    local model=$1 tid=$2 model_grid_dir=$3
    local log="${LOG_DIR}/${model}_task_${tid}_${TS}.log"
    echo "  [Task $((tid+1))/$NUM_TASKS] Starting..."
    "$PYTHON" "${BASELINE_DIR}/tnns_test.py" \
        --dataset "$DATASET" --table_idx "$TABLE_IDX" \
        --model "$model" --grid \
        --task_id "$tid" --num_tasks "$NUM_TASKS" \
        --grid_hidden "$GRID_HIDDEN" --grid_layers "$GRID_LAYERS" \
        --grid_lr "$GRID_LR" --grid_wd "$GRID_WD" --grid_bs "$GRID_BS" \
        --grid_epochs "$GRID_EPOCHS" --grid_patience "$GRID_PATIENCE" \
        --grid_output_dir "$model_grid_dir" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --log_dir "$LOG_DIR" \
        --artifact_dir "$ARTIFACT_DIR" \
        --device cuda:0 --seed "$SEED" \
        > "$log" 2>&1 &
    echo "    Log: $log"
}

merge_and_train() {
    local model=$1 model_grid_dir=$2
    local model_result_dir="$RESULTS_DIR/$model"
    mkdir -p "$model_result_dir"
    local log="${LOG_DIR}/${model}_merge_${TS}.log"
    echo "Merging $model results + final training..."
    "$PYTHON" "${BASELINE_DIR}/tnns_test.py" \
        --dataset "$DATASET" --table_idx "$TABLE_IDX" \
        --model "$model" --merge_results \
        --grid_output_dir "$model_grid_dir" \
        --grid_epochs "$GRID_EPOCHS" --grid_patience "$GRID_PATIENCE" \
        --epochs "$FINAL_EPOCHS" --patience "$FINAL_PATIENCE" \
        --num_runs "$NUM_RUNS" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --log_dir "$LOG_DIR" \
        --artifact_dir "$ARTIFACT_DIR" \
        --save_results "$model_result_dir/final_${NUM_RUNS}runs_${TS}.json" \
        --seed "$SEED" --device cuda:0 \
        2>&1 | tee "$log"
}

TOTAL=${#MODELS[@]}
IDX=0

for model in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    MODEL_GRID_DIR="$GRID_ROOT/$model/$TS"
    mkdir -p "$MODEL_GRID_DIR"
    echo "=== [$IDX/$TOTAL] $model ==="

    PIDS=()
    for ((i=0; i<NUM_TASKS; i++)); do
        run_task "$model" "$i" "$MODEL_GRID_DIR"
        PIDS+=($!)
        [ $i -lt $((NUM_TASKS-1)) ] && sleep "$TASK_DELAY"
    done

    echo "Waiting for $model tasks..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || { echo "  [Task $((i+1))] FAILED"; FAILED=$((FAILED+1)); }
    done
    if [[ "$FAILED" -gt 0 ]]; then
        echo "ERROR: $model had $FAILED failed task(s). See $LOG_DIR" >&2
        exit 1
    fi

    merge_and_train "$model" "$MODEL_GRID_DIR"
    echo "[OK] $model done."
    echo ""
done

END=$(date +%s)
ELAPSED=$((END - START))
printf "\n=== All Done === (Elapsed: %dh %dm %ds)\n" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Grid:      $GRID_ROOT"
echo "Results:   $RESULTS_DIR"
echo "Logs:      $LOG_DIR"
echo "Artifacts: $ARTIFACT_DIR/$DATA_TAG"
