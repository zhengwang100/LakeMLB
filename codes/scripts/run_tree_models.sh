#!/bin/bash
# Grid search + repeated evaluation for tree-based classification models.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "$SCRIPT_DIR/../baseline" && pwd)"
RESULTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/results"

NUM_RUNS=10
NUM_THREADS=0
SEED=42
DEVICE=1
GRID_PATIENCE=50
DATASET="mstraffic"
TABLE_IDX=0
DATA_NAME=""
DATA_TAG=""
MODELS=("xgboost" "catboost" "lightgbm")

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --dataset NAME         LakeMLB dataset name. Default: mstraffic.
  --table_idx N          Table index inside --dataset. Default: 0.
  --data_name NAME       Parquet-based dataset name supported by tree_models.py.
                         When set, --dataset/--table_idx are ignored.
  --data_tag NAME        Output directory tag. Auto-inferred when omitted.
  --device N             CUDA device index passed to tree_models.py. Default: $DEVICE.
  --num_runs N           Repeated runs after grid search. Default: $NUM_RUNS.
  --num_threads N        Threads inside each tree learner. 0=library default/all. Default: $NUM_THREADS.
  --seed N               Base random seed. Default: $SEED.
  --grid_patience N      Early stopping rounds for grid search. Default: $GRID_PATIENCE.
  --models LIST          Comma-separated models. Default: xgboost,catboost,lightgbm.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"; shift 2 ;;
        --table_idx)
            TABLE_IDX="$2"; shift 2 ;;
        --data_name)
            DATA_NAME="$2"; shift 2 ;;
        --data_tag)
            DATA_TAG="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --num_runs)
            NUM_RUNS="$2"; shift 2 ;;
        --num_threads)
            NUM_THREADS="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --grid_patience)
            GRID_PATIENCE="$2"; shift 2 ;;
        --models)
            IFS=',' read -r -a MODELS <<< "$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

if [[ -z "$DATA_TAG" ]]; then
    if [[ -n "$DATA_NAME" ]]; then
        DATA_TAG="$DATA_NAME"
    else
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
            nnstocks:0) DATA_TAG="nnstocks_nnlist" ;;
            nnstocks:1) DATA_TAG="nnstocks_nnwiki" ;;
            nnstocks:2) DATA_TAG="nnstocks_da" ;;
            nnstocks:3) DATA_TAG="nnstocks_fa" ;;
            dsmusic:0) DATA_TAG="dsmusic_discogs" ;;
            dsmusic:1) DATA_TAG="dsmusic_spotify" ;;
            dsmusic:2) DATA_TAG="dsmusic_da" ;;
            dsmusic:3) DATA_TAG="dsmusic_fa" ;;
            agbooks:0) DATA_TAG="agbooks_amazon" ;;
            agbooks:1) DATA_TAG="agbooks_goodreads" ;;
            agbooks:2) DATA_TAG="agbooks_da" ;;
            agbooks:3) DATA_TAG="agbooks_fa" ;;
            *) DATA_TAG="${DATASET}_table${TABLE_IDX}" ;;
        esac
    fi
fi

RESULTS_DIR="$RESULTS_ROOT/tree_models/$DATA_TAG"
LOG_DIR="$RESULTS_ROOT/logs/tree_models/$DATA_TAG"
ARTIFACT_DIR="$RESULTS_ROOT/artifacts/tree_models"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$ARTIFACT_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_tree_models_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== Tree Models: Grid Search + ${NUM_RUNS} Runs ==="
echo "Dataset tag : $DATA_TAG"
if [[ -n "$DATA_NAME" ]]; then
    echo "Data name   : $DATA_NAME"
else
    echo "Dataset     : $DATASET"
    echo "Table idx   : $TABLE_IDX"
fi
echo "Models      : ${MODELS[*]}"
echo "Threads     : $NUM_THREADS"
echo "Seed        : $SEED"
echo "Device      : $DEVICE"
echo "Patience    : $GRID_PATIENCE"
echo "Results     : $RESULTS_DIR"
echo "Logs        : $LOG_DIR"
echo "Artifacts   : $ARTIFACT_DIR/$DATA_TAG"
echo "Main log    : $MAIN_LOG"
echo ""

START=$(date +%s)
cd "$BASELINE_DIR"

COMMON_ARGS=(
    --grid
    --num_runs "$NUM_RUNS"
    --num_threads "$NUM_THREADS"
    --seed "$SEED"
    --device "$DEVICE"
    --grid_patience "$GRID_PATIENCE"
    --log_dir "$LOG_DIR"
    --artifact_dir "$ARTIFACT_DIR"
)

if [[ -n "$DATA_NAME" ]]; then
    COMMON_ARGS+=(--data_name "$DATA_NAME")
else
    COMMON_ARGS+=(--dataset "$DATASET" --table_idx "$TABLE_IDX")
fi

TOTAL=${#MODELS[@]}
IDX=0

for model in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    MODEL_DIR="$RESULTS_DIR/$model"
    mkdir -p "$MODEL_DIR"

    echo "--- [$IDX/$TOTAL] $model ---"
    python tree_models.py \
        --model "$model" \
        "${COMMON_ARGS[@]}" \
        --grid_results "$MODEL_DIR/grid_search_${TS}.json" \
        --save_results "$MODEL_DIR/final_${NUM_RUNS}runs_${TS}.json"

    echo "[OK] $model done."
    echo ""
done

END=$(date +%s)
ELAPSED=$((END - START))

echo "=== All Done ==="
printf "Elapsed: %dh %dm %ds\n" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Results:   $RESULTS_DIR"
echo "Logs:      $LOG_DIR"
echo "Artifacts: $ARTIFACT_DIR/$DATA_TAG"
