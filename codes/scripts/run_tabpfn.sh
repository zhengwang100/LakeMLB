#!/bin/bash
# Repeated TabPFN foundation-model experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "$SCRIPT_DIR/../baseline" && pwd)"
RESULTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/results"

PYTHON="${PYTHON:-python}"
GPU_ID=1
NUM_RUNS=10
MAX_JOBS=1
SEED=42
DATASET="mstraffic"
TABLE_IDX=0
DATA_TAG=""
N_ESTIMATORS=8
MODEL_PATH=""

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --dataset NAME    LakeMLB dataset name. Default: $DATASET.
  --table_idx N     Table index inside --dataset. Default: $TABLE_IDX.
  --data_tag NAME   Output directory tag. Auto-inferred when omitted.
  --gpu N           Physical GPU id exposed via CUDA_VISIBLE_DEVICES. Default: $GPU_ID.
  --num_runs N      Repeated runs. Default: $NUM_RUNS.
  --max_jobs N      Concurrent repeated runs on the selected GPU. Default: $MAX_JOBS.
  --seed N          Base random seed. Default: $SEED.
  --n_estimators N  TabPFN n_estimators. Default: $N_ESTIMATORS.
  --model_path PATH Optional local TabPFN v3 checkpoint path.
  -h, --help        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --table_idx) TABLE_IDX="$2"; shift 2 ;;
        --data_tag) DATA_TAG="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --num_runs) NUM_RUNS="$2"; shift 2 ;;
        --max_jobs) MAX_JOBS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --n_estimators) N_ESTIMATORS="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
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

RESULTS_DIR="$RESULTS_ROOT/foundation/tabpfn/$DATA_TAG"
LOG_DIR="$RESULTS_ROOT/logs/foundation/tabpfn/$DATA_TAG"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_tabpfn_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== TabPFN Foundation Model: ${NUM_RUNS} Runs ==="
echo "Dataset  : $DATASET[$TABLE_IDX] ($DATA_TAG)"
echo "GPU      : physical cuda:$GPU_ID exposed as cuda:0"
echo "Max jobs : $MAX_JOBS"
echo "Seed     : $SEED"
echo "n_estimators: $N_ESTIMATORS"
echo "Model path  : ${MODEL_PATH:-auto}"
echo "Results  : $RESULTS_DIR"
echo "Logs     : $LOG_DIR"
echo "Main log : $MAIN_LOG"
echo ""

START=$(date +%s)
RUN_JSONS=()
PIDS=()

wait_for_slot() {
    while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
        sleep 5
    done
}

random_seed() {
    "$PYTHON" -c 'import secrets; print(secrets.randbelow(2**31 - 1))'
}

run_one() {
    local run_id=$1 run_seed=$2 run_json=$3 run_log=$4
    local extra_args=()
    if [[ -n "$MODEL_PATH" ]]; then
        extra_args+=(--model_path "$MODEL_PATH")
    fi
    echo "--- [Run $run_id/$NUM_RUNS] seed=$run_seed ---"
    (
        cd "$BASELINE_DIR"
        "$PYTHON" tabpfnv2_extend.py \
            --dataset "$DATASET" \
            --table_idx "$TABLE_IDX" \
            --num_runs 1 \
            --seed "$run_seed" \
            --device cuda:0 \
            --n_estimators "$N_ESTIMATORS" \
            "${extra_args[@]}" \
            --save_results "$run_json"
    ) 2>&1 | tee "$run_log"
    echo "[OK] run $run_id saved -> $run_json"
    echo ""
}

for ((i=1; i<=NUM_RUNS; i++)); do
    RUN_SEED="$(random_seed)"
    RUN_JSON="$RESULTS_DIR/run_${i}_seed${RUN_SEED}_${TS}.json"
    RUN_LOG="$LOG_DIR/run_${i}_seed${RUN_SEED}_${TS}.log"
    RUN_JSONS+=("$RUN_JSON")
    wait_for_slot
    run_one "$i" "$RUN_SEED" "$RUN_JSON" "$RUN_LOG" &
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done
if [[ "$FAILED" -gt 0 ]]; then
    echo "ERROR: $FAILED TabPFN run(s) failed. See logs in $LOG_DIR" >&2
    exit 1
fi

SUMMARY_JSON="$RESULTS_DIR/summary_${NUM_RUNS}runs_${TS}.json"
"$PYTHON" "$BASELINE_DIR/merge_foundation_results.py" \
    --model tabpfn \
    --dataset "$DATA_TAG" \
    --log_path "$MAIN_LOG" \
    --output "$SUMMARY_JSON" \
    --inputs "${RUN_JSONS[@]}"

END=$(date +%s)
ELAPSED=$((END - START))
printf "\n=== All Done === (Elapsed: %dh %dm %ds)\n" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Summary: $SUMMARY_JSON"
echo "Results: $RESULTS_DIR"
echo "Logs:    $LOG_DIR"
