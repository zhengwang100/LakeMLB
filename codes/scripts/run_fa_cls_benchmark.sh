#!/bin/bash
# =============================================================================
# Single-table classification benchmark on nnstocks_fa
# (NNStocksDataset table_idx=3, mask_nnlist.pt)
#
# Methods: Tree (XGBoost/CatBoost/LightGBM) + TNN (5 models), 5 runs each
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$SCRIPT_DIR/../baseline"

TABLE_IDX=3
DATA_NAME="nnstocks_fa"
RESULTS_DIR="$SCRIPT_DIR/../results/${DATA_NAME}_cls_benchmark"
LOG_DIR="$SCRIPT_DIR/../results/logs/${DATA_NAME}_cls_benchmark"
ARTIFACT_DIR="$SCRIPT_DIR/../results/artifacts/${DATA_NAME}_cls_benchmark"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$ARTIFACT_DIR"

DEVICE="cuda:1"
SEED=42
RUNS=5

# Parallel batches (5 runs each, 1 parallel batch = sequential)
TREE_PARALLEL=5   # 5 batches × 1 run = 5 processes in parallel
TNN_PARALLEL=1    # 1 batch × 5 runs = sequential (GPU-heavy)

echo "================================================================"
echo "  NNStocks FA Classification Benchmark  [$DATA_NAME]"
echo "  Results dir : $RESULTS_DIR"
echo "  Device      : $DEVICE   TABLE_IDX: $TABLE_IDX   Runs: $RUNS"
echo "================================================================"
echo ""
TS=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_fa_cls_benchmark_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1
echo "  Main log    : $MAIN_LOG"
START=$(date +%s)

cd "$BASELINE"

# ── helper: launch batches in background ─────────────────────────────────────
launch_batches() {
    local -n _LB_PIDS=$1
    local -n _LB_BOUTS=$2
    local OUT_BASE="$3"
    local TOTAL_RUNS="$4"
    local NPAR="$5"
    local SCRIPT="$6"
    shift 6

    local RPBATCH=$(( (TOTAL_RUNS + NPAR - 1) / NPAR ))
    for ((i=0; i<NPAR; i++)); do
        local REMAINING=$(( TOTAL_RUNS - i * RPBATCH ))
        [ "$REMAINING" -le 0 ] && break
        local RUNS_I=$(( REMAINING < RPBATCH ? REMAINING : RPBATCH ))
        local BSEED
        BSEED="$(python -c 'import secrets; print(secrets.randbelow(2**31 - 1))')"
        local BOUT="${OUT_BASE}_b${i}.json"
        _LB_BOUTS+=("$BOUT")
        python "$SCRIPT" \
            --num_runs "$RUNS_I" --seed "$BSEED" \
            "$@" \
            --save_results "$BOUT" \
            > "$LOG_DIR/$(basename "${OUT_BASE}")_b${i}.log" 2>&1 &
        _LB_PIDS+=($!)
        echo "    batch $i : runs=$RUNS_I  seed=$BSEED  (PID=$!)"
    done
}

# ── helper: wait + merge → final JSON ────────────────────────────────────────
wait_and_merge() {
    local -n _WM_PIDS=$1
    local -n _WM_BOUTS=$2
    local OUTPUT="$3"
    for pid in "${_WM_PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  FAILED (PID=$pid) - see $LOG_DIR/$(basename "${OUTPUT%.json}")_b*.log"; exit 1
        fi
    done
    python "$SCRIPT_DIR/merge_batch_results.py" \
        --inputs "${_WM_BOUTS[@]}" --output "$OUTPUT"
    rm -f "${_WM_BOUTS[@]}"
}

# ── 1. Tree models ────────────────────────────────────────────────────────────
for MODEL in xgboost catboost lightgbm; do
    echo "[Tree] $MODEL  ($RUNS runs, $TREE_PARALLEL parallel batches) ..."
    PIDS=(); BOUTS=()
    launch_batches PIDS BOUTS \
        "$RESULTS_DIR/${MODEL}" "$RUNS" "$TREE_PARALLEL" \
        tree_models.py \
        --model "$MODEL" --table_idx "$TABLE_IDX" --device 1 \
        --log_dir "$LOG_DIR" --artifact_dir "$ARTIFACT_DIR"
    wait_and_merge PIDS BOUTS "$RESULTS_DIR/${MODEL}.json"
    echo "[Tree] $MODEL done."
    echo ""
done

# ── 2. TNN models ─────────────────────────────────────────────────────────────
for MODEL in fttransformer tabtransformer excelformer saint tromptnet; do
    echo "[TNN] $MODEL  ($RUNS runs, $TNN_PARALLEL parallel batches) ..."
    PIDS=(); BOUTS=()
    launch_batches PIDS BOUTS \
        "$RESULTS_DIR/tnn_${MODEL}" "$RUNS" "$TNN_PARALLEL" \
        tnns_test.py \
        --model "$MODEL" --table_idx "$TABLE_IDX" \
        --device "$DEVICE" \
        --epochs 500 --lr 1e-3 --wd 1e-4 --batch_size 256 --patience 10 \
        --log_dir "$LOG_DIR" --artifact_dir "$ARTIFACT_DIR"
    wait_and_merge PIDS BOUTS "$RESULTS_DIR/tnn_${MODEL}.json"
    echo "[TNN] $MODEL done."
    echo ""
done

# ── 3. Aggregate results ──────────────────────────────────────────────────────
echo "Aggregating results..."
python "$SCRIPT_DIR/aggregate_cls_results.py" \
    --results_dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/summary.csv"

echo ""
END=$(date +%s)
ELAPSED=$((END - START))
printf "Elapsed: %dh %dm %ds\n" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "All done!  Summary: $RESULTS_DIR/summary.csv"
echo "Logs: $LOG_DIR"
echo "Artifacts: $ARTIFACT_DIR"
