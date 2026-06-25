#!/bin/bash
# =============================================================================
# Single-table classification benchmark on stocks_wiki_llm_1nn
# (NNStocksDataset table_idx=4, mask_nnlist.pt)
#
# Methods:
#   Tree  : XGBoost / CatBoost / LightGBM
#   TNN   : FTTransformer / TabTransformer / ExcelFormer / SAINT / TromptNet
#   TabPFN: TabPFN v2 + ManyClassClassifier
#   TransTab: transtab_single (single-table)
#   CARTE : carte_single (single-table)
#
# Results: $RESULTS_DIR/  (one JSON per method + summary.csv)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$SCRIPT_DIR/../baseline"
RESULTS_DIR="$SCRIPT_DIR/../results/nnstocks_tfidf1nn_cls_benchmark"
mkdir -p "$RESULTS_DIR"

# ── settings (edit here) ──────────────────────────────────────────────────────
DEVICE="cuda:1"
SEED=42

# Table index in NNStocksDataset (8 = stocks_wiki_tfidf_1nn)
TABLE_IDX=8
DATA_NAME="stocks_wiki_tfidf_1nn"

TREE_RUNS=10
TNN_RUNS=10
TABPFN_RUNS=5
TRANSTAB_RUNS=5
CARTE_RUNS=5

# Parallel batches per method (1 = sequential)
TREE_PARALLEL=5
TNN_PARALLEL=2
TABPFN_PARALLEL=1
TRANSTAB_PARALLEL=1
CARTE_PARALLEL=1
# ──────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  NNStocks 1-NN Classification Benchmark  [$DATA_NAME]"
echo "  Results dir  : $RESULTS_DIR"
echo "  Device       : $DEVICE"
echo "  TABLE_IDX    : $TABLE_IDX"
echo "================================================================"
echo ""

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
        local RUNS=$(( REMAINING < RPBATCH ? REMAINING : RPBATCH ))
        local BSEED=$(( SEED + i * 100000 ))
        local BOUT="${OUT_BASE}_b${i}.json"
        _LB_BOUTS+=("$BOUT")
        python "$SCRIPT" \
            --num_runs "$RUNS" --seed "$BSEED" \
            "$@" \
            --save_results "$BOUT" \
            > "${OUT_BASE}_b${i}.log" 2>&1 &
        _LB_PIDS+=($!)
        echo "    batch $i : runs=$RUNS  seed=$BSEED  (PID=$!)"
    done
}

# ── helper: wait + merge → final JSON ────────────────────────────────────────
wait_and_merge() {
    local -n _WM_PIDS=$1
    local -n _WM_BOUTS=$2
    local OUTPUT="$3"
    for pid in "${_WM_PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  FAILED (PID=$pid) — see ${OUTPUT%.json}_b*.log"; exit 1
        fi
    done
    python "$SCRIPT_DIR/merge_batch_results.py" \
        --inputs "${_WM_BOUTS[@]}" --output "$OUTPUT"
    rm -f "${_WM_BOUTS[@]}" "${_WM_BOUTS[@]/%.json/.log}"
}

# ── 1. Tree models ────────────────────────────────────────────────────────────
for MODEL in xgboost catboost lightgbm; do
    echo "[Tree] $MODEL  ($TREE_RUNS runs, $TREE_PARALLEL parallel batches) ..."
    PIDS=(); BOUTS=()
    launch_batches PIDS BOUTS \
        "$RESULTS_DIR/${MODEL}" "$TREE_RUNS" "$TREE_PARALLEL" \
        tree_models.py \
        --model "$MODEL" --table_idx "$TABLE_IDX" --device 1
    wait_and_merge PIDS BOUTS "$RESULTS_DIR/${MODEL}.json"
    echo "[Tree] $MODEL done."
    echo ""
done

# ── 2. TNN models ─────────────────────────────────────────────────────────────
for MODEL in fttransformer tabtransformer excelformer saint tromptnet; do
    echo "[TNN] $MODEL  ($TNN_RUNS runs, $TNN_PARALLEL parallel batches) ..."
    PIDS=(); BOUTS=()
    launch_batches PIDS BOUTS \
        "$RESULTS_DIR/tnn_${MODEL}" "$TNN_RUNS" "$TNN_PARALLEL" \
        tnns_test.py \
        --model "$MODEL" --table_idx "$TABLE_IDX" \
        --epochs 200 --patience 50 --device "$DEVICE"
    wait_and_merge PIDS BOUTS "$RESULTS_DIR/tnn_${MODEL}.json"
    echo "[TNN] $MODEL done."
    echo ""
done

# ── 3. TabPFN v2 (ManyClass) ─────────────────────────────────────────────────
echo "[TabPFN] ($TABPFN_RUNS runs, $TABPFN_PARALLEL parallel batches) ..."
PIDS=(); BOUTS=()
launch_batches PIDS BOUTS \
    "$RESULTS_DIR/tabpfn" "$TABPFN_RUNS" "$TABPFN_PARALLEL" \
    tabpfnv2_extend.py \
    --table_idx "$TABLE_IDX" --device "$DEVICE"
wait_and_merge PIDS BOUTS "$RESULTS_DIR/tabpfn.json"
echo "[TabPFN] done."
echo ""

# ── 4. TransTab single ────────────────────────────────────────────────────────
echo "[TransTab] ($TRANSTAB_RUNS runs, $TRANSTAB_PARALLEL parallel batches) ..."
PIDS=(); BOUTS=()
launch_batches PIDS BOUTS \
    "$RESULTS_DIR/transtab_single" "$TRANSTAB_RUNS" "$TRANSTAB_PARALLEL" \
    transtab_single.py \
    --data_name "$DATA_NAME" --num_epoch 100 --patience 20 --device "$DEVICE"
wait_and_merge PIDS BOUTS "$RESULTS_DIR/transtab_single.json"
echo "[TransTab] done."
echo ""

# ── 5. CARTE single ───────────────────────────────────────────────────────────
echo "[CARTE] ($CARTE_RUNS runs, $CARTE_PARALLEL parallel batches) ..."
PIDS=(); BOUTS=()
launch_batches PIDS BOUTS \
    "$RESULTS_DIR/carte_single" "$CARTE_RUNS" "$CARTE_PARALLEL" \
    carte_single.py \
    --data_name "$DATA_NAME" --mask_basename nnlist \
    --num_model 5 --device "$DEVICE"
wait_and_merge PIDS BOUTS "$RESULTS_DIR/carte_single.json"
echo "[CARTE] done."
echo ""

# ── 6. Aggregate into CSV ─────────────────────────────────────────────────────
echo "================================================================"
echo "  Aggregating results → $RESULTS_DIR/summary.csv"
echo "================================================================"
python "$SCRIPT_DIR/aggregate_cls_results.py" \
    --results_dir "$RESULTS_DIR" \
    --output_csv  "$RESULTS_DIR/summary.csv"

echo ""
echo "All done!  Summary: $RESULTS_DIR/summary.csv"
