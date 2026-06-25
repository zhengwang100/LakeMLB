#!/bin/bash
# =============================================================================
# TabICL classification benchmark on NNStocks 1nn/2nn/4nn/8nn
# 5 runs per table, results aggregated to summary.csv
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$SCRIPT_DIR/../baseline"
RESULTS_DIR="$SCRIPT_DIR/../results/tabicl_nnstocks_benchmark"
mkdir -p "$RESULTS_DIR"

DEVICE="cuda:0"
SEED=42
RUNS=5

echo "================================================================"
echo "  TabICL NNStocks Benchmark  (1nn / 2nn / 4nn / 8nn / tfidf_1nn)"
echo "  Results dir : $RESULTS_DIR"
echo "  Device      : $DEVICE   Runs: $RUNS"
echo "================================================================"
echo ""

cd "$BASELINE"

# ── run one table ─────────────────────────────────────────────────────────────
run_table() {
    local TABLE_IDX=$1
    local DATA_TAG=$2
    local OUT="$RESULTS_DIR/tabicl_${DATA_TAG}.json"
    echo "[TabICL] $DATA_TAG  (table_idx=$TABLE_IDX, runs=$RUNS) ..."
    python tabicl_nnstocks.py \
        --table_idx   "$TABLE_IDX" \
        --num_runs    "$RUNS" \
        --seed        "$SEED" \
        --device      "$DEVICE" \
        --save_results "$OUT" \
        > "$RESULTS_DIR/tabicl_${DATA_TAG}.log" 2>&1
    echo "[TabICL] $DATA_TAG done."
    echo ""
}

run_table 4 "1nn"
run_table 5 "2nn"
run_table 6 "4nn"
run_table 7 "8nn"
run_table 8 "tfidf_1nn"

# ── aggregate ─────────────────────────────────────────────────────────────────
echo "Aggregating results..."
python - << 'PYEOF'
import json, os, csv, numpy as np

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'tabicl_nnstocks_benchmark')

rows = []
for tag in ['1nn', '2nn', '4nn', '8nn', 'tfidf_1nn']:
    p = os.path.join(results_dir, f'tabicl_{tag}.json')
    if not os.path.exists(p):
        print(f"  MISSING: {p}")
        continue
    with open(p) as f:
        d = json.load(f)
    s = d['statistics']
    rows.append({
        'Dataset':  d['dataset'],
        'Num_runs': d['num_runs'],
        'Acc_mean': round(s['test_acc_mean'], 4),
        'Acc_std':  round(s['test_acc_std'],  4),
        'Acc':      f"{s['test_acc_mean']:.4f}±{s['test_acc_std']:.4f}",
    })

csv_path = os.path.join(results_dir, 'summary.csv')
if rows:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Dataset':<30} {'Runs':>5}  {'Acc (mean±std)'}")
    print('-' * 55)
    for r in rows:
        print(f"{r['Dataset']:<30} {r['Num_runs']:>5}  {r['Acc']}")
    print(f"\nCSV saved → {csv_path}")
PYEOF

echo ""
echo "All done!  Summary: $RESULTS_DIR/summary.csv"
