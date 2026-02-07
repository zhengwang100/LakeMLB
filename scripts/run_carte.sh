#!/bin/bash
# Run CARTE experiments (single-table / multi-table) N times each.
# Usage:
#   ./run_carte.sh              # run both modes
#   ./run_carte.sh single       # single-table only
#   ./run_carte.sh multi        # multi-table only

set -euo pipefail

################ Config ################

BASELINE_DIR="$(cd "$(dirname "$0")/../baseline" && pwd)"
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
LOG_DIR="${RESULTS_DIR}/carte"
NUM_RUNS=5
TS=$(date +"%Y%m%d_%H%M%S")

# Task definitions: name|script
TASKS=(
    "single|carte_single.py"
    "multi|carte_joint.py"
)

########################################

MODE="${1:-all}"  # default: run all

mkdir -p "$LOG_DIR"

run_task() {
    local name=$1 script=$2
    local total_log="${LOG_DIR}/${name}_total_${TS}.log"

    echo "=== CARTE ${name}-table: ${NUM_RUNS} runs ==="
    echo "Script: ${BASELINE_DIR}/${script}"
    echo "Start:  $(date)"
    echo ""

    for i in $(seq 1 "$NUM_RUNS"); do
        local run_log="${LOG_DIR}/${name}_run${i}_${TS}.log"
        echo "[${i}/${NUM_RUNS}] Running..."

        cd "$BASELINE_DIR"
        if python "$script" 2>&1 | tee -a "$run_log" "$total_log"; then
            echo "  OK (run $i)"
        else
            echo "  FAIL (run $i, exit: ${PIPESTATUS[0]})"
        fi
    done

    echo ""
    echo "[Done] ${name}-table | Log: $total_log"
    echo ""
}

for entry in "${TASKS[@]}"; do
    IFS='|' read -r name script <<< "$entry"
    case "$MODE" in
        all)    run_task "$name" "$script" ;;
        "$name") run_task "$name" "$script" ;;
    esac
done

echo "=== All Done === $(date)"

