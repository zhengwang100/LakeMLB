#!/bin/bash
# Run TabICL classification script N times and report accuracy statistics.

set -euo pipefail

PYTHON="/home/pfy/.conda/envs/lake/bin/python"
SCRIPT="tabicl_clf.py" # tabpfnv2.py
NUM_RUNS=10
OUTPUT_DIR="../results/fundation_models"
PREFIX="tabicl"

cd "$(dirname "$0")/../baseline"
mkdir -p "$OUTPUT_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
SUMMARY="${OUTPUT_DIR}/${PREFIX}_summary_${TS}.txt"

log() { echo "$*" | tee -a "$SUMMARY"; }

log "=== TabICL Repeated Runs ==="
log "Script: $SCRIPT | Runs: $NUM_RUNS"
log "Start: $(date)"
log ""

declare -a accs

for i in $(seq 1 "$NUM_RUNS"); do
    log "[${i}/${NUM_RUNS}] Running..."
    RUN_LOG="${OUTPUT_DIR}/${PREFIX}_run${i}_${TS}.log"

    if $PYTHON "$SCRIPT" > "$RUN_LOG" 2>&1; then
        acc=$(grep "Test Accuracy:" "$RUN_LOG" | tail -1 | awk '{print $3}')
        accs+=("$acc")
        log "  OK - Accuracy: $acc"
    else
        log "  FAIL"
    fi
done

log ""
log "=== Results ==="
for i in "${!accs[@]}"; do
    log "Run $((i+1)): ${accs[$i]}"
done

if [ ${#accs[@]} -gt 0 ]; then
    sum=0
    for a in "${accs[@]}"; do sum=$(echo "$sum + $a" | bc); done
    mean=$(echo "scale=4; $sum / ${#accs[@]}" | bc)
    log "Mean: $mean"
fi

log ""
log "End: $(date)"
log "Results saved in: $OUTPUT_DIR/"
