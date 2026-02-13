#!/bin/bash
# Grid search + repeated evaluation for tree-based models (XGBoost, CatBoost, LightGBM).

set -euo pipefail

cd "$(dirname "$0")/../baseline"
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$RESULTS_DIR/tree_models"

NUM_RUNS=10
SEED=42
GRID_PATIENCE=50
TS=$(date +"%Y%m%d_%H%M%S")
MODELS=("xgboost" "catboost" "lightgbm")

echo "=== Tree Models: Grid Search + ${NUM_RUNS} Runs ==="
echo "Models: ${MODELS[*]} | Seed: $SEED | Patience: $GRID_PATIENCE"
echo ""

TOTAL=${#MODELS[@]}
IDX=0

for model in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    echo "--- [$IDX/$TOTAL] $model ---"

    python tree_models.py \
        --model "$model" \
        --grid \
        --num_runs "$NUM_RUNS" \
        --seed "$SEED" \
        --grid_patience "$GRID_PATIENCE" \
        --save_results "$RESULTS_DIR/tree_models/${model}_grid_final_${TS}.json"

    echo "[OK] $model done."
    echo ""
done

echo "=== All Done ==="
echo "Results: $RESULTS_DIR/tree_models/"
for model in "${MODELS[@]}"; do
    echo "  - ${model}_grid_final_${TS}.json"
done
