#!/bin/bash
# Repeated CARTE experiments for single-table and multi-table scenarios.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "$SCRIPT_DIR/../baseline" && pwd)"
RESULTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/results"

PYTHON="${PYTHON:-python}"
GPU_ID=1
NUM_RUNS=10
MAX_JOBS=1
SEED=42
NUM_MODEL=5
MODE="joint"
DATA_TAG=""
DATA_NAME="maryland"
TARGET_DATA_NAME="maryland"
SOURCE_DATA_NAME="seattle"
MASK_BASENAME="maryland"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --mode NAME             Scenario: single or joint. Default: $MODE.
  --data_name NAME        CARTE single-table data name. Default: $DATA_NAME.
  --target_data_name NAME CARTE joint target table. Default: $TARGET_DATA_NAME.
  --source_data_name NAME Comma-separated CARTE joint source tables. Default: $SOURCE_DATA_NAME.
  --mask_basename NAME    Mask file basename, i.e. mask_<name>.pt. Default: $MASK_BASENAME.
  --data_tag NAME         Output directory tag. Auto-inferred when omitted.
  --gpu N          Physical GPU id exposed via CUDA_VISIBLE_DEVICES. Default: $GPU_ID.
  --num_runs N     Repeated runs. Default: $NUM_RUNS.
  --max_jobs N     Concurrent repeated runs on the selected GPU. Default: $MAX_JOBS.
  --seed N         Base random seed. Default: $SEED.
  --num_model N    CARTE ensemble models per run. Default: $NUM_MODEL.
  -h, --help       Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --data_name)
            DATA_NAME="$2"; shift 2 ;;
        --target_data_name)
            TARGET_DATA_NAME="$2"; shift 2 ;;
        --source_data_name)
            SOURCE_DATA_NAME="$2"; shift 2 ;;
        --mask_basename)
            MASK_BASENAME="$2"; shift 2 ;;
        --data_tag)
            DATA_TAG="$2"; shift 2 ;;
        --gpu)
            GPU_ID="$2"; shift 2 ;;
        --num_runs)
            NUM_RUNS="$2"; shift 2 ;;
        --max_jobs)
            MAX_JOBS="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --num_model)
            NUM_MODEL="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

case "$MODE" in
    single|joint) ;;
    multi|multitable) MODE="joint" ;;
    *)
        echo "Unknown --mode: $MODE" >&2
        usage
        exit 1 ;;
esac

if [[ "$MODE" == "single" ]]; then
    [[ -z "$DATA_TAG" ]] && DATA_TAG="$DATA_NAME"
    PY_SCRIPT="carte_single.py"
    MODEL_NAME="carte_single"
else
    [[ -z "$DATA_TAG" ]] && DATA_TAG="${TARGET_DATA_NAME}__src_${SOURCE_DATA_NAME//,/+}"
    PY_SCRIPT="carte_joint.py"
    MODEL_NAME="carte_joint"
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

RESULTS_DIR="$RESULTS_ROOT/transfer/carte/$MODE/$DATA_TAG"
LOG_DIR="$RESULTS_ROOT/logs/transfer/carte/$MODE/$DATA_TAG"
ARTIFACT_DIR="$RESULTS_ROOT/artifacts/transfer/carte/$MODE/$DATA_TAG"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$ARTIFACT_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_carte_${MODE}_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== CARTE ${MODE}: ${NUM_RUNS} Runs ==="
echo "Script    : $PY_SCRIPT"
echo "Data tag  : $DATA_TAG"
echo "Data name : $DATA_NAME"
echo "Target    : $TARGET_DATA_NAME"
echo "Sources   : $SOURCE_DATA_NAME"
echo "Mask      : mask_${MASK_BASENAME}.pt"
echo "GPU       : physical cuda:$GPU_ID exposed as cuda:0"
echo "Max jobs  : $MAX_JOBS"
echo "Num model : $NUM_MODEL"
echo "Seed      : $SEED"
echo "Results   : $RESULTS_DIR"
echo "Logs      : $LOG_DIR"
echo "Artifacts : $ARTIFACT_DIR"
echo "Main log  : $MAIN_LOG"
echo ""

START=$(date +%s)
RUN_JSONS=()
PIDS=()

run_one() {
    local run_id=$1 run_seed=$2 run_json=$3 run_log=$4 model_path=$5
    echo "--- [Run $run_id/$NUM_RUNS] seed=$run_seed ---"
    (
        cd "$BASELINE_DIR"
        if [[ "$MODE" == "single" ]]; then
            "$PYTHON" "$PY_SCRIPT" \
                --data_name "$DATA_NAME" \
                --mask_basename "$MASK_BASENAME" \
                --num_model "$NUM_MODEL" \
                --device cuda:0 \
                --num_runs 1 \
                --seed "$run_seed" \
                --save_results "$run_json"
        else
            "$PYTHON" "$PY_SCRIPT" \
                --target_data_name "$TARGET_DATA_NAME" \
                --source_data_name "$SOURCE_DATA_NAME" \
                --mask_basename "$MASK_BASENAME" \
                --num_model "$NUM_MODEL" \
                --device cuda:0 \
                --seed "$run_seed" \
                --model_output "$model_path" \
                --save_results "$run_json"
        fi
    ) 2>&1 | tee "$run_log"
    echo "[OK] run $run_id saved -> $run_json"
    echo ""
}

wait_for_slot() {
    while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
        sleep 5
    done
}

random_seed() {
    "$PYTHON" -c 'import secrets; print(secrets.randbelow(2**31 - 1))'
}

for ((i=1; i<=NUM_RUNS; i++)); do
    RUN_SEED="$(random_seed)"
    RUN_JSON="$RESULTS_DIR/run_${i}_seed${RUN_SEED}_${TS}.json"
    RUN_LOG="$LOG_DIR/run_${i}_seed${RUN_SEED}_${TS}.log"
    MODEL_PATH="$ARTIFACT_DIR/run_${i}_seed${RUN_SEED}/${MODEL_NAME}.pkl"
    RUN_JSONS+=("$RUN_JSON")

    wait_for_slot
    run_one "$i" "$RUN_SEED" "$RUN_JSON" "$RUN_LOG" "$MODEL_PATH" &
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done
if [[ "$FAILED" -gt 0 ]]; then
    echo "ERROR: $FAILED CARTE run(s) failed. See logs in $LOG_DIR" >&2
    exit 1
fi

SUMMARY_JSON="$RESULTS_DIR/summary_${NUM_RUNS}runs_${TS}.json"
"$PYTHON" - "$SUMMARY_JSON" "$MAIN_LOG" "$MODEL_NAME" "$MODE" "$DATA_TAG" "$DATA_NAME" "$TARGET_DATA_NAME" "$SOURCE_DATA_NAME" "$MASK_BASENAME" "${RUN_JSONS[@]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

summary_path = Path(sys.argv[1])
main_log = sys.argv[2]
model_name = sys.argv[3]
mode = sys.argv[4]
dataset_tag = sys.argv[5]
data_name = sys.argv[6]
target_data_name = sys.argv[7]
source_data_name = sys.argv[8]
mask_basename = sys.argv[9]
run_paths = [Path(p) for p in sys.argv[10:]]
runs = []
for idx, path in enumerate(run_paths, start=1):
    with path.open() as f:
        data = json.load(f)
    metrics = data["metrics"]
    runs.append({
        "run_id": idx,
        "path": str(path),
        "seed": data["seed"],
        "runtime": data["runtime"],
        "model_output": data.get("model_output"),
        **metrics,
    })

metric_names = sorted({
    key
    for run in runs
    for key, value in run.items()
    if key not in {"run_id", "path", "seed", "model_output"} and value is not None
})
stats = {}
for name in metric_names:
    values = np.array([r[name] for r in runs], dtype=float)
    stats[f"{name}_mean"] = float(values.mean())
    stats[f"{name}_std"] = float(values.std())
    stats[f"{name}_min"] = float(values.min())
    stats[f"{name}_max"] = float(values.max())

output = {
    "model": model_name,
    "task": "classification",
    "mode": mode,
    "dataset": dataset_tag,
    "data_name": data_name if mode == "single" else None,
    "target_data_name": target_data_name,
    "source_datasets": [s for s in source_data_name.split(",") if s],
    "mask_basename": mask_basename,
    "num_runs": len(runs),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "log_path": main_log,
    "individual_runs": runs,
    "statistics": stats,
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Summary saved -> {summary_path}")
if "test_acc_mean" in stats:
    print(f"Test Acc: {stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.4f}")
PY

END=$(date +%s)
ELAPSED=$((END - START))
printf "\n=== All Done === (Elapsed: %dh %dm %ds)\n" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Summary:   $SUMMARY_JSON"
echo "Results:   $RESULTS_DIR"
echo "Logs:      $LOG_DIR"
echo "Artifacts: $ARTIFACT_DIR"
