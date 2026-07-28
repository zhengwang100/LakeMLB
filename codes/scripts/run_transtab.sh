#!/bin/bash
# Repeated TransTab experiments for single-table and transfer-learning scenarios.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "$SCRIPT_DIR/../baseline" && pwd)"
RESULTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/results"

PYTHON="${PYTHON:-python}"
GPU_ID=1
NUM_RUNS=10
MAX_JOBS=1
SEED=42
PRETRAIN_EPOCHS=100
FINETUNE_EPOCHS=100
MODE="transfer"
DATASET="mstraffic"
TABLE_IDX=0
AUX_DATASET=""
AUX_TABLE_IDX=""
DATA_TAG=""

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --mode NAME             Scenario: single, transfer, contrastive. Default: $MODE.
                          single      -> transtab_single.py
                          transfer    -> transtab_transfer.py, labeled auxiliary table
                          contrastive -> transtab_transfer_cl.py, unlabeled auxiliary table
  --dataset NAME          Dataset family. Defaults depend on --mode.
  --table_idx N           Task table index. Default: $TABLE_IDX.
  --aux_dataset NAME      Auxiliary dataset family for transfer/contrastive modes.
  --aux_table_idx N       Auxiliary table index for transfer/contrastive modes.
  --data_tag NAME         Output directory tag. Auto-inferred when omitted.
  --gpu N                Physical GPU id exposed via CUDA_VISIBLE_DEVICES. Default: $GPU_ID.
  --num_runs N           Repeated runs. Default: $NUM_RUNS.
  --max_jobs N           Concurrent repeated runs on the selected GPU. Default: $MAX_JOBS.
  --seed N               Base random seed. Default: $SEED.
  --pretrain_epochs N    TransTab pretraining epochs. Default: $PRETRAIN_EPOCHS.
  --finetune_epochs N    TransTab fine-tuning epochs. Default: $FINETUNE_EPOCHS.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --dataset)
            DATASET="$2"; shift 2 ;;
        --table_idx)
            TABLE_IDX="$2"; shift 2 ;;
        --aux_dataset)
            AUX_DATASET="$2"; shift 2 ;;
        --aux_table_idx)
            AUX_TABLE_IDX="$2"; shift 2 ;;
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
        --pretrain_epochs)
            PRETRAIN_EPOCHS="$2"; shift 2 ;;
        --finetune_epochs)
            FINETUNE_EPOCHS="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

case "$MODE" in
    single|transfer|contrastive) ;;
    cl) MODE="contrastive" ;;
    *)
        echo "Unknown --mode: $MODE" >&2
        usage
        exit 1 ;;
esac

if [[ "$MODE" == "single" ]]; then
    [[ -z "$DATA_TAG" ]] && DATA_TAG="${DATASET}_table${TABLE_IDX}"
    PY_SCRIPT="transtab_single.py"
    MODEL_NAME="transtab_single"
elif [[ "$MODE" == "transfer" ]]; then
    [[ -z "$AUX_DATASET" ]] && AUX_DATASET="$DATASET"
    [[ -z "$AUX_TABLE_IDX" ]] && AUX_TABLE_IDX=1
    [[ -z "$DATA_TAG" ]] && DATA_TAG="${DATASET}_table${TABLE_IDX}__aux_${AUX_DATASET}_table${AUX_TABLE_IDX}"
    PY_SCRIPT="transtab_transfer.py"
    MODEL_NAME="transtab_transfer"
elif [[ "$MODE" == "contrastive" ]]; then
    [[ -z "$AUX_DATASET" ]] && AUX_DATASET="$DATASET"
    [[ -z "$AUX_TABLE_IDX" ]] && AUX_TABLE_IDX=1
    [[ -z "$DATA_TAG" ]] && DATA_TAG="${DATASET}_table${TABLE_IDX}__aux_${AUX_DATASET}_table${AUX_TABLE_IDX}"
    PY_SCRIPT="transtab_transfer_cl.py"
    MODEL_NAME="transtab_transfer_cl"
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

RESULTS_DIR="$RESULTS_ROOT/transfer/transtab/$MODE/$DATA_TAG"
LOG_DIR="$RESULTS_ROOT/logs/transfer/transtab/$MODE/$DATA_TAG"
ARTIFACT_DIR="$RESULTS_ROOT/artifacts/transfer/transtab/$MODE/$DATA_TAG"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$ARTIFACT_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_transtab_${MODE}_${TS}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== TransTab ${MODE}: ${NUM_RUNS} Runs ==="
echo "Script    : $PY_SCRIPT"
echo "Task      : $DATASET[$TABLE_IDX]"
echo "Aux       : ${AUX_DATASET:-n/a}${AUX_TABLE_IDX:+[$AUX_TABLE_IDX]}"
echo "Data tag  : $DATA_TAG"
echo "GPU       : physical cuda:$GPU_ID exposed as cuda:0"
echo "Max jobs  : $MAX_JOBS"
echo "Epochs    : pretrain=$PRETRAIN_EPOCHS finetune=$FINETUNE_EPOCHS"
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
    local run_id=$1 run_seed=$2 run_dir=$3 ckpt_dir=$4 pretrain_dir=$5 run_json=$6 run_log=$7
    echo "--- [Run $run_id/$NUM_RUNS] seed=$run_seed ---"
    (
        cd "$BASELINE_DIR"
        if [[ "$MODE" == "single" ]]; then
            "$PYTHON" "$PY_SCRIPT" \
                --dataset "$DATASET" \
                --table_idx "$TABLE_IDX" \
                --work_dir "$run_dir/data" \
                --ckpt_dir "$ckpt_dir" \
                --num_epoch "$FINETUNE_EPOCHS" \
                --device cuda:0 \
                --num_runs 1 \
                --seed "$run_seed" \
                --save_results "$run_json"
        else
            "$PYTHON" "$PY_SCRIPT" \
                --dataset "$DATASET" \
                --table_idx "$TABLE_IDX" \
                --aux_dataset "$AUX_DATASET" \
                --aux_table_idx "$AUX_TABLE_IDX" \
                --work_dir "$run_dir/data" \
                --ckpt_dir "$ckpt_dir" \
                --pretrain_dir "$pretrain_dir" \
                --num_epoch_pretrain "$PRETRAIN_EPOCHS" \
                --num_epoch_finetune "$FINETUNE_EPOCHS" \
                --device cuda:0 \
                --seed "$run_seed" \
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
    RUN_DIR="$ARTIFACT_DIR/run_${i}_seed${RUN_SEED}"
    CKPT_DIR="$RUN_DIR/checkpoint"
    PRETRAIN_DIR="$RUN_DIR/pretrained"
    RUN_JSON="$RESULTS_DIR/run_${i}_seed${RUN_SEED}_${TS}.json"
    RUN_LOG="$LOG_DIR/run_${i}_seed${RUN_SEED}_${TS}.log"
    RUN_JSONS+=("$RUN_JSON")

    wait_for_slot
    run_one "$i" "$RUN_SEED" "$RUN_DIR" "$CKPT_DIR" "$PRETRAIN_DIR" "$RUN_JSON" "$RUN_LOG" &
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done
if [[ "$FAILED" -gt 0 ]]; then
    echo "ERROR: $FAILED TransTab run(s) failed. See logs in $LOG_DIR" >&2
    exit 1
fi

SUMMARY_JSON="$RESULTS_DIR/summary_${NUM_RUNS}runs_${TS}.json"
"$PYTHON" - "$SUMMARY_JSON" "$MAIN_LOG" "$MODEL_NAME" "$DATA_TAG" "$MODE" "${RUN_JSONS[@]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

summary_path = Path(sys.argv[1])
main_log = sys.argv[2]
model_name = sys.argv[3]
dataset_tag = sys.argv[4]
mode = sys.argv[5]
run_paths = [Path(p) for p in sys.argv[6:]]
runs = []
for idx, path in enumerate(run_paths, start=1):
    with path.open() as f:
        data = json.load(f)
    metrics = data.get("metrics")
    if metrics is None:
        run = data["individual_runs"][0]
        metrics = {
            "accuracy": run.get("test_acc"),
            "f1": run.get("test_f1"),
        }
    runs.append({
        "run_id": idx,
        "path": str(path),
        "seed": data.get("seed", data.get("individual_runs", [{}])[0].get("seed")),
        "runtime": data.get("runtime", data.get("statistics", {}).get("total_runtime")),
        **metrics,
        "ckpt_dir": data.get("ckpt_dir"),
        "pretrain_dir": data.get("pretrain_dir"),
    })

metric_names = sorted({
    key
    for run in runs
    for key, value in run.items()
    if key not in {"run_id", "path", "seed", "ckpt_dir", "pretrain_dir"} and value is not None
})
stats = {}
for name in metric_names:
    values = np.array([r[name] for r in runs if r.get(name) is not None], dtype=float)
    if len(values) == 0:
        continue
    stats[f"{name}_mean"] = float(values.mean())
    stats[f"{name}_std"] = float(values.std())
    stats[f"{name}_min"] = float(values.min())
    stats[f"{name}_max"] = float(values.max())

output = {
    "model": model_name,
    "task": "classification",
    "mode": mode,
    "dataset": dataset_tag,
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
if "accuracy_mean" in stats:
    print(f"Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
if "f1_mean" in stats:
    print(f"F1:       {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
PY

END=$(date +%s)
ELAPSED=$((END - START))
printf "\n=== All Done === (Elapsed: %dh %dm %ds)\n" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "Summary:   $SUMMARY_JSON"
echo "Results:   $RESULTS_DIR"
echo "Logs:      $LOG_DIR"
echo "Artifacts: $ARTIFACT_DIR"
