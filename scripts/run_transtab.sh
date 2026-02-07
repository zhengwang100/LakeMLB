#!/bin/bash
# Run TransTab scripts repeatedly with concurrency control.

set -euo pipefail

################ Config ################

PYTHON="/home/pfy/.conda/envs/lake/bin/python"
GPU_ID=1
export CUDA_VISIBLE_DEVICES=$GPU_ID

REPEAT=5
MAX_JOBS=5
BASELINE_DIR="$(cd "$(dirname "$0")/../baseline" && pwd)"
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
LOG_DIR="$RESULTS_DIR/transtab"

SCRIPTS=(
    "transtab_single.py"
)

########################################

# Concurrency limiter: wait until running jobs < MAX_JOBS
run_with_limit() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 5
    done
    "$@" &
}

cd "$BASELINE_DIR" || { echo "ERROR: $BASELINE_DIR not found"; exit 1; }
mkdir -p "$LOG_DIR"

echo "=== TransTab Repeated Runs ==="
echo "Python: $PYTHON | GPU: $GPU_ID | Repeat: $REPEAT | MaxJobs: $MAX_JOBS"
echo "Scripts: ${SCRIPTS[*]}"
echo ""

for script in "${SCRIPTS[@]}"; do
    script_path="$BASELINE_DIR/$script"
    if [ ! -f "$script_path" ]; then
        echo "WARNING: $script_path not found, skipping"
        continue
    fi

    base=$(basename "$script" .py)

    for ((i=1; i<=REPEAT; i++)); do
        log_file="${LOG_DIR}/${base}_run${i}.log"
        ckpt_dir="${BASELINE_DIR}/checkpoint_${base}_run${i}"
        pretrain_dir="${BASELINE_DIR}/ckpt_${base}_run${i}_pretrained"

        cmd=("$PYTHON" "$script_path")

        case "$script" in
            transtab_single.py)
                cmd+=(--ckpt_dir "$ckpt_dir" --device cuda:0)
                ;;
            transtab_clf_simplified.py)
                cmd+=(--ckpt_dir "$ckpt_dir" --pretrain_dir "$pretrain_dir" --device cuda:0)
                ;;
        esac

        echo "Launch: $script (run $i/$REPEAT) -> $log_file"
        run_with_limit "${cmd[@]}" > "$log_file" 2>&1
    done
done

echo "All tasks submitted, waiting..."
wait
echo "All done."
