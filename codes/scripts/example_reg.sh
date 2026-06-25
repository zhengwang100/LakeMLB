#!/bin/bash
# Examples for regression baselines (GACars price regression).
# All Python scripts are in baseline/.
# table_idx: 4=German Reg  5=Australian Reg  6=DA Reg  7=FA Reg

cd "$(dirname "$0")/../baseline"

# ===================== Tree Models – Regression =====================

# --- Single run (quick smoke test) ---
python tree_models_reg.py --model xgboost   --table_idx 4 --seed 42 --num_runs 1
python tree_models_reg.py --model catboost  --table_idx 4 --seed 42 --num_runs 1
python tree_models_reg.py --model lightgbm  --table_idx 4 --seed 42 --num_runs 1

# --- Full 10-run evaluation (German Reg) ---
python tree_models_reg.py --model xgboost   --table_idx 4 --seed 42 --num_runs 10
python tree_models_reg.py --model catboost  --table_idx 4 --seed 42 --num_runs 10
python tree_models_reg.py --model lightgbm  --table_idx 4 --seed 42 --num_runs 10

# --- Grid search then 10-run evaluation ---
python tree_models_reg.py --model xgboost  --table_idx 4 --seed 42 --num_runs 10 --grid
python tree_models_reg.py --model catboost --table_idx 4 --seed 42 --num_runs 10 --grid
python tree_models_reg.py --model lightgbm --table_idx 4 --seed 42 --num_runs 10 --grid


# ===================== TabPFN v2 – Regression (GPU) =====================
# In-context learner; fits on train+val, evaluates on test.

CUDA_VISIBLE_DEVICES=0 python tabpfnv2_reg.py --table_idx 4 --num_runs 5 --device cuda:0 --seed 42


# ===================== Tabular Neural Networks – Regression (GPU) =====================

# --- Single run (quick test) ---
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model fttransformer  --table_idx 4 --epochs 3  --num_runs 1 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model tabtransformer --table_idx 4 --epochs 3  --num_runs 1 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model excelformer    --table_idx 4 --epochs 3  --num_runs 1 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model saint          --table_idx 4 --epochs 3  --num_runs 1 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model tromptnet      --table_idx 4 --epochs 3  --num_runs 1 --device cuda:0 --seed 42

# --- Full 5-run evaluation (German Reg, 200 epochs, early stopping patience=50) ---
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model fttransformer  --table_idx 4 --epochs 200 --patience 50 --num_runs 5 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model tabtransformer --table_idx 4 --epochs 200 --patience 50 --num_runs 5 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model excelformer    --table_idx 4 --epochs 200 --patience 50 --num_runs 5 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model saint          --table_idx 4 --epochs 200 --patience 50 --num_runs 5 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test_reg.py --model tromptnet      --table_idx 4 --epochs 200 --patience 50 --num_runs 5 --device cuda:0 --seed 42


# ===================== TransTab – Regression (GPU) =====================

# --- Single-table (German Reg only) ---
# Quick smoke test (1 run, 3 epochs)
CUDA_VISIBLE_DEVICES=0 python transtab_single_reg.py \
    --num_epoch 3 --num_runs 1 --device cuda:0 --seed 42

# Full 5-run evaluation (50 epochs, early-stop patience=10)
CUDA_VISIBLE_DEVICES=0 python transtab_single_reg.py \
    --num_epoch 50 --patience 10 --num_runs 5 --device cuda:0 --seed 42

# --- Transfer learning (Pretrain on Australian Reg → Fine-tune on German Reg) ---
# Quick smoke test (1 run, 3 pretrain + 3 finetune epochs)
CUDA_VISIBLE_DEVICES=0 python transtab_transfer_reg.py \
    --num_epoch_pretrain 3 --num_epoch_finetune 3 --num_runs 1 --device cuda:0 --seed 42

# Full 5-run evaluation
CUDA_VISIBLE_DEVICES=0 python transtab_transfer_reg.py \
    --num_epoch_pretrain 50 --num_epoch_finetune 50 \
    --patience 10 --num_runs 5 --device cuda:0 --seed 42


# ===================== CARTE – Regression (GPU) =====================
# Note: graph building (~20 min for joint) is done once and reused across runs.

# --- Single-table (German Reg only) ---
# Smoke test (1 run, ensemble=1)
CUDA_VISIBLE_DEVICES=1 python carte_single_reg.py \
    --num_runs 1 --num_model 1 --device cuda:1 --seed 0

# Full 5-run evaluation (ensemble=5)
CUDA_VISIBLE_DEVICES=1 python carte_single_reg.py \
    --num_runs 5 --num_model 5 --device cuda:1 --seed 0

# --- Joint (German Reg + Australian Reg, CARTE native multi-table) ---
# Smoke test
CUDA_VISIBLE_DEVICES=1 python carte_joint_reg.py \
    --num_runs 1 --num_model 1 --device cuda:1 --seed 0

# Full 5-run evaluation
CUDA_VISIBLE_DEVICES=1 python carte_joint_reg.py \
    --num_runs 5 --num_model 5 --device cuda:1 --seed 0

# --- DA / FA augmented single-table variants ---
CUDA_VISIBLE_DEVICES=1 python carte_single_reg.py \
    --data_name gacars_da_reg --num_runs 5 --num_model 5 --device cuda:1 --seed 0
CUDA_VISIBLE_DEVICES=1 python carte_single_reg.py \
    --data_name gacars_fa_reg --num_runs 5 --num_model 5 --device cuda:1 --seed 0
