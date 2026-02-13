#!/bin/bash
# Quick examples showing how to run each baseline method.
# All Python scripts are located in baseline/.

cd "$(dirname "$0")/../baseline"

# ===================== Tree Models (CPU) =====================
# Available models: xgboost, catboost, lightgbm

python tree_models.py --model xgboost --seed 42 --num_runs 1
python tree_models.py --model catboost --seed 42 --num_runs 1
python tree_models.py --model lightgbm --seed 42 --num_runs 1

# ===================== Tabular Neural Networks (GPU) =====================
# Available models: fttransformer, tabtransformer, excelformer, saint, tromptnet

CUDA_VISIBLE_DEVICES=0 python tnns_test.py --model fttransformer --epochs 50 --lr 1e-3 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test.py --model tabtransformer --epochs 50 --lr 1e-3 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test.py --model excelformer   --epochs 50 --lr 1e-3 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test.py --model saint         --epochs 50 --lr 1e-3 --device cuda:0 --seed 42
CUDA_VISIBLE_DEVICES=0 python tnns_test.py --model tromptnet     --epochs 50 --lr 1e-3 --device cuda:0 --seed 42

# ===================== TransTab (GPU) =====================
# Single-table training (no transfer learning)
CUDA_VISIBLE_DEVICES=0 python transtab_single.py --num_epoch 50 --device cuda:0

# Transfer learning: pretrain on auxiliary table, fine-tune on task table
CUDA_VISIBLE_DEVICES=0 python transtab_transfer.py --num_epoch_pretrain 50 --num_epoch_finetune 50 --device cuda:0

# Contrastive learning: unsupervised pretrain, then fine-tune
CUDA_VISIBLE_DEVICES=0 python transtab_transfer_cl.py

# ===================== CARTE (GPU) =====================
# Single-table classification
# CUDA_VISIBLE_DEVICES=0 python carte_single.py

# Multi-table classification (with auxiliary source tables)
# CUDA_VISIBLE_DEVICES=0 python carte_joint.py

# ===================== Foundation Models (GPU) =====================
# TabPFN v2
CUDA_VISIBLE_DEVICES=0 python tabpfnv2.py --device cuda:0

# TabICL
# CUDA_VISIBLE_DEVICES=0 python tabicl_clf.py --device cuda:0
