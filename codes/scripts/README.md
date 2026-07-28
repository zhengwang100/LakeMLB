# Experiment Scripts

This directory contains the six shell entry points used for the classification
experiments. Baseline implementations are located in `codes/baseline/`. Run all
commands from the repository root.

Unless specified otherwise, the scripts use table 0 of `mstraffic` and perform
10 repeated runs. Use `bash codes/scripts/<script>.sh --help` for the complete
argument list.

## Tree-Based Models

Run hyperparameter search and repeated evaluation for XGBoost, CatBoost, and
LightGBM:

```bash
bash codes/scripts/run_tree_models.sh \
  --dataset mstraffic --table_idx 0 --device 0 --num_runs 10
```

Use `--models` to select a subset of models and `--num_threads` to control the
number of CPU threads used by each learner.

## Tabular Neural Networks

Run FTTransformer, TabTransformer, ExcelFormer, SAINT, and TromptNet:

```bash
bash codes/scripts/run_nn_grid_search.sh \
  --dataset mstraffic --table_idx 0 --gpu 0 --num_runs 10 --num_tasks 2
```

Use `--models` to select a subset and `--num_tasks` to control parallel
hyperparameter-search tasks.

## TransTab

The unified TransTab script supports single-table learning, supervised transfer,
and contrastive transfer:

```bash
# Single-table learning
bash codes/scripts/run_transtab.sh \
  --mode single --dataset mstraffic --table_idx 0 --gpu 0 --num_runs 10

# Supervised transfer from a labeled auxiliary table
bash codes/scripts/run_transtab.sh \
  --mode transfer \
  --dataset mstraffic --table_idx 0 \
  --aux_dataset mstraffic --aux_table_idx 1 \
  --gpu 0 --num_runs 10 --max_jobs 1

# Contrastive transfer from an unlabeled auxiliary table
bash codes/scripts/run_transtab.sh \
  --mode contrastive \
  --dataset nnstocks --table_idx 0 \
  --aux_dataset nnstocks --aux_table_idx 1 \
  --gpu 0 --num_runs 10 --max_jobs 1
```

## CARTE

CARTE uses preprocessed `data_name` identifiers rather than
`--dataset/--table_idx`:

```bash
# Single-table learning
bash codes/scripts/run_carte.sh \
  --mode single \
  --data_name maryland --mask_basename maryland \
  --gpu 0 --num_runs 10 --max_jobs 1

# Multi-table transfer
bash codes/scripts/run_carte.sh \
  --mode joint \
  --target_data_name maryland --source_data_name seattle \
  --mask_basename maryland \
  --gpu 0 --num_runs 10 --max_jobs 1
```

**Data preprocessing.** The preprocessed CARTE data are excluded from this
artifact due to the submission size limit. To reproduce the experiments, process
the raw data using the preprocessing procedure specified by CARTE. The dataset
definitions used in this study are retained in
`codes/lib/carte_ai/scripts/preprocess_raw.py`.

## Foundation Models

Run TabPFN v3:

```bash
bash codes/scripts/run_tabpfn.sh \
  --dataset mstraffic --table_idx 0 --gpu 0 --num_runs 10 --max_jobs 1
```

Run TabICLv2:

```bash
bash codes/scripts/run_tabicl.sh \
  --dataset mstraffic --table_idx 0 --gpu 0 --num_runs 10 --max_jobs 1
```

Use `--model_path` to specify a local checkpoint. TabICL additionally supports
`--checkpoint` and `--no_auto_download`.

## Data and Outputs

- `--dataset` selects a dataset and `--table_idx` selects a table.
- `--gpu N` exposes physical GPU `N`; the tree-model script uses `--device N`.
- `--max_jobs` controls concurrent repeated runs on one GPU. Use `1` when GPU
  memory is limited.
- Results, logs, and model artifacts are written under `codes/results/`.

Dataset indices and the reproduction protocol are documented in the repository
README. Public data and pretrained weights excluded from this artifact are
documented in [`../lib/README.md`](../lib/README.md).
