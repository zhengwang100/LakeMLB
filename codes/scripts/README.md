# LakeMLB Experiment Scripts

LakeMLB provides six maintained entry points for classification experiments.
Run all commands from the repository root with the `lake` Conda environment:

```bash
cd /home/pfy/LakeMLB
conda activate lake
```

Use `bash codes/scripts/<script>.sh --help` to view all supported options for
an entry point.

## Dataset Selection

All methods except CARTE select a table with:

```text
--dataset <dataset_name> --table_idx <table_index>
```

`table_idx` follows the order of `processed_filenames` in the corresponding
dataset class under `codes/lib/rllm/datasets/lakemlb/`. See
`codes/data/README.md` for task tables, auxiliary tables, labels, and class
counts.

CARTE uses independently preprocessed data names rather than
`--dataset/--table_idx`. Its dataset definitions are maintained in
`codes/lib/carte_ai/scripts/preprocess_raw.py`.

## Maintained Entry Points

| Family | Script | Methods |
|---|---|---|
| Tree-based | `run_tree_models.sh` | XGBoost, CatBoost, LightGBM |
| Deep tabular | `run_nn_grid_search.sh` | FT-Transformer, TabTransformer, ExcelFormer, SAINT, TromptNet |
| Transfer learning | `run_transtab.sh` | TransTab single-table, labeled transfer, and contrastive transfer |
| Transfer learning | `run_carte.sh` | CARTE single-table and joint multi-table learning |
| Foundation model | `run_tabpfn.sh` | TabPFN v3 |
| Foundation model | `run_tabicl.sh` | TabICLv2 |

### Tree-Based Models

The three models run sequentially. Each selected model first performs grid
search and then repeats training with its best configuration. Use `--models`
to select a subset and `--num_threads` to control CPU threads inside each
learner.

```bash
bash codes/scripts/run_tree_models.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --device 1 \
  --models xgboost,catboost,lightgbm \
  --num_runs 10 \
  --num_threads 16
```

### Deep Tabular Models

Each selected model performs grid search before repeated training with the
best configuration. `--num_tasks` partitions the grid into parallel tasks.
Final repeated runs are sequential to reduce GPU out-of-memory risk.

```bash
bash codes/scripts/run_nn_grid_search.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --gpu 1 \
  --models fttransformer,tabtransformer,excelformer,saint,tromptnet \
  --num_tasks 2 \
  --num_runs 10
```

### TransTab

Single-table classification:

```bash
bash codes/scripts/run_transtab.sh \
  --mode single \
  --dataset mstraffic \
  --table_idx 0 \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
```

Transfer from a labeled auxiliary table:

```bash
bash codes/scripts/run_transtab.sh \
  --mode transfer \
  --dataset mstraffic \
  --table_idx 0 \
  --aux_dataset mstraffic \
  --aux_table_idx 1 \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
```

Contrastive pretraining with an unlabeled auxiliary table:

```bash
bash codes/scripts/run_transtab.sh \
  --mode contrastive \
  --dataset nnstocks \
  --table_idx 0 \
  --aux_dataset nnstocks \
  --aux_table_idx 1 \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
```

### CARTE

Single-table mode:

```bash
bash codes/scripts/run_carte.sh \
  --mode single \
  --data_name maryland \
  --mask_basename maryland \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
```

Joint multi-table mode:

```bash
bash codes/scripts/run_carte.sh \
  --mode joint \
  --target_data_name maryland \
  --source_data_name seattle \
  --mask_basename maryland \
  --gpu 1 \
  --num_runs 10 \
  --max_jobs 1
```

Pass multiple auxiliary tables as a comma-separated
`--source_data_name` value. CARTE requires
`codes/lib/FastText/cc.en.300.bin`.

### TabPFN v3

```bash
bash codes/scripts/run_tabpfn.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --gpu 1 \
  --n_estimators 8 \
  --num_runs 10 \
  --max_jobs 1
```

By default, TabPFN loads its v3 checkpoint from the package cache. Use
`--model_path` to select a local checkpoint explicitly.

### TabICLv2

```bash
bash codes/scripts/run_tabicl.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --gpu 1 \
  --n_estimators 8 \
  --batch_size 8 \
  --num_runs 10 \
  --max_jobs 1
```

The default checkpoint is `tabicl-classifier-v2-20260212.ckpt`. The runner can
download it automatically or use a local file supplied through `--model_path`.

## Output Directories

Every runner records the actual per-run seed, metrics, runtime, and terminal
output.

```text
codes/results/tree_models/            Tree-model results
codes/results/grid_search/tnns/       Deep-model grid-search results
codes/results/tnns/                   Final deep-model results
codes/results/transfer/               TransTab and CARTE results
codes/results/foundation/             TabPFN and TabICL results
codes/results/artifacts/              Selected weights and model artifacts
codes/results/logs/                   Experiment logs
codes/results/nohup/                  Optional outer nohup logs
```

## Detached Execution

Use `nohup` for long experiments that must continue after an SSH disconnect:

```bash
mkdir -p codes/results/nohup

nohup bash -lc '
conda activate lake
cd /home/pfy/LakeMLB
bash codes/scripts/run_nn_grid_search.sh \
  --dataset mstraffic --table_idx 0 --gpu 1 --num_tasks 2 --num_runs 10
' > codes/results/nohup/tnns_mstraffic_$(date +"%Y%m%d_%H%M%S").out 2>&1 &
```

Monitor the outer log with:

```bash
tail -f codes/results/nohup/<log_file>.out
```

Repeated-run seeds are generated from system entropy and stored in log files,
JSON results, and output filenames.
