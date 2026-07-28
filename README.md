# LakeMLB Experimental Artifact

This repository contains the anonymized datasets, baseline implementations,
modified third-party code, and experiment entry points used in the study. It is
organized as a self-contained experimental artifact for double-blind review.

## Artifact Contents

```text
LakeMLB/
├── benckmark/
│   ├── join_based/
│   │   ├── agbooks.zip
│   │   ├── dsmusic.zip
│   │   └── nnstocks.zip
│   └── union_based/
│       ├── mstraffic.zip
│       ├── ncbuilding.zip
│       └── nctaxi.zip
├── codes/
│   ├── baseline/
│   ├── lib/
│   ├── process_data/
│   ├── scripts/
│   ├── data/                 # Generated locally; omitted from submission
│   └── results/              # Generated locally; omitted from submission
├── requirements.txt
└── README.md
```

### `benckmark/`

This directory contains four full dataset archives and two reduced example
archives.

- `union_based/`: MSTraffic, NCBuilding, and NCTaxi.
- `join_based/`: NNStocks, DSMusic, and AGBooks.

Every archive is flat and contains exactly seven files:

- two original source tables;
- one data-augmented table with the `_da` suffix;
- one feature-augmented table with the `_fa` suffix;
- two train/validation/test mask files;
- one `mapping.csv` describing the row correspondence used by feature
  augmentation.

Because the artifact is limited to 50 MB, `agbooks.zip` and `nctaxi.zip`
contain deterministic 10% examples generated with seed 42. Each example
contains 10,000 task rows, 10,000 auxiliary rows, 12,100 DA rows, 10,000 FA
rows, and 10,000 mappings. Task rows are sampled jointly by class and data
split, preserving exact class balance and the 70/10/20 train/validation/test
ratio. The additional 2,100 DA rows are training-only.

For NCTaxi, the labeled auxiliary table is also class-balanced, and all
auxiliary rows referenced by the sampled FA mappings are retained. The
unlabeled AGBooks auxiliary table is sampled uniformly. Additional DA rows are
sampled proportionally by class. FA rows and mappings follow the sampled task
indices exactly. To avoid duplicating the long text already stored in the
tables, example mappings retain the task index, auxiliary index, and similarity
score. NCTaxi auxiliary indices are reindexed to the sampled auxiliary table;
AGBooks auxiliary indices retain their original Goodreads provenance.

These two examples validate data construction and experiment execution. The
reported AGBooks and NCTaxi results were obtained using the full-scale tables
with 100,000 rows in each original table.

No network download is performed by the dataset classes. When the extracted
files are absent, the corresponding local archive is read from `benckmark/` and
unpacked into `codes/data/table_<dataset>/raw/`.

### `codes/baseline/`

This directory contains the experiment implementations:

- `tree_models.py`: XGBoost, CatBoost, and LightGBM;
- `tnns_models.py` and `tnns_test.py`: FTTransformer, TabTransformer,
  ExcelFormer, SAINT, and TromptNet;
- `transtab_single.py`: TransTab single-table learning;
- `transtab_transfer.py`: TransTab supervised transfer;
- `transtab_transfer_cl.py`: TransTab contrastive transfer;
- `carte_single.py` and `carte_joint.py`: CARTE experiments;
- `tabpfnv2_extend.py`: TabPFN v3 experiments;
- `tabicl_v2.py`: TabICLv2 experiments;
- `merge_foundation_results.py`: aggregation of foundation-model runs;
- `utils.py` and `transtab_lakemlb_utils.py`: shared loading, evaluation, and
  experiment utilities.

### `codes/lib/`

This directory retains the locally modified portions of third-party libraries:

- `rllm/`: dataset definitions and tabular neural-network components;
- `transtab/`: modified TransTab data loading and training components;
- `carte_ai/`: modified CARTE source and retained preprocessing definitions.

Only files required by the experiments or modified for fair comparison are
included. Large public model assets, model caches, and CARTE-preprocessed data
are excluded because of the artifact size limit. The omitted asset inventory
and expected paths are documented in `codes/lib/README.md`.

### `codes/process_data/`

This directory contains the retained data-construction and maintenance
utilities, including the generic data-augmentation and feature-augmentation
pipelines. These scripts document how the derived DA and FA tables were
created. Standard experiment reproduction uses the prepared tables in the
local dataset archives and does not require rerunning these utilities.

### `codes/scripts/`

This directory contains the six experiment entry points:

```text
run_tree_models.sh
run_nn_grid_search.sh
run_transtab.sh
run_carte.sh
run_tabpfn.sh
run_tabicl.sh
```

Concise usage, experiment modes, and output conventions are provided in
`codes/scripts/README.md`. Dataset indices and the reproduction protocol are
documented below.

### `codes/data/`

This directory is intentionally omitted from the submitted artifact. It is a
runtime cache rather than an independent source of data.

For the RLLM-based pipelines, the first dataset construction automatically:

1. locates the appropriate ZIP archive under `benckmark/`;
2. extracts its seven files into `codes/data/table_<dataset>/raw/`;
3. creates four processed `TableData` files under
   `codes/data/table_<dataset>/processed/`.

The archive locations and extraction targets must retain the directory layout
shown above.

### `codes/results/`

This directory is intentionally omitted from the submitted artifact. Experiment
scripts create it automatically and store:

- individual-run metrics;
- aggregate statistics;
- hyperparameter-search records;
- logs;
- trained checkpoints and other model artifacts.

### `requirements.txt`

This file contains the pinned direct Python dependencies used by the
experiments. PyTorch and PyTorch Geometric builds must be selected to match the
CUDA runtime on the evaluation machine.

## Dataset Tables

Each dataset exposes four tables with a consistent index convention:

| Dataset | Index 0 | Index 1 | Index 2 | Index 3 |
| --- | --- | --- | --- | --- |
| MSTraffic | Maryland | Seattle | MSTraffic-DA | MSTraffic-FA |
| NCBuilding | New York | Chicago | NCBuilding-DA | NCBuilding-FA |
| NCTaxi | New York Taxi | Chicago Taxi | NCTaxi-DA | NCTaxi-FA |
| NNStocks | Stock List | Wikipedia | NNStocks-DA | NNStocks-FA |
| DSMusic | Discogs | Spotify | DSMusic-DA | DSMusic-FA |
| AGBooks | Amazon | Goodreads | AGBooks-DA | AGBooks-FA |

Index 0 is the primary task table. Index 1 is the auxiliary table. Index 2 is
the data-augmented table, and index 3 is the feature-augmented table.

The auxiliary tables in MSTraffic, NCBuilding, and NCTaxi contain labels and can
be used for supervised transfer. The auxiliary tables in NNStocks, DSMusic, and
AGBooks are unlabeled and are intended for contrastive or unsupervised transfer.

## Environment Preparation

The experiments were tested with Python 3.9.21. Create and activate a compatible
environment, then install the pinned dependencies:

```bash
conda create --name lake python=3.9.21 pip
conda activate lake
pip install -r requirements.txt
```

CUDA-enabled PyTorch and PyTorch Geometric packages must match the CUDA runtime
available on the evaluation machine.

## Data Preparation Check

Run commands from the repository root. The following check reconstructs the
small NNStocks dataset from its local archive and verifies that four tables are
available:

```bash
PYTHONPATH=codes/lib python - <<'PY'
from rllm.datasets import NNStocksDataset

dataset = NNStocksDataset(cached_dir="codes/data")
print("Number of tables:", len(dataset))
for index, table in enumerate(dataset.data_list):
    print(index, table.df.shape, table.target_col)
PY
```

The same extraction and processing mechanism is used by the other five dataset
classes.

## Running Experiments

The commands below launch the standard experiment workflows. Use
`bash codes/scripts/<script>.sh --help` to inspect all available options.

### Tree-Based Models

This command performs hyperparameter search and repeated evaluation for
XGBoost, CatBoost, and LightGBM:

```bash
bash codes/scripts/run_tree_models.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --device 0 \
  --num_runs 10
```

The complete search space contains 108 configurations per model.

### Tabular Neural Networks

This command evaluates FTTransformer, TabTransformer, ExcelFormer, SAINT, and
TromptNet:

```bash
bash codes/scripts/run_nn_grid_search.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --gpu 0 \
  --num_runs 10 \
  --num_tasks 2
```

The complete search space contains 81 configurations per model.
`--num_tasks` controls the number of parallel grid-search shards.

### TransTab

Single-table learning:

```bash
bash codes/scripts/run_transtab.sh \
  --mode single \
  --dataset mstraffic \
  --table_idx 0 \
  --gpu 0 \
  --num_runs 10
```

Supervised transfer with a labeled auxiliary table:

```bash
bash codes/scripts/run_transtab.sh \
  --mode transfer \
  --dataset mstraffic \
  --table_idx 0 \
  --aux_dataset mstraffic \
  --aux_table_idx 1 \
  --gpu 0 \
  --num_runs 10 \
  --max_jobs 1
```

Contrastive transfer with an unlabeled auxiliary table:

```bash
bash codes/scripts/run_transtab.sh \
  --mode contrastive \
  --dataset nnstocks \
  --table_idx 0 \
  --aux_dataset nnstocks \
  --aux_table_idx 1 \
  --gpu 0 \
  --num_runs 10 \
  --max_jobs 1
```

### Foundation Models

TabPFN v3:

```bash
bash codes/scripts/run_tabpfn.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --gpu 0 \
  --num_runs 10 \
  --max_jobs 1
```

TabICLv2:

```bash
bash codes/scripts/run_tabicl.sh \
  --dataset mstraffic \
  --table_idx 0 \
  --gpu 0 \
  --num_runs 10 \
  --max_jobs 1
```

Local checkpoints can be supplied through the model-path options exposed by the
scripts. Public pretrained checkpoints are not included in this artifact.

### CARTE

CARTE requires its own preprocessed single-table representation and a FastText
model. These large assets are not included in the submitted artifact.

The dataset-specific preprocessing definitions are retained in
`codes/lib/carte_ai/scripts/preprocess_raw.py`. After preparing the required
CARTE data and model assets, run:

```bash
# Single-table learning
bash codes/scripts/run_carte.sh \
  --mode single \
  --data_name maryland \
  --mask_basename maryland \
  --gpu 0 \
  --num_runs 10 \
  --max_jobs 1

# Multi-table transfer
bash codes/scripts/run_carte.sh \
  --mode joint \
  --target_data_name maryland \
  --source_data_name seattle \
  --mask_basename maryland \
  --gpu 0 \
  --num_runs 10 \
  --max_jobs 1
```

## Reproduction Notes

- Run all commands from the repository root.
- `--dataset` selects one of the six dataset families.
- `--table_idx` follows the four-table convention documented above.
- `--gpu N` sets the physical GPU exposed to the Python process.
- The tree-model script uses `--device N` instead of `--gpu N`.
- `--max_jobs` controls concurrent repeated runs on one GPU. Use `1` when GPU
  memory is limited.
- Repeated runs use independently generated seeds. Every realized seed is
  recorded in the output JSON and log.
- Tree and TNN workflows perform full hyperparameter search before repeated
  evaluation; they are substantially more expensive than a smoke test.
- The fixed train/validation/test masks included in each dataset archive must
  not be regenerated when reproducing the reported protocol.
- `mapping.csv` documents the FA pairing. The provided `_fa.csv` files already
  contain the corresponding augmented features.
- `codes/data` and `codes/results` can be deleted safely before distribution;
  they are reconstructed during execution.
- Additional command examples and output conventions are documented in
  `codes/scripts/README.md`.
