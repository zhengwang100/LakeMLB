# Third-Party Libraries and Local Resources

This directory contains third-party source code, model files, and runtime data
required by the LakeMLB experiments.

The GitHub repository is intended to retain only the following two third-party
libraries adapted for LakeMLB:

- `transtab/`: modified from
  [RyanWangZf/transtab](https://github.com/RyanWangZf/transtab).
- `carte_ai/`: modified from
  [soda-inria/carte](https://github.com/soda-inria/carte).

These changes provide consistent classification data interfaces, fixed data
splits, multi-table inputs, and experimental settings. They are not official
releases of the corresponding upstream projects. Users must follow the
upstream licenses and cite the original papers when using these libraries.

The rLLM source tree, model weights, Hugging Face caches, FastText files, and
CARTE raw/preprocessed data are not committed because of their size. They must
be prepared locally before reproducing the experiments.


## Expected Directory Structure

After local resources have been prepared, the relevant portion of `codes/lib/`
should have the following structure:

```text
codes/lib/
├── README.md
├── transtab/
├── carte_ai/
│   └── data/
│       ├── __init__.py
│       ├── load_data.py
│       ├── data_raw/
│       │   ├── <data_name>.csv
│       │   └── mask_<name>.pt
│       ├── data_singletable/
│       │   └── <data_name>/
│       │       ├── raw.parquet
│       │       └── config_data.json
│       └── etc/
│           └── kg_pretrained.pt
├── rllm/
│   └── datasets/lakemlb/
└── FastText/
    └── cc.en.300.bin
```


## 1. rLLM

The LakeMLB tree-based, deep tabular, TransTab, TabPFN, and TabICL experiments
use rLLM's `TableData` abstraction and LakeMLB dataset classes.

Upstream source:

- [rllm-team/rllm](https://github.com/rllm-team/rllm)

The base source tree can be prepared with:

```bash
git clone https://github.com/rllm-team/rllm.git codes/lib/rllm
```

Cloning the upstream repository alone is not sufficient. The current
experiment code also requires:

```text
codes/lib/rllm/datasets/lakemlb/
codes/lib/rllm/datasets/__init__.py
```

These files must provide the six LakeMLB dataset classes, processed-table
ordering, target columns, download locations, and fixed-mask processing.
`rllm.datasets.lakemlb` is a LakeMLB-specific integration and is not part of a
standard upstream rLLM installation.

Before public release, use one of the following approaches:

1. Publish `datasets/lakemlb/` and the required registration changes as a small
   patch with LakeMLB.
2. Provide a fixed rLLM branch or commit containing these changes and record
   its commit SHA here.
3. Move the LakeMLB dataset classes out of the third-party rLLM tree and
   publish them as a first-party LakeMLB module.

Until one of these options is implemented, the publicly retained files alone
cannot reproduce the experiments other than CARTE.

## 2. TransTab

LakeMLB uses the modified source in `codes/lib/transtab/`. Do not install a
second TransTab version with `pip` that could override this copy. The
experiment code places `codes/lib` on `PYTHONPATH` so that the adapted source
is imported first.

Upstream source and documentation:

- [RyanWangZf/transtab](https://github.com/RyanWangZf/transtab)
- [TransTab documentation](https://transtab.readthedocs.io/)
- [bert-base-uncased tokenizer/model](https://huggingface.co/google-bert/bert-base-uncased)

TransTab training does not require a separate pretrained TransTab checkpoint.
The current implementation uses a BERT tokenizer for column names and cell
text. The tokenizer files included with the source support offline execution.
If no local tokenizer is available, Transformers downloads the
`bert-base-uncased` tokenizer from Hugging Face.

## 3. CARTE

LakeMLB uses the modified source in `codes/lib/carte_ai/`. Reproducing CARTE
requires all of the following:

1. The LakeMLB-adapted CARTE source.
2. The English FastText vectors in `cc.en.300.bin`.
3. The CARTE pretrained checkpoint `kg_pretrained.pt`.
4. The CSV file, fixed split mask, and preprocessed output for each experiment
   table.

### 3.1 FastText

The LakeMLB CARTE runners read:

```text
codes/lib/FastText/cc.en.300.bin
```

The official CARTE examples use the file with the same name from
[hi-paris/fastText](https://huggingface.co/hi-paris/fastText/tree/main).
Download it to the required location with `huggingface_hub`:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='hi-paris/fastText', filename='cc.en.300.bin', local_dir='codes/lib/FastText')"
```

Verified local file information:

```text
Size: approximately 6.8 GB
SHA256: 14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca
```

CARTE's bundled `scripts/download_data.py` downloads FastText to
`codes/lib/carte_ai/data/etc/`, while LakeMLB's `carte_single.py` and
`carte_joint.py` read it from `codes/lib/FastText/`. When using CARTE's
download script, copy the file or create a symbolic link at the location
expected by LakeMLB.

### 3.2 CARTE Pretrained Checkpoint

The adapted CARTE code uses `load_pretrain=True` by default and reads:

```text
codes/lib/carte_ai/data/etc/kg_pretrained.pt
```

Verified local file information:

```text
Size: 40,212,939 bytes (approximately 39 MB)
SHA256: 21a08bf7d4dd29ff895ad2bf91abe24a4ddbcd0770c73f9b6ced1e016ab0c597
```

This file is not generated by `preprocess_raw.py`, and CARTE's
`download_data.py` does not provide a separate download step for it. Because
`codes/lib/carte_ai/data/` is not committed, the checkpoint must be published
separately as a GitHub Release or Hugging Face resource, with a stable download
link added here. The default CARTE experiments cannot run without this file.

### 3.3 CARTE Raw Data and Masks

CARTE does not read the `.pt` tables generated by rLLM. It uses its own
preprocessed data directory. Dataset definitions are maintained in:

```text
codes/lib/carte_ai/scripts/preprocess_raw.py
```

For a dataset named `<data_name>`, the preprocessing script expects:

```text
codes/lib/carte_ai/data/data_raw/<data_name>.csv
```

Place the fixed split mask in the same directory. Its filename is determined
by the `--mask_basename` runner option:

```text
codes/lib/carte_ai/data/data_raw/mask_<mask_basename>.pt
```

Each mask file must be a dictionary containing three Boolean tensors whose
lengths match the corresponding task table:

```text
train_mask
val_mask
test_mask
```

CSV files can be copied from the LakeMLB working-data directories under
`codes/data/table_<dataset>/raw/`. The copied filename must exactly match the
dataset definition in `preprocess_raw.py`.

### 3.4 Generate CARTE Preprocessed Data

Run the following command from the repository root:

```bash
PYTHONPATH="$PWD/codes/lib:$PYTHONPATH" \
python codes/lib/carte_ai/scripts/preprocess_raw.py \
  -dt <data_name_1> <data_name_2>
```

Successful preprocessing creates:

```text
codes/lib/carte_ai/data/data_singletable/<data_name>/raw.parquet
codes/lib/carte_ai/data/data_singletable/<data_name>/config_data.json
```

Use the following command for a one-run single-table check:

```bash
bash codes/scripts/run_carte.sh \
  --mode single \
  --data_name <data_name> \
  --mask_basename <mask_basename> \
  --gpu 0 \
  --num_runs 1
```

CARTE upstream resources:

- [soda-inria/carte](https://github.com/soda-inria/carte)
- [CARTE benchmark data](https://huggingface.co/datasets/inria-soda/carte-benchmark)
- [hi-paris/fastText](https://huggingface.co/hi-paris/fastText/tree/main)

The CARTE benchmark resource contains data for the upstream CARTE experiments,
not the six LakeMLB datasets. LakeMLB CSV files and fixed masks must still be
prepared through this project's data download or construction workflow.

## 4. Optional Model Caches

The following resources do not need to be placed under `codes/lib` to run the
standard baselines. Their official packages use user-level caches or accept an
explicit model path:

- [TabPFN official repository](https://github.com/PriorLabs/TabPFN)
- [TabPFN-3 classifier checkpoint](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/tabpfn-v3-classifier-v3_default.ckpt)
- [TabICLv2 official repository](https://github.com/soda-inria/tabicl)
- [TabICL checkpoints](https://huggingface.co/jingang/TabICL): the current
  classification runner uses `tabicl-classifier-v2-20260212.ckpt`.
- [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased):
  used only when a tokenizer or data-construction utility requires an online
  download.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):
  used only by rLLM's optional retrieval module and not required by the current
  LakeMLB baselines.

TabPFN and TabICL checkpoints are managed by their official Python packages.
They should not be copied into this directory or committed to the LakeMLB
repository.
