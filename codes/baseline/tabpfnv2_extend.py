"""
TabPFN v3 classification benchmark for LakeMLB tables.

Supports generic LakeMLB datasets and multi-run evaluation.
"""
import sys
import argparse
import json
import os
import os.path as osp
import secrets
import time
from datetime import datetime

sys.path.append("./")
sys.path.append("../")
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', "lib"))
sys.path.insert(0, osp.dirname(__file__))

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
import importlib.metadata as importlib_metadata

from rllm.types import ColType
from rllm.datasets.lakemlb import (
    AGBooksDataset,
    DSMusicDataset,
    MSTrafficDataset,
    NCBuildingDataset,
    NCTaxiDataset,
    NNStocksDataset,
)

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = osp.dirname(osp.realpath(__file__))
DATA_DIR    = osp.abspath(osp.join(SCRIPT_DIR, '..', 'data'))
RESULTS_DIR = osp.abspath(osp.join(SCRIPT_DIR, '..', 'results', 'tabpfn'))
os.makedirs(RESULTS_DIR, exist_ok=True)

_NNSTOCKS_TABLE_NAMES = {
    0: "nnstocks_nnlist",
    1: "nnstocks_nnwiki",
    2: "nnstocks_da",
    3: "nnstocks_fa",
}

_DATASET_REGISTRY = {
    "mstraffic": MSTrafficDataset,
    "ncbuilding": NCBuildingDataset,
    "nctaxi": NCTaxiDataset,
    "nnstocks": NNStocksDataset,
    "dsmusic": DSMusicDataset,
    "agbooks": AGBooksDataset,
}

_TABLE_TAGS = {
    "mstraffic": {
        0: "mstraffic_maryland",
        1: "mstraffic_seattle",
        2: "mstraffic_da",
        3: "mstraffic_fa",
    },
    "ncbuilding": {
        0: "ncbuilding_newyork",
        1: "ncbuilding_chicago",
        2: "ncbuilding_da",
        3: "ncbuilding_fa",
    },
    "nctaxi": {
        0: "nctaxi_newyork_taxi",
        1: "nctaxi_chicago_taxi",
        2: "nctaxi_da",
        3: "nctaxi_fa",
    },
    "dsmusic": {
        0: "dsmusic_discogs",
        1: "dsmusic_spotify",
        2: "dsmusic_da",
        3: "dsmusic_fa",
    },
    "agbooks": {
        0: "agbooks_amazon",
        1: "agbooks_goodreads",
        2: "agbooks_da",
        3: "agbooks_fa",
    },
    "nnstocks": _NNSTOCKS_TABLE_NAMES,
}


def get_dataset_tag(dataset_name: str, table_idx: int) -> str:
    return _TABLE_TAGS.get(dataset_name, {}).get(
        table_idx, f"{dataset_name}_table{table_idx}"
    )

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mstraffic",
                    choices=sorted(_DATASET_REGISTRY.keys()))
parser.add_argument("--table_idx", type=int, default=0)
parser.add_argument("--f_dim",        type=int,   default=32)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--device",       type=str,   default="cuda:1")
parser.add_argument("--num_runs",     type=int,   default=5)
parser.add_argument("--n_estimators", type=int,   default=8)
parser.add_argument("--fit_mode",     type=str,   default="fit_preprocessors",
                    choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"])
parser.add_argument("--model_path",   type=str,   default=None,
                    help="Optional local TabPFN v3 checkpoint path. Defaults to TabPFN auto path.")
parser.add_argument("--save_results", type=str,   default=None)
args = parser.parse_args()

script_start = time.perf_counter()
data_tag = get_dataset_tag(args.dataset, args.table_idx)
if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(RESULTS_DIR, f"tabpfn_{data_tag}_{args.num_runs}runs_{ts}.json")

tabpfn_package_version = importlib_metadata.version("tabpfn")
model_version = ModelVersion.V3

print(f"TabPFN {model_version.value}  Dataset: {data_tag}  Runs: {args.num_runs}")
print(f"Package version: tabpfn=={tabpfn_package_version}")
print(f"Results → {args.save_results}")

dataset_class = _DATASET_REGISTRY[args.dataset]
dataset  = dataset_class(cached_dir=DATA_DIR, force_reload=False)
if args.table_idx < 0 or args.table_idx >= len(dataset.data_list):
    raise IndexError(
        f"table_idx={args.table_idx} is out of range for {args.dataset}; "
        f"available range is 0..{len(dataset.data_list) - 1}."
    )
data_raw = dataset[args.table_idx]
if not (
    hasattr(data_raw, "train_mask")
    and hasattr(data_raw, "val_mask")
    and hasattr(data_raw, "test_mask")
):
    raise ValueError(
        f"{args.dataset}[{args.table_idx}] must have train_mask, val_mask, "
        "and test_mask for TabPFN evaluation."
    )

features = [c for c in data_raw.col_types if c != data_raw.target_col]
data_df  = data_raw.df

train_mask = (data_raw.train_mask.cpu().numpy()
              if isinstance(data_raw.train_mask, torch.Tensor)
              else data_raw.train_mask)
val_mask   = (data_raw.val_mask.cpu().numpy()
              if isinstance(data_raw.val_mask, torch.Tensor)
              else data_raw.val_mask)
test_mask  = (data_raw.test_mask.cpu().numpy()
              if isinstance(data_raw.test_mask, torch.Tensor)
              else data_raw.test_mask)

X = data_df[features].copy()
y = data_df[data_raw.target_col].copy()
categorical_features_indices = [
    idx for idx, col in enumerate(features)
    if data_raw.col_types.get(col) == ColType.CATEGORICAL
]

# type coercion
for col in X.columns:
    if col in data_raw.col_types:
        if data_raw.col_types[col] == ColType.CATEGORICAL:
            X[col] = X[col].astype(str)
        elif data_raw.col_types[col] == ColType.NUMERICAL:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)

# train+val merged for TabPFN (single pass)
train_val_mask = train_mask | val_mask
X_train = X[train_val_mask].reset_index(drop=True)
y_train = y[train_val_mask].reset_index(drop=True)
X_test  = X[test_mask].reset_index(drop=True)
y_test  = y[test_mask].reset_index(drop=True)

n_classes = y_train.nunique()
print(f"Dataset: train+val={X_train.shape[0]}, test={X_test.shape[0]}, classes={n_classes}")
print(f"Categorical features: {len(categorical_features_indices)} / {len(features)}")

# ── multi-run ─────────────────────────────────────────────────────────────────
all_runs = []

for run_id in range(args.num_runs):
    run_start = time.perf_counter()
    seed = args.seed if args.num_runs == 1 else secrets.randbelow(2**31 - 1)

    clf = TabPFNClassifier.create_default_for_version(
        model_version,
        ignore_pretraining_limits=True,
        device=args.device,
        random_state=seed,
        n_estimators=args.n_estimators,
        fit_mode=args.fit_mode,
        categorical_features_indices=categorical_features_indices,
        **({"model_path": args.model_path} if args.model_path else {}),
    )

    clf.fit(X_train, y_train)
    preds       = clf.predict(X_test)
    test_acc    = accuracy_score(y_test, preds)
    test_f1     = f1_score(y_test, preds, average="macro", zero_division=0)

    runtime = time.perf_counter() - run_start
    all_runs.append({
        "run_id": run_id + 1,
        "seed": seed,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "runtime": runtime,
        "model_path": str(getattr(clf, "model_path", "")),
    })
    print(
        f"  Run {run_id+1}/{args.num_runs}: test_acc={test_acc:.4f}  "
        f"test_f1={test_f1:.4f}  "
        f"seed={seed} time={runtime:.2f}s"
    )

# ── statistics & save ─────────────────────────────────────────────────────────
vals  = [r["test_acc"] for r in all_runs]
f1_vals = [r["test_f1"] for r in all_runs]
runtimes = [r["runtime"] for r in all_runs]
stats = {
    "test_acc_mean": float(np.mean(vals)),
    "test_acc_std":  float(np.std(vals)),
    "test_acc_min":  float(np.min(vals)),
    "test_acc_max":  float(np.max(vals)),
    "test_f1_mean": float(np.mean(f1_vals)),
    "test_f1_std": float(np.std(f1_vals)),
    "test_f1_min": float(np.min(f1_vals)),
    "test_f1_max": float(np.max(f1_vals)),
    "runtime_mean": float(np.mean(runtimes)),
    "runtime_std": float(np.std(runtimes)),
    "runtime_min": float(np.min(runtimes)),
    "runtime_max": float(np.max(runtimes)),
    "total_runtime": time.perf_counter() - script_start,
}
print(f"\nTest Acc: {stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.4f}")

output = {
    "model":           "tabpfn",
    "task":            "classification",
    "dataset":         data_tag,
    "dataset_name":    args.dataset,
    "table_idx":       args.table_idx,
    "device":          args.device,
    "package_version": tabpfn_package_version,
    "model_version":   model_version.value,
    "n_estimators":    args.n_estimators,
    "fit_mode":        args.fit_mode,
    "model_path":      args.model_path,
    "categorical_features_indices": categorical_features_indices,
    "num_runs":        len(all_runs),
    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "individual_runs": all_runs,
    "statistics":      stats,
}
os.makedirs(osp.dirname(args.save_results), exist_ok=True)
with open(args.save_results, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.save_results}")
