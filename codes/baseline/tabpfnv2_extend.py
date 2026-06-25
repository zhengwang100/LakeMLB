"""
TabPFN v2 + ManyClassClassifier for classification (>10 classes supported).

Supports NNStocksDataset (table_idx 4-7) and multi-run evaluation.
"""
import sys
import argparse
import json
import random
import os
import os.path as osp
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
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn_extensions.many_class import ManyClassClassifier

from rllm.types import ColType
from rllm.datasets import NNStocksDataset
from rllm.transforms.table_transforms import DefaultTableTransform

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = osp.dirname(osp.realpath(__file__))
DATA_DIR    = osp.abspath(osp.join(SCRIPT_DIR, '..', 'data'))
RESULTS_DIR = osp.abspath(osp.join(SCRIPT_DIR, '..', 'results', 'tabpfn'))
os.makedirs(RESULTS_DIR, exist_ok=True)

_NNSTOCKS_TABLE_NAMES = {
    4: "stocks_wiki_llm_1nn",
    5: "stocks_wiki_llm_2nn",
    6: "stocks_wiki_llm_4nn",
    7: "stocks_wiki_llm_8nn",
    8: "stocks_wiki_tfidf_1nn",
}

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--table_idx", type=int, default=4,
                    choices=[4, 5, 6, 7, 8],
                    help="NNStocksDataset index: 4=llm_1nn,5=llm_2nn,6=llm_4nn,7=llm_8nn,8=tfidf_1nn")
parser.add_argument("--f_dim",        type=int,   default=32)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--device",       type=str,   default="cuda:1")
parser.add_argument("--num_runs",     type=int,   default=5)
parser.add_argument("--save_results", type=str,   default=None)
args = parser.parse_args()

data_tag = _NNSTOCKS_TABLE_NAMES.get(args.table_idx, f"nnstocks_{args.table_idx}")
if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(RESULTS_DIR, f"tabpfn_{data_tag}_{args.num_runs}runs_{ts}.json")

print(f"TabPFN v2 (ManyClass)  Dataset: {data_tag}  Runs: {args.num_runs}")
print(f"Results → {args.save_results}")

# ── load dataset once (raw DataFrame + masks via NNStocksDataset) ─────────────
dataset  = NNStocksDataset(cached_dir=DATA_DIR, force_reload=False)
data_raw = dataset[args.table_idx]

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

# type coercion (mirrors tabpfnv2.py)
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

# ── multi-run ─────────────────────────────────────────────────────────────────
all_runs = []

for run_id in range(args.num_runs):
    seed = args.seed if run_id == 0 else args.seed + random.randint(1, 10000)

    base_clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        ignore_pretraining_limits=True,
        device=args.device,
    )
    clf = ManyClassClassifier(
        estimator=base_clf,
        alphabet_size=10,
        random_state=seed,
        verbose=0,
    )

    clf.fit(X_train, y_train)
    preds       = clf.predict(X_test)
    test_acc    = accuracy_score(y_test, preds)

    all_runs.append({"run_id": run_id + 1, "seed": seed, "test_acc": test_acc})
    print(f"  Run {run_id+1}/{args.num_runs}: test_acc={test_acc:.4f}  seed={seed}")

# ── statistics & save ─────────────────────────────────────────────────────────
vals  = [r["test_acc"] for r in all_runs]
stats = {
    "test_acc_mean": float(np.mean(vals)),
    "test_acc_std":  float(np.std(vals)),
    "test_acc_min":  float(np.min(vals)),
    "test_acc_max":  float(np.max(vals)),
}
print(f"\nTest Acc: {stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.4f}")

output = {
    "model":           "tabpfn_manyclass",
    "task":            "classification",
    "dataset":         data_tag,
    "num_runs":        len(all_runs),
    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "individual_runs": all_runs,
    "statistics":      stats,
}
os.makedirs(osp.dirname(args.save_results), exist_ok=True)
with open(args.save_results, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.save_results}")
