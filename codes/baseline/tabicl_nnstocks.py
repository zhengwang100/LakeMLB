"""
TabICL classification on NNStocksDataset (table_idx 4-7).

Mirrors the interface of tabpfnv2_extend.py:
  --table_idx  4-7
  --num_runs   N runs with different seeds, report mean±std
  --save_results  path to output JSON
"""
import sys
import argparse
import json
import os
import os.path as osp
import random
from datetime import datetime
from pathlib import Path

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', "lib"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier

from rllm.types import ColType
from rllm.datasets import NNStocksDataset

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = osp.dirname(osp.realpath(__file__))
DATA_DIR    = osp.abspath(osp.join(SCRIPT_DIR, '..', 'data'))
RESULTS_DIR = osp.abspath(osp.join(SCRIPT_DIR, '..', 'results', 'tabicl'))
os.makedirs(RESULTS_DIR, exist_ok=True)

_NNSTOCKS_TABLE_NAMES = {
    4: "stocks_wiki_llm_1nn",
    5: "stocks_wiki_llm_2nn",
    6: "stocks_wiki_llm_4nn",
    7: "stocks_wiki_llm_8nn",
    8: "stocks_wiki_tfidf_1nn",
}

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--table_idx",    type=int, default=4, choices=[4, 5, 6, 7, 8])
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--num_runs",     type=int, default=5)
parser.add_argument("--device",       type=str, default="cuda:1")
parser.add_argument("--n_estimators", type=int, default=4)
parser.add_argument("--batch_size",   type=int, default=8)
parser.add_argument("--model_path",   type=str,
    default="../lib/huggingface/hub/models--jingang--TabICL-clf/snapshots/main/tabicl-classifier-v1.1-0506.ckpt")
parser.add_argument("--save_results", type=str, default=None)
parser.add_argument("--verbose",      action="store_true")
args = parser.parse_args()

# resolve model path
model_path = Path(args.model_path).expanduser()
if not model_path.is_absolute():
    model_path = Path(__file__).parent / model_path
args.model_path = str(model_path.resolve())

data_tag = _NNSTOCKS_TABLE_NAMES.get(args.table_idx, f"nnstocks_{args.table_idx}")
if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(RESULTS_DIR, f"tabicl_{data_tag}_{args.num_runs}runs_{ts}.json")

print(f"TabICL  Dataset: {data_tag}  Runs: {args.num_runs}")
print(f"Results → {args.save_results}")

# ── load dataset once ──────────────────────────────────────────────────────────
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

# TabICL is in-context: merge train+val for fitting (same as TabPFN)
train_val_mask = train_mask | val_mask
X = data_df[features].copy()
y = data_df[data_raw.target_col].copy()

X_train = X[train_val_mask].reset_index(drop=True)
y_train = y[train_val_mask].reset_index(drop=True)
X_test  = X[test_mask].reset_index(drop=True)
y_test  = y[test_mask].reset_index(drop=True)

for col in X_train.columns:
    if col in data_raw.col_types:
        if data_raw.col_types[col] == ColType.CATEGORICAL:
            X_train[col] = X_train[col].astype(str)
            X_test[col]  = X_test[col].astype(str)
        elif data_raw.col_types[col] == ColType.NUMERICAL:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype(float)
            X_test[col]  = pd.to_numeric(X_test[col],  errors='coerce').fillna(0).astype(float)

n_classes = y_train.nunique()
print(f"Dataset: train+val={X_train.shape[0]}, test={X_test.shape[0]}, classes={n_classes}")

# ── multi-run ──────────────────────────────────────────────────────────────────
all_runs = []

for run_id in range(args.num_runs):
    seed = args.seed + run_id * 100000

    clf = TabICLClassifier(
        n_estimators=args.n_estimators,
        norm_methods=["none", "power"],
        feat_shuffle_method="latin",
        class_shift=True,
        outlier_threshold=4.0,
        softmax_temperature=0.9,
        average_logits=True,
        use_hierarchical=True,
        batch_size=args.batch_size,
        use_amp=True,
        model_path=args.model_path,
        allow_auto_download=False,
        device=args.device,
        random_state=seed,
        verbose=args.verbose,
    )

    clf.fit(X_train, y_train)
    preds    = clf.predict(X_test)
    test_acc = accuracy_score(y_test, preds)

    all_runs.append({"run_id": run_id + 1, "seed": seed, "test_acc": test_acc})
    print(f"  Run {run_id+1}/{args.num_runs}: test_acc={test_acc:.4f}  seed={seed}")

# ── statistics & save ──────────────────────────────────────────────────────────
vals  = [r["test_acc"] for r in all_runs]
stats = {
    "test_acc_mean": float(np.mean(vals)),
    "test_acc_std":  float(np.std(vals)),
    "test_acc_min":  float(np.min(vals)),
    "test_acc_max":  float(np.max(vals)),
}
print(f"\nTest Acc: {stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.4f}")

output = {
    "model":           "tabicl",
    "task":            "classification",
    "dataset":         data_tag,
    "num_runs":        len(all_runs),
    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "individual_runs": all_runs,
    "statistics":      stats,
}
os.makedirs(osp.dirname(args.save_results) or ".", exist_ok=True)
with open(args.save_results, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.save_results}")
