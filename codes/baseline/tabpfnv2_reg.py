"""
TabPFN v2 for regression (GACars price regression).

Task    : price_in_euro regression (continuous)
Tables  : 4=German Reg, 5=Australian Reg, 6=DA Reg, 7=FA Reg
Metrics : RMSE, MAE, R²  (mean ± std across runs)

TabPFN is an in-context learner – no gradient training.
Multi-run variability comes from the ensemble random_state.
"""

import sys
import argparse
import os
import os.path as osp
import json
import random
from datetime import datetime

sys.path.append("./")
sys.path.append("../")
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', "lib"))
sys.path.insert(0, osp.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from rllm.types import ColType
from rllm.datasets.lakemlb.gacars import GACarsDataset
from rllm.transforms.table_transforms import DefaultTableTransform


TABLE_NAMES = {
    4: "german_reg",
    5: "australian_reg",
    6: "gacars_da_reg",
    7: "gacars_fa_reg",
}

SCRIPT_DIR  = osp.dirname(osp.realpath(__file__))
DATA_DIR    = osp.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = osp.join(SCRIPT_DIR, "..", "results", "tabpfnv2_reg")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="TabPFN v2 regression on GACars price prediction"
)
parser.add_argument("--f_dim", type=int, default=32,
                    help="Feature embedding dim for DefaultTableTransform")
parser.add_argument("--table_idx", type=int, default=4, choices=[4, 5, 6, 7],
                    help="GACarsDataset table: 4=German Reg, 5=Australian Reg, "
                         "6=DA Reg, 7=FA Reg")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_runs", type=int, default=5,
                    help="Number of runs (different ensemble random_state each run)")
parser.add_argument("--save_results", type=str, default=None)
args = parser.parse_args()

table_name = TABLE_NAMES.get(args.table_idx, f"table_{args.table_idx}")
if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(
        RESULTS_DIR,
        f"tabpfnv2_{table_name}_{args.num_runs}runs_{ts}.json"
    )


# ── load dataset ──────────────────────────────────────────────────────────────
print(f"Loading GACarsDataset[{args.table_idx}] ({table_name}) …")
dataset  = GACarsDataset(cached_dir=DATA_DIR, force_reload=False)
data_raw = dataset[args.table_idx]

transform = DefaultTableTransform(out_dim=args.f_dim)
transform(data_raw)

# Feature columns (exclude target)
features = [c for c in data_raw.col_types.keys() if c != data_raw.target_col]
data_df  = data_raw.df

X = data_df[features].copy()
y = data_df[data_raw.target_col].copy()

# Masks
def to_np(mask):
    return mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

train_mask = to_np(data_raw.train_mask)
val_mask   = to_np(data_raw.val_mask)
test_mask  = to_np(data_raw.test_mask)

# Combine train+val for TabPFN (no gradient training; using more context is fine)
fit_mask = train_mask | val_mask

X_fit  = X[fit_mask].reset_index(drop=True)
y_fit  = y[fit_mask].reset_index(drop=True)
X_test = X[test_mask].reset_index(drop=True)
y_test = y[test_mask].reset_index(drop=True)

# Also keep train-only split for reporting train metrics
X_train = X[train_mask].reset_index(drop=True)
y_train = y[train_mask].reset_index(drop=True)

# Type coercion: categorical -> str, numerical -> float
for col in X_fit.columns:
    if col not in data_raw.col_types:
        continue
    if data_raw.col_types[col] == ColType.CATEGORICAL:
        for df_ in [X_fit, X_test, X_train]:
            df_[col] = df_[col].astype(str)
    elif data_raw.col_types[col] == ColType.NUMERICAL:
        for df_ in [X_fit, X_test, X_train]:
            df_[col] = pd.to_numeric(df_[col], errors="coerce").fillna(0).astype(float)

y_fit_vals  = y_fit.to_numpy().astype(float)
y_test_vals = y_test.to_numpy().astype(float)

print(f"  fit (train+val)={len(X_fit)}  test={len(X_test)}")
print(f"  y range=[{y_fit_vals.min():.4f}, {y_fit_vals.max():.4f}]  "
      f"mean={y_fit_vals.mean():.4f}  std={y_fit_vals.std():.4f}")


# ── helpers ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ── multi-run evaluation ──────────────────────────────────────────────────────
print(f"\nRunning {args.num_runs} experiments …")
all_results = []

random.seed(args.seed)
seeds = [args.seed] + [args.seed + random.randint(1, 10000)
                       for _ in range(args.num_runs - 1)]

for run_id, seed in enumerate(seeds):
    regr = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        random_state=seed,
        ignore_pretraining_limits=True,
        device=args.device,
    )

    regr.fit(X_fit, y_fit_vals)
    test_pred = regr.predict(X_test)
    m = compute_metrics(y_test_vals, test_pred)

    run_result = {
        "run_id": run_id + 1,
        "seed":   seed,
        **{f"test_{k}": v for k, v in m.items()},
    }
    all_results.append(run_result)
    print(f"  Run {run_id+1}/{args.num_runs}: "
          f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")


# ── statistics ────────────────────────────────────────────────────────────────
def compute_statistics(results):
    metrics = ["test_rmse", "test_mae", "test_r2"]
    stats = {}
    for m in metrics:
        vals = [r[m] for r in results]
        stats[f"{m}_mean"] = float(np.mean(vals))
        stats[f"{m}_std"]  = float(np.std(vals))
        stats[f"{m}_min"]  = float(np.min(vals))
        stats[f"{m}_max"]  = float(np.max(vals))
    return stats


def fmt(stats, m):
    return f"{stats[f'{m}_mean']:.4f}±{stats[f'{m}_std']:.4f}"

stats = compute_statistics(all_results)
print(f"\n{'='*60}")
print(f"Summary ({args.num_runs} runs)  [mean ± std]")
print(f"{'='*60}")
print(f"  Test RMSE={fmt(stats,'test_rmse')}  "
      f"MAE={fmt(stats,'test_mae')}  R²={fmt(stats,'test_r2')}")
print(f"{'='*60}")


# ── save ──────────────────────────────────────────────────────────────────────
output = {
    "model":    "tabpfnv2",
    "task":     "price_regression",
    "dataset":  f"GACarsDataset / {table_name}",
    "num_runs": args.num_runs,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "note":     "fit on train+val (TabPFN in-context learner)",
    "individual_runs": all_results,
    "statistics":      stats,
}
with open(args.save_results, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.save_results}")
