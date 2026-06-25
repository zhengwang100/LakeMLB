"""
CARTE single-table regression on GACars.

Defaults: german_reg (price_in_euro).
Also supports gacars_da_reg / gacars_fa_reg via --data_name.
Metrics : RMSE, MAE, R²  (mean ± std across runs).
"""
import os
from pathlib import Path
import sys
import os.path as osp

os.chdir(Path().cwd().parent)
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', "lib"))
sys.path.insert(0, osp.dirname(__file__))

import argparse
import json
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sklearn_shuffle

# Use threading backend so joblib workers share the parent process's memory
# (avoids pickling CUDA tensors across loky subprocesses → BrokenProcessPool)
joblib.parallel_backend('threading')

from carte_ai.src.carte_estimator import CARTERegressor
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer
from carte_ai.configs.directory import config_directory

# ── default mask paths per dataset ───────────────────────────────────────────
SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
GACARS_RAW = osp.abspath(osp.join(SCRIPT_DIR, '..', 'data', 'table_gacars', 'raw'))
CARTE_RAW  = config_directory['data_raw']

DEFAULT_MASK = {
    "german_reg":     osp.join(CARTE_RAW,  "german_mask_reg.pt"),
    "gacars_fa_reg":  osp.join(CARTE_RAW,  "german_mask_reg.pt"),  # same rows
    "gacars_da_reg":  osp.join(GACARS_RAW, "mask_da_reg.pt"),
}
FASTTEXT_PATH = osp.join(SCRIPT_DIR, '..', "lib", "FastText", "cc.en.300.bin")
RESULTS_DIR   = osp.abspath(osp.join(SCRIPT_DIR, '..', 'results', 'carte_reg'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='CARTE single-table regression')
parser.add_argument('--data_name',    type=str,   default='german_reg')
parser.add_argument('--mask_path',    type=str,   default=None,
                    help='Override default mask path')
parser.add_argument('--num_model',    type=int,   default=5,
                    help='Bagging ensemble size')
parser.add_argument('--num_runs',     type=int,   default=5)
parser.add_argument('--seed',         type=int,   default=0)
parser.add_argument('--device',       type=str,   default='cuda:0')
parser.add_argument('--save_results', type=str,   default=None)
args = parser.parse_args()

if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(
        RESULTS_DIR,
        f"carte_single_{args.data_name}_{args.num_runs}runs_{ts}.json"
    )

mask_path = args.mask_path or DEFAULT_MASK.get(args.data_name)
if mask_path is None:
    raise ValueError(f"No default mask for '{args.data_name}'. Pass --mask_path.")

print(f"Dataset   : {args.data_name}")
print(f"Mask      : {mask_path}")
print(f"Device    : {args.device}")
print(f"Num runs  : {args.num_runs}  (ensemble size per run: {args.num_model})")


# ── data helpers ──────────────────────────────────────────────────────────────
def _load_data(data_name):
    df = pd.read_parquet(
        f"{config_directory['data_singletable']}/{data_name}/raw.parquet"
    )
    df.fillna(value=np.nan, inplace=True)
    with open(f"{config_directory['data_singletable']}/{data_name}/config_data.json") as f:
        cfg = json.load(f)
    return df, cfg


def _load_masks(path):
    masks = torch.load(path, weights_only=False)
    tn = masks['train_mask'].sum().item()
    vn = masks['val_mask'].sum().item()
    en = masks['test_mask'].sum().item()
    tot = masks['train_mask'].numel()
    print(f"  Masks: train={tn} ({tn/tot*100:.1f}%), "
          f"val={vn} ({vn/tot*100:.1f}%), "
          f"test={en} ({en/tot*100:.1f}%)")
    return masks


def _split(data, target_name, masks):
    X = data.drop(columns=target_name)
    y = data[target_name].to_numpy(dtype=float)
    tm = masks['train_mask'].numpy()
    vm = masks['val_mask'].numpy()
    em = masks['test_mask'].numpy()
    return X[tm], X[vm], X[em], y[tm], y[vm], y[em]


# ── fixed-split subclass ──────────────────────────────────────────────────────
class CARTERegressorFixedSplit(CARTERegressor):
    """CARTERegressor that uses pre-defined train / val index arrays."""

    def __init__(self, fixed_train_idx, fixed_val_idx, **kwargs):
        super().__init__(**kwargs)
        self.fixed_train_idx = np.array(fixed_train_idx)
        self.fixed_val_idx   = np.array(fixed_val_idx)

    def _set_train_valid_split(self):
        return [(self.fixed_train_idx, self.fixed_val_idx)
                for _ in range(self.num_model)]


# ── load & graph-transform once (shared across all runs) ─────────────────────
print("\n[1] Loading data …")
data, cfg = _load_data(args.data_name)
target = cfg["target_name"]
print(f"  {len(data)} rows  target='{target}'  "
      f"y∈[{data[target].min():.3f}, {data[target].max():.3f}]")

print("\n[2] Loading masks …")
masks = _load_masks(mask_path)

print("\n[3] Splitting …")
X_tr, X_va, X_te, y_tr, y_va, y_te = _split(data, target, masks)
print(f"  train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")

print("\n[4] Building graphs (done once, reused per run) …")
X_tr_sh, y_tr_sh = sklearn_shuffle(X_tr, y_tr, random_state=0)
X_va_sh, y_va_sh = sklearn_shuffle(X_va, y_va, random_state=0)
X_tv   = pd.concat([X_tr_sh, X_va_sh], axis=0, ignore_index=True)
y_tv   = np.concatenate([y_tr_sh, y_va_sh])

preprocessor = Table2GraphTransformer(
    fasttext_model_path=FASTTEXT_PATH,
    num_transformer=StandardScaler(),
)
X_tv_graphs = preprocessor.fit_transform(X_tv, y=y_tv)
X_te_graphs = preprocessor.transform(X_te)
print(f"  train+val graphs={len(X_tv_graphs)}, test graphs={len(X_te_graphs)}")

carte_train_idx = np.arange(len(y_tr))
carte_val_idx   = np.arange(len(y_tr), len(y_tr) + len(y_va))


# ── multi-run experiment ──────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }

random.seed(args.seed)
seeds = [args.seed + i for i in range(args.num_runs)]

all_results = []
print(f"\n[5] Running {args.num_runs} experiments …")

for run_id, seed in enumerate(seeds):
    print(f"\n  [Run {run_id+1}/{args.num_runs}]  random_state={seed}")
    estimator = CARTERegressorFixedSplit(
        fixed_train_idx=carte_train_idx,
        fixed_val_idx=carte_val_idx,
        loss="squared_error",
        scoring="r2_score",
        num_model=args.num_model,
        n_jobs=args.num_model,
        random_state=seed,
        device=args.device,
        num_layers=1,
        batch_size=256,
        learning_rate=1e-3,
        early_stopping_patience=40,
        val_size=0.125,
        disable_pbar=True,
    )
    estimator.fit(X=X_tv_graphs, y=y_tv)

    y_pred = estimator.predict(X_te_graphs)
    m = compute_metrics(y_te, y_pred)
    run_res = {"run_id": run_id+1, "seed": seed,
               **{f"test_{k}": v for k, v in m.items()}}
    all_results.append(run_res)
    print(f"    Test RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")


# ── statistics & save ─────────────────────────────────────────────────────────
def summarise(results):
    stats = {}
    for k in ["test_rmse", "test_mae", "test_r2"]:
        vals = [r[k] for r in results]
        stats[f"{k}_mean"] = float(np.mean(vals))
        stats[f"{k}_std"]  = float(np.std(vals))
    return stats

stats = summarise(all_results)
fmt = lambda k: f"{stats[f'{k}_mean']:.4f}±{stats[f'{k}_std']:.4f}"

print(f"\n{'='*60}")
print(f"Summary ({args.num_runs} runs) [mean ± std]")
print(f"{'='*60}")
print(f"  RMSE={fmt('test_rmse')}  MAE={fmt('test_mae')}  R²={fmt('test_r2')}")
print(f"{'='*60}")

output = {
    "model":      "carte_single",
    "task":       "price_regression",
    "data_name":  args.data_name,
    "num_runs":   args.num_runs,
    "num_model":  args.num_model,
    "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "individual_runs": all_results,
    "statistics":      stats,
}
with open(args.save_results, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.save_results}")
