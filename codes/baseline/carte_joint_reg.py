"""
CARTE multi-table regression on GACars.

Default : target = german_reg  (price_in_euro)
          source = australian_reg  (Price / AUD, labelled)

CARTE's native multi-table learning: the source table is used during joint
training via CARTEMultitableRegressor.  All evaluation is on the target
table's test split.

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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sklearn_shuffle

from carte_ai.src.carte_estimator import CARTEMultitableRegressor
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer
from carte_ai.configs.directory import config_directory

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
GACARS_RAW = osp.abspath(osp.join(SCRIPT_DIR, '..', 'data', 'table_gacars', 'raw'))
CARTE_RAW  = config_directory['data_raw']

DEFAULT_MASK = {
    "german_reg":    osp.join(CARTE_RAW,  "german_mask_reg.pt"),
    "gacars_fa_reg": osp.join(CARTE_RAW,  "german_mask_reg.pt"),
    "gacars_da_reg": osp.join(GACARS_RAW, "mask_da_reg.pt"),
}
FASTTEXT_PATH = osp.join(SCRIPT_DIR, '..', "lib", "FastText", "cc.en.300.bin")
RESULTS_DIR   = osp.abspath(osp.join(SCRIPT_DIR, '..', 'results', 'carte_reg'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='CARTE multi-table regression')
parser.add_argument('--target_data',  type=str,   default='german_reg')
parser.add_argument('--source_data',  type=str,   nargs='+',
                    default=['australian_reg'],
                    help='One or more source dataset names')
parser.add_argument('--mask_path',    type=str,   default=None,
                    help='Override default mask path for the target table')
parser.add_argument('--num_model',    type=int,   default=5)
parser.add_argument('--num_runs',     type=int,   default=1)
parser.add_argument('--seed',         type=int,   default=0)
parser.add_argument('--device',       type=str,   default='cuda:1')
parser.add_argument('--save_results', type=str,   default=None)
args = parser.parse_args()

if args.save_results is None:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = "+".join(args.source_data)
    args.save_results = osp.join(
        RESULTS_DIR,
        f"carte_joint_{args.target_data}_{src}_{args.num_runs}runs_{ts}.json"
    )

mask_path = args.mask_path or DEFAULT_MASK.get(args.target_data)
if mask_path is None:
    raise ValueError(f"No default mask for '{args.target_data}'. Pass --mask_path.")

print(f"Target    : {args.target_data}")
print(f"Source(s) : {args.source_data}")
print(f"Mask      : {mask_path}")
print(f"Device    : {args.device}")
print(f"Num runs  : {args.num_runs}  (ensemble size: {args.num_model})")


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
class CARTEMultitableRegressorFixedSplit(CARTEMultitableRegressor):
    """CARTEMultitableRegressor with pre-defined train / val index arrays."""

    def __init__(self, fixed_train_idx, fixed_val_idx, **kwargs):
        super().__init__(**kwargs)
        self.fixed_train_idx = np.array(fixed_train_idx)
        self.fixed_val_idx   = np.array(fixed_val_idx)

    def _set_train_valid_split(self):
        return [(self.fixed_train_idx, self.fixed_val_idx)
                for _ in range(self.num_model)]


# ── multi-table graph preparation ─────────────────────────────────────────────
def prepare_multitable(data_t, cfg_t, masks_t, data_s_all, cfg_s_all):
    """
    Build graph lists for target (train+val, test) and all sources.

    For source tables:
    - labelled  → keep original float y; drop NaN-y rows
    - unlabelled → y set to NaN (float sentinel)

    The `d.domain` attribute marks the table origin (0 = target, 1+ = source).
    """
    # -- target table --
    Xt_tr, Xt_va, Xt_te, yt_tr, yt_va, yt_te = _split(
        data_t, cfg_t["target_name"], masks_t
    )
    Xt_tr_sh, yt_tr_sh = sklearn_shuffle(Xt_tr, yt_tr, random_state=0)
    Xt_va_sh, yt_va_sh = sklearn_shuffle(Xt_va, yt_va, random_state=0)
    Xt_tv = pd.concat([Xt_tr_sh, Xt_va_sh], axis=0, ignore_index=True)
    yt_tv = np.concatenate([yt_tr_sh, yt_va_sh])

    graph = Table2GraphTransformer(
        fasttext_model_path=FASTTEXT_PATH,
        num_transformer=StandardScaler(),
    )
    Xt_carte_tv = graph.fit_transform(X=Xt_tv, y=yt_tv)
    Xt_carte_te = graph.transform(Xt_te)

    for d in Xt_carte_tv + Xt_carte_te:
        d.domain = 0

    carte_train_idx = np.arange(len(yt_tr))
    carte_val_idx   = np.arange(len(yt_tr), len(yt_tr) + len(yt_va))

    # -- source tables --
    Xs_carte = {}
    domain_marker = 1
    for name, df_s in data_s_all.items():
        cfg_s = cfg_s_all[name]
        is_unlabeled = cfg_s["target_name"] is None

        if is_unlabeled:
            Xs_temp    = graph.fit_transform(X=df_s, y=None)
            Xs_pruned  = Xs_temp
        else:
            Xs_temp = graph.fit_transform(
                X=df_s.drop(columns=cfg_s["target_name"]),
                y=df_s[cfg_s["target_name"]].to_numpy(dtype=float),
            )
            # drop rows whose target is NaN
            ys = np.array([d.y.cpu().item() for d in Xs_temp])
            keep       = ~np.isnan(ys)
            Xs_pruned  = [d for d, k in zip(Xs_temp, keep) if k]

        for d in Xs_pruned:
            if is_unlabeled:
                # float NaN sentinel for unlabelled regression source
                d.y = torch.tensor([float('nan')], dtype=torch.float)
            else:
                # keep as float (regression)
                d.y = torch.tensor([d.y.cpu().item()], dtype=torch.float)
            d.domain = domain_marker
        Xs_carte[name] = Xs_pruned
        domain_marker += 1

    return (Xt_carte_tv, Xt_carte_te, Xs_carte,
            yt_tv, yt_te, carte_train_idx, carte_val_idx)


# ── load data ─────────────────────────────────────────────────────────────────
print("\n[1] Loading data …")
data_t, cfg_t = _load_data(args.target_data)
tgt = cfg_t["target_name"]
print(f"  Target: {args.target_data}  ({len(data_t)} rows)  "
      f"target='{tgt}'  y∈[{data_t[tgt].min():.3f}, {data_t[tgt].max():.3f}]")

data_s_all, cfg_s_all = {}, {}
for nm in args.source_data:
    df_s, cfg_s = _load_data(nm)
    data_s_all[nm] = df_s
    cfg_s_all[nm]  = cfg_s
    stgt = cfg_s["target_name"]
    print(f"  Source: {nm}  ({len(df_s)} rows)  target='{stgt}'")

print("\n[2] Loading masks …")
masks_t = _load_masks(mask_path)

print("\n[3] Building graphs (done once, reused per run) …")
(Xt_tv, Xt_te, Xs_carte,
 yt_tv, yt_te,
 carte_train_idx, carte_val_idx) = prepare_multitable(
    data_t, cfg_t, masks_t, data_s_all, cfg_s_all
)
print(f"  Target train+val={len(yt_tv)}, test={len(yt_te)}")
for nm, graphs in Xs_carte.items():
    print(f"  Source {nm}: {len(graphs)} graphs")


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
print(f"\n[4] Running {args.num_runs} experiments …")

for run_id, seed in enumerate(seeds):
    print(f"\n  [Run {run_id+1}/{args.num_runs}]  random_state={seed}")

    estimator = CARTEMultitableRegressorFixedSplit(
        fixed_train_idx=carte_train_idx,
        fixed_val_idx=carte_val_idx,
        source_data=Xs_carte,
        loss="squared_error",
        scoring="r2_score",
        num_model=args.num_model,
        n_jobs=5,   # loky (process) backend: graphs are pre-built CPU tensors → safe to pickle
        random_state=seed,
        device=args.device,
        num_layers=1,
        batch_size=256,
        learning_rate=1e-3,
        early_stopping_patience=40,
        val_size=0.125,
        disable_pbar=True,
    )
    estimator.fit(Xt_tv, yt_tv)

    y_pred = estimator.predict(Xt_te)
    m = compute_metrics(yt_te, y_pred)
    run_res = {"run_id": run_id+1, "seed": seed,
               **{f"test_{k}": v for k, v in m.items()}}
    all_results.append(run_res)
    print(f"    Test RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")

    # print domain weights for first run only (diagnostic)
    if run_id == 0:
        print(f"    Domain weights: "
              + "  ".join(f"{d}:{w:.4f}" for d, w in
                          zip(estimator.source_list_total_, estimator.weights_)))


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
    "model":       "carte_joint",
    "task":        "price_regression",
    "target_data": args.target_data,
    "source_data": args.source_data,
    "num_runs":    args.num_runs,
    "num_model":   args.num_model,
    "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "individual_runs": all_results,
    "statistics":      stats,
}
with open(args.save_results, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.save_results}")
