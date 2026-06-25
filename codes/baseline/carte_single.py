"""
CARTE single-table classification with fixed train/val/test split.

Supports parameterized data_name / mask_basename, multi-run evaluation.
"""
import os
import sys
import argparse
import json
import random
from pathlib import Path
from datetime import datetime

import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', "lib"))
sys.path.insert(0, osp.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sklearn_shuffle
from carte_ai.src.carte_estimator import CARTEClassifier
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer
from carte_ai.configs.directory import config_directory

# ── args ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = osp.dirname(osp.realpath(__file__))
RESULTS_DIR = osp.abspath(osp.join(SCRIPT_DIR, '..', 'results', 'carte_cls'))
os.makedirs(RESULTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description='CARTE Single Table Classification')
parser.add_argument('--data_name',    type=str, default='stocks_wiki_llm_1nn',
                    help='Dataset name (must match data_singletable/<name>/)')
parser.add_argument('--mask_basename', type=str, default='nnlist',
                    help='Mask file basename (e.g. nnlist → mask_nnlist.pt)')
parser.add_argument('--num_model',    type=int, default=5,
                    help='Number of CARTE ensemble models per run')
parser.add_argument('--device',       type=str, default='cuda:0')
parser.add_argument('--num_runs',     type=int, default=5)
parser.add_argument('--seed',         type=int, default=0)
parser.add_argument('--save_results', type=str, default=None)
args = parser.parse_args()

if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(
        RESULTS_DIR, f"carte_single_{args.data_name}_{args.num_runs}runs_{ts}.json"
    )

print(f"CARTE Single Table [{args.data_name}]  mask={args.mask_basename}  runs={args.num_runs}")
print("=" * 80)

# ── helpers ───────────────────────────────────────────────────────────────────
def _load_data(data_name):
    data_pd = pd.read_parquet(
        f"{config_directory['data_singletable']}/{data_name}/raw.parquet"
    )
    data_pd.fillna(value=np.nan, inplace=True)
    with open(f"{config_directory['data_singletable']}/{data_name}/config_data.json") as f:
        config_data = json.load(f)
    return data_pd, config_data


def _load_masks(mask_basename):
    mask_path = osp.join(config_directory['data_raw'], f"mask_{mask_basename}.pt")
    if not osp.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    masks = torch.load(mask_path, weights_only=False)
    train_n = masks['train_mask'].sum().item()
    val_n   = masks['val_mask'].sum().item()
    test_n  = masks['test_mask'].sum().item()
    total   = masks['train_mask'].numel()
    print(f"Masks: train={train_n}({train_n/total*100:.1f}%) "
          f"val={val_n}({val_n/total*100:.1f}%) "
          f"test={test_n}({test_n/total*100:.1f}%)")
    return masks


def _split(data, target_name, masks):
    X = data.drop(columns=target_name)
    y = data[target_name].to_numpy()
    tm  = masks['train_mask'].numpy()
    vm  = masks['val_mask'].numpy()
    tsm = masks['test_mask'].numpy()
    return X[tm], X[vm], X[tsm], y[tm], y[vm], y[tsm]


class CARTEClassifierFixedSplit(CARTEClassifier):
    """CARTE Classifier with fixed train/val indices."""
    def __init__(self, fixed_train_idx, fixed_val_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_train_idx = np.array(fixed_train_idx)
        self.fixed_val_idx   = np.array(fixed_val_idx)

    def _set_train_valid_split(self):
        assert self.fixed_train_idx.max() < len(self.X_)
        assert self.fixed_val_idx.max()   < len(self.X_)
        return [(self.fixed_train_idx, self.fixed_val_idx)
                for _ in range(self.num_model)]


# ── load data & masks once ────────────────────────────────────────────────────
print("\n[1] Loading data...")
data, data_config = _load_data(args.data_name)
print(f"Loaded {len(data)} samples")

print("\n[2] Loading masks...")
masks = _load_masks(args.mask_basename)

print("\n[3] Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = _split(
    data, data_config["target_name"], masks
)

print("\n[4] Building graph representation...")
fasttext_path = osp.join(SCRIPT_DIR, '..', "lib", "FastText", "cc.en.300.bin")
preprocessor  = Table2GraphTransformer(
    fasttext_model_path=fasttext_path,
    num_transformer=StandardScaler(),
)

X_tv_shuf, y_tv_shuf = sklearn_shuffle(
    pd.concat([X_train, X_val], axis=0, ignore_index=True),
    np.concatenate([y_train, y_val]),
    random_state=0,
)
X_train_val_graphs = preprocessor.fit_transform(X_tv_shuf, y=y_tv_shuf)
X_test_graphs      = preprocessor.transform(X_test)
print(f"Graphs: train+val={len(X_train_val_graphs)}, test={len(X_test_graphs)}")

carte_train_idx = np.arange(len(y_train))
carte_val_idx   = np.arange(len(y_train), len(y_train) + len(y_val))

# ── multi-run ─────────────────────────────────────────────────────────────────
print("\n[5] Multi-run training...")
print("=" * 80)

all_runs = []

for run_id in range(args.num_runs):
    seed = args.seed if run_id == 0 else args.seed + random.randint(1, 10000)

    fixed_params = {
        "num_model":             args.num_model,
        "disable_pbar":          True,
        "random_state":          seed,
        "device":                args.device,
        "n_jobs":                args.num_model,
        "loss":                  "categorical_crossentropy",
        "scoring":               "accuracy",
        "num_layers":            1,
        "batch_size":            256,
        "learning_rate":         1e-3,
        "dropout":               0,
        "val_size":              0.125,
        "early_stopping_patience": 40,
    }
    estimator = CARTEClassifierFixedSplit(
        fixed_train_idx=carte_train_idx,
        fixed_val_idx=carte_val_idx,
        **fixed_params,
    )
    estimator.fit(X=X_train_val_graphs, y=y_tv_shuf)
    preds    = estimator.predict(X_test_graphs)
    test_acc = accuracy_score(y_test, preds)

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
    "model":           "carte_single",
    "task":            "classification",
    "dataset":         args.data_name,
    "num_runs":        len(all_runs),
    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "individual_runs": all_runs,
    "statistics":      stats,
}
os.makedirs(osp.dirname(args.save_results) or ".", exist_ok=True)
with open(args.save_results, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.save_results}")
