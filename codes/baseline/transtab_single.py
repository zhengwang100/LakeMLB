"""
TransTab single-dataset classification.

Supports multiple datasets via --data_name.
Metrics: accuracy (mean ± std across runs).
"""
import sys
import os
import os.path as osp
import argparse
import json
import random
import shutil
from datetime import datetime

sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', "lib"))

import transtab
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from rllm.types import ColType

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = osp.dirname(osp.realpath(__file__))
NNSTOCKS_RAW_DIR = osp.abspath(osp.join(SCRIPT_DIR, '..', 'data', 'table_nnstocks', 'raw'))
RESULTS_DIR     = osp.abspath(osp.join(SCRIPT_DIR, '..', 'results', 'transtab_cls'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# carte data_raw for mask files
sys.path.insert(0, osp.join(SCRIPT_DIR, '..', 'lib'))
from carte_ai.configs.directory import config_directory as _CARTE_CFG

# ── per-dataset column types ───────────────────────────────────────────────────
_NNSTOCKS_COLS = {
    "symbol":        ColType.CATEGORICAL,
    "name":          ColType.CATEGORICAL,
    "lastsale":      ColType.NUMERICAL,
    "netchange":     ColType.NUMERICAL,
    "pctchange":     ColType.NUMERICAL,
    "volume":        ColType.NUMERICAL,
    "marketCap":     ColType.NUMERICAL,
    "country":       ColType.CATEGORICAL,
    "ipoyear":       ColType.NUMERICAL,
    "sector":        ColType.CATEGORICAL,
    "url":           ColType.CATEGORICAL,
    "wiki_title":    ColType.CATEGORICAL,
    "wiki_url":      ColType.CATEGORICAL, #
    "company_type":  ColType.CATEGORICAL,
    "traded_as":     ColType.CATEGORICAL,
    "founded":       ColType.CATEGORICAL,
    "headquarters":  ColType.CATEGORICAL,
    "num_locations": ColType.CATEGORICAL,
    "area_served":   ColType.CATEGORICAL,
    "key_people":    ColType.CATEGORICAL,
    "services":      ColType.CATEGORICAL,
    "revenue":       ColType.CATEGORICAL,
    "operating_income": ColType.CATEGORICAL,
    "net_income":    ColType.CATEGORICAL,
    "total_assets":  ColType.CATEGORICAL,
    "total_equity":  ColType.CATEGORICAL,
    "num_employees": ColType.CATEGORICAL,
    "subsidiaries":  ColType.CATEGORICAL,
    "website":       ColType.CATEGORICAL, #
    "founders":      ColType.CATEGORICAL,
    "formerly":      ColType.CATEGORICAL,
    "products":      ColType.CATEGORICAL,
    "isin":          ColType.CATEGORICAL,
}

DATA_CONFIGS = {
    "stocks_wiki_llm_1nn": {
        "data_dir":  NNSTOCKS_RAW_DIR,
        "csv":       "stocks_wiki_llm_1nn.csv",
        "mask":      osp.join(_CARTE_CFG['data_raw'], "mask_nnlist.pt"),
        "target":    "sector",
        "num_class": 11,
        "cols":      _NNSTOCKS_COLS,
    },
    "stocks_wiki_llm_2nn": {
        "data_dir":  NNSTOCKS_RAW_DIR,
        "csv":       "stocks_wiki_llm_2nn.csv",
        "mask":      osp.join(_CARTE_CFG['data_raw'], "mask_nnlist.pt"),
        "target":    "sector",
        "num_class": 11,
        "cols":      _NNSTOCKS_COLS,
    },
    "stocks_wiki_llm_4nn": {
        "data_dir":  NNSTOCKS_RAW_DIR,
        "csv":       "stocks_wiki_llm_4nn.csv",
        "mask":      osp.join(_CARTE_CFG['data_raw'], "mask_nnlist.pt"),
        "target":    "sector",
        "num_class": 11,
        "cols":      _NNSTOCKS_COLS,
    },
    "stocks_wiki_llm_8nn": {
        "data_dir":  NNSTOCKS_RAW_DIR,
        "csv":       "stocks_wiki_llm_8nn.csv",
        "mask":      osp.join(_CARTE_CFG['data_raw'], "mask_nnlist.pt"),
        "target":    "sector",
        "num_class": 11,
        "cols":      _NNSTOCKS_COLS,
    },
    "stocks_wiki_tfidf_1nn": {
        "data_dir":  NNSTOCKS_RAW_DIR,
        "csv":       "stocks_wiki_tfidf_1nn.csv",
        "mask":      osp.join(_CARTE_CFG['data_raw'], "mask_nnlist.pt"),
        "target":    "sector",
        "num_class": 11,
        "cols":      _NNSTOCKS_COLS,
    },
}

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='TransTab single-dataset classification')
parser.add_argument('--data_name',    type=str,   default='stocks_wiki_llm_1nn',
                    choices=list(DATA_CONFIGS.keys()))
parser.add_argument('--ckpt_dir',     type=str,   default='./ckpt_transtab_cls')
parser.add_argument('--num_epoch',    type=int,   default=100)
parser.add_argument('--patience',     type=int,   default=20)
parser.add_argument('--device',       type=str,   default='cuda:0')
parser.add_argument('--num_runs',     type=int,   default=5)
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--save_results', type=str,   default=None)
args = parser.parse_args()

if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(
        RESULTS_DIR, f"transtab_single_{args.data_name}_{args.num_runs}runs_{ts}.json"
    )

cfg        = DATA_CONFIGS[args.data_name]
target_col = cfg["target"]
num_class  = cfg["num_class"]
col_types  = cfg["cols"]

print(f"Data    : {args.data_name}")
print(f"Device  : {args.device}")
print(f"Epochs  : {args.num_epoch}  Patience: {args.patience}")
print(f"Runs    : {args.num_runs}")

# ── dataset config & data loading ────────────────────────────────────────────
task_config = transtab.create_dataset_config(
    col_types_dict=col_types,
    target_col=target_col,
    mask_path=cfg["mask"],
)

allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(
    [cfg["data_dir"]],
    dataset_config={cfg["data_dir"]: task_config},
    filename=cfg["csv"],
)
x_test, y_test = testset[0]
print(f"Train: {len(trainset[0][0])}, Val: {len(valset[0][0])}, Test: {len(x_test)}")

# ── multi-run ─────────────────────────────────────────────────────────────────
all_runs = []

for run_id in range(args.num_runs):
    seed = args.seed + run_id * 100000
    run_ckpt = osp.join(args.ckpt_dir, f"seed_{seed}")
    os.makedirs(run_ckpt, exist_ok=True)

    model = transtab.build_classifier(
        categorical_columns=cat_cols,
        numerical_columns=num_cols,
        binary_columns=bin_cols,
        num_class=num_class,
        num_layer=4,
        device=args.device,
    )

    transtab.train(
        model, trainset, valset,
        num_epoch=args.num_epoch,
        eval_metric='val_loss',
        eval_less_is_better=True,
        output_dir=run_ckpt,
    )

    model.load(run_ckpt)
    ypred_prob = transtab.predict(model, x_test, y_test)
    preds      = np.argmax(ypred_prob, axis=1)
    test_acc   = accuracy_score(y_test, preds)

    all_runs.append({"run_id": run_id + 1, "seed": seed, "test_acc": test_acc})
    print(f"  Run {run_id+1}/{args.num_runs}: test_acc={test_acc:.4f}  seed={seed}")

    # clean up checkpoint to save disk
    shutil.rmtree(run_ckpt, ignore_errors=True)

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
    "model":           "transtab_single",
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
