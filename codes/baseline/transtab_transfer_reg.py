"""
TransTab transfer-learning regression on GACars.

Stage 1 – Pretrain  : Australian Reg (price in AUD, target = 'Price')
Stage 2 – Fine-tune : German Reg     (price in EUR, target = 'price_in_euro')

Both tables are labelled → supervised transfer (identical structure to
transtab_transfer.py but uses build_regressor).

Metrics: RMSE, MAE, R²  (mean ± std across runs, evaluated on German test set)

Key fix vs. transtab.load_data():
  - transtab lowercases CSV columns AND applies LabelEncoder to y.
    For regression (continuous y) LabelEncoder converts prices to integer
    indices, producing completely wrong training targets. We use a custom
    loader that keeps raw float y values and lowercases column names to
    match transtab's internal processing.
"""
import sys
import os
import os.path as osp
import argparse
import json
import random
from datetime import datetime

sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', "lib"))

import numpy as np
import pandas as pd
import torch
import transtab
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rllm.types import ColType

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = osp.dirname(osp.realpath(__file__))
# transtab loads its tokenizer from './transtab/tokenizer' (relative path);
# ensure cwd is the baseline dir so that path resolves correctly.
os.chdir(SCRIPT_DIR)
RAW_DIR     = osp.abspath(osp.join(SCRIPT_DIR, '..', 'data', 'table_gacars', 'raw'))
RESULTS_DIR = osp.abspath(osp.join(SCRIPT_DIR, '..', 'results', 'transtab_reg'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='TransTab transfer regression (GACars)')
parser.add_argument('--ckpt_dir',            type=str,   default=None)
parser.add_argument('--pretrain_dir',        type=str,   default=None)
parser.add_argument('--num_epoch_pretrain',  type=int,   default=50)
parser.add_argument('--num_epoch_finetune',  type=int,   default=50)
parser.add_argument('--lr',                  type=float, default=1e-4)
parser.add_argument('--batch_size',          type=int,   default=64)
parser.add_argument('--patience',            type=int,   default=10)
parser.add_argument('--num_runs',            type=int,   default=5)
parser.add_argument('--seed',                type=int,   default=42)
parser.add_argument('--device',              type=str,   default='cuda:0')
parser.add_argument('--save_results',        type=str,   default=None)
args = parser.parse_args()

if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(
        RESULTS_DIR,
        f"transtab_transfer_german_reg_{args.num_runs}runs_{ts}.json",
    )

# Unique checkpoint dirs per parallel batch (avoids cross-process checkpoint clobbering)
if args.ckpt_dir is None:
    args.ckpt_dir = f"./ckpt_transfer_reg_{args.seed}"
if args.pretrain_dir is None:
    args.pretrain_dir = osp.join(args.ckpt_dir, "pretrained")

print(f"Device           : {args.device}")
print(f"Pretrain epochs  : {args.num_epoch_pretrain}  (Australian Reg → Price/AUD)")
print(f"Finetune epochs  : {args.num_epoch_finetune}  (German Reg    → price_in_euro/EUR)")
print(f"Num runs         : {args.num_runs}")

# ── column types (ALL LOWERCASE – transtab lowercases CSV columns internally) ─

# German Reg (task table)
task_col_types = {
    "id":                        ColType.NUMERICAL,
    "brand":                     ColType.CATEGORICAL,
    "model":                     ColType.CATEGORICAL,
    "color":                     ColType.CATEGORICAL,
    "registration_date":         ColType.CATEGORICAL,
    "year":                      ColType.NUMERICAL,
    "price_in_euro":             ColType.NUMERICAL,   # regression target
    "power_kw":                  ColType.NUMERICAL,
    "power_ps":                  ColType.NUMERICAL,
    "transmission_type":         ColType.CATEGORICAL,
    "fuel_type":                 ColType.CATEGORICAL,
    "fuel_consumption_l_100km":  ColType.CATEGORICAL,
    "fuel_consumption_g_km":     ColType.CATEGORICAL,
    "mileage_in_km":             ColType.NUMERICAL,
    "offer_description":         ColType.CATEGORICAL,
}
TASK_TARGET = "price_in_euro"

# Australian Reg (auxiliary table) – keys lowercase to match load_data lowercasing
aux_col_types = {
    "brand":             ColType.CATEGORICAL,
    "year":              ColType.NUMERICAL,
    "model":             ColType.CATEGORICAL,
    "car/suv":           ColType.CATEGORICAL,
    "title":             ColType.CATEGORICAL,
    "usedornew":         ColType.CATEGORICAL,
    "transmission":      ColType.CATEGORICAL,
    "engine":            ColType.CATEGORICAL,
    "drivetype":         ColType.CATEGORICAL,
    "fueltype":          ColType.CATEGORICAL,
    "fuelconsumption":   ColType.CATEGORICAL,
    "kilometres":        ColType.NUMERICAL,
    "colourextint":      ColType.CATEGORICAL,
    "location":          ColType.CATEGORICAL,
    "cylindersinengine": ColType.CATEGORICAL,
    "bodytype":          ColType.CATEGORICAL,
    "doors":             ColType.CATEGORICAL,
    "seats":             ColType.CATEGORICAL,
    "price":             ColType.NUMERICAL,   # regression target (AUD)
}
AUX_TARGET = "price"


# ── custom regression data loader ─────────────────────────────────────────────
def load_regression_data(csv_path, col_types_dict, target_col, mask_path):
    """
    Load regression data from CSV + mask file.

    transtab.load_data() applies LabelEncoder to y, converting continuous
    prices to integer indices – wrong for regression. This function keeps
    raw float y values and matches transtab's column pre-processing
    (lowercase, MinMaxScaler on num, str cast on cat).

    Returns
    -------
    trainset, valset, testset : tuple (X: pd.DataFrame, y: pd.Series)
    cat_cols, num_cols, bin_cols : list[str]  (all lowercase)
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]   # mirrors transtab

    target_lower = target_col.lower()
    y_all = df[target_lower].astype(float)
    X_all = df.drop(columns=[target_lower])

    cat_cols, num_cols, bin_cols = [], [], []
    for col, ctype in col_types_dict.items():
        col_l = col.lower()
        if col_l == target_lower or col_l not in X_all.columns:
            continue
        ts = str(ctype).lower()
        if 'binary' in ts or ('bin' in ts and 'numerical' not in ts):
            bin_cols.append(col_l)
        elif 'categorical' in ts or 'cat' in ts:
            cat_cols.append(col_l)
        else:
            num_cols.append(col_l)

    # Numerical pre-processing (mirrors transtab: fillna + MinMaxScaler)
    if num_cols:
        for c in num_cols:
            X_all[c] = pd.to_numeric(X_all[c], errors='coerce')
            m = X_all[c].mode()
            X_all[c] = X_all[c].fillna(m.iloc[0] if not m.empty else 0)
        X_all[num_cols] = MinMaxScaler().fit_transform(X_all[num_cols])

    # Categorical pre-processing
    for c in cat_cols:
        m = X_all[c].mode()
        X_all[c] = X_all[c].fillna(m.iloc[0] if not m.empty else "Unknown")
        X_all[c] = X_all[c].astype(str)

    X_all = X_all[bin_cols + num_cols + cat_cols]

    masks = torch.load(mask_path, weights_only=False)
    def split(mask):
        idx = mask.numpy() if hasattr(mask, 'numpy') else mask
        return X_all[idx].reset_index(drop=True), y_all[idx].reset_index(drop=True)

    return split(masks['train_mask']), split(masks['val_mask']), split(masks['test_mask']), \
           cat_cols, num_cols, bin_cols


# ── load datasets ─────────────────────────────────────────────────────────────
(train1, val1, test1, cat_cols1, num_cols1, bin_cols1) = load_regression_data(
    csv_path       = osp.join(RAW_DIR, 'australian_reg.csv'),
    col_types_dict = aux_col_types,
    target_col     = AUX_TARGET,
    mask_path      = osp.join(RAW_DIR, 'australian_mask_reg.pt'),
)
(train2, val2, test2, cat_cols2, num_cols2, bin_cols2) = load_regression_data(
    csv_path       = osp.join(RAW_DIR, 'german_reg.csv'),
    col_types_dict = task_col_types,
    target_col     = TASK_TARGET,
    mask_path      = osp.join(RAW_DIR, 'german_mask_reg.pt'),
)

x_test, y_test = test2
print(f"\nAux  : train={len(train1[0])}, val={len(val1[0])}, test={len(test1[0])}")
print(f"Task : train={len(train2[0])}, val={len(val2[0])}, test={len(x_test)}")
print(f"Aux  cat={len(cat_cols1)}, num={len(num_cols1)}")
print(f"Task cat={len(cat_cols2)}, num={len(num_cols2)}")
print(f"Task y range=[{y_test.min():.2f}, {y_test.max():.2f}]  "
      f"mean={y_test.mean():.2f}  std={y_test.std():.2f}")


# ── custom regression predict (avoids sigmoid in transtab.predict) ────────────
def predict_regression(model, x, eval_batch_size=256):
    model.eval()
    preds = []
    for i in range(0, len(x), eval_batch_size):
        bx = x.iloc[i:i + eval_batch_size]
        with torch.no_grad():
            output, _ = model(bx, None)
        preds.append(output.flatten().cpu().numpy())
    return np.concatenate(preds, 0)


def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ── multi-run transfer-learning experiment ────────────────────────────────────
random.seed(args.seed)
seeds = [args.seed] + [args.seed + random.randint(1, 10000)
                       for _ in range(args.num_runs - 1)]

all_results = []
print(f"\nRunning {args.num_runs} transfer-learning experiments …")

for run_id, seed in enumerate(seeds):
    print(f"\n[Run {run_id+1}/{args.num_runs}]  seed={seed}")
    print("-" * 40)
    transtab.random_seed(seed)

    ckpt     = osp.join(args.ckpt_dir,   f"run_{run_id+1}")
    pre_ckpt = osp.join(args.pretrain_dir, f"run_{run_id+1}")
    os.makedirs(ckpt,     exist_ok=True)
    os.makedirs(pre_ckpt, exist_ok=True)

    # ---- Stage 1: pretrain on Australian auxiliary table --------------------
    print("  Stage 1 – Pretrain on Australian Reg …")
    model = transtab.build_regressor(
        categorical_columns=cat_cols1,
        numerical_columns=num_cols1,
        binary_columns=bin_cols1,
        num_layer=4,
        device=args.device,
    )
    transtab.train(
        model, train1, val1,
        num_epoch=args.num_epoch_pretrain,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        eval_metric='val_loss',
        eval_less_is_better=True,
        output_dir=ckpt,
    )
    # transtab.train loads the best checkpoint at the end by default;
    # save it explicitly for reproducibility.
    model.save(pre_ckpt)
    print(f"  Pretrained model saved → {pre_ckpt}")

    # ---- Stage 2: fine-tune on German task table ----------------------------
    print("  Stage 2 – Fine-tune on German Reg …")
    model.load(pre_ckpt)

    # Extend feature extractor to recognise German column names.
    # feature_extractor.update() *extends* the column lists, so both
    # Australian (from pretraining) and German columns are known.
    model.update({'cat': cat_cols2, 'num': num_cols2, 'bin': bin_cols2})

    transtab.train(
        model, train2, val2,
        num_epoch=args.num_epoch_finetune,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        eval_metric='val_loss',
        eval_less_is_better=True,
        output_dir=ckpt,
    )
    model.load(ckpt)

    # ---- Evaluate on German test set ----------------------------------------
    y_pred = predict_regression(model, x_test)
    m = compute_metrics(y_test.to_numpy(), y_pred)

    run_result = {"run_id": run_id + 1, "seed": seed,
                  **{f"test_{k}": v for k, v in m.items()}}
    all_results.append(run_result)
    print(f"  Test RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R²={m['r2']:.4f}")


# ── statistics & save ─────────────────────────────────────────────────────────
def compute_statistics(results):
    stats = {}
    for m in ["test_rmse", "test_mae", "test_r2"]:
        vals = [r[m] for r in results]
        stats[f"{m}_mean"] = float(np.mean(vals))
        stats[f"{m}_std"]  = float(np.std(vals))
        stats[f"{m}_min"]  = float(np.min(vals))
        stats[f"{m}_max"]  = float(np.max(vals))
    return stats

stats = compute_statistics(all_results)

def fmt(s, m): return f"{s[f'{m}_mean']:.4f}±{s[f'{m}_std']:.4f}"
print(f"\n{'='*60}")
print(f"Summary ({args.num_runs} runs)  [mean ± std]")
print(f"{'='*60}")
print(f"  Test RMSE={fmt(stats,'test_rmse')}  "
      f"MAE={fmt(stats,'test_mae')}  R²={fmt(stats,'test_r2')}")
print(f"{'='*60}")

output = {
    "model":           "transtab_transfer",
    "task":            "price_regression",
    "dataset_task":    "GACarsDataset / german_reg",
    "dataset_aux":     "GACarsDataset / australian_reg",
    "num_runs":        args.num_runs,
    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "individual_runs": all_results,
    "statistics":      stats,
}
with open(args.save_results, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.save_results}")
