"""
TransTab single-table regression on GACars (German Reg).

Train and evaluate on the task table only (no transfer learning).
Task    : price_in_euro regression (continuous)
Metrics : RMSE, MAE, R²  (mean ± std across runs)

Key fix vs. transtab.load_data():
  - transtab internally lowercases all column names AND applies LabelEncoder to y.
    LabelEncoder converts continuous prices into integer indices, which is wrong
    for regression. We use a custom loader that keeps raw float y values.
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
RAW_DIR     = osp.abspath(osp.join(SCRIPT_DIR, '..', 'data', 'table_gacars', 'raw'))
RESULTS_DIR = osp.abspath(osp.join(SCRIPT_DIR, '..', 'results', 'transtab_reg'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='TransTab single-table regression (GACars)')
parser.add_argument('--data_name',    type=str,   default='german_reg',
                    choices=['german_reg', 'gacars_da_reg', 'gacars_fa_reg'])
parser.add_argument('--ckpt_dir',     type=str,   default=None,
                    help='Checkpoint dir; defaults to ./ckpt_single_reg_<seed> to avoid conflicts when running parallel batches')
parser.add_argument('--num_epoch',    type=int,   default=50)
parser.add_argument('--lr',           type=float, default=1e-4)
parser.add_argument('--batch_size',   type=int,   default=64)
parser.add_argument('--patience',     type=int,   default=10)
parser.add_argument('--num_runs',     type=int,   default=5)
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--device',       type=str,   default='cuda:0')
parser.add_argument('--save_results', type=str,   default=None)
args = parser.parse_args()

if args.ckpt_dir is None:
    args.ckpt_dir = f'./ckpt_single_reg_{args.seed}'

if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(
        RESULTS_DIR, f"transtab_single_{args.data_name}_{args.num_runs}runs_{ts}.json"
    )

print(f"Data        : {args.data_name}")
print(f"Device      : {args.device}")
print(f"Epochs      : {args.num_epoch}  Patience: {args.patience}")
print(f"Num runs    : {args.num_runs}")

# ── per-dataset config ─────────────────────────────────────────────────────────
_GERMAN_COLS = {
    "id":                        ColType.NUMERICAL,
    "brand":                     ColType.CATEGORICAL,
    "model":                     ColType.CATEGORICAL,
    "color":                     ColType.CATEGORICAL,
    "registration_date":         ColType.CATEGORICAL,
    "year":                      ColType.NUMERICAL,
    "power_kw":                  ColType.NUMERICAL,
    "power_ps":                  ColType.NUMERICAL,
    "transmission_type":         ColType.CATEGORICAL,
    "fuel_type":                 ColType.CATEGORICAL,
    "fuel_consumption_l_100km":  ColType.CATEGORICAL,
    "fuel_consumption_g_km":     ColType.CATEGORICAL,
    "mileage_in_km":             ColType.NUMERICAL,
    "offer_description":         ColType.CATEGORICAL,
}
_DA_AUX_COLS = {
    "car/suv":              ColType.CATEGORICAL,
    "title":                ColType.CATEGORICAL,
    "usedornew":            ColType.CATEGORICAL,
    "transmission":         ColType.CATEGORICAL,
    "engine":               ColType.CATEGORICAL,
    "drivetype":            ColType.CATEGORICAL,
    "fuelconsumption":      ColType.CATEGORICAL,
    "kilometres":           ColType.NUMERICAL,
    "colourextint":         ColType.CATEGORICAL,
    "location":             ColType.CATEGORICAL,
    "cylindersinengine":    ColType.CATEGORICAL,
    "bodytype":             ColType.CATEGORICAL,
    "doors":                ColType.CATEGORICAL,
    "seats":                ColType.CATEGORICAL,
}
_FA_AUX_COLS = {
    **_DA_AUX_COLS,
    # Australian Brand/Year/Model lowercase to brand/year/model, clashing with
    # German columns → deduplicated to brand_aux / year_aux / model_aux
    "brand_aux":    ColType.CATEGORICAL,
    "year_aux":     ColType.NUMERICAL,
    "model_aux":    ColType.CATEGORICAL,
    "fueltype":     ColType.CATEGORICAL,
    "price":        ColType.NUMERICAL,     # Australian Price (feature, not target)
}

DATA_CONFIGS = {
    "german_reg": {
        "csv":    osp.join(RAW_DIR, "german_reg.csv"),
        "mask":   osp.join(RAW_DIR, "german_mask_reg.pt"),
        "target": "price_in_euro",
        "cols":   {**_GERMAN_COLS, "price_in_euro": ColType.NUMERICAL},
    },
    "gacars_da_reg": {
        "csv":    osp.join(RAW_DIR, "gacars_da_reg.csv"),
        "mask":   osp.join(RAW_DIR, "mask_da_reg.pt"),
        "target": "price",
        "cols":   {**_GERMAN_COLS, "price": ColType.NUMERICAL, **_DA_AUX_COLS},
    },
    "gacars_fa_reg": {
        "csv":    osp.join(RAW_DIR, "gacars_fa_reg.csv"),
        "mask":   osp.join(RAW_DIR, "german_mask_reg.pt"),
        "target": "price_in_euro",
        "cols":   {**_GERMAN_COLS, "price_in_euro": ColType.NUMERICAL, **_FA_AUX_COLS},
    },
}

cfg = DATA_CONFIGS[args.data_name]
task_col_types = cfg["cols"]
TASK_TARGET    = cfg["target"]


# ── custom regression data loader ─────────────────────────────────────────────
def load_regression_data(csv_path, col_types_dict, target_col, mask_path):
    """
    Load regression data from CSV + mask file.

    transtab.load_data() applies LabelEncoder to y, converting continuous
    prices to integer indices – wrong for regression. This function keeps
    raw float y values and matches transtab's own column pre-processing
    (lowercase columns, MinMaxScaler on num, str cast on cat).

    Returns
    -------
    trainset, valset, testset : tuple (X: pd.DataFrame, y: pd.Series)
    cat_cols, num_cols, bin_cols : list[str]  (all lowercase)
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]          # mirrors transtab

    # Deduplicate column names that clash after lowercasing (e.g. FA table has
    # both German "brand" and Australian "Brand" → both become "brand").
    # Keep the first occurrence unchanged; rename duplicates with "_aux" suffix.
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_aux{seen[c] - 1 if seen[c] > 2 else ''}")
        else:
            seen[c] = 1
            new_cols.append(c)
    df.columns = new_cols

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
        X_s = X_all[idx].reset_index(drop=True)
        y_s = y_all[idx].reset_index(drop=True)
        return X_s, y_s

    trainset = split(masks['train_mask'])
    valset   = split(masks['val_mask'])
    testset  = split(masks['test_mask'])

    return trainset, valset, testset, cat_cols, num_cols, bin_cols


# ── load data ─────────────────────────────────────────────────────────────────
trainset, valset, testset, cat_cols, num_cols, bin_cols = load_regression_data(
    csv_path       = cfg["csv"],
    col_types_dict = task_col_types,
    target_col     = TASK_TARGET,
    mask_path      = cfg["mask"],
)
x_test, y_test = testset
print(f"Train={len(trainset[0])}, Val={len(valset[0])}, Test={len(x_test)}")
print(f"cat={len(cat_cols)}, num={len(num_cols)}, bin={len(bin_cols)}")
print(f"y range=[{y_test.min():.2f}, {y_test.max():.2f}]  "
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


# ── multi-run experiment ──────────────────────────────────────────────────────
random.seed(args.seed)
seeds = [args.seed] + [args.seed + random.randint(1, 10000)
                       for _ in range(args.num_runs - 1)]

all_results = []
print(f"\nRunning {args.num_runs} experiments …")

for run_id, seed in enumerate(seeds):
    print(f"\n[Run {run_id+1}/{args.num_runs}]  seed={seed}")
    print("-" * 40)
    transtab.random_seed(seed)

    ckpt = osp.join(args.ckpt_dir, f"run_{run_id+1}")
    os.makedirs(ckpt, exist_ok=True)

    model = transtab.build_regressor(
        categorical_columns=cat_cols,
        numerical_columns=num_cols,
        binary_columns=bin_cols,
        num_layer=4,
        device=args.device,
    )

    transtab.train(
        model, trainset, valset,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        eval_metric='val_loss',
        eval_less_is_better=True,
        output_dir=ckpt,
    )

    model.load(ckpt)
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
    "model":     "transtab_single",
    "task":      "price_regression",
    "dataset":   f"GACarsDataset / {args.data_name}",
    "num_runs":  args.num_runs,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "individual_runs": all_results,
    "statistics": stats,
}
with open(args.save_results, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.save_results}")
