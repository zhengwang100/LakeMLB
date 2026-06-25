"""
Tree-based models (XGBoost, CatBoost, LightGBM) for regression.

Task    : price regression on GACarsDataset (price_in_euro, continuous)
Tables  : 4=German Reg, 5=Australian Reg, 6=DA Reg, 7=FA Reg
Metrics : RMSE, MAE, R²  (mean ± std across runs)

Available models: xgboost, catboost, lightgbm
"""

import sys
import os
import os.path as osp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', "lib"))

import argparse
import itertools
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

from lib.rllm.transforms.table_transforms import DefaultTableTransform
from lib.rllm.datasets.lakemlb.gacars import GACarsDataset
from utils import set_seed, get_device


AVAILABLE_MODELS = ["xgboost", "catboost", "lightgbm"]
_DATA_CACHE: Dict = {}

SCRIPT_DIR  = osp.dirname(osp.abspath(__file__))
DATA_DIR    = osp.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = osp.join(SCRIPT_DIR, "..", "results", "tree_models_reg")
os.makedirs(RESULTS_DIR, exist_ok=True)

TABLE_NAMES = {
    4: "german_reg",
    5: "australian_reg",
    6: "gacars_da_reg",
    7: "gacars_fa_reg",
}


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tree models for GACars regression task"
    )
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=AVAILABLE_MODELS)
    parser.add_argument("--table_idx", type=int, default=4,
                        choices=[4, 5, 6, 7],
                        help="GACarsDataset table index: 4=German Reg, "
                             "5=Australian Reg, 6=DA Reg, 7=FA Reg")

    # Experiment settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--save_results", type=str, default=None)
    parser.add_argument("--force_reload", action="store_true", default=False)

    # XGBoost hyperparameters
    parser.add_argument("--xgb_n_estimators", type=int, default=500)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_lr", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample", type=float, default=0.9)

    # CatBoost hyperparameters
    parser.add_argument("--cat_iterations", type=int, default=500)
    parser.add_argument("--cat_depth", type=int, default=6)
    parser.add_argument("--cat_lr", type=float, default=0.05)

    # LightGBM hyperparameters
    parser.add_argument("--lgb_num_boost_round", type=int, default=500)
    parser.add_argument("--lgb_num_leaves", type=int, default=64)
    parser.add_argument("--lgb_lr", type=float, default=0.03)
    parser.add_argument("--lgb_feature_fraction", type=float, default=0.9)
    parser.add_argument("--lgb_bagging_fraction", type=float, default=0.9)

    # Grid search
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--grid_patience", type=int, default=50)

    return parser.parse_args()


# ── data loading ──────────────────────────────────────────────────────────────
def load_data(
    table_idx: int = 4,
    device: torch.device = None,
    emb_dim: int = 32,
    force_reload: bool = False,
) -> Tuple:
    global _DATA_CACHE
    cache_key = f"gacars_{table_idx}_{emb_dim}"
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    if device is None:
        device = torch.device("cpu")

    table_transform = DefaultTableTransform(out_dim=emb_dim)
    dataset = GACarsDataset(
        cached_dir=DATA_DIR,
        force_reload=force_reload,
        transform=table_transform,
        device=device,
    )

    data = dataset.data_list[table_idx]
    # Regression: keep labels as float
    data.y = data.y.float().to(device)

    train_mask = data.train_mask.cpu().numpy()
    val_mask   = data.val_mask.cpu().numpy()
    test_mask  = data.test_mask.cpu().numpy()

    feat_dict = data.get_feat_dict()
    feat_list = []
    for key in sorted(feat_dict.keys()):
        feat_tensor = feat_dict[key]
        if feat_tensor.dim() == 1:
            feat_tensor = feat_tensor.unsqueeze(1)
        feat_list.append(feat_tensor.cpu().numpy())

    X = np.concatenate(feat_list, axis=1)
    y = data.y.cpu().numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    result = (X_train, y_train, X_val, y_val, X_test, y_test)
    _DATA_CACHE[cache_key] = result

    print(f"Dataset loaded: train={len(y_train)}, val={len(y_val)}, "
          f"test={len(y_test)}, dim={X_train.shape[1]}")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}]  "
          f"mean={y.mean():.4f}  std={y.std():.4f}")

    return result


# ── model training ────────────────────────────────────────────────────────────
def train_xgboost(
    X_train, y_train, X_val, y_val,
    config: Dict, seed: int, verbose: bool = False
) -> Tuple[XGBRegressor, Dict]:
    model = XGBRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["lr"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample"],
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=seed,
        early_stopping_rounds=config.get("early_stopping_rounds", 50),
        verbosity=1 if verbose else 0,
    )
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
    return model, {
        "training_time": time.time() - t0,
        "best_iteration": getattr(model, "best_iteration", config["n_estimators"]),
    }


def train_catboost(
    X_train, y_train, X_val, y_val,
    config: Dict, seed: int, verbose: bool = False
) -> Tuple[CatBoostRegressor, Dict]:
    model = CatBoostRegressor(
        iterations=config["iterations"],
        depth=config["depth"],
        learning_rate=config["lr"],
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=verbose,
        early_stopping_rounds=config.get("early_stopping_rounds", 50),
    )
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=verbose)
    return model, {
        "training_time": time.time() - t0,
        "best_iteration": model.get_best_iteration(),
    }


def train_lightgbm(
    X_train, y_train, X_val, y_val,
    config: Dict, seed: int, verbose: bool = False
) -> Tuple[lgb.Booster, Dict]:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": config["lr"],
        "num_leaves": config["num_leaves"],
        "max_depth": -1,
        "feature_fraction": config["feature_fraction"],
        "bagging_fraction": config["bagging_fraction"],
        "bagging_freq": 5,
        "seed": seed,
        "verbosity": 1 if verbose else -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set   = lgb.Dataset(X_val,   label=y_val, reference=train_set)
    t0 = time.time()
    gbm = lgb.train(
        params,
        train_set,
        num_boost_round=config["num_boost_round"],
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.get("early_stopping_rounds", 50)),
            lgb.log_evaluation(period=50 if verbose else 0),
        ],
    )
    return gbm, {
        "training_time": time.time() - t0,
        "best_iteration": gbm.best_iteration,
    }


# ── evaluation ────────────────────────────────────────────────────────────────
def compute_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_model(
    model, model_type: str,
    X_train, y_train, X_val, y_val, X_test, y_test
) -> Dict:
    if model_type == "lightgbm":
        train_pred = model.predict(X_train)
        val_pred   = model.predict(X_val)
        test_pred  = model.predict(X_test)
    else:
        train_pred = model.predict(X_train)
        val_pred   = model.predict(X_val)
        test_pred  = model.predict(X_test)

    results = {}
    for split, y_true, y_pred in [
        ("train", y_train, train_pred),
        ("val",   y_val,   val_pred),
        ("test",  y_test,  test_pred),
    ]:
        m = compute_reg_metrics(y_true, y_pred)
        for k, v in m.items():
            results[f"{split}_{k}"] = v
    return results


# ── single / multi-run experiment ─────────────────────────────────────────────
def run_single_experiment(
    model_type: str, config: Dict,
    X_train, y_train, X_val, y_val, X_test, y_test,
    seed: int, verbose: bool = False
) -> Dict:
    set_seed(seed)

    if model_type == "xgboost":
        model, train_metrics = train_xgboost(
            X_train, y_train, X_val, y_val, config, seed, verbose)
    elif model_type == "catboost":
        model, train_metrics = train_catboost(
            X_train, y_train, X_val, y_val, config, seed, verbose)
    elif model_type == "lightgbm":
        model, train_metrics = train_lightgbm(
            X_train, y_train, X_val, y_val, config, seed, verbose)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    eval_results = evaluate_model(
        model, model_type,
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    return {**eval_results, **train_metrics, "_model": model}


def run_multiple_experiments(
    model_type: str, config: Dict,
    data: Tuple, num_runs: int, base_seed: int,
    verbose: bool = False
) -> List[Dict]:
    X_train, y_train, X_val, y_val, X_test, y_test = data
    results = []
    print(f"\nRunning {num_runs} experiments …")
    for run_id in range(num_runs):
        seed = base_seed if run_id == 0 else base_seed + random.randint(1, 10000)
        r = run_single_experiment(
            model_type, config,
            X_train, y_train, X_val, y_val, X_test, y_test,
            seed, verbose=False
        )
        model = r.pop("_model")
        run_result = {"run_id": run_id + 1, "seed": seed, **r}
        results.append(run_result)
        print(
            f"  Run {run_id+1}/{num_runs}: "
            f"val RMSE={r['val_rmse']:.4f}  MAE={r['val_mae']:.4f}  R²={r['val_r2']:.4f} | "
            f"test RMSE={r['test_rmse']:.4f}  MAE={r['test_mae']:.4f}  R²={r['test_r2']:.4f} | "
            f"time={r['training_time']:.1f}s"
        )
    return results


# ── statistics & reporting ────────────────────────────────────────────────────
def compute_statistics(results: List[Dict]) -> Dict:
    metrics = [
        "train_rmse", "train_mae", "train_r2",
        "val_rmse",   "val_mae",   "val_r2",
        "test_rmse",  "test_mae",  "test_r2",
        "training_time",
    ]
    stats = {}
    for m in metrics:
        vals = [r[m] for r in results]
        stats[f"{m}_mean"] = float(np.mean(vals))
        stats[f"{m}_std"]  = float(np.std(vals))
        stats[f"{m}_min"]  = float(np.min(vals))
        stats[f"{m}_max"]  = float(np.max(vals))
    return stats


def print_statistics(stats: Dict, num_runs: int):
    def fmt(m): return f"{stats[f'{m}_mean']:.4f}±{stats[f'{m}_std']:.4f}"
    print(f"\n{'='*60}")
    print(f"Summary ({num_runs} runs)  [mean ± std]")
    print(f"{'='*60}")
    print(f"  Train  RMSE={fmt('train_rmse')}  MAE={fmt('train_mae')}  R²={fmt('train_r2')}")
    print(f"  Val    RMSE={fmt('val_rmse')}  MAE={fmt('val_mae')}  R²={fmt('val_r2')}")
    print(f"  Test   RMSE={fmt('test_rmse')}  MAE={fmt('test_mae')}  R²={fmt('test_r2')}")
    print(f"  Time   {fmt('training_time')}s")
    print(f"{'='*60}")


def save_results_to_file(
    results: List[Dict], stats: Dict, config: Dict,
    model_name: str, table_name: str, save_path: str
):
    output = {
        "model": model_name,
        "task": "price_regression",
        "dataset": f"GACarsDataset / {table_name}",
        "config": config,
        "num_runs": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "individual_runs": results,
        "statistics": stats,
    }
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {save_path}")


# ── grid search ───────────────────────────────────────────────────────────────
def get_grid_search_space(model_type: str) -> Dict:
    if model_type == "xgboost":
        return {
            "n_estimators": [300, 500, 1000],
            "max_depth": [4, 6, 8],
            "lr": [0.01, 0.03, 0.05],
            "subsample": [0.8, 0.9],
            "colsample": [0.8, 0.9],
        }
    elif model_type == "catboost":
        return {
            "iterations": [300, 500, 1000],
            "depth": [4, 6, 8],
            "lr": [0.01, 0.03, 0.05],
        }
    elif model_type == "lightgbm":
        return {
            "num_boost_round": [300, 500, 1000],
            "num_leaves": [31, 64, 127],
            "lr": [0.01, 0.03, 0.05],
            "feature_fraction": [0.8, 0.9],
            "bagging_fraction": [0.8, 0.9],
        }
    else:
        raise ValueError(f"Unknown model: {model_type}")


def run_grid_search(
    model_type: str, data: Tuple, base_seed: int,
    early_stopping_rounds: int = 50
) -> Tuple[Dict, List[Dict]]:
    X_train, y_train, X_val, y_val, X_test, y_test = data
    space = get_grid_search_space(model_type)
    combos = list(itertools.product(*space.values()))
    print(f"\nGrid search: {model_type.upper()}, {len(combos)} combinations")

    best_val_rmse = float("inf")
    best_config = None
    all_results = []

    for idx, combo in enumerate(combos):
        config = dict(zip(space.keys(), combo))
        config["early_stopping_rounds"] = early_stopping_rounds
        r = run_single_experiment(
            model_type, config,
            X_train, y_train, X_val, y_val, X_test, y_test,
            base_seed, verbose=False
        )
        r.pop("_model", None)
        record = {"config": config, **{k: r[k] for k in
                  ["val_rmse", "val_mae", "val_r2",
                   "test_rmse", "test_mae", "test_r2"]}}
        all_results.append(record)

        print(f"  [{idx+1}/{len(combos)}] val RMSE={r['val_rmse']:.4f}  "
              f"test RMSE={r['test_rmse']:.4f}")

        if r["val_rmse"] < best_val_rmse:
            best_val_rmse = r["val_rmse"]
            best_config = config.copy()

    print(f"\nBest config: {best_config}  val RMSE={best_val_rmse:.4f}")
    return best_config, all_results


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    table_name = TABLE_NAMES.get(args.table_idx, f"table_{args.table_idx}")
    print(f"\nModel : {args.model.upper()}")
    print(f"Table : {table_name}  (index={args.table_idx})")

    data = load_data(
        table_idx=args.table_idx,
        device=device,
        emb_dim=32,
        force_reload=args.force_reload,
    )

    if args.save_results is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_results = osp.join(
            RESULTS_DIR,
            f"{args.model}_{table_name}_{args.num_runs}runs_{ts}.json"
        )

    # Build default config from args
    if args.model == "xgboost":
        config = {
            "n_estimators": args.xgb_n_estimators,
            "max_depth": args.xgb_max_depth,
            "lr": args.xgb_lr,
            "subsample": args.xgb_subsample,
            "colsample": args.xgb_colsample,
            "early_stopping_rounds": 50,
        }
    elif args.model == "catboost":
        config = {
            "iterations": args.cat_iterations,
            "depth": args.cat_depth,
            "lr": args.cat_lr,
            "early_stopping_rounds": 50,
        }
    elif args.model == "lightgbm":
        config = {
            "num_boost_round": args.lgb_num_boost_round,
            "num_leaves": args.lgb_num_leaves,
            "lr": args.lgb_lr,
            "feature_fraction": args.lgb_feature_fraction,
            "bagging_fraction": args.lgb_bagging_fraction,
            "early_stopping_rounds": 50,
        }

    if args.grid:
        best_config, grid_results = run_grid_search(
            args.model, data, args.seed, args.grid_patience
        )
        grid_save = osp.join(
            RESULTS_DIR,
            f"{args.model}_{table_name}_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(grid_save, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "table": table_name,
                       "grid_results": grid_results, "best_config": best_config},
                      f, indent=2)
        print(f"Grid search saved → {grid_save}")
        config = best_config

    print(f"\nConfig: {config}")

    results = run_multiple_experiments(
        model_type=args.model,
        config=config,
        data=data,
        num_runs=args.num_runs,
        base_seed=args.seed,
        verbose=False,
    )

    stats = compute_statistics(results)
    print_statistics(stats, args.num_runs)
    save_results_to_file(results, stats, config, args.model, table_name, args.save_results)

    # Re-train best run and show per-split summary
    best_run = min(results, key=lambda x: x["val_rmse"])
    print(f"\nBest run: #{best_run['run_id']}  (seed={best_run['seed']}, "
          f"val RMSE={best_run['val_rmse']:.4f})")
    print(f"  → test RMSE={best_run['test_rmse']:.4f}  "
          f"MAE={best_run['test_mae']:.4f}  R²={best_run['test_r2']:.4f}")


if __name__ == "__main__":
    main()
