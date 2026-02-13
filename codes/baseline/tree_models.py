"""
Tree-based models (XGBoost, CatBoost, LightGBM) for classification.

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
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

from lib.rllm.transforms.table_transforms import DefaultTableTransform
from lib.rllm.datasets import MSTrafficDataset
from utils import set_seed, get_device


AVAILABLE_MODELS = ["xgboost", "catboost", "lightgbm"]
_DATA_CACHE = {}


SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = osp.join(SCRIPT_DIR, "..", "results", "tree_models")
os.makedirs(RESULTS_DIR, exist_ok=True)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=AVAILABLE_MODELS,
        help="Tree model to use"
    )
    
    # Experiment settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--save_results", type=str, default=None)
    parser.add_argument("--force_reload", action="store_true", default=True)
    parser.add_argument("--xgb_n_estimators", type=int, default=500)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_lr", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample", type=float, default=0.9)
    parser.add_argument("--cat_iterations", type=int, default=500)
    parser.add_argument("--cat_depth", type=int, default=6)
    parser.add_argument("--cat_lr", type=float, default=0.05)
    parser.add_argument("--lgb_num_boost_round", type=int, default=500)
    parser.add_argument("--lgb_num_leaves", type=int, default=64)
    parser.add_argument("--lgb_lr", type=float, default=0.03)
    parser.add_argument("--lgb_feature_fraction", type=float, default=0.9)
    parser.add_argument("--lgb_bagging_fraction", type=float, default=0.9)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--grid_patience", type=int, default=50)
    
    return parser.parse_args()


def load_data(
    dataset_class,
    dataset_name: str = "nnstocks",
    device: torch.device = None,
    emb_dim: int = 32,
    force_reload: bool = False,
    **dataset_kwargs
) -> Tuple:
    global _DATA_CACHE
    
    cache_key = f"{dataset_name}_{emb_dim}"
    
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]
    
    if device is None:
        device = torch.device('cpu')
    
    table_transform = DefaultTableTransform(out_dim=emb_dim)
    if 'cached_dir' not in dataset_kwargs:
        dataset_kwargs['cached_dir'] = DATA_DIR
    
    dataset = dataset_class(
        force_reload=force_reload,
        transform=table_transform,
        device=device,
        **dataset_kwargs
    )
    
    data = dataset.data_list[0]
    data.y = data.y.long().to(device)
    
    if not (hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask')):
        raise ValueError(f"Dataset {dataset_name} must have train_mask, val_mask, test_mask")
    
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    feat_dict = data.get_feat_dict()
    feat_list = []
    for key in sorted(feat_dict.keys()):
        feat_tensor = feat_dict[key]
        if feat_tensor.dim() == 1:
            feat_tensor = feat_tensor.unsqueeze(1)
        feat_list.append(feat_tensor.cpu().numpy())
    
    X = np.concatenate(feat_list, axis=1)
    y = data.y.cpu().numpy()
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    num_classes = len(np.unique(y))
    result = (X_train, y_train, X_val, y_val, X_test, y_test, num_classes)
    _DATA_CACHE[cache_key] = result
    
    print(f"Dataset loaded: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, classes={num_classes}, dim={X_train.shape[1]}")
    
    return result


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: Dict, seed: int, verbose: bool = False
) -> Tuple[XGBClassifier, Dict]:
    
    num_classes = len(np.unique(y_train))
    
    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["lr"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample"],
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
        tree_method="hist",
        random_state=seed,
        early_stopping_rounds=config.get("early_stopping_rounds", 50),
        verbosity=1 if verbose else 0
    )
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )
    training_time = time.time() - start_time
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else config["n_estimators"]
    
    metrics = {
        "training_time": training_time,
        "best_iteration": best_iteration
    }
    
    return model, metrics


def train_catboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: Dict, seed: int, verbose: bool = False
) -> Tuple[CatBoostClassifier, Dict]:
    
    num_classes = len(np.unique(y_train))
    
    model = CatBoostClassifier(
        iterations=config["iterations"],
        depth=config["depth"],
        learning_rate=config["lr"],
        loss_function="MultiClass" if num_classes > 2 else "Logloss",
        random_seed=seed,
        verbose=verbose,
        early_stopping_rounds=config.get("early_stopping_rounds", 50)
    )
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=verbose
    )
    training_time = time.time() - start_time
    best_iteration = model.get_best_iteration()
    
    metrics = {
        "training_time": training_time,
        "best_iteration": best_iteration
    }
    
    return model, metrics


def train_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: Dict, seed: int, verbose: bool = False
) -> Tuple[lgb.Booster, Dict]:
    
    num_classes = len(np.unique(y_train))
    is_binary = (num_classes == 2)
    
    params = {
        "objective": "binary" if is_binary else "multiclass",
        "metric": "binary_logloss" if is_binary else "multi_logloss",
        "learning_rate": config["lr"],
        "num_leaves": config["num_leaves"],
        "max_depth": -1,
        "feature_fraction": config["feature_fraction"],
        "bagging_fraction": config["bagging_fraction"],
        "bagging_freq": 5,
        "seed": seed,
        "verbosity": 1 if verbose else -1
    }
    
    if not is_binary:
        params["num_class"] = num_classes
    
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    start_time = time.time()
    gbm = lgb.train(
        params,
        train_set,
        num_boost_round=config["num_boost_round"],
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.get("early_stopping_rounds", 50)),
            lgb.log_evaluation(period=50 if verbose else 0)
        ]
    )
    training_time = time.time() - start_time
    
    metrics = {
        "training_time": training_time,
        "best_iteration": gbm.best_iteration
    }
    
    return gbm, metrics


def predict_labels(model, X: np.ndarray, model_type: str) -> np.ndarray:
    if model_type == "lightgbm":
        pred = model.predict(X)
        if pred.ndim == 1:
            return (pred > 0.5).astype(int)
        else:
            return np.argmax(pred, axis=1)
    else:
        return model.predict(X)


def evaluate_model(
    model, X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    model_type: str
) -> Dict:
    
    train_pred = predict_labels(model, X_train, model_type)
    val_pred = predict_labels(model, X_val, model_type)
    test_pred = predict_labels(model, X_test, model_type)
    
    results = {
        "train_acc": accuracy_score(y_train, train_pred),
        "train_f1": f1_score(y_train, train_pred, average='macro'),
        "val_acc": accuracy_score(y_val, val_pred),
        "val_f1": f1_score(y_val, val_pred, average='macro'),
        "test_acc": accuracy_score(y_test, test_pred),
        "test_f1": f1_score(y_test, test_pred, average='macro'),
    }
    
    return results


def run_single_experiment(
    model_type: str,
    config: Dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    seed: int,
    verbose: bool = False
) -> Dict:
    set_seed(seed)
    
    if model_type == "xgboost":
        model, train_metrics = train_xgboost(
            X_train, y_train, X_val, y_val, config, seed, verbose
        )
    elif model_type == "catboost":
        model, train_metrics = train_catboost(
            X_train, y_train, X_val, y_val, config, seed, verbose
        )
    elif model_type == "lightgbm":
        model, train_metrics = train_lightgbm(
            X_train, y_train, X_val, y_val, config, seed, verbose
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    eval_results = evaluate_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test, model_type
    )
    results = {**eval_results, **train_metrics}
    return results


def run_multiple_experiments(
    model_type: str,
    config: Dict,
    data: Tuple,
    num_runs: int,
    base_seed: int,
    verbose: bool = False
) -> List[Dict]:
    X_train, y_train, X_val, y_val, X_test, y_test, _ = data
    results = []
    
    print(f"\nRunning {num_runs} experiments...")
    
    for run_id in range(num_runs):
        seed = base_seed if run_id == 0 else base_seed + random.randint(1, 10000)
        
        result = run_single_experiment(
            model_type, config, X_train, y_train, X_val, y_val,
            X_test, y_test, seed, verbose=False
        )
        
        run_result = {"run_id": run_id + 1, "seed": seed, **result}
        results.append(run_result)
        
        print(f"Run {run_id+1}/{num_runs}: train={result['train_acc']:.4f}, val={result['val_acc']:.4f}, test={result['test_acc']:.4f}, time={result['training_time']:.2f}s")
    
    return results


def compute_statistics(results: List[Dict]) -> Dict:
    
    metrics = ["train_acc", "train_f1", "val_acc", "val_f1", "test_acc", "test_f1", "training_time"]
    stats = {}
    
    for metric in metrics:
        values = [r[metric] for r in results]
        stats[f"{metric}_mean"] = float(np.mean(values))
        stats[f"{metric}_std"] = float(np.std(values))
        stats[f"{metric}_min"] = float(np.min(values))
        stats[f"{metric}_max"] = float(np.max(values))
    
    return stats


def print_statistics(stats: Dict, num_runs: int):
    print(f"\nSummary ({num_runs} runs):")
    print(f"Train: {stats['train_acc_mean']:.4f}±{stats['train_acc_std']:.4f} (F1: {stats['train_f1_mean']:.4f}±{stats['train_f1_std']:.4f})")
    print(f"Val:   {stats['val_acc_mean']:.4f}±{stats['val_acc_std']:.4f} (F1: {stats['val_f1_mean']:.4f}±{stats['val_f1_std']:.4f})")
    print(f"Test:  {stats['test_acc_mean']:.4f}±{stats['test_acc_std']:.4f} (F1: {stats['test_f1_mean']:.4f}±{stats['test_f1_std']:.4f})")
    print(f"Time:  {stats['training_time_mean']:.2f}±{stats['training_time_std']:.2f}s")


def save_results_to_file(
    results: List[Dict],
    stats: Dict,
    config: Dict,
    model_name: str,
    save_path: str
):
    
    output = {
        "model": model_name,
        "task": "stock_sector_classification",
        "dataset": "NNStocks",
        "config": config,
        "num_runs": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "individual_runs": results,
        "statistics": stats,
    }
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {save_path}")


def get_grid_search_space(model_type: str) -> Dict:
    
    if model_type == "xgboost":
        return {
            "n_estimators": [300, 500, 1000],
            "max_depth": [4, 6, 8],
            "lr": [0.01, 0.03, 0.05],
            "subsample": [0.8, 0.9],
            "colsample": [0.8, 0.9]
        }
    elif model_type == "catboost":
        return {
            "iterations": [300, 500, 1000],
            "depth": [4, 6, 8],
            "lr": [0.01, 0.03, 0.05]
        }
    elif model_type == "lightgbm":
        return {
            "num_boost_round": [300, 500, 1000],
            "num_leaves": [31, 64, 127],
            "lr": [0.01, 0.03, 0.05],
            "feature_fraction": [0.8, 0.9],
            "bagging_fraction": [0.8, 0.9]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_grid_search(
    model_type: str,
    data: Tuple,
    base_seed: int,
    early_stopping_rounds: int = 50
) -> Tuple[Dict, List[Dict]]:
    X_train, y_train, X_val, y_val, X_test, y_test, _ = data
    
    space = get_grid_search_space(model_type)
    all_combinations = list(itertools.product(*space.values()))
    
    print(f"\nGrid search: {model_type.upper()}, {len(all_combinations)} combinations")
    
    best_val_acc = -1.0
    best_config = None
    all_results = []
    
    for idx, combination in enumerate(all_combinations):
        config = dict(zip(space.keys(), combination))
        config["early_stopping_rounds"] = early_stopping_rounds
        
        result = run_single_experiment(
            model_type, config, X_train, y_train, X_val, y_val,
            X_test, y_test, base_seed, verbose=False
        )
        
        result_record = {
            "config": config,
            "val_acc": result["val_acc"],
            "test_acc": result["test_acc"],
            "val_f1": result["val_f1"],
            "test_f1": result["test_f1"]
        }
        all_results.append(result_record)
        
        print(f"[{idx+1}/{len(all_combinations)}] val={result['val_acc']:.4f}, test={result['test_acc']:.4f}")
        
        if result["val_acc"] > best_val_acc:
            best_val_acc = result["val_acc"]
            best_config = config.copy()
    
    print(f"\nBest config: {best_config}, val_acc={best_val_acc:.4f}")
    return best_config, all_results


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    
    print(f"\nModel: {args.model.upper()}")
    
    data = load_data(
        dataset_class=MSTrafficDataset,
        dataset_name="mstraffic",
        device=device,
        emb_dim=32,
        force_reload=args.force_reload,
        cached_dir=DATA_DIR
    )
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = data
    
    if args.save_results is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_results = osp.join(
            RESULTS_DIR,
            f"{args.model}_{args.num_runs}runs_{timestamp}.json"
        )
    
    if args.model == "xgboost":
        config = {
            "n_estimators": args.xgb_n_estimators,
            "max_depth": args.xgb_max_depth,
            "lr": args.xgb_lr,
            "subsample": args.xgb_subsample,
            "colsample": args.xgb_colsample,
            "early_stopping_rounds": 50
        }
    elif args.model == "catboost":
        config = {
            "iterations": args.cat_iterations,
            "depth": args.cat_depth,
            "lr": args.cat_lr,
            "early_stopping_rounds": 50
        }
    elif args.model == "lightgbm":
        config = {
            "num_boost_round": args.lgb_num_boost_round,
            "num_leaves": args.lgb_num_leaves,
            "lr": args.lgb_lr,
            "feature_fraction": args.lgb_feature_fraction,
            "bagging_fraction": args.lgb_bagging_fraction,
            "early_stopping_rounds": 50
        }
    
    if args.grid:
        best_config, grid_results = run_grid_search(
            args.model, data, args.seed, args.grid_patience
        )
        
        grid_save_path = osp.join(
            RESULTS_DIR,
            f"{args.model}_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(grid_save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "model": args.model,
                "grid_results": grid_results,
                "best_config": best_config
            }, f, indent=2)
        print(f"Grid search saved: {grid_save_path}")
        config = best_config
    
    print(f"\nConfig: {config}")
    
    results = run_multiple_experiments(
        model_type=args.model,
        config=config,
        data=data,
        num_runs=args.num_runs,
        base_seed=args.seed,
        verbose=False
    )
    
    stats = compute_statistics(results)
    print_statistics(stats, args.num_runs)
    save_results_to_file(results, stats, config, args.model, args.save_results)
    
    best_run = max(results, key=lambda x: x["val_acc"])
    print(f"\nBest run: {best_run['run_id']} (seed={best_run['seed']})")
    
    set_seed(best_run["seed"])
    if args.model == "xgboost":
        model, _ = train_xgboost(X_train, y_train, X_val, y_val, config, best_run["seed"])
    elif args.model == "catboost":
        model, _ = train_catboost(X_train, y_train, X_val, y_val, config, best_run["seed"])
    else:
        model, _ = train_lightgbm(X_train, y_train, X_val, y_val, config, best_run["seed"])
    
    test_pred = predict_labels(model, X_test, args.model)
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred, digits=4))


if __name__ == "__main__":
    main()

