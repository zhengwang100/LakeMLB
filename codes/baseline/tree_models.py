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
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

from lib.rllm.transforms.table_transforms import DefaultTableTransform
from lib.rllm.data.table_data import TableData
from lib.rllm.types import ColType
from lib.rllm.datasets import (
    MSTrafficDataset,
    NCBuildingDataset,
    GACarsDataset,
    NNStocksDataset,
    LHStocksDataset,
    DSMusicDataset,
    NCTaxiDataset,
    AGBooksDataset,
)
from utils import set_seed, get_device, generate_random_seed

_NNSTOCKS_TABLE_NAMES = {
    3: "nnstocks_fa",
    4: "stocks_wiki_llm_1nn",
    5: "t1_enriched_rank2",
    6: "t1_enriched_rank4",
    7: "t1_enriched_rank8",
    8: "stocks_wiki_tfidf_1nn",
    9: "t1_enriched_random",
}

_DATASET_REGISTRY = {
    "mstraffic": MSTrafficDataset,
    "ncbuilding": NCBuildingDataset,
    "gacars": GACarsDataset,
    "nnstocks": NNStocksDataset,
    "lhstocks": LHStocksDataset,
    "dsmusic": DSMusicDataset,
    "nctaxi": NCTaxiDataset,
    "agbooks": AGBooksDataset,
}

_TABLE_TAGS = {
    "mstraffic": {
        0: "mstraffic_maryland",
        1: "mstraffic_seattle",
        2: "mstraffic_da",
        3: "mstraffic_fa",
    },
    "nctaxi": {
        0: "nctaxi_newyork_taxi",
        1: "nctaxi_chicago_taxi",
    },
    "dsmusic": {
        0: "dsmusic_discogs",
        1: "dsmusic_spotify",
        2: "dsmusic_da",
        3: "dsmusic_fa",
        4: "dsmusic_1nn",
        5: "dsmusic_2nn",
        6: "dsmusic_4nn",
        7: "dsmusic_8nn",
        8: "dsmusic_random",
    },
    "agbooks": {
        0: "agbooks_amazon",
        1: "agbooks_goodreads",
        2: "agbooks_amazon_enriched",
        4: "agbooks_amazon_no_features",
        5: "agbooks_amazon_no_features_10k",
        6: "agbooks_1nn",
        7: "agbooks_2nn",
        8: "agbooks_4nn",
        9: "agbooks_8nn",
        10: "agbooks_random",
    },
    "nnstocks": _NNSTOCKS_TABLE_NAMES,
}


def get_dataset_tag(dataset_name: str, table_idx: int) -> str:
    return _TABLE_TAGS.get(dataset_name, {}).get(
        table_idx, f"{dataset_name}_table{table_idx}"
    )

# ── per-dataset parquet configs (carte-preprocessed) ──────────────────────────
_NNSTOCKS_COL_TYPES = {
    "symbol":        ColType.CATEGORICAL,
    "name":          ColType.CATEGORICAL,
    "lastsale":      ColType.NUMERICAL,
    "netchange":     ColType.NUMERICAL,
    "pctchange":     ColType.NUMERICAL,
    "volume":        ColType.NUMERICAL,
    "marketCap":     ColType.NUMERICAL,
    "country":       ColType.CATEGORICAL,
    "url":           ColType.CATEGORICAL,
    "wiki_title":    ColType.CATEGORICAL,
    "company_type":  ColType.CATEGORICAL,
    "traded_as":     ColType.CATEGORICAL,
    "founded":       ColType.CATEGORICAL,
    "headquarters":  ColType.CATEGORICAL,
    "key_people":    ColType.CATEGORICAL,
    "revenue":       ColType.CATEGORICAL,
    "net_income":    ColType.CATEGORICAL,
    "total_assets":  ColType.CATEGORICAL,
    "num_employees": ColType.CATEGORICAL,
    "products":      ColType.CATEGORICAL,
}

_PARQUET_CONFIGS = {
    "stocks_wiki_llm_1nn": {"parquet": "stocks_wiki_llm_1nn", "mask": "mask_nnlist", "target": "sector", "col_types": _NNSTOCKS_COL_TYPES},
    "stocks_wiki_llm_2nn": {"parquet": "stocks_wiki_llm_2nn", "mask": "mask_nnlist", "target": "sector", "col_types": _NNSTOCKS_COL_TYPES},
    "stocks_wiki_llm_4nn": {"parquet": "stocks_wiki_llm_4nn", "mask": "mask_nnlist", "target": "sector", "col_types": _NNSTOCKS_COL_TYPES},
    "stocks_wiki_llm_8nn": {"parquet": "stocks_wiki_llm_8nn", "mask": "mask_nnlist", "target": "sector", "col_types": _NNSTOCKS_COL_TYPES},
}


AVAILABLE_MODELS = ["xgboost", "catboost", "lightgbm"]
_DATA_CACHE = {}


SCRIPT_DIR  = osp.dirname(osp.abspath(__file__))
DATA_DIR    = osp.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = osp.join(SCRIPT_DIR, "..", "results", "tree_models")
LOG_DIR     = osp.join(SCRIPT_DIR, "..", "results", "logs", "tree_models")
ARTIFACT_DIR = osp.join(SCRIPT_DIR, "..", "results", "artifacts", "tree_models")
LIB_DIR     = osp.join(SCRIPT_DIR, "..", "lib")
for _d in (RESULTS_DIR, LOG_DIR, ARTIFACT_DIR):
    os.makedirs(_d, exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(log_dir: str, model: str, data_tag: str) -> Optional[str]:
    os.makedirs(log_dir, exist_ok=True)
    log_path = osp.join(
        log_dir,
        f"{model}_{data_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    print(f"Log file: {log_path}")
    return log_path


def save_tree_model(model, model_type: str, model_path: str):
    os.makedirs(osp.dirname(model_path), exist_ok=True)
    if model_type in {"xgboost", "catboost"}:
        model.save_model(model_path)
    elif model_type == "lightgbm":
        model.save_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def model_extension(model_type: str) -> str:
    return {
        "xgboost": "json",
        "catboost": "cbm",
        "lightgbm": "txt",
    }[model_type]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=AVAILABLE_MODELS,
        help="Tree model to use"
    )
    parser.add_argument("--dataset", type=str, default="mstraffic",
                        choices=sorted(_DATASET_REGISTRY.keys()),
                        help="LakeMLB dataset name. Default: mstraffic.")
    parser.add_argument("--table_idx", type=int, default=0,
                        help="Table index inside the selected LakeMLB dataset. "
                             "Default: 0, the task table for most datasets.")
    parser.add_argument("--data_name", type=str, default=None,
                        help="(Optional) parquet-based loading override. "
                             "When set, --dataset/--table_idx are ignored.")

    # Experiment settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--num_threads", type=int, default=0,
                        help="Threads used inside each tree learner. "
                             "0 means library default / all available threads.")
    parser.add_argument("--save_results", type=str, default=None)
    parser.add_argument("--grid_results", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=LOG_DIR)
    parser.add_argument("--artifact_dir", type=str, default=ARTIFACT_DIR)
    parser.add_argument("--force_reload", action="store_true", default=False)
    parser.add_argument("--xgb_n_estimators", type=int, default=500)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_lr", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample", type=float, default=0.9)
    parser.add_argument("--cat_iterations", type=int, default=500)
    parser.add_argument("--cat_depth", type=int, default=6)
    parser.add_argument("--cat_lr", type=float, default=0.05)
    parser.add_argument("--cat_subsample", type=float, default=0.9)
    parser.add_argument("--cat_rsm", type=float, default=0.9)
    parser.add_argument("--lgb_num_boost_round", type=int, default=500)
    parser.add_argument("--lgb_num_leaves", type=int, default=63)
    parser.add_argument("--lgb_lr", type=float, default=0.03)
    parser.add_argument("--lgb_feature_fraction", type=float, default=0.9)
    parser.add_argument("--lgb_bagging_fraction", type=float, default=0.9)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--resume_grid", action="store_true",
                        help="Resume grid search from <grid_results>.partial when available.")
    parser.add_argument("--grid_patience", type=int, default=50)
    
    return parser.parse_args()


def load_data(
    dataset_class,
    dataset_name: str = "nnstocks",
    device: torch.device = None,
    emb_dim: int = 32,
    force_reload: bool = False,
    table_idx: int = 0,
    **dataset_kwargs
) -> Tuple:
    global _DATA_CACHE
    
    cache_key = f"{dataset_name}_{table_idx}_{emb_dim}"
    
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
    if table_idx < 0 or table_idx >= len(dataset.data_list):
        raise IndexError(
            f"table_idx={table_idx} is out of range for {dataset_name}; "
            f"available range is 0..{len(dataset.data_list) - 1}."
        )
    
    data = dataset.data_list[table_idx]
    data.y = data.y.long().to(device)
    
    if not (hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask')):
        raise ValueError(
            f"{dataset_name}[{table_idx}] must have train_mask, val_mask, "
            "and test_mask for supervised tree-model evaluation."
        )
    
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
    
    print(
        f"Dataset loaded: {dataset_name}[{table_idx}], train={len(y_train)}, "
        f"val={len(y_val)}, test={len(y_test)}, classes={num_classes}, "
        f"dim={X_train.shape[1]}"
    )
    
    return result


def load_data_from_parquet(
    data_name: str,
    emb_dim: int = 32,
    device: torch.device = None,
) -> tuple:
    """Load dataset from carte-preprocessed parquet + mask file."""
    global _DATA_CACHE
    cache_key = f"{data_name}_{emb_dim}"
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    from carte_ai.configs.directory import config_directory as carte_config_directory

    cfg = _PARQUET_CONFIGS[data_name]
    parquet_path = osp.join(carte_config_directory['data_singletable'], cfg['parquet'], "raw.parquet")
    mask_path    = osp.join(carte_config_directory['data_raw'], f"{cfg['mask']}.pt")

    df    = pd.read_parquet(parquet_path)
    masks = torch.load(mask_path, weights_only=False)

    data = TableData(
        df=df,
        col_types=cfg['col_types'],
        target_col=cfg['target'],
        train_mask=masks['train_mask'],
        val_mask=masks['val_mask'],
        test_mask=masks['test_mask'],
    )
    DefaultTableTransform(out_dim=emb_dim)(data)

    if device is None:
        device = torch.device('cpu')
    data.y = data.y.long().to(device)

    train_mask = data.train_mask.cpu().numpy()
    val_mask   = data.val_mask.cpu().numpy()
    test_mask  = data.test_mask.cpu().numpy()

    feat_dict = data.get_feat_dict()
    feat_list = []
    for key in sorted(feat_dict.keys()):
        t = feat_dict[key]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        feat_list.append(t.cpu().numpy())

    X = np.concatenate(feat_list, axis=1)
    y = data.y.cpu().numpy()

    result = (X[train_mask], y[train_mask], X[val_mask], y[val_mask],
              X[test_mask], y[test_mask], len(np.unique(y)))
    _DATA_CACHE[cache_key] = result

    X_train, y_train, _, _, X_test, y_test, num_classes = result
    print(f"Parquet loaded [{data_name}]: train={len(y_train)}, test={len(y_test)}, "
          f"classes={num_classes}, dim={X_train.shape[1]}")
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
        n_jobs=config.get("num_threads", 0) or None,
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
    
    model_params = {
        "iterations": config["iterations"],
        "depth": config["depth"],
        "learning_rate": config["lr"],
        "loss_function": "MultiClass" if num_classes > 2 else "Logloss",
        "random_seed": seed,
        "verbose": verbose,
        "early_stopping_rounds": config.get("early_stopping_rounds", 50),
        "thread_count": config.get("num_threads", 0) or -1,
        "rsm": config.get("rsm", 1.0),
    }
    if config.get("subsample", 1.0) < 1.0:
        model_params["bootstrap_type"] = "Bernoulli"
        model_params["subsample"] = config["subsample"]

    model = CatBoostClassifier(
        **model_params,
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
        "num_threads": config.get("num_threads", 0) or 0,
        "force_col_wise": True,
        "deterministic": True,
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
            lgb.log_evaluation(period=50)
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
    verbose: bool = False,
    save_path: Optional[str] = None,
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
    if save_path:
        save_tree_model(model, model_type, save_path)
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
        seed = generate_random_seed()
        
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
    save_path: str,
    dataset_tag: str,
    grid_results_path: Optional[str] = None,
    best_grid_model_path: Optional[str] = None,
    log_path: Optional[str] = None,
):
    
    output = {
        "model": model_name,
        "task": "classification",
        "dataset": dataset_tag,
        "config": config,
        "num_runs": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grid_results_path": grid_results_path,
        "best_grid_model_path": best_grid_model_path,
        "log_path": log_path,
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
            "n_estimators": [300, 1000],
            "max_depth": [6, 8],
            "lr": [0.01, 0.05],
            "subsample": [0.9],
            "colsample": [0.8]
        }
    elif model_type == "catboost":
        return {
            "iterations": [300, 1000],
            "depth": [6, 8],
            "lr": [0.01, 0.05],
            "subsample": [0.9],
            "rsm": [0.8],
        }
    elif model_type == "lightgbm":
        return {
            "num_boost_round": [300, 1000],
            "num_leaves": [63, 127],
            "lr": [0.01, 0.05],
            "feature_fraction": [0.8],
            "bagging_fraction": [0.9]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def is_better_grid_result(result: Dict, best_result: Optional[Dict]) -> bool:
    if best_result is None:
        return True
    return (
        result["val_acc"],
        result["val_f1"],
    ) > (
        best_result["val_acc"],
        best_result["val_f1"],
    )


def run_grid_search(
    model_type: str,
    data: Tuple,
    base_seed: int,
    early_stopping_rounds: int = 50,
    artifact_dir: Optional[str] = None,
    data_tag: str = "dataset",
    num_threads: int = 0,
    partial_results_path: Optional[str] = None,
    resume_grid: bool = False,
) -> Tuple[Dict, List[Dict], Dict]:
    X_train, y_train, X_val, y_val, X_test, y_test, _ = data
    
    space = get_grid_search_space(model_type)
    all_combinations = list(itertools.product(*space.values()))
    
    print(f"\nGrid search: {model_type.upper()}, {len(all_combinations)} combinations")
    
    best_config = None
    best_result = None
    best_model_path = None
    best_metadata_path = None
    all_results = []
    completed_configs = set()
    grid_start = time.perf_counter()

    if resume_grid and partial_results_path and osp.exists(partial_results_path):
        with open(partial_results_path, "r", encoding="utf-8") as f:
            partial_payload = json.load(f)
        all_results = partial_payload.get("grid_results", [])
        for record in all_results:
            completed_configs.add(json.dumps(record["config"], sort_keys=True))
            if is_better_grid_result(record, best_result):
                best_result = record.copy()
                best_config = record["config"].copy()
        print(
            f"Resuming grid search from {partial_results_path}: "
            f"{len(all_results)}/{len(all_combinations)} completed"
        )
    
    for idx, combination in enumerate(all_combinations):
        combo_start = time.perf_counter()
        config = dict(zip(space.keys(), combination))
        config["early_stopping_rounds"] = early_stopping_rounds
        config["num_threads"] = num_threads
        config_key = json.dumps(config, sort_keys=True)
        if config_key in completed_configs:
            print(f"[{idx+1}/{len(all_combinations)}] Skip completed config: {config}")
            continue
        
        result = run_single_experiment(
            model_type, config, X_train, y_train, X_val, y_val,
            X_test, y_test, base_seed, verbose=False,
        )
        
        result_record = {
            "config": config,
            "val_acc": result["val_acc"],
            "test_acc": result["test_acc"],
            "val_f1": result["val_f1"],
            "test_f1": result["test_f1"],
            "training_time": result["training_time"],
            "best_iteration": result.get("best_iteration"),
        }
        all_results.append(result_record)
        combo_time = time.perf_counter() - combo_start
        
        print(
            f"[{idx+1}/{len(all_combinations)}] "
            f"val_acc={result['val_acc']:.4f}, val_f1={result['val_f1']:.4f}, "
            f"test_acc={result['test_acc']:.4f}, time={combo_time:.2f}s"
        )
        if is_better_grid_result(result_record, best_result):
            best_result = result_record.copy()
            best_config = config.copy()

        if partial_results_path:
            os.makedirs(osp.dirname(partial_results_path), exist_ok=True)
            with open(partial_results_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_type,
                    "dataset": data_tag,
                    "completed": len(all_results),
                    "total": len(all_combinations),
                    "last_config": config,
                    "current_best_result": best_result,
                    "grid_results": all_results,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2, ensure_ascii=False)

    grid_time = time.perf_counter() - grid_start
    print(
        f"\nBest config: {best_config}, "
        f"val_acc={best_result['val_acc']:.4f}, val_f1={best_result['val_f1']:.4f}"
    )
    if artifact_dir and best_config:
        best_grid_path = osp.join(
            artifact_dir,
            data_tag,
            model_type,
            "grid",
            f"best_grid_seed{base_seed}.{model_extension(model_type)}",
        )
        model, model_metrics = (
            train_xgboost(X_train, y_train, X_val, y_val, best_config, base_seed)
            if model_type == "xgboost"
            else train_catboost(X_train, y_train, X_val, y_val, best_config, base_seed)
            if model_type == "catboost"
            else train_lightgbm(X_train, y_train, X_val, y_val, best_config, base_seed)
        )
        save_tree_model(model, model_type, best_grid_path)
        meta_path = osp.join(osp.dirname(best_grid_path), "best_grid_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_type,
                "dataset": data_tag,
                "config": best_config,
                "metrics": model_metrics,
                "selection": {
                    "primary": "val_acc",
                    "tie_breaker": "val_f1",
                },
                "best_grid_result": best_result,
                "model_path": best_grid_path,
                "grid_time": grid_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2, ensure_ascii=False)
        best_model_path = best_grid_path
        best_metadata_path = meta_path
        print(f"Best grid model: {best_grid_path}")
        print(f"Best grid metadata: {meta_path}")
    print(f"Grid search time: {grid_time:.2f}s")
    return best_config, all_results, {
        "best_result": best_result,
        "best_model_path": best_model_path,
        "best_metadata_path": best_metadata_path,
        "grid_time": grid_time,
    }


def main():
    run_start = time.perf_counter()
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    
    if args.data_name:
        data_tag = args.data_name
    else:
        data_tag = get_dataset_tag(args.dataset, args.table_idx)
    log_path = setup_logging(args.log_dir, args.model, data_tag)
    print(f"\nModel: {args.model.upper()}  Dataset: {data_tag}")

    if args.data_name and args.data_name in _PARQUET_CONFIGS:
        data = load_data_from_parquet(
            data_name=args.data_name,
            emb_dim=32,
            device=device,
        )
    elif args.data_name:
        raise ValueError(
            f"Unknown data_name={args.data_name!r}. Available parquet configs: "
            f"{sorted(_PARQUET_CONFIGS.keys())}"
        )
    else:
        dataset_class = _DATASET_REGISTRY[args.dataset]
        data = load_data(
            dataset_class=dataset_class,
            dataset_name=args.dataset,
            device=device,
            emb_dim=32,
            force_reload=args.force_reload,
            table_idx=args.table_idx,
            cached_dir=DATA_DIR
        )
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = data

    if args.save_results is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_results = osp.join(
            RESULTS_DIR,
            data_tag,
            args.model,
            f"final_{args.num_runs}runs_{timestamp}.json"
        )
    if args.grid_results is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.grid_results = osp.join(
            RESULTS_DIR,
            data_tag,
            args.model,
            f"grid_search_{timestamp}.json"
        )
    
    if args.model == "xgboost":
        config = {
            "n_estimators": args.xgb_n_estimators,
            "max_depth": args.xgb_max_depth,
            "lr": args.xgb_lr,
            "subsample": args.xgb_subsample,
            "colsample": args.xgb_colsample,
            "early_stopping_rounds": 50,
            "num_threads": args.num_threads,
        }
    elif args.model == "catboost":
        config = {
            "iterations": args.cat_iterations,
            "depth": args.cat_depth,
            "lr": args.cat_lr,
            "subsample": args.cat_subsample,
            "rsm": args.cat_rsm,
            "early_stopping_rounds": 50,
            "num_threads": args.num_threads,
        }
    elif args.model == "lightgbm":
        config = {
            "num_boost_round": args.lgb_num_boost_round,
            "num_leaves": args.lgb_num_leaves,
            "lr": args.lgb_lr,
            "feature_fraction": args.lgb_feature_fraction,
            "bagging_fraction": args.lgb_bagging_fraction,
            "early_stopping_rounds": 50,
            "num_threads": args.num_threads,
        }
    
    if args.grid:
        best_config, grid_results, grid_summary = run_grid_search(
            args.model, data, args.seed, args.grid_patience,
            artifact_dir=args.artifact_dir,
            data_tag=data_tag,
            num_threads=args.num_threads,
            partial_results_path=f"{args.grid_results}.partial",
            resume_grid=args.resume_grid,
        )
        
        os.makedirs(osp.dirname(args.grid_results), exist_ok=True)
        with open(args.grid_results, 'w', encoding='utf-8') as f:
            json.dump({
                "model": args.model,
                "dataset": data_tag,
                "grid_results": grid_results,
                "best_config": best_config,
                "best_result": grid_summary["best_result"],
                "best_grid_model_path": grid_summary["best_model_path"],
                "best_grid_metadata_path": grid_summary["best_metadata_path"],
                "grid_time": grid_summary["grid_time"],
                "selection": {
                    "primary": "val_acc",
                    "tie_breaker": "val_f1",
                },
                "early_stopping_rounds": args.grid_patience,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        print(f"Grid search saved: {args.grid_results}")
        config = best_config
    else:
        grid_summary = {
            "best_model_path": None,
        }
    
    print(f"\nConfig: {config}")
    
    final_start = time.perf_counter()
    results = run_multiple_experiments(
        model_type=args.model,
        config=config,
        data=data,
        num_runs=args.num_runs,
        base_seed=args.seed,
        verbose=False
    )
    
    stats = compute_statistics(results)
    stats["final_training_total_time"] = time.perf_counter() - final_start
    stats["total_runtime"] = time.perf_counter() - run_start
    print_statistics(stats, args.num_runs)
    save_results_to_file(
        results,
        stats,
        config,
        args.model,
        args.save_results,
        dataset_tag=data_tag,
        grid_results_path=args.grid_results if args.grid else None,
        best_grid_model_path=grid_summary["best_model_path"],
        log_path=log_path,
    )
    
    best_run = max(results, key=lambda x: x["val_acc"])
    print(f"\nBest run: {best_run['run_id']} (seed={best_run['seed']})")
    
    set_seed(best_run["seed"])
    if args.model == "xgboost":
        model, model_metrics = train_xgboost(X_train, y_train, X_val, y_val, config, best_run["seed"])
    elif args.model == "catboost":
        model, model_metrics = train_catboost(X_train, y_train, X_val, y_val, config, best_run["seed"])
    else:
        model, model_metrics = train_lightgbm(X_train, y_train, X_val, y_val, config, best_run["seed"])
    _ = model_metrics
    
    test_pred = predict_labels(model, X_test, args.model)
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred, digits=4))
    print(f"Total runtime: {time.perf_counter() - run_start:.2f}s")


if __name__ == "__main__":
    main()
