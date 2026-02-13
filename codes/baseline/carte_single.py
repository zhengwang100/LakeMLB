import os
from pathlib import Path
import sys
import os.path as osp
os.chdir(Path().cwd().parent)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', "lib"))
sys.path.insert(0, osp.dirname(__file__))

import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from carte_ai.src.carte_estimator import CARTEClassifier
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer
from carte_ai.configs.directory import config_directory


def _load_data(data_name):
    """Load the preprocessed data."""
    data_pd_dir = f"{config_directory['data_singletable']}/{data_name}/raw.parquet"
    data_pd = pd.read_parquet(data_pd_dir)
    data_pd.fillna(value=np.nan, inplace=True)
    config_data_dir = f"{config_directory['data_singletable']}/{data_name}/config_data.json"
    with open(config_data_dir) as f:
        config_data = json.load(f)
    return data_pd, config_data


def load_fixed_mask(data_name):
    """Load fixed train/val/test masks."""
    mask_path = osp.join(config_directory['data_raw'], f"mask_{data_name}.pt")
    if not osp.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    masks = torch.load(mask_path, weights_only=False)
    train_n = masks['train_mask'].sum().item()
    val_n = masks['val_mask'].sum().item()
    test_n = masks['test_mask'].sum().item()
    total = masks['train_mask'].numel()
    print(f"Loaded masks: train={train_n} ({train_n/total*100:.1f}%), val={val_n} ({val_n/total*100:.1f}%), test={test_n} ({test_n/total*100:.1f}%)")
    
    return masks


def apply_fixed_mask(data, target_name, masks):
    """Split data using fixed masks."""
    X = data.drop(columns=target_name)
    y = data[target_name].to_numpy()
    
    train_mask = masks['train_mask'].numpy()
    val_mask = masks['val_mask'].numpy()
    test_mask = masks['test_mask'].numpy()
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class CARTEClassifierFixedSplit(CARTEClassifier):
    """CARTE Classifier with fixed train/val split using predefined indices."""
    
    def __init__(self, fixed_train_idx, fixed_val_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_train_idx = np.array(fixed_train_idx)
        self.fixed_val_idx = np.array(fixed_val_idx)
    
    def _set_train_valid_split(self):
        """Override: return fixed train/val indices instead of random split."""
        assert self.fixed_train_idx.max() < len(self.X_), f"train_idx out of range: {self.fixed_train_idx.max()} >= {len(self.X_)}"
        assert self.fixed_val_idx.max() < len(self.X_), f"val_idx out of range: {self.fixed_val_idx.max()} >= {len(self.X_)}"
        
        splits = [(self.fixed_train_idx, self.fixed_val_idx) for _ in range(self.num_model)]
        return splits


data_name = "dsmusic_enriched"
mask_basename = "dsmusic"

print(f"CARTE Single Table - Fixed split (mask_{mask_basename}.pt)")
print("=" * 80)

print("\n[1] Loading data...")
data, data_config = _load_data(data_name)
print(f"Loaded {len(data)} samples")

print("\n[2] Loading fixed masks...")
masks = load_fixed_mask(mask_basename)

print("\n[3] Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = apply_fixed_mask(
    data, data_config["target_name"], masks
)

print("\n[4] Merging train+val (shuffled to avoid class clustering)...")
from sklearn.utils import shuffle as sklearn_shuffle
X_train_shuffled, y_train_shuffled = sklearn_shuffle(X_train, y_train, random_state=0)
X_val_shuffled, y_val_shuffled = sklearn_shuffle(X_val, y_val, random_state=0)

X_train_val = pd.concat([X_train_shuffled, X_val_shuffled], axis=0, ignore_index=True)
y_train_val = np.concatenate([y_train_shuffled, y_val_shuffled])
print(f"Merged: train({len(y_train)}) + val({len(y_val)}) = {len(y_train_val)}")

print("\n[5] Converting to graph structure...")
fasttext_model_path = osp.join(osp.dirname(__file__), '..', "lib", "FastText", "cc.en.300.bin")
preprocessor = Table2GraphTransformer(
    fasttext_model_path=fasttext_model_path,
    num_transformer=StandardScaler()
)

X_train_val_graphs = preprocessor.fit_transform(X_train_val, y=y_train_val)
X_test_graphs = preprocessor.transform(X_test)
print(f"Graphs: train+val={len(X_train_val_graphs)}, test={len(X_test_graphs)}")

print("\n[6] Creating CARTE indices...")
carte_train_idx = np.arange(len(y_train))
carte_val_idx = np.arange(len(y_train), len(y_train) + len(y_val))

print("\n[7] Training model...")
print("=" * 80)

fixed_params = {
    "num_model": 5,
    "disable_pbar": False,
    "random_state": 0,
    "device": "cuda:0",
    "n_jobs": 5,
    "loss": "categorical_crossentropy",
    "scoring": "accuracy",
    "num_layers": 1,
    "batch_size": 256,
    "learning_rate": 1e-3,
    # "max_epoch": 1,
    "dropout": 0,
    "val_size": 0.125,
    "early_stopping_patience": 40,
}

estimator = CARTEClassifierFixedSplit(
    fixed_train_idx=carte_train_idx,
    fixed_val_idx=carte_val_idx,
    **fixed_params
)

estimator.fit(X=X_train_val_graphs, y=y_train_val)

print("\n[8] Evaluating...")
print("=" * 80)

preds_test = estimator.predict(X_test_graphs)
accuracy = accuracy_score(y_test, preds_test)
print(f"\nTest accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y_test, preds_test, digits=4))

