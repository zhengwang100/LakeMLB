import os
from pathlib import Path
import sys
import os.path as osp
os.chdir(Path().cwd().parent)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', "lib"))
sys.path.insert(0, osp.dirname(__file__))
import torch
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer
from carte_ai.src.carte_estimator import CARTEMultitableClassifer
from carte_ai.configs.directory import config_directory


def _load_data(data_name):
    data_pd = pd.read_parquet(f"{config_directory['data_singletable']}/{data_name}/raw.parquet")
    data_pd.fillna(value=np.nan, inplace=True)
    with open(f"{config_directory['data_singletable']}/{data_name}/config_data.json") as f:
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


def _prepare_multitable_fixed(
    data_t, config_t, masks_t,
    data_s_total, config_s_total
):
    """Prepare multi-table data with fixed split."""
    Xt_train, Xt_val, Xt_test, yt_train, yt_val, yt_test = apply_fixed_mask(
        data_t, config_t["target_name"], masks_t
    )
    
    from sklearn.utils import shuffle as sklearn_shuffle
    Xt_train_shuffled, yt_train_shuffled = sklearn_shuffle(Xt_train, yt_train, random_state=0)
    Xt_val_shuffled, yt_val_shuffled = sklearn_shuffle(Xt_val, yt_val, random_state=0)
    
    Xt_train_val = pd.concat([Xt_train_shuffled, Xt_val_shuffled], axis=0, ignore_index=True)
    yt_train_val = np.concatenate([yt_train_shuffled, yt_val_shuffled])
    
    fasttext_model_path = osp.join(osp.dirname(__file__), '..', "lib", "FastText", "cc.en.300.bin")
    graph = Table2GraphTransformer(fasttext_model_path=fasttext_model_path, num_transformer=StandardScaler())
    Xt_carte_train_val = graph.fit_transform(X=Xt_train_val, y=yt_train_val)
    Xt_carte_test = graph.transform(Xt_test)
    
    for d in Xt_carte_train_val + Xt_carte_test:
        d.domain = 0
    
    carte_train_idx = np.arange(len(yt_train))
    carte_val_idx = np.arange(len(yt_train), len(yt_train) + len(yt_val))
    
    Xs_carte = {}
    domain_marker = 1
    for name, df_s in data_s_total.items():
        cfg_s = config_s_total[name]
        is_unlabeled = cfg_s["target_name"] is None
        
        if is_unlabeled:
            Xs_temp = graph.fit_transform(X=df_s, y=None)
            Xs_pruned = Xs_temp
        else:
            Xs_temp = graph.fit_transform(
                X=df_s.drop(columns=cfg_s["target_name"]),
                y=df_s[cfg_s["target_name"]].to_numpy()
            )
            ys = np.array([d.y.cpu().item() for d in Xs_temp])
            keep = ~np.isnan(ys)
            Xs_pruned = [d for d, k in zip(Xs_temp, keep) if k]
        
        for d in Xs_pruned:
            if is_unlabeled:
                d.y = torch.tensor([-1], dtype=torch.long)
            else:
                d.y = torch.tensor([d.y.cpu().item()], dtype=torch.long)
            d.domain = domain_marker
        Xs_carte[name] = Xs_pruned
        domain_marker += 1
    
    return (Xt_carte_train_val, Xt_carte_test, Xs_carte, 
            yt_train_val, yt_test, carte_train_idx, carte_val_idx)


class CARTEMultitableClassiferFixedSplit(CARTEMultitableClassifer):
    """CARTE Multi-table Classifier with fixed train/val split."""
    
    def __init__(self, fixed_train_idx, fixed_val_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_train_idx = np.array(fixed_train_idx)
        self.fixed_val_idx = np.array(fixed_val_idx)
    
    def _set_train_valid_split(self):
        """Override: return fixed train/val indices instead of random split."""
        splits = [(self.fixed_train_idx, self.fixed_val_idx) for _ in range(self.num_model)]
        return splits


target_data_name = "maryland"
source_data_name = ["seattle"]
mask_basename = "maryland"

print(f"CARTE Multi-Table - Fixed split (mask_{mask_basename}.pt)")
print("=" * 80)

print("\n[1] Loading data...")
data_t, config_t = _load_data(target_data_name)
print(f"Target table: {target_data_name} ({len(data_t)} samples)")

data_s_total, config_s_total = {}, {}
for nm in source_data_name:
    df, cfg = _load_data(nm)
    data_s_total[nm] = df.copy()
    config_s_total[nm] = cfg
    print(f"Source table: {nm} ({len(df)} samples)")

print("\n[2] Loading fixed masks...")
masks_t = load_fixed_mask(mask_basename)

print("\n[3] Preparing multi-table data...")
(Xt_tr_val, Xt_te, Xs_carte, yt_tr_val, yt_te, 
 carte_train_idx, carte_val_idx) = _prepare_multitable_fixed(
    data_t, config_t, masks_t,
    data_s_total, config_s_total
)

print(f"Target train+val: {len(yt_tr_val)} graphs")
print(f"Target test: {len(yt_te)} graphs")
for name, data_list in Xs_carte.items():
    print(f"Source {name}: {len(data_list)} graphs")

print("\n[4] Training model...")
print("=" * 80)

fixed_params = {
    "source_data": Xs_carte,
    "num_model": 5,
    "n_jobs": 5,
    "random_state": 0,
    "disable_pbar": False,
    "loss": "categorical_crossentropy",
    "scoring": "accuracy",
    "device": "cuda:0",
    "num_layers": 1,
    "batch_size": 256,
    "learning_rate": 1e-3,
    # "max_epoch": 200,
    # "dropout": 0.1,
    "val_size": 0.2,
    # "early_stopping_patience": 50,
    # "target_fraction": 0.5,
}

estimator = CARTEMultitableClassiferFixedSplit(
    fixed_train_idx=carte_train_idx,
    fixed_val_idx=carte_val_idx,
    **fixed_params
)
estimator.fit(Xt_tr_val, yt_tr_val)

print("\nDomain information:")
print("=" * 80)
print(f"Domains: {estimator.source_list_total_}")
print(f"Domain count: {len(estimator.source_list_total_)}")
print(f"Models trained: {len(estimator.model_list_)}")
print(f"\nLearned domain weights:")
for domain, w in zip(estimator.source_list_total_, estimator.weights_):
    print(f"  {domain:>15}: {w:.4f}")

print("\n[5] Evaluating...")
print("=" * 80)

Xt_train_only = [Xt_tr_val[i] for i in carte_train_idx]
yt_train_only = yt_tr_val[carte_train_idx]

y_pred_train = estimator.predict(Xt_train_only)
y_pred_test = estimator.predict(Xt_te)

acc_train = accuracy_score(yt_train_only, y_pred_train)
acc_test = accuracy_score(yt_te, y_pred_test)

print(f"\nTrain accuracy: {acc_train:.4f}")
print(f"Test accuracy: {acc_test:.4f}")
print(f"Overfitting: {acc_train - acc_test:.4f}")
print("\nClassification report:")
print(classification_report(yt_te, y_pred_test, digits=4))

