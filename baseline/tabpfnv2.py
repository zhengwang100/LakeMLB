import sys
import argparse
import os.path as osp

sys.path.append("./")
sys.path.append("../")
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', "lib"))
sys.path.insert(0, osp.dirname(__file__))

import random
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from rllm.types import ColType
from rllm.datasets.lakemlb import MSTrafficDataset
# from rllm.datasets.lakemlb import LHStocksDataset
from rllm.transforms.table_transforms import DefaultTableTransform
# from utils import set_global_seed


parser = argparse.ArgumentParser()
parser.add_argument("--f_dim", type=int, default=32)
parser.add_argument("--h_dim", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--with_val", action='store_true')
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()


seed = random.randint(0, 100000)
# set_global_seed(seed)

path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = MSTrafficDataset(cached_dir=path)
data_raw = dataset[0]

transform = DefaultTableTransform(out_dim=args.f_dim)
transform(data_raw)

clf = TabPFNClassifier.create_default_for_version(
    ModelVersion.V2, 
    random_state=seed, 
    ignore_pretraining_limits=True,
    device=args.device
)

features = list(data_raw.col_types.keys())
if data_raw.target_col in features:
    features.remove(data_raw.target_col)
data_df = data_raw.df

X = data_df[features]
y = data_df[data_raw.target_col]

train_mask = data_raw.train_mask.cpu().numpy() if isinstance(data_raw.train_mask, torch.Tensor) else data_raw.train_mask
test_mask = data_raw.test_mask.cpu().numpy() if isinstance(data_raw.test_mask, torch.Tensor) else data_raw.test_mask
X_train = X[train_mask].reset_index(drop=True)
y_train = y[train_mask].reset_index(drop=True)
X_test = X[test_mask].reset_index(drop=True)
y_test = y[test_mask].reset_index(drop=True)
if args.with_val:
    val_mask = data_raw.val_mask.cpu().numpy() if isinstance(data_raw.val_mask, torch.Tensor) else data_raw.val_mask
    X_val = X[val_mask].reset_index(drop=True)
    y_val = y[val_mask].reset_index(drop=True)
X_train = X_train.copy()
X_test = X_test.copy()

for col in X_train.columns:
    if col in data_raw.col_types:
        if data_raw.col_types[col] == ColType.CATEGORICAL:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
        elif data_raw.col_types[col] == ColType.NUMERICAL:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype(float)
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(float)

n_classes = y_train.nunique()
print(f"Dataset: train={X_train.shape[0]}, test={X_test.shape[0]}, classes={n_classes}")

if n_classes > 10:
    print(f"ERROR: {n_classes} classes exceeds TabPFN limit of 10")
    sys.exit(1)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {test_accuracy:.4f}")
