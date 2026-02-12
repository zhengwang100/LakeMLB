import sys
import argparse
import os.path as osp
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from tabicl import TabICLClassifier

sys.path.append("./")
sys.path.append("../")
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', "lib"))

from rllm.types import ColType
from rllm.datasets.lakemlb import MSTrafficDataset
from rllm.datasets.lakemlb import LHStocksDataset
from rllm.transforms.table_transforms import DefaultTableTransform
# from utils import set_global_seed


parser = argparse.ArgumentParser()
parser.add_argument("--f_dim", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--n_estimators", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--use_hierarchical", action='store_true', default=True)
parser.add_argument("--model_path", type=str, 
                    default="../lib/huggingface/hub/models--jingang--TabICL-clf/snapshots/main/tabicl-classifier-v1.1-0506.ckpt")
parser.add_argument("--verbose", action='store_true')
args = parser.parse_args()

if args.model_path:
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path
    args.model_path = str(model_path.resolve())


seed = random.randint(0, 100000)
# set_global_seed(seed)

path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = LHStocksDataset(cached_dir=path)
data_raw = dataset[0]

transform = DefaultTableTransform(out_dim=args.f_dim)
transform(data_raw)

clf = TabICLClassifier(
    n_estimators=args.n_estimators,
    norm_methods=["none", "power"],
    feat_shuffle_method="latin",
    class_shift=True,
    outlier_threshold=4.0,
    softmax_temperature=0.9,
    average_logits=True,
    use_hierarchical=args.use_hierarchical,
    batch_size=args.batch_size,
    use_amp=True,
    model_path=args.model_path,
    allow_auto_download=(args.model_path is None),
    device=args.device,
    random_state=seed,
    verbose=args.verbose,
)

features = list(data_raw.col_types.keys())
if data_raw.target_col in features:
    features.remove(data_raw.target_col)
data_df = data_raw.df

train_mask = data_raw.train_mask.cpu().numpy() if isinstance(data_raw.train_mask, torch.Tensor) else data_raw.train_mask
test_mask = data_raw.test_mask.cpu().numpy() if isinstance(data_raw.test_mask, torch.Tensor) else data_raw.test_mask

X = data_df[features]
y = data_df[data_raw.target_col]

X_train = X[train_mask].reset_index(drop=True)
y_train = y[train_mask].reset_index(drop=True)
X_test = X[test_mask].reset_index(drop=True)
y_test = y[test_mask].reset_index(drop=True)

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

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {test_accuracy:.4f}")

