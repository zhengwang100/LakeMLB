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
from tabpfn_extensions.many_class import ManyClassClassifier

from rllm.types import ColType
from datasets.mstraffic_datasets import MSTrafficDataset
from datasets.ncbuilding_datasets import NCBuildingDataset
from rllm.transforms.table_transforms import DefaultTableTransform
# from utils import set_global_seed


parser = argparse.ArgumentParser()
parser.add_argument("--f_dim", type=int, default=32)
parser.add_argument("--h_dim", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--use_many_class", action='store_true', 
                    help="Use ManyClassClassifier for datasets with >10 classes. Default: True for >10 classes")
args = parser.parse_args()


seed = random.randint(0, 100000)
# set_global_seed(seed)
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data")
dataset = NCBuildingDataset(cached_dir=path)
data_raw = dataset[0]

transform = DefaultTableTransform(out_dim=args.f_dim)
transform(data_raw)

base_clf = TabPFNClassifier.create_default_for_version(
    ModelVersion.V2,
    ignore_pretraining_limits=True,
    device=args.device
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

n_classes = y_train.nunique()
print(f"Dataset: train={X_train.shape[0]}, test={X_test.shape[0]}, classes={n_classes}")

if n_classes > 10:
    clf = ManyClassClassifier(
        estimator=base_clf,
        alphabet_size=10,
        random_state=seed,
        verbose=0
    )
else:
    clf = base_clf

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {test_accuracy:.4f}")
