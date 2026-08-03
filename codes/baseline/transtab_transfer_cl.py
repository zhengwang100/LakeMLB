"""TransTab contrastive transfer learning for LakeMLB tables."""
import argparse
import json
import os
import os.path as osp
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, osp.join(osp.dirname(__file__), "..", "lib"))

import numpy as np
import torch
import transtab
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from transtab_lakemlb_utils import prepare_table


parser = argparse.ArgumentParser(description="TransTab contrastive transfer learning")
parser.add_argument("--dataset", type=str, default="mstraffic", help="Task dataset family.")
parser.add_argument("--table_idx", type=int, default=0, help="Task table index.")
parser.add_argument("--aux_dataset", type=str, default="mstraffic", help="Auxiliary dataset family.")
parser.add_argument("--aux_table_idx", type=int, default=1, help="Auxiliary table index.")
parser.add_argument("--work_dir", type=str, default=None)
parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
parser.add_argument("--pretrain_dir", type=str, default="./ckpt_cl/pretrained")
parser.add_argument("--num_epoch_pretrain", type=int, default=100)
parser.add_argument("--num_epoch_finetune", type=int, default=100)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_results", type=str, default=None)
args = parser.parse_args()

run_start = time.perf_counter()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

work_dir = args.work_dir or osp.join(args.ckpt_dir, "data")
task_dir = osp.join(work_dir, "task")
aux_dir = osp.join(work_dir, "aux")

task_table = prepare_table(
    args.dataset, args.table_idx, task_dir, "task", args.seed,
    require_target=True, use_target=True,
)
aux_table = prepare_table(
    args.aux_dataset, args.aux_table_idx, aux_dir, "aux", args.seed + 1,
    require_target=False, use_target=False,
)

print("Arguments:")
print(f"  Task: {args.dataset}[{args.table_idx}] ({task_table.tag})")
print(f"  Aux : {args.aux_dataset}[{args.aux_table_idx}] ({aux_table.tag}, unlabeled)")
print(f"  Checkpoint directory: {args.ckpt_dir}")
print(f"  Pretrain directory: {args.pretrain_dir}")
print(f"  Pretrain epochs: {args.num_epoch_pretrain}")
print(f"  Finetune epochs: {args.num_epoch_finetune}")
print(f"  Device: {args.device}")
print(f"  Seed: {args.seed}")

print("Stage 1: Contrastive pretraining on unlabeled auxiliary table")
allset1, trainset1, valset1, testset1, cat_cols1, num_cols1, bin_cols1 = transtab.load_data(
    [aux_table.csv_dir],
    dataset_config={aux_table.csv_dir: aux_table.config},
    filename=aux_table.csv_name,
)
print(f"Aux train={len(trainset1[0][0])}, val={len(valset1[0][0])}, test={len(testset1[0][0])}")

model_pretrain, collate_fn = transtab.build_contrastive_learner(
    categorical_columns=cat_cols1,
    numerical_columns=num_cols1,
    binary_columns=bin_cols1,
    supervised=False,
    num_partition=4,
    overlap_ratio=0.5,
)
transtab.train(
    model_pretrain,
    trainset1,
    valset1,
    collate_fn=collate_fn,
    num_epoch=args.num_epoch_pretrain,
    lr=1e-4,
    eval_metric="val_loss",
    eval_less_is_better=True,
    output_dir=args.pretrain_dir,
)
print(f"Contrastive learning model saved to {args.pretrain_dir}")

print("Stage 2: Fine-tuning on task table")
allset2, trainset2, valset2, testset2, cat_cols2, num_cols2, bin_cols2 = transtab.load_data(
    [task_table.csv_dir],
    dataset_config={task_table.csv_dir: task_table.config},
    filename=task_table.csv_name,
)
print(f"Task train={len(trainset2[0][0])}, val={len(valset2[0][0])}, test={len(testset2[0][0])}, classes={task_table.num_classes}")

model_downstream = transtab.build_classifier(
    categorical_columns=cat_cols2,
    numerical_columns=num_cols2,
    binary_columns=bin_cols2,
    num_class=task_table.num_classes,
    checkpoint=args.pretrain_dir,
    device=args.device,
)
model_downstream.update({"cat": cat_cols2, "num": num_cols2, "bin": bin_cols2, "num_class": task_table.num_classes})
transtab.train(
    model_downstream,
    trainset2,
    valset2,
    num_epoch=args.num_epoch_finetune,
    eval_metric="val_loss",
    eval_less_is_better=True,
    output_dir=args.ckpt_dir,
)

x_test, y_test = testset2[0]
ypred_prob = transtab.predict(model_downstream, x_test, y_test)
preds = np.argmax(ypred_prob, axis=1)
try:
    auc_score = roc_auc_score(y_test, ypred_prob, multi_class="ovr")
except ValueError:
    auc_score = float("nan")
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted", zero_division=0)
recall = recall_score(y_test, preds, average="weighted", zero_division=0)
f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

print("\nTest Performance:")
print(f"  AUC:       {auc_score:.4f}")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, digits=4, zero_division=0))

runtime = time.perf_counter() - run_start
print(f"Runtime: {runtime:.2f}s")

if args.save_results:
    output = {
        "model": "transtab_transfer_cl",
        "task": "classification",
        "dataset_name": args.dataset,
        "table_idx": args.table_idx,
        "dataset": task_table.tag,
        "auxiliary_dataset_name": args.aux_dataset,
        "auxiliary_table_idx": args.aux_table_idx,
        "auxiliary_dataset": aux_table.tag,
        "seed": args.seed,
        "num_epoch_pretrain": args.num_epoch_pretrain,
        "num_epoch_finetune": args.num_epoch_finetune,
        "device": args.device,
        "ckpt_dir": args.ckpt_dir,
        "pretrain_dir": args.pretrain_dir,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runtime": runtime,
        "metrics": {
            "auc": float(auc_score),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
    }
    os.makedirs(os.path.dirname(args.save_results) or ".", exist_ok=True)
    with open(args.save_results, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved -> {args.save_results}")
