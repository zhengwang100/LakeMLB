"""TransTab single-table classification for LakeMLB tables."""
import argparse
import json
import os
import os.path as osp
import random
import secrets
import shutil
import sys
import time
from datetime import datetime

sys.path.insert(0, osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, osp.join(osp.dirname(__file__), "..", "lib"))

import numpy as np
import torch
import transtab
from sklearn.metrics import accuracy_score, f1_score

from transtab_lakemlb_utils import prepare_table


SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
RESULTS_DIR = osp.abspath(osp.join(SCRIPT_DIR, "..", "results", "transtab_cls"))
os.makedirs(RESULTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="TransTab single-table classification")
parser.add_argument("--dataset", type=str, default="nnstocks")
parser.add_argument("--table_idx", type=int, default=4)
parser.add_argument("--work_dir", type=str, default=None)
parser.add_argument("--ckpt_dir", type=str, default="./ckpt_transtab_cls")
parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_results", type=str, default=None)
args = parser.parse_args()

script_start = time.perf_counter()
if args.save_results is None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(
        RESULTS_DIR, f"transtab_single_{args.dataset}_table{args.table_idx}_{args.num_runs}runs_{ts}.json"
    )

print(f"Dataset : {args.dataset}[{args.table_idx}]")
print(f"Device  : {args.device}")
print(f"Epochs  : {args.num_epoch}  Patience: {args.patience}")
print(f"Runs    : {args.num_runs}")

all_runs = []

for run_id in range(args.num_runs):
    seed = args.seed if args.num_runs == 1 else secrets.randbelow(2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_start = time.perf_counter()
    run_work_dir = args.work_dir or osp.join(args.ckpt_dir, f"seed_{seed}", "data")
    run_ckpt = osp.join(args.ckpt_dir, f"seed_{seed}", "checkpoint")
    os.makedirs(run_ckpt, exist_ok=True)

    table = prepare_table(
        args.dataset, args.table_idx, run_work_dir, "task", seed,
        require_target=True, use_target=True,
    )
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(
        [table.csv_dir],
        dataset_config={table.csv_dir: table.config},
        filename=table.csv_name,
    )
    x_test, y_test = testset[0]
    print(
        f"Run {run_id+1}: table={table.tag}, train={len(trainset[0][0])}, "
        f"val={len(valset[0][0])}, test={len(x_test)}, classes={table.num_classes}"
    )

    model = transtab.build_classifier(
        categorical_columns=cat_cols,
        numerical_columns=num_cols,
        binary_columns=bin_cols,
        num_class=table.num_classes,
        num_layer=4,
        device=args.device,
    )
    transtab.train(
        model, trainset, valset,
        num_epoch=args.num_epoch,
        eval_metric="val_loss",
        eval_less_is_better=True,
        output_dir=run_ckpt,
    )

    model.load(run_ckpt)
    ypred_prob = transtab.predict(model, x_test, y_test)
    preds = np.argmax(ypred_prob, axis=1)
    test_acc = accuracy_score(y_test, preds)
    test_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    runtime = time.perf_counter() - run_start

    all_runs.append({
        "run_id": run_id + 1,
        "seed": seed,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "runtime": runtime,
        "ckpt_dir": run_ckpt,
        "work_dir": run_work_dir,
    })
    print(
        f"  Run {run_id+1}/{args.num_runs}: test_acc={test_acc:.4f}  "
        f"test_f1={test_f1:.4f}  seed={seed} time={runtime:.2f}s"
    )

    shutil.rmtree(run_ckpt, ignore_errors=True)

vals = [r["test_acc"] for r in all_runs]
f1_vals = [r["test_f1"] for r in all_runs]
runtimes = [r["runtime"] for r in all_runs]
stats = {
    "test_acc_mean": float(np.mean(vals)),
    "test_acc_std": float(np.std(vals)),
    "test_acc_min": float(np.min(vals)),
    "test_acc_max": float(np.max(vals)),
    "test_f1_mean": float(np.mean(f1_vals)),
    "test_f1_std": float(np.std(f1_vals)),
    "test_f1_min": float(np.min(f1_vals)),
    "test_f1_max": float(np.max(f1_vals)),
    "runtime_mean": float(np.mean(runtimes)),
    "runtime_std": float(np.std(runtimes)),
    "runtime_min": float(np.min(runtimes)),
    "runtime_max": float(np.max(runtimes)),
    "total_runtime": time.perf_counter() - script_start,
}
print(f"\nTest Acc: {stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.4f}")

output = {
    "model": "transtab_single",
    "task": "classification",
    "dataset_name": args.dataset,
    "table_idx": args.table_idx,
    "dataset": table.tag if all_runs else f"{args.dataset}_table{args.table_idx}",
    "num_runs": len(all_runs),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "individual_runs": all_runs,
    "statistics": stats,
    "seed": all_runs[0]["seed"] if len(all_runs) == 1 else args.seed,
    "runtime": stats["total_runtime"],
    "metrics": {
        "accuracy": stats["test_acc_mean"],
        "f1": stats["test_f1_mean"],
    },
    "ckpt_dir": args.ckpt_dir,
    "pretrain_dir": None,
}
os.makedirs(osp.dirname(args.save_results) or ".", exist_ok=True)
with open(args.save_results, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Results saved -> {args.save_results}")
