"""
Tabular Neural Networks for regression (GACars price regression).

Task    : price_in_euro regression (continuous)
Tables  : 4=German Reg, 5=Australian Reg, 6=DA Reg, 7=FA Reg
Loss    : MSE
Metrics : RMSE, MAE, R²  (mean ± std across runs)
Models  : fttransformer, tabtransformer, excelformer, saint, tromptnet

tnns_models.py is shared with the classification version (unchanged).
"""

import sys
import os
import os.path as osp

_THIS_DIR    = osp.abspath(osp.dirname(__file__))
PROJECT_ROOT = osp.abspath(osp.join(_THIS_DIR, ".."))
LIB_ROOT     = osp.abspath(osp.join(PROJECT_ROOT, "lib"))
for _p in reversed([_THIS_DIR, LIB_ROOT, PROJECT_ROOT]):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import itertools
import json
import random
import glob
import fcntl
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lib.rllm.transforms.table_transforms import DefaultTableTransform
from lib.rllm.datasets.lakemlb.gacars import GACarsDataset
from utils import (
    set_seed, parse_list_of_ints, parse_list_of_floats, get_device,
    get_batch, to_device, save_model, load_model, print_grid_config
)
from tnns_models import create_model, AVAILABLE_MODELS

_DATA_CACHE: Dict = {}

TABLE_NAMES = {
    4: "german_reg",
    5: "australian_reg",
    6: "gacars_da_reg",
    7: "gacars_fa_reg",
}

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="fttransformer",
                    choices=AVAILABLE_MODELS)
parser.add_argument("--table_idx", type=int, default=4,
                    choices=[4, 5, 6, 7],
                    help="GACarsDataset table: 4=German Reg, 5=Australian Reg, "
                         "6=DA Reg, 7=FA Reg")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

# Grid search
parser.add_argument("--grid", action="store_true", default=False)
parser.add_argument("--grid_hidden", type=str, default="32,64,128")
parser.add_argument("--grid_layers", type=str, default="2,3,4")
parser.add_argument("--grid_lr", type=str, default="1e-3,1e-4,5e-4")
parser.add_argument("--grid_wd", type=str, default="1e-4,1e-3,5e-4")
parser.add_argument("--grid_bs", type=str, default="512")
parser.add_argument("--grid_epochs", type=int, default=100)
parser.add_argument("--grid_patience", type=int, default=10)

# Parallel grid search
parser.add_argument("--task_id", type=int, default=0)
parser.add_argument("--num_tasks", type=int, default=1)
parser.add_argument("--grid_output_dir", type=str, default=None)
parser.add_argument("--merge_results", action="store_true", default=False)
parser.add_argument("--skip_final_train", action="store_true", default=False)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument("--save_results", type=str, default=None)

args = parser.parse_args()

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = osp.join(PROJECT_ROOT, "data")
RESULTS_DIR = osp.join(PROJECT_ROOT, "results", "tnns_reg")
CKPT_DIR    = osp.join(RESULTS_DIR, "checkpoints")
for _d in (RESULTS_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)

table_name = TABLE_NAMES.get(args.table_idx, f"table_{args.table_idx}")
if args.save_results is None and args.num_runs > 1:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(
        RESULTS_DIR,
        f"{args.model}_{table_name}_{args.num_runs}runs_{ts}.json"
    )
    print(f"Results will be auto-saved to: {args.save_results}")


# ── dataset ───────────────────────────────────────────────────────────────────
def build_dataset(emb_dim: int, gpu_device: torch.device):
    """Load GACarsDataset (regression table) on CPU; batches moved to GPU during training."""
    global _DATA_CACHE
    cache_key = f"{args.table_idx}_{emb_dim}"

    if cache_key in _DATA_CACHE:
        cached = _DATA_CACHE[cache_key]
        print(f"Using cached dataset (table={table_name}, emb_dim={emb_dim})")
        return cached["data"], cached["train_idx"], cached["val_idx"], cached["test_idx"]

    print(f"Loading GACarsDataset[{args.table_idx}] ({table_name}, emb_dim={emb_dim}) …")

    lock_file = osp.join(DATA_DIR, ".data_load_reg.lock")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(lock_file, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            table_transform = DefaultTableTransform(out_dim=emb_dim)
            dataset = GACarsDataset(
                cached_dir=DATA_DIR,
                force_reload=False,
                transform=table_transform,
                device=torch.device("cpu"),
            )
            data = dataset.data_list[args.table_idx]
            # Regression: keep labels as float32
            data.y = data.y.float()

            train_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1)
            val_idx   = torch.nonzero(data.val_mask,   as_tuple=False).view(-1)
            test_idx  = torch.nonzero(data.test_mask,  as_tuple=False).view(-1)

            _DATA_CACHE[cache_key] = {
                "data": data, "train_idx": train_idx,
                "val_idx": val_idx, "test_idx": test_idx,
            }
            print(f"Loaded: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
            y_all = data.y.numpy()
            print(f"  y range=[{y_all.min():.4f}, {y_all.max():.4f}]  "
                  f"mean={y_all.mean():.4f}  std={y_all.std():.4f}")
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    return data, train_idx, val_idx, test_idx


# ── training & evaluation ─────────────────────────────────────────────────────
def train_epoch(model, optimizer, data, train_indices, batch_size, device,
                gradient_accumulation_steps=1) -> float:
    model.train()
    total_loss = 0.0
    perm = train_indices[torch.randperm(train_indices.size(0))]

    optimizer.zero_grad()
    accum_step = 0

    for start in range(0, perm.size(0), batch_size):
        batch_idx = perm[start:start + batch_size]
        batch = to_device(get_batch(data, batch_idx), device)

        pred  = model(batch)          # shape (B,) for regression
        loss  = F.mse_loss(pred, batch.y)
        (loss / gradient_accumulation_steps).backward()

        accum_step += 1
        if accum_step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * batch_idx.size(0)
        del batch, pred, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if accum_step % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / train_indices.size(0)   # mean MSE over samples


@torch.no_grad()
def evaluate(model, data, indices, batch_size, device) -> Dict:
    """Return RMSE, MAE, R² for the given split."""
    model.eval()
    preds_all, y_all = [], []

    for start in range(0, indices.size(0), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = to_device(get_batch(data, batch_idx), device)
        preds_all.append(model(batch).cpu().numpy())
        y_all.append(batch.y.cpu().numpy())
        del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    preds = np.concatenate(preds_all)
    y     = np.concatenate(y_all)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae":  float(mean_absolute_error(y, preds)),
        "r2":   float(r2_score(y, preds)),
    }


# ── single training run ───────────────────────────────────────────────────────
def run_training(config, epochs, patience, device, model_name,
                 save_path=None, seed=None,
                 eval_test_each_epoch=False, verbose=False,
                 gradient_accumulation_steps=1):
    if seed is not None:
        set_seed(seed)

    hidden_dim = config["hidden_dim"]
    data, train_idx, val_idx, test_idx = build_dataset(hidden_dim, device)
    model = create_model(model_name, config, data, device, task="regression")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )

    best_val_rmse = float("inf")
    best_test_metrics: Dict = {}
    best_epoch  = 0
    no_improve  = 0
    best_state  = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, optimizer, data, train_idx,
            config["batch_size"], device, gradient_accumulation_steps
        )
        val_m = evaluate(model, data, val_idx, config["batch_size"], device)

        if eval_test_each_epoch:
            test_m = evaluate(model, data, test_idx, config["batch_size"], device)
        else:
            test_m = {}

        if verbose:
            msg = (f"[Epoch {epoch:03d}] train_mse={train_loss:.4f} | "
                   f"val_rmse={val_m['rmse']:.4f} mae={val_m['mae']:.4f} r2={val_m['r2']:.4f}")
            if eval_test_each_epoch:
                msg += f" | test_rmse={test_m['rmse']:.4f}"
            print(msg)

        # Early stopping: minimize val RMSE
        if val_m["rmse"] < best_val_rmse:
            best_val_rmse  = val_m["rmse"]
            best_test_metrics = test_m
            best_epoch     = epoch
            no_improve     = 0
            if save_path:
                save_model(model, save_path)
            else:
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"[EarlyStop] best val RMSE={best_val_rmse:.4f} @ epoch {best_epoch}")
                break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # If test was not evaluated each epoch, compute it now on best checkpoint
    if not eval_test_each_epoch:
        if save_path and os.path.exists(save_path):
            load_model(model, save_path, device)
        elif best_state is not None:
            model.load_state_dict(best_state)
        best_test_metrics = evaluate(model, data, test_idx, config["batch_size"], device)

    return {
        "best_val_rmse": best_val_rmse,
        "best_epoch":    best_epoch,
        **{f"test_{k}": v for k, v in best_test_metrics.items()},
    }


# ── multi-run ─────────────────────────────────────────────────────────────────
def run_multiple_experiments(
    config: Dict, model_name: str, device: torch.device,
    num_runs: int, base_seed: int, epochs: int, patience: int,
    gradient_accumulation_steps: int = 1,
) -> List[Dict]:
    results = []
    print(f"\n{'='*60}")
    print(f"Running {num_runs} experiments …")
    print(f"{'='*60}\n")

    for run_id in range(num_runs):
        seed = base_seed if run_id == 0 else base_seed + random.randint(1, 10000)
        print(f"\n[Run {run_id+1}/{num_runs}]  seed={seed}")
        print("-" * 40)

        r = run_training(
            config=config, epochs=epochs, patience=patience,
            device=device, model_name=model_name,
            save_path=None, seed=seed,
            eval_test_each_epoch=False, verbose=False,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        run_result = {
            "run_id":        run_id + 1,
            "seed":          seed,
            "best_val_rmse": r["best_val_rmse"],
            "test_rmse":     r["test_rmse"],
            "test_mae":      r["test_mae"],
            "test_r2":       r["test_r2"],
            "best_epoch":    r["best_epoch"],
        }
        results.append(run_result)
        print(f"  Val  RMSE={r['best_val_rmse']:.4f}")
        print(f"  Test RMSE={r['test_rmse']:.4f}  MAE={r['test_mae']:.4f}  R²={r['test_r2']:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ── statistics ────────────────────────────────────────────────────────────────
def compute_statistics(results: List[Dict]) -> Dict:
    metrics = ["best_val_rmse", "test_rmse", "test_mae", "test_r2"]
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
    print(f"  Val  RMSE={fmt('best_val_rmse')}")
    print(f"  Test RMSE={fmt('test_rmse')}  MAE={fmt('test_mae')}  R²={fmt('test_r2')}")
    print(f"{'='*60}")


def save_results_to_file(results, stats, config, model_name, save_path):
    output = {
        "model":    model_name,
        "task":     "price_regression",
        "dataset":  f"GACarsDataset / {table_name}",
        "config":   config,
        "num_runs": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "individual_runs": results,
        "statistics":      stats,
    }
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {save_path}")


# ── grid search ───────────────────────────────────────────────────────────────
def get_all_combinations(space: Dict) -> List[Tuple]:
    return list(itertools.product(
        space["hidden_dim"], space["layers"],
        space["lr"], space["wd"], space["batch_size"]
    ))


def get_task_combinations(all_combinations, task_id, num_tasks):
    total    = len(all_combinations)
    per_task = total // num_tasks
    remainder = total % num_tasks
    if task_id < remainder:
        start = task_id * (per_task + 1)
        end   = start + per_task + 1
    else:
        start = task_id * per_task + remainder
        end   = start + per_task
    return all_combinations[start:end]


def save_grid_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Grid results saved → {output_path}")


def load_grid_results(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_all_grid_results(grid_output_dir, model_name):
    pattern = osp.join(grid_output_dir, f"{model_name}_grid_task_*.json")
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        raise FileNotFoundError(f"No result files: {pattern}")
    print(f"Found {len(result_files)} result files to merge")

    all_results = []
    for f in result_files:
        print(f"  Loading: {f}")
        all_results.extend(load_grid_results(f))

    # Minimize val_rmse
    best_result = min(all_results, key=lambda x: x["best_val"])
    best_cfg    = best_result["config"]

    merged_path = osp.join(grid_output_dir, f"{model_name}_grid_merged.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name, "table": table_name,
            "total_combinations": len(all_results),
            "best_config": best_cfg,
            "best_val_rmse": best_result["best_val"],
            "best_test_rmse": best_result["best_test"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "all_results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Merged results saved → {merged_path}")
    return best_cfg, all_results


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"Model : {args.model.upper()}")
    print(f"Table : {table_name}  (index={args.table_idx})")
    print(f"Device: {device}")
    print("Task  : Regression (MSE loss)\n")

    if args.grid_output_dir is None:
        args.grid_output_dir = osp.join(RESULTS_DIR, "grid_search")
    os.makedirs(args.grid_output_dir, exist_ok=True)

    # ── merge mode ───────────────────────────────────────────────────────────
    if args.merge_results:
        print("Merging parallel grid search results …")
        best_cfg, all_results = merge_all_grid_results(args.grid_output_dir, args.model)
        best = min(all_results, key=lambda x: x["best_val"])
        print(f"Best config: {best_cfg}")
        print(f"  val RMSE={best['best_val']:.4f}  test RMSE={best['best_test']:.4f}")

        if not args.skip_final_train and args.num_runs > 1:
            results = run_multiple_experiments(
                config=best_cfg, model_name=args.model, device=device,
                num_runs=args.num_runs, base_seed=args.seed,
                epochs=args.epochs, patience=args.patience,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            stats = compute_statistics(results)
            print_statistics(stats, args.num_runs)
            if args.save_results:
                save_results_to_file(results, stats, best_cfg, args.model, args.save_results)
        return

    # ── grid search mode ─────────────────────────────────────────────────────
    if args.grid:
        space = {
            "hidden_dim": parse_list_of_ints(args.grid_hidden),
            "layers":     parse_list_of_ints(args.grid_layers),
            "lr":         parse_list_of_floats(args.grid_lr),
            "wd":         parse_list_of_floats(args.grid_wd),
            "batch_size": parse_list_of_ints(args.grid_bs),
        }
        all_combinations = get_all_combinations(space)
        total = len(all_combinations)

        if args.num_tasks > 1:
            task_combinations = get_task_combinations(all_combinations, args.task_id, args.num_tasks)
            print(f"Parallel Grid Search – Task {args.task_id+1}/{args.num_tasks}")
            print(f"  Total: {total}  This task: {len(task_combinations)}")
        else:
            task_combinations = all_combinations
            print_grid_config(space, total)

        task_results = []
        best_cfg, best_val_rmse, best_test_rmse = None, float("inf"), float("inf")

        for comb_idx, (hd, ly, lr, wd, bs) in enumerate(task_combinations):
            global_idx = all_combinations.index((hd, ly, lr, wd, bs)) + 1
            cfg = {"hidden_dim": hd, "layers": ly, "lr": lr, "wd": wd, "batch_size": bs}

            if args.num_tasks > 1:
                print(f"\n[Task {args.task_id+1}] [{comb_idx+1}/{len(task_combinations)}] "
                      f"(Global {global_idx}/{total}) {cfg}")
            else:
                print(f"\n[Grid {global_idx}/{total}] {cfg}")

            r = run_training(
                cfg, epochs=args.grid_epochs, patience=args.grid_patience,
                device=device, model_name=args.model, save_path=None,
                seed=args.seed, eval_test_each_epoch=False,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            print(f"  val_rmse={r['best_val_rmse']:.4f}  "
                  f"test_rmse={r['test_rmse']:.4f}  @ epoch {r['best_epoch']}")

            task_results.append({
                "config":     cfg,
                "best_val":   r["best_val_rmse"],
                "best_test":  r["test_rmse"],
                "best_epoch": r["best_epoch"],
                "global_idx": global_idx,
            })

            if r["best_val_rmse"] < best_val_rmse:
                best_val_rmse  = r["best_val_rmse"]
                best_test_rmse = r["test_rmse"]
                best_cfg = cfg

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save task results (parallel mode)
        if args.num_tasks > 1:
            task_out = osp.join(
                args.grid_output_dir,
                f"{args.model}_grid_task_{args.task_id:03d}.json"
            )
            save_grid_results(task_results, task_out)
            print(f"\nTask {args.task_id+1} done. "
                  f"Best val RMSE={best_val_rmse:.4f} test RMSE={best_test_rmse:.4f}")
            return

        print(f"\nBest config: {best_cfg}")
        print(f"val RMSE={best_val_rmse:.4f}  test RMSE={best_test_rmse:.4f}")
        cfg = best_cfg

    # ── single-config mode ───────────────────────────────────────────────────
    else:
        cfg = {
            "hidden_dim": 32,
            "layers":     3,
            "lr":         args.lr,
            "wd":         args.wd,
            "batch_size": args.batch_size,
        }
        print(f"Config: {cfg}\n")

    # ── multi-run final evaluation ────────────────────────────────────────────
    if args.num_runs > 1:
        results = run_multiple_experiments(
            config=cfg, model_name=args.model, device=device,
            num_runs=args.num_runs, base_seed=args.seed,
            epochs=args.epochs, patience=args.patience,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        stats = compute_statistics(results)
        print_statistics(stats, args.num_runs)
        if args.save_results:
            save_results_to_file(results, stats, cfg, args.model, args.save_results)

        best_run = min(results, key=lambda x: x["val_rmse"] if "val_rmse" in x else x["best_val_rmse"])
        print(f"\nBest run: #{best_run['run_id']}  (seed={best_run['seed']})")
        print(f"  → test RMSE={best_run['test_rmse']:.4f}  "
              f"MAE={best_run['test_mae']:.4f}  R²={best_run['test_r2']:.4f}")

    else:
        # Single run
        r = run_training(
            config=cfg, epochs=args.epochs, patience=args.patience,
            device=device, model_name=args.model, save_path=None,
            seed=args.seed, eval_test_each_epoch=True, verbose=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        print(f"\nResult:")
        print(f"  Val  RMSE={r['best_val_rmse']:.4f}  @ epoch {r['best_epoch']}")
        print(f"  Test RMSE={r['test_rmse']:.4f}  MAE={r['test_mae']:.4f}  R²={r['test_r2']:.4f}")


if __name__ == "__main__":
    main()
