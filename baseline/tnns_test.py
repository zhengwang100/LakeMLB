
import sys
import os
import os.path as osp

# Minimal sys.path bootstrap so this script can be run directly.
_THIS_DIR = osp.abspath(osp.dirname(__file__))
PROJECT_ROOT = osp.abspath(osp.join(_THIS_DIR, ".."))
LIB_ROOT = osp.abspath(osp.join(PROJECT_ROOT, "lib"))
REPO_ROOT = osp.abspath(osp.join(_THIS_DIR, "../.."))  # Repo root (/home/pfy/devrepo/lakemlb0204)

# Precedence (highest first): this dir -> repo root -> lib -> project root
for _p in reversed([_THIS_DIR, REPO_ROOT, LIB_ROOT, PROJECT_ROOT]):
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

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from lib.rllm.transforms.table_transforms import DefaultTableTransform
from datasets.mstraffic_datasets import MSTrafficDataset
from utils import (
    set_seed, parse_list_of_ints, parse_list_of_floats, get_device,
    get_batch, to_device, save_model, load_model, print_grid_config
)
from tnns_models import create_model, AVAILABLE_MODELS

_DATA_CACHE = {}


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="fttransformer",
                    choices=AVAILABLE_MODELS,
                    help="Model to use")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of gradient accumulation steps (larger effective batch)")

parser.add_argument("--grid", action="store_true", default=False)
parser.add_argument("--grid_hidden", type=str, default="32,64,128")
parser.add_argument("--grid_layers", type=str, default="2,3,4")
parser.add_argument("--grid_lr", type=str, default="1e-3,1e-4,5e-4")
parser.add_argument("--grid_wd", type=str, default="1e-4,1e-3,5e-4")
parser.add_argument("--grid_bs", type=str, default="512")
parser.add_argument("--grid_epochs", type=int, default=100)
parser.add_argument("--grid_patience", type=int, default=10)

# Parallel grid search arguments
parser.add_argument("--task_id", type=int, default=0,
                    help="Task ID for parallel grid search (0-indexed)")
parser.add_argument("--num_tasks", type=int, default=1,
                    help="Total number of parallel tasks")
parser.add_argument("--grid_output_dir", type=str, default=None,
                    help="Directory to save intermediate grid search results")
parser.add_argument("--merge_results", action="store_true", default=False,
                    help="Merge results from all parallel tasks")
parser.add_argument("--skip_final_train", action="store_true", default=False,
                    help="Skip final training after grid search")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--num_runs", type=int, default=5,
                    help="Number of runs with different seeds")
parser.add_argument("--save_results", type=str, default=None,
                    help="Path to save results (JSON format)")

args = parser.parse_args()


# ======================== Paths ========================
DATA_DIR = osp.join(PROJECT_ROOT, "data")
RESULTS_DIR = osp.join(PROJECT_ROOT, "results")
CKPT_DIR = osp.join(RESULTS_DIR, "checkpoints")
for _d in (RESULTS_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)
if args.save_results is None and args.num_runs > 1:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_results = osp.join(RESULTS_DIR, f"{args.model}_{args.num_runs}runs_{timestamp}.json")
    print(f"Results will be auto-saved to: {args.save_results}")


# ======================== Dataset (memory-efficient) ========================
def build_dataset(emb_dim: int, gpu_device: torch.device):
    """Load dataset on CPU, only move batches to GPU during training.
    
    Key changes:
    - Dataset kept on CPU
    - Only batches moved to GPU during training
    - Cache stored on CPU to save GPU memory
    """
    global _DATA_CACHE
    
    cache_key = f"{emb_dim}"  # does not depend on device
    
    if cache_key in _DATA_CACHE:
        cached = _DATA_CACHE[cache_key]
        print(f"Using cached dataset (emb_dim={emb_dim}) from CPU")
        return cached["data"], cached["train_idx"], cached["val_idx"], cached["test_idx"]
    
    print(f"Loading dataset (emb_dim={emb_dim}) to CPU...")
    
    lock_file = osp.join(DATA_DIR, ".data_load.lock")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(lock_file, 'w') as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            table_transform = DefaultTableTransform(out_dim=emb_dim)

            dataset = MSTrafficDataset(
                cached_dir=DATA_DIR,
                force_reload=True,
                transform=table_transform,
                device=torch.device('cpu')
            )
            data = dataset.data_list[0]
            data.y = data.y.long()

            train_indices = torch.nonzero(data.train_mask, as_tuple=False).view(-1)
            val_indices = torch.nonzero(data.val_mask, as_tuple=False).view(-1)
            test_indices = torch.nonzero(data.test_mask, as_tuple=False).view(-1)
            
            _DATA_CACHE[cache_key] = {
                "data": data,
                "train_idx": train_indices,
                "val_idx": val_indices,
                "test_idx": test_indices
            }
            
            print(f"Dataset loaded to CPU. Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    return data, train_indices, val_indices, test_indices


# ======================== Training & evaluation (memory-efficient) ========================
def train_epoch(model, optimizer, data, train_indices, batch_size, device, 
                gradient_accumulation_steps=1) -> float:
    """Train one epoch with gradient accumulation support."""
    model.train()
    total_loss = 0.0
    perm = train_indices[torch.randperm(train_indices.size(0))]
    
    optimizer.zero_grad()
    accum_step = 0
    
    for start in range(0, perm.size(0), batch_size):
        batch_idx = perm[start:start + batch_size]
        
        batch = get_batch(data, batch_idx)
        batch = to_device(batch, device)
        
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y)
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        accum_step += 1
        if accum_step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * batch_idx.size(0) * gradient_accumulation_steps
        
        del batch, logits, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    if accum_step % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / train_indices.size(0)


@torch.no_grad()
def evaluate(model, data, indices, batch_size, device) -> float:
    """Compute accuracy with memory-efficient batching."""
    model.eval()
    correct = 0
    
    for start in range(0, indices.size(0), batch_size):
        batch_idx = indices[start:start + batch_size]
        
        batch = get_batch(data, batch_idx)
        batch = to_device(batch, device)
        
        preds = model(batch).argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        
        del batch, preds
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return correct / indices.size(0)


def run_training(config, epochs, patience, device, model_name, save_path=None, seed=None,
                 eval_test_each_epoch=True, verbose=False, gradient_accumulation_steps=1):
    """Run training with early stopping and memory management."""
    if seed is not None:
        set_seed(seed)

    hidden_dim = config["hidden_dim"]
    data, train_idx, val_idx, test_idx = build_dataset(hidden_dim, device)
    
    model = create_model(model_name, config, data, device, task="classification")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["wd"]
    )

    best_val, best_test, best_epoch = 0.0, -1.0, 0
    no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, optimizer, data, train_idx, 
                                config["batch_size"], device, gradient_accumulation_steps)
        val_acc = evaluate(model, data, val_idx, config["batch_size"], device)
        
        if eval_test_each_epoch:
            test_acc = evaluate(model, data, test_idx, config["batch_size"], device)
        else:
            test_acc = -1.0

        if verbose:
            msg = f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | val_acc={val_acc:.4f}"
            if eval_test_each_epoch:
                msg += f" | test_acc={test_acc:.4f}"
            print(msg)

        if val_acc > best_val:
            best_val, best_test, best_epoch = val_acc, test_acc, epoch
            no_improve = 0
            if save_path:
                save_model(model, save_path)
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"[EarlyStop] No improvement for {patience} epochs. "
                          f"Best val acc={best_val:.4f} @ epoch {best_epoch}")
                break
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not eval_test_each_epoch:
        if save_path and os.path.exists(save_path):
            load_model(model, save_path, device)
        elif best_state is not None:
            model.load_state_dict(best_state)
        best_test = evaluate(model, data, test_idx, config["batch_size"], device)

    return {
        "best_val": best_val,
        "best_test": best_test,
        "best_epoch": best_epoch,
    }


# ======================== Multiple Runs ========================
def run_multiple_experiments(
    config: Dict,
    model_name: str,
    device: torch.device,
    num_runs: int,
    base_seed: int,
    epochs: int,
    patience: int,
    gradient_accumulation_steps: int = 1,
) -> List[Dict]:
    """Run experiments multiple times with different seeds."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running {num_runs} experiments with different seeds...")
    print(f"{'='*60}\n")
    
    for run_id in range(num_runs):
        if run_id == 0:
            seed = base_seed
        else:
            seed = base_seed + random.randint(1, 10000)
        print(f"\n[Run {run_id+1}/{num_runs}]")
        print("-" * 40)
        
        result = run_training(
            config=config,
            epochs=epochs,
            patience=patience,
            device=device,
            model_name=model_name,
            save_path=None,
            seed=seed,
            eval_test_each_epoch=False,
            verbose=False,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        run_result = {
            "run_id": run_id + 1,
            "seed": seed,
            "best_val_acc": result["best_val"],
            "test_acc": result["best_test"],
            "best_epoch": result["best_epoch"],
        }
        results.append(run_result)
        
        print(f"  Val Acc: {run_result['best_val_acc']:.4f}")
        print(f"  Test Acc: {run_result['test_acc']:.4f}")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute mean and std of multiple runs."""
    metrics = ["best_val_acc", "test_acc"]
    stats = {}
    
    for metric in metrics:
        values = [r[metric] for r in results]
        stats[f"{metric}_mean"] = float(np.mean(values))
        stats[f"{metric}_std"] = float(np.std(values))
        stats[f"{metric}_min"] = float(np.min(values))
        stats[f"{metric}_max"] = float(np.max(values))
    
    return stats


def save_results_to_file(
    results: List[Dict],
    stats: Dict,
    config: Dict,
    model_name: str,
    save_path: str
):
    """Save results to JSON file."""
    output = {
        "model": model_name,
        "task": "classification",
        "config": config,
        "num_runs": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "individual_runs": results,
        "statistics": stats,
    }
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {save_path}")


def print_statistics(stats: Dict, num_runs: int):
    """Print statistics in a nice format."""
    print("\n" + "="*60)
    print(f"Summary Statistics ({num_runs} runs):")
    print("="*60)
    print(f"Val Acc:  {stats['best_val_acc_mean']:.4f} ± {stats['best_val_acc_std']:.4f}")
    print(f"Test Acc: {stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.4f}")
    print("="*60)


# ======================== Grid Search Functions ========================
def get_all_combinations(space: Dict) -> List[Tuple]:
    """Generate all parameter combinations."""
    return list(itertools.product(
        space["hidden_dim"], space["layers"],
        space["lr"], space["wd"], space["batch_size"]
    ))


def get_task_combinations(all_combinations: List[Tuple], task_id: int, num_tasks: int) -> List[Tuple]:
    """Get combinations assigned to this task."""
    total = len(all_combinations)
    per_task = total // num_tasks
    remainder = total % num_tasks
    
    if task_id < remainder:
        start = task_id * (per_task + 1)
        end = start + per_task + 1
    else:
        start = task_id * per_task + remainder
        end = start + per_task
    
    return all_combinations[start:end]


def save_grid_results(results: List[Dict], output_path: str):
    """Save grid search results."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Grid results saved to: {output_path}")


def load_grid_results(input_path: str) -> List[Dict]:
    """Load grid search results."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_all_grid_results(grid_output_dir: str, model_name: str) -> Tuple[Dict, List[Dict]]:
    """Merge results from all parallel tasks."""
    pattern = osp.join(grid_output_dir, f"{model_name}_grid_task_*.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        raise FileNotFoundError(f"No result files found matching: {pattern}")
    
    print(f"Found {len(result_files)} result files to merge")
    
    all_results = []
    for f in sorted(result_files):
        print(f"  Loading: {f}")
        task_results = load_grid_results(f)
        all_results.extend(task_results)
    
    print(f"Total combinations: {len(all_results)}")
    
    best_result = max(all_results, key=lambda x: x["best_val"])
    best_cfg = best_result["config"]
    
    merged_path = osp.join(grid_output_dir, f"{model_name}_grid_merged.json")
    merged_output = {
        "model": model_name,
        "total_combinations": len(all_results),
        "best_config": best_cfg,
        "best_val_acc": best_result["best_val"],
        "best_test_acc": best_result["best_test"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "all_results": all_results
    }
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged_output, f, indent=2, ensure_ascii=False)
    print(f"Merged results saved to: {merged_path}")
    
    return best_cfg, all_results


# ======================== Main ========================
def main():
    set_seed(args.seed)
    device = get_device(args.device)
    
    print(f"Using model: {args.model.upper()}")
    print(f"Device: {device}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print("Task: Classification (Memory-Efficient Version)\n")
    
    if args.grid_output_dir is None:
        args.grid_output_dir = osp.join(RESULTS_DIR, "grid_search")
    os.makedirs(args.grid_output_dir, exist_ok=True)

    # ==================== Merge Results Mode ====================
    if args.merge_results:
        print("="*60)
        print("Merging parallel grid search results...")
        print("="*60)
        
        best_cfg, all_results = merge_all_grid_results(args.grid_output_dir, args.model)
        
        print(f"\nBest config found: {best_cfg}")
        best_result = max(all_results, key=lambda x: x["best_val"])
        print(f"Best val acc: {best_result['best_val']:.4f}")
        print(f"Best test acc: {best_result['best_test']:.4f}")
        
        if not args.skip_final_train:
            print("\n" + "="*60)
            print("Final training with best config")
            print("="*60)
            
            if args.num_runs > 1:
                results = run_multiple_experiments(
                    config=best_cfg,
                    model_name=args.model,
                    device=device,
                    num_runs=args.num_runs,
                    base_seed=args.seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    gradient_accumulation_steps=args.gradient_accumulation_steps
                )
                
                stats = compute_statistics(results)
                print_statistics(stats, args.num_runs)
                
                if args.save_results:
                    save_results_to_file(results, stats, best_cfg, args.model, args.save_results)
        return

    # ==================== Grid Search Mode ====================
    if args.grid:
        space = {
            "hidden_dim": parse_list_of_ints(args.grid_hidden),
            "layers": parse_list_of_ints(args.grid_layers),
            "lr": parse_list_of_floats(args.grid_lr),
            "wd": parse_list_of_floats(args.grid_wd),
            "batch_size": parse_list_of_ints(args.grid_bs),
        }
        
        all_combinations = get_all_combinations(space)
        total = len(all_combinations)
        
        if args.num_tasks > 1:
            task_combinations = get_task_combinations(all_combinations, args.task_id, args.num_tasks)
            print("="*60)
            print(f"Parallel Grid Search - Task {args.task_id + 1}/{args.num_tasks}")
            print(f"Total combinations: {total}")
            print(f"This task's combinations: {len(task_combinations)}")
            print("="*60)
        else:
            task_combinations = all_combinations
            print_grid_config(space, total)

        task_results = []
        best_cfg, best_val, best_test = None, -1.0, -1.0

        for comb_idx, (hd, ly, lr, wd, bs) in enumerate(task_combinations):
            global_idx = all_combinations.index((hd, ly, lr, wd, bs)) + 1
            cfg = {"hidden_dim": hd, "layers": ly, "lr": lr, "wd": wd, "batch_size": bs}
            
            if args.num_tasks > 1:
                print(f"\n[Task {args.task_id + 1}] [{comb_idx + 1}/{len(task_combinations)}] (Global {global_idx}/{total}) {cfg}")
            else:
                print(f"\n[Grid {global_idx}/{total}] {cfg}")

            result = run_training(
                cfg, epochs=args.grid_epochs, patience=args.grid_patience,
                device=device, model_name=args.model, save_path=None, seed=args.seed,
                eval_test_each_epoch=False, gradient_accumulation_steps=args.gradient_accumulation_steps
            )

            print(f"  val_acc={result['best_val']:.4f} test_acc={result['best_test']:.4f} @ epoch {result['best_epoch']}")
            
            task_results.append({
                "config": cfg,
                "best_val": result["best_val"],
                "best_test": result["best_test"],
                "best_epoch": result["best_epoch"],
                "global_idx": global_idx
            })
            
            if result["best_val"] > best_val:
                best_val = result["best_val"]
                best_test = result["best_test"]
                best_cfg = cfg

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save task results
        if args.num_tasks > 1:
            task_output_path = osp.join(
                args.grid_output_dir, 
                f"{args.model}_grid_task_{args.task_id:03d}.json"
            )
            save_grid_results(task_results, task_output_path)
            
            print("\n" + "="*60)
            print(f"Task {args.task_id + 1} completed!")
            print(f"Best config in this task: {best_cfg}")
            print(f"Best val acc={best_val:.4f}, Test acc={best_test:.4f}")
            print("="*60)
            return

        print("\n" + "="*60)
        print(f"Best config: {best_cfg}")
        print(f"Best val acc={best_val:.4f}, Test acc={best_test:.4f}")
        print("="*60)

    else:
        # Single config training
        cfg = {
            "hidden_dim": 32,
            "layers": 3,
            "lr": args.lr,
            "wd": args.wd,
            "batch_size": args.batch_size
        }
        print(f"Config: {cfg}\n")
        
        if args.num_runs > 1:
            results = run_multiple_experiments(
                config=cfg,
                model_name=args.model,
                device=device,
                num_runs=args.num_runs,
                base_seed=args.seed,
                epochs=args.epochs,
                patience=args.patience,
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
            
            stats = compute_statistics(results)
            print_statistics(stats, args.num_runs)
            
            if args.save_results:
                save_results_to_file(results, stats, cfg, args.model, args.save_results)


if __name__ == "__main__":
    main()

