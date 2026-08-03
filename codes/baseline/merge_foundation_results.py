#!/usr/bin/env python3
"""Merge repeated TabPFN or TabICL run files into one summary."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    args = parser.parse_args()

    runs = []
    for idx, path_str in enumerate(args.inputs, start=1):
        path = Path(path_str)
        with path.open() as f:
            data = json.load(f)
        run = data["individual_runs"][0]
        runs.append({
            "run_id": idx,
            "path": str(path),
            "seed": run["seed"],
            "test_acc": run["test_acc"],
            "runtime": run.get("runtime", data.get("statistics", {}).get("total_runtime")),
        })

    stats = {}
    for name in ["test_acc", "runtime"]:
        values = np.array([r[name] for r in runs], dtype=float)
        stats[f"{name}_mean"] = float(values.mean())
        stats[f"{name}_std"] = float(values.std())
        stats[f"{name}_min"] = float(values.min())
        stats[f"{name}_max"] = float(values.max())

    output = {
        "model": args.model,
        "task": "classification",
        "dataset": args.dataset,
        "num_runs": len(runs),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_path": args.log_path,
        "individual_runs": runs,
        "statistics": stats,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Summary saved -> {output_path}")
    print(f"Test Acc: {stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.4f}")
    print(f"Runtime:  {stats['runtime_mean']:.2f}s ± {stats['runtime_std']:.2f}s")


if __name__ == "__main__":
    main()
