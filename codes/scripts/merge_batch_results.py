#!/usr/bin/env python3
"""Merge multiple batch JSON result files into one, recomputing statistics."""
import argparse
import json
import numpy as np
from pathlib import Path


def merge(input_files: list[str], output_file: str) -> None:
    all_runs: list[dict] = []
    meta: dict = {}

    for f in input_files:
        with open(f) as fp:
            d = json.load(fp)
        if not meta:
            meta = {k: v for k, v in d.items()
                    if k not in ("individual_runs", "statistics", "num_runs")}
        all_runs.extend(d.get("individual_runs", []))

    # Re-number run_ids sequentially
    for i, r in enumerate(all_runs, 1):
        r["run_id"] = i

    # Recompute statistics over all merged runs
    skip = {"run_id", "seed"}
    numeric_keys = [k for k in all_runs[0] if k not in skip]
    stats: dict = {}
    for key in numeric_keys:
        vals = [r[key] for r in all_runs if key in r]
        stats[f"{key}_mean"] = float(np.mean(vals))
        stats[f"{key}_std"]  = float(np.std(vals))
        stats[f"{key}_min"]  = float(np.min(vals))
        stats[f"{key}_max"]  = float(np.max(vals))

    result = {**meta, "num_runs": len(all_runs),
              "individual_runs": all_runs, "statistics": stats}

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"  merged {len(input_files)} batches "
          f"({len(all_runs)} runs total) → {output_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge batch JSON result files.")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Batch JSON files to merge")
    p.add_argument("--output", required=True,
                   help="Output merged JSON file")
    args = p.parse_args()
    merge(args.inputs, args.output)
