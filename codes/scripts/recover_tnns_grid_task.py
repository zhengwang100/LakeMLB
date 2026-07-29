#!/usr/bin/env python3
"""Recover a TNNS grid shard from logs and append rerun results.

This is useful when a parallel grid shard crashed before writing its
`*_grid_task_XXX.json`. It parses completed combinations from the shard log,
then appends JSON results from a small rerun of the missing combinations.
"""

import argparse
import ast
import json
import os
import re
from typing import Dict, List


CONFIG_RE = re.compile(r"\(Global\s+(\d+)/\d+\)\s+(\{.*\})")
RESULT_RE = re.compile(
    r"val_acc=([0-9.]+)\s+test_acc=([0-9.]+)\s+@\s+epoch\s+(\d+)"
)


def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("all_results"), list):
            return payload["all_results"]
        if isinstance(payload.get("grid_results"), list):
            return payload["grid_results"]
    raise ValueError(f"Unsupported grid result format: {path}")


def parse_log(path: str) -> List[Dict]:
    records: List[Dict] = []
    pending = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            config_match = CONFIG_RE.search(line)
            if config_match:
                pending = {
                    "global_idx": int(config_match.group(1)),
                    "config": ast.literal_eval(config_match.group(2)),
                }
                continue
            result_match = RESULT_RE.search(line)
            if result_match and pending is not None:
                records.append(
                    {
                        "config": pending["config"],
                        "best_val": float(result_match.group(1)),
                        "best_test": float(result_match.group(2)),
                        "best_epoch": int(result_match.group(3)),
                        "training_time": None,
                        "global_idx": pending["global_idx"],
                    }
                )
                pending = None
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Crashed shard log path.")
    parser.add_argument(
        "--extra_json",
        nargs="*",
        default=[],
        help="JSON result files from rerun missing combinations.",
    )
    parser.add_argument("--output", required=True, help="Recovered task JSON path.")
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Expected number of records in the recovered shard.",
    )
    args = parser.parse_args()

    records = parse_log(args.log)
    for extra_path in args.extra_json:
        records.extend(load_results(extra_path))

    by_global_idx: Dict[int, Dict] = {}
    for record in records:
        idx = int(record["global_idx"])
        by_global_idx[idx] = record

    merged = [by_global_idx[idx] for idx in sorted(by_global_idx)]
    if args.expected is not None and len(merged) != args.expected:
        raise RuntimeError(
            f"Recovered {len(merged)} records, expected {args.expected}. "
            f"Global indices: {sorted(by_global_idx)}"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    best = max(merged, key=lambda item: item["best_val"])
    print(f"Saved {len(merged)} records -> {args.output}")
    print(f"Best global_idx={best['global_idx']} val={best['best_val']} config={best['config']}")


if __name__ == "__main__":
    main()
