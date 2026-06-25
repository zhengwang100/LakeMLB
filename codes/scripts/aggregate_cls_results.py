"""
Aggregate classification benchmark JSON results into a single CSV.

Usage
-----
python aggregate_cls_results.py \
    --results_dir ../results/nnstocks1nn_cls_benchmark \
    --output_csv  ../results/nnstocks1nn_cls_benchmark/summary.csv

Each JSON must contain a "statistics" dict with at least:
    test_acc_mean, test_acc_std
"""
import argparse
import json
import os
import csv
from pathlib import Path

METHOD_NAMES = {
    "xgboost":            "XGBoost",
    "catboost":           "CatBoost",
    "lightgbm":           "LightGBM",
    "tnn_fttransformer":  "FTTransformer",
    "tnn_tabtransformer": "TabTransformer",
    "tnn_excelformer":    "ExcelFormer",
    "tnn_saint":          "SAINT",
    "tnn_tromptnet":      "TromptNet",
    "tabpfn":             "TabPFN v2",
    "transtab_single":    "TransTab (single)",
    "carte_single":       "CARTE (single)",
}

ROW_ORDER = [
    "xgboost", "catboost", "lightgbm",
    "tnn_fttransformer", "tnn_tabtransformer", "tnn_excelformer",
    "tnn_saint", "tnn_tromptnet",
    "tabpfn",
    "transtab_single",
    "carte_single",
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_row(stem, data):
    stats    = data.get("statistics", {})
    num_runs = data.get("num_runs", "?")
    method   = METHOD_NAMES.get(stem, stem)

    def g(key, default="N/A"):
        v = stats.get(key)
        return f"{v:.4f}" if isinstance(v, (int, float)) else default

    acc_mean = g("test_acc_mean")
    acc_std  = g("test_acc_std")
    return {
        "Method":    method,
        "Num_runs":  num_runs,
        "Acc_mean":  acc_mean,
        "Acc_std":   acc_std,
        "Acc":       f"{acc_mean}±{acc_std}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_csv",  type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_files  = {p.stem: p for p in results_dir.glob("*.json")}

    ordered_stems = [s for s in ROW_ORDER if s in json_files]
    extra_stems   = [s for s in sorted(json_files) if s not in ROW_ORDER]
    all_stems     = ordered_stems + extra_stems

    rows    = []
    missing = []

    for stem in all_stems:
        try:
            data = load_json(json_files[stem])
            rows.append(extract_row(stem, data))
        except Exception as e:
            print(f"  [WARN] Failed to read {json_files[stem].name}: {e}")

    for stem in ROW_ORDER:
        if stem not in json_files:
            missing.append(stem)
    if missing:
        print(f"\n[WARN] Missing result files: {missing}")

    fieldnames = ["Method", "Num_runs", "Acc_mean", "Acc_std", "Acc"]
    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # pretty-print
    print(f"\n{'Method':<22} {'Runs':>4}  {'Acc (mean±std)':^20}")
    print("-" * 52)
    for row in rows:
        print(f"{row['Method']:<22} {str(row['Num_runs']):>4}  {row['Acc']:^20}")

    print(f"\nCSV saved → {args.output_csv}")


if __name__ == "__main__":
    main()
