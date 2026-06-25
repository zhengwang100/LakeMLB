"""
Aggregate benchmark JSON results into a single CSV.

Usage
-----
python aggregate_results.py \
    --results_dir ../results/german_reg_benchmark \
    --output_csv  ../results/german_reg_benchmark/summary.csv

Each JSON in results_dir must contain a "statistics" dict with keys:
    test_rmse_mean, test_rmse_std,
    test_mae_mean,  test_mae_std,
    test_r2_mean,   test_r2_std
"""
import argparse
import json
import os
import glob
import csv
from pathlib import Path

# ── display name mapping  (filename stem → pretty method name) ────────────────
METHOD_NAMES = {
    "xgboost":          "XGBoost",
    "catboost":         "CatBoost",
    "lightgbm":         "LightGBM",
    "tnn_fttransformer":  "FTTransformer",
    "tnn_tabtransformer": "TabTransformer",
    "tnn_excelformer":    "ExcelFormer",
    "tnn_saint":          "SAINT",
    "tnn_tromptnet":      "TromptNet",
    "tabpfn":           "TabPFN v2",
    "transtab_single":    "TransTab (single)",
    "carte_single":       "CARTE (single)",
    "transtab_transfer":  "TransTab (transfer)",
    "carte_joint":        "CARTE (joint)",
}

# Desired output row order
ROW_ORDER = [
    "xgboost", "catboost", "lightgbm",
    "tnn_fttransformer", "tnn_tabtransformer", "tnn_excelformer",
    "tnn_saint", "tnn_tromptnet",
    "tabpfn",
    "transtab_single",
    "carte_single",
    "transtab_transfer",
    "carte_joint",
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_row(stem, data):
    stats = data.get("statistics", {})
    num_runs = data.get("num_runs", "?")
    method = METHOD_NAMES.get(stem, stem)

    def g(key, default="N/A"):
        v = stats.get(key)
        return f"{v:.4f}" if isinstance(v, (int, float)) else default

    return {
        "Method":    method,
        "Num_runs":  num_runs,
        "RMSE_mean": g("test_rmse_mean"),
        "RMSE_std":  g("test_rmse_std"),
        "MAE_mean":  g("test_mae_mean"),
        "MAE_std":   g("test_mae_std"),
        "R2_mean":   g("test_r2_mean"),
        "R2_std":    g("test_r2_std"),
        # convenience combined strings (mean±std)
        "RMSE":      f"{g('test_rmse_mean')}±{g('test_rmse_std')}",
        "MAE":       f"{g('test_mae_mean')}±{g('test_mae_std')}",
        "R2":        f"{g('test_r2_mean')}±{g('test_r2_std')}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_csv",  type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_files  = {p.stem: p for p in results_dir.glob("*.json")}

    rows = []
    missing = []

    # Follow preferred display order first
    ordered_stems = [s for s in ROW_ORDER if s in json_files]
    extra_stems   = [s for s in sorted(json_files) if s not in ROW_ORDER]
    all_stems     = ordered_stems + extra_stems

    for stem in all_stems:
        path = json_files[stem]
        try:
            data = load_json(path)
            rows.append(extract_row(stem, data))
        except Exception as e:
            print(f"  [WARN] Failed to read {path.name}: {e}")

    # Report any expected files that are absent
    for stem in ROW_ORDER:
        if stem not in json_files:
            missing.append(stem)

    if missing:
        print(f"\n[WARN] Missing result files: {missing}")

    # Write CSV
    fieldnames = [
        "Method", "Num_runs",
        "RMSE_mean", "RMSE_std",
        "MAE_mean",  "MAE_std",
        "R2_mean",   "R2_std",
        "RMSE", "MAE", "R2",
    ]

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Pretty-print to console
    print(f"\n{'Method':<22} {'Runs':>4}  {'RMSE':^18}  {'MAE':^18}  {'R²':^18}")
    print("-" * 85)
    for row in rows:
        print(f"{row['Method']:<22} {str(row['Num_runs']):>4}  "
              f"{row['RMSE']:^18}  {row['MAE']:^18}  {row['R2']:^18}")

    print(f"\nCSV saved → {args.output_csv}")


if __name__ == "__main__":
    main()
