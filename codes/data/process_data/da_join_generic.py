"""
Generic DA builder for joinable table pairs.

The script constructs pseudo-labeled auxiliary samples from a 1-NN pairing and
then appends them to the target table for training-time data augmentation.

Input options:
1. --enriched_table: a table produced by concatenating target rows with their
   matched auxiliary rows. Target columns are removed, except the label column.
2. --matched_aux_table: the matched auxiliary table only, row-aligned with the
   target table. Target labels are copied by row position.

Example:
  python da_join_generic.py \
    --task_table target.csv \
    --enriched_table target_enriched.csv \
    --label_col label \
    --mask_file mask_target.pt \
    --output_dir out/join_da
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def read_csv_auto(path, encoding):
    if encoding != "auto":
        return pd.read_csv(path, encoding=encoding), encoding

    last_error = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "iso-8859-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not read CSV file {path!r}") from last_error


def normalize_column_name(name):
    return str(name).lower().replace("_", "").replace(" ", "").replace("/", "").replace("-", "")


def require_column(df, column, name):
    if column not in df.columns:
        raise ValueError(f"{name} column {column!r} not found. Columns: {list(df.columns)}")


def load_mask(mask_file):
    mask = torch.load(mask_file, weights_only=False)
    for key in ("train_mask", "val_mask", "test_mask"):
        if key not in mask:
            raise ValueError(f"Mask file must contain {key!r}")
        if mask[key].dim() != 1:
            raise ValueError(f"{key} must be a 1-D tensor")
    return mask


def extract_labeled_aux_from_enriched(enriched_df, task_df, label_col):
    task_cols_to_remove = [col for col in task_df.columns if col != label_col]
    cols_to_keep = [col for col in enriched_df.columns if col not in task_cols_to_remove]
    if label_col not in cols_to_keep:
        raise ValueError(
            f"Label column {label_col!r} must be present in enriched table after removing target columns"
        )
    aux_with_label = enriched_df[cols_to_keep].copy()
    return aux_with_label.drop_duplicates().reset_index(drop=True)


def build_labeled_aux_from_matched(task_df, matched_aux_df, label_col):
    if len(task_df) != len(matched_aux_df):
        raise ValueError(
            f"matched_aux_table must be row-aligned with task_table: "
            f"{len(matched_aux_df)} rows vs {len(task_df)} target rows"
        )
    aux = matched_aux_df.copy()
    if label_col in aux.columns:
        aux = aux.drop(columns=[label_col])
    aux[label_col] = task_df[label_col].to_numpy()
    return aux.drop_duplicates().reset_index(drop=True)


def align_columns(task_df, aux_df, label_col):
    task_cols = [col for col in task_df.columns if col != label_col]
    aux_cols = [col for col in aux_df.columns if col != label_col]
    task_by_norm = {normalize_column_name(col): col for col in task_cols}

    mapping = {}
    for aux_col in aux_cols:
        task_col = task_by_norm.get(normalize_column_name(aux_col))
        if task_col is not None:
            mapping[aux_col] = task_col
    return mapping


def sample_count(task_df, aux_df, mask, sample_ratio, cap_basis):
    if cap_basis == "train":
        base = int(mask["train_mask"].sum().item())
    elif cap_basis == "all":
        base = len(task_df)
    else:
        raise ValueError(f"Unsupported cap basis: {cap_basis}")
    return min(int(base * sample_ratio), len(aux_df))


def append_with_unified_schema(task_df, aux_df, col_mapping):
    rename_dict = {
        aux_col: task_col
        for aux_col, task_col in col_mapping.items()
        if aux_col in aux_df.columns and aux_col != task_col
    }
    aux_aligned = aux_df.rename(columns=rename_dict).copy()

    aux_only_cols = [col for col in aux_aligned.columns if col not in task_df.columns]
    task_extended = task_df.copy()
    for col in aux_only_cols:
        task_extended[col] = np.nan

    task_only_cols = [col for col in task_extended.columns if col not in aux_aligned.columns]
    for col in task_only_cols:
        aux_aligned[col] = np.nan

    all_columns = list(task_df.columns) + aux_only_cols
    combined = pd.concat(
        [task_extended[all_columns], aux_aligned[all_columns]],
        ignore_index=True,
    )
    return combined, aux_only_cols


def extend_mask(mask, n_task, n_added, output_path):
    for key in ("train_mask", "val_mask", "test_mask"):
        if len(mask[key]) != n_task:
            raise ValueError(f"{key} length {len(mask[key])} does not match target rows {n_task}")

    new_mask = dict(mask)
    new_mask["train_mask"] = torch.cat([mask["train_mask"].bool(), torch.ones(n_added, dtype=torch.bool)])
    new_mask["val_mask"] = torch.cat([mask["val_mask"].bool(), torch.zeros(n_added, dtype=torch.bool)])
    new_mask["test_mask"] = torch.cat([mask["test_mask"].bool(), torch.zeros(n_added, dtype=torch.bool)])
    torch.save(new_mask, output_path)
    return new_mask


def save_mapping(mapping, path):
    pd.DataFrame(
        [{"aux_column": aux_col, "task_column": task_col} for aux_col, task_col in mapping.items()]
    ).to_csv(path, index=False, encoding="utf-8-sig")


def main(args):
    if bool(args.enriched_table) == bool(args.matched_aux_table):
        raise ValueError("Provide exactly one of --enriched_table or --matched_aux_table")

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    task_df, task_encoding = read_csv_auto(args.task_table, args.encoding)
    require_column(task_df, args.label_col, "Label")
    mask = load_mask(args.mask_file)

    if args.enriched_table:
        source_df, source_encoding = read_csv_auto(args.enriched_table, args.encoding)
        labeled_aux_df = extract_labeled_aux_from_enriched(source_df, task_df, args.label_col)
        source_name = args.enriched_table
    else:
        source_df, source_encoding = read_csv_auto(args.matched_aux_table, args.encoding)
        labeled_aux_df = build_labeled_aux_from_matched(task_df, source_df, args.label_col)
        source_name = args.matched_aux_table

    labeled_aux_df = labeled_aux_df[labeled_aux_df[args.label_col].notna()].copy()
    col_mapping = align_columns(task_df, labeled_aux_df, args.label_col)

    n_to_add = sample_count(task_df, labeled_aux_df, mask, args.sample_ratio, args.cap_basis)
    if n_to_add > 0:
        sampled_aux = labeled_aux_df.sample(n=n_to_add, random_state=args.seed).copy()
    else:
        sampled_aux = labeled_aux_df.iloc[0:0].copy()

    combined_df, aux_only_cols = append_with_unified_schema(task_df, sampled_aux, col_mapping)

    table_path = output_dir / args.output_table
    mask_path = output_dir / args.output_mask
    labeled_aux_path = output_dir / "labeled_aux_table.csv"
    col_mapping_path = output_dir / "col_mapping.csv"

    combined_df.to_csv(table_path, index=False, encoding="utf-8-sig")
    labeled_aux_df.to_csv(labeled_aux_path, index=False, encoding="utf-8-sig")
    save_mapping(col_mapping, col_mapping_path)
    extend_mask(mask, len(task_df), len(sampled_aux), mask_path)

    print("=" * 72)
    print("Join DA")
    print("=" * 72)
    print(f"Target table: {args.task_table} ({task_df.shape}, encoding={task_encoding})")
    print(f"Pairing source: {source_name} ({source_df.shape}, encoding={source_encoding})")
    print(f"Labeled auxiliary rows after dedup/dropna: {len(labeled_aux_df)}")
    print(f"Column matches: {len(col_mapping)}")
    print(f"Cap basis: {args.cap_basis}, ratio: {args.sample_ratio}")
    print(f"Appended rows: {len(sampled_aux)}")
    print(f"Aux-only columns retained: {len(aux_only_cols)}")
    print(f"Combined table:     {table_path} {combined_df.shape}")
    print(f"Mask:               {mask_path}")
    print(f"Labeled aux table:  {labeled_aux_path}")
    print(f"Column mapping:     {col_mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic DA builder for joinable table pairs.")
    parser.add_argument("--task_table", required=True, help="Target/task table CSV")
    parser.add_argument("--enriched_table", help="CSV containing target rows concatenated with matched aux rows")
    parser.add_argument("--matched_aux_table", help="Row-aligned matched auxiliary CSV")
    parser.add_argument("--label_col", required=True, help="Target label column")
    parser.add_argument("--mask_file", required=True, help="Original target-table mask .pt")
    parser.add_argument("--encoding", default="auto", help="CSV encoding, or auto")

    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_table", default="combined_table.csv", help="Output CSV filename")
    parser.add_argument("--output_mask", default="mask_da.pt", help="Output mask filename")
    parser.add_argument("--sample_ratio", type=float, default=0.3, help="Sample append cap ratio")
    parser.add_argument(
        "--cap_basis",
        choices=("train", "all"),
        default="train",
        help="Use train_mask size or all target rows as the sample-ratio base",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling random seed")

    main(parser.parse_args())
