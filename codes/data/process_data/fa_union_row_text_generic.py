"""
Generic FA builder for unionable table pairs with row-text 1-NN matching.

Feature Augmentation (FA) keeps the number of target-table rows unchanged. For
each target row, this script builds a row-level text sequence from column
name-value pairs, retrieves the nearest auxiliary row in BERT embedding space,
and horizontally concatenates the matched auxiliary attributes to the target row.

Example:
  python fa_union_row_text_generic.py \
    --task_table target.csv \
    --aux_table auxiliary.csv \
    --output_dir out/fa_union \
    --output_table target_fa.csv
"""

import argparse
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_LOCAL_BERT = SCRIPT_DIR / "models" / "bert-base-uncased"
DEFAULT_BERT = str(
    SCRIPT_LOCAL_BERT
    if SCRIPT_LOCAL_BERT.exists()
    else "bert-base-uncased"
)


def read_csv_auto(path, encoding):
    if encoding != "auto":
        return pd.read_csv(path, encoding=encoding, low_memory=False), encoding

    last_error = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "iso-8859-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False), enc
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not read CSV file {path!r}") from last_error


def parse_column_list(value):
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_text_columns(df, include_cols, exclude_cols, table_name):
    if include_cols:
        missing = [col for col in include_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{table_name} include columns not found: {missing}")
        return include_cols

    missing_excluded = [col for col in exclude_cols if col not in df.columns]
    if missing_excluded:
        raise ValueError(f"{table_name} exclude columns not found: {missing_excluded}")
    return [col for col in df.columns if col not in set(exclude_cols)]


def row_to_text(row, columns):
    parts = []
    for col in columns:
        value = row[col]
        if pd.isna(value):
            continue
        value = str(value).strip()
        if value:
            parts.append(f"{col}: {value}")
    return ", ".join(parts)


def dataframe_to_row_texts(df, columns, desc):
    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        texts.append(row_to_text(row, columns))
    return texts


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_arg!r}, but CUDA is not available")
    return device


def compute_cls_embeddings(texts, tokenizer, model, device, batch_size, max_length):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding rows"):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            cls = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls.astype("float32", copy=False))
    return np.vstack(embeddings)


def build_hnsw_index(embeddings, hnsw_m, hnsw_ef_construction, hnsw_ef_search):
    embeddings = np.ascontiguousarray(embeddings.astype("float32", copy=False))
    faiss.normalize_L2(embeddings)

    index = faiss.IndexHNSWFlat(embeddings.shape[1], hnsw_m)
    index.hnsw.efConstruction = hnsw_ef_construction
    index.hnsw.efSearch = hnsw_ef_search
    index.add(embeddings)
    return index


def search_1nn(index, query_embeddings):
    query_embeddings = np.ascontiguousarray(query_embeddings.astype("float32", copy=False))
    faiss.normalize_L2(query_embeddings)
    squared_l2_distances, indices = index.search(query_embeddings, 1)
    cosine_similarity = 1.0 - squared_l2_distances.reshape(-1) / 2.0
    return cosine_similarity, indices.reshape(-1)


def apply_similarity_threshold(indices, similarities, threshold):
    if threshold is None:
        return indices
    return np.where(similarities >= threshold, indices, -1)


def create_row_mapping(task_texts, aux_texts, indices, similarities):
    aux_text_preview = []
    for idx in indices:
        aux_text_preview.append(aux_texts[int(idx)] if idx >= 0 else "")
    return pd.DataFrame(
        {
            "task_index": np.arange(len(task_texts), dtype=np.int64),
            "aux_index": indices.astype(np.int64),
            "task_text": task_texts,
            "aux_text": aux_text_preview,
            "cosine_similarity": similarities.astype(float),
            "cosine_distance": (1.0 - similarities).astype(float),
        }
    )


def rename_aux_columns(task_columns, aux_columns, aux_prefix):
    task_columns = set(task_columns)
    mapping = {}
    for col in aux_columns:
        mapping[col] = f"{aux_prefix}{col}" if col in task_columns else col
    return mapping


def build_augmented_table(task_df, aux_df, aux_indices, aux_prefix):
    col_mapping = rename_aux_columns(task_df.columns, aux_df.columns, aux_prefix)
    aux_renamed = aux_df.rename(columns=col_mapping)

    valid_mask = aux_indices >= 0
    safe_indices = np.where(valid_mask, aux_indices, 0)
    matched_aux = aux_renamed.iloc[safe_indices].reset_index(drop=True).copy()
    if not valid_mask.all():
        matched_aux.loc[~valid_mask, :] = np.nan

    augmented = pd.concat([task_df.reset_index(drop=True), matched_aux], axis=1)
    if augmented.columns.duplicated().any():
        duplicates = augmented.columns[augmented.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate columns after augmentation: {duplicates}")
    return augmented, matched_aux, col_mapping


def save_column_mapping(col_mapping, output_path):
    rows = [
        {
            "aux_column": aux_col,
            "augmented_column": augmented_col,
            "renamed": aux_col != augmented_col,
        }
        for aux_col, augmented_col in col_mapping.items()
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def print_similarity_stats(similarities, indices):
    matched = indices >= 0
    matched_sims = similarities[matched]
    print("\nMatch statistics:")
    print(f"  matched rows:      {int(matched.sum())}/{len(indices)}")
    if len(matched_sims) == 0:
        print("  no rows passed the threshold")
        return
    print(f"  mean similarity:   {float(np.mean(matched_sims)):.4f}")
    print(f"  median similarity: {float(np.median(matched_sims)):.4f}")
    print(f"  min similarity:    {float(np.min(matched_sims)):.4f}")
    print(f"  max similarity:    {float(np.max(matched_sims)):.4f}")
    print(f"  > 0.8:             {int(np.sum(matched_sims > 0.8))}")
    print(f"  0.5 - 0.8:         {int(np.sum((matched_sims >= 0.5) & (matched_sims <= 0.8)))}")
    print(f"  < 0.5:             {int(np.sum(matched_sims < 0.5))}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    task_include_cols = parse_column_list(args.task_text_cols)
    aux_include_cols = parse_column_list(args.aux_text_cols)
    task_exclude_cols = parse_column_list(args.exclude_task_cols)
    aux_exclude_cols = parse_column_list(args.exclude_aux_cols)

    print("=" * 72)
    print("Union FA: Row-Text BERT [CLS] + FAISS HNSW 1-NN")
    print("=" * 72)

    print("\n[1/7] Loading CSV files")
    task_df, task_encoding = read_csv_auto(args.task_table, args.encoding)
    aux_df, aux_encoding = read_csv_auto(args.aux_table, args.encoding)
    print(f"  task table: {args.task_table} {task_df.shape}, encoding={task_encoding}")
    print(f"  aux table:  {args.aux_table} {aux_df.shape}, encoding={aux_encoding}")

    task_text_cols = resolve_text_columns(task_df, task_include_cols, task_exclude_cols, "task")
    aux_text_cols = resolve_text_columns(aux_df, aux_include_cols, aux_exclude_cols, "aux")
    print(f"  task row-text columns: {len(task_text_cols)}")
    print(f"  aux row-text columns:  {len(aux_text_cols)}")

    print("\n[2/7] Converting rows to text")
    task_texts = dataframe_to_row_texts(task_df, task_text_cols, "Task rows")
    aux_texts = dataframe_to_row_texts(aux_df, aux_text_cols, "Aux rows")
    print(f"  task example: {task_texts[0][:160] if task_texts else ''}")
    print(f"  aux example:  {aux_texts[0][:160] if aux_texts else ''}")

    print("\n[3/7] Loading BERT model")
    device = resolve_device(args.device)
    print(f"  model:  {args.bert}")
    print(f"  device: {device}")
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    model = BertModel.from_pretrained(args.bert).to(device)

    print("\n[4/7] Computing auxiliary row embeddings")
    aux_embeddings = compute_cls_embeddings(
        aux_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )

    print("\n[5/7] Computing target row embeddings")
    task_embeddings = compute_cls_embeddings(
        task_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )

    print("\n[6/7] Building FAISS index and searching 1-NN")
    index = build_hnsw_index(
        aux_embeddings,
        args.hnsw_m,
        args.hnsw_ef_construction,
        args.hnsw_ef_search,
    )
    similarities, aux_indices = search_1nn(index, task_embeddings)
    aux_indices = apply_similarity_threshold(aux_indices, similarities, args.threshold)

    print("\n[7/7] Saving augmented feature table and mappings")
    mapping = create_row_mapping(task_texts, aux_texts, aux_indices, similarities)
    augmented, matched_aux, col_mapping = build_augmented_table(
        task_df,
        aux_df,
        aux_indices,
        args.aux_prefix,
    )

    augmented_path = output_dir / args.output_table
    mapping_path = output_dir / args.output_mapping
    matched_aux_path = output_dir / args.output_matched_aux
    column_mapping_path = output_dir / args.output_column_mapping

    augmented.to_csv(augmented_path, index=False, encoding="utf-8-sig")
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    matched_aux.to_csv(matched_aux_path, index=False, encoding="utf-8-sig")
    save_column_mapping(col_mapping, column_mapping_path)

    print_similarity_stats(similarities, aux_indices)
    print("\nOutputs:")
    print(f"  augmented table: {augmented_path} {augmented.shape}")
    print(f"  row mapping:     {mapping_path}")
    print(f"  matched aux:     {matched_aux_path} {matched_aux.shape}")
    print(f"  column mapping:  {column_mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic FA builder for unionable table pairs using row-text 1-NN matching."
    )
    parser.add_argument("--task_table", required=True, help="Target/task table CSV")
    parser.add_argument("--aux_table", required=True, help="Auxiliary table CSV")
    parser.add_argument("--encoding", default="auto", help="CSV encoding, or auto")

    parser.add_argument(
        "--task_text_cols",
        default="",
        help="Comma-separated target columns used for row text. Empty means all except excluded columns.",
    )
    parser.add_argument(
        "--aux_text_cols",
        default="",
        help="Comma-separated auxiliary columns used for row text. Empty means all except excluded columns.",
    )
    parser.add_argument(
        "--exclude_task_cols",
        default="",
        help="Comma-separated target columns excluded from row text when --task_text_cols is empty.",
    )
    parser.add_argument(
        "--exclude_aux_cols",
        default="",
        help="Comma-separated auxiliary columns excluded from row text when --aux_text_cols is empty.",
    )

    parser.add_argument("--bert", default=DEFAULT_BERT, help="BERT model name or local path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch_size", type=int, default=16, help="BERT embedding batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Tokenizer max sequence length")
    parser.add_argument("--hnsw_m", type=int, default=32, help="FAISS HNSW M parameter")
    parser.add_argument("--hnsw_ef_construction", type=int, default=64, help="FAISS HNSW efConstruction")
    parser.add_argument("--hnsw_ef_search", type=int, default=64, help="FAISS HNSW efSearch")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional cosine-similarity threshold. Default keeps every 1-NN match.",
    )

    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_table", default="augmented_table.csv", help="Output augmented CSV filename")
    parser.add_argument("--output_mapping", default="row_mapping.csv", help="Output row mapping CSV filename")
    parser.add_argument(
        "--output_matched_aux",
        default="matched_aux_table.csv",
        help="Output matched auxiliary rows CSV filename",
    )
    parser.add_argument(
        "--output_column_mapping",
        default="column_mapping.csv",
        help="Output auxiliary column rename mapping CSV filename",
    )
    parser.add_argument(
        "--aux_prefix",
        default="aux_",
        help="Prefix for auxiliary columns that conflict with target-table column names.",
    )

    main(parser.parse_args())
