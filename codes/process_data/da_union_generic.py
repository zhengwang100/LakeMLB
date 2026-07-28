"""Build union-based data augmentation for weakly aligned tables.

The script aligns columns by normalized names, maps auxiliary labels to target
labels with BERT and FAISS 1-NN, samples relabeled auxiliary rows, and appends
them as training-only examples.
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
DEFAULT_BERT = os.environ.get(
    "BERT_MODEL_PATH",
    str(SCRIPT_LOCAL_BERT if SCRIPT_LOCAL_BERT.exists() else "bert-base-uncased"),
)


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
    return (
        str(name)
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("/", "")
        .replace("-", "")
    )


def require_column(df, column, name):
    if column not in df.columns:
        raise ValueError(
            f"{name} column {column!r} not found. Columns: {list(df.columns)}"
        )


def align_columns(task_df, aux_df, task_label, aux_label):
    task_cols = [col for col in task_df.columns if col != task_label]
    aux_cols = [col for col in aux_df.columns if col != aux_label]
    task_by_norm = {normalize_column_name(col): col for col in task_cols}

    mapping = {}
    for aux_col in aux_cols:
        task_col = task_by_norm.get(normalize_column_name(aux_col))
        if task_col is not None:
            mapping[aux_col] = task_col
    return mapping


def load_unique_labels(df, label_col):
    labels = df[label_col].dropna().astype(str).map(str.strip)
    labels = labels[labels != ""].unique().tolist()
    if not labels:
        raise ValueError(f"No non-empty labels found in column {label_col!r}")
    return sorted(labels)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device_arg!r}, but CUDA is not available"
        )
    return device


def compute_cls_embeddings(texts, tokenizer, model, device, batch_size, max_length):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding labels"):
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


def align_labels(
    task_labels, aux_labels, tokenizer, model, device, batch_size, max_length
):
    task_emb = compute_cls_embeddings(
        task_labels, tokenizer, model, device, batch_size, max_length
    )
    aux_emb = compute_cls_embeddings(
        aux_labels, tokenizer, model, device, batch_size, max_length
    )

    task_emb = np.ascontiguousarray(task_emb.astype("float32", copy=False))
    aux_emb = np.ascontiguousarray(aux_emb.astype("float32", copy=False))
    faiss.normalize_L2(task_emb)
    faiss.normalize_L2(aux_emb)

    index = faiss.IndexHNSWFlat(task_emb.shape[1], 32)
    index.hnsw.efConstruction = 64
    index.hnsw.efSearch = 64
    index.add(task_emb)

    distances, indices = index.search(aux_emb, 1)
    cosine_similarity = 1.0 - distances.reshape(-1) / 2.0
    nearest = indices.reshape(-1)

    rows = []
    mapping = {}
    for i, aux_label in enumerate(aux_labels):
        task_label = task_labels[int(nearest[i])]
        mapping[aux_label] = task_label
        rows.append(
            {
                "aux_label": aux_label,
                "task_label": task_label,
                "cosine_similarity": float(cosine_similarity[i]),
                "cosine_distance": float(1.0 - cosine_similarity[i]),
            }
        )
    return mapping, pd.DataFrame(rows)


def load_mask(mask_file):
    mask = torch.load(mask_file, weights_only=False)
    for key in ("train_mask", "val_mask", "test_mask"):
        if key not in mask:
            raise ValueError(f"Mask file must contain {key!r}")
        if mask[key].dim() != 1:
            raise ValueError(f"{key} must be a 1-D tensor")
    return mask


def sample_count(task_df, aux_df, mask, sample_ratio, cap_basis):
    if cap_basis == "train":
        base = int(mask["train_mask"].sum().item())
    elif cap_basis == "all":
        base = len(task_df)
    else:
        raise ValueError(f"Unsupported cap basis: {cap_basis}")
    return min(int(base * sample_ratio), len(aux_df))


def relabel_and_sample_aux(
    aux_df, aux_label, task_label, label_mapping, n_to_add, seed
):
    aux = aux_df.copy()
    aux[task_label] = aux[aux_label].astype(str).map(str.strip).map(label_mapping)
    aux = aux[aux[task_label].notna()].copy()
    if aux_label != task_label and aux_label in aux.columns:
        aux = aux.drop(columns=[aux_label])
    if n_to_add > len(aux):
        n_to_add = len(aux)
    if n_to_add == 0:
        return aux.iloc[0:0].copy(), 0
    return aux.sample(n=n_to_add, random_state=seed).copy(), n_to_add


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

    task_only_cols = [
        col for col in task_extended.columns if col not in aux_aligned.columns
    ]
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
            raise ValueError(
                f"{key} length {len(mask[key])} does not match target rows {n_task}"
            )

    new_mask = dict(mask)
    new_mask["train_mask"] = torch.cat(
        [mask["train_mask"].bool(), torch.ones(n_added, dtype=torch.bool)]
    )
    new_mask["val_mask"] = torch.cat(
        [mask["val_mask"].bool(), torch.zeros(n_added, dtype=torch.bool)]
    )
    new_mask["test_mask"] = torch.cat(
        [mask["test_mask"].bool(), torch.zeros(n_added, dtype=torch.bool)]
    )
    torch.save(new_mask, output_path)
    return new_mask


def save_mapping(mapping, path):
    pd.DataFrame(
        [
            {"aux_column": aux_col, "task_column": task_col}
            for aux_col, task_col in mapping.items()
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    task_df, task_encoding = read_csv_auto(args.task_table, args.encoding)
    aux_df, aux_encoding = read_csv_auto(args.aux_table, args.encoding)
    require_column(task_df, args.task_label, "Target label")
    require_column(aux_df, args.aux_label, "Auxiliary label")

    mask = load_mask(args.mask_file)
    col_mapping = align_columns(task_df, aux_df, args.task_label, args.aux_label)

    print("=" * 72)
    print("Union DA")
    print("=" * 72)
    print(
        f"Target table: {args.task_table} ({task_df.shape}, encoding={task_encoding})"
    )
    print(f"Aux table:    {args.aux_table} ({aux_df.shape}, encoding={aux_encoding})")
    print(f"Column matches: {len(col_mapping)}")

    task_labels = load_unique_labels(task_df, args.task_label)
    aux_labels = load_unique_labels(aux_df, args.aux_label)
    print(f"Target labels: {len(task_labels)}")
    print(f"Aux labels:    {len(aux_labels)}")

    device = resolve_device(args.device)
    print(f"Loading BERT: {args.bert} on {device}")
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    model = BertModel.from_pretrained(args.bert).to(device)
    label_mapping, label_mapping_df = align_labels(
        task_labels,
        aux_labels,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )

    n_to_add = sample_count(task_df, aux_df, mask, args.sample_ratio, args.cap_basis)
    aux_sampled, n_added = relabel_and_sample_aux(
        aux_df,
        args.aux_label,
        args.task_label,
        label_mapping,
        n_to_add,
        args.seed,
    )
    combined_df, aux_only_cols = append_with_unified_schema(
        task_df, aux_sampled, col_mapping
    )

    table_path = output_dir / args.output_table
    mask_path = output_dir / args.output_mask
    col_mapping_path = output_dir / "col_mapping.csv"
    label_mapping_path = output_dir / "label_mapping.csv"

    combined_df.to_csv(table_path, index=False, encoding="utf-8-sig")
    extend_mask(mask, len(task_df), n_added, mask_path)
    save_mapping(col_mapping, col_mapping_path)
    label_mapping_df.to_csv(label_mapping_path, index=False, encoding="utf-8-sig")

    print(f"Cap basis: {args.cap_basis}, ratio: {args.sample_ratio}")
    print(f"Appended rows: {n_added}")
    print(f"Aux-only columns retained: {len(aux_only_cols)}")
    print(f"Combined table: {table_path} {combined_df.shape}")
    print(f"Mask:           {mask_path}")
    print(f"Column mapping: {col_mapping_path}")
    print(f"Label mapping:  {label_mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic DA builder for unionable table pairs."
    )
    parser.add_argument("--task_table", required=True, help="Target/task table CSV")
    parser.add_argument("--aux_table", required=True, help="Auxiliary table CSV")
    parser.add_argument(
        "--task_label", required=True, help="Label column in target table"
    )
    parser.add_argument(
        "--aux_label", required=True, help="Label column in auxiliary table"
    )
    parser.add_argument(
        "--mask_file", required=True, help="Original target-table mask .pt"
    )
    parser.add_argument("--encoding", default="auto", help="CSV encoding, or auto")

    parser.add_argument(
        "--bert", default=DEFAULT_BERT, help="BERT model name or local path"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="BERT embedding batch size"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Tokenizer max sequence length"
    )

    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--output_table", default="combined_table.csv", help="Output CSV filename"
    )
    parser.add_argument(
        "--output_mask", default="mask_da.pt", help="Output mask filename"
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=0.3, help="Sample append cap ratio"
    )
    parser.add_argument(
        "--cap_basis",
        choices=("train", "all"),
        default="train",
        help="Use train_mask size or all target rows as the sample-ratio base",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling random seed")

    main(parser.parse_args())
