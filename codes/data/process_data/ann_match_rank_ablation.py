"""Generate exact cosine rank-NN and random table-enrichment ablations."""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_CANDIDATES = [
    SCRIPT_DIR / "models" / "bert-base-uncased",
    SCRIPT_DIR.parents[1] / "lib" / "models" / "bert-base-uncased",
]
LOCAL_DEFAULT_MODEL = next(
    (path for path in LOCAL_MODEL_CANDIDATES if path.exists()), None
)
DEFAULT_MODEL = str(LOCAL_DEFAULT_MODEL or "bert-base-uncased")


def parse_int_list(value):
    ranks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not ranks or ranks[0] < 1:
        raise argparse.ArgumentTypeError("ranks must contain positive integers")
    return ranks


def parse_str_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but CUDA is unavailable")
    return device


def load_table(path, text_col, encoding):
    df = pd.read_csv(path, encoding=encoding, low_memory=False)
    if text_col not in df.columns:
        raise ValueError(
            f"Column {text_col!r} not found in {path}. Columns: {list(df.columns)}"
        )
    return df, df[text_col].fillna("").astype(str).tolist()


def compute_cls_embeddings(
    texts, tokenizer, model, device, batch_size, max_length, description
):
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc=description):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            cls = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
            batches.append(cls.astype("float32", copy=False))
    embeddings = np.ascontiguousarray(np.vstack(batches), dtype="float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def exact_cosine_search(reference_embeddings, query_embeddings, max_rank):
    index = faiss.IndexFlatIP(reference_embeddings.shape[1])
    index.add(reference_embeddings)
    similarities, indices = index.search(query_embeddings, max_rank)
    return similarities, indices


def rename_conflicting_columns(t1_columns, t2_columns, prefix):
    t1_columns = set(t1_columns)
    return [f"{prefix}{col}" if col in t1_columns else col for col in t2_columns]


def save_match(
    name,
    match_type,
    rank,
    t1_df,
    t2_df,
    t1_texts,
    t2_texts,
    t2_indices,
    similarities,
    output_dir,
    output_prefix,
    drop_t2_cols,
    t2_prefix,
    encoding,
):
    output_path = output_dir / f"{output_prefix}_{name}.csv"
    mapping_path = output_dir / f"{output_prefix}_{name}_mapping.csv"

    matched_t2_full = t2_df.iloc[t2_indices].reset_index(drop=True)
    matched_t2 = matched_t2_full.drop(columns=drop_t2_cols, errors="ignore").copy()
    matched_t2.columns = rename_conflicting_columns(
        t1_df.columns, matched_t2.columns, t2_prefix
    )
    enriched = pd.concat([t1_df.reset_index(drop=True), matched_t2], axis=1)
    if enriched.columns.duplicated().any():
        duplicates = enriched.columns[enriched.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate enriched columns: {duplicates}")
    enriched.to_csv(output_path, index=False, encoding=encoding)

    mapping_data = {
        "t1_index": np.arange(len(t1_df), dtype=np.int64),
        "t2_index": t2_indices.astype(np.int64),
        "match_type": match_type,
        "rank": rank,
        "t1_text": t1_texts,
        "t2_text": [t2_texts[index] for index in t2_indices],
        "cosine_similarity": similarities.astype(np.float32),
        "cosine_distance": (1.0 - similarities).astype(np.float32),
    }
    if "Unnamed: 0" in matched_t2_full.columns:
        mapping_data["t2_source_index"] = matched_t2_full["Unnamed: 0"].to_numpy()
    mapping = pd.DataFrame(mapping_data)
    mapping.to_csv(mapping_path, index=False, encoding=encoding)

    stats = {
        "name": name,
        "match_type": match_type,
        "rank": rank,
        "output": str(output_path),
        "mapping": str(mapping_path),
        "rows": len(enriched),
        "columns": len(enriched.columns),
        "unique_t2_matches": int(np.unique(t2_indices).size),
        "similarity_mean": float(np.mean(similarities)),
        "similarity_median": float(np.median(similarities)),
        "similarity_min": float(np.min(similarities)),
        "similarity_max": float(np.max(similarities)),
    }
    print(
        f"  {name}: rows={stats['rows']}, cols={stats['columns']}, "
        f"unique_t2={stats['unique_t2_matches']}, "
        f"mean_similarity={stats['similarity_mean']:.4f}"
    )
    return stats


def main(args):
    start_time = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print("=" * 72)
    print("Exact cosine rank-NN + random matching")
    print("=" * 72)
    t1_df, t1_texts = load_table(args.t1, args.t1_col, args.encoding)
    t2_df, t2_texts = load_table(args.t2, args.t2_col, args.encoding)
    if len(t2_df) < max(args.ranks):
        raise ValueError(
            f"T2 has {len(t2_df)} rows, fewer than requested rank {max(args.ranks)}"
        )
    print(f"T1: {args.t1} ({len(t1_df)} rows), text={args.t1_col}")
    print(f"T2: {args.t2} ({len(t2_df)} rows), text={args.t2_col}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Ranks: {args.ranks}; random={args.include_random}; seed={args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    t2_embeddings = compute_cls_embeddings(
        t2_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
        "Embedding T2",
    )
    t1_embeddings = compute_cls_embeddings(
        t1_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
        "Embedding T1",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    top_similarities, top_indices = exact_cosine_search(
        t2_embeddings, t1_embeddings, max(args.ranks)
    )

    outputs = []
    for rank in args.ranks:
        outputs.append(
            save_match(
                name=f"{rank}nn",
                match_type="rank_nn",
                rank=rank,
                t1_df=t1_df,
                t2_df=t2_df,
                t1_texts=t1_texts,
                t2_texts=t2_texts,
                t2_indices=top_indices[:, rank - 1],
                similarities=top_similarities[:, rank - 1],
                output_dir=output_dir,
                output_prefix=args.output_prefix,
                drop_t2_cols=args.drop_t2_cols,
                t2_prefix=args.t2_prefix,
                encoding=args.encoding,
            )
        )

    if args.include_random:
        rng = np.random.default_rng(args.seed)
        random_indices = rng.integers(0, len(t2_df), size=len(t1_df), dtype=np.int64)
        random_similarities = np.sum(
            t1_embeddings * t2_embeddings[random_indices], axis=1
        )
        outputs.append(
            save_match(
                name="random",
                match_type="random",
                rank=None,
                t1_df=t1_df,
                t2_df=t2_df,
                t1_texts=t1_texts,
                t2_texts=t2_texts,
                t2_indices=random_indices,
                similarities=random_similarities,
                output_dir=output_dir,
                output_prefix=args.output_prefix,
                drop_t2_cols=args.drop_t2_cols,
                t2_prefix=args.t2_prefix,
                encoding=args.encoding,
            )
        )

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "t1": str(Path(args.t1).resolve()),
        "t1_col": args.t1_col,
        "t2": str(Path(args.t2).resolve()),
        "t2_col": args.t2_col,
        "model": args.model,
        "pooling": "last_hidden_state_cls",
        "search": "faiss.IndexFlatIP_exact_cosine",
        "ranks": args.ranks,
        "include_random": args.include_random,
        "random_seed": args.seed,
        "drop_t2_cols": args.drop_t2_cols,
        "runtime_seconds": time.perf_counter() - start_time,
        "outputs": outputs,
    }
    metadata_path = output_dir / f"{args.output_prefix}_matching_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print(f"Metadata: {metadata_path}")
    print(f"Runtime: {metadata['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate exact rank-NN and random enriched-table ablations."
    )
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t1_col", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--t2_col", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_prefix", default="enriched")
    parser.add_argument("--ranks", type=parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--include_random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--drop_t2_cols", type=parse_str_list, default=[])
    parser.add_argument("--t2_prefix", default="t2_")
    main(parser.parse_args())
