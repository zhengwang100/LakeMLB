"""
Generic BERT + FAISS HNSW 1-NN matching for two CSV tables.

The script does no domain-specific preprocessing. It directly embeds the
specified text columns with BERT [CLS], normalizes embeddings with L2 norm,
builds a FAISS HNSW index over T2, and retrieves the nearest T2 row for each
T1 row.

conda activate lake

python process_book/ann_match_generic.py \
  --t1 process_book/amazon_books.csv \
  --t1_col title \
  --t2 process_book/goodreads_books_full.csv \
  --t2_col title \
  --bert bert-base-uncased \
  --device cuda:0 \
  --batch_size 64 \
  --max_length 128 \
  --map_out process_book/ann_title_mapping.csv \
  --t2_match process_book/goodreads_ann_matched_subset.csv \
  --t1_out process_book/amazon_ann_enriched.csv

Join配对,使用的通用脚本。 即join-FA
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
LOCAL_MODEL_CANDIDATES = [
    SCRIPT_DIR / "models" / "bert-base-uncased",
    SCRIPT_DIR.parents[1] / "lib" / "models" / "bert-base-uncased",
]
LOCAL_DEFAULT_MODEL = next(
    (path for path in LOCAL_MODEL_CANDIDATES if path.exists()), None
)
DEFAULT_MODEL = str(LOCAL_DEFAULT_MODEL or "bert-base-uncased")


def load_csv(path, text_col, encoding):
    df = pd.read_csv(path, encoding=encoding)
    if text_col not in df.columns:
        raise ValueError(f"Column {text_col!r} not found in {path}. Columns: {list(df.columns)}")
    texts = df[text_col].fillna("").astype(str).tolist()
    return texts, df


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available")
    return device


def compute_cls_embeddings(texts, tokenizer, model, device, batch_size, max_length):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
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
    return index, embeddings


def search_1nn(index, query_embeddings):
    query_embeddings = np.ascontiguousarray(query_embeddings.astype("float32", copy=False))
    faiss.normalize_L2(query_embeddings)
    distances, indices = index.search(query_embeddings, 1)
    cosine_similarity = 1.0 - distances / 2.0
    return cosine_similarity.reshape(-1), indices.reshape(-1), query_embeddings


def rename_t2_columns_for_enriched(t1_columns, t2_columns, t2_prefix):
    t1_columns = set(t1_columns)
    renamed = []
    for col in t2_columns:
        if col in t1_columns:
            renamed.append(f"{t2_prefix}{col}")
        else:
            renamed.append(col)
    return renamed


def save_outputs(
    t1_df,
    t2_df,
    t1_texts,
    t2_texts,
    t2_indices,
    similarities,
    map_out,
    t2_match_out,
    t1_out,
    t2_prefix,
):
    matched_t2 = t2_df.iloc[t2_indices].reset_index(drop=True)
    matched_t2.to_csv(t2_match_out, index=False, encoding="utf-8-sig")

    mapping = pd.DataFrame(
        {
            "t1_index": np.arange(len(t1_df), dtype=np.int64),
            "t2_index": t2_indices.astype(np.int64),
            "t1_text": t1_texts,
            "t2_text": [t2_texts[i] for i in t2_indices],
            "cosine_similarity": similarities,
        }
    )
    mapping.to_csv(map_out, index=False, encoding="utf-8-sig")

    enriched_t2 = matched_t2.copy()
    enriched_t2.columns = rename_t2_columns_for_enriched(t1_df.columns, enriched_t2.columns, t2_prefix)
    enriched = pd.concat([t1_df.reset_index(drop=True), enriched_t2], axis=1)
    if enriched.columns.duplicated().any():
        duplicates = enriched.columns[enriched.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate columns after enrichment: {duplicates}")
    enriched.to_csv(t1_out, index=False, encoding="utf-8-sig")

    return mapping, matched_t2, enriched


def print_similarity_stats(similarities):
    print("\nMatch statistics:")
    print(f"  mean similarity:   {float(np.mean(similarities)):.4f}")
    print(f"  median similarity: {float(np.median(similarities)):.4f}")
    print(f"  min similarity:    {float(np.min(similarities)):.4f}")
    print(f"  max similarity:    {float(np.max(similarities)):.4f}")
    print(f"  > 0.8:             {int(np.sum(similarities > 0.8))}")
    print(f"  0.5 - 0.8:         {int(np.sum((similarities >= 0.5) & (similarities <= 0.8)))}")
    print(f"  < 0.5:             {int(np.sum(similarities < 0.5))}")


def main(args):
    os.makedirs(os.path.dirname(args.map_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.t2_match) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.t1_out) or ".", exist_ok=True)

    print("=" * 72)
    print("Generic BERT [CLS] + FAISS HNSW 1-NN Matching")
    print("=" * 72)

    print("\n[1/6] Loading CSV files")
    t1_texts, t1_df = load_csv(args.t1, args.t1_col, args.encoding)
    t2_texts, t2_df = load_csv(args.t2, args.t2_col, args.encoding)
    print(f"  T1 rows: {len(t1_df)}")
    print(f"  T2 rows: {len(t2_df)}")
    print(f"  T1 match column: {args.t1_col}")
    print(f"  T2 match column: {args.t2_col}")

    print("\n[2/6] Loading BERT model")
    device = resolve_device(args.device)
    print(f"  model: {args.bert}")
    print(f"  device: {device}")
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    model = BertModel.from_pretrained(args.bert).to(device)

    print("\n[3/6] Computing T2 embeddings")
    emb_t2 = compute_cls_embeddings(
        t2_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )

    print("\n[4/6] Computing T1 embeddings")
    emb_t1 = compute_cls_embeddings(
        t1_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )

    print("\n[5/6] Building FAISS HNSW index and searching 1-NN")
    index, _ = build_hnsw_index(
        emb_t2,
        args.hnsw_m,
        args.hnsw_ef_construction,
        args.hnsw_ef_search,
    )
    similarities, t2_indices, _ = search_1nn(index, emb_t1)

    print("\n[6/6] Saving outputs")
    mapping, matched_t2, enriched = save_outputs(
        t1_df=t1_df,
        t2_df=t2_df,
        t1_texts=t1_texts,
        t2_texts=t2_texts,
        t2_indices=t2_indices,
        similarities=similarities,
        map_out=args.map_out,
        t2_match_out=args.t2_match,
        t1_out=args.t1_out,
        t2_prefix=args.t2_prefix,
    )

    print(f"  mapping:          {args.map_out} ({len(mapping)} rows)")
    print(f"  matched T2 subset:{args.t2_match} ({len(matched_t2)} rows)")
    print(f"  enriched T1:      {args.t1_out} ({len(enriched)} rows, {len(enriched.columns)} columns)")
    print_similarity_stats(similarities)
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic BERT [CLS] + FAISS HNSW 1-NN matcher for two CSV files."
    )

    parser.add_argument("--t1", required=True, help="Path to T1 CSV")
    parser.add_argument("--t1_col", required=True, help="T1 text column used for matching")
    parser.add_argument("--t2", required=True, help="Path to T2 CSV")
    parser.add_argument("--t2_col", required=True, help="T2 text column used for matching")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")

    parser.add_argument("--bert", default=DEFAULT_MODEL, help="BERT model name or local path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--batch_size", type=int, default=32, help="BERT embedding batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Tokenizer max sequence length")

    parser.add_argument("--hnsw_m", type=int, default=32, help="FAISS HNSW M parameter")
    parser.add_argument(
        "--hnsw_ef_construction",
        type=int,
        default=64,
        help="FAISS HNSW efConstruction parameter",
    )
    parser.add_argument(
        "--hnsw_ef_search",
        type=int,
        default=64,
        help="FAISS HNSW efSearch parameter",
    )

    parser.add_argument("--map_out", required=True, help="Output mapping CSV")
    parser.add_argument("--t2_match", required=True, help="Output matched T2 subset CSV")
    parser.add_argument("--t1_out", required=True, help="Output enriched T1 CSV")
    parser.add_argument(
        "--t2_prefix",
        default="t2_",
        help="Prefix for T2 columns that conflict with T1 columns in enriched output",
    )

    main(parser.parse_args())
