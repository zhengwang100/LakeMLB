import os
import os.path as osp
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, osp.join(osp.dirname(__file__), "..", "lib"))

from transtab.dataset import create_dataset_config
from rllm.datasets import (
    MSTrafficDataset,
    NCBuildingDataset,
    NNStocksDataset,
    DSMusicDataset,
    NCTaxiDataset,
    AGBooksDataset,
)


DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "data"))

DATASET_REGISTRY = {
    "mstraffic": MSTrafficDataset,
    "ncbuilding": NCBuildingDataset,
    "nnstocks": NNStocksDataset,
    "dsmusic": DSMusicDataset,
    "nctaxi": NCTaxiDataset,
    "agbooks": AGBooksDataset,
}

TABLE_TAGS = {
    "mstraffic": {
        0: "mstraffic_maryland",
        1: "mstraffic_seattle",
        2: "mstraffic_da",
        3: "mstraffic_fa",
    },
    "ncbuilding": {
        0: "ncbuilding_newyork",
        1: "ncbuilding_chicago",
        2: "ncbuilding_da",
        3: "ncbuilding_fa",
    },
    "nctaxi": {
        0: "nctaxi_newyork_taxi",
        1: "nctaxi_chicago_taxi",
        2: "nctaxi_da",
        3: "nctaxi_fa",
    },
    "dsmusic": {
        0: "dsmusic_discogs",
        1: "dsmusic_spotify",
        2: "dsmusic_da",
        3: "dsmusic_fa",
    },
    "agbooks": {
        0: "agbooks_amazon",
        1: "agbooks_goodreads",
        2: "agbooks_da",
        3: "agbooks_fa",
    },
    "nnstocks": {
        0: "nnstocks_nnlist",
        1: "nnstocks_nnwiki",
        2: "nnstocks_da",
        3: "nnstocks_fa",
    },
}


@dataclass
class PreparedTable:
    dataset_name: str
    table_idx: int
    tag: str
    csv_dir: str
    csv_name: str
    mask_path: str
    config: dict
    target_col: Optional[str]
    num_classes: Optional[int]


def get_table_tag(dataset_name: str, table_idx: int) -> str:
    return TABLE_TAGS.get(dataset_name, {}).get(table_idx, f"{dataset_name}_table{table_idx}")


def load_table(dataset_name: str, table_idx: int):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choices: {sorted(DATASET_REGISTRY)}")
    dataset = DATASET_REGISTRY[dataset_name](cached_dir=DATA_DIR, force_reload=False)
    if table_idx < 0 or table_idx >= len(dataset.data_list):
        raise IndexError(
            f"table_idx={table_idx} out of range for {dataset_name}; "
            f"available range is 0..{len(dataset.data_list) - 1}."
        )
    return dataset[table_idx]


def _mask_to_numpy(mask):
    if isinstance(mask, torch.Tensor):
        return mask.cpu().numpy()
    return np.asarray(mask)


def _random_masks(num_rows: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_rows)
    train_end = int(num_rows * 0.8)
    val_end = int(num_rows * 0.9)
    train_mask = np.zeros(num_rows, dtype=bool)
    val_mask = np.zeros(num_rows, dtype=bool)
    test_mask = np.zeros(num_rows, dtype=bool)
    train_mask[idx[:train_end]] = True
    val_mask[idx[train_end:val_end]] = True
    test_mask[idx[val_end:]] = True
    return {
        "train_mask": torch.as_tensor(train_mask),
        "val_mask": torch.as_tensor(val_mask),
        "test_mask": torch.as_tensor(test_mask),
    }


def _get_masks(table, seed: int) -> dict:
    if all(hasattr(table, name) for name in ("train_mask", "val_mask", "test_mask")):
        return {
            "train_mask": torch.as_tensor(_mask_to_numpy(table.train_mask), dtype=torch.bool),
            "val_mask": torch.as_tensor(_mask_to_numpy(table.val_mask), dtype=torch.bool),
            "test_mask": torch.as_tensor(_mask_to_numpy(table.test_mask), dtype=torch.bool),
        }
    return _random_masks(len(table.df), seed)


def prepare_table(
    dataset_name: str,
    table_idx: int,
    work_dir: str,
    role: str,
    seed: int,
    require_target: bool,
    use_target: bool,
) -> PreparedTable:
    table = load_table(dataset_name, table_idx)
    target_col = table.target_col if use_target else None
    if require_target and target_col is None:
        raise ValueError(f"{dataset_name}[{table_idx}] has no target_col, cannot use it as a supervised table.")

    df = table.df.copy()
    col_types = dict(table.col_types)
    if not use_target and table.target_col is not None and table.target_col in df.columns:
        df = df.drop(columns=[table.target_col])
        col_types.pop(table.target_col, None)

    os.makedirs(work_dir, exist_ok=True)
    csv_name = f"{role}.csv"
    csv_path = osp.join(work_dir, csv_name)
    mask_path = osp.join(work_dir, f"{role}_mask.pt")
    df.to_csv(csv_path, index=False)
    torch.save(_get_masks(table, seed), mask_path)

    config = create_dataset_config(
        col_types_dict=col_types,
        target_col=target_col,
        mask_path=mask_path,
    )
    num_classes = None
    if target_col is not None:
        num_classes = int(table.df[target_col].nunique(dropna=True))

    return PreparedTable(
        dataset_name=dataset_name,
        table_idx=table_idx,
        tag=get_table_tag(dataset_name, table_idx),
        csv_dir=work_dir,
        csv_name=csv_name,
        mask_path=mask_path,
        config=config,
        target_col=target_col,
        num_classes=num_classes,
    )
