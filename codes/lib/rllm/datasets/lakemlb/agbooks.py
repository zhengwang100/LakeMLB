from typing import Optional, List
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset


class AGBooksDataset(Dataset):
    r"""AGBooksDataset is a tabular dataset designed for weakly related
    table scenarios in Data Lake(House) settings.

    The dataset comprises three book metadata tables: a task table from Amazon
    books, an auxiliary table from Goodreads books, and an enriched Amazon table
    with Goodreads features. The default task is to predict Amazon book categories.

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        force_reload (bool): If set to `True`, this dataset will be re-process again.
        transform: Optional transform to be applied on the data.
        device: Optional device to move the transformed data to.

    .. parsed-literal::

        Table1: amazon
        --------------
            Statics:
            Name        Records     Features
            Size        100,000     11

        Table2: goodreads
        -----------------
            Statics:
            Name        Records     Features
            Size        100,000     20

        Table3: amazon_enriched
        -----------------------
            Statics:
            Name        Records     Features
            Size        100,000     31
    """

    def __init__(
        self,
        cached_dir: str,
        force_reload: Optional[bool] = False,
        transform=None,
        device=None,
    ) -> None:
        self.name = "table_agbooks"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        self.data_list: List[TableData] = [
            TableData.load(self.processed_paths[0]),
            TableData.load(self.processed_paths[1]),
            TableData.load(self.processed_paths[2]),
            TableData.load(self.processed_paths[3]),
            TableData.load(self.processed_paths[4]),
            TableData.load(self.processed_paths[5]),
            TableData.load(self.processed_paths[6]),
            TableData.load(self.processed_paths[7]),
            TableData.load(self.processed_paths[8]),
            TableData.load(self.processed_paths[9]),
            TableData.load(self.processed_paths[10]),
        ]
        self.transform = transform
        if self.transform is not None:
            for i, data in enumerate(self.data_list):
                self.data_list[i] = (
                    self.transform(data).to(device)
                    if device is not None
                    else self.transform(data)
                )

    @property
    def raw_filenames(self):
        return [
            "amazon.csv",
            "goodreads.csv",
            "amazon_enriched.csv",
            "agbooks_da.csv",
            "amazon_mask.pt",
            "mask_da.pt",
            "agbooks_1nn.csv",
            "agbooks_2nn.csv",
            "agbooks_4nn.csv",
            "agbooks_8nn.csv",
            "agbooks_random.csv",
        ]

    @property
    def processed_filenames(self):
        return [
            "amazon_data.pt",
            "goodreads_data.pt",
            "amazon_enriched_data.pt",
            "agbooks_da_data.pt",
            "amazon_no_features_data.pt",
            "amazon_no_features_10k_data.pt",
            "agbooks_1nn_data.pt",
            "agbooks_2nn_data.pt",
            "agbooks_4nn_data.pt",
            "agbooks_8nn_data.pt",
            "agbooks_random_data.pt",
        ]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        # Amazon Books Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[0])
        col_types = {
            "parent_asin": ColType.CATEGORICAL,
            "title": ColType.CATEGORICAL,
            "main_category": ColType.CATEGORICAL,
            "average_rating": ColType.NUMERICAL,
            "rating_number": ColType.NUMERICAL,
            "price": ColType.CATEGORICAL,
            "store": ColType.CATEGORICAL,
            "features": ColType.CATEGORICAL,
            "description": ColType.CATEGORICAL,
            "details": ColType.CATEGORICAL,
            "categories": ColType.CATEGORICAL,
        }
        amazon_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=amazon_df,
            col_types=col_types,
            target_col="categories",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        # Amazon Books Data without the long free-text `features` column.
        amazon_no_features_df = amazon_df.drop(columns=["features"])
        amazon_no_features_col_types = {
            key: value for key, value in col_types.items() if key != "features"
        }
        TableData(
            df=amazon_no_features_df,
            col_types=amazon_no_features_col_types,
            target_col="categories",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[4])

        # A stratified 10k subset of the no-features Amazon table.
        rng = np.random.default_rng(20260702)
        train_mask_np = masks["train_mask"].cpu().numpy()
        val_mask_np = masks["val_mask"].cpu().numpy()
        test_mask_np = masks["test_mask"].cpu().numpy()
        sample_indices = []
        split_specs = [
            (train_mask_np, 175),
            (val_mask_np, 25),
            (test_mask_np, 50),
        ]
        for label in sorted(amazon_no_features_df["categories"].dropna().unique()):
            label_mask = amazon_no_features_df["categories"].eq(label).to_numpy()
            for split_mask, n_samples in split_specs:
                candidates = np.flatnonzero(label_mask & split_mask)
                if len(candidates) < n_samples:
                    raise ValueError(
                        f"Not enough rows for label={label!r}: "
                        f"need {n_samples}, found {len(candidates)}."
                    )
                sample_indices.extend(rng.choice(candidates, size=n_samples, replace=False).tolist())
        sampled_df = amazon_no_features_df.iloc[sample_indices].reset_index(drop=True)
        sampled_train_mask = torch.zeros(len(sampled_df), dtype=torch.bool)
        sampled_val_mask = torch.zeros(len(sampled_df), dtype=torch.bool)
        sampled_test_mask = torch.zeros(len(sampled_df), dtype=torch.bool)
        for start in range(0, len(sampled_df), 250):
            sampled_train_mask[start:start + 175] = True
            sampled_val_mask[start + 175:start + 200] = True
            sampled_test_mask[start + 200:start + 250] = True
        TableData(
            df=sampled_df,
            col_types=amazon_no_features_col_types,
            target_col="categories",
            train_mask=sampled_train_mask,
            val_mask=sampled_val_mask,
            test_mask=sampled_test_mask,
        ).save(self.processed_paths[5])

        # Goodreads Books Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[1])
        col_types = {
            "book_id": ColType.CATEGORICAL,
            "title": ColType.CATEGORICAL,
            "title_without_series": ColType.CATEGORICAL,
            "average_rating": ColType.NUMERICAL,
            "ratings_count": ColType.NUMERICAL,
            "text_reviews_count": ColType.NUMERICAL,
            "publication_year": ColType.NUMERICAL,
            "publication_month": ColType.NUMERICAL,
            "publication_day": ColType.NUMERICAL,
            "publisher": ColType.CATEGORICAL,
            "num_pages": ColType.NUMERICAL,
            "language_code": ColType.CATEGORICAL,
            "format": ColType.CATEGORICAL,
            "isbn": ColType.CATEGORICAL,
            "isbn13": ColType.CATEGORICAL,
            "is_ebook": ColType.CATEGORICAL,
            "kindle_asin": ColType.CATEGORICAL,
            "author_ids": ColType.CATEGORICAL,
            "similar_books": ColType.CATEGORICAL,
            "description": ColType.CATEGORICAL,
        }
        goodreads_df = pd.read_csv(csv_path, low_memory=False)
        TableData(
            df=goodreads_df,
            col_types=col_types,
            target_col=None,
        ).save(self.processed_paths[1])

        # Amazon Enriched Books Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[2])
        col_types = {
            "parent_asin": ColType.CATEGORICAL,
            "title": ColType.CATEGORICAL,
            "main_category": ColType.CATEGORICAL,
            "average_rating": ColType.NUMERICAL,
            "rating_number": ColType.NUMERICAL,
            "price": ColType.CATEGORICAL,
            "store": ColType.CATEGORICAL,
            "features": ColType.CATEGORICAL,
            "description": ColType.CATEGORICAL,
            "details": ColType.CATEGORICAL,
            "categories": ColType.CATEGORICAL,
            "book_id": ColType.CATEGORICAL,
            "goodreads_title": ColType.CATEGORICAL,
            "title_without_series": ColType.CATEGORICAL,
            "goodreads_average_rating": ColType.NUMERICAL,
            "ratings_count": ColType.NUMERICAL,
            "text_reviews_count": ColType.NUMERICAL,
            "publication_year": ColType.NUMERICAL,
            "publication_month": ColType.NUMERICAL,
            "publication_day": ColType.NUMERICAL,
            "publisher": ColType.CATEGORICAL,
            "num_pages": ColType.NUMERICAL,
            "language_code": ColType.CATEGORICAL,
            "format": ColType.CATEGORICAL,
            "isbn": ColType.CATEGORICAL,
            "isbn13": ColType.CATEGORICAL,
            "is_ebook": ColType.CATEGORICAL,
            "kindle_asin": ColType.CATEGORICAL,
            "author_ids": ColType.CATEGORICAL,
            "similar_books": ColType.CATEGORICAL,
            "goodreads_description": ColType.CATEGORICAL,
        }
        enriched_col_types = col_types.copy()
        amazon_enriched_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=amazon_enriched_df,
            col_types=col_types,
            target_col="categories",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[2])

        # Merged Data(DA)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[3])
        col_types = {
            "parent_asin": ColType.CATEGORICAL,
            "title": ColType.CATEGORICAL,
            "main_category": ColType.CATEGORICAL,
            "average_rating": ColType.NUMERICAL,
            "rating_number": ColType.NUMERICAL,
            "price": ColType.CATEGORICAL,
            "store": ColType.CATEGORICAL,
            "features": ColType.CATEGORICAL,
            "description": ColType.CATEGORICAL,
            "details": ColType.CATEGORICAL,
            "categories": ColType.CATEGORICAL,
            "book_id": ColType.CATEGORICAL,
            "goodreads_title": ColType.CATEGORICAL,
            "title_without_series": ColType.CATEGORICAL,
            "goodreads_average_rating": ColType.NUMERICAL,
            "ratings_count": ColType.NUMERICAL,
            "text_reviews_count": ColType.NUMERICAL,
            "publication_year": ColType.NUMERICAL,
            "publication_month": ColType.NUMERICAL,
            "publication_day": ColType.NUMERICAL,
            "publisher": ColType.CATEGORICAL,
            "num_pages": ColType.NUMERICAL,
            "language_code": ColType.CATEGORICAL,
            "format": ColType.CATEGORICAL,
            "isbn": ColType.CATEGORICAL,
            "isbn13": ColType.CATEGORICAL,
            "is_ebook": ColType.CATEGORICAL,
            "kindle_asin": ColType.CATEGORICAL,
            "author_ids": ColType.CATEGORICAL,
            "similar_books": ColType.CATEGORICAL,
            "goodreads_description": ColType.CATEGORICAL,
        }
        agbooks_da_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[5])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=agbooks_da_df,
            col_types=col_types,
            target_col="categories",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[3])

        # Title-matching rank and random ablation tables.
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        for raw_idx, processed_idx in zip(range(6, 11), range(6, 11)):
            csv_path = osp.join(self.raw_dir, self.raw_filenames[raw_idx])
            matched_df = pd.read_csv(csv_path, low_memory=False)
            TableData(
                df=matched_df,
                col_types=enriched_col_types,
                target_col="categories",
                train_mask=masks["train_mask"],
                val_mask=masks["val_mask"],
                test_mask=masks["test_mask"],
            ).save(self.processed_paths[processed_idx])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        raise RuntimeError(
            "AGBooksDataset raw files are not bundled for download. "
            f"Please place {self.raw_filenames} under {self.raw_dir}."
        )

    def __len__(self):
        return 11

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]
