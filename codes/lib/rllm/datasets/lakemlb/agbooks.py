from typing import Optional, List
import os
import os.path as osp
import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset

from .local_archive import extract_local_archive


class AGBooksDataset(Dataset):
    r"""AGBooksDataset is a tabular dataset designed for weakly related
    table scenarios in Data Lake(House) settings.

    The dataset comprises an Amazon task table, a Goodreads auxiliary table,
    and the corresponding data-augmentation (DA) and feature-augmentation (FA)
    tables. The default task is to predict Amazon book categories. The
    anonymous artifact contains a deterministic 10% example of the full-scale
    dataset.

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
            Size        10,000      11

        Table2: goodreads
        -----------------
            Statics:
            Name        Records     Features
            Size        10,000      20

        Table3: agbooks_da
        ------------------
            Statics:
            Name        Records     Features
            Size        12,100      31

        Table4: agbooks_fa
        ------------------
            Statics:
            Name        Records     Features
            Size        10,000      31
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
            "agbooks_da.csv",
            "agbooks_fa.csv",
            "amazon_mask.pt",
            "mask_da.pt",
            "mapping.csv",
        ]

    @property
    def processed_filenames(self):
        return [
            "amazon_data.pt",
            "goodreads_data.pt",
            "agbooks_da_data.pt",
            "agbooks_fa_data.pt",
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

        # Merged Data (FA)
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
        agbooks_fa_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=agbooks_fa_df,
            col_types=col_types,
            target_col="categories",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[3])

        # Merged Data (DA)
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
        ).save(self.processed_paths[2])

    def download(self):
        extract_local_archive(
            "agbooks", "join", self.raw_dir, self.raw_filenames
        )

    def __len__(self):
        return 4

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]
