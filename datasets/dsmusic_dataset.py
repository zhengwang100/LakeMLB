from typing import Optional, List
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.transforms.table_transforms import TabTransformerTransform

class DSMusicDataset(Dataset):
    def __init__(self, cached_dir: str, force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "table_dsmusic"
        root = os.path.join(cached_dir, self.name)
        super().__init__(root, force_reload=force_reload)

        self.data_list: List[TableData] = [
            TableData.load(self.processed_paths[0])
        ]
        self.transform = transform
        if self.transform is not None:
            if device is not None:
                self.data_list[0] = self.transform(self.data_list[0]).to(device)
            else:
                self.data_list[0] = self.transform(self.data_list[0])

    @property
    def raw_filenames(self):
        return ["dsmusic_da.csv", "mask_da.pt"]


    @property
    def processed_filenames(self):
        return ["music_data_da.pt"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[0])
        masks_path = osp.join(self.raw_dir, self.raw_filenames[1])

        df = pd.read_csv(csv_path)

        col_types = {
            # #T1
            "title": ColType.CATEGORICAL,
            "release_year": ColType.NUMERICAL,
            "artists": ColType.CATEGORICAL,
            "genres": ColType.CATEGORICAL,
            "region": ColType.CATEGORICAL,
            #merged
            "track_id": ColType.CATEGORICAL,
            "album_name": ColType.CATEGORICAL,
            "popularity": ColType.NUMERICAL,
            "duration_ms": ColType.NUMERICAL,
            "explicit": ColType.CATEGORICAL,
            "danceability": ColType.NUMERICAL,
            "energy": ColType.NUMERICAL,
            "key": ColType.NUMERICAL,
            "loudness": ColType.NUMERICAL,
            "mode": ColType.NUMERICAL,
            "speechiness": ColType.NUMERICAL,
            "acousticness": ColType.NUMERICAL,
            "instrumentalness": ColType.NUMERICAL,
            "liveness": ColType.NUMERICAL,
            "valence": ColType.NUMERICAL,
            "tempo": ColType.NUMERICAL,
            "time_signature": ColType.NUMERICAL,
            "track_genre": ColType.CATEGORICAL,
        }

        masks = torch.load(masks_path, weights_only=False)
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="genres",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        )
        data.save(self.processed_paths[0])

    def download(self):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[0]
