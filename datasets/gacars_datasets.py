from typing import Optional, List
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset
from rllm.utils.download import download_url
from rllm.utils.extract import extract_zip
from rllm.transforms.table_transforms import TabTransformerTransform


class GACarsDataset(Dataset):
    # url = "https://github.com/FeiyuPan/LakeMLB_datasets/raw/refs/heads/main/datasets/Crash20K.zip"

    def __init__(self, cached_dir: str, force_reload: Optional[bool] = False, transform=None, device=None) -> None:
        self.name = "table_gacars"
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
        return ["german.csv", "mask.pt"]

    @property
    def processed_filenames(self):
        return ["gacars_data_1.pt"]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[0])
        masks_path = osp.join(self.raw_dir, self.raw_filenames[1])

        df_temp = pd.read_csv(csv_path, nrows=0, encoding='latin-1', low_memory=False) #'utf-8-sig'
        df_temp.columns = df_temp.columns.str.strip()
        
        col_types = {
            "id":ColType.NUMERICAL,
            "brand":ColType.CATEGORICAL,
            "model":ColType.CATEGORICAL,
            "color":ColType.CATEGORICAL,
            "registration_date":ColType.CATEGORICAL,
            "year":ColType.NUMERICAL,
            "price_in_euro":ColType.CATEGORICAL,
            "power_kw": ColType.NUMERICAL,
            "power_ps": ColType.NUMERICAL,
            "transmission_type": ColType.CATEGORICAL,
            "fuel_type": ColType.CATEGORICAL,
            "fuel_consumption_l_100km": ColType.CATEGORICAL,
            "fuel_consumption_g_km": ColType.CATEGORICAL,
            "mileage_in_km": ColType.NUMERICAL,
            "offer_description": ColType.CATEGORICAL,
            # aux table cols
            # "Brand": ColType.CATEGORICAL, #da
            # "Year": ColType.NUMERICAL, #da
            # "Model": ColType.CATEGORICAL, #da
            # "Car/Suv": ColType.CATEGORICAL,
            # "Title": ColType.CATEGORICAL,
            # "UsedOrNew": ColType.CATEGORICAL,
            # "Transmission": ColType.CATEGORICAL,
            # "Engine": ColType.CATEGORICAL,
            # "DriveType": ColType.CATEGORICAL,
            # "FuelType": ColType.CATEGORICAL, #da
            # "FuelConsumption": ColType.CATEGORICAL,
            # "Kilometres": ColType.NUMERICAL,
            # "ColourExtInt": ColType.CATEGORICAL,
            # "Location": ColType.CATEGORICAL,
            # "CylindersinEngine": ColType.CATEGORICAL,
            # "BodyType": ColType.CATEGORICAL,
            # "Doors": ColType.CATEGORICAL,
            # "Seats": ColType.CATEGORICAL, 
            # "Price": ColType.CATEGORICAL, #da
        }
        
        dtype_dict = {}
        for col_name, col_type in col_types.items():
            if col_type == ColType.CATEGORICAL and col_name in df_temp.columns:
                dtype_dict[col_name] = str
        
        df = pd.read_csv(csv_path, encoding='latin-1', low_memory=False, dtype=dtype_dict)
        df.columns = df.columns.str.strip()
        
        for col_name, col_type in col_types.items():
            if col_type == ColType.CATEGORICAL and col_name in df.columns:
                df[col_name] = df[col_name].astype(str)
                df[col_name] = df[col_name].replace(['nan', 'None', 'NaN', '<NA>'], 'Missing')
            elif col_type == ColType.NUMERICAL and col_name in df.columns:
                df[col_name] = df[col_name].fillna(0)
        
        masks = torch.load(masks_path, weights_only=False)
        data = TableData(
            df=df,
            col_types=col_types,
            target_col="price_in_euro",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        )
        data.save(self.processed_paths[0])

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        all_files_exist = all(
            osp.exists(osp.join(self.raw_dir, fname)) 
            for fname in self.raw_filenames
        )
        if all_files_exist:
            print(f"Raw files already exist in {self.raw_dir}, skipping download.")
            return
        if not hasattr(self, 'url'):
            raise AttributeError(
                f"Raw files not found in {self.raw_dir} and no download URL is configured. "
                f"Please manually place the following files in {self.raw_dir}: {self.raw_filenames}"
            )
        
        path = download_url(self.url, self.raw_dir, "Crash20K.zip")
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError
        return self.data_list[0]
