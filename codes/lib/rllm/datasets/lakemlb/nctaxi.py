from typing import Optional, List
import os
import os.path as osp
import pandas as pd
import torch

from rllm.types import ColType
from rllm.data.table_data import TableData
from rllm.datasets.dataset import Dataset

from .local_archive import extract_local_archive


class NCTaxiDataset(Dataset):
    r"""NCTaxiDataset is a tabular dataset designed for weakly related
    table scenarios in Data Lake(House) settings.

    The dataset comprises two taxi trip tables: a task table from New York
    taxi trips and an auxiliary table from Chicago taxi trips. The default
    task is to predict the drop-off location of New York taxi trips. The
    anonymous artifact contains a deterministic 10% example of the full-scale
    dataset.

    Args:
        cached_dir (str): Root directory where dataset should be saved.
        force_reload (bool): If set to `True`, this dataset will be re-process again.
        transform: Optional transform to be applied on the data.
        device: Optional device to move the transformed data to.

    .. parsed-literal::

        Table1: newyork_taxi
        --------------------
            Statics:
            Name        Records     Features
            Size        10,000      19

        Table2: chicago_taxi
        --------------------
            Statics:
            Name        Records     Features
            Size        10,000      19

        Table3: nctaxi_da
        -----------------
            Statics:
            Name        Records     Features
            Size        12,100      36

        Table4: nctaxi_fa
        -----------------
            Statics:
            Name        Records     Features
            Size        10,000      38
    """

    def __init__(
        self,
        cached_dir: str,
        force_reload: Optional[bool] = False,
        transform=None,
        device=None,
    ) -> None:
        self.name = "table_nctaxi"
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
            "newyork_taxi.csv",
            "chicago_taxi.csv",
            "nctaxi_da.csv",
            "nctaxi_fa.csv",
            "newyork_taxi_mask.pt",
            "mask_da.pt",
            "mapping.csv",
        ]

    @property
    def processed_filenames(self):
        return [
            "newyork_taxi_data.pt",
            "chicago_taxi_data.pt",
            "nctaxi_da_data.pt",
            "nctaxi_fa_data.pt",
        ]

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        # New York Taxi Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[0])
        col_types = {
            "vendorid": ColType.CATEGORICAL,
            "tpep_pickup_datetime": ColType.CATEGORICAL,
            "tpep_dropoff_datetime": ColType.CATEGORICAL,
            "passenger_count": ColType.NUMERICAL,
            "trip_distance": ColType.NUMERICAL,
            "ratecodeid": ColType.CATEGORICAL,
            "store_and_fwd_flag": ColType.CATEGORICAL,
            "pulocationid": ColType.CATEGORICAL,
            "dolocationid": ColType.CATEGORICAL,
            "payment_type": ColType.CATEGORICAL,
            "fare_amount": ColType.NUMERICAL,
            "extra": ColType.NUMERICAL,
            "mta_tax": ColType.NUMERICAL,
            "tip_amount": ColType.NUMERICAL,
            "tolls_amount": ColType.NUMERICAL,
            "improvement_surcharge": ColType.NUMERICAL,
            "total_amount": ColType.NUMERICAL,
            "congestion_surcharge": ColType.NUMERICAL,
            "airport_fee": ColType.NUMERICAL,
        }
        newyork_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=newyork_df,
            col_types=col_types,
            target_col="dolocationid",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[0])

        # Chicago Taxi Data
        csv_path = osp.join(self.raw_dir, self.raw_filenames[1])
        col_types = {
            "trip_id": ColType.CATEGORICAL,
            "taxi_id": ColType.CATEGORICAL,
            "trip_start_timestamp": ColType.CATEGORICAL,
            "trip_end_timestamp": ColType.CATEGORICAL,
            "trip_seconds": ColType.NUMERICAL,
            "trip_miles": ColType.NUMERICAL,
            "pickup_census_tract": ColType.CATEGORICAL,
            "pickup_community_area": ColType.CATEGORICAL,
            "dropoff_community_area": ColType.CATEGORICAL,
            "fare": ColType.NUMERICAL,
            "tips": ColType.NUMERICAL,
            "tolls": ColType.NUMERICAL,
            "extras": ColType.NUMERICAL,
            "trip_total": ColType.NUMERICAL,
            "payment_type": ColType.CATEGORICAL,
            "company": ColType.CATEGORICAL,
            "pickup_centroid_latitude": ColType.NUMERICAL,
            "pickup_centroid_longitude": ColType.NUMERICAL,
            "pickup_centroid_location": ColType.CATEGORICAL,
        }
        chicago_df = pd.read_csv(csv_path, low_memory=False)
        TableData(
            df=chicago_df,
            col_types=col_types,
            target_col="dropoff_community_area",
        ).save(self.processed_paths[1])

        # Merged Data(DA)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[2])
        col_types = {
            "vendorid": ColType.CATEGORICAL,
            "tpep_pickup_datetime": ColType.CATEGORICAL,
            "tpep_dropoff_datetime": ColType.CATEGORICAL,
            "passenger_count": ColType.NUMERICAL,
            "trip_distance": ColType.NUMERICAL,
            "ratecodeid": ColType.CATEGORICAL,
            "store_and_fwd_flag": ColType.CATEGORICAL,
            "pulocationid": ColType.CATEGORICAL,
            "dolocationid": ColType.CATEGORICAL,
            "payment_type": ColType.CATEGORICAL,
            "fare_amount": ColType.NUMERICAL,
            "extra": ColType.NUMERICAL,
            "mta_tax": ColType.NUMERICAL,
            "tip_amount": ColType.NUMERICAL,
            "tolls_amount": ColType.NUMERICAL,
            "improvement_surcharge": ColType.NUMERICAL,
            "total_amount": ColType.NUMERICAL,
            "congestion_surcharge": ColType.NUMERICAL,
            "airport_fee": ColType.NUMERICAL,
            # aux table cols
            "trip_id": ColType.CATEGORICAL,
            "taxi_id": ColType.CATEGORICAL,
            "trip_start_timestamp": ColType.CATEGORICAL,
            "trip_end_timestamp": ColType.CATEGORICAL,
            "trip_seconds": ColType.NUMERICAL,
            "trip_miles": ColType.NUMERICAL,
            "pickup_census_tract": ColType.CATEGORICAL,
            "pickup_community_area": ColType.CATEGORICAL,
            "fare": ColType.NUMERICAL,
            "tips": ColType.NUMERICAL,
            "tolls": ColType.NUMERICAL,
            "extras": ColType.NUMERICAL,
            "trip_total": ColType.NUMERICAL,
            "company": ColType.CATEGORICAL,
            "pickup_centroid_latitude": ColType.NUMERICAL,
            "pickup_centroid_longitude": ColType.NUMERICAL,
            "pickup_centroid_location": ColType.CATEGORICAL,
        }
        nctaxi_da_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[5])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=nctaxi_da_df,
            col_types=col_types,
            target_col="dolocationid",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[2])

        # Merged Data(FA)
        csv_path = osp.join(self.raw_dir, self.raw_filenames[3])
        col_types = {
            "vendorid": ColType.CATEGORICAL,
            "tpep_pickup_datetime": ColType.CATEGORICAL,
            "tpep_dropoff_datetime": ColType.CATEGORICAL,
            "passenger_count": ColType.NUMERICAL,
            "trip_distance": ColType.NUMERICAL,
            "ratecodeid": ColType.CATEGORICAL,
            "store_and_fwd_flag": ColType.CATEGORICAL,
            "pulocationid": ColType.CATEGORICAL,
            "dolocationid": ColType.CATEGORICAL,
            "payment_type": ColType.CATEGORICAL,
            "fare_amount": ColType.NUMERICAL,
            "extra": ColType.NUMERICAL,
            "mta_tax": ColType.NUMERICAL,
            "tip_amount": ColType.NUMERICAL,
            "tolls_amount": ColType.NUMERICAL,
            "improvement_surcharge": ColType.NUMERICAL,
            "total_amount": ColType.NUMERICAL,
            "congestion_surcharge": ColType.NUMERICAL,
            "airport_fee": ColType.NUMERICAL,
            # aux table cols
            "trip_id": ColType.CATEGORICAL,
            "taxi_id": ColType.CATEGORICAL,
            "trip_start_timestamp": ColType.CATEGORICAL,
            "trip_end_timestamp": ColType.CATEGORICAL,
            "trip_seconds": ColType.NUMERICAL,
            "trip_miles": ColType.NUMERICAL,
            "pickup_census_tract": ColType.CATEGORICAL,
            "pickup_community_area": ColType.CATEGORICAL,
            "dropoff_community_area": ColType.CATEGORICAL,
            "fare": ColType.NUMERICAL,
            "tips": ColType.NUMERICAL,
            "tolls": ColType.NUMERICAL,
            "extras": ColType.NUMERICAL,
            "trip_total": ColType.NUMERICAL,
            "aux_payment_type": ColType.CATEGORICAL,
            "company": ColType.CATEGORICAL,
            "pickup_centroid_latitude": ColType.NUMERICAL,
            "pickup_centroid_longitude": ColType.NUMERICAL,
            "pickup_centroid_location": ColType.CATEGORICAL,
        }
        nctaxi_fa_df = pd.read_csv(csv_path, low_memory=False)
        masks_path = osp.join(self.raw_dir, self.raw_filenames[4])
        masks = torch.load(masks_path, weights_only=False)
        TableData(
            df=nctaxi_fa_df,
            col_types=col_types,
            target_col="dolocationid",
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"],
        ).save(self.processed_paths[3])

    def download(self):
        extract_local_archive(
            "nctaxi", "union", self.raw_dir, self.raw_filenames
        )

    def __len__(self):
        return 4

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.data_list):
            raise IndexError
        return self.data_list[index]
