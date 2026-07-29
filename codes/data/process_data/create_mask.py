import pandas as pd
import numpy as np
import torch
import random

def split_masks_temporal_ratio(csv_file,
                               label_col='COLLISIONTYPE',
                               time_col=None,
                               train_ratio=0.7,
                               val_ratio=0.1,
                               test_ratio=0.2,
                               date_format='%m/%d/%Y %I:%M:%S %p',
                               seed=42,
                               output_path='mask_ratio_70_10_20.pt'):
    """
    Groups by class, then either:
      • If `time_col` is provided and exists, sorts by time (oldest→newest);
      • Otherwise shuffles randomly.
    Then splits each group into train/val/test by the given ratios,
    and saves the masks as a .pt file.

    Args:
        csv_file (str): Path to the CSV file.
        label_col (str): Name of the column to group by.
        time_col (str or None): Name of the time column, or None to skip temporal splitting.
        train_ratio (float): Ratio for the training set.
        val_ratio (float): Ratio for the validation set.
        test_ratio (float): Ratio for the test set.
        date_format (str): Format string for pd.to_datetime (only used if time_col is set).
        seed (int): Random seed for reproducibility.
        output_path (str): Path to save the output mask file.

    Returns:
        dict: Contains three torch.BoolTensors: 'train_mask', 'val_mask', 'test_mask'.
    """
    # 1. Read CSV
    df = pd.read_csv(csv_file) #, encoding='gbk' NewYork

    # 2. If a valid time_col is given, parse it
    temporal = False
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], format=date_format)
        temporal = True

    n_total = len(df)
    print(f"Total samples: {n_total:,}")

    # 3. Initialize empty masks
    train_mask = np.zeros(n_total, dtype=bool)
    val_mask   = np.zeros(n_total, dtype=bool)
    test_mask  = np.zeros(n_total, dtype=bool)

    rng = random.Random(seed)

    # 4. Process each class
    for cls, group in df.groupby(label_col):
        idx_list = group.index.tolist()

        if temporal:
            # Sort by timestamp ascending
            idx_list = group.sort_values(time_col).index.tolist()
        else:
            # Shuffle randomly
            rng.shuffle(idx_list)

        m = len(idx_list)
        n_train = int(m * train_ratio)
        n_val   = int(m * val_ratio)
        # remaining goes to test
        n_test  = m - n_train - n_val

        # Split indices
        train_idx = idx_list[:n_train]
        val_idx   = idx_list[n_train:n_train + n_val]
        test_idx  = idx_list[n_train + n_val:]

        # Mark masks
        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        test_mask[test_idx]   = True

        # Print per-class summary
        mode = "temporal" if temporal else "random"
        print(f"Class {cls!r} ({mode}): total={m}, train={n_train}, "
              f"val={n_val}, test={n_test}")

    # 5. Global summary
    print(f"\nAfter split: train={train_mask.sum():,}  "
          f"val={val_mask.sum():,}  test={test_mask.sum():,}")

    # 6. Save masks
    mask = {
        'train_mask': torch.from_numpy(train_mask),
        'val_mask':   torch.from_numpy(val_mask),
        'test_mask':  torch.from_numpy(test_mask),
    }
    torch.save(mask, output_path)
    print(f"Masks saved to {output_path}")

    return mask


if __name__ == "__main__":
    # Example without temporal splitting:
    mask = split_masks_temporal_ratio(
        csv_file='./NYCsale/t2.csv',
        label_col='SALE PRICE',
        time_col=None,  # -> will perform a random split
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=42,
        output_path='./NYCsale/T2_mask.pt'
    )

    # # Example with temporal splitting:
    # mask = split_masks_temporal_ratio(
    #     csv_file='./CNYBuilding/NewYork.csv',
    #     label_col='StatuteCodes',
    #     time_col='ApprovedDate',  # -> will sort by this column
    #     train_ratio=0.7,
    #     val_ratio=0.1,
    #     test_ratio=0.2,
    #     date_format='%m/%d/%Y', #'%m/%d/%Y %I:%M:%S %p'
    #     seed=42,
    #     output_path='./CNYBuilding/CNYBuilding_T2_mask.pt'
    # )
