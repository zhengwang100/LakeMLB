"""
Utility functions for training and evaluation.
"""
import os
import random
import numpy as np
import torch
from typing import List, Optional, Union


# ======================== Seed & Device ========================
def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_id: Optional[Union[int, str]] = None) -> torch.device:
    """Get torch device.
    
    Args:
        device_id: Can be:
            - None: auto-select cuda if available, else cpu
            - int: device index (e.g., 0, 1) -> "cuda:0", "cuda:1"
            - str: device string (e.g., "cuda:1", "cpu", "cuda") -> used directly
    """
    if device_id is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle string inputs (e.g., "cuda:1", "cpu")
    if isinstance(device_id, str):
        return torch.device(device_id)
    
    # Handle integer inputs (backward compatibility)
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


# ======================== Data Utilities ========================
def get_batch(data, indices):
    """Extract a batch from data given indices."""
    from types import SimpleNamespace
    batch = SimpleNamespace()
    
    if hasattr(data, 'feat_dict'):
        batch.feat_dict = {}
        for key, val in data.feat_dict.items():
            if isinstance(val, torch.Tensor):
                batch.feat_dict[key] = val[indices]
            else:
                batch.feat_dict[key] = val
    
    if hasattr(data, 'y'):
        batch.y = data.y[indices]
    
    if hasattr(data, 'metadata'):
        batch.metadata = data.metadata
    
    return batch


def to_device(batch, device):
    """Move batch to device."""
    from types import SimpleNamespace
    
    if hasattr(batch, 'feat_dict'):
        for key in batch.feat_dict:
            if isinstance(batch.feat_dict[key], torch.Tensor):
                batch.feat_dict[key] = batch.feat_dict[key].to(device)
    
    if hasattr(batch, 'y'):
        batch.y = batch.y.to(device)
    
    return batch


# ======================== Model I/O ========================
def save_model(model, path: str):
    """Save model state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path: str, device):
    """Load model state dict."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


# ======================== Argument Parsing ========================
def parse_list_of_ints(s: str) -> List[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",")]


def parse_list_of_floats(s: str) -> List[float]:
    """Parse comma-separated floats."""
    return [float(x.strip()) for x in s.split(",")]


# ======================== Grid Search Utilities ========================
def print_grid_config(space: dict, total: int):
    """Print grid search configuration."""
    print("\n" + "="*60)
    print("Grid Search Configuration:")
    for key, vals in space.items():
        print(f"  {key}: {vals}")
    print(f"Total combinations: {total}")
    print("="*60 + "\n")
