"""
Wrapper for pyglib interface.
"""
import warnings

try:
    import pyg_lib  # noqa: F401
    WITH_PYG_LIB = True

    # Check torch ops registered
    import torch
    assert hasattr(torch.ops.pyg, 'hetero_neighbor_sample'), \
        "pyg_lib is installed, but torch ops are not registered."
except (ImportError, OSError, AssertionError) as exc:
    WITH_PYG_LIB = False
    warnings.warn(f"pyg_lib is unavailable or incompatible: {exc}")
