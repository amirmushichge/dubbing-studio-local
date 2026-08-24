from __future__ import annotations

import torch

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def model_load_options(device: str, dtype: str) -> dict:
    """Build consistent Transformers loading options for CUDA, MPS and CPU."""
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return {"dtype": DTYPES[dtype], "device_map": device}


def empty_device_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
