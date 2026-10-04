"""The one preprocessing path, used by training, evaluation, inference, Grad-CAM and the app.

    open -> RGB -> bilinear resize to 224 x 224 (no crop) -> [0, 1] -> ImageNet mean/std

This is the original project's preprocessing, unchanged. ConvNeXt-Tiny's torchvision weights
were trained with the same ImageNet mean/std, so the normalisation is the correct one for the
pretrained weights. torchvision's reference transform for those weights resizes to 236 and
centre-crops 224; a centre crop would cut off the periphery of a fundus photograph, so the plain
resize of the original project is kept. No augmentation is applied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.config import load_config

Image.MAX_IMAGE_PIXELS = None
_P = load_config()["preprocessing"]
INPUT_SIZE: int = _P["input_size"]
MEAN = torch.tensor(_P["mean"]).view(1, 3, 1, 1)
STD = torch.tensor(_P["std"]).view(1, 3, 1, 1)


def load_resized(path: str | Path, size: int = INPUT_SIZE) -> np.ndarray:
    """uint8 (size, size, 3) RGB array."""
    with Image.open(path) as raw:
        return np.asarray(raw.convert("RGB").resize((size, size), Image.BILINEAR))


def to_tensor(batch_uint8: np.ndarray) -> torch.Tensor:
    """uint8 (B, H, W, 3) or (H, W, 3) -> normalised float (B, 3, H, W)."""
    array = np.array(batch_uint8)          # copy: cache memmaps are read-only
    if array.ndim == 3:
        array = array[None]
    x = torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(255)
    return (x - MEAN) / STD


def preprocess(path: str | Path) -> torch.Tensor:
    """One image file -> (1, 3, 224, 224) model input."""
    return to_tensor(load_resized(path))
