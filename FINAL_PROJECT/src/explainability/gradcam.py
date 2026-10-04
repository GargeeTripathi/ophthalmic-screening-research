"""Grad-CAM for ConvNeXt-Tiny.

TARGET LAYER
The target is the output of the last stage, `model.last_stage(...)` = torchvision
`features[7]`, a (768, 7, 7) feature map - the last layer that still has spatial layout.
ResNet50's target (`layer4`) does not exist in this network, and ConvNeXt's final LayerNorm
sits AFTER global pooling, so it has no spatial map to explain.

    weights_c = mean over the 7x7 map of d(logit_class) / d(feature map)
    cam       = ReLU(sum_c weights_c * feature_map_c), upsampled to 224 x 224, scaled to [0, 1]

A heatmap shows which regions moved the class score. It is not evidence that the model found a
lesion: this project has no lesion annotations to check that against.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.data.preprocessing import INPUT_SIZE


def grad_cam(model, tensor: torch.Tensor, class_index: int | None = None):
    """Returns (cam as float array (224, 224) in [0, 1], class index explained, logits)."""
    model.eval()
    stem_out = model.stem_features(tensor)
    with torch.enable_grad():
        feature_map = model.last_stage(stem_out.requires_grad_(False))
        feature_map.retain_grad()
        logits = model.classify(feature_map)
        index = int(logits[0].argmax()) if class_index is None else class_index
        model.zero_grad(set_to_none=True)
        logits[0, index].backward()
    weights = feature_map.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feature_map.detach()).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max() if float(cam.max()) > 0 else cam
    return cam.numpy(), index, logits.detach()[0]


def overlay(image_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet heatmap over the resized RGB image. Returns uint8 (224, 224, 3)."""
    import matplotlib
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    blended = (1 - alpha * cam[..., None]) * (image_uint8 / 255.0) + alpha * cam[..., None] * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
