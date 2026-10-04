"""ImageNet-pretrained ConvNeXt-Tiny for four-class fundus screening.

torchvision's ConvNeXt is `features` (8 modules) -> avgpool -> classifier, where
classifier = LayerNorm2d(768) -> Flatten -> Linear(768, 1000). It is NOT a ResNet: there is no
`.fc`, and the final normalisation lives in the classifier, not in the backbone.

    features[0]     stem (4x4 patchify conv + LayerNorm)
    features[1]     stage 1 (96 ch)       features[2]  downsample
    features[3]     stage 2 (192 ch)      features[4]  downsample
    features[5]     stage 3 (384 ch)      features[6]  downsample
    features[7]     stage 4 (768 ch)

The split mirrors the original project's partial fine-tuning (everything up to the last stage
frozen, last stage + new head trained):

    stem  = features[0:6]   frozen, always in eval()
    tail  = features[6:8]   trained
    head  = global average pool -> LayerNorm(768) -> Linear(768, n_classes)

The head keeps ConvNeXt's own normalise-then-classify structure with a fresh Linear layer.
Checkpoints store only tail + head (the stem is unchanged ImageNet weights from torchvision).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

CUT = 6            # features[:CUT] frozen, features[CUT:] trained
POOLED_DIM = 768


class ConvNeXtScreener(nn.Module):
    def __init__(self, n_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        parts = list(convnext_tiny(weights=weights).features)
        self.stem = nn.Sequential(*parts[:CUT])
        self.tail = nn.Sequential(*parts[CUT:])
        self.norm = nn.LayerNorm(POOLED_DIM)
        self.fc = nn.Linear(POOLED_DIM, n_classes)
        for p in self.stem.parameters():
            p.requires_grad_(False)
        self.stem.eval()

    def train(self, mode: bool = True):          # the frozen stem must never leave eval()
        super().train(mode)
        self.stem.eval()
        return self

    def stem_features(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen part. (B, 3, 224, 224) -> (B, 384, 14, 14)."""
        with torch.no_grad():
            return self.stem(x)

    def last_stage(self, stem_out: torch.Tensor) -> torch.Tensor:
        """Trained stage. (B, 384, 14, 14) -> (B, 768, 7, 7). This is the Grad-CAM target."""
        return self.tail(stem_out)

    def classify(self, feature_map: torch.Tensor) -> torch.Tensor:
        pooled = torch.flatten(nn.functional.adaptive_avg_pool2d(feature_map, 1), 1)
        return self.fc(self.norm(pooled))

    def forward_from_stem(self, stem_out: torch.Tensor) -> torch.Tensor:
        return self.classify(self.last_stage(stem_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_stem(self.stem_features(x))

    def trainable_parameters(self):
        return list(self.tail.parameters()) + self.head_parameters()

    def head_parameters(self):
        return list(self.norm.parameters()) + list(self.fc.parameters())

    def trained_state(self) -> dict:
        """Everything except the frozen ImageNet stem."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith("stem.")}


def save_checkpoint(path: Path, state: dict, meta: dict) -> None:
    """Trained weights in half precision (about 30 MB) plus the metadata needed to use them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.half() for k, v in state.items()}, "meta": meta}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Returns (model in eval mode, meta). The stem comes from torchvision's ImageNet weights."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run `python scripts/train.py` first")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    meta = bundle["meta"]
    model = ConvNeXtScreener(n_classes=len(meta["labels"]), pretrained=True)
    missing, unexpected = model.load_state_dict(
        {k: v.float() for k, v in bundle["state"].items()}, strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    assert all(k.startswith("stem.") for k in missing), f"checkpoint is missing trained keys: {missing}"
    return model.to(device).eval(), meta
