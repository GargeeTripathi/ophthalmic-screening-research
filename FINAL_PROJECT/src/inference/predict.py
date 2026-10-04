"""Inference with the final ConvNeXt-Tiny checkpoint.

    image -> preprocessing -> ConvNeXt-Tiny -> temperature-scaled probabilities -> class + confidence

The checkpoint path comes from configs/config.yaml (model.checkpoint). Nothing here names a
model or a file directly.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.data.preprocessing import preprocess
from src.models.convnext_tiny import load_checkpoint
from src.utils.config import load_config, repo_path


class Predictor:
    """Loads once, predicts many. CPU is enough: about 0.1 s per image."""

    def __init__(self, checkpoint: str | Path | None = None):
        config = load_config()
        self.checkpoint = Path(checkpoint) if checkpoint else repo_path(config["model"]["checkpoint"])
        self.model, self.meta = load_checkpoint(self.checkpoint)
        self.labels: list[str] = self.meta["labels"]
        self.class_names: dict = self.meta["class_names"]
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.architecture = config["model"]["display_name"]

    @torch.no_grad()
    def logits(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)

    def predict(self, image_path: str | Path) -> dict:
        """Class, confidence and full distribution, from the image alone."""
        logits = self.logits(preprocess(image_path))[0]
        probabilities = torch.softmax(logits / self.temperature, dim=0).numpy()
        raw = torch.softmax(logits, dim=0).numpy()
        best = int(probabilities.argmax())
        return {
            "image": str(image_path),
            "predicted_class": self.labels[best],
            "predicted_name": self.class_names[self.labels[best]],
            "confidence": round(float(probabilities[best]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.labels, probabilities)},
            "uncalibrated_probabilities": {c: round(float(p), 6) for c, p in zip(self.labels, raw)},
            "model": self.architecture,
            "checkpoint": self.checkpoint.name,
            "temperature": self.temperature,
        }
