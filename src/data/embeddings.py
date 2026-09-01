"""Image embeddings for near-duplicate detection.

Why embeddings rather than hashing: byte hashing only finds byte-identical files, and the
earlier audit of this data showed that under-counts - the same eye re-encoded or resized
by a different upstream dataset slips through. A perceptual hash is not the answer either;
on fundus images (bright disc, dark surround, low entropy) an 8x8 dHash saturates and
returns tens of thousands of meaningless "matches".

So: ImageNet-pretrained ResNet50 penultimate features, L2-normalised, compared by cosine
similarity. This is a detector whose threshold must be CALIBRATED against hand-labelled
pairs before it is trusted - see scripts/find_duplicates.py. We do not assume it is right.

Embeddings are cached to disk because they are slow to compute and never change for a
given image. Nothing downstream should recompute them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

Image.MAX_IMAGE_PIXELS = None

EMBEDDING_DIM = 2048
INPUT_SIZE = 224

# Plain resize, not the FOV crop used for training. Dedup asks "is this the same picture",
# and cropping away the border would discard evidence that two files differ.
TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        try:
            with Image.open(self.paths[index]) as image:
                tensor = TRANSFORM(image.convert("RGB"))
        except Exception:  # noqa: BLE001 - an unreadable file becomes a zero vector
            tensor = torch.zeros(3, INPUT_SIZE, INPUT_SIZE)
        return tensor, index


def build_extractor() -> torch.nn.Module:
    """ResNet50 with the classifier removed - outputs the 2048-d pooled feature."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


@torch.inference_mode()
def embed_paths(paths: list[str], batch_size: int = 32, workers: int = 0,
                verbose: bool = True) -> np.ndarray:
    """Return an (N, 2048) array of L2-normalised embeddings, in the order given."""
    model = build_extractor()
    loader = DataLoader(
        ImagePathDataset(paths), batch_size=batch_size, num_workers=workers, shuffle=False
    )

    output = np.zeros((len(paths), EMBEDDING_DIM), dtype=np.float32)
    done = 0
    for batch, indices in loader:
        features = model(batch)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        output[indices.numpy()] = features.numpy()
        done += len(indices)
        if verbose and done % (batch_size * 20) < batch_size:
            print(f"    embedded {done}/{len(paths)}")
    return output


def save(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids, dtype=object), vectors=vectors)


def load(path: Path) -> tuple[list[str], np.ndarray]:
    stored = np.load(path, allow_pickle=True)
    return list(stored["ids"]), stored["vectors"]
