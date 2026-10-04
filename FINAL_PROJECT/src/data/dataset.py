"""Image and frozen-stem caches, and the minibatch iterator used for training.

WHY A STEM CACHE
The first six ConvNeXt modules are frozen and held in eval(), and no augmentation is used, so
their output for a given image never changes. Computing it once (about 1.5 GB on disk) and
training the last stage from the cache gives the same result as running the full network every
epoch, and makes CPU training practical. The original project cached ResNet50 layer3 outputs for
the same reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import INPUT_SIZE, load_resized, to_tensor
from src.utils.config import load_config, repo_path


def cache_dir() -> Path:
    path = repo_path(load_config()["data"]["cache_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ids_match(path: Path, ids: list[str]) -> bool:
    return path.is_file() and json.loads(path.read_text()) == ids


def image_cache(frame: pd.DataFrame, tag: str = "all"):
    """uint8 (N, 224, 224, 3) memmap of every resized image, built once."""
    ids = sorted(frame.image_id.unique())
    data_path = cache_dir() / f"images_{tag}_{INPUT_SIZE}.npy"
    ids_path = cache_dir() / f"images_{tag}_{INPUT_SIZE}_ids.json"
    if not (data_path.is_file() and _ids_match(ids_path, ids)):
        paths = frame.drop_duplicates("image_id").set_index("image_id").image_path
        store = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.uint8,
                                          shape=(len(ids), INPUT_SIZE, INPUT_SIZE, 3))
        for n, image_id in enumerate(ids):
            store[n] = load_resized(paths[image_id])
            if n % 500 == 0:
                print(f"  resized {n}/{len(ids)} images", flush=True)
        store.flush()
        del store
        ids_path.write_text(json.dumps(ids))
    return np.load(data_path, mmap_mode="r"), {i: n for n, i in enumerate(ids)}


def stem_cache(model, frame: pd.DataFrame, tag: str = "all", batch_size: int = 32):
    """float32 (N, 384, 14, 14) memmap of the frozen stem's output for every image."""
    images, index = image_cache(frame, tag)
    ids = sorted(index, key=index.get)
    data_path = cache_dir() / f"stem_{tag}_{INPUT_SIZE}.npy"
    ids_path = cache_dir() / f"stem_{tag}_{INPUT_SIZE}_ids.json"
    if not (data_path.is_file() and _ids_match(ids_path, ids)):
        model.eval()
        probe = model.stem_features(to_tensor(images[:1]))
        store = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float32,
                                          shape=(len(ids), *probe.shape[1:]))
        for start in range(0, len(ids), batch_size):
            store[start:start + batch_size] = model.stem_features(
                to_tensor(images[start:start + batch_size])).numpy()
            if (start // batch_size) % 20 == 0:
                print(f"  frozen-stem features {start}/{len(ids)}", flush=True)
        store.flush()
        del store
        ids_path.write_text(json.dumps(ids))
    return np.load(data_path, mmap_mode="r"), index


class Batches:
    """Minibatches of cached stem features and integer labels."""

    def __init__(self, frame, features, index, labels, batch_size, train=False, drop_last=False):
        self.frame = frame.reset_index(drop=True)
        self.features, self.index, self.labels = features, index, labels
        self.batch_size, self.train, self.drop_last = batch_size, train, train and drop_last

    def __iter__(self):
        order = np.random.permutation(len(self.frame)) if self.train else np.arange(len(self.frame))
        for start in range(0, len(order), self.batch_size):
            rows = self.frame.iloc[order[start:start + self.batch_size]]
            if self.drop_last and len(rows) < self.batch_size:
                break
            # read the memmap in ascending position; order inside a batch does not matter
            rows = rows.assign(_pos=[self.index[i] for i in rows.image_id]).sort_values("_pos")
            x = torch.from_numpy(np.array(self.features[rows._pos.to_numpy()]))
            y = torch.tensor([self.labels.index(c) for c in rows.label])
            yield x, y, list(rows.image_id)
