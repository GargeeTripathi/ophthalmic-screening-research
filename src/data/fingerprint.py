"""FOV-normalised image fingerprints for near-duplicate detection.

WHY NOT EMBEDDINGS
------------------
The first attempt used ImageNet ResNet50 features and cosine similarity. Measured against
ground truth (merged-benchmark images that share a filename with an ODIR image, i.e. the
same eye at a different scale), it does not work in this domain:

    known duplicates      median cosine 0.929, 1st percentile 0.806
    random unrelated ODIR pairs   99th percentile 0.900, maximum 0.955

The distributions overlap, so no threshold separates them. Two reasons: fundus images are
visually homogeneous (a bright disc on a dark surround), and the copies differ by a large
downscale plus re-encode, which ImageNet features are not invariant to.

WHAT WORKS
----------
Undo the variation instead of hoping the model is invariant to it. Crop to the retinal
field of view, convert to greyscale, resize to a fixed small grid, and mean-centre and
L2-normalise so the comparison is a correlation rather than a brightness match.

Measured on the same ground truth (250 known duplicates vs 250 controls):

    ROC-AUC 0.9994

That is a detector worth trusting, and it is explainable in one sentence. It needs no
model, no GPU and no torch.

CHOOSING THE THRESHOLD - AND A MISTAKE WORTH RECORDING
------------------------------------------------------
The first attempt calibrated on a BALANCED set (1,654 known duplicates vs 1,654 random
non-duplicates) and picked 0.98, where precision measured 0.999. Applied to the real
search - all 12.79 million within-ODIR pairs - that threshold returned 26,634 pairs which
chained transitively into clusters swallowing 37% of the dataset.

The error was the base rate. Precision on a balanced sample does not transfer to a search
where true positives are one in a million; a 0.1% false-positive rate there means ~15,000
false pairs. Calibration negatives must reflect the search space, so they are now the
HARDEST negatives available: for each image, its best-scoring non-matching counterpart.

True duplicates turn out to be near-exact: median 0.9999, 5th percentile 0.9994. The
operating point is therefore 0.999, which retains ~95% of true duplicates and returns 2
within-ODIR pairs rather than 26,634.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

GRID = 64                    # fingerprint side length -> GRID*GRID dimensions
FOV_THRESHOLD = 0.10         # fraction of max intensity that counts as "inside the retina"
DUPLICATE_THRESHOLD = 0.999  # calibrated against hard negatives - see the module docstring
CANDIDATE_THRESHOLD = 0.90   # generous, so calibration can see the decision region


def fingerprint(path: str | Path, grid: int = GRID) -> np.ndarray:
    """Return a (grid*grid,) unit vector, or a zero vector if the file is unreadable.

    A zero vector scores 0 against everything, so a bad file never matches anything
    rather than crashing the run or silently matching everything.
    """
    try:
        with Image.open(path) as image:
            array = np.array(image.convert("L"), dtype=np.float32, copy=True)
    except Exception:  # noqa: BLE001
        return np.zeros(grid * grid, dtype=np.float32)

    # Crop to the retinal disc. Fundus photographs sit on a black background whose size
    # and aspect ratio differ between upstream datasets; cropping removes that difference,
    # which is exactly the variation that defeated the embedding approach.
    peak = float(array.max())
    if peak > 0:
        inside = array > peak * FOV_THRESHOLD
        rows, cols = np.where(inside)
        if len(rows) > 10:
            array = array[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    array = np.array(
        Image.fromarray(array).resize((grid, grid), Image.LANCZOS),
        dtype=np.float32, copy=True,
    )

    # Mean-centre then L2-normalise: a dot product between two fingerprints is then the
    # Pearson correlation, so exposure and contrast differences do not count as differences.
    array -= array.mean()
    norm = float(np.linalg.norm(array))
    return (array.ravel() / norm) if norm else array.ravel()


def fingerprint_paths(paths: list[str], verbose: bool = True) -> np.ndarray:
    """Fingerprint every path, in order. Returns (N, GRID*GRID)."""
    output = np.zeros((len(paths), GRID * GRID), dtype=np.float32)
    for index, path in enumerate(paths):
        output[index] = fingerprint(path)
        if verbose and (index + 1) % 500 == 0:
            print(f"    fingerprinted {index + 1}/{len(paths)}")
    return output


def save(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids, dtype=object), vectors=vectors)


def load(path: Path) -> tuple[list[str], np.ndarray]:
    stored = np.load(path, allow_pickle=True)
    return list(stored["ids"]), stored["vectors"]
