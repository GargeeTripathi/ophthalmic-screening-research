"""Compute and cache image embeddings for both manifests.

Slow (CPU, a few minutes for ~9k images) but run once. Everything downstream reads the
cache. Re-run only if the manifests change.

    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --force     # recompute even if cached
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embeddings import embed_paths, save  # noqa: E402
from src.utils.config import output_dir  # noqa: E402

DATASETS = {
    "odir": "odir_manifest.csv",
    "merged": "merged_manifest.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute existing caches")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifests = output_dir("manifests")
    processed = output_dir("processed")

    print("=" * 72)
    print("BUILD EMBEDDINGS  (ResNet50 ImageNet, 2048-d, L2-normalised)")
    print("=" * 72)

    for name, filename in DATASETS.items():
        manifest_path = manifests / filename
        cache_path = processed / f"embeddings_{name}.npz"

        if not manifest_path.is_file():
            print(f"\n{name}: manifest missing ({manifest_path}) - skipping")
            continue
        if cache_path.is_file() and not args.force:
            print(f"\n{name}: cached at {cache_path} - skipping (use --force to redo)")
            continue

        manifest = pd.read_csv(manifest_path, encoding="utf-8")
        print(f"\n{name}: embedding {len(manifest)} images...")
        started = time.time()
        vectors = embed_paths(manifest["image_path"].tolist(), batch_size=args.batch_size)
        elapsed = time.time() - started

        save(cache_path, manifest["image_id"].tolist(), vectors)
        print(f"  done in {elapsed / 60:.1f} min  ({len(manifest) / elapsed:.1f} img/s)")
        print(f"  written: {cache_path}  shape {vectors.shape}")

    print("\n" + "=" * 72)
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
