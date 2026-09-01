"""Compute and cache FOV-normalised fingerprints for both manifests.

Replaces the ImageNet-embedding approach for duplicate detection - see the docstring of
src/data/fingerprint.py for the measurement that motivated the change.

    python scripts/build_fingerprints.py
    python scripts/build_fingerprints.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fingerprint import GRID, fingerprint_paths, save  # noqa: E402
from src.utils.config import output_dir  # noqa: E402

DATASETS = {"odir": "odir_manifest.csv", "merged": "merged_manifest.csv"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifests, processed = output_dir("manifests"), output_dir("processed")
    print("=" * 72)
    print(f"BUILD FINGERPRINTS  (FOV crop -> greyscale -> {GRID}x{GRID} -> normalised)")
    print("=" * 72)

    for name, filename in DATASETS.items():
        manifest_path = manifests / filename
        cache_path = processed / f"fingerprints_{name}.npz"
        if not manifest_path.is_file():
            print(f"\n{name}: manifest missing - skipping")
            continue
        if cache_path.is_file() and not args.force:
            print(f"\n{name}: cached - skipping (use --force to redo)")
            continue

        manifest = pd.read_csv(manifest_path, encoding="utf-8")
        print(f"\n{name}: fingerprinting {len(manifest)} images...")
        started = time.time()
        vectors = fingerprint_paths(manifest["image_path"].tolist())
        elapsed = time.time() - started
        save(cache_path, manifest["image_id"].tolist(), vectors)

        dead = int((vectors == 0).all(axis=1).sum())
        print(f"  done in {elapsed / 60:.1f} min  ({len(manifest) / elapsed:.1f} img/s)")
        print(f"  written: {cache_path}  shape {vectors.shape}")
        if dead:
            print(f"  WARNING: {dead} unreadable image(s) produced zero fingerprints")

    print("\n" + "=" * 72)
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
