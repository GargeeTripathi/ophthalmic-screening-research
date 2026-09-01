"""Package everything the Colab run needs into one small zip.

WHY THIS EXISTS
---------------
The obvious instruction — "upload the ODIR Training Images folder to Drive" — means moving
about 7 GB of native-resolution photographs, most of which the run throws away: training
resizes everything to 384x384 anyway. Resizing first turns that into roughly 200 MB.

The bundle holds resized images, the split manifests, and nothing else. Paths inside it are
relative, so the absolute D:\\PBL locations in the manifests never travel.

    python scripts/build_colab_bundle.py
    python scripts/build_colab_bundle.py --size 384 --quality 90
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import output_dir  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

# The ladder needs both: the split most published work uses, and the corrected one.
SPLITS = ["splits_image_level.csv", "splits_patient_disjoint.csv",
          "splits_patient_disjoint_dedup.csv"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--out", default="colab_bundle.zip")
    args = parser.parse_args()

    splits_dir, manifests = output_dir("splits"), output_dir("manifests")
    # The RAW manifest, not the deduplicated one. The image_level rung is defined by
    # containing the duplicates - that is the leakage it exists to measure - so bundling only
    # the clean set would leave two of its rows pointing at images that are not there, and
    # trimming those rows instead would silently apply deduplication to the one rung whose
    # whole purpose is to lack it.
    manifest = pd.read_csv(manifests / "odir_manifest.csv", encoding="utf-8")
    out_path = Path(args.out)

    print("=" * 72)
    print("BUILD COLAB BUNDLE")
    print("=" * 72)
    print(f"\n{len(manifest)} images -> {args.size}x{args.size} JPEG q{args.quality}")

    started = time.time()
    skipped = []
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for position, row in enumerate(manifest.itertuples(), start=1):
            try:
                with Image.open(row.image_path) as image:
                    resized = image.convert("RGB").resize(
                        (args.size, args.size), Image.BILINEAR)
                buffer = BytesIO()
                resized.save(buffer, format="JPEG", quality=args.quality)
                bundle.writestr(f"images/{row.original_filename}", buffer.getvalue())
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the bundle
                skipped.append((row.image_id, str(exc)))
            if position % 500 == 0:
                print(f"    {position}/{len(manifest)}  "
                      f"({out_path.stat().st_size / 1e6:.0f} MB so far)")

        # Splits carry absolute local paths; strip them so the bundle is portable.
        for name in SPLITS:
            path = splits_dir / name
            if not path.is_file():
                print(f"  missing split: {name}")
                continue
            frame = pd.read_csv(path, encoding="utf-8")
            frame = frame[["image_id", "patient_id", "eye", "label",
                           "original_filename", "fold"]]
            bundle.writestr(f"splits/{name}", frame.to_csv(index=False))

        bundle.writestr("bundle.json", json.dumps({
            "images": int(len(manifest)) - len(skipped),
            "image_size": args.size,
            "jpeg_quality": args.quality,
            "splits": SPLITS,
            "note": "resized copies of ODIR-5K for training only; not redistributed with "
                    "the public repository",
        }, indent=2))

    # --- verify before anyone uploads it -------------------------------------------------
    # A split row pointing at an image the bundle does not contain fails in Colab, mid-fold,
    # after the upload. Cheaper to catch here.
    problems = []
    with zipfile.ZipFile(out_path) as check:
        bundled = {n.split("/", 1)[1] for n in check.namelist() if n.startswith("images/")}
        for name in SPLITS:
            if f"splits/{name}" not in check.namelist():
                continue
            frame = pd.read_csv(BytesIO(check.read(f"splits/{name}")))
            missing = set(frame["original_filename"]) - bundled
            if missing:
                problems.append(f"{name}: {len(missing)} row(s) reference missing images, "
                                f"e.g. {sorted(missing)[:3]}")
            if "image_level" not in name:
                spanning = (frame.groupby("patient_id")["fold"].nunique() > 1).sum()
                if spanning:
                    problems.append(f"{name}: {spanning} patient(s) span folds in a split "
                                    "that must be patient-disjoint")

    elapsed = time.time() - started
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nwritten {out_path}  {size_mb:.0f} MB  in {elapsed / 60:.1f} min")
    if skipped:
        print(f"  skipped {len(skipped)} unreadable image(s): {skipped[:3]}")
    print(f"\n  original ODIR folder is ~7 GB; this bundle is {size_mb:.0f} MB "
          f"({7000 / size_mb:.0f}x smaller)")
    if problems:
        print("\nBUNDLE IS NOT USABLE:")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    print("\nverified: every split row resolves to a bundled image, and every "
          "patient-disjoint split is disjoint")
    print("Upload this single file to Drive, then run colab/train_models.ipynb.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
