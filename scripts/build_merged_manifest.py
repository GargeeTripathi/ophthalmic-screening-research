"""Build the provenance manifest for the merged four-class benchmark - the audit subject.

This dataset is NEVER used to train the primary model. It exists in this project for two
reasons: it is the object of the provenance audit (RQ1), and its non-ODIR portions serve
as cross-source test data.

    python scripts/build_merged_manifest.py

Exit codes:
    0  manifest written, every image assigned a source
    2  some images could not be assigned a source - the rules need extending
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import parse_filename  # noqa: E402
from src.provenance.features import describe_image  # noqa: E402
from src.utils.config import load_config, output_dir, source_path  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build() -> pd.DataFrame:
    root = source_path("merged_root")
    class_names = load_config("data")["class_names"]      # code -> folder name
    folder_to_code = {folder: code for code, folder in class_names.items()}

    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        code = folder_to_code.get(folder.name)
        if code is None:
            print(f"  skipping unexpected folder: {folder.name}")
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            features = describe_image(path)
            patient_id, eye = parse_filename(path.name)
            rows.append({
                "image_id": f"merged_{folder.name}_{path.stem}",
                "image_path": str(path),
                "label": code,
                "class_folder": folder.name,
                # Present only for patient_eye-named files; null elsewhere. Note the ids
                # are NOT comparable across sources - two datasets can both have "1234".
                "patient_id_raw": patient_id,
                "eye": eye,
                **features,
            })
        print(f"  {folder.name:<22} {sum(1 for r in rows if r['class_folder'] == folder.name):>5} images")

    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 72)
    print("BUILD MERGED-BENCHMARK PROVENANCE MANIFEST")
    print("=" * 72)
    print("\nScanning class folders...")
    manifest = build()

    print("\n" + "-" * 72)
    print("SOURCE x CLASS  (counts)")
    print("-" * 72)
    crosstab = pd.crosstab(manifest["source"], manifest["label"])
    print(crosstab.to_string())

    print("\n" + "-" * 72)
    print("SOURCE x CLASS  (row %, i.e. what class does each source imply?)")
    print("-" * 72)
    row_pct = (crosstab.div(crosstab.sum(axis=1), axis=0) * 100).round(1)
    print(row_pct.to_string())

    # How predictable is the label from the source alone? Always guessing the source's
    # majority class gives a lower bound on what a provenance-only model can achieve.
    majority_correct = int(crosstab.max(axis=1).sum())
    base_rate = int(manifest["label"].value_counts().max())
    print("\n" + "-" * 72)
    print("PROVENANCE PREDICTIVENESS (lower bound, no model)")
    print("-" * 72)
    print(f"  images                                   {len(manifest)}")
    print(f"  correct by always-guess-majority-class    {base_rate}"
          f"  ({base_rate / len(manifest) * 100:.1f}%)   <- baseline")
    print(f"  correct by source -> majority class       {majority_correct}"
          f"  ({majority_correct / len(manifest) * 100:.1f}%)   <- provenance alone")

    print("\n" + "-" * 72)
    print("EXTENSION x CLASS")
    print("-" * 72)
    print(pd.crosstab(manifest["file_extension"], manifest["label"]).to_string())

    unassigned = manifest[manifest["source"] == "unassigned"]
    unreadable = manifest[manifest["width"].isna()]

    directory = output_dir("manifests")
    manifest_path = directory / "merged_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")

    summary = {
        "total_images": int(len(manifest)),
        "sources": manifest["source"].value_counts().to_dict(),
        "source_by_class": crosstab.to_dict(),
        "majority_class_baseline": base_rate / len(manifest),
        "source_majority_accuracy": majority_correct / len(manifest),
        "unassigned": int(len(unassigned)),
        "unreadable": int(len(unreadable)),
    }
    summary_path = directory / "merged_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"WRITTEN\n  {manifest_path}\n  {summary_path}")
    if len(unassigned):
        print(f"\nRESULT: {len(unassigned)} image(s) unassigned - extend the rules in "
              "src/provenance/features.py")
        print(unassigned[["original_filename", "file_extension", "width", "height"]].head(10).to_string())
        return 2
    print("\nRESULT: OK - every image assigned a source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
