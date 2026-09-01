"""Build the four split variants used by the protocol ladder (E1).

    python scripts/build_splits.py

Writes to data/splits/, one CSV per protocol rung:

    splits_image_level.csv           rung 0 - stratified by class, patient ignored (WRONG)
    splits_patient_disjoint.csv      rung 1 - grouped by patient
    splits_patient_disjoint_dedup.csv rung 2 - grouped by patient, duplicates removed
    splits_provenance_controlled.csv rung 3 - rung 2 restricted to one acquisition source

Rung 3 note: ODIR carries no reliable per-image source label (no EXIF, 91 fragmented
resolutions), so "provenance control" here means restricting to the dominant resolution
cluster - a proxy, and reported as one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import (  # noqa: E402
    assert_patient_disjoint,
    build_folds,
    build_image_level_folds,
    describe_folds,
)
from src.utils.config import load_config, output_dir  # noqa: E402


def main() -> int:
    config = load_config("data")["splits"]
    n_folds, seed = config["n_folds"], config["seed"]
    manifests, splits_dir = output_dir("manifests"), output_dir("splits")

    raw = pd.read_csv(manifests / "odir_manifest.csv", encoding="utf-8")
    clean_path = manifests / "odir_manifest_clean.csv"
    clean = pd.read_csv(clean_path, encoding="utf-8") if clean_path.is_file() else raw

    print("=" * 72)
    print(f"BUILD SPLITS  ({n_folds}-fold, seed {seed})")
    print("=" * 72)

    variants = {}

    # --- rung 0: the split most published work uses ------------------------------------
    image_level = build_image_level_folds(raw, n_folds, seed)
    spanning = int((image_level.groupby("patient_id")["fold"].nunique() > 1).sum())
    variants["image_level"] = image_level
    print(f"\nrung 0  image_level              {len(image_level)} images, "
          f"{spanning} patients span folds  <- the leakage being measured")

    # --- rung 1: patient-disjoint ------------------------------------------------------
    patient_disjoint = build_folds(raw, n_folds, seed)
    assert_patient_disjoint(patient_disjoint)
    variants["patient_disjoint"] = patient_disjoint
    print(f"rung 1  patient_disjoint         {len(patient_disjoint)} images, "
          "0 patients span folds (asserted)")

    # --- rung 2: + deduplication -------------------------------------------------------
    dedup = build_folds(clean, n_folds, seed)
    assert_patient_disjoint(dedup)
    variants["patient_disjoint_dedup"] = dedup
    print(f"rung 2  patient_disjoint_dedup   {len(dedup)} images "
          f"({len(raw) - len(clean)} duplicates removed)")

    # --- rung 3: + provenance control ---------------------------------------------------
    sizes = clean.assign(res=clean["width"].astype(str) + "x" + clean["height"].astype(str))
    dominant = sizes["res"].value_counts().idxmax()
    controlled = sizes[sizes["res"] == dominant].drop(columns="res")
    provenance = build_folds(controlled, n_folds, seed)
    assert_patient_disjoint(provenance)
    variants["provenance_controlled"] = provenance
    print(f"rung 3  provenance_controlled    {len(provenance)} images "
          f"(restricted to {dominant}, the dominant resolution cluster)")

    summary = {}
    for name, frame in variants.items():
        path = splits_dir / f"splits_{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        description = describe_folds(frame)
        summary[name] = {
            "images": int(len(frame)),
            "patients": int(frame["patient_id"].nunique()),
            "patients_spanning_folds": int(
                (frame.groupby("patient_id")["fold"].nunique() > 1).sum()),
            "per_fold": description.to_dict(orient="index"),
        }
        print(f"\n{name}")
        print(description.to_string())

    (splits_dir / "splits_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"WRITTEN to {splits_dir}")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
