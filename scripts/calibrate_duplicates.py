"""Calibrate the duplicate detector against hand labels, then emit the final groups.

Run this after filling in `is_duplicate` in data/manifests/duplicate_review_sample.csv.

The point of this script is honesty about the detector: it reports precision and recall
at a range of thresholds, picks the operating point, and records both the choice and its
measured error rates so the report can state them rather than assert the detector works.

    python scripts/calibrate_duplicates.py
    python scripts/calibrate_duplicates.py --threshold 0.97   # override the auto choice

Outputs (data/manifests/):
    duplicate_threshold_sweep.csv   precision/recall/F1 at each candidate threshold
    duplicate_groups.csv            confirmed duplicate groups at the chosen threshold
    deduplication_report.json       updated with the calibration result
    odir_manifest_clean.csv         manifest with within-ODIR duplicates collapsed

Exit codes:
    0  calibrated and written
    2  the review sample is not labelled yet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dedup import detector_scores, group_duplicates  # noqa: E402
from src.utils.config import output_dir  # noqa: E402

SWEEP = np.arange(0.90, 1.001, 0.005)
# Duplicates that survive into a split inflate the test score, so a false positive
# (dropping a distinct image) is far cheaper than a false negative (keeping a duplicate).
# The operating point therefore targets precision only high enough to avoid gutting the
# dataset, and otherwise favours recall.
MIN_PRECISION = 0.90


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the automatically selected operating point")
    args = parser.parse_args()

    manifests = output_dir("manifests")
    review_path = manifests / "duplicate_review_sample.csv"
    if not review_path.is_file():
        print(f"ERROR: {review_path} not found. Run scripts/find_duplicates.py first.")
        return 2

    review = pd.read_csv(review_path, encoding="utf-8")
    labelled = review[review["is_duplicate"].astype(str).str.strip().isin({"0", "1"})].copy()
    labelled["is_duplicate"] = labelled["is_duplicate"].astype(int)

    print("=" * 72)
    print("CALIBRATE DUPLICATE DETECTOR")
    print("=" * 72)
    print(f"\nreviewed pairs: {len(labelled)} / {len(review)}")
    if len(labelled) < 20:
        print("\nERROR: fewer than 20 labelled pairs. Fill in `is_duplicate` "
              "(1 = same eye, 0 = different eyes) and re-run.")
        return 2
    print(f"  labelled duplicates    : {int(labelled['is_duplicate'].sum())}")
    print(f"  labelled non-duplicates: {int((1 - labelled['is_duplicate']).sum())}")

    print("\n" + "-" * 72)
    print("THRESHOLD SWEEP")
    print("-" * 72)
    rows = [detector_scores(labelled, float(t)) for t in SWEEP]
    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False))
    sweep.to_csv(manifests / "duplicate_threshold_sweep.csv", index=False)

    if args.threshold is not None:
        chosen = args.threshold
        reason = "manual override"
    else:
        eligible = sweep[sweep["precision"] >= MIN_PRECISION]
        if eligible.empty:
            chosen = float(sweep.loc[sweep["f1"].idxmax(), "threshold"])
            reason = f"no threshold reached precision {MIN_PRECISION}; best F1 used"
        else:
            # Lowest threshold meeting the precision floor = highest recall at acceptable cost.
            chosen = float(eligible.loc[eligible["threshold"].idxmin(), "threshold"])
            reason = f"lowest threshold with precision >= {MIN_PRECISION}"

    at_chosen = detector_scores(labelled, chosen)
    print("\n" + "-" * 72)
    print("OPERATING POINT")
    print("-" * 72)
    print(f"  threshold {chosen:.3f}   ({reason})")
    for key in ("precision", "recall_within_pool", "f1", "false_positive", "false_negative"):
        print(f"  {key:<20} {at_chosen[key]}")
    print("\n  NOTE: recall is measured within the reviewed candidate pool only. Pairs "
          "below\n  the candidate threshold were never shown, so true recall over the "
          "whole dataset\n  is unknown. State this in the report.")

    # --- apply the threshold ----------------------------------------------------------
    odir = pd.read_csv(manifests / "odir_manifest.csv", encoding="utf-8")
    within = pd.read_csv(manifests / "duplicate_candidates_within_odir.csv", encoding="utf-8")
    across = pd.read_csv(manifests / "duplicate_candidates_odir_vs_merged.csv", encoding="utf-8")

    confirmed_within = within[within["similarity"] >= chosen]
    confirmed_across = across[across["similarity"] >= chosen]

    groups = group_duplicates(confirmed_within, n_items=len(odir))
    group_rows = [
        {"group_id": gid, "image_id": odir.loc[index, "image_id"],
         "label": odir.loc[index, "label"], "patient_id": odir.loc[index, "patient_id"],
         "keep": position == 0}
        for gid, members in enumerate(groups)
        for position, index in enumerate(sorted(members))
    ]
    group_frame = pd.DataFrame(group_rows)
    group_frame.to_csv(manifests / "duplicate_groups.csv", index=False, encoding="utf-8")

    dropped = set(group_frame.loc[~group_frame["keep"], "image_id"]) if len(group_frame) else set()
    clean = odir[~odir["image_id"].isin(dropped)].copy()
    clean.to_csv(manifests / "odir_manifest_clean.csv", index=False, encoding="utf-8")

    print("\n" + "-" * 72)
    print("APPLIED")
    print("-" * 72)
    print(f"  confirmed within-ODIR pairs      {len(confirmed_within)}")
    print(f"  duplicate groups                 {len(groups)}")
    print(f"  images dropped as redundant      {len(dropped)}")
    print(f"  clean manifest                   {len(clean)} images "
          f"({clean['patient_id'].nunique()} patients)")
    print(f"  merged images matching ODIR      {confirmed_across['b_id'].nunique() if len(confirmed_across) else 0}"
          "   <- cannot serve as cross-source test data")

    report_path = manifests / "deduplication_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report.update({
        "status": "CALIBRATED",
        "reviewed_pairs": int(len(labelled)),
        "chosen_threshold": chosen,
        "threshold_reason": reason,
        "detector_at_threshold": at_chosen,
        "recall_caveat": "measured within the reviewed candidate pool only",
        "duplicate_groups": len(groups),
        "images_dropped": len(dropped),
        "clean_images": int(len(clean)),
        "clean_patients": int(clean["patient_id"].nunique()),
        "merged_images_matching_odir": int(confirmed_across["b_id"].nunique()) if len(confirmed_across) else 0,
    })
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"WRITTEN\n  {manifests / 'duplicate_groups.csv'}"
          f"\n  {manifests / 'odir_manifest_clean.csv'}\n  {report_path}")
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
