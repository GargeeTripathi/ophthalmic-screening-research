"""Detect near-duplicates using FOV-normalised fingerprints, calibrated against ground truth.

No manual labelling is needed. Ground truth comes for free: merged-benchmark images that
share a filename with an ODIR image are the same eye at a different scale. That gives ~1650
positives and a large negative pool, which is a better calibration set than a hand-labelled
sample of 100 - bigger, objective, and reproducible.

The script also re-measures the ImageNet-embedding detector on the same ground truth, when
the embedding cache exists, so the report can state why that approach was rejected.

    python scripts/find_duplicates.py

Outputs (data/manifests/):
    duplicate_detector_calibration.csv   precision/recall/F1 by threshold, both methods
    duplicate_candidates_within_odir.csv
    duplicate_candidates_odir_vs_merged.csv
    duplicate_groups.csv
    odir_manifest_clean.csv
    deduplication_report.json
    duplicate_review/                    a small visual spot-check of the ground-truth assumption
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dedup import find_candidate_pairs, group_duplicates  # noqa: E402
from src.data.fingerprint import CANDIDATE_THRESHOLD, DUPLICATE_THRESHOLD, load  # noqa: E402
from src.utils.config import output_dir  # noqa: E402

SWEEP = np.round(np.r_[np.arange(0.90, 0.99, 0.01), [0.995, 0.998, 0.999, 0.9995]], 4)
SPOT_CHECK = 12
THUMB = 300


def ground_truth(odir: pd.DataFrame, merged: pd.DataFrame, seed: int = 42):
    """Positive and negative ODIR<->merged index pairs.

    ASSUMPTION: a shared original filename means the same eye. ODIR's `<id>_<eye>.jpg`
    convention is dataset-specific and the merged benchmark is partly built from ODIR, so
    this is strong - but it is an inference, and the spot-check images let a human confirm
    it. Recorded in the report as an assumption, not a fact.
    """
    rng = np.random.default_rng(seed)
    by_name = {name: index for index, name in enumerate(odir["original_filename"])}

    positives = [(by_name[name], k) for k, name in enumerate(merged["original_filename"])
                 if name in by_name]
    negative_pool = [k for k, name in enumerate(merged["original_filename"])
                     if name not in by_name]
    negatives = [(int(rng.integers(0, len(odir))), int(k))
                 for k in rng.choice(negative_pool, min(len(positives), len(negative_pool)),
                                     replace=False)]
    return positives, negatives


def hard_negatives(odir_vectors, merged_vectors, negative_pool, block: int = 512):
    """For each non-matching merged image, its BEST-scoring ODIR counterpart.

    Random negatives are easy and make precision look better than it is. The real search
    ranges over millions of pairs, so the operating point has to survive the hardest
    negative each image can find - that is what these are.
    """
    scores = []
    pool = np.asarray(negative_pool)
    for start in range(0, len(pool), block):
        chunk = pool[start:start + block]
        scores.append((merged_vectors[chunk] @ odir_vectors.T).max(axis=1))
    return np.concatenate(scores)


def score_pairs(pairs, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not pairs:
        return np.array([])
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return (left[a] * right[b]).sum(axis=1)


def sweep_scores(positive: np.ndarray, negative: np.ndarray, method: str) -> pd.DataFrame:
    rows = []
    for threshold in SWEEP:
        tp = int((positive >= threshold).sum())
        fn = int((positive < threshold).sum())
        fp = int((negative >= threshold).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"method": method, "threshold": float(threshold),
                     "true_positive": tp, "false_positive": fp, "false_negative": fn,
                     "precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(f1, 4)})
    return pd.DataFrame(rows)


def annotate(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    out = pairs.copy()
    for side, frame, column in (("a", left, "index_a"), ("b", right, "index_b")):
        out[f"{side}_id"] = frame.loc[out[column], "image_id"].to_numpy()
        out[f"{side}_path"] = frame.loc[out[column], "image_path"].to_numpy()
        out[f"{side}_label"] = frame.loc[out[column], "label"].to_numpy()
    out["cross_class"] = out["a_label"] != out["b_label"]
    return out


def write_spot_check(pairs, odir, merged, scores, directory: Path) -> None:
    """Side-by-side images so a human can confirm the same-filename assumption."""
    directory.mkdir(parents=True, exist_ok=True)
    order = np.argsort(scores)          # weakest matches first - the informative ones
    chosen = list(order[:SPOT_CHECK // 2]) + list(order[-SPOT_CHECK // 2:])
    for position, index in enumerate(chosen, start=1):
        left_index, right_index = pairs[index]
        try:
            canvas = Image.new("RGB", (THUMB * 2 + 10, THUMB), "white")
            for offset, path in ((0, odir.loc[left_index, "image_path"]),
                                 (THUMB + 10, merged.loc[right_index, "image_path"])):
                with Image.open(path) as image:
                    canvas.paste(image.convert("RGB").resize((THUMB, THUMB)), (offset, 0))
            name = f"spot_{position:02d}_sim{scores[index]:.3f}.jpg"
            canvas.save(directory / name, quality=88)
        except Exception as exc:  # noqa: BLE001
            print(f"    could not render spot-check {position}: {exc}")


def main() -> int:
    manifests, processed = output_dir("manifests"), output_dir("processed")
    odir = pd.read_csv(manifests / "odir_manifest.csv", encoding="utf-8")
    merged = pd.read_csv(manifests / "merged_manifest.csv", encoding="utf-8")
    _, fp_odir = load(processed / "fingerprints_odir.npz")
    _, fp_merged = load(processed / "fingerprints_merged.npz")

    print("=" * 72)
    print("DUPLICATE DETECTION - FOV-normalised fingerprints")
    print("=" * 72)

    # --- calibration against free ground truth ----------------------------------------
    positives, negatives = ground_truth(odir, merged)
    print(f"\nGround truth: {len(positives)} positives (shared filename), "
          f"{len(negatives)} negatives")

    pos_fp = score_pairs(positives, fp_odir, fp_merged)
    by_name = {n: i for i, n in enumerate(odir["original_filename"])}
    pool = [k for k, n in enumerate(merged["original_filename"]) if n not in by_name]
    print("  scoring hard negatives (best ODIR match for each non-matching image)...")
    neg_fp = hard_negatives(fp_odir, fp_merged, pool)
    calibration = sweep_scores(pos_fp, neg_fp, "fingerprint")
    print(f"  true duplicates : median {np.median(pos_fp):.4f}  p5 {np.percentile(pos_fp, 5):.4f}")
    print(f"  hard negatives  : median {np.median(neg_fp):.4f}  max {neg_fp.max():.4f}")

    # Re-measure the rejected embedding detector on identical pairs, for the report.
    embedding_cache = processed / "embeddings_odir.npz"
    if embedding_cache.is_file() and (processed / "embeddings_merged.npz").is_file():
        _, em_odir = load(embedding_cache)
        _, em_merged = load(processed / "embeddings_merged.npz")
        calibration = pd.concat([
            calibration,
            sweep_scores(score_pairs(positives, em_odir, em_merged),
                         hard_negatives(em_odir, em_merged, pool), "imagenet_embedding"),
        ], ignore_index=True)

    print("\n" + "-" * 72)
    print("DETECTOR CALIBRATION  (selected thresholds)")
    print("-" * 72)
    shown = calibration[calibration["threshold"].isin([0.95, 0.98, 0.99, 0.995, 0.999])]
    print(shown.to_string(index=False))
    calibration.to_csv(manifests / "duplicate_detector_calibration.csv", index=False)

    at_operating = calibration[
        (calibration["method"] == "fingerprint")
        & (np.isclose(calibration["threshold"], DUPLICATE_THRESHOLD))
    ].iloc[0].to_dict()
    print(f"\nOperating point {DUPLICATE_THRESHOLD}: "
          f"precision {at_operating['precision']}, recall {at_operating['recall']}")

    write_spot_check(positives, odir, merged, pos_fp, manifests / "duplicate_review")

    # --- apply -------------------------------------------------------------------------
    print("\nSearching within ODIR...")
    within = annotate(
        find_candidate_pairs(fp_odir, threshold=CANDIDATE_THRESHOLD, verbose=False),
        odir, odir)
    confirmed_within = within[within["similarity"] >= DUPLICATE_THRESHOLD] if len(within) else within
    print(f"  candidates {len(within)}   confirmed {len(confirmed_within)}   "
          f"cross-class {int(confirmed_within['cross_class'].sum()) if len(confirmed_within) else 0}")

    print("Searching ODIR vs merged...")
    across = annotate(
        find_candidate_pairs(fp_odir, fp_merged, threshold=CANDIDATE_THRESHOLD, verbose=False),
        odir, merged)
    confirmed_across = across[across["similarity"] >= DUPLICATE_THRESHOLD] if len(across) else across
    contaminated = int(confirmed_across["b_id"].nunique()) if len(confirmed_across) else 0
    print(f"  candidates {len(across)}   confirmed {len(confirmed_across)}   "
          f"merged images also in ODIR: {contaminated}")

    within.to_csv(manifests / "duplicate_candidates_within_odir.csv", index=False, encoding="utf-8")
    across.to_csv(manifests / "duplicate_candidates_odir_vs_merged.csv", index=False, encoding="utf-8")

    groups = group_duplicates(confirmed_within, n_items=len(odir)) if len(confirmed_within) else []
    group_rows = [
        {"group_id": gid, "image_id": odir.loc[i, "image_id"], "label": odir.loc[i, "label"],
         "patient_id": odir.loc[i, "patient_id"], "keep": position == 0}
        for gid, members in enumerate(groups)
        for position, i in enumerate(sorted(members))
    ]
    group_frame = pd.DataFrame(group_rows,
                               columns=["group_id", "image_id", "label", "patient_id", "keep"])
    group_frame.to_csv(manifests / "duplicate_groups.csv", index=False, encoding="utf-8")

    if groups:
        sizes = sorted((len(g) for g in groups), reverse=True)
        print(f"  group sizes: largest {sizes[0]}, median {int(np.median(sizes))}")
        if sizes[0] > 10:
            print("  WARNING: a large group suggests transitive chaining - check the threshold")
    dropped = set(group_frame.loc[~group_frame["keep"], "image_id"]) if len(group_frame) else set()
    clean = odir[~odir["image_id"].isin(dropped)].copy()
    clean.to_csv(manifests / "odir_manifest_clean.csv", index=False, encoding="utf-8")

    print("\n" + "-" * 72)
    print("RESULT")
    print("-" * 72)
    print(f"  duplicate groups in ODIR      {len(groups)}")
    print(f"  images dropped                {len(dropped)}")
    print(f"  clean manifest                {len(clean)} images, "
          f"{clean['patient_id'].nunique()} patients")
    print(f"  merged images also in ODIR    {contaminated} "
          f"({contaminated / len(merged) * 100:.1f}%) - excluded from cross-source test")

    report = {
        "method": "FOV-normalised 64x64 fingerprint, cosine similarity",
        "rejected_method": "ImageNet ResNet50 embedding cosine - see calibration table",
        "ground_truth_assumption": "shared original filename between ODIR and the merged "
                                   "benchmark means the same eye; spot-check images provided",
        "ground_truth_positives": len(positives),
        "ground_truth_negatives": len(negatives),
        "candidate_threshold": CANDIDATE_THRESHOLD,
        "operating_threshold": DUPLICATE_THRESHOLD,
        "detector_at_operating_point": at_operating,
        "recall_caveat": "recall is measured over ODIR<->merged ground-truth pairs; recall "
                         "for within-ODIR duplicates is not directly measurable",
        "within_odir_duplicate_groups": len(groups),
        "within_odir_images_dropped": len(dropped),
        "clean_images": int(len(clean)),
        "clean_patients": int(clean["patient_id"].nunique()),
        "merged_images_contaminated_by_odir": contaminated,
        "status": "CALIBRATED",
    }
    (manifests / "deduplication_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nWritten to {manifests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
