"""E5 - transfer to acquisition sources the model never saw.

Train on all of ODIR, test on the parts of the merged benchmark that are NOT ODIR copies.
Deduplication established which those are: 1,626 merged images are ODIR images at a
different scale and are excluded here, so nothing in the test set was seen in training.

WHAT THIS MEASURES, PRECISELY
-----------------------------
Transfer under a documented change of acquisition source. It is NOT "generalisation" in the
loose sense, and the report should not use that word unqualified. Each remaining source
carries only a subset of the classes, which is itself a consequence of the provenance
confound - so per-source performance is reported separately rather than pooled into one
number that would hide the structure.

    eyepacs_jpeg   diabetic retinopathy only
    crop_jpg256    cataract and glaucoma only
    png_highres    cataract and glaucoma only

Because each source is class-restricted, plain accuracy is compared against the accuracy of
always guessing that source's majority class. Beating that is the bar.

    python scripts/cross_source_transfer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embeddings import load  # noqa: E402
from src.utils.config import load_config, output_dir  # noqa: E402

SEED = 42
DUPLICATE_THRESHOLD = 0.999


def contaminated_ids(manifests: Path) -> set[str]:
    """Merged images confirmed to be ODIR copies. They cannot serve as held-out test data."""
    path = manifests / "duplicate_candidates_odir_vs_merged.csv"
    if not path.is_file():
        return set()
    pairs = pd.read_csv(path, encoding="utf-8")
    confirmed = pairs[pairs["similarity"] >= DUPLICATE_THRESHOLD]
    return set(confirmed["b_id"]) if len(confirmed) else set()


def main() -> int:
    labels = load_config("data")["classes"]
    manifests, processed, results = (output_dir("manifests"), output_dir("processed"),
                                     output_dir("results"))

    odir = pd.read_csv(manifests / "odir_manifest_clean.csv", encoding="utf-8")
    merged = pd.read_csv(manifests / "merged_manifest.csv", encoding="utf-8")
    odir_ids, odir_vectors = load(processed / "embeddings_odir.npz")
    merged_ids, merged_vectors = load(processed / "embeddings_merged.npz")

    odir_index = {i: p for p, i in enumerate(odir_ids)}
    merged_index = {i: p for p, i in enumerate(merged_ids)}

    print("=" * 72)
    print("E5 - CROSS-SOURCE TRANSFER")
    print("=" * 72)

    # --- train on every ODIR image ------------------------------------------------------
    train_rows = odir[odir["image_id"].isin(odir_index)]
    X_train = odir_vectors[[odir_index[i] for i in train_rows["image_id"]]]
    y_train = train_rows["label"].to_numpy()
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000, n_jobs=-1))]).fit(X_train, y_train)
    print(f"\ntrained on {len(train_rows)} ODIR images "
          f"({train_rows['patient_id'].nunique()} patients)")

    # --- test on merged images that are NOT ODIR copies ---------------------------------
    excluded = contaminated_ids(manifests)
    test_rows = merged[~merged["image_id"].isin(excluded) & merged["image_id"].isin(merged_index)]
    print(f"merged images: {len(merged)}   confirmed ODIR copies removed: "
          f"{len(merged) - len(test_rows)}   usable test images: {len(test_rows)}")

    collected = []
    print("\n" + "-" * 72)
    print(f"{'source':<16}{'n':>7}{'classes':>10}{'majority':>11}{'accuracy':>11}{'macro-F1':>11}")
    print("-" * 72)

    for source, block in test_rows.groupby("source"):
        X = merged_vectors[[merged_index[i] for i in block["image_id"]]]
        y = block["label"].to_numpy()
        predicted = model.predict(X)

        present = sorted(set(y))
        majority = float(pd.Series(y).value_counts(normalize=True).max())
        accuracy = float((predicted == y).mean())
        macro = float(f1_score(y, predicted, average="macro", zero_division=0))

        entry = {
            "source": source, "n": int(len(block)), "classes_present": present,
            "majority_class_accuracy": round(majority, 4),
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro, 4),
            "lift_over_majority": round(accuracy - majority, 4),
            "predicted_distribution": pd.Series(predicted).value_counts().to_dict(),
            "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist(),
            "labels": labels,
        }
        collected.append(entry)
        print(f"{source:<16}{entry['n']:>7}{','.join(present):>10}"
              f"{majority:>11.4f}{accuracy:>11.4f}{macro:>11.4f}")

    pooled_accuracy = float(np.mean([e["accuracy"] for e in collected])) if collected else 0.0

    print("\n" + "-" * 72)
    print("READING THIS TABLE")
    print("-" * 72)
    print("  Each source carries only a subset of classes - itself a symptom of the")
    print("  provenance confound - so 'majority' is the accuracy of always guessing that")
    print("  source's dominant class. Beating it is the bar. Pooling these into a single")
    print("  number would hide exactly the structure the audit is about.")

    (results / "E5_cross_source_transfer.json").write_text(json.dumps({
        "train_images": int(len(train_rows)),
        "train_patients": int(train_rows["patient_id"].nunique()),
        "excluded_as_odir_copies": int(len(merged) - len(test_rows)),
        "per_source": collected,
        "mean_accuracy_across_sources": round(pooled_accuracy, 4),
        "interpretation": "transfer under a documented change of acquisition source; not "
                          "generalisation in the unqualified sense",
    }, indent=2, default=str), encoding="utf-8")

    pd.DataFrame([{k: v for k, v in e.items()
                   if k not in ("confusion_matrix", "labels", "predicted_distribution",
                                "classes_present")}
                  for e in collected]).to_csv(
        results / "E5_cross_source_transfer.csv", index=False)

    print(f"\nWRITTEN to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
