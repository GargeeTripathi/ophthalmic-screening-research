"""E0a / E0b / E0c - how much of the merged benchmark's label is acquisition provenance?

E0a  metadata only     Predict the CLASS from file properties alone - extension, encoding,
                       dimensions, byte size, filename convention. No pixels at all.
E0b  normalised pixels Predict the SOURCE from FOV-normalised fingerprints. Answers whether
                       provenance survives the preprocessing that removes the obvious cues.
E0c  geometry control  Two controls that guard against the probe secretly reading pathology:
                       (i) predict source from geometry alone, no pixels;
                       (ii) predict class from pixels WITHIN a single source, where
                            provenance is constant and only pathology can help.

    python scripts/provenance_probe.py

Writes results/E0_provenance_probes.json and a per-experiment CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fingerprint import load  # noqa: E402
from src.provenance.features import PROBE_FEATURES  # noqa: E402
from src.utils.config import output_dir  # noqa: E402

FOLDS = 5
SEED = 42

CATEGORICAL = ["file_extension", "image_format", "colour_mode",
               "filename_convention", "resolution_bucket"]
NUMERIC = ["width", "height", "size_bytes", "aspect_ratio"]
GEOMETRY_ONLY = ["width", "height", "size_bytes", "aspect_ratio"]


def evaluate(X, y, model, name: str, description: str) -> dict:
    """Cross-validated accuracy and macro-F1 against a majority-class baseline."""
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    predicted = cross_val_predict(model, X, y, cv=splitter)
    baseline = cross_val_predict(
        DummyClassifier(strategy="most_frequent"), X, y, cv=splitter)

    result = {
        "experiment": name,
        "description": description,
        "n": int(len(y)),
        "classes": sorted(map(str, pd.unique(y))),
        "baseline_accuracy": round(float(accuracy_score(y, baseline)), 4),
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "macro_f1": round(float(f1_score(y, predicted, average="macro")), 4),
    }
    result["lift_over_baseline"] = round(result["accuracy"] - result["baseline_accuracy"], 4)
    result["confusion_matrix"] = confusion_matrix(y, predicted).tolist()
    return result


def tabular_model(categorical: list[str], numeric: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ])),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    ])


def main() -> int:
    manifests, processed, results = output_dir("manifests"), output_dir("processed"), output_dir("results")
    merged = pd.read_csv(manifests / "merged_manifest.csv", encoding="utf-8")
    _, fingerprints = load(processed / "fingerprints_merged.npz")

    print("=" * 72)
    print("E0 - PROVENANCE PROBES on the merged benchmark")
    print("=" * 72)
    collected = []

    # --- E0a: can file metadata alone predict the disease class? -----------------------
    print("\nE0a  metadata only -> CLASS")
    features = merged[[c for c in PROBE_FEATURES]].copy()
    categorical = [c for c in CATEGORICAL if c in features.columns]
    numeric = [c for c in NUMERIC if c in features.columns]
    features[numeric] = features[numeric].fillna(-1)
    e0a = evaluate(features, merged["label"], tabular_model(categorical, numeric),
                   "E0a", "predict class from file metadata only, no pixels")
    collected.append(e0a)
    print(f"  baseline {e0a['baseline_accuracy']:.3f}  ->  probe {e0a['accuracy']:.3f}   "
          f"(+{e0a['lift_over_baseline']:.3f})   macro-F1 {e0a['macro_f1']:.3f}")

    # --- E0b: does provenance survive FOV normalisation? -------------------------------
    print("\nE0b  FOV-normalised pixels -> SOURCE")
    pixel_model = LogisticRegression(max_iter=2000, n_jobs=-1)
    e0b = evaluate(fingerprints, merged["source"].to_numpy(), pixel_model,
                   "E0b", "predict acquisition source from FOV-normalised pixels")
    collected.append(e0b)
    print(f"  baseline {e0b['baseline_accuracy']:.3f}  ->  probe {e0b['accuracy']:.3f}   "
          f"(+{e0b['lift_over_baseline']:.3f})   macro-F1 {e0b['macro_f1']:.3f}")
    if e0b["lift_over_baseline"] < 0.05:
        print("  STOP CONDITION MET: provenance does not survive normalisation.")

    # --- E0c(i): geometry only -> source ------------------------------------------------
    print("\nE0c-i  geometry only -> SOURCE  (no pixels, no encoding)")
    geometry = merged[GEOMETRY_ONLY].fillna(-1)
    e0c1 = evaluate(geometry, merged["source"].to_numpy(),
                    Pipeline([("s", StandardScaler()),
                              ("c", RandomForestClassifier(n_estimators=300,
                                                           random_state=SEED, n_jobs=-1))]),
                    "E0c-i", "predict source from image geometry alone")
    collected.append(e0c1)
    print(f"  baseline {e0c1['baseline_accuracy']:.3f}  ->  probe {e0c1['accuracy']:.3f}   "
          f"(+{e0c1['lift_over_baseline']:.3f})")

    # --- E0c(ii): within one source, can pixels still predict class? -------------------
    # This is the anti-circularity control. Inside a single source, provenance is constant,
    # so anything the probe achieves must come from retinal content.
    print("\nE0c-ii  pixels -> CLASS, within a single source")
    for source in merged["source"].unique():
        mask = (merged["source"] == source).to_numpy()
        subset_labels = merged.loc[mask, "label"]
        if subset_labels.nunique() < 2:
            print(f"  {source:<16} single class only - not evaluable")
            continue
        result = evaluate(fingerprints[mask], subset_labels.to_numpy(), pixel_model,
                          f"E0c-ii[{source}]",
                          f"predict class from pixels within source={source}")
        collected.append(result)
        print(f"  {source:<16} baseline {result['baseline_accuracy']:.3f}  ->  "
              f"{result['accuracy']:.3f}   (+{result['lift_over_baseline']:.3f})   "
              f"n={result['n']}")

    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"}
                          for r in collected])
    frame.to_csv(results / "E0_provenance_probes.csv", index=False)
    (results / "E0_provenance_probes.json").write_text(
        json.dumps(collected, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print(frame[["experiment", "n", "baseline_accuracy", "accuracy",
                 "lift_over_baseline", "macro_f1"]].to_string(index=False))
    print(f"\nWRITTEN to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
