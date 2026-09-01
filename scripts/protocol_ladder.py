"""E1 - the protocol ladder. The project's headline experiment.

Question: how much of a reported score on this task is leakage rather than learning?

Method: hold the model fixed and vary ONLY the split. Four rungs, from the split most
published work uses down to a patient-disjoint, deduplicated, provenance-controlled one.
Any difference between rungs is attributable to the protocol, because nothing else moves.

The classifier is a multinomial logistic head on frozen ImageNet ResNet50 features. This is
deliberate. The question is about the SPLIT, not about reaching state of the art, and a
frozen-feature head trains in seconds - so all four rungs run identically, many times, on a
CPU. Absolute numbers here are lower than a fine-tuned backbone would give; the comparison
between rungs is what the experiment is for. Fine-tuned baselines are E2/E3 and run on GPU.

    python scripts/protocol_ladder.py

Writes results/E1_protocol_ladder.{json,csv} and pooled out-of-fold predictions per rung.

RESULT AND ITS LIMITATION (recorded after the first run)
--------------------------------------------------------
On ODIR this ladder found NO significant inflation: macro-F1 0.5496 (image-level) vs 0.5444
(patient-disjoint), McNemar p = 0.75, confidence intervals overlapping throughout.

That is a real negative result for ODIR, but it must be reported with its limitation. A
frozen linear head has almost no capacity to memorise, and leakage inflates a score by
letting a model recall a near-copy it has already seen. A model that cannot memorise cannot
benefit from leakage, so this measures a LOWER BOUND on protocol inflation, not the
inflation a fine-tuned backbone would show. Establishing that needs E2/E3 on GPU.

The provenance confound itself is not in ODIR - it is in the merged benchmark, where E0a
shows file metadata alone predicts the class at 0.667 against a 0.260 baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embeddings import load  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    classification_summary,
    macro_f1_of,
    mcnemar_exact,
    patient_bootstrap_ci,
)
from src.utils.config import load_config, output_dir  # noqa: E402

RUNGS = [
    ("image_level", "stratified by class, patient identity ignored (as commonly published)"),
    ("patient_disjoint", "grouped by patient"),
    ("patient_disjoint_dedup", "grouped by patient, near-duplicates removed"),
    ("provenance_controlled", "as above, restricted to one resolution cluster"),
]


def head() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)),
    ])


def run_rung(split: pd.DataFrame, vectors: np.ndarray, index: dict[str, int],
             labels: list[str]) -> pd.DataFrame:
    """Pooled out-of-fold predictions for one split variant."""
    rows = split[split["image_id"].isin(index)].reset_index(drop=True)
    X = vectors[[index[i] for i in rows["image_id"]]]
    y = rows["label"].to_numpy()

    predictions = np.empty(len(rows), dtype=object)
    scores = np.zeros((len(rows), len(labels)), dtype=float)

    for fold in sorted(rows["fold"].unique()):
        test_mask = (rows["fold"] == fold).to_numpy()
        model = head().fit(X[~test_mask], y[~test_mask])
        predictions[test_mask] = model.predict(X[test_mask])
        probabilities = model.predict_proba(X[test_mask])
        for position, code in enumerate(model.classes_):
            scores[test_mask, labels.index(code)] = probabilities[:, position]

    return pd.DataFrame({
        "image_id": rows["image_id"], "patient_id": rows["patient_id"],
        "y_true": y, "y_pred": predictions,
        **{f"p_{code}": scores[:, i] for i, code in enumerate(labels)},
    })


def main() -> int:
    labels = load_config("data")["classes"]
    manifests, processed = output_dir("manifests"), output_dir("processed")
    splits_dir, results = output_dir("splits"), output_dir("results")

    ids, vectors = load(processed / "embeddings_odir.npz")
    index = {image_id: position for position, image_id in enumerate(ids)}

    print("=" * 72)
    print("E1 - PROTOCOL LADDER  (frozen ResNet50 features + logistic head)")
    print("=" * 72)

    collected, predictions_by_rung = [], {}

    for name, description in RUNGS:
        path = splits_dir / f"splits_{name}.csv"
        if not path.is_file():
            print(f"\n{name}: split missing - run scripts/build_splits.py first")
            continue
        split = pd.read_csv(path, encoding="utf-8")
        print(f"\n{name}  ({len(split)} images, {split['patient_id'].nunique()} patients)")
        print(f"  {description}")

        predicted = run_rung(split, vectors, index, labels)
        predictions_by_rung[name] = predicted
        predicted.to_csv(results / f"E1_predictions_{name}.csv", index=False, encoding="utf-8")

        summary = classification_summary(
            predicted["y_true"], predicted["y_pred"], labels,
            y_score=predicted[[f"p_{c}" for c in labels]].to_numpy())
        interval = patient_bootstrap_ci(predicted, macro_f1_of)

        record = {"rung": name, "description": description, **summary,
                  "macro_f1_ci": interval}
        collected.append(record)
        print(f"  accuracy {summary['accuracy']:.4f}   macro-F1 {summary['macro_f1']:.4f}"
              f"   95% CI [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]")

    # --- the comparison the experiment exists for --------------------------------------
    print("\n" + "-" * 72)
    print("INFLATION ATTRIBUTABLE TO THE PROTOCOL")
    print("-" * 72)
    comparisons = []
    if "image_level" in predictions_by_rung:
        reference = predictions_by_rung["image_level"]
        base_f1 = macro_f1_of(reference)
        for name in [r[0] for r in RUNGS[1:]]:
            if name not in predictions_by_rung:
                continue
            other = predictions_by_rung[name]
            delta = base_f1 - macro_f1_of(other)
            entry = {"from": "image_level", "to": name,
                     "macro_f1_drop": round(float(delta), 4)}

            # Paired test only where the two rungs score the same images.
            shared = set(reference["image_id"]) & set(other["image_id"])
            if len(shared) > 100:
                a = reference[reference["image_id"].isin(shared)].sort_values("image_id")
                b = other[other["image_id"].isin(shared)].sort_values("image_id")
                entry["paired_n"] = len(shared)
                entry["mcnemar"] = mcnemar_exact(
                    (a["y_true"].to_numpy() == a["y_pred"].to_numpy()),
                    (b["y_true"].to_numpy() == b["y_pred"].to_numpy()))
            comparisons.append(entry)
            print(f"  image_level -> {name:<26} macro-F1 drop {delta:+.4f}"
                  + (f"   McNemar p={entry['mcnemar']['p_value']}" if "mcnemar" in entry else ""))

    frame = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ("per_class", "confusion_matrix", "macro_f1_ci", "labels")}
                          | {"ci_low": r["macro_f1_ci"]["ci_low"],
                             "ci_high": r["macro_f1_ci"]["ci_high"]}
                          for r in collected])
    frame.to_csv(results / "E1_protocol_ladder.csv", index=False)
    (results / "E1_protocol_ladder.json").write_text(
        json.dumps({"rungs": collected, "comparisons": comparisons}, indent=2, default=str),
        encoding="utf-8")

    print("\n" + "=" * 72)
    print(frame[["rung", "n", "accuracy", "macro_f1", "ci_low", "ci_high"]].to_string(index=False))
    print(f"\nWRITTEN to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
