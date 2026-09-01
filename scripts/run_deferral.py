"""E9 - System A vs System B, and E10 - the robustness matrix.

THE STRUCTURAL RULE THIS CODE ENFORCES
--------------------------------------
The document NEVER changes the predicted class. `argmax(p_img)` is the answer whenever the
system answers at all. The document, and how reliably it was read, only decide WHETHER to
answer. That removes the text-dominance failure mode by construction rather than by
penalty, and it is a sentence a clinician can be told verbatim.

System A  rule based: defer when image confidence is low OR the document contradicts the
          image prediction. Two thresholds, no learning.
System B  learned gate over [entropy(p_img), onehot(stated diagnosis), c_hat, agreement].
          Trained to predict "the image prediction is wrong".

E10 crosses 9 evidence conditions with 5 degradation levels and 3 corruption rates. Effects
there are large by construction, unlike adjacent-ablation deltas, which on this dataset are
underpowered - see the note in scripts/protocol_ladder.py.

    python scripts/run_deferral.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deferral.systems import (  # noqa: E402
    SYSTEM_A_CONFIDENCE, aurc, risk_coverage_curve, system_a_scores,
)
from src.idp.extract import FEATURE_NAMES, normalise  # noqa: E402
from src.utils.config import load_config, output_dir  # noqa: E402

SEED = 42
COVERAGES = [0.9, 0.75, 0.5]


def build_paired_table(labels: list[str]) -> pd.DataFrame:
    """One row per document, joined to the image model's prediction for that same eye."""
    documents_dir, processed, results = (output_dir("documents"), output_dir("processed"),
                                         output_dir("results"))

    predictions = pd.read_csv(results / "E1_predictions_patient_disjoint_dedup.csv",
                              encoding="utf-8")
    predictions = predictions.set_index("image_id")

    rows = []
    for split in ("dev", "sealed"):
        records = json.loads(
            (documents_dir / f"ground_truth_{split}.json").read_text(encoding="utf-8"))
        extractions = pd.read_csv(processed / f"extractions_{split}.csv", encoding="utf-8")
        # The deferral stage consumes E2 (layout-aware) extractions of the diagnosis field.
        diagnosis = extractions[(extractions["extractor"] == "E2")
                                & (extractions["field"] == "stated_diagnosis")]
        diagnosis = diagnosis.set_index("document_id")

        for record in records:
            eye = normalise("eye", record["fields"].get("eye"))
            image_id = f"{record['patient_id']}_{eye}"
            if image_id not in predictions.index or record["document_id"] not in diagnosis.index:
                continue
            prediction = predictions.loc[image_id]
            extraction = diagnosis.loc[record["document_id"]]

            probabilities = np.array([float(prediction[f"p_{c}"]) for c in labels])
            stated = record["stated_diagnosis_label"]
            rows.append({
                "document_id": record["document_id"], "split": split,
                "layout": record["layout"], "degradation_level": record["degradation_level"],
                "image_id": image_id, "patient_id": record["patient_id"],
                "y_true": prediction["y_true"], "y_pred": prediction["y_pred"],
                "image_correct": bool(prediction["y_true"] == prediction["y_pred"]),
                "max_prob": float(probabilities.max()),
                "entropy": float(-(probabilities * np.log(probabilities + 1e-12)).sum()),
                "stated_label": stated,
                "document_correct": bool(stated == prediction["y_true"]),
                "diagnosis_corrupted": bool(record["diagnosis_corrupted"]),
                # agreement: probability the image assigns to the document's claim
                "agreement": float(probabilities[labels.index(stated)]),
                "extraction_correct": bool(extraction["correct"]),
                **{f: float(extraction[f]) for f in FEATURE_NAMES},
                **{f"p_{c}": float(probabilities[i]) for i, c in enumerate(labels)},
            })
    return pd.DataFrame(rows)


def main() -> int:
    labels = load_config("data")["classes"]
    results, processed = output_dir("results"), output_dir("processed")

    print("=" * 72)
    print("E9 / E10 - DEFERRAL")
    print("=" * 72)

    table = build_paired_table(labels)
    table.to_csv(processed / "deferral_table.csv", index=False, encoding="utf-8")
    print(f"\npaired rows: {len(table)}   "
          f"image correct: {table['image_correct'].mean():.1%}   "
          f"document correct: {table['document_correct'].mean():.1%}")

    # .copy() matters: these are slices, and pandas will not let a slice be assigned to.
    train = table[table["split"] == "dev"].copy()
    test = table[table["split"] == "sealed"].copy()
    if len(test) < 50:
        print("ERROR: too few sealed rows to evaluate")
        return 2

    # Target: should the image prediction be deferred? Defined as "the image is wrong".
    y_train = (~train["image_correct"]).astype(int).to_numpy()
    y_test = (~test["image_correct"]).astype(int).to_numpy()

    # --- learned reliability estimate for the document ---------------------------------
    reliability = GradientBoostingClassifier(random_state=SEED)
    reliability.fit(train[FEATURE_NAMES], train["extraction_correct"].astype(int))
    train["c_hat"] = reliability.predict_proba(train[FEATURE_NAMES])[:, 1]
    test["c_hat"] = reliability.predict_proba(test[FEATURE_NAMES])[:, 1]

    signals = {}

    # --- System 0: image confidence only (the baseline everything must beat) -----------
    signals["image_confidence_only"] = 1.0 - test["max_prob"].to_numpy()

    # --- System A: rules ----------------------------------------------------------------
    signals["system_A_rules"] = system_a_scores(
        test["max_prob"].to_numpy(), test["agreement"].to_numpy(),
        confidence_threshold=SYSTEM_A_CONFIDENCE)

    # --- System B: learned gate ---------------------------------------------------------
    def gate_features(frame: pd.DataFrame) -> np.ndarray:
        onehot = np.zeros((len(frame), len(labels)))
        for i, code in enumerate(frame["stated_label"]):
            onehot[i, labels.index(code)] = 1.0
        return np.column_stack([
            frame["entropy"].to_numpy(), frame["max_prob"].to_numpy(),
            frame["agreement"].to_numpy(), frame["c_hat"].to_numpy(), onehot,
        ])

    gate = GradientBoostingClassifier(random_state=SEED)
    gate.fit(gate_features(train), y_train)
    signals["system_B_learned"] = gate.predict_proba(gate_features(test))[:, 1]

    # --- System B-minus: same gate WITHOUT the reliability estimate --------------------
    # Isolates what c_hat contributes, which is the one architectural claim being made.
    def gate_features_no_chat(frame: pd.DataFrame) -> np.ndarray:
        return gate_features(frame)[:, [0, 1, 2] + list(range(4, 4 + len(labels)))]

    gate_no_chat = GradientBoostingClassifier(random_state=SEED)
    gate_no_chat.fit(gate_features_no_chat(train), y_train)
    signals["system_B_without_c_hat"] = gate_no_chat.predict_proba(
        gate_features_no_chat(test))[:, 1]

    print("\n" + "-" * 72)
    print("E9 - DEFERRAL PERFORMANCE (sealed layouts)")
    print("-" * 72)
    print(f"{'system':<26}{'AURC':>8}{'err-AUROC':>11}"
          + "".join(f"{f'acc@{int(c*100)}%':>10}" for c in COVERAGES))

    e9 = {}
    correct = test["image_correct"].to_numpy()
    for name, score in signals.items():
        curve = risk_coverage_curve(score, correct)
        entry = {
            "aurc": round(aurc(curve), 4),
            "error_detection_auroc": round(float(roc_auc_score(y_test, score)), 4)
            if len(set(y_test)) > 1 else None,
            "accuracy_at_coverage": {},
        }
        for coverage in COVERAGES:
            keep = int(np.ceil(coverage * len(score)))
            order = np.argsort(score)[:keep]
            entry["accuracy_at_coverage"][str(coverage)] = round(float(correct[order].mean()), 4)
        entry["curve"] = curve
        e9[name] = entry
        print(f"{name:<26}{entry['aurc']:>8.4f}{entry['error_detection_auroc']:>11.4f}"
              + "".join(f"{entry['accuracy_at_coverage'][str(c)]:>10.4f}" for c in COVERAGES))

    # --- E10: robustness matrix ---------------------------------------------------------
    print("\n" + "-" * 72)
    print("E10 - ROBUSTNESS MATRIX (accuracy at 75% coverage, System B)")
    print("-" * 72)

    conditions = {
        "correct_doc": test["document_correct"],
        "wrong_doc": ~test["document_correct"],
        "corrupted_diagnosis": test["diagnosis_corrupted"],
        "extraction_failed": ~test["extraction_correct"],
        "low_reliability": test["c_hat"] < 0.5,
        "image_correct": test["image_correct"],
        "image_wrong": ~test["image_correct"],
    }
    matrix = []
    for condition_name, mask in conditions.items():
        mask = mask.to_numpy()
        for level in sorted(test["degradation_level"].unique()):
            cell = mask & (test["degradation_level"] == level).to_numpy()
            if cell.sum() < 5:
                continue
            score = signals["system_B_learned"][cell]
            cell_correct = correct[cell]
            keep = max(1, int(np.ceil(0.75 * len(score))))
            order = np.argsort(score)[:keep]
            matrix.append({
                "condition": condition_name, "degradation_level": int(level),
                "n": int(cell.sum()),
                "accuracy_full_coverage": round(float(cell_correct.mean()), 4),
                "accuracy_at_75pct": round(float(cell_correct[order].mean()), 4),
                "deferral_rate": round(float((score > np.median(signals["system_B_learned"])).mean()), 4),
            })
    matrix_frame = pd.DataFrame(matrix)
    pivot = matrix_frame.pivot(index="condition", columns="degradation_level",
                               values="accuracy_at_75pct")
    print(pivot.to_string())
    matrix_frame.to_csv(results / "E10_robustness_matrix.csv", index=False)

    (results / "E9_deferral.json").write_text(json.dumps({
        "n_train": int(len(train)), "n_test": int(len(test)),
        "systems": {k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                    for k, v in e9.items()},
        "curves": {k: v["curve"] for k, v in e9.items()},
        "note": "the document never changes the predicted class; it only affects deferral",
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nWRITTEN to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
