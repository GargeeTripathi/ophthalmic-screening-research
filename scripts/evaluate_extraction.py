"""E7 - extraction benchmark, and E8 - the reliability study.

E7  Both extractors, on development layouts and on the sealed layouts, across five
    degradation levels. Per-field precision/recall/F1, exact match, error taxonomy.

E8  The reliability question this project exists to answer:
        Is raw OCR confidence a usable estimate of whether an extracted clinical field is
        CORRECT, and does a learned estimator do better?
    Trained on development layouts only, evaluated on the sealed ones.

    python scripts/evaluate_extraction.py

Writes results/E7_extraction.{json,csv}, results/E8_reliability.json and
data/processed/extractions_{split}.csv (the per-field table the deferral stage consumes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import expected_calibration_error  # noqa: E402
from src.idp.extract import EXTRACTORS, FEATURE_NAMES, error_category, is_correct  # noqa: E402
from src.idp.ocr import load as load_ocr  # noqa: E402
from src.utils.config import load_config, output_dir  # noqa: E402

SEED = 42


def extract_split(split: str, fields: list[str]) -> pd.DataFrame:
    """Run both extractors over one split, one row per (document, extractor, field)."""
    documents_dir, processed = output_dir("documents"), output_dir("processed")
    records = json.loads((documents_dir / f"ground_truth_{split}.json").read_text(encoding="utf-8"))
    tokens_by_document = load_ocr(processed / f"ocr_{split}.json")

    rows = []
    for record in records:
        tokens = tokens_by_document.get(record["document_id"], [])
        for extractor_name, extractor in EXTRACTORS.items():
            extracted = extractor(tokens, fields)
            for name in fields:
                result = extracted[name]
                truth = record["fields"].get(name)
                rows.append({
                    "document_id": record["document_id"],
                    "split": split, "layout": record["layout"],
                    "degradation_level": record["degradation_level"],
                    "extractor": extractor_name, "field": name,
                    "predicted": result.value, "truth": truth,
                    "correct": bool(is_correct(name, result.value, truth)),
                    "category": error_category(name, result.value, truth),
                    "truth_present": truth is not None,
                    "predicted_present": result.value is not None,
                    **{f: result.features.get(f, 0.0) for f in FEATURE_NAMES},
                })
    return pd.DataFrame(rows)


def field_scores(frame: pd.DataFrame) -> dict:
    """Precision/recall/F1 over field PRESENCE-and-correctness."""
    predicted = frame["predicted_present"]
    truth = frame["truth_present"]
    correct = frame["correct"]

    true_positive = int((predicted & truth & correct).sum())
    false_positive = int((predicted & (~truth | ~correct)).sum())
    false_negative = int((truth & (~predicted | ~correct)).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "exact_match": round(float(correct.mean()), 4), "n": int(len(frame))}


def main() -> int:
    idp = load_config("idp")
    fields = idp["fields"]
    processed, results = output_dir("processed"), output_dir("results")

    print("=" * 72)
    print("E7 - EXTRACTION BENCHMARK")
    print("=" * 72)

    frames = {}
    for split in ("dev", "sealed"):
        frames[split] = extract_split(split, fields)
        frames[split].to_csv(processed / f"extractions_{split}.csv", index=False, encoding="utf-8")
    everything = pd.concat(frames.values(), ignore_index=True)

    print("\nOVERALL by extractor and split")
    overall = []
    for (extractor, split), block in everything.groupby(["extractor", "split"]):
        overall.append({"extractor": extractor, "split": split, **field_scores(block)})
    overall_frame = pd.DataFrame(overall)
    print(overall_frame.to_string(index=False))

    print("\nBY DEGRADATION LEVEL (F1)")
    by_level = (everything.groupby(["extractor", "split", "degradation_level"])
                .apply(lambda b: field_scores(b)["f1"], include_groups=False)
                .unstack(level="degradation_level"))
    print(by_level.to_string())

    print("\nBY FIELD (F1, sealed layouts)")
    sealed = everything[everything["split"] == "sealed"]
    by_field = (sealed.groupby(["extractor", "field"])
                .apply(lambda b: field_scores(b)["f1"], include_groups=False)
                .unstack(level="extractor"))
    print(by_field.to_string())

    print("\nERROR TAXONOMY (counts, sealed layouts)")
    taxonomy = pd.crosstab(sealed["category"], sealed["extractor"])
    print(taxonomy.to_string())

    # pandas gives tuple keys for a MultiIndex; JSON only accepts scalar keys.
    def stringify(mapping: dict) -> dict:
        return {str(k): (stringify(v) if isinstance(v, dict) else v)
                for k, v in mapping.items()}

    (results / "E7_extraction.json").write_text(json.dumps({
        "overall": overall,
        "by_level": stringify(by_level.to_dict()),
        "by_field_sealed": stringify(by_field.to_dict()),
        "error_taxonomy_sealed": stringify(taxonomy.to_dict()),
        "caveat": "synthetic documents; a technical evaluation under controlled "
                  "degradation, not clinical validation",
    }, indent=2, default=str), encoding="utf-8")
    overall_frame.to_csv(results / "E7_extraction.csv", index=False)

    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("E8 - RELIABILITY: is OCR confidence a good correctness estimate?")
    print("=" * 72)

    reliability = {}
    for extractor in EXTRACTORS:
        train = frames["dev"][(frames["dev"]["extractor"] == extractor)
                              & frames["dev"]["predicted_present"]]
        test = frames["sealed"][(frames["sealed"]["extractor"] == extractor)
                                & frames["sealed"]["predicted_present"]]
        if len(train) < 50 or len(test) < 50 or test["correct"].nunique() < 2:
            print(f"\n{extractor}: too few usable rows - skipped")
            continue

        y_train = train["correct"].astype(int).to_numpy()
        y_test = test["correct"].astype(int).to_numpy()

        # Signal A: raw OCR confidence, used directly as P(correct).
        raw = test["ocr_confidence"].to_numpy()

        # Signal B: learned, calibrated correctness estimator.
        model = GradientBoostingClassifier(random_state=SEED)
        model.fit(train[FEATURE_NAMES], y_train)
        learned = model.predict_proba(test[FEATURE_NAMES])[:, 1]

        entry = {"n_train": int(len(train)), "n_test": int(len(test)),
                 "test_correct_rate": round(float(y_test.mean()), 4)}
        for name, scores in (("raw_ocr_confidence", raw), ("learned_estimator", learned)):
            probabilities = np.column_stack([1 - scores, scores])
            entry[name] = {
                "auroc": round(float(roc_auc_score(y_test, scores)), 4)
                if len(set(y_test)) > 1 else None,
                "brier": round(float(brier_score_loss(y_test, np.clip(scores, 0, 1))), 4),
                "ece": expected_calibration_error(probabilities, y_test.astype(str),
                                                  labels=["0", "1"]),
            }
        fraction, mean_predicted = calibration_curve(
            y_test, np.clip(learned, 0, 1), n_bins=8, strategy="quantile")
        entry["learned_calibration_curve"] = {
            "mean_predicted": [round(float(v), 4) for v in mean_predicted],
            "observed_fraction": [round(float(v), 4) for v in fraction],
        }
        entry["feature_importance"] = dict(zip(
            FEATURE_NAMES, [round(float(v), 4) for v in model.feature_importances_]))
        reliability[extractor] = entry

        print(f"\n{extractor}  (train {len(train)} dev rows, test {len(test)} sealed rows, "
              f"{y_test.mean():.1%} correct)")
        for name in ("raw_ocr_confidence", "learned_estimator"):
            e = entry[name]
            print(f"  {name:<22} AUROC {e['auroc']}   Brier {e['brier']}   ECE {e['ece']}")
        print(f"  top features: {sorted(entry['feature_importance'].items(), key=lambda kv: -kv[1])[:3]}")

    (results / "E8_reliability.json").write_text(
        json.dumps(reliability, indent=2, default=str), encoding="utf-8")

    print(f"\nWRITTEN to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
