"""E11 - is the result a disease detector, or a demographics detector?

THE WORRY
---------
The audit found ODIR is *less confounded*, not clean:

    cataract patients average 66.8 years, diabetic-retinopathy patients 55.7 - an 11-year gap
    cataract is 57.3% female, glaucoma 41.5%
    23.2% of glaucoma rows name a second condition in the annotation

Deep networks read age and sex from fundus photographs accurately - that is established
literature - so a model could score well on cataract-vs-rest by being partly an age detector.
Flagging this in a limitations section is not enough; it is testable.

FOUR TESTS
----------
C1  demographics-only     Predict the class from age and sex ALONE, no image. Whatever this
                          achieves is available to any model without looking at a retina.
C2  age-matched           Re-evaluate on an age-matched subsample, so the age distribution is
                          equalised across classes and the demographic route is closed.
C3  sex-matched           The same for sex.
C4  multi-finding removed Drop rows whose annotation names a second condition, since those
                          carry label noise concentrated in the smallest class.

If image performance survives C2-C4 roughly intact, the demographic confound is not driving
it. If it collapses, that is the finding and it must be reported as such.

    python scripts/confound_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import macro_f1_of, patient_bootstrap_ci  # noqa: E402
from src.utils.config import load_config, output_dir  # noqa: E402

SEED = 42
AGE_BIN = 5          # years per stratum when age-matching


def cataract_auroc(frame: pd.DataFrame, labels: list[str]) -> float:
    """Cataract-vs-rest AUROC - the project's stated primary clinical outcome."""
    truth = (frame["y_true"] == "C").astype(int)
    if truth.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(truth, frame["p_C"]))


def age_matched(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Subsample so every age stratum holds equal numbers of each class.

    Within a stratum the classes are balanced by downsampling to the smallest, which closes
    the route from age to label without reweighting or modelling anything.
    """
    rng = np.random.default_rng(seed)
    frame = frame.dropna(subset=["age"]).copy()
    frame["age_bin"] = (frame["age"] // AGE_BIN).astype(int)

    kept = []
    for _, stratum in frame.groupby("age_bin"):
        counts = stratum["label"].value_counts()
        if len(counts) < 2:
            continue                       # a single-class stratum cannot be matched
        take = int(counts.min())
        for _, block in stratum.groupby("label"):
            picked = rng.choice(len(block), size=take, replace=False)
            kept.append(block.iloc[picked])
    return pd.concat(kept, ignore_index=True) if kept else frame.head(0)


def sex_matched(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kept = []
    for _, stratum in frame.dropna(subset=["sex"]).groupby("sex"):
        counts = stratum["label"].value_counts()
        if len(counts) < 2:
            continue
        take = int(counts.min())
        for _, block in stratum.groupby("label"):
            picked = rng.choice(len(block), size=take, replace=False)
            kept.append(block.iloc[picked])
    return pd.concat(kept, ignore_index=True) if kept else frame.head(0)


def evaluate(frame: pd.DataFrame, labels: list[str], name: str, note: str) -> dict:
    interval = patient_bootstrap_ci(frame, macro_f1_of, n_resamples=500)
    return {
        "condition": name, "note": note,
        "n": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "macro_f1": interval["point"],
        "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
        "cataract_auroc": round(cataract_auroc(frame, labels), 4),
        "class_counts": frame["y_true"].value_counts().to_dict(),
    }


def main() -> int:
    labels = load_config("data")["classes"]
    manifests, results = output_dir("manifests"), output_dir("results")

    predictions = pd.read_csv(results / "E1_predictions_patient_disjoint_dedup.csv",
                              encoding="utf-8")
    manifest = pd.read_csv(manifests / "odir_manifest.csv", encoding="utf-8")
    joined = predictions.merge(
        manifest[["image_id", "age", "sex", "multi_finding", "label"]],
        on="image_id", how="left")

    print("=" * 72)
    print("E11 - CONFOUND SENSITIVITY")
    print("=" * 72)

    # --- C1: how far do demographics alone get you? -------------------------------------
    print("\nC1  demographics only -> CLASS  (age and sex, no image)")
    demographics = joined.dropna(subset=["age", "sex"])
    X = pd.get_dummies(demographics[["age", "sex"]], columns=["sex"])
    y = demographics["y_true"].to_numpy()
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                     n_jobs=-1))])
    predicted = cross_val_predict(model, X, y, cv=5)
    baseline = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, y, cv=5)

    demo_accuracy = float((predicted == y).mean())
    demo_baseline = float((baseline == y).mean())
    demo_macro = float(f1_score(y, predicted, average="macro", zero_division=0))
    print(f"  baseline {demo_baseline:.4f}  ->  demographics {demo_accuracy:.4f}   "
          f"(+{demo_accuracy - demo_baseline:.4f})   macro-F1 {demo_macro:.4f}")
    print("  Anything here is available to any model without looking at a retina.")

    # --- C2-C4: does image performance survive the controls? ----------------------------
    conditions = [
        (joined, "unrestricted", "all held-out predictions"),
        (age_matched(joined), "age_matched",
         f"classes balanced within {AGE_BIN}-year age strata"),
        (sex_matched(joined), "sex_matched", "classes balanced within each sex"),
        (joined[~joined["multi_finding"].fillna(False)], "multi_finding_removed",
         "rows whose annotation names a second condition dropped"),
    ]

    print("\n" + "-" * 72)
    print(f"{'condition':<24}{'n':>7}{'patients':>10}{'macro-F1':>11}"
          f"{'95% CI':>20}{'cataract AUROC':>16}")
    print("-" * 72)

    collected = []
    for frame, name, note in conditions:
        if len(frame) < 100:
            print(f"{name:<24} too few rows ({len(frame)}) - skipped")
            continue
        entry = evaluate(frame, labels, name, note)
        collected.append(entry)
        interval = f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"
        print(f"{name:<24}{entry['n']:>7}{entry['patients']:>10}{entry['macro_f1']:>11.4f}"
              f"{interval:>20}{entry['cataract_auroc']:>16.4f}")

    reference = collected[0] if collected else None
    verdict_lines = []
    if reference:
        for entry in collected[1:]:
            delta = entry["macro_f1"] - reference["macro_f1"]
            overlaps = not (entry["ci_high"] < reference["ci_low"]
                            or entry["ci_low"] > reference["ci_high"])
            verdict_lines.append(
                f"  {entry['condition']:<24} macro-F1 {delta:+.4f} vs unrestricted; "
                f"CIs {'overlap' if overlaps else 'DO NOT overlap'}")

    print("\n" + "-" * 72)
    print("DOES IMAGE PERFORMANCE SURVIVE THE CONTROLS?")
    print("-" * 72)
    for line in verdict_lines:
        print(line)
    print("\n  Overlapping intervals mean the control did not materially change the result,")
    print("  i.e. the confound was not driving it. Non-overlap would be the finding.")

    payload = {
        "demographics_only": {
            "baseline_accuracy": round(demo_baseline, 4),
            "accuracy": round(demo_accuracy, 4),
            "lift_over_baseline": round(demo_accuracy - demo_baseline, 4),
            "macro_f1": round(demo_macro, 4),
            "n": int(len(demographics)),
            "note": "age and sex only - no image. Establishes what is obtainable without "
                    "looking at a retina.",
        },
        "conditions": collected,
        "comparisons": verdict_lines,
        "method_note": "matching is by downsampling within strata, so it closes the "
                       "demographic route without reweighting or modelling",
    }
    (results / "E11_confound_sensitivity.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(collected).drop(columns=["class_counts"]).to_csv(
        results / "E11_confound_sensitivity.csv", index=False)

    print(f"\nWRITTEN to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
