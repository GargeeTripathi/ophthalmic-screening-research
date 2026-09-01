"""E13 - seed stability and multiple-comparison correction.

TWO GAPS THIS CLOSES
--------------------
1. SEED STABILITY. Every result so far rests on one split seed (42). A finding that only
   holds for one random partition is not a finding. The capacity ladder is re-run across
   three seeds; if the leakage effect is real it should appear in all three.

2. MULTIPLE COMPARISONS. The project reports roughly a dozen paired tests. At alpha = 0.05
   one spurious "significant" result in twenty is expected by construction. Holm-Bonferroni
   is applied across the whole family, and both raw and corrected p-values are reported so a
   reader can see what the correction did.

Holm is used rather than plain Bonferroni because it is uniformly more powerful at the same
family-wise error rate, which matters when the smallest class has 182 patients.

    python scripts/statistical_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embeddings import load  # noqa: E402
from src.data.splits import assert_patient_disjoint, build_folds, build_image_level_folds  # noqa: E402
from src.evaluation.metrics import macro_f1_of, mcnemar_exact  # noqa: E402
from src.utils.config import load_config, output_dir  # noqa: E402

SEEDS = [42, 7, 2024]

# Only the models that made the point in E1b - the cheapest and the sharpest.
MODELS = {
    "logistic": lambda: Pipeline([("scale", StandardScaler()),
                                  ("clf", LogisticRegression(max_iter=3000, n_jobs=-1))]),
    "knn_1": lambda: KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1),
}


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm step-down correction over a family of tests.

    Sort ascending; test the k-th smallest against alpha / (m - k). Stop at the first
    failure - everything after it stays non-significant regardless of its own p-value.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    corrected, still_rejecting = {}, True
    for k, (name, p) in enumerate(ordered):
        threshold = alpha / (m - k)
        if still_rejecting and p <= threshold:
            significant = True
        else:
            still_rejecting = False
            significant = False
        corrected[name] = {
            "p_raw": round(float(p), 6),
            "holm_threshold": round(threshold, 6),
            "significant_after_correction": significant,
        }
    return corrected


def out_of_fold(split: pd.DataFrame, vectors: np.ndarray, index: dict[str, int],
                build) -> pd.DataFrame:
    rows = split[split["image_id"].isin(index)].reset_index(drop=True)
    X = vectors[[index[i] for i in rows["image_id"]]]
    y = rows["label"].to_numpy()
    predicted = np.empty(len(rows), dtype=object)
    for fold in sorted(rows["fold"].unique()):
        test = (rows["fold"] == fold).to_numpy()
        predicted[test] = build().fit(X[~test], y[~test]).predict(X[test])
    return pd.DataFrame({"image_id": rows["image_id"], "patient_id": rows["patient_id"],
                         "y_true": y, "y_pred": predicted})


def main() -> int:
    config = load_config("data")["splits"]
    manifests, processed, results = (output_dir("manifests"), output_dir("processed"),
                                     output_dir("results"))
    manifest = pd.read_csv(manifests / "odir_manifest_clean.csv", encoding="utf-8")
    ids, vectors = load(processed / "embeddings_odir.npz")
    index = {image_id: position for position, image_id in enumerate(ids)}

    print("=" * 72)
    print("E13 - SEED STABILITY AND MULTIPLE-COMPARISON CORRECTION")
    print("=" * 72)

    # --- seed stability ------------------------------------------------------------------
    print(f"\nRe-running the leakage comparison across seeds {SEEDS}")
    print(f"\n{'seed':<8}{'model':<12}{'image-level':>13}{'patient-disj':>14}"
          f"{'inflation':>11}{'McNemar p':>12}")
    print("-" * 72)

    rows, family = [], {}
    for seed in SEEDS:
        leaky_split = build_image_level_folds(manifest, config["n_folds"], seed)
        clean_split = build_folds(manifest, config["n_folds"], seed)
        assert_patient_disjoint(clean_split)

        for model_name, build in MODELS.items():
            leaky = out_of_fold(leaky_split, vectors, index, build)
            clean = out_of_fold(clean_split, vectors, index, build)
            inflation = macro_f1_of(leaky) - macro_f1_of(clean)

            shared = sorted(set(leaky["image_id"]) & set(clean["image_id"]))
            a = leaky.set_index("image_id").loc[shared]
            b = clean.set_index("image_id").loc[shared]
            test = mcnemar_exact((a["y_true"] == a["y_pred"]).to_numpy(),
                                 (b["y_true"] == b["y_pred"]).to_numpy())

            key = f"leakage[{model_name},seed{seed}]"
            family[key] = test["p_value"]
            rows.append({"seed": seed, "model": model_name,
                         "image_level_macro_f1": round(macro_f1_of(leaky), 4),
                         "patient_disjoint_macro_f1": round(macro_f1_of(clean), 4),
                         "inflation": round(inflation, 4),
                         "p_raw": test["p_value"], "discordant": test["discordant"]})
            print(f"{seed:<8}{model_name:<12}{rows[-1]['image_level_macro_f1']:>13.4f}"
                  f"{rows[-1]['patient_disjoint_macro_f1']:>14.4f}{inflation:>+11.4f}"
                  f"{test['p_value']:>12}")

    frame = pd.DataFrame(rows)
    print("\n" + "-" * 72)
    print("STABILITY ACROSS SEEDS")
    print("-" * 72)
    stability = frame.groupby("model")["inflation"].agg(["mean", "std", "min", "max"]).round(4)
    print(stability.to_string())

    # --- multiple-comparison correction ---------------------------------------------------
    # Fold in the other paired tests the project reports, so the correction covers the whole
    # family rather than only the tests run in this script.
    ladder = results / "E1_protocol_ladder.json"
    if ladder.is_file():
        payload = json.loads(ladder.read_text(encoding="utf-8"))
        for comparison in payload.get("comparisons", []):
            if "mcnemar" in comparison:
                family[f"E1[{comparison['to']}]"] = comparison["mcnemar"]["p_value"]

    capacity = results / "E1b_capacity_ladder.json"
    if capacity.is_file():
        payload = json.loads(capacity.read_text(encoding="utf-8"))
        for comparison in payload.get("comparisons", []):
            family[f"E1b[{comparison['model']}]"] = comparison["mcnemar"]["p_value"]

    corrected = holm_bonferroni(family)
    print("\n" + "-" * 72)
    print(f"HOLM-BONFERRONI OVER {len(family)} PAIRED TESTS  (family-wise alpha = 0.05)")
    print("-" * 72)
    print(f"{'test':<32}{'p raw':>12}{'threshold':>12}{'significant':>13}")
    for name, entry in sorted(corrected.items(), key=lambda kv: kv[1]["p_raw"]):
        mark = "yes" if entry["significant_after_correction"] else "no"
        print(f"{name:<32}{entry['p_raw']:>12}{entry['holm_threshold']:>12}{mark:>13}")

    survivors = [n for n, e in corrected.items() if e["significant_after_correction"]]
    print(f"\n  {len(survivors)} of {len(family)} tests survive correction")

    frame.to_csv(results / "E13_seed_stability.csv", index=False)
    (results / "E13_statistical_validation.json").write_text(json.dumps({
        "seeds": SEEDS,
        "per_run": rows,
        "stability": stability.to_dict(orient="index"),
        "family_size": len(family),
        "holm_bonferroni": corrected,
        "survivors": survivors,
        "note": "Holm rather than plain Bonferroni: uniformly more powerful at the same "
                "family-wise error rate, which matters when the smallest class has 182 "
                "patients",
    }, indent=2, default=str), encoding="utf-8")

    print(f"\nWRITTEN to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
