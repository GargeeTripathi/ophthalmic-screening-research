"""E1b - does protocol inflation depend on the model's capacity to MEMORISE?

WHY THIS EXPERIMENT EXISTS
--------------------------
E1 found no significant inflation between an image-level split and a patient-disjoint one.
That result carries an obvious objection: it used a frozen linear head, which has almost no
capacity to memorise, and leakage inflates a score precisely by letting a model recall a
near-copy it has already seen. A model that cannot memorise cannot benefit from leakage, so
E1 measured a lower bound.

The usual fix is a fine-tuned backbone on a GPU. But the question is about MEMORISATION, not
about compute, so it can be answered directly and on a CPU by varying capacity while holding
features and splits fixed:

    logistic     linear on 2048-d features - minimal memorisation
    mlp          one hidden layer - some
    knn_5        retrieval over 5 neighbours - mostly memorisation
    knn_1        retrieval of the single nearest training image - PURE memorisation

1-NN is the sharpest instrument available. If a patient's fellow eye is in the training set,
1-NN will retrieve it. If inflation exists and is exploitable, it appears here or nowhere.

READING THE RESULT
------------------
    inflation grows with capacity  -> leakage is real and only memorisers exploit it;
                                      E1's null was an artefact of model choice
    inflation flat and near zero   -> patient leakage is genuinely not exploitable on this
                                      dataset, and E1's null stands on much firmer ground

Either outcome is publishable, and neither can be argued away as "you used too weak a model".

    python scripts/capacity_ladder.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embeddings import load  # noqa: E402
from src.evaluation.metrics import macro_f1_of, mcnemar_exact, patient_bootstrap_ci  # noqa: E402
from src.utils.config import load_config, output_dir  # noqa: E402

SEED = 42

# Ordered by how much each can memorise. That ordering is the independent variable.
MODELS = {
    "logistic": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, n_jobs=-1)),
    ]),
    "mlp": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                              random_state=SEED, early_stopping=True)),
    ]),
    "knn_5": lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1),
    "knn_1": lambda: KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1),
}

RUNGS = ["image_level", "patient_disjoint", "patient_disjoint_dedup"]


def out_of_fold(split: pd.DataFrame, vectors: np.ndarray, index: dict[str, int],
                build) -> pd.DataFrame:
    rows = split[split["image_id"].isin(index)].reset_index(drop=True)
    X = vectors[[index[i] for i in rows["image_id"]]]
    y = rows["label"].to_numpy()

    predicted = np.empty(len(rows), dtype=object)
    for fold in sorted(rows["fold"].unique()):
        test = (rows["fold"] == fold).to_numpy()
        model = build().fit(X[~test], y[~test])
        predicted[test] = model.predict(X[test])

    return pd.DataFrame({"image_id": rows["image_id"], "patient_id": rows["patient_id"],
                         "y_true": y, "y_pred": predicted})


def main() -> int:
    load_config("data")
    processed, splits_dir, results = (output_dir("processed"), output_dir("splits"),
                                      output_dir("results"))
    ids, vectors = load(processed / "embeddings_odir.npz")
    index = {image_id: position for position, image_id in enumerate(ids)}

    print("=" * 72)
    print("E1b - CAPACITY LADDER: is protocol inflation a function of memorisation?")
    print("=" * 72)

    records, predictions = [], {}
    for model_name, build in MODELS.items():
        print(f"\n{model_name}")
        for rung in RUNGS:
            path = splits_dir / f"splits_{rung}.csv"
            if not path.is_file():
                continue
            split = pd.read_csv(path, encoding="utf-8")
            predicted = out_of_fold(split, vectors, index, build)
            predictions[(model_name, rung)] = predicted

            interval = patient_bootstrap_ci(predicted, macro_f1_of, n_resamples=500)
            records.append({"model": model_name, "rung": rung,
                            "n": int(len(predicted)),
                            "macro_f1": interval["point"],
                            "ci_low": interval["ci_low"], "ci_high": interval["ci_high"]})
            print(f"  {rung:<26} macro-F1 {interval['point']:.4f}  "
                  f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]")

    frame = pd.DataFrame(records)
    frame.to_csv(results / "E1b_capacity_ladder.csv", index=False)

    # --- the question the experiment was built to answer --------------------------------
    print("\n" + "-" * 72)
    print("INFLATION FROM IGNORING PATIENT IDENTITY, BY MODEL CAPACITY")
    print("-" * 72)
    print(f"{'model':<12}{'image-level':>13}{'patient-disj':>14}{'inflation':>11}{'McNemar p':>12}")

    comparisons = []
    for model_name in MODELS:
        leaky = predictions.get((model_name, "image_level"))
        clean = predictions.get((model_name, "patient_disjoint"))
        if leaky is None or clean is None:
            continue
        inflation = macro_f1_of(leaky) - macro_f1_of(clean)

        shared = sorted(set(leaky["image_id"]) & set(clean["image_id"]))
        a = leaky.set_index("image_id").loc[shared]
        b = clean.set_index("image_id").loc[shared]
        test = mcnemar_exact((a["y_true"] == a["y_pred"]).to_numpy(),
                             (b["y_true"] == b["y_pred"]).to_numpy())

        entry = {"model": model_name,
                 "image_level_macro_f1": round(macro_f1_of(leaky), 4),
                 "patient_disjoint_macro_f1": round(macro_f1_of(clean), 4),
                 "inflation": round(inflation, 4),
                 "paired_n": len(shared), "mcnemar": test}
        comparisons.append(entry)
        print(f"{model_name:<12}{entry['image_level_macro_f1']:>13.4f}"
              f"{entry['patient_disjoint_macro_f1']:>14.4f}{inflation:>+11.4f}"
              f"{test['p_value']:>12}")

    verdict = ("inflation grows with memorisation capacity - E1's null was an artefact of "
               "using a low-capacity head"
               if len(comparisons) >= 2
               and comparisons[-1]["inflation"] > comparisons[0]["inflation"] + 0.02
               else "inflation does not grow with capacity - even a pure memoriser gains "
                    "little from ignoring patient identity, so E1's null stands")

    print("\n" + "=" * 72)
    print("VERDICT:", verdict)

    (results / "E1b_capacity_ladder.json").write_text(json.dumps({
        "rationale": "leakage is exploited by memorisation, so capacity is the right "
                     "variable to sweep - not compute",
        "models_by_capacity": list(MODELS),
        "per_cell": records,
        "comparisons": comparisons,
        "verdict": verdict,
    }, indent=2, default=str), encoding="utf-8")

    print(f"WRITTEN to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
