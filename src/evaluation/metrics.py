"""Metrics and resampling used across every experiment.

Two things here are not the textbook default, and both matter for this dataset:

PATIENT-LEVEL BOOTSTRAP
    2,148 of 2,910 patients contribute two eyes. Resampling images treats those two rows as
    independent observations, which they are not, and the resulting interval is too narrow.
    Resampling patients - taking all of a patient's rows together - is the honest version.

POOLED OUT-OF-FOLD PREDICTIONS
    A single test fold holds ~1,000 images, which needs a 2-3 point accuracy gap to reach
    80% power on a paired test. Pooling every fold's held-out predictions gives ~5,000 and
    roughly halves the detectable difference. Cross-validated paired testing is standard;
    the alternative here is an underpowered study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

BOOTSTRAP_RESAMPLES = 2000
SEED = 42


def classification_summary(y_true, y_pred, labels: list[str],
                           y_score: np.ndarray | None = None) -> dict:
    """Headline metrics plus per-class detail."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    summary = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": {
            code: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                   "f1": round(float(f), 4), "support": int(s)}
            for code, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

    if y_score is not None:
        try:
            summary["macro_auroc"] = round(
                float(roc_auc_score(y_true, y_score, multi_class="ovr",
                                    average="macro", labels=labels)), 4)
        except ValueError:
            summary["macro_auroc"] = None
    return summary


def patient_bootstrap_ci(frame: pd.DataFrame, metric, n_resamples: int = BOOTSTRAP_RESAMPLES,
                         alpha: float = 0.05, seed: int = SEED) -> dict:
    """Percentile bootstrap CI, resampling PATIENTS rather than images.

    `frame` needs a `patient_id` column. `metric` takes a DataFrame and returns a float.
    """
    rng = np.random.default_rng(seed)
    patients = frame["patient_id"].to_numpy()
    unique = np.unique(patients)
    by_patient = {p: np.where(patients == p)[0] for p in unique}

    point = float(metric(frame))
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_patient[p] for p in picked])
        try:
            draws[i] = metric(frame.iloc[rows])
        except Exception:  # noqa: BLE001 - a degenerate resample should not abort the CI
            draws[i] = np.nan

    draws = draws[~np.isnan(draws)]
    return {
        "point": round(point, 4),
        "ci_low": round(float(np.percentile(draws, 100 * alpha / 2)), 4),
        "ci_high": round(float(np.percentile(draws, 100 * (1 - alpha / 2))), 4),
        "n_resamples": int(len(draws)),
        "resample_unit": "patient_id",
    }


def macro_f1_of(frame: pd.DataFrame) -> float:
    return float(f1_score(frame["y_true"], frame["y_pred"], average="macro", zero_division=0))


def accuracy_of(frame: pd.DataFrame) -> float:
    return float(accuracy_score(frame["y_true"], frame["y_pred"]))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired per-item correctness.

    Only the discordant pairs carry information - items both models get right, or both
    get wrong, say nothing about which is better.
    """
    from scipy.stats import binomtest

    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only

    if discordant == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "p_value": 1.0,
                "note": "models made identical errors - no evidence either way"}

    result = binomtest(a_only, discordant, 0.5, alternative="two-sided")
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": round(float(result.pvalue), 6),
    }


def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray,
                               labels: list[str], n_bins: int = 15) -> float:
    """ECE over the predicted-class confidence."""
    index = {code: i for i, code in enumerate(labels)}
    truth = np.array([index[c] for c in y_true])
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == truth).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.sum() == 0:
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return round(float(error), 4)
