"""Tests for split construction and evaluation metrics.

The patient-disjointness test is the single most important test in this repository. If it
ever fails, every downstream number is inflated and nothing else in the project is safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    assert_patient_disjoint,
    build_folds,
    build_image_level_folds,
    describe_folds,
    train_test_for_fold,
)
from src.evaluation.metrics import (
    accuracy_of,
    classification_summary,
    expected_calibration_error,
    macro_f1_of,
    mcnemar_exact,
    patient_bootstrap_ci,
)


@pytest.fixture
def toy_manifest() -> pd.DataFrame:
    """80 patients, most contributing two eyes - the structure that causes leakage."""
    rows = []
    for patient in range(80):
        label = ["N", "D", "G", "C"][patient % 4]
        for eye in (["left", "right"] if patient % 5 else ["left"]):
            rows.append({"image_id": f"p{patient}_{eye}", "patient_id": f"p{patient}",
                         "eye": eye, "label": label})
    return pd.DataFrame(rows)


# --- splits -----------------------------------------------------------------------------

def test_patient_disjoint_folds_are_actually_disjoint(toy_manifest):
    assigned = build_folds(toy_manifest, n_folds=5, seed=42)
    assert_patient_disjoint(assigned)                      # raises if violated
    assert (assigned.groupby("patient_id")["fold"].nunique() == 1).all()


def test_every_row_gets_a_fold(toy_manifest):
    assigned = build_folds(toy_manifest, n_folds=5, seed=42)
    assert (assigned["fold"] >= 0).all()
    assert sorted(assigned["fold"].unique()) == [0, 1, 2, 3, 4]


def test_image_level_split_does_leak_patients(toy_manifest):
    # Rung 0 exists to reproduce the leakage published work has. If it stopped leaking, the
    # protocol ladder would have nothing to measure.
    assigned = build_image_level_folds(toy_manifest, n_folds=5, seed=42)
    spanning = (assigned.groupby("patient_id")["fold"].nunique() > 1).sum()
    assert spanning > 0


def test_assert_patient_disjoint_raises_on_a_leaky_split(toy_manifest):
    assigned = build_image_level_folds(toy_manifest, n_folds=5, seed=42)
    with pytest.raises(AssertionError, match="span multiple folds"):
        assert_patient_disjoint(assigned)


def test_folds_are_deterministic(toy_manifest):
    a = build_folds(toy_manifest, n_folds=5, seed=42)
    b = build_folds(toy_manifest, n_folds=5, seed=42)
    assert (a["fold"] == b["fold"]).all()


def test_train_and_test_partition_the_data(toy_manifest):
    assigned = build_folds(toy_manifest, n_folds=5, seed=42)
    train, test = train_test_for_fold(assigned, 0)
    assert len(train) + len(test) == len(assigned)
    assert set(train["patient_id"]).isdisjoint(test["patient_id"])


def test_describe_folds_covers_every_class(toy_manifest):
    description = describe_folds(build_folds(toy_manifest, n_folds=5, seed=42))
    for code in ("N", "D", "G", "C"):
        assert code in description.columns


# --- metrics ------------------------------------------------------------------------------

def test_classification_summary_on_a_perfect_prediction():
    labels = ["N", "D"]
    y = ["N", "N", "D", "D"]
    summary = classification_summary(y, y, labels)
    assert summary["accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary["per_class"]["N"]["support"] == 2


def test_patient_bootstrap_ci_brackets_the_point_estimate():
    frame = pd.DataFrame({
        "patient_id": [f"p{i // 2}" for i in range(60)],
        "y_true": ["N", "D"] * 30,
        "y_pred": ["N", "D"] * 25 + ["D", "N"] * 5,
    })
    interval = patient_bootstrap_ci(frame, macro_f1_of, n_resamples=200)
    assert interval["ci_low"] <= interval["point"] <= interval["ci_high"]
    assert interval["resample_unit"] == "patient_id"


def test_bootstrap_resamples_patients_not_rows():
    # A patient's two eyes must move together. If they did not, the interval would be too
    # narrow and every significance claim in the project would be anti-conservative.
    frame = pd.DataFrame({
        "patient_id": ["p0", "p0", "p1", "p1"],
        "y_true": ["N", "N", "D", "D"],
        "y_pred": ["N", "N", "D", "D"],
    })
    seen = []
    def metric(block):
        seen.append(len(block))
        return accuracy_of(block)
    patient_bootstrap_ci(frame, metric, n_resamples=25)
    # Every resample draws 2 patients x 2 eyes = 4 rows; an odd count means rows were split.
    assert all(size % 2 == 0 for size in seen)


def test_mcnemar_uses_only_discordant_pairs():
    a = np.array([True, True, False, False, True])
    b = np.array([True, False, True, False, True])
    result = mcnemar_exact(a, b)
    assert result["a_only"] == 1 and result["b_only"] == 1
    assert result["discordant"] == 2


def test_mcnemar_reports_no_evidence_when_models_agree():
    same = np.array([True, False, True])
    result = mcnemar_exact(same, same)
    assert result["discordant"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_is_significant_for_a_lopsided_split():
    a = np.array([True] * 20 + [False] * 2)
    b = np.array([False] * 20 + [False] * 2)
    assert mcnemar_exact(a, b)["p_value"] < 0.01


def test_perfect_calibration_scores_near_zero_ece():
    probabilities = np.array([[0.0, 1.0]] * 10 + [[1.0, 0.0]] * 10)
    y = np.array(["1"] * 10 + ["0"] * 10)
    assert expected_calibration_error(probabilities, y, labels=["0", "1"]) < 0.01


def test_confident_and_wrong_scores_high_ece():
    probabilities = np.array([[0.0, 1.0]] * 10)
    y = np.array(["0"] * 10)
    assert expected_calibration_error(probabilities, y, labels=["0", "1"]) > 0.9
