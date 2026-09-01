"""Deferral scoring and risk-coverage evaluation.

A deferral score is "how much this case should be handed to a clinician". Higher means
defer sooner. Every system in the study emits one, so they are directly comparable on the
same risk-coverage axis.

WHY RISK-COVERAGE RATHER THAN ACCURACY
--------------------------------------
A system that answers 40% of cases at 99% accuracy and one that answers 100% at 88% are not
comparable on a single accuracy number. The risk-coverage curve shows accuracy on the
retained cases at every coverage level, and AURC summarises it. Accuracy at a few fixed
coverages is reported alongside so readers can compare at a matched operating point.
"""

from __future__ import annotations

import numpy as np

# Below this confidence a screening prediction is not worth acting on unreviewed. Chosen
# on the development split, not tuned on the sealed one.
SYSTEM_A_CONFIDENCE = 0.60
# Below this agreement the document is treated as contradicting the image prediction.
SYSTEM_A_AGREEMENT = 0.25


def system_a_scores(max_probability: np.ndarray, agreement: np.ndarray,
                    confidence_threshold: float = SYSTEM_A_CONFIDENCE,
                    agreement_threshold: float = SYSTEM_A_AGREEMENT) -> np.ndarray:
    """Rule-based deferral: low confidence OR document contradiction.

    Returned as a continuous score rather than a boolean so it can be compared on the same
    risk-coverage curve as the learned gate. Cases triggering a rule are ranked above all
    others; within each group, lower confidence defers first.
    """
    triggered = (max_probability < confidence_threshold) | (agreement < agreement_threshold)
    # 1 - confidence keeps the ordering sensible inside each group; +1 lifts every
    # rule-triggered case above every untriggered one.
    return (1.0 - max_probability) + triggered.astype(float)


def risk_coverage_curve(deferral_score: np.ndarray, correct: np.ndarray,
                        n_points: int = 20) -> list[dict]:
    """Accuracy on retained cases as coverage falls from 100% to 5%."""
    order = np.argsort(deferral_score)          # most confident first
    ordered_correct = np.asarray(correct)[order]

    points = []
    for coverage in np.linspace(1.0, 0.05, n_points):
        keep = max(1, int(np.ceil(coverage * len(ordered_correct))))
        retained = ordered_correct[:keep]
        points.append({
            "coverage": round(float(keep / len(ordered_correct)), 4),
            "accuracy": round(float(retained.mean()), 4),
            "risk": round(float(1 - retained.mean()), 4),
            "n_retained": int(keep),
        })
    return points


def aurc(curve: list[dict]) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    coverage = np.array([p["coverage"] for p in curve])
    risk = np.array([p["risk"] for p in curve])
    order = np.argsort(coverage)
    return float(np.trapezoid(risk[order], coverage[order]))
