"""Patient-disjoint cross-validation splits.

The whole project rests on this file being right, so it is deliberately small and every
invariant is asserted rather than assumed.

WHY GROUPING IS NOT ENOUGH ON ITS OWN
-------------------------------------
2,148 of the 2,910 ODIR patients contribute both eyes. Two eyes photographed in one session
on one camera are highly correlated, so a random image-level split puts a near-copy of a
training image into the test set and the score measures recall, not generalisation.

WHY STRATIFICATION IS AWKWARD HERE
----------------------------------
354 patients have different diagnoses in their two eyes, so a patient does not have "a
class". StratifiedGroupKFold handles exactly this: it groups by patient while keeping the
per-fold class distribution as close as it can at the row level. It cannot be perfect, so
`describe_folds` reports the residual imbalance instead of hiding it.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build_folds(manifest: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return the manifest with an added integer `fold` column."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    assigned = manifest.copy().reset_index(drop=True)
    assigned["fold"] = -1
    for fold, (_, test_index) in enumerate(
        splitter.split(assigned, y=assigned["label"], groups=assigned["patient_id"])
    ):
        assigned.loc[test_index, "fold"] = fold
    if (assigned["fold"] < 0).any():
        raise RuntimeError("some rows were not assigned to a fold")
    return assigned


def build_image_level_folds(manifest: pd.DataFrame, n_folds: int = 5,
                            seed: int = 42) -> pd.DataFrame:
    """Deliberately WRONG splits - stratified by class, ignoring patient identity.

    This exists only as rung 0 of the protocol ladder. It reproduces the split most
    published work on this task uses, so the inflation it causes can be measured rather
    than asserted. It must never be used for a reported model result.
    """
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    assigned = manifest.copy().reset_index(drop=True)
    assigned["fold"] = -1
    for fold, (_, test_index) in enumerate(splitter.split(assigned, y=assigned["label"])):
        assigned.loc[test_index, "fold"] = fold
    return assigned


def assert_patient_disjoint(assigned: pd.DataFrame) -> None:
    """Fail loudly if any patient appears in more than one fold."""
    per_patient = assigned.groupby("patient_id")["fold"].nunique()
    offenders = per_patient[per_patient > 1]
    if len(offenders):
        raise AssertionError(
            f"{len(offenders)} patient(s) span multiple folds, e.g. "
            f"{list(offenders.index[:5])}"
        )


def describe_folds(assigned: pd.DataFrame) -> pd.DataFrame:
    """Per-fold size, patient count and class distribution."""
    rows = []
    for fold, block in assigned.groupby("fold"):
        row = {
            "fold": fold,
            "images": len(block),
            "patients": block["patient_id"].nunique(),
        }
        counts = block["label"].value_counts()
        for code in sorted(assigned["label"].unique()):
            row[code] = int(counts.get(code, 0))
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")


def train_test_for_fold(assigned: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard cross-validation view: one fold held out, the rest for training."""
    test = assigned[assigned["fold"] == fold]
    train = assigned[assigned["fold"] != fold]
    return train, test
