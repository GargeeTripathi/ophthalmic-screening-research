"""The patient-disjoint split, copied unchanged from the original project.

It was built with sklearn StratifiedGroupKFold(5, shuffle=True, random_state=42), grouped on
patient_id and stratified on label, after removing 2 duplicate images. It is NOT regenerated
here: changing the model must not change which patients are tested.
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import images_dir, load_config, repo_path


def load_split(require_images: bool = True) -> pd.DataFrame:
    """Split table with an `image_path` column resolved for this machine."""
    frame = pd.read_csv(repo_path(load_config()["data"]["split_file"]), encoding="utf-8")
    root = images_dir()
    if root is None:
        if require_images:
            raise FileNotFoundError(
                "ODIR images not found. Set ODIR_IMAGES (or data.images_dir in configs/config.yaml) "
                "to the folder containing 0_left.jpg, 0_right.jpg, ... - see data/README.md")
        frame["image_path"] = None
    else:
        frame["image_path"] = [str(root / name) for name in frame.original_filename]
    return frame


def leakage_report(frame: pd.DataFrame) -> dict:
    """The checks that must hold before any training: no patient or image in two folds."""
    return {
        "images": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "patients_in_more_than_one_fold": int((frame.groupby("patient_id").fold.nunique() > 1).sum()),
        "duplicate_image_ids": int(frame.image_id.duplicated().sum()),
        "duplicate_filenames": int(frame.original_filename.duplicated().sum()),
    }


def fold_sets(frame: pd.DataFrame, k: int, n_folds: int = 5):
    """test = fold k, validation = fold (k + 1) % n, train = the rest. Patient overlap asserted."""
    val_fold = (k + 1) % n_folds
    test = frame[frame.fold == k]
    val = frame[frame.fold == val_fold]
    train = frame[~frame.fold.isin([k, val_fold])]
    sets = {"train": set(train.patient_id), "validation": set(val.patient_id), "test": set(test.patient_id)}
    overlap = {f"{a}&{b}": len(sets[a] & sets[b])
               for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    assert not any(overlap.values()), f"patient overlap between sets: {overlap}"
    return train, val, test, val_fold
