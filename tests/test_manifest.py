"""Tests for the manifest builder.

The pure-function tests run anywhere. The manifest-file tests skip if the manifest has
not been built yet, so a fresh clone still passes `pytest` before any data exists.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.manifest import (
    has_multiple_findings,
    keyword_for_eye,
    parse_filename,
    single_label,
)
from src.utils.config import load_config, output_dir

MANIFEST_PATH = output_dir("manifests", create=False) / "odir_manifest.csv"


# --- pure functions -------------------------------------------------------------------

@pytest.mark.parametrize("filename,patient,eye", [
    ("1234_left.jpg", "1234", "left"),
    ("0_right.jpg", "0", "right"),
    ("10003_LEFT.JPEG", "10003", "left"),
    ("notanimage.txt", None, None),
    ("_26_2782553.jpg", None, None),      # merged-dataset style, must not parse as ODIR
    ("1234_middle.jpg", None, None),
])
def test_parse_filename(filename, patient, eye):
    assert parse_filename(filename) == (patient, eye)


@pytest.mark.parametrize("raw,expected", [
    ("['N']", "N"),
    ("['C']", "C"),
    ("['N', 'D']", None),                 # multi-label rows are not single-label
    ("[]", None),
    ("garbage", None),
])
def test_single_label(raw, expected):
    assert single_label(raw) == expected


def test_keyword_is_read_from_the_matching_eye():
    # Getting this backwards silently mislabels half the dataset, so it is worth a test.
    row = pd.Series({
        "Left-Diagnostic Keywords": "cataract",
        "Right-Diagnostic Keywords": "normal fundus",
    })
    assert keyword_for_eye(row, "left") == "cataract"
    assert keyword_for_eye(row, "right") == "normal fundus"


@pytest.mark.parametrize("keyword,expected", [
    ("glaucoma", False),
    ("normal fundus", False),
    ("glaucoma，diabetic retinopathy", True),   # full-width comma, as in the source data
    ("cataract, mild nonproliferative retinopathy", True),
])
def test_multi_finding_detection(keyword, expected):
    assert has_multiple_findings(keyword) is expected


# --- the built manifest ---------------------------------------------------------------

needs_manifest = pytest.mark.skipif(
    not MANIFEST_PATH.is_file(), reason="manifest not built yet"
)


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH, encoding="utf-8")


@needs_manifest
def test_manifest_has_the_declared_columns(manifest):
    assert list(manifest.columns) == load_config("data")["manifest_columns"]


@needs_manifest
def test_image_ids_are_unique(manifest):
    assert manifest["image_id"].is_unique


@needs_manifest
def test_labels_are_within_the_declared_class_set(manifest):
    assert set(manifest["label"]) <= set(load_config("data")["classes"])


@needs_manifest
def test_no_patient_contributes_more_than_two_eyes(manifest):
    # One patient, at most one left and one right. More than two rows means the
    # filename parsing collapsed distinct patients onto one id.
    assert manifest.groupby("patient_id").size().max() <= 2


@needs_manifest
def test_patient_id_and_eye_are_consistent_with_the_image_id(manifest):
    rebuilt = manifest["patient_id"] + "_" + manifest["eye"]
    assert (rebuilt == manifest["image_id"]).all()


@needs_manifest
def test_no_rows_are_missing_critical_fields(manifest):
    for column in ("image_path", "patient_id", "eye", "label", "age", "sex"):
        assert manifest[column].notna().all(), f"{column} has nulls"


@needs_manifest
def test_every_image_was_readable(manifest):
    assert manifest["width"].notna().all()
    assert (manifest["width"] > 0).all()
