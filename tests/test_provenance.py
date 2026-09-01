"""Tests for provenance feature extraction and source assignment.

The most important test in this file is `test_probe_features_exclude_label_and_source`.
The entire provenance audit rests on source being derived without reference to the class;
if that ever stops being true, the headline result becomes circular and meaningless.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.provenance.features import (
    PROBE_FEATURES,
    assign_source,
    filename_convention,
    resolution_bucket,
)
from src.utils.config import output_dir

MANIFEST_PATH = output_dir("manifests", create=False) / "merged_manifest.csv"


def features(extension, convention, bucket):
    return {
        "file_extension": extension,
        "filename_convention": convention,
        "resolution_bucket": bucket,
    }


# --- the anti-circularity guard -------------------------------------------------------

def test_probe_features_exclude_label_and_source():
    # The E0a probe must work from raw acquisition properties. If `source` (our own rule
    # output) or anything label-derived leaked in, the probe would be predicting its own
    # input and the audit would prove nothing.
    forbidden = {"source", "label", "class_folder", "image_id", "image_path"}
    assert forbidden.isdisjoint(PROBE_FEATURES)


def test_assign_source_ignores_any_label_key():
    # Passing a label must not change the answer. This pins the rule's signature-level
    # independence from the class, not just its current implementation.
    base = features(".jpeg", "patient_eye", "small_512")
    assert assign_source(base) == assign_source({**base, "label": "N", "class_folder": "normal"})


# --- filename conventions -------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("1234_left.jpg", "patient_eye"),
    ("10003_right.jpeg", "patient_eye"),
    ("_26_2782553.jpg", "underscore_index_hash"),
    ("_0_4015166.jpg", "underscore_index_hash"),
    ("random_name.jpg", "other"),
    ("IMG20240101.png", "other"),
])
def test_filename_convention(filename, expected):
    assert filename_convention(filename) == expected


# --- resolution buckets ---------------------------------------------------------------

@pytest.mark.parametrize("width,height,expected", [
    (256, 256, "tiny_256"),
    (300, 300, "tiny_256"),
    (512, 512, "small_512"),
    (600, 400, "small_512"),
    (1024, 768, "medium"),
    (2592, 1728, "large"),
    (None, None, "unknown"),
])
def test_resolution_bucket(width, height, expected):
    assert resolution_bucket(width, height) == expected


# --- source rules ---------------------------------------------------------------------

@pytest.mark.parametrize("extension,convention,bucket,expected", [
    (".jpeg", "patient_eye", "small_512", "eyepacs_jpeg"),
    (".png", "underscore_index_hash", "large", "png_highres"),
    (".jpg", "patient_eye", "small_512", "odir_jpg512"),
    (".jpg", "underscore_index_hash", "tiny_256", "crop_jpg256"),
    (".jpg", "patient_eye", "tiny_256", "crop_jpg256"),
    (".jpg", "other", "large", "native_jpg"),
    (".bmp", "other", "unknown", "unassigned"),
])
def test_assign_source(extension, convention, bucket, expected):
    assert assign_source(features(extension, convention, bucket)) == expected


def test_extension_dominates_for_jpeg():
    # Every .jpeg in this collection is an EyePACS export, whatever else is true of it.
    for bucket in ("tiny_256", "small_512", "medium", "large"):
        assert assign_source(features(".jpeg", "other", bucket)) == "eyepacs_jpeg"


# --- the built manifest ---------------------------------------------------------------

needs_manifest = pytest.mark.skipif(
    not MANIFEST_PATH.is_file(), reason="merged manifest not built yet"
)


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH, encoding="utf-8")


@needs_manifest
def test_every_image_has_a_source(manifest):
    assert (manifest["source"] != "unassigned").all()


@needs_manifest
def test_image_ids_are_unique(manifest):
    assert manifest["image_id"].is_unique


@needs_manifest
def test_source_is_a_pure_function_of_acquisition_features(manifest):
    # Two images with identical acquisition properties must receive the same source.
    # A violation would mean something outside PROBE_FEATURES influenced the assignment.
    key = ["file_extension", "filename_convention", "resolution_bucket"]
    assert (manifest.groupby(key)["source"].nunique() == 1).all()


@needs_manifest
def test_no_source_spans_more_classes_than_expected(manifest):
    # Sanity check on the finding itself: at least one source must be perfectly
    # class-predictive, otherwise the audit's premise does not hold for this dataset.
    per_source_classes = manifest.groupby("source")["label"].nunique()
    assert (per_source_classes == 1).any(), "no source is perfectly class-predictive"
