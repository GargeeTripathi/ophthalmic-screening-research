"""Mapping a referral's free-text diagnosis onto a class.

The two failures this guards against were both found in real output, not imagined:
"normal" is a substring of "abnormality", and RapidOCR drops inter-word spaces on some
layouts (returning "Noabnormalitydetected" at 0.98 confidence).
"""

from __future__ import annotations

import pytest

from src.app.pipeline import ScreeningPipeline

classify = ScreeningPipeline.classify_stated_diagnosis


@pytest.mark.parametrize("text,expected", [
    ("No abnormality detected", "N"),
    ("Within normal limits", "N"),
    ("Nuclear sclerotic cataract", "C"),
    ("Non-proliferative diabetic retinopathy", "D"),
    ("NPDR, moderate", "D"),
    ("Glaucoma suspect", "G"),
    ("", None),
    (None, None),
    ("blurred vision", None),
])
def test_maps_stated_diagnosis(text, expected):
    assert classify(None, text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Noabnormalitydetected", "N"),
    ("Withinnormallimits", "N"),
    ("Diabeticretinopathy", "D"),
])
def test_survives_ocr_dropping_spaces(text, expected):
    assert classify(None, text) == expected


@pytest.mark.parametrize("text", [
    "Abnormality detected on the disc",
    "abnormality present, urgent review",
    "Abnormalitydetected",
])
def test_an_asserted_abnormality_is_never_normal(text):
    """The false negative that matters: 'normal' hides inside 'abnormality'."""
    assert classify(None, text) != "N"
