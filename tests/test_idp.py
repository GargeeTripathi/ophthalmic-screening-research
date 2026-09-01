"""Tests for document generation, extraction and deferral scoring.

The value-normalisation and error-taxonomy tests matter most: if "OD" and "Right" are not
treated as the same answer, extraction F1 is wrong in a way that looks plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.deferral.systems import aurc, risk_coverage_curve, system_a_scores
from src.idp.extract import error_category, is_correct, normalise
from src.idp.ocr import Token, as_text
from src.utils.config import load_config


# --- value normalisation ---------------------------------------------------------------

@pytest.mark.parametrize("field,a,b", [
    ("eye", "OD", "Right"),
    ("eye", "os", "LEFT"),
    ("eye", "OU", "Both"),
    ("sex", "M", "male"),
    ("sex", "f", "Female"),
    ("iop", "16 mmHg", "16"),
    ("visual_acuity", "6 / 12", "6/12"),
    ("stated_diagnosis", "Cataract.", "cataract"),
])
def test_equivalent_values_normalise_together(field, a, b):
    assert is_correct(field, a, b)


@pytest.mark.parametrize("field,a,b", [
    ("eye", "OD", "Left"),
    ("sex", "M", "Female"),
    ("iop", "16", "18"),
    ("visual_acuity", "6/12", "6/18"),
])
def test_different_values_stay_different(field, a, b):
    assert not is_correct(field, a, b)


def test_normalise_passes_none_through():
    assert normalise("eye", None) is None


# --- error taxonomy ---------------------------------------------------------------------

def test_absent_field_correctly_left_absent():
    assert error_category("iop", None, None) == "correct_absent"


def test_extracting_an_absent_field_is_a_false_positive():
    # This is what the distractor fields exist to provoke.
    assert error_category("iop", "16", None) == "false_positive"


def test_missing_a_present_field():
    assert error_category("iop", None, "16") == "missing_field"


def test_correct_extraction():
    assert error_category("iop", "16", "16") == "correct"


def test_malformed_value_is_ocr_corruption():
    # "l6" is not a plausible IOP, so the failure is transcription, not selection.
    assert error_category("iop", "l6", "16") == "ocr_corruption"


def test_plausible_but_wrong_value():
    assert error_category("iop", "18", "16") == "wrong_value"


# --- OCR text reconstruction -------------------------------------------------------------

def test_tokens_group_into_lines_by_row():
    tokens = [
        Token("Age:", 0.9, 0, 0, 40, 20),
        Token("62", 0.9, 50, 0, 70, 20),
        Token("Sex:", 0.9, 0, 40, 40, 60),
        Token("M", 0.9, 50, 40, 60, 60),
    ]
    assert as_text(tokens).splitlines() == ["Age: 62", "Sex: M"]


def test_empty_token_list_gives_empty_text():
    assert as_text([]) == ""


# --- deferral scoring ---------------------------------------------------------------------

def test_system_a_defers_low_confidence_and_contradictions():
    confidence = np.array([0.95, 0.40, 0.95])
    agreement = np.array([0.90, 0.90, 0.05])   # third case: document contradicts the image
    scores = system_a_scores(confidence, agreement)
    assert scores[0] < scores[1]               # low confidence defers before high
    assert scores[0] < scores[2]               # contradiction defers before agreement


def test_risk_coverage_starts_at_full_coverage():
    curve = risk_coverage_curve(np.array([0.1, 0.2, 0.3, 0.4]),
                                np.array([True, True, False, False]))
    assert curve[0]["coverage"] == 1.0
    assert curve[0]["accuracy"] == 0.5


def test_a_perfect_ranker_beats_a_useless_one_on_aurc():
    correct = np.array([True] * 8 + [False] * 2)
    perfect = np.arange(10)                    # wrong cases ranked last -> deferred first
    useless = np.zeros(10)
    assert aurc(risk_coverage_curve(perfect, correct)) <= aurc(risk_coverage_curve(useless, correct))


# --- corpus configuration ------------------------------------------------------------------

def test_sealed_layouts_are_not_used_for_development():
    layouts = load_config("idp")["layouts"]
    assert set(layouts["dev"]).isdisjoint(layouts["sealed"])


def test_every_field_has_at_least_one_label_synonym():
    from src.idp.generator import LABELS
    for name in load_config("idp")["fields"]:
        assert name in LABELS and LABELS[name], f"{name} has no label synonyms"
