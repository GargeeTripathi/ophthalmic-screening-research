"""The interface's contracts with scripts/build_site.py.

The static site does not re-render the app; it post-processes the captured HTML with two
regexes. Change the markup without changing them and the published site silently keeps a
dead upload form, or loses its whole body. These tests are those two regexes.
"""

from __future__ import annotations

import re

from src.app.render import case_review, worklist

CLASS_DISPLAY = {"N": "No referable abnormality detected", "D": "Diabetic retinopathy",
                 "G": "Glaucoma", "C": "Cataract"}

WRAP = re.compile(r"<div class=wrap>(.*?)</div>\s*<footer", re.S)
FORM = re.compile(r"<form class=upload.*?</form>", re.S)


def make_case(decision: str = "DEFER", confidence: float = 0.54) -> dict:
    return {
        "image_id": "odir_1043_left", "predicted_class": "C", "predicted_name": "cataract",
        "confidence": confidence,
        "probabilities": {"C": 0.54, "N": 0.28, "G": 0.12, "D": 0.06},
        "decision": decision,
        "deferral_reasons": ["the referral diagnosis contradicts the image finding"]
        if decision == "DEFER" else [],
        "document_present": True,
        "document_fields": {"patient_id": "odir_1043", "stated_diagnosis": "normal"},
        "stated_diagnosis_class": "N", "agreement": 0.08, "extraction_confidence": 0.41,
        "concordant": False, "calibrated": True, "image_path": "", "document_path": None,
        "class_display": CLASS_DISPLAY, "calibration_note": "calibrated, T = 0.83",
    }


def test_worklist_carries_the_upload_panel():
    page = worklist([make_case()])
    assert FORM.search(page), "build_site.py strips the panel by this pattern"
    assert 'name="image"' in page or "name=image" in page
    assert "name=document" in page


def test_both_screens_expose_the_body_marker():
    assert WRAP.search(worklist([make_case()]))
    assert WRAP.search(case_review(make_case(), [("Why deferred", "Because.")]))


def test_case_review_has_no_upload_panel():
    assert not FORM.search(case_review(make_case(), []))


def test_empty_queue_renders():
    page = worklist([])
    assert "queue is empty" in page
    assert WRAP.search(page)


def test_deferral_is_stated_before_the_class_name():
    """The design rule: a case the system declined to answer must not read as an answer."""
    page = case_review(make_case(decision="DEFER"), [])
    assert page.index("DEFER") < page.index("Cataract")
