"""Two field extractors, and the features the reliability estimator learns from.

E1 - rule + fuzzy
    Scan the reconstructed text for a known label synonym, allowing OCR character errors
    via fuzzy matching, then take the value that follows it on the same line.

E2 - layout aware
    Work from OCR word boxes instead of reconstructed text. Find the label token, then take
    the nearest token to its RIGHT on the same row, or failing that the nearest BELOW.
    This is the only one of the two that can handle a two-column form or a table, where the
    value is not textually adjacent to its label.

Both are frozen before the sealed layouts are opened. Neither is tuned on D-F.

EXTRACTION FEATURES
-------------------
Every extraction carries the evidence behind it: how confident OCR was over the matched
span, how well the label matched, whether the value has a plausible format, how close a
distractor field was, and where on the page it sat. Those become the input to the learned
correctness estimator, which is the point of the whole reliability study.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field

from rapidfuzz import fuzz, process

from src.idp.generator import LABELS
from src.idp.ocr import Token, as_text

FUZZY_MIN = 78          # below this a label match is not credible even with OCR noise

# A cheap plausibility check per field. Not validation - just "does this look like the
# kind of string this field holds", which turns out to matter more than OCR confidence.
FORMAT_PATTERNS = {
    "age": re.compile(r"^\d{1,3}$"),
    "sex": re.compile(r"^(male|female|m|f)$", re.IGNORECASE),
    "eye": re.compile(r"^(left|right|both|od|os|ou|l|r)$", re.IGNORECASE),
    "exam_date": re.compile(r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}$"),
    "visual_acuity": re.compile(r"^(6\s*/\s*\d{1,3}|20\s*/\s*\d{1,3})$"),
    "iop": re.compile(r"^\d{1,2}(\.\d)?$"),
    "patient_id": re.compile(r"^[A-Za-z0-9_\-]{2,32}$"),
    "stated_diagnosis": re.compile(r"^[A-Za-z][A-Za-z ,\-]{3,60}$"),
    "referral_reason": re.compile(r"^[A-Za-z][A-Za-z ,\-]{3,60}$"),
}

# Label strings that are NOT targets but look like them.
DISTRACTOR_LABELS = ["Insurance ID", "Clinic Address", "Appointment Ref"]

SYNONYM_TO_FIELD = {synonym.lower(): name
                    for name, synonyms in LABELS.items() for synonym in synonyms}


@dataclass
class Extraction:
    field: str
    value: str | None
    matched_label: str | None = None
    features: dict = dataclass_field(default_factory=dict)


def _clean(value: str) -> str:
    return value.strip(" :;-–—\t").strip()


def _format_valid(field_name: str, value: str | None) -> float:
    if value is None:
        return 0.0
    pattern = FORMAT_PATTERNS.get(field_name)
    return 1.0 if pattern and pattern.match(value.strip()) else 0.0


def _label_variants(field_name: str) -> list[str]:
    return LABELS[field_name]


# --- E1: rule + fuzzy over reconstructed text -----------------------------------------

def extract_rule(tokens: list[Token], fields: list[str]) -> dict[str, Extraction]:
    text = as_text(tokens)
    lines = [line for line in text.split("\n") if line.strip()]
    mean_confidence = (sum(t.confidence for t in tokens) / len(tokens)) if tokens else 0.0

    results: dict[str, Extraction] = {}
    for name in fields:
        variants = _label_variants(name)
        best = Extraction(field=name, value=None)
        best_score = 0.0

        for line in lines:
            # Split on the first separator; the label is on the left, value on the right.
            parts = re.split(r"[:\-–]", line, maxsplit=1)
            if len(parts) != 2:
                continue
            candidate_label, candidate_value = _clean(parts[0]), _clean(parts[1])
            if not candidate_value:
                continue

            match = process.extractOne(candidate_label, variants, scorer=fuzz.ratio)
            if match is None or match[1] < FUZZY_MIN or match[1] <= best_score:
                continue

            # Reject if the label matches a distractor better than any target synonym.
            distractor = process.extractOne(candidate_label, DISTRACTOR_LABELS, scorer=fuzz.ratio)
            distractor_score = distractor[1] if distractor else 0.0
            if distractor_score > match[1]:
                continue

            best_score = match[1]
            best = Extraction(
                field=name, value=candidate_value, matched_label=match[0],
                features={
                    "ocr_confidence": round(mean_confidence, 4),
                    "fuzzy_score": round(float(match[1]) / 100, 4),
                    "format_valid": _format_valid(name, candidate_value),
                    "distractor_margin": round((match[1] - distractor_score) / 100, 4),
                    "layout_position": round(lines.index(line) / max(len(lines), 1), 4),
                    "value_length": len(candidate_value),
                },
            )
        results[name] = best
    return results


# --- E2: layout aware, over word boxes -------------------------------------------------

def extract_layout(tokens: list[Token], fields: list[str]) -> dict[str, Extraction]:
    if not tokens:
        return {name: Extraction(field=name, value=None) for name in fields}

    page_bottom = max(t.y1 for t in tokens) or 1.0
    results: dict[str, Extraction] = {}

    for name in fields:
        variants = _label_variants(name)
        best = Extraction(field=name, value=None)
        best_score = 0.0

        for index, token in enumerate(tokens):
            probe = _clean(token.text)
            if not probe:
                continue
            match = process.extractOne(probe, variants, scorer=fuzz.partial_ratio)
            if match is None or match[1] < FUZZY_MIN or match[1] <= best_score:
                continue
            distractor = process.extractOne(probe, DISTRACTOR_LABELS, scorer=fuzz.partial_ratio)
            distractor_score = distractor[1] if distractor else 0.0
            if distractor_score > match[1]:
                continue

            # The value may already sit in the same token ("Age: 62"), otherwise look right
            # on the same row, otherwise directly below.
            value, source_token = None, token
            if ":" in token.text or "-" in token.text:
                tail = _clean(re.split(r"[:\-–]", token.text, maxsplit=1)[-1])
                if tail and not any(tail.lower() == v.lower() for v in variants):
                    value = tail

            if value is None:
                row = [t for t in tokens
                       if abs(t.centre_y - token.centre_y) < 14 and t.x0 > token.x1 - 2]
                if row:
                    source_token = min(row, key=lambda t: t.x0)
                    value = _clean(source_token.text)

            if value is None:
                below = [t for t in tokens
                         if t.y0 > token.y1 and abs(t.x0 - token.x0) < 60]
                if below:
                    source_token = min(below, key=lambda t: t.y0)
                    value = _clean(source_token.text)

            if not value:
                continue

            best_score = match[1]
            best = Extraction(
                field=name, value=value, matched_label=match[0],
                features={
                    "ocr_confidence": round(float(source_token.confidence), 4),
                    "fuzzy_score": round(float(match[1]) / 100, 4),
                    "format_valid": _format_valid(name, value),
                    "distractor_margin": round((match[1] - distractor_score) / 100, 4),
                    "layout_position": round(token.y0 / page_bottom, 4),
                    "value_length": len(value),
                },
            )
        results[name] = best
    return results


EXTRACTORS = {"E1": extract_rule, "E2": extract_layout}

FEATURE_NAMES = ["ocr_confidence", "fuzzy_score", "format_valid",
                 "distractor_margin", "layout_position", "value_length"]


# --- scoring ---------------------------------------------------------------------------

def normalise(field_name: str, value: str | None) -> str | None:
    """Canonical form, so "OD" and "Right" count as the same answer."""
    if value is None:
        return None
    text = value.strip().lower().rstrip(".")
    if field_name == "eye":
        for canonical, forms in (("right", {"right", "od", "r"}),
                                 ("left", {"left", "os", "l"}),
                                 ("both", {"both", "ou"})):
            if text in forms:
                return canonical
    if field_name == "sex":
        if text in {"male", "m"}:
            return "male"
        if text in {"female", "f"}:
            return "female"
    if field_name in {"visual_acuity", "iop"}:
        return re.sub(r"\s+", "", text).replace("mmhg", "")
    return re.sub(r"\s+", " ", text)


def is_correct(field_name: str, predicted: str | None, truth: str | None) -> bool:
    return normalise(field_name, predicted) == normalise(field_name, truth)


def error_category(field_name: str, predicted: str | None, truth: str | None) -> str:
    """Six-category taxonomy - counts per category are more informative than an F1 alone."""
    if truth is None and predicted is None:
        return "correct_absent"
    if truth is None and predicted is not None:
        return "false_positive"
    if truth is not None and predicted is None:
        return "missing_field"
    if is_correct(field_name, predicted, truth):
        return "correct"
    if _format_valid(field_name, predicted) == 0.0:
        return "ocr_corruption"
    if fuzz.ratio(str(predicted).lower(), str(truth).lower()) > 70:
        return "partial_value"
    return "wrong_value"
