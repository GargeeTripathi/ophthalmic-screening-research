"""OCR wrapper returning text, word boxes and per-token confidence.

Engine choice: RapidOCR (ONNX Runtime). The original plan was Tesseract, but that needs a
system binary installed outside Python, which makes the project harder to reproduce and
impossible to run in a clean environment without admin rights. RapidOCR is pip-installable,
runs on CPU, works offline, and returns exactly what the reliability study needs - a box, a
string and a confidence per detected region.

The confidence values matter here more than the text. The reliability study asks whether
OCR confidence is a usable estimate of whether an extracted CLINICAL FIELD is correct, and
the answer is not obvious: a region can be transcribed perfectly and still be the wrong
region, which no OCR confidence can tell you.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ENGINE: Any = None


@dataclass
class Token:
    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def centre_y(self) -> float:
        return (self.y0 + self.y1) / 2

    def to_dict(self) -> dict:
        return {"text": self.text, "confidence": round(self.confidence, 4),
                "x0": round(self.x0, 1), "y0": round(self.y0, 1),
                "x1": round(self.x1, 1), "y1": round(self.y1, 1)}


def engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def read(image_path: str | Path) -> list[Token]:
    """Return detected tokens, ordered top-to-bottom then left-to-right."""
    result, _ = engine()(str(image_path))
    if not result:
        return []

    tokens = []
    for box, text, score in result:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        tokens.append(Token(str(text), float(score), min(xs), min(ys), max(xs), max(ys)))

    tokens.sort(key=lambda t: (round(t.centre_y / 12), t.x0))
    return tokens


def as_text(tokens: list[Token]) -> str:
    """Reconstruct reading-order text, one line per row band."""
    if not tokens:
        return ""
    lines, current, last_row = [], [], None
    for token in tokens:
        row = round(token.centre_y / 12)
        if last_row is not None and row != last_row:
            lines.append(" ".join(current))
            current = []
        current.append(token.text)
        last_row = row
    lines.append(" ".join(current))
    return "\n".join(lines)


def save(path: Path, records: dict[str, list[Token]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {doc_id: [t.to_dict() for t in tokens] for doc_id, tokens in records.items()}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load(path: Path) -> dict[str, list[Token]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {doc_id: [Token(**t) for t in tokens] for doc_id, tokens in payload.items()}
