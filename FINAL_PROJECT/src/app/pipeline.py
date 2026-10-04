"""End-to-end screening pipeline: one fundus image, one optional referral, one decision.

The image model is the final ConvNeXt-Tiny checkpoint named in configs/config.yaml, loaded
through src/inference/predict.py - the same code path scripts/evaluate.py measured.

THE STRUCTURAL RULE, ENFORCED IN CODE
-------------------------------------
`predict_image()` takes only the image. The document is not a parameter and cannot reach it.
`screen()` then decides whether to ACCEPT that prediction or DEFER it, and only that second
step sees the referral.

    from src.app.pipeline import ScreeningPipeline
    result = ScreeningPipeline().screen("fundus.jpg", "referral.jpg")
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.inference.predict import Predictor
from src.utils.config import load_config, repo_path


@dataclass
class ScreeningResult:
    image_id: str
    predicted_class: str
    predicted_name: str
    confidence: float
    probabilities: dict
    decision: str                       # ACCEPT or DEFER
    deferral_reasons: list = field(default_factory=list)
    document_present: bool = False
    document_fields: dict = field(default_factory=dict)
    stated_diagnosis_class: str | None = None
    agreement: float | None = None
    extraction_confidence: float | None = None
    concordant: bool | None = None
    calibrated: bool = False
    disclaimer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ScreeningPipeline:
    """Loads once, screens many. Everything is CPU-only."""

    DISCLAIMER = ("Academic screening prototype. Not a medical device and not a diagnosis. "
                  "A qualified clinician reviews every case and makes every decision.")

    def __init__(self, checkpoint: str | Path | None = None):
        settings = load_config()
        self.predictor = Predictor(checkpoint)
        self.labels = self.predictor.labels
        self.class_names = self.predictor.class_names
        self.temperature = self.predictor.temperature
        self.weights_source = ("torchvision ImageNet stem + fine-tuned stage 4 and head from "
                               f"{self.predictor.checkpoint.name}")
        meta = self.predictor.meta
        # Same shape the templates and the server already read.
        self.config = {
            "labels": self.labels,
            "class_names": self.class_names,
            "image_model": {
                "architecture": f"ImageNet {settings['model']['display_name']}, stage 4 + head fine-tuned",
                "calibrated": True,
                "temperature": self.temperature,
                "calibration_note": (
                    f"temperature scaling, T={self.temperature:.3f}, fitted on {meta['temperature_fitted_on']}, "
                    f"patient-disjoint from training and from the held-out test fold {meta['test_fold']}. "
                    "Confidence is comparable across cases but is not a probability of the patient "
                    "having the disease."),
            },
            "deferral": {"system": "A - rules", **settings["deferral"]},
        }

        self._reliability = None
        reliability_path = repo_path("checkpoints", "reliability.joblib")
        if reliability_path.is_file():
            import joblib
            self._reliability = joblib.load(reliability_path)

        self._idp_fields = load_config("idp")["fields"]

    # ------------------------------------------------------------------ image pathway
    def predict_image(self, image_path: str | Path) -> tuple[str, float, dict]:
        """Class, confidence and full distribution - from the IMAGE ALONE.

        The referral document is deliberately not a parameter here. That is the safety
        guarantee, expressed as a signature rather than as a comment.
        """
        out = self.predictor.predict(image_path)
        return out["predicted_class"], out["confidence"], out["probabilities"]

    # --------------------------------------------------------------- document pathway
    def read_document(self, document_path: str | Path) -> tuple[dict, float | None]:
        """Extracted fields plus an estimate of how reliably they were read."""
        from src.idp.extract import extract_layout
        from src.idp.ocr import read

        tokens = read(document_path)
        extracted = extract_layout(tokens, self._idp_fields)
        fields = {name: result.value for name, result in extracted.items()
                  if result.value is not None}

        confidence = None
        diagnosis = extracted.get("stated_diagnosis")
        if self._reliability is not None and diagnosis is not None and diagnosis.value:
            names = self._reliability["features"]
            # Named columns, because the estimator was fitted from a DataFrame — a bare list
            # scores identically but warns on every document read.
            features = pd.DataFrame([[diagnosis.features.get(f, 0.0) for f in names]],
                                    columns=names)
            confidence = float(self._reliability["model"].predict_proba(features)[0, 1])
        return fields, confidence

    def classify_stated_diagnosis(self, text: str | None) -> str | None:
        """Map free-text diagnosis onto one of the four classes, or None if unrecognised.

        Two traps this has to survive, both seen in real OCR output:

        1. "normal" is a substring of "abNORMALity", so a referral reading "abnormality
           present" used to classify as N. That is a false negative in the one direction a
           screening tool must never fail in, so N is only assigned via an explicit negative
           phrase, never via a bare "normal" that sits inside "abnormality".
        2. RapidOCR drops inter-word spaces on some layouts and returns
           "Noabnormalitydetected" at 0.98 confidence, so matching runs over a
           space-stripped probe rather than the raw text.
        """
        if not text:
            return None
        probe = re.sub(r"[^a-z]", "", text.lower())
        for code, keywords in (("C", ("cataract", "sclerosis", "sclerotic")),
                               ("D", ("diabetic", "retinopathy", "npdr")),
                               ("G", ("glaucoma",)),
                               ("N", ("noabnormality", "nabnormality", "withinnormal",
                                      "unremarkable", "noreferable"))):
            if any(word in probe for word in keywords):
                return code
        # A bare "normal" only counts once no abnormality has been asserted.
        if "normal" in probe and "abnormal" not in probe:
            return "N"
        return None

    # ------------------------------------------------------------------------ decision
    def screen(self, image_path: str | Path,
               document_path: str | Path | None = None) -> ScreeningResult:
        predicted, confidence, probabilities = self.predict_image(image_path)

        result = ScreeningResult(
            image_id=Path(image_path).stem,
            predicted_class=predicted,
            predicted_name=self.class_names[predicted],
            confidence=round(confidence, 4),
            probabilities=probabilities,
            decision="ACCEPT",
            calibrated=self.config["image_model"]["calibrated"],
            disclaimer=self.DISCLAIMER,
        )

        if document_path is not None:
            fields, extraction_confidence = self.read_document(document_path)
            stated = self.classify_stated_diagnosis(fields.get("stated_diagnosis"))
            result.document_present = True
            result.document_fields = fields
            result.stated_diagnosis_class = stated
            result.extraction_confidence = (round(extraction_confidence, 4)
                                            if extraction_confidence is not None else None)
            if stated is not None:
                result.agreement = probabilities[stated]
                result.concordant = stated == predicted

        # System A: the two rules that beat the learned gate.
        rules = self.config["deferral"]
        reasons = []
        if confidence < rules["confidence_threshold"]:
            reasons.append(f"image confidence {confidence:.0%} is below the "
                           f"{rules['confidence_threshold']:.0%} threshold")
        if result.agreement is not None and result.agreement < rules["agreement_threshold"]:
            reasons.append("the referral diagnosis contradicts the image finding")
        if (result.extraction_confidence is not None
                and result.extraction_confidence < rules["reliability_threshold"]):
            reasons.append("the referral fields were read with low reliability")

        result.deferral_reasons = reasons
        result.decision = "DEFER" if reasons else "ACCEPT"
        return result
