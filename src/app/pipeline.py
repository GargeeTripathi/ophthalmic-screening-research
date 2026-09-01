"""End-to-end screening pipeline: one fundus image, one optional referral, one decision.

This is the deployed system. It loads the artefacts `scripts/export_model.py` froze to disk,
so what runs here is exactly what was evaluated — inference cannot silently drift from the
numbers in the report.

THE STRUCTURAL RULE, ENFORCED IN CODE
-------------------------------------
`predict_image()` takes only the image. The document is not a parameter and cannot reach it.
`screen()` then decides whether to ACCEPT that prediction or DEFER it, and only that second
step sees the referral. A reader can verify the guarantee by reading two function signatures
rather than by trusting a comment.

    from src.app.pipeline import ScreeningPipeline
    result = ScreeningPipeline().screen("fundus.jpg", "referral.jpg")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

from src.utils.config import load_config, output_dir

Image.MAX_IMAGE_PIXELS = None
INPUT_SIZE = 224
FOV_THRESHOLD = 0.10

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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

    def __init__(self, checkpoint_dir: Path | None = None):
        self.checkpoints = checkpoint_dir or output_dir("checkpoints")
        config_path = self.checkpoints / "deployment.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"{config_path} not found - run scripts/export_model.py first")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.labels = self.config["labels"]
        self.class_names = self.config["class_names"]

        # Two ways to load, and both must produce identical predictions.
        #
        #   full     checkpoints/screening_model.pt  - 94 MB, backbone weights baked in.
        #            Not committed; used locally where it already exists.
        #   portable checkpoints/screening_head.pt   - 33 KB, head + temperature only.
        #            Committed. The backbone comes from torchvision, which is where the
        #            94 MB came from in the first place.
        #
        # A deployment that only has the repository takes the portable path.
        full = self.checkpoints / self.config["image_model"]["path"]
        portable = self.checkpoints / self.config["image_model"].get(
            "portable_path", "screening_head.pt")

        backbone = resnet50(weights=None)
        backbone.fc = nn.Identity()

        if full.is_file():
            bundle = torch.load(full, map_location="cpu", weights_only=False)
            backbone.load_state_dict(bundle["backbone_state_dict"])
            self.weights_source = "local checkpoint"
        elif portable.is_file():
            from torchvision.models import ResNet50_Weights
            pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            pretrained.fc = nn.Identity()
            backbone.load_state_dict(pretrained.state_dict())
            bundle = torch.load(portable, map_location="cpu", weights_only=False)
            self.weights_source = "torchvision ImageNet weights + committed head"
        else:
            raise FileNotFoundError(
                f"no model found - expected {full.name} or {portable.name} in "
                f"{self.checkpoints}. Run scripts/export_model.py.")

        backbone.eval()
        self.backbone = backbone
        self.head_weight = bundle["head_weight"]
        self.head_bias = bundle["head_bias"]
        # Temperature scaling, fitted on a held-out patient-disjoint fold at export time.
        self.temperature = float(bundle.get("temperature", 1.0))

        self._reliability = None
        reliability_path = self.checkpoints / "reliability.joblib"
        if reliability_path.is_file():
            import joblib
            self._reliability = joblib.load(reliability_path)

        self._idp_fields = load_config("idp")["fields"]

    # ------------------------------------------------------------------ image pathway
    # PREPROCESSING MUST MATCH src/data/embeddings.py EXACTLY - two steps, both of which
    # were wrong in a first version and produced the same silent symptom.
    #
    #   1. Resize plainly to 224x224. Cropping to the retinal field of view here (as an
    #      earlier version did) frames the image differently from the cached embeddings.
    #   2. L2-NORMALISE the pooled feature before the head. This was the dominant bug: the
    #      cached embeddings have norm 1.0, live features have norm ~12, and the head was
    #      fitted on the former. Cosine similarity between the two is 1.000000 - the
    #      direction is identical, only the scale differs - so logits came out ~12x too
    #      large and every softmax saturated to 100%.
    #
    # The held-out confidence distribution is the check: 95th percentile 0.90, nothing above
    # 0.995, 31% below the 0.60 deferral threshold. If live inference disagrees with that,
    # the preprocessing has drifted.
    #
    # Train/serve skew of this kind is silent. The system does not error; it just lies about
    # how sure it is - which, for a tool whose entire purpose is knowing when not to answer,
    # is the worst failure available to it.

    def predict_image(self, image_path: str | Path) -> tuple[str, float, dict]:
        """Class, confidence and full distribution — from the IMAGE ALONE.

        The referral document is deliberately not a parameter here. That is the safety
        guarantee, expressed as a signature rather than as a comment.
        """
        with Image.open(image_path) as raw:
            resized = raw.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        tensor = TRANSFORM(resized).unsqueeze(0)

        with torch.no_grad():
            features = self.backbone(tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)   # step 2 above
            logits = (features @ self.head_weight.T + self.head_bias) / self.temperature
            probabilities = torch.softmax(logits, dim=1)[0].numpy()

        best = int(probabilities.argmax())
        return (self.labels[best], float(probabilities[best]),
                {code: round(float(p), 4) for code, p in zip(self.labels, probabilities)})

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
            features = [[diagnosis.features.get(f, 0.0)
                         for f in self._reliability["features"]]]
            confidence = float(self._reliability["model"].predict_proba(features)[0, 1])
        return fields, confidence

    def classify_stated_diagnosis(self, text: str | None) -> str | None:
        """Map free-text diagnosis onto one of the four classes, or None if unrecognised."""
        if not text:
            return None
        probe = text.lower()
        for code, keywords in (("C", ("cataract", "sclerosis", "sclerotic")),
                               ("D", ("diabetic", "retinopathy", "npdr")),
                               ("G", ("glaucoma",)),
                               ("N", ("normal", "no abnormality", "within normal"))):
            if any(word in probe for word in keywords):
                return code
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
                and result.extraction_confidence < 0.5):
            reasons.append("the referral fields were read with low reliability")

        result.deferral_reasons = reasons
        result.decision = "DEFER" if reasons else "ACCEPT"
        return result

    def to_case(self, result: ScreeningResult) -> dict:
        """Shape a result for the chatbot and the interface."""
        return {
            "image_id": result.image_id,
            "y_pred": result.predicted_class,
            "max_prob": result.confidence,
            "agreement": result.agreement or 0.0,
            "stated_label": result.stated_diagnosis_class,
            "deferred": result.decision == "DEFER",
            "deferral_reason": "; ".join(result.deferral_reasons),
            "extraction_confidence": result.extraction_confidence,
            "document_fields": result.document_fields,
            "calibrated": result.calibrated,
        }
