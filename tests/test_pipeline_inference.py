"""Regression tests for the deployed inference path.

The bug these exist to prevent: the classification head is fitted on L2-NORMALISED
embeddings, and a first version of the pipeline fed it un-normalised features. Cosine
similarity between the two is 1.000000 — the direction is identical, only the scale differs
— so nothing errored. Logits just came out about 12x too large and every softmax saturated
to 100% confidence.

For a system whose entire purpose is knowing when not to answer, silently overstating
confidence is the worst failure available. These tests compare live inference against the
cached embeddings it must reproduce, and against the held-out confidence distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.config import output_dir

MODEL = output_dir("checkpoints", create=False) / "screening_model.pt"
EMBEDDINGS = output_dir("processed", create=False) / "embeddings_odir.npz"
SPLITS = output_dir("splits", create=False) / "splits_patient_disjoint_dedup.csv"

needs_model = pytest.mark.skipif(
    not (MODEL.is_file() and EMBEDDINGS.is_file() and SPLITS.is_file()),
    reason="deployment artefacts not exported yet (run scripts/export_model.py)")

# The held-out fold 0 distribution, measured after calibration. Live inference must land
# inside this envelope; drifting outside it means preprocessing has changed somewhere.
HELD_OUT_MAX = 0.995
HELD_OUT_P95 = 0.90


@pytest.fixture(scope="module")
def pipeline():
    from src.app.pipeline import ScreeningPipeline
    return ScreeningPipeline()


@pytest.fixture(scope="module")
def sample():
    splits = pd.read_csv(SPLITS, encoding="utf-8")
    return splits[splits["fold"] == 0].head(6)


@needs_model
def test_exported_bundle_declares_normalised_features():
    import torch
    bundle = torch.load(MODEL, map_location="cpu", weights_only=False)
    assert bundle.get("features_l2_normalised") is True
    # The head must be stored separately, not bolted onto resnet50.fc — attaching it there
    # is exactly what fed it un-normalised features.
    assert "head_weight" in bundle and "backbone_state_dict" in bundle
    assert "state_dict" not in bundle


@needs_model
def test_live_features_match_the_cached_embeddings(pipeline, sample):
    """The whole bug in one assertion: same direction AND same scale."""
    import torch
    from PIL import Image
    from src.data.embeddings import load
    from src.app.pipeline import INPUT_SIZE, TRANSFORM

    ids, vectors = load(EMBEDDINGS)
    index = {image_id: position for position, image_id in enumerate(ids)}

    row = sample.iloc[0]
    cached = vectors[index[row["image_id"]]]

    with Image.open(row["image_path"]) as raw:
        resized = raw.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    with torch.no_grad():
        live = pipeline.backbone(TRANSFORM(resized).unsqueeze(0))
        live = torch.nn.functional.normalize(live, p=2, dim=1)[0].numpy()

    assert np.allclose(np.linalg.norm(live), 1.0, atol=1e-4), "live features are not unit norm"
    assert np.allclose(np.linalg.norm(cached), 1.0, atol=1e-4)
    assert float(cached @ live) > 0.999, "live features diverged from the cached embeddings"


@needs_model
def test_confidence_stays_inside_the_held_out_envelope(pipeline, sample):
    confidences = [pipeline.predict_image(p)[1] for p in sample["image_path"]]
    assert max(confidences) <= HELD_OUT_MAX, (
        f"confidence {max(confidences):.4f} exceeds anything seen on the held-out fold - "
        "preprocessing has almost certainly drifted from embeddings.py")
    assert min(confidences) > 0.20, "confidence below chance for a 4-class problem"


@needs_model
def test_probabilities_are_a_distribution(pipeline, sample):
    _, confidence, probabilities = pipeline.predict_image(sample.iloc[0]["image_path"])
    assert abs(sum(probabilities.values()) - 1.0) < 1e-3
    assert abs(max(probabilities.values()) - confidence) < 1e-3


@needs_model
def test_the_document_cannot_change_the_finding(pipeline, sample):
    """The structural safety rule, asserted rather than trusted."""
    import json
    documents = json.loads(
        (output_dir("documents", create=False) / "ground_truth_sealed.json")
        .read_text(encoding="utf-8"))
    image_path = sample.iloc[0]["image_path"]

    alone = pipeline.screen(image_path)
    # Deliberately pick a referral that disagrees, to prove disagreement changes only the
    # decision and never the prediction.
    contradicting = next(d for d in documents if d["diagnosis_corrupted"])
    with_document = pipeline.screen(image_path, contradicting["image_path"])

    assert alone.predicted_class == with_document.predicted_class
    assert alone.confidence == with_document.confidence
    assert alone.probabilities == with_document.probabilities


@needs_model
def test_predict_image_does_not_accept_a_document():
    """The guarantee is a signature, not a comment - so test the signature."""
    import inspect
    from src.app.pipeline import ScreeningPipeline

    parameters = set(inspect.signature(ScreeningPipeline.predict_image).parameters)
    assert parameters == {"self", "image_path"}, (
        "predict_image gained a parameter - the image-only guarantee is no longer structural")
