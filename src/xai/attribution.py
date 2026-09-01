"""Grad-CAM++ attribution and the metrics that decide whether to believe it.

WHAT MAY AND MAY NOT BE CLAIMED
-------------------------------
These maps show WHERE attribution falls and whether removing that region changes the
prediction. They do NOT show that the model is looking at a lesion. We have no lesion masks
and no ophthalmologist, so any sentence of the form "the model attends to the pathology" is
unsupported and must not appear in the report or in a figure caption.

The three measurements below are chosen because each can be computed without annotation:

FOV-mass ratio      Fraction of positive attribution inside the retinal disc. The mask comes
                    free from the same field-of-view crop used for deduplication. A model
                    exploiting border or letterbox artefacts scores low; one attending to
                    retinal tissue scores high. This is evidence about the shortcut question,
                    not about clinical correctness.

Deletion AUC        Progressively blank the highest-attribution regions and watch the
                    predicted probability fall. A faithful map collapses quickly. Known
                    caveat: masking creates out-of-distribution inputs, so this conflates
                    attribution quality with behaviour on corrupted images. Reported with
                    that stated.

Sanity check        Adebayo et al.: randomise the model's weights layer by layer. A faithful
                    method degrades toward noise. If a method passes this only weakly, it is
                    dropped from the scientific claims rather than explained away.
"""

from __future__ import annotations

import numpy as np

try:                                    # torch is a phase-3 dependency
    import torch
    import torch.nn.functional as F
except ImportError:                     # pragma: no cover
    torch = None


def retinal_mask(image: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    """Boolean mask of the retinal disc, from intensity alone.

    Fundus photographs sit on a black background. Anything above a fraction of the peak
    intensity is inside the field of view. Same rule as the deduplication fingerprint, so
    the two analyses agree about what counts as retina.
    """
    grey = image.mean(axis=2) if image.ndim == 3 else image
    peak = float(grey.max())
    return grey > peak * threshold if peak > 0 else np.zeros_like(grey, dtype=bool)


def grad_cam_plus_plus(model, image_tensor, target_class: int, target_layer):
    """Grad-CAM++ heat map, upsampled to the input resolution and scaled to [0, 1].

    Chosen over plain Grad-CAM because fundus images often carry several small lesions and
    Grad-CAM++ weights multiple activation peaks more evenly.
    """
    if torch is None:
        raise ImportError("torch is required for attribution")

    activations, gradients = {}, {}
    forward = target_layer.register_forward_hook(
        lambda _m, _i, out: activations.__setitem__("value", out))
    backward = target_layer.register_full_backward_hook(
        lambda _m, _gi, gout: gradients.__setitem__("value", gout[0]))

    try:
        model.zero_grad()
        output = model(image_tensor)
        output[0, target_class].backward()

        activation = activations["value"][0]           # (C, H, W)
        gradient = gradients["value"][0]

        squared = gradient.pow(2)
        cubed = gradient.pow(3)
        summed = activation.sum(dim=(1, 2), keepdim=True)
        denominator = 2 * squared + summed * cubed
        alpha = squared / torch.where(denominator != 0, denominator,
                                      torch.ones_like(denominator))
        weights = (alpha * F.relu(gradient)).sum(dim=(1, 2))

        cam = F.relu((weights[:, None, None] * activation).sum(dim=0))
        cam = F.interpolate(cam[None, None], size=image_tensor.shape[-2:],
                            mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max()).detach().cpu().numpy() if cam.max() > 0 else cam.detach().cpu().numpy()
    finally:
        forward.remove()
        backward.remove()


def fov_mass_ratio(attribution: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of total attribution that lands inside the retinal field of view.

    Interpretation is deliberately narrow: high means attribution is on retinal tissue.
    It says nothing about which tissue, or whether that tissue is diseased.
    """
    total = float(attribution.sum())
    if total <= 0:
        return 0.0
    return float(attribution[mask].sum() / total)


def deletion_curve(model, image_tensor, attribution: np.ndarray, target_class: int,
                   steps: int = 20):
    """Predicted probability as the most-attributed pixels are progressively blanked."""
    if torch is None:
        raise ImportError("torch is required for attribution")

    flat = attribution.flatten()
    order = np.argsort(-flat)                     # most important first
    total = len(order)
    probabilities = []

    with torch.no_grad():
        working = image_tensor.clone()
        for step in range(steps + 1):
            if step > 0:
                take = order[int((step - 1) * total / steps): int(step * total / steps)]
                rows, cols = np.unravel_index(take, attribution.shape)
                working[0, :, rows, cols] = 0.0
            probability = torch.softmax(model(working), dim=1)[0, target_class]
            probabilities.append(float(probability))

    fractions = np.linspace(0, 1, steps + 1)
    return {"fraction_removed": fractions.tolist(),
            "probability": probabilities,
            "auc": float(np.trapezoid(probabilities, fractions))}


def randomisation_sanity_check(model, image_tensor, target_class: int, target_layer,
                               build_fresh) -> dict:
    """Adebayo et al. model-randomisation test.

    Compare the map from the trained model against one from a randomly initialised model of
    the same architecture. A faithful method produces very different maps; a method that
    ignores the weights produces similar ones and must be dropped.
    """
    trained = grad_cam_plus_plus(model, image_tensor, target_class, target_layer)
    random_model, random_layer = build_fresh()
    random_map = grad_cam_plus_plus(random_model, image_tensor, target_class, random_layer)

    a, b = trained.flatten(), random_map.flatten()
    if a.std() == 0 or b.std() == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])

    return {
        "correlation_with_random_model": round(correlation, 4),
        "passes": bool(abs(correlation) < 0.5),
        "note": "a faithful method should NOT correlate with a randomly initialised model; "
                "if this fails, drop the method from the scientific claims",
    }
