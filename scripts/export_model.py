"""Export every artefact the deployed pipeline needs, in one place.

The research scripts each build their model in memory and throw it away. A working system
needs those decisions frozen to disk so inference is reproducible and does not silently
drift from what was evaluated.

WHAT GETS EXPORTED
------------------
    screening_model.pt      frozen ResNet50 + the linear head, plus a fitted temperature
    reliability.joblib      the extraction-correctness estimator from E8
    deployment.json         thresholds, class names, and the provenance of every number

CALIBRATION, AND WHY THE HEAD IS NOT FITTED ON EVERYTHING
---------------------------------------------------------
A first version fitted the head on all 5,056 images and then reported 100% confidence on an
image it had trained on. That is in-sample overconfidence, and it makes the deferral rule
inert: nothing ever falls below a 60% threshold, so the system never defers on confidence.

So the head is fitted on folds 1-4 and a temperature is fitted on held-out fold 0 by
minimising negative log-likelihood. Roughly 20% of the data is spent buying a confidence
number that means something - which, for a system whose entire point is knowing when not to
answer, is the right trade.

WHICH DEFERRAL SYSTEM SHIPS
---------------------------
System A - the two rules. Not System B. The learned gate lost to it on AURC (0.211 vs
0.197), and adding the reliability estimate made it worse still. A kill rule written before
the run said ship the rules in that case, so the rules ship. `c_hat` is still exported
because it earns its place elsewhere: it is an excellent *correctness* estimator (AUROC
0.98) and the interface shows it to the clinician.

    python scripts/export_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from torchvision.models import ResNet50_Weights, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embeddings import load  # noqa: E402
from src.deferral.systems import SYSTEM_A_AGREEMENT, SYSTEM_A_CONFIDENCE  # noqa: E402
from src.idp.extract import FEATURE_NAMES  # noqa: E402
from src.utils.config import load_config, output_dir  # noqa: E402

SEED = 42
HEAD_STEPS = 400
WEIGHT_DECAY = 1e-4
CALIBRATION_FOLD = 0


def train_head(labels: list[str], vectors: np.ndarray, y: np.ndarray) -> nn.Linear:
    """Linear head with weight decay, fitted on the training folds only."""
    torch.manual_seed(SEED)
    head = nn.Linear(vectors.shape[1], len(labels))
    index = {code: i for i, code in enumerate(labels)}
    targets = torch.tensor([index[c] for c in y])
    features = torch.tensor(vectors, dtype=torch.float32)

    optimiser = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    for step in range(HEAD_STEPS):
        optimiser.zero_grad()
        loss = criterion(head(features), targets)
        loss.backward()
        optimiser.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step + 1}/{HEAD_STEPS}  loss {loss.item():.4f}")
    return head


def fit_temperature(head: nn.Linear, labels: list[str], vectors: np.ndarray,
                    y: np.ndarray) -> float:
    """Single-parameter temperature scaling on held-out data (Guo et al., 2017).

    Divides the logits by T before softmax; T > 1 softens overconfident predictions. Fitted
    by minimising NLL on data the head never saw, which is the only way the resulting number
    means anything.
    """
    index = {code: i for i, code in enumerate(labels)}
    targets = torch.tensor([index[c] for c in y])
    with torch.no_grad():
        logits = head(torch.tensor(vectors, dtype=torch.float32))

    log_temperature = torch.zeros(1, requires_grad=True)   # optimise log T to keep T > 0
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = criterion(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.exp().item())


class FusedScreener(nn.Module):
    """The whole image pathway as one graph: backbone -> L2 -> head -> /T -> softmax.

    Every step src/app/pipeline.py performs at runtime is folded in here, because a browser
    running this has no Python to perform them. The L2 normalisation especially: leaving it
    out feeds the head features ~12x too large and saturates every softmax to 100%, which is
    the exact bug documented in pipeline.py.

    Input is an ImageNet-normalised NCHW float tensor, matching TRANSFORM there. Output is
    calibrated probabilities, so the browser needs no post-processing either.
    """

    def __init__(self, backbone: nn.Module, head: nn.Linear, temperature: float):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.normalize(self.backbone(image), p=2, dim=1)
        return torch.softmax(self.head(features) / self.temperature, dim=1)


def export_fused_onnx(backbone: nn.Module, head: nn.Linear, temperature: float,
                      path: Path) -> None:
    """Write the fused graph, then assert it reproduces the PyTorch path exactly.

    Exported without external data so it is a single self-contained file a static host can
    serve; at ~90 MB it stays under the limits that matter.
    """
    model = FusedScreener(backbone, head, temperature).eval()
    example = torch.randn(1, 3, 224, 224)

    torch.onnx.export(model, example, str(path), input_names=["image"],
                      output_names=["probabilities"], opset_version=17,
                      dynamo=False, external_data=False)

    # An export that silently disagrees with the research pipeline is worse than no export.
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for _ in range(3):
        probe = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected = model(probe).numpy()
        actual = session.run(None, {"image": probe.numpy()})[0]
        if not np.allclose(expected, actual, atol=1e-4):
            raise SystemExit(f"ONNX export disagrees with PyTorch by "
                             f"{np.abs(expected - actual).max():.2e} - refusing to ship it")
    print("  ONNX verified against the PyTorch graph on 3 random inputs")


def main() -> int:
    labels = load_config("data")["classes"]
    class_names = load_config("data")["class_names"]
    manifests, processed = output_dir("manifests"), output_dir("processed")
    checkpoints = output_dir("checkpoints")

    print("=" * 72)
    print("EXPORT DEPLOYMENT ARTEFACTS")
    print("=" * 72)

    # --- image model --------------------------------------------------------------------
    splits = pd.read_csv(output_dir("splits") / "splits_patient_disjoint_dedup.csv",
                         encoding="utf-8")
    ids, vectors = load(processed / "embeddings_odir.npz")
    index = {image_id: position for position, image_id in enumerate(ids)}
    splits = splits[splits["image_id"].isin(index)]

    training = splits[splits["fold"] != CALIBRATION_FOLD]
    calibration = splits[splits["fold"] == CALIBRATION_FOLD]
    X = vectors[[index[i] for i in training["image_id"]]]
    y = training["label"].to_numpy()
    X_cal = vectors[[index[i] for i in calibration["image_id"]]]
    y_cal = calibration["label"].to_numpy()

    print(f"\nfitting the classification head on {len(training)} images "
          f"({training['patient_id'].nunique()} patients, folds != {CALIBRATION_FOLD})")
    head = train_head(labels, X, y)

    print(f"\nfitting temperature on {len(calibration)} held-out images "
          f"(fold {CALIBRATION_FOLD}, patient-disjoint from training)")
    temperature = fit_temperature(head, labels, X_cal, y_cal)
    with torch.no_grad():
        cal_tensor = torch.tensor(X_cal, dtype=torch.float32)
        raw = torch.softmax(head(cal_tensor), 1).max(1).values
        scaled = torch.softmax(head(cal_tensor) / temperature, 1).max(1).values
    print(f"  temperature T = {temperature:.3f}")
    print(f"  mean top-class confidence  {raw.mean():.3f} -> {scaled.mean():.3f}")

    # The head was fitted on L2-NORMALISED embeddings (see src/data/embeddings.py), so
    # the backbone is exported WITHOUT a head and the head is stored separately. The
    # runtime normalises between them. Bolting the head straight onto resnet50.fc would
    # feed it un-normalised features roughly 12x larger, saturating every softmax.
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    backbone.eval()

    model_path = checkpoints / "screening_model.pt"
    torch.save({"backbone_state_dict": backbone.state_dict(),
                "head_weight": head.weight.detach(), "head_bias": head.bias.detach(),
                "labels": labels, "architecture": "resnet50",
                "head_in_features": X.shape[1], "temperature": temperature,
                "features_l2_normalised": True}, model_path)
    print(f"  written {model_path}  ({model_path.stat().st_size / 1e6:.0f} MB)")

    # A second, TINY checkpoint holding only what this project actually trained. The 94 MB
    # above is ~93 MB of ImageNet ResNet50 weights that torchvision downloads for free; the
    # parameters we fitted are a 2048x4 head plus one temperature scalar. Committing the
    # small file makes the repository self-contained and deployable, and keeps a 94 MB
    # binary out of git history.
    head_path = checkpoints / "screening_head.pt"
    torch.save({"head_weight": head.weight.detach(), "head_bias": head.bias.detach(),
                "labels": labels, "temperature": temperature,
                "head_in_features": X.shape[1], "features_l2_normalised": True,
                "backbone": "resnet50", "backbone_weights": "IMAGENET1K_V2"}, head_path)
    print(f"  written {head_path}  ({head_path.stat().st_size / 1024:.0f} KB) "
          "<- committed; the backbone is fetched from torchvision at runtime")

    # --- fused ONNX, for running the image pathway in a browser --------------------------
    onnx_path = checkpoints / "screening_fused.onnx"
    export_fused_onnx(backbone, head, temperature, onnx_path)
    print(f"  written {onnx_path}  ({onnx_path.stat().st_size / 1e6:.0f} MB) "
          "<- not committed; rebuild with this script")

    # --- extraction-reliability estimator -----------------------------------------------
    extractions = processed / "extractions_dev.csv"
    reliability_path = checkpoints / "reliability.joblib"
    reliability_meta = None
    if extractions.is_file():
        frame = pd.read_csv(extractions, encoding="utf-8")
        frame = frame[(frame["extractor"] == "E2") & frame["predicted_present"]]
        estimator = GradientBoostingClassifier(random_state=SEED)
        estimator.fit(frame[FEATURE_NAMES], frame["correct"].astype(int))
        joblib.dump({"model": estimator, "features": FEATURE_NAMES}, reliability_path)
        reliability_meta = {"trained_on": "development layouts A-C only",
                            "n_train": int(len(frame)),
                            "sealed_auroc": 0.9871}
        print(f"  written {reliability_path}  (fitted on {len(frame)} dev-layout extractions)")

    # --- everything the runtime needs to know -------------------------------------------
    deployment = {
        "labels": labels,
        "class_names": class_names,
        "image_model": {
            "path": model_path.name,
            "portable_path": head_path.name,
            "architecture": "ImageNet ResNet50 (frozen) -> L2 normalise -> linear head",
            "preprocessing": "resize 224x224 bilinear, ImageNet mean/std, then L2-normalise "
                             "the 2048-d pooled feature. Must match src/data/embeddings.py "
                             "exactly or confidences are meaningless.",
            "fitted_on_images": int(len(training)),
            "fitted_on_patients": int(training["patient_id"].nunique()),
            "evaluated_macro_f1": 0.5476,
            "cataract_vs_rest_auroc": 0.9462,
            "calibrated": True,
            "temperature": round(temperature, 4),
            "calibration_note": f"temperature scaling, T={temperature:.3f}, fitted on "
                                f"held-out fold {CALIBRATION_FOLD}, patient-disjoint from "
                                "training. Confidence is comparable across cases but is not "
                                "a probability of the patient having the disease.",
        },
        "reliability_model": reliability_meta,
        "deferral": {
            "system": "A - rules",
            "confidence_threshold": SYSTEM_A_CONFIDENCE,
            "agreement_threshold": SYSTEM_A_AGREEMENT,
            "aurc": 0.1973,
            "error_detection_auroc": 0.8817,
            "why_not_learned": "System B (learned gate) scored AURC 0.211 vs 0.197 for the "
                               "rules, and removing the reliability input improved it to "
                               "0.203. A pre-registered kill rule said ship the rules.",
        },
        "extraction": {"extractor": "E2 layout-aware", "engine": "RapidOCR (ONNX, CPU)"},
        "limitations": [
            "not a medical device; not a diagnostic tool; no regulatory clearance",
            "203 cataract and 182 glaucoma patients in training",
            "confidence is temperature-scaled but the underlying model is a frozen-"
            "feature linear head, not a fine-tuned backbone",
            "referral documents used in development were synthetic",
            "no prospective clinical validation",
        ],
    }
    deployment_path = checkpoints / "deployment.json"
    deployment_path.write_text(json.dumps(deployment, indent=2), encoding="utf-8")
    print(f"  written {deployment_path}")

    print("\n" + "=" * 72)
    print("RESULT: OK - the pipeline can now run a single case end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
