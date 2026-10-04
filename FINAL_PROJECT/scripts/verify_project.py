"""Verify that the final project is complete and runnable. Prints a PASS / FAIL summary.

    python scripts/verify_project.py

Works without the ODIR images: it uses a generated test image for the inference check.
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        CHECKS.append((name, True, str(detail or "")))
    except Exception as exc:  # noqa: BLE001 - every failure is reported, none aborts the run
        CHECKS.append((name, False, f"{type(exc).__name__}: {exc}"))


def required_files():
    needed = ["README.md", "PROJECT_EXPLANATION.md", "requirements.txt", ".gitignore", "configs/config.yaml",
              "data/README.md", "data/manifests/splits_patient_disjoint_dedup.csv",
              "src/models/convnext_tiny.py", "src/data/preprocessing.py", "src/training/train.py",
              "src/evaluation/evaluate.py", "src/explainability/gradcam.py", "src/inference/predict.py",
              "src/app/pipeline.py", "scripts/train.py", "scripts/evaluate.py", "scripts/predict.py",
              "scripts/gradcam.py", "scripts/serve.py", "notebooks/final_results.ipynb",
              "docs/MODEL.md", "docs/TRAINING.md", "docs/EVALUATION.md", "docs/DEPLOYMENT.md", "docs/MODEL_SELECTION.md"]
    missing = [f for f in needed if not (ROOT / f).is_file()]
    assert not missing, f"missing: {missing}"
    return f"{len(needed)} files"


def config_valid():
    from src.utils.config import load_config
    c = load_config()
    assert c["model"]["name"] == "convnext_tiny" and c["model"]["pretrained"] is True
    assert c["model"]["n_classes"] == len(c["data"]["labels"]) == len(c["data"]["class_names"]) == 4
    assert c["preprocessing"]["input_size"] == 224
    return f"model {c['model']['name']}, labels {c['data']['labels']}"


def split_is_leak_free():
    from src.data.splits import leakage_report, load_split
    report = leakage_report(load_split(require_images=False))
    assert report["patients_in_more_than_one_fold"] == 0 and report["duplicate_image_ids"] == 0, report
    return f"{report['images']} images, {report['patients']} patients, 0 patients across folds"


def model_instantiates():
    import torch
    from src.models.convnext_tiny import ConvNeXtScreener
    net = ConvNeXtScreener(4, pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 4) and torch.isfinite(out).all()
    return "input (2, 3, 224, 224) -> output (2, 4), finite"


def gradients_flow():
    from src.training.train import sanity_check
    return sanity_check(4)["status"] + " (loss, gradients on trained layers only, optimiser step)"


def checkpoint_loads():
    from src.models.convnext_tiny import load_checkpoint
    from src.utils.config import load_config, repo_path
    path = repo_path(load_config()["model"]["checkpoint"])
    assert path.is_file(), f"{path} missing - run scripts/train.py"
    _, meta = load_checkpoint(path)
    assert meta["architecture"] == "convnext_tiny" and len(meta["labels"]) == 4 and not meta["smoke_test"]
    return f"{path.name} ({path.stat().st_size / 1e6:.1f} MB), test fold {meta['test_fold']}, T={meta['temperature']}"


def inference_works():
    import numpy as np
    from PIL import Image
    from src.inference.predict import Predictor
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.jpg"
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(path)
        out = Predictor().predict(path)
    assert out["predicted_class"] in out["probabilities"] and len(out["probabilities"]) == 4
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-2 and 0 <= out["confidence"] <= 1
    return f"4 probabilities summing to 1, model {out['model']}, checkpoint {out['checkpoint']}"


def gradcam_works():
    import torch
    from src.explainability.gradcam import grad_cam
    from src.inference.predict import Predictor
    cam, index, _ = grad_cam(Predictor().model, torch.randn(1, 3, 224, 224))
    assert cam.shape == (224, 224) and cam.min() >= 0 and cam.max() <= 1 and cam.max() > 0
    return f"heatmap (224, 224) in [0, 1] for class index {index}"


def app_pipeline_works():
    import numpy as np
    from PIL import Image
    from src.app.pipeline import ScreeningPipeline
    from src.app.render import worklist
    pipeline = ScreeningPipeline()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.jpg"
        Image.fromarray(np.full((256, 256, 3), 120, dtype=np.uint8)).save(path)
        result = pipeline.screen(path)
    assert result.decision in {"ACCEPT", "DEFER"} and "ConvNeXt" in pipeline.config["image_model"]["architecture"]
    assert "<html" in worklist([]).lower()
    return f"screen() -> {result.decision}; worklist page renders"


def no_stale_backbone():
    """No code path in the final project may load the old backbone."""
    import re
    hits = []
    for path in list((ROOT / "src").rglob("*.py")) + [p for p in (ROOT / "scripts").glob("*.py") if p.name != "verify_project.py"]:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            if re.search(r"resnet50\s*\(|ResNet50_Weights|import.*resnet|screening_head\.pt|screening_model\.pt", code):
                hits.append(f"{path.relative_to(ROOT)}:{n}")
    assert not hits, f"old-backbone code references: {hits}"
    return "no ResNet50 constructor, weights or checkpoint referenced in src/ or scripts/"


def no_absolute_paths():
    import re
    hits = []
    for path in [p for p in ROOT.rglob("*") if p.suffix in {".py", ".yaml", ".md", ".txt", ".ipynb", ".json", ".csv"}
                 and not any(part in {".git", "cache", "uploads", "__pycache__", ".pytest_cache"} for part in p.parts)]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"[A-Z]:[\\/](?:Users|PBL|rem_optical)|/content/drive/MyDrive/[A-Za-z]", text) and path.name not in {"verify_project.py", "final_results.ipynb", "train_log.txt"}:
            hits.append(str(path.relative_to(ROOT)))
    assert not hits, f"machine-specific paths in: {hits}"
    return "no machine-specific absolute paths in tracked text files"


def metrics_exist():
    metrics = json.loads((ROOT / "results/metrics/metrics.json").read_text(encoding="utf-8"))
    for key in ("accuracy", "macro_precision", "macro_recall_sensitivity", "macro_specificity", "macro_f1",
                "macro_auroc_ovr", "macro_auprc_ovr", "per_class", "confusion_matrix"):
        assert key in metrics, f"metrics.json lacks {key}"
    assert all(k in v for v in metrics["per_class"].values() for k in ("TP", "TN", "FP", "FN"))
    needed = ["results/metrics/metrics.csv", "results/metrics/predictions.csv", "results/metrics/classification_report.txt",
              "results/final_results.csv", "results/final_results.md", "results/final_results.xlsx",
              "results/training/training_history.csv"]
    missing = [f for f in needed if not (ROOT / f).is_file()]
    assert not missing, f"missing: {missing}"
    return f"macro-F1 {metrics['macro_f1']}, accuracy {metrics['accuracy']}, {metrics['n_images']} test images, folds {metrics['folds']}"


def plots_exist():
    needed = ["results/training/training_history.png", "results/confusion_matrices/confusion_matrix.png",
              "results/roc/roc_curve.png", "results/pr/precision_recall_curve.png", "results/gradcam/gradcam_overview.png"]
    missing = [f for f in needed if not (ROOT / f).is_file() or (ROOT / f).stat().st_size < 5000]
    assert not missing, f"missing or empty: {missing}"
    return f"{len(needed)} figures, {len(list((ROOT / 'results/gradcam').glob('*.png')))} Grad-CAM images"


def docs_match_metrics():
    metrics = json.loads((ROOT / "results/metrics/metrics.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{metrics['macro_f1']:.4f}" in readme and f"{metrics['accuracy']:.4f}" in readme, \
        "README does not quote the measured macro-F1 / accuracy"
    return "README quotes the measured accuracy and macro-F1"


def main() -> int:
    for name, fn in [("required files exist", required_files), ("configuration is valid", config_valid),
                     ("split has no patient leakage", split_is_leak_free), ("model instantiates, output shape correct", model_instantiates),
                     ("loss, gradients and optimiser step", gradients_flow), ("checkpoint exists and loads", checkpoint_loads),
                     ("preprocessing + inference", inference_works), ("Grad-CAM", gradcam_works),
                     ("app pipeline and page rendering", app_pipeline_works), ("no stale ResNet50 code", no_stale_backbone),
                     ("no machine-specific paths", no_absolute_paths), ("metrics files", metrics_exist),
                     ("result plots", plots_exist), ("documentation matches results", docs_match_metrics)]:
        check(name, fn)
    width = max(len(n) for n, _, _ in CHECKS)
    print("\nFINAL PROJECT VERIFICATION\n" + "=" * 26)
    for name, ok, detail in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [n for n, ok, _ in CHECKS if not ok]
    print("=" * 26 + f"\n{'ALL CHECKS PASSED' if not failed else f'{len(failed)} CHECK(S) FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
