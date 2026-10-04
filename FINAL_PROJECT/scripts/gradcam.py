"""Grad-CAM examples for the deployed ConvNeXt-Tiny model, on its held-out test fold.

    python scripts/gradcam.py              # 2 correct + 2 incorrect per class
    python scripts/gradcam.py --per-class 3

Needs results/metrics/predictions_deployed_model.csv (written by scripts/evaluate.py).
Each saved panel shows the original image and the Grad-CAM overlay, titled with the ground
truth, the predicted class and the confidence. A model-randomisation sanity check (Adebayo et
al. 2018) is run on the same images: if the heatmap barely changes when the trained weights
are replaced by random ones, the method is not explaining the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocessing import load_resized, to_tensor  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.explainability.gradcam import grad_cam, overlay  # noqa: E402
from src.inference.predict import Predictor  # noqa: E402
from src.models.convnext_tiny import ConvNeXtScreener  # noqa: E402
from src.utils.config import load_config, repo_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    config = load_config()
    results = repo_path(config["outputs"]["results"])
    out = results / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(results / "metrics" / "predictions_deployed_model.csv")
    paths = load_split().set_index("image_id").image_path
    predictor = Predictor()
    labels, names = predictor.labels, predictor.class_names

    torch.manual_seed(0)
    randomised = ConvNeXtScreener(len(labels), pretrained=False).eval()      # same architecture, random weights

    chosen = []
    for code in labels:
        of_class = predictions[predictions.y_true == code].sort_values("confidence", ascending=False)
        chosen += [(r, "correct") for r in of_class[of_class.y_pred == code].head(args.per_class).itertuples()]
        chosen += [(r, "incorrect") for r in of_class[of_class.y_pred != code].head(args.per_class).itertuples()]

    records, panels = [], []
    for row, kind in chosen:
        pixels = load_resized(paths[row.image_id])
        tensor = to_tensor(pixels)
        cam, index, _ = grad_cam(predictor.model, tensor)
        random_cam, _, _ = grad_cam(randomised, tensor, index)
        rho = float(spearmanr(cam.ravel(), random_cam.ravel()).statistic)
        blended = overlay(pixels, cam)
        title = (f"True: {names[row.y_true].replace('_', ' ')}  |  Predicted: {names[row.y_pred].replace('_', ' ')}  |  "
                 f"Confidence: {row.confidence:.1%}  ({kind})")
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.5))
        axes[0].imshow(pixels)
        axes[0].set_title("Original (resized to 224 x 224)", fontsize=9)
        axes[1].imshow(blended)
        axes[1].set_title("Grad-CAM for the predicted class", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{title}\n{row.image_id}", fontsize=10)
        fig.tight_layout()
        name = f"{row.y_true}_{kind}_{row.image_id}.png"
        fig.savefig(out / name, dpi=140)
        plt.close(fig)
        panels.append((pixels, blended, row, kind))
        records.append({"file": name, "image_id": row.image_id, "true": row.y_true, "predicted": row.y_pred,
                        "confidence": round(float(row.confidence), 4), "kind": kind,
                        "cam_nonzero_fraction": round(float((cam > 0.2).mean()), 3),
                        "spearman_vs_randomised_model": round(rho, 3)})
        print(f"  {name}", flush=True)

    # one overview sheet for slides
    columns = 4
    rows_n = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows_n, columns * 2, figsize=(columns * 4.6, rows_n * 2.6))
    for ax in np.ravel(axes):
        ax.axis("off")
    for n, (pixels, blended, row, kind) in enumerate(panels):
        r, c = divmod(n, columns)
        axes[r, 2 * c].imshow(pixels)
        axes[r, 2 * c + 1].imshow(blended)
        axes[r, 2 * c].set_title(f"true {row.y_true} / pred {row.y_pred} ({row.confidence:.0%})", fontsize=8,
                                 color="darkgreen" if kind == "correct" else "firebrick")
    fig.suptitle("ConvNeXt-Tiny Grad-CAM (target: last stage, features[7]) - green titles correct, red incorrect", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "gradcam_overview.png", dpi=130)
    plt.close(fig)

    rhos = [r["spearman_vs_randomised_model"] for r in records]
    summary = {"model": config["model"]["display_name"], "checkpoint": predictor.checkpoint.name,
               "target_layer": "last_stage = torchvision features[7], output (768, 7, 7)",
               "n_examples": len(records),
               "sanity_check": {"method": "model randomisation (Adebayo et al. 2018): Spearman rank correlation between "
                                          "the trained model's heatmap and a randomly initialised model's heatmap",
                                "mean_spearman": round(float(np.mean(rhos)), 3),
                                "max_abs_spearman": round(float(np.max(np.abs(rhos))), 3),
                                "passes": bool(abs(np.mean(rhos)) < 0.5),
                                "pass_rule": "|mean correlation| < 0.5: the heatmap depends on the trained weights"},
               "examples": records,
               "caveat": "Heatmaps show where the class score is sensitive. No lesion annotations exist in this "
                         "project, so they are not evidence that the model located pathology."}
    (out / "gradcam_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(records)} panels + overview -> {out}; sanity check mean Spearman {summary['sanity_check']['mean_spearman']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
