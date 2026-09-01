"""E12 - the explainability battery.

WHY THIS RUNS WITHOUT A GPU
---------------------------
Grad-CAM needs gradients to flow through the backbone; it does not need the backbone to
have been fine-tuned. So the model here is the same one E1 evaluated - a frozen ImageNet
ResNet50 with a linear head - reassembled in torch so gradients can be taken. The head is
fitted on the cached embeddings, which makes it numerically the same classifier E1 scored.

WHAT THAT MEANS FOR THE CLAIMS
------------------------------
Attributions reflect ImageNet features weighted by a linear probe, NOT a fine-tuned
diagnostic model. That is stated wherever these numbers appear. Re-run after the Colab
baselines land to get attributions for the fine-tuned model.

WHAT MAY AND MAY NOT BE CLAIMED
-------------------------------
Permitted: attribution falls inside or outside the retinal field of view; removing the
highlighted region does or does not change the prediction; the method does or does not pass
a randomisation sanity check.

Forbidden: that the model is looking at a lesion. There are no lesion masks and no
ophthalmologist. The FOV-mass ratio answers a shortcut question, not a clinical one.

    python scripts/run_xai.py
    python scripts/run_xai.py --n 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embeddings import load  # noqa: E402
from src.xai.attribution import (  # noqa: E402
    deletion_curve, fov_mass_ratio, grad_cam_plus_plus, retinal_mask,
)
from src.utils.config import load_config, output_dir  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SEED = 42
INPUT_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(labels: list[str], vectors: np.ndarray, y: np.ndarray, pretrained: bool = True):
    """Frozen ResNet50 + a linear head fitted on the cached embeddings.

    Numerically the same classifier E1 evaluated, but expressed in torch so Grad-CAM can
    take gradients through it.
    """
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    backbone.fc = nn.Linear(2048, len(labels))

    if pretrained:
        # Fit the head on the cached embeddings - seconds, and it matches E1's classifier.
        index = {code: i for i, code in enumerate(labels)}
        targets = torch.tensor([index[c] for c in y])
        features = torch.tensor(vectors, dtype=torch.float32)
        optimiser = torch.optim.Adam(backbone.fc.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        for _ in range(300):
            optimiser.zero_grad()
            loss = criterion(backbone.fc(features), targets)
            loss.backward()
            optimiser.step()

    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)      # Grad-CAM needs the graph, not training
    return backbone, backbone.layer4[-1]


def load_image(path: str) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    return TRANSFORM(rgb).unsqueeze(0), np.array(rgb, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80, help="images to attribute over")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    labels = load_config("data")["classes"]
    manifests, processed, results = (output_dir("manifests"), output_dir("processed"),
                                     output_dir("results"))

    # Fit the head on the training folds and attribute ONLY over the held-out fold.
    # Attributing over images the head was fitted on would make every map in-sample.
    splits = pd.read_csv(output_dir("splits") / "splits_patient_disjoint_dedup.csv",
                         encoding="utf-8")
    ids, vectors = load(processed / "embeddings_odir.npz")
    index = {image_id: position for position, image_id in enumerate(ids)}

    splits = splits[splits["image_id"].isin(index)]
    held_out = splits[splits["fold"] == 0]
    training = splits[splits["fold"] != 0]

    X = vectors[[index[i] for i in training["image_id"]]]
    y = training["label"].to_numpy()
    usable = held_out

    print("=" * 72)
    print("E12 - EXPLAINABILITY BATTERY")
    print("=" * 72)
    print(f"\nmodel: frozen ImageNet ResNet50 + linear head fitted on {len(training)} "
          "training-fold embeddings")
    print("       (the same classifier E1 scored - NOT a fine-tuned diagnostic model)")
    print(f"attribution set: fold 0, held out from the head - {len(held_out)} images")

    model, target_layer = build_model(labels, X, y)

    # Stratified sample so every class is represented in the attribution analysis.
    per_class = max(1, args.n // len(labels))
    sample = pd.concat([block.sample(min(len(block), per_class), random_state=SEED)
                        for _, block in usable.groupby("label")], ignore_index=True)
    print(f"attributing over {len(sample)} images "
          f"({sample['label'].value_counts().to_dict()})")

    rows = []
    for position, row in enumerate(sample.itertuples(), start=1):
        tensor, rgb = load_image(row.image_path)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        predicted = int(probabilities.argmax())

        attribution = grad_cam_plus_plus(model, tensor, predicted, target_layer)
        mask = retinal_mask(rgb)
        deletion = deletion_curve(model, tensor, attribution, predicted, steps=10)

        rows.append({
            "image_id": row.image_id, "true_label": row.label,
            "predicted_label": labels[predicted],
            "correct": labels[predicted] == row.label,
            "confidence": float(probabilities[predicted]),
            "fov_mass_ratio": round(fov_mass_ratio(attribution, mask), 4),
            "fov_area_fraction": round(float(mask.mean()), 4),
            "deletion_auc": round(deletion["auc"], 4),
        })
        if position % 20 == 0:
            print(f"    {position}/{len(sample)}")

    frame = pd.DataFrame(rows)
    frame.to_csv(results / "E12_xai_per_image.csv", index=False, encoding="utf-8")

    # --- interpretation -----------------------------------------------------------------
    # The right baseline for FOV-mass is the mask's own area fraction: uniform attribution
    # would score exactly that. Anything above it means attribution concentrates on retina.
    mean_fov = float(frame["fov_mass_ratio"].mean())
    mean_area = float(frame["fov_area_fraction"].mean())

    print("\n" + "-" * 72)
    print("FOV-MASS RATIO  (fraction of attribution inside the retinal field of view)")
    print("-" * 72)
    print(f"  observed                        {mean_fov:.4f}")
    print(f"  uniform-attribution baseline    {mean_area:.4f}   (the mask's area fraction)")
    print(f"  concentration ratio             {mean_fov / mean_area:.3f}x")
    print("\n  by class:")
    print(frame.groupby("true_label")[["fov_mass_ratio", "deletion_auc", "confidence"]]
          .mean().round(4).to_string())

    print("\n" + "-" * 72)
    print("DELETION AUC  (lower = removing highlighted pixels collapses the prediction)")
    print("-" * 72)
    print(f"  mean {frame['deletion_auc'].mean():.4f}   "
          f"correct predictions {frame[frame['correct']]['deletion_auc'].mean():.4f}   "
          f"incorrect {frame[~frame['correct']]['deletion_auc'].mean():.4f}")

    # --- sanity check --------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("RANDOMISATION SANITY CHECK  (Adebayo et al.)")
    print("-" * 72)
    random_model, random_layer = build_model(labels, X, y, pretrained=False)
    correlations = []
    for row in sample.head(12).itertuples():
        tensor, _ = load_image(row.image_path)
        with torch.no_grad():
            predicted = int(torch.softmax(model(tensor), dim=1)[0].argmax())
        trained_map = grad_cam_plus_plus(model, tensor, predicted, target_layer).flatten()
        random_map = grad_cam_plus_plus(random_model, tensor, predicted, random_layer).flatten()
        if trained_map.std() > 0 and random_map.std() > 0:
            correlations.append(float(np.corrcoef(trained_map, random_map)[0, 1]))

    mean_correlation = float(np.mean(correlations)) if correlations else 0.0
    passes = abs(mean_correlation) < 0.5
    print(f"  mean correlation with a randomly initialised model: {mean_correlation:+.4f}")
    print(f"  verdict: {'PASS' if passes else 'FAIL'} "
          f"({'attributions depend on the learned weights' if passes else 'attributions are weight-independent - DROP this method from the claims'})")

    summary = {
        "model": "frozen ImageNet ResNet50 + linear head on cached embeddings",
        "head_fitted_on": int(len(training)),
        "attributed_over": "held-out fold 0 - the head never saw these images",
        "model_caveat": "not a fine-tuned diagnostic model; re-run after the Colab baselines",
        "n_images": int(len(frame)),
        "fov_mass_ratio_mean": round(mean_fov, 4),
        "uniform_attribution_baseline": round(mean_area, 4),
        "concentration_ratio": round(mean_fov / mean_area, 4) if mean_area else None,
        "fov_mass_by_class": frame.groupby("true_label")["fov_mass_ratio"].mean().round(4).to_dict(),
        "deletion_auc_mean": round(float(frame["deletion_auc"].mean()), 4),
        "deletion_auc_correct": round(float(frame[frame["correct"]]["deletion_auc"].mean()), 4),
        "deletion_auc_incorrect": round(float(frame[~frame["correct"]]["deletion_auc"].mean()), 4),
        "randomisation_correlation": round(mean_correlation, 4),
        "randomisation_passes": bool(passes),
        "permitted_claims": [
            "attribution falls inside or outside the retinal field of view",
            "removing highlighted regions does or does not change the prediction",
            "the method does or does not pass a randomisation sanity check",
        ],
        "forbidden_claims": [
            "the model is looking at a lesion",
            "attribution demonstrates clinical validity",
        ],
        "deletion_caveat": "masking creates out-of-distribution inputs, so deletion AUC "
                           "conflates attribution quality with behaviour on corrupted images",
    }
    (results / "E12_xai.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    print(f"\nWRITTEN to {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
