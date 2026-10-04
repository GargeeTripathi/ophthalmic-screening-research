# Project Explanation

## Problem

Detect ophthalmic disease from colour fundus photographs: classify one image as **normal, diabetic retinopathy, glaucoma or cataract**, and say how confident the model is so that uncertain cases go to a clinician. The earlier phases of the project showed that published accuracies on this dataset depend heavily on how the data is split, so the evaluation here is patient-disjoint throughout.

## Dataset

- **ODIR-5K** (Ocular Disease Intelligent Recognition), Kaggle release; fundus photographs of both eyes of each patient, from hospitals in China.
- Four single-label classes are used: normal (N) 2,871, diabetic retinopathy (D) 1,608, cataract (C) 293, glaucoma (G) 284 images. Imbalance about 10:1.
- **5,056 images from 2,909 patients** after removing 2 duplicate images.
- **Split:** 5 folds, `StratifiedGroupKFold` grouped by patient and stratified by label, seed 42, built in the original project and reused here unchanged. No patient appears in two folds.
- For test fold k: validation = fold (k + 1) mod 5, training = the other three folds (about 3,033 / 1,011 / 1,012 images).

## Preprocessing

1. Open the image, convert to RGB.
2. Resize to 224 x 224 with bilinear interpolation. No crop, so the retinal periphery is kept.
3. Scale to [0, 1] and normalise with the ImageNet mean and standard deviation.

No augmentation. This is the original project's preprocessing; the ImageNet normalisation is also what ConvNeXt-Tiny's pretrained weights expect. The same function (`src/data/preprocessing.py`) is used for training, evaluation, inference, Grad-CAM and the app.

## Model

ConvNeXt-Tiny pretrained on ImageNet-1k (torchvision). The stem and stages 1 to 3 are frozen; stage 4 is fine-tuned; the 1,000-class ImageNet classifier is replaced by global average pooling, LayerNorm(768) and a new Linear(768, 4). ConvNeXt has no `fc` layer like ResNet, and its final normalisation is part of the classifier, so the head was rebuilt for this architecture rather than swapped by name. Details: [docs/MODEL.md](docs/MODEL.md).

## Training

- Phase 1: 2 epochs, head only, AdamW, learning rate 1e-3.
- Phase 2: up to 8 epochs, stage 4 + head, AdamW, learning rate 1e-4 with cosine annealing, weight decay 1e-4.
- Early stopping with patience 3 on validation macro-F1; the best validation epoch is kept.
- Loss: cross-entropy with balanced class weights computed on the training folds (fold 0: N 0.4403, D 0.7858, G 4.4342, C 4.3329).
- Batch size 24, seed 42 + fold.
- The test fold is scored once, after training and model selection.

This is the recipe validated in the model comparison, which in turn copied the original ResNet50 recipe. Details: [docs/TRAINING.md](docs/TRAINING.md).

## Evaluation

Accuracy, macro precision, recall (sensitivity), specificity and F1, one-vs-rest AUROC and AUPRC, per-class TP / TN / FP / FN, confusion matrix, ROC and precision-recall curves, patient-level bootstrap 95% confidence intervals, and an exact McNemar test against the previous model. Details: [docs/EVALUATION.md](docs/EVALUATION.md).

## Explainability

Grad-CAM on the output of the last ConvNeXt stage (a 768-channel 7 x 7 map), the last layer with spatial layout. 16 examples (correct and incorrect, every class) are in `results/gradcam/`. A model-randomisation sanity check gave a mean rank correlation of 0.058 between the trained model's heatmaps and a randomly initialised model's (passes the rule |correlation| < 0.5, meaning the maps depend on the trained weights). Heatmaps are not validated against lesion annotations.

## Application

`python scripts/serve.py` starts a local web app. A fundus photograph goes through preprocessing, ConvNeXt-Tiny and temperature scaling to give a finding and a confidence. If a referral letter is uploaded, it is read by OCR and its stated diagnosis is compared with the image finding. The case is **deferred to a clinician** when confidence is below 60%, when the letter contradicts the image, or when the letter was read unreliably. The letter can never change the image finding: `predict_image()` takes only the image.

## Results

| Metric | ConvNeXt-Tiny |
|---|---:|
| Accuracy | 0.6879 (95% CI 0.6744 to 0.702) |
| Precision (macro) | 0.6624 |
| Recall / Sensitivity (macro) | 0.6664 |
| Specificity (macro) | 0.8537 |
| F1-score (macro) | 0.6630 (95% CI 0.6442 to 0.6812) |
| AUROC (macro, one-vs-rest) | 0.8618 |
| AUPRC (macro, one-vs-rest) | 0.7170 |
| Parameters | 27.82 M total, 15.48 M trained |
| Inference time | 124.6 ms per image (laptop CPU, batch 1) |

| Class | Images | Precision | Recall / Sensitivity | Specificity | F1 | AUROC | AUPRC | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 2871 | 0.7209 | 0.7673 | 0.6096 | 0.7434 | 0.7623 | 0.7753 | 2203 | 1332 | 853 | 668 |
| diabetic retinopathy | 1608 | 0.6247 | 0.5498 | 0.8460 | 0.5848 | 0.8006 | 0.6929 | 884 | 2917 | 531 | 724 |
| glaucoma | 284 | 0.4962 | 0.4577 | 0.9723 | 0.4762 | 0.8989 | 0.4784 | 130 | 4640 | 132 | 154 |
| cataract | 293 | 0.8080 | 0.8908 | 0.9870 | 0.8474 | 0.9853 | 0.9216 | 261 | 4701 | 62 | 32 |

The app serves one checkpoint, `checkpoints/convnext_tiny/best_model.pth` (the fold-0 model, fixed in the config before training). Run file by file through the real inference path on its own held-out fold (1012 images, 582 patients):

| Metric | Deployed checkpoint |
|---|---:|
| Accuracy | 0.6976 |
| Precision / Recall / Specificity (macro) | 0.7176 / 0.6546 / 0.8613 |
| F1-score (macro) | 0.6790 (95% CI 0.6334 to 0.7211) |
| AUROC / AUPRC (macro) | 0.8789 / 0.7498 |
| Glaucoma recall | 0.4107 |
| Cataract recall | 0.8136 |
| Calibration error (ECE) | 0.1106 before, 0.0202 after temperature scaling (T = 1.9582) |
| End-to-end time | 223 ms per image including file decoding (CPU) |

The cached-stem path used in training and the file-by-file path agree on 100.0% of predictions (largest probability difference 1e-06).

On the 5056 images both models predicted (same patient-disjoint folds), ConvNeXt-Tiny scores macro-F1 0.663 against 0.6421 for the original project's ResNet50. Exact McNemar test on paired correctness: ConvNeXt-Tiny right where ResNet50 was wrong on 612 images, the reverse on 544; p = 0.048724, so the difference is only borderline significant (just under the 0.05 level) and should be described as a modest improvement. McNemar tests overall correctness (accuracy), not macro-F1.

## What changed from the ResNet50 project

| Component | Original project | Final project |
|---|---|---|
| Backbone | ImageNet ResNet50 | ImageNet ConvNeXt-Tiny |
| Trained layers | `layer4` + `fc(2048, 4)` | stage 4 + `LayerNorm(768)` + `Linear(768, 4)` |
| Deployed app model | frozen ResNet50 features + L2 normalisation + linear head | the fine-tuned ConvNeXt-Tiny itself |
| Checkpoint | `screening_head.pt` | `checkpoints/convnext_tiny/best_model.pth` |
| Temperature scaling | fitted on the same fold it was evaluated on | fitted on the validation fold, evaluated on the test fold |
| Grad-CAM target | `layer4` | last stage (`features[7]`) |
| Configuration | several YAML files and constants in scripts | one `configs/config.yaml` |
| Paths | absolute Windows paths in manifests | image folder from `ODIR_IMAGES`; no absolute path in any tracked file |
| Split, classes, preprocessing, recipe, metrics code | | unchanged |

## Limitations

- **Not a medical device.** Academic prototype with no clinical validation and no regulatory clearance.
- **Modest absolute performance.** Macro-F1 is 0.6630. Glaucoma is the weakest class (recall 0.4577, F1 0.4762); the system is not a glaucoma screener.
- **One dataset.** ODIR-5K, from hospitals in China. No external test set; the original project showed the earlier model did not transfer to other image sources.
- **Small minority classes.** 284 glaucoma and 293 cataract images against 2,871 normal.
- **Training recipe held fixed, not optimised.** 224 px input, no augmentation, only the last stage fine-tuned, one seed per fold. These settings were kept identical to the original project and the comparison so results stay comparable.
- **Audit experiments were not re-run.** The leakage, duplicate, image-source and demographic analyses of the original project used ResNet50. They describe the dataset and protocol and are not repeated here.
- **Deferral rules were not re-tuned.** The confidence and agreement thresholds, and the deferral results (AURC), come from the original project's model. They have not been re-measured for ConvNeXt-Tiny.
- **Referral letters are synthetic.**
- **Grad-CAM is not validated against lesions.** There are no lesion annotations in this project.
