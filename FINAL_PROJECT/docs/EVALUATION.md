# Evaluation

## What is measured

**Pooled out-of-fold (headline).** Each fold model predicts its own held-out fold; predictions are pooled so each of the 5056 images is scored once by a model that never saw that patient. Folds evaluated: [0, 1, 2, 3, 4].

**Deployed checkpoint.** `best_model.pth` run file by file through the real inference path on its held-out fold.

## Multiclass conventions

- Precision, recall (sensitivity), specificity and F1 are computed per class (one-vs-rest) and macro-averaged, so each class counts equally despite the imbalance.
- AUROC and AUPRC are one-vs-rest per class, macro-averaged. AUPRC must be read against class prevalence (5.6% for glaucoma), which is drawn on the precision-recall plot.
- TP / TN / FP / FN are per class, one-vs-rest.
- Predictions use argmax; no threshold was tuned.
- 95% confidence intervals resample **patients** (2,000 bootstrap resamples), because two eyes of one patient are not independent.

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

Per-fold test macro-F1: fold 0: 0.679, fold 1: 0.63, fold 2: 0.6587, fold 3: 0.6789, fold 4: 0.6627.

## Deployed checkpoint

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

## Comparison with the previous model

On the 5056 images both models predicted (same patient-disjoint folds), ConvNeXt-Tiny scores macro-F1 0.663 against 0.6421 for the original project's ResNet50. Exact McNemar test on paired correctness: ConvNeXt-Tiny right where ResNet50 was wrong on 612 images, the reverse on 544; p = 0.048724, so the difference is only borderline significant (just under the 0.05 level) and should be described as a modest improvement. McNemar tests overall correctness (accuracy), not macro-F1.

## Files

| File | Content |
|---|---|
| `results/metrics/metrics.json` | every metric, per class and macro, confidence intervals, deployed-model block |
| `results/metrics/metrics.csv` | per-class table |
| `results/metrics/classification_report.txt` | scikit-learn report |
| `results/metrics/predictions.csv` | pooled out-of-fold predictions with probabilities |
| `results/metrics/predictions_deployed_model.csv` | deployed checkpoint on its test fold |
| `results/confusion_matrices/confusion_matrix.png` | counts and row percentages |
| `results/roc/roc_curve.png`, `results/pr/precision_recall_curve.png` | one-vs-rest curves |
| `results/final_results.csv` / `.xlsx` / `.md` | the summary table |

## Reproduce

```bash
python scripts/evaluate.py
```
