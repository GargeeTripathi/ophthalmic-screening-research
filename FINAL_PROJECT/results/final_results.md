# Final results - ConvNeXt-Tiny

Measured by `python scripts/evaluate.py`. Pooled out-of-fold test predictions on the patient-disjoint split; every image is scored once by a model that never saw its patient.

| Metric | ConvNeXt-Tiny |
|---|---:|
| Accuracy | 0.6879 |
| Precision (macro) | 0.6624 |
| Recall (macro) | 0.6664 |
| Sensitivity (macro) | 0.6664 |
| Specificity (macro) | 0.8537 |
| F1-score (macro) | 0.663 [95% CI 0.6442, 0.6812] |
| AUROC (macro, one-vs-rest) | 0.8618 |
| AUPRC (macro, one-vs-rest) | 0.717 |
| Parameters | 27.82 M (15.48 M trained) |
| Inference Time | 124.6 ms per image (CPU, batch 1, network only) |
| Test images / patients | 5056 / 2909 (pooled out-of-fold, folds [0, 1, 2, 3, 4]) |
