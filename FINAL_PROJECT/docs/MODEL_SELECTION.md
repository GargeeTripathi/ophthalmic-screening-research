# Model Selection

## The comparison experiment

Six ImageNet-pretrained backbones were compared with everything else held fixed: the same 5,056 images, the same patient-disjoint folds, the same partial fine-tuning recipe, class weights, optimiser, schedule, early stopping, seed and evaluation code. Only the backbone changed. The experiment was run on Google Colab (Tesla T4).

Source of the numbers below: the comparison's own results table, as output by the Colab run. The comparison's result files are not part of this repository.

| Model | Accuracy | Precision | Recall | Specificity | Macro-F1 | AUROC | AUPRC | Parameters (M) | GFLOPs | Inference (ms/img, T4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 0.6814 | 0.6503 | 0.6363 | 0.8510 | 0.6413 | 0.8385 | 0.6728 | 23.52 | 8.174 | 5.80 |
| ResNet152 | 0.6576 | 0.6293 | 0.6348 | 0.8468 | 0.6312 | 0.8399 | 0.6659 | 58.15 | 23.023 | 16.35 |
| InceptionV3 | 0.6684 | 0.6295 | 0.6166 | 0.8443 | 0.6202 | 0.8350 | 0.6641 | 21.79 | 11.422 | 15.27 |
| EfficientNetV2-S | 0.6940 | 0.6884 | 0.6725 | 0.8573 | **0.6785** | 0.8520 | 0.7115 | 20.18 | 5.697 | 17.16 |
| **ConvNeXt-Tiny** | **0.6949** | 0.6780 | 0.6714 | **0.8598** | 0.6731 | **0.8717** | **0.7307** | 27.82 | 8.910 | 5.95 |
| DenseNet121 | 0.6541 | 0.6191 | 0.6193 | 0.8413 | 0.6183 | 0.8394 | 0.6612 | 6.96 | 5.666 | 20.45 |

The original project's ResNet50 (CPU run) scored accuracy 0.6744, macro-F1 0.6421, AUROC 0.8399 on the same folds.

## Why ConvNeXt-Tiny

- **Best on the ranking metrics.** Highest AUROC and AUPRC of the six, and highest accuracy and specificity. Ranking quality is what the confidence and deferral steps depend on.
- **Tied for best macro-F1.** EfficientNetV2-S is 0.005 higher (0.6785 against 0.6731), a difference too small to separate two models on this dataset.
- **Fast.** 5.95 ms per image, the same as ResNet50 and about a third of EfficientNetV2-S (17.16 ms).
- **Depth did not help.** ResNet152 scored below ResNet50, as the ODIR base paper also found.

## How sure is the improvement

The six backbones span only 0.62 to 0.68 macro-F1, and the comparison's confidence intervals and significance tests were not available when this document was written. For this final project the paired test was therefore run again here, between the ConvNeXt-Tiny trained in this repository and the original project's ResNet50 predictions:

On the 5056 images both models predicted (same patient-disjoint folds), ConvNeXt-Tiny scores macro-F1 0.663 against 0.6421 for the original project's ResNet50. Exact McNemar test on paired correctness: ConvNeXt-Tiny right where ResNet50 was wrong on 612 images, the reverse on 544; p = 0.048724, so the difference is only borderline significant (just under the 0.05 level) and should be described as a modest improvement. McNemar tests overall correctness (accuracy), not macro-F1.

## This repository against the comparison run

The comparison trained on a GPU; this repository's model was trained on a CPU with the same recipe and seed. Floating-point and library differences mean the numbers are close but not identical:

| | Comparison (Colab T4) | This repository (CPU) |
|---|---:|---:|
| Accuracy | 0.6949 | 0.6879 |
| Macro-F1 | 0.6731 | 0.6630 |
| AUROC | 0.8717 | 0.8618 |
| AUPRC | 0.7307 | 0.7170 |

The numbers reported everywhere else in this project are the right-hand column, measured from the checkpoints in this repository.
