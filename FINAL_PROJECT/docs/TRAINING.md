# Training

## Configuration (from `configs/config.yaml`)

| Setting | Value |
|---|---|
| Model | ConvNeXt-Tiny |
| Pretrained | ImageNet-1k (`ConvNeXt_Tiny_Weights.IMAGENET1K_V1`) |
| Optimizer | AdamW, weight decay 0.0001 |
| Learning rate | phase 1: 0.001 (head only); phase 2: 0.0001 (stage 4 + head) |
| Scheduler | cosine annealing over phase 2 (`CosineAnnealingLR`, T_max = 8) |
| Batch size | 24 (last incomplete training batch dropped) |
| Epochs | 2 head-only + at most 8 fine-tuning |
| Early stopping | patience 3 on validation macro-F1 |
| Loss | cross-entropy with class weights |
| Class weights | sklearn `balanced`, from the training folds only. Fold 0: N 0.4403, D 0.7858, G 4.4342, C 4.3329 |
| Seed | 42 + fold index |
| Augmentation | none |
| Input | 224 x 224, ImageNet normalisation |

Every value is the one used in the model-comparison experiment, which copied the original project's ResNet50 recipe. Nothing was tuned for this run.

## Protocol

For each test fold k: validation = fold (k + 1) mod 5, training = the remaining three folds. Patient overlap between the three sets is asserted to be zero before training. The best phase-2 epoch is chosen on validation macro-F1. The test fold is scored once, afterwards.

## Class imbalance

Handled by the loss weights above. No resampling and no rebalancing of validation or test data. Reported with macro-averaged metrics so the minority classes count equally.

## Augmentation

| Augmentation | Probability | Range | Reason |
|---|---|---|---|
| none | - | - | The original project's final run and the comparison used none. It is held fixed so the only change from the original project is the backbone. Adding augmentation is the most obvious way to improve the model and is left as future work. |

## What was run

Hardware: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 14 threads, CPU only, torch 2.14.0+cpu. Training time 76.4 minutes in total, after a one-off cache build.

| Test fold | Validation fold | Train / val / test images | Best epoch (phase 2) | Validation macro-F1 | Test macro-F1 | Temperature | Minutes |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 1 | 3033 / 1011 / 1012 | 7 | 0.6387 | 0.679 | 1.9582 | 14.9 |
| 1 | 2 | 3034 / 1011 / 1011 | 3 | 0.6739 | 0.63 | 1.3143 | 14.6 |
| 2 | 3 | 3034 / 1011 / 1011 | 7 | 0.6531 | 0.6587 | 1.8639 | 15.6 |
| 3 | 4 | 3034 / 1011 / 1011 | 7 | 0.6792 | 0.6789 | 1.8623 | 17.6 |
| 4 | 0 | 3033 / 1012 / 1011 | 3 | 0.6788 | 0.6627 | 1.1639 | 13.7 |

Curves: `results/training/training_history.png`; per-epoch values: `results/training/training_history.csv` (fold, phase, epoch, learning rate, training and validation loss, accuracy and macro-F1).

## The frozen-stem cache

Because the stem is frozen, in eval mode, and no augmentation is used, its output for an image never changes. It is computed once (about 1.5 GB in `data/cache/`) and stage 4 is trained from the cache. This is numerically the same as running the whole network every epoch and makes CPU training practical. The original project cached ResNet50 `layer3` outputs for the same reason. Evaluation of the deployed checkpoint does not use the cache: it runs the full network on image files.

## Checkpoints

- `checkpoints/convnext_tiny/fold_k.pt`: best validation epoch of each fold.
- `checkpoints/convnext_tiny/best_model.pth`: the deployed model, a copy of the best checkpoint of fold 0 (`model.deployed_fold`). The deployed fold is fixed in `configs/config.yaml` before training, not chosen by test performance.
- `checkpoints/convnext_tiny/last_model.pth`: the last epoch of the same fold (not selected; kept for reference, not committed).

## Commands

```bash
python scripts/train.py --smoke
python scripts/train.py
python scripts/train.py --skip-completed
```
