# Model

## Architecture

ConvNeXt-Tiny (Liu et al. 2022), torchvision implementation, weights `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`.

torchvision's model is `features` (8 modules) -> `avgpool` -> `classifier`, with `classifier = LayerNorm2d(768) -> Flatten -> Linear(768, 1000)`.

| torchvision module | Role | Output (224 px input) | In this project |
|---|---|---|---|
| `features[0]` | 4 x 4 patchify stem + LayerNorm | 96 x 56 x 56 | frozen |
| `features[1]` | stage 1 | 96 x 56 x 56 | frozen |
| `features[2]`, `[3]` | downsample, stage 2 | 192 x 28 x 28 | frozen |
| `features[4]`, `[5]` | downsample, stage 3 | 384 x 14 x 14 | frozen |
| `features[6]`, `[7]` | downsample, stage 4 | 768 x 7 x 7 | **trained** |
| global average pool | | 768 | |
| `LayerNorm(768)` | | 768 | **trained (new)** |
| `Linear(768, 4)` | | 4 logits | **trained (new)** |

Parameters: 27.82 M in total, 15.48 M trained, 12.35 M frozen.

## Why the head is not a one-line replacement

A ResNet ends in `avgpool -> fc`, so replacing `fc` is enough. ConvNeXt has no `fc`; its blocks emit un-normalised activations and rely on the LayerNorm inside the classifier. The head here keeps that normalise-then-classify structure with a new 4-class Linear layer. LayerNorm2d before pooling and LayerNorm after pooling are not numerically identical; the comparison experiment used LayerNorm after pooling, and this project keeps that so results match the experiment that selected the model.

## Frozen stem

The stem is frozen and always in eval mode (`train()` is overridden), the same partial fine-tuning as the original project (ResNet50 `conv1` to `layer3` frozen, `layer4` trained).

## Checkpoint format

`checkpoints/convnext_tiny/best_model.pth` is a dictionary:

- `state`: the trained tensors only (stage 4, LayerNorm, Linear), half precision, about 30 MB.
- `meta`: architecture, labels, class names, input size, mean / std, test and validation fold, best epoch, validation macro-F1, temperature.

The frozen stem is not stored: it is the unchanged ImageNet weights, which torchvision downloads on first use (about 110 MB, cached). `src/models/convnext_tiny.py:load_checkpoint` rebuilds the model and asserts that only stem keys are absent.

## Sanity check before training

`scripts/train.py` first runs `sanity_check()`: input (2, 3, 224, 224) -> stem output (2, 384, 14, 14) -> logits (2, 4); finite outputs and loss; gradients on every trained parameter and none on the stem; one optimiser step changes the head. Result of the recorded run: PASS.
