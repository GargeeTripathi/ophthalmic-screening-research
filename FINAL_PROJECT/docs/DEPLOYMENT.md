# Deployment

## Command-line inference

```bash
python scripts/predict.py --image path/to/fundus.jpg
python scripts/predict.py --image path/to/fundus.jpg --gradcam heatmap.png
```

Prints the predicted class, the temperature-scaled confidence and all four probabilities as JSON. The first run downloads the ImageNet ConvNeXt-Tiny weights (about 110 MB) for the frozen stem.

## Local web app

```bash
python scripts/serve.py                    # http://127.0.0.1:8000
python scripts/serve.py --port 8080 --no-browser --cases 0
```

Standard-library HTTP server; no web framework. Flow for each case:

```text
upload -> src/app/pipeline.py: ScreeningPipeline.screen()
            predict_image(image)                 src/inference/predict.py -> ConvNeXt-Tiny -> temperature scaling
            read_document(letter)   [optional]   src/idp: RapidOCR -> field extraction -> reliability estimate
            decision rules                       ACCEPT or DEFER
       -> src/app/render.py: worklist and case-review pages
```

Deferral rules (from `configs/config.yaml`): confidence below 0.6; the letter's stated diagnosis has probability below 0.25 under the image model; the letter was read with reliability below 0.5.

- With `ODIR_IMAGES` set, the app pre-screens held-out cases from the deployed model's test fold, paired with synthetic referral letters from `data/sample_referrals/`.
- Without the images it starts upload-only.
- Uploads are limited to 12 MB and stored under `data/uploads/` (not committed).

## Which model is served

Exactly one: the checkpoint at `model.checkpoint` in `configs/config.yaml`, loaded by `src/inference/predict.py:Predictor`. The CLI, the evaluation of the deployed model, Grad-CAM and the app all go through that class. There is no ResNet50 code or checkpoint in this project.

## Not re-measured for this model

The deferral thresholds and the deferral evaluation (risk-coverage, AURC) were established in the original project with its ResNet50-based model on 180 synthetic letters. They are carried over unchanged and have not been re-evaluated with ConvNeXt-Tiny. The OCR module and its reliability estimator are independent of the image model and are unchanged.

## Google Colab

`notebooks/final_results.ipynb` runs the full project. The dataset is found in Drive or downloaded from Kaggle; it is never committed.

## Verification

```bash
python scripts/verify_project.py
```
