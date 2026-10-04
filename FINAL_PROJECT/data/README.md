# Data

## What is in this folder

| Path | Content |
|---|---|
| `manifests/splits_patient_disjoint_dedup.csv` | The split used for every result: 5,056 images, 2,909 patients, 5 folds, grouped by patient. Copied unchanged from the original project (only the machine-specific path column was dropped). |
| `manifests/splits_summary.json`, `manifests/deduplication_report.json` | How the split and the duplicate removal were produced in the original project. |
| `sample_referrals/` | 39 synthetic referral letters (generated, not real patients) used by the app's demo queue. |
| `cache/` | Created by training: resized images and frozen-stem features. Not committed. |

## The fundus images are not included

ODIR-5K is public but not ours to redistribute. Download it from Kaggle:

    https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

Then tell the project where the training JPEGs are (the folder containing `0_left.jpg`,
`0_right.jpg`, ...), in one of three ways:

1. environment variable `ODIR_IMAGES`
2. a `.env` file in the project root (copy `.env.example`)
3. `data.images_dir` in `configs/config.yaml`

Images are matched to the split by file name, so no path in the split file needs editing.

## Classes

| Code | Class | Images |
|---|---|---:|
| N | normal | 2,871 |
| D | diabetic retinopathy | 1,608 |
| C | cataract | 293 |
| G | glaucoma | 284 |

## Leakage safeguards (inherited, and re-checked before every training run)

- Folds are grouped by `patient_id`: both eyes of a patient are always in the same fold.
- 0 patients appear in more than one fold; 0 duplicate image ids (asserted in code).
- 2 duplicate images were removed in the original project by a calibrated fingerprint detector.
- For test fold k, validation is fold (k + 1) % 5 and training is the other three folds.
