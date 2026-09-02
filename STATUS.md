# Where the project stands — paused 2026-09-01

**The full local pipeline has run end to end.** Everything is on disk, every step is cached,
and `reports/RESULTS.md` is generated from `results/` rather than typed by hand.

---

## Resume

```bash
cd D:/ophthalmic-screening-research
.venv\Scripts\python.exe scripts/health_check.py     # HEALTHY
.venv\Scripts\python.exe -m pytest                   # 131 passed
.venv\Scripts\python.exe scripts/run_all.py --list   # what is done, what is not
```

Everything below is already done. `run_all.py` rebuilds anything missing and skips the rest;
`requirements.lock.txt` pins the exact versions this was measured on.

---

## What exists

| Layer | Output |
|---|---|
| Data | `odir_manifest.csv` (5,058 images / 2,910 patients, counts asserted), `merged_manifest.csv` (4,217, every image source-labelled), `odir_manifest_clean.csv` |
| Splits | 4 protocol rungs, patient-disjointness asserted |
| Dedup | calibrated detector, `duplicate_groups.csv`, comparison against the rejected embedding method |
| Experiments | E0a/b/c, E1, E1b, **E3**, E5, E7, E8, E9, E10, E11, E12, E13 — all in `results/` |
| IDP | 360 synthetic referrals, layouts D–F sealed and opened once, OCR cached |
| Figures | 14, in `figures/` (fig10 now includes the fine-tuned point) |
| Report | `reports/RESULTS.md`, `reports/PAPER_OUTLINE.md` |
| Demo | `reports/worklist.html` — self-contained, opens in any browser |
| Reproducibility | `scripts/run_all.py` (one command, cached stages), `requirements.lock.txt` (45 pinned versions) |
| Tests | 131 passing |

---

## Findings

### RQ1 — provenance: **confirmed, large**
Majority baseline 26.0%. A four-row source lookup table alone: 69.1%. A metadata-only
classifier: **66.7%**, without seeing a pixel. Every diabetic-retinopathy image is a `.jpeg`
and no other class contains one.

Within a single source the signal vanishes (`odir_jpg512`: 0.640 → 0.644, lift +0.004). So
provenance explains the DR and normal classes almost entirely and contributes little to
separating cataract from glaucoma. Both halves belong in the paper.

### RQ2 — patient leakage: **initially falsified, then confirmed by E1b**

E1 found nothing: macro-F1 0.5496 → 0.5444, McNemar p = 0.75. But that used a frozen linear
head, which cannot memorise — and leakage is exploited *by* memorisation.

E1b swept capacity instead of compute, holding features and splits fixed:

| model | inflation from ignoring patient identity | McNemar p |
|---|---:|---:|
| logistic | +0.0052 | 0.75 |
| mlp | +0.0274 | 0.20 |
| knn_5 | +0.0207 | **0.00065** |
| **knn_1 (pure retriever)** | **+0.0453** | **~0** |

**Patient leakage is real and is exploitable — by models that memorise.** A 1-NN retriever
gains +0.045 macro-F1 from ignoring patient identity, at p ≈ 0. E1's null was an artefact of
model choice, and the caveat written before the run turned out to be the finding. The
practical message: patient-disjoint splitting matters *more* as capacity rises.

### E3 — fine-tuned GPU confirmation: **done**

`colab/train_models.ipynb` ran on a T4. Two real problems surfaced and were fixed before it
could complete:
- **7 GB upload avoided.** `scripts/build_colab_bundle.py` resizes to 384px first, cutting
  the upload to 77 MB (91x smaller) - almost all the discarded resolution was thrown away
  by training anyway.
- **A silent bundling bug.** The bundle initially packed images from the *deduplicated*
  manifest while two of three splits reference the *raw* one, so two rows pointed at images
  the bundle did not contain - a `FileNotFoundError` that would only have surfaced mid-fold,
  after upload. Fixed by bundling the raw manifest (the `image_level` rung is *defined* by
  containing the duplicates) and the builder now verifies itself before reporting success.
- **Runtime-reset friction**, twice: `DRIVE_ROOT` undefined after a Drive remount, and
  Drive not actually mounted despite `drive.mount()` appearing to succeed. Both diagnosed
  live and fixed with a find-and-remount cell rather than blind retries.

Result: see RQ2 above. Confirms E1b on a real deployable model.

### E5 — cross-source transfer: **the model does not transfer**

| source | classes | majority-class acc | model acc |
|---|---|---:|---:|
| crop_jpg256 | C,G | 0.517 | 0.330 |
| eyepacs_jpeg | D | 1.000 | 0.297 |
| png_highres | C,G | 0.502 | 0.104 |
| odir_jpg512 | C,G,N | 0.449 | 0.507 |

Below majority-class guessing on three of four unseen acquisition sources. Caveat: the
in-domain model is itself weak (macro-F1 0.55), so this is partly a weak model and not purely
a domain gap.

### E11 — is it a disease detector or a demographics detector? **Disease.**

The 11-year age gap between cataract and DR patients has been flagged since the first audit.
It is now tested rather than merely disclaimed.

| condition | n | macro-F1 | cataract AUROC |
|---|---:|---:|---:|
| unrestricted | 5,056 | 0.5476 | 0.9462 |
| age-matched | 964 | 0.5385 | **0.9414** |
| sex-matched | 972 | 0.5302 | **0.9406** |
| multi-finding removed | 4,679 | 0.5478 | 0.9560 |

Every CI overlaps the unrestricted one. And the sharpest control: **demographics alone score
0.556 against a 0.568 majority baseline** — age and sex carry no usable class signal at all,
performing *worse* than always guessing. The confound is real in the data but is not what the
model is using.

### RQ3 — OCR confidence: **confirmed, and the mechanism is visible**
| Signal | E1 AUROC | E2 AUROC |
|---|---:|---:|
| Raw OCR confidence | 0.52 | 0.64 |
| Learned estimator | **0.98** | **0.99** |

Raw OCR confidence is close to chance at predicting whether a clinical field was extracted
correctly. Feature importances explain it: `format_valid` 0.55 and `value_length` 0.31
dominate; `ocr_confidence` contributes 0.08. Semantic extraction errors survive high OCR
confidence, which is exactly why the raw signal fails.

### RQ5 — document evidence for deferral: **confirmed**
| System | AURC ↓ | error-detection AUROC | acc@50% coverage |
|---|---:|---:|---:|
| image confidence only | 0.389 | 0.588 | 0.589 |
| **System A (rules)** | **0.197** | **0.882** | 0.867 |
| System B (learned) | 0.211 | 0.869 | 0.889 |
| System B without `c_hat` | 0.203 | 0.871 | 0.867 |

Document evidence roughly halves AURC and lifts error detection from 0.59 to 0.88.

### E12 — explainability: **passes, with narrow claims**

| measure | value |
|---|---:|
| FOV-mass ratio | **0.836** |
| uniform-attribution baseline | 0.535 |
| concentration ratio | **1.56x** |
| deletion AUC | 0.469 |
| randomisation sanity check | **PASS** (r = −0.094) |

Attribution concentrates on retinal tissue at 1.56× what uniform attribution would give, and
the maps depend on the learned weights. That is all it licenses: it answers a *shortcut*
question, not a clinical one. No lesion masks, no ophthalmologist, so "the model looks at the
lesion" stays forbidden.

The head is fitted on training folds and attribution runs only over held-out fold 0 — an
earlier version attributed over images the head had been fitted on, which was in-sample and
wrong.

### E13 — seed stability and multiple comparisons

Leakage inflation, re-run across three split seeds:

| model | mean inflation | sd | range |
|---|---:|---:|---|
| **knn_1** | **+0.0453** | 0.0027 | [+0.0434, +0.0484] |
| logistic | −0.0031 | 0.0085 | [−0.0126, +0.0038] |

The memoriser's effect is large, tight and significant at every seed. The linear head's
hovers around zero and **flips sign** between seeds — noise, not signal.

Holm-Bonferroni over all 13 paired tests: **5 survive, and all five are the
memorisation-based ones.** The instructive failure is `leakage[logistic,seed7]`, raw
p = 0.037 — it would have been reported as "significant" uncorrected, and its effect was
*negative*. That single row is the argument for correcting.

### RQ4 / the architectural claim: **killed by the pre-registered rule**
System A (two rules, no learning) beats System B (learned gate). Removing `c_hat` from
System B makes it *better*, not worse. The kill criterion written before the run said: if the
learned gate does not beat the rules, ship the rules and drop the extraction-confidence claim.

**So that is what the paper says.** The learned deferral gate and the reliability-estimate
input are reported as a negative result. `c_hat` remains valuable for RQ3 — it is an excellent
*correctness* estimator — it simply adds nothing to the *deferral* decision on top of image
confidence and document agreement.

Caveat: only 180 sealed rows, so an AURC gap of 0.01 is well within noise. The honest
statement is "the learned gate did not improve on the rules", not "the rules are better".

---

## Needs a GPU, so needs you

`colab/train_models.ipynb` — E2 backbone selection, E3 fine-tuned baselines.

Upload to `MyDrive/ophthalmic-screening-research/`:
- `data/manifests/odir_manifest_clean.csv`
- `data/splits/splits_*.csv`
- ODIR `Training Images` → `data/raw/odir_images/` (~7 GB, one-time)

Then bring `results/E3_predictions_*.csv` back and re-run `protocol_ladder.py` and
`make_figures.py`.

**Nothing is blocked on it any more.** Two questions that looked GPU-bound were answered
on CPU instead:

- **RQ2** — E1b swept memorisation capacity rather than compute.
- **E12, the XAI battery** — Grad-CAM needs gradients through the backbone, not *fine-tuned*
  backbone weights, so the frozen ResNet50 plus a linear head fitted on held-out folds
  supports the full battery.

The fine-tuned baselines are now about absolute performance and comparability with published
numbers, not about settling any research question.

Also worth knowing: the worklist demo currently defers **24 of 24** cases, because the
frozen-head model is rarely above 60% confident. That is the deferral rule behaving correctly
on a weak model, not a bug — it should fall sharply once the fine-tuned model replaces it.

---

## Corrections made along the way, recorded so they are not repeated

1. **Duplicate threshold.** Calibrating on a *balanced* set gave 0.98 at apparent precision
   0.999. Applied to the real 12.79M-pair search it returned 26,634 pairs, chaining into
   clusters covering 37% of the dataset. Precision on a balanced sample does not transfer to a
   search where positives are one in a million. Recalibrated against hard negatives; operating
   point 0.999, returning 2 within-ODIR pairs.
2. **Embeddings failed as a duplicate detector** and were replaced by FOV-normalised
   fingerprints. Kept as a reported methods note: 0% recall at 0.999 vs 97% at precision 0.99.
3. **The chatbot claimed calibration that had not happened.** It said confidences were
   temperature-scaled; they are not, until the GPU baselines run.
4. **OCR engine changed from Tesseract to RapidOCR** — pip-installable, CPU, offline, no
   system binary and no admin rights, which keeps the project reproducible in a clean
   environment.

All four are documented in the source files themselves, not only here.
