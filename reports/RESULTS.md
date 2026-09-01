# Results

Every number here is read from `results/` by `scripts/summarise_results.py`. Nothing is transcribed by hand.

**Not a medical device. Synthetic documents are not clinical data. No clinical validation is claimed.**

## 1. Dataset

ODIR-5K, four-class subset: **5058 images** from **2910 patients**.

| class | images | patients | multi-finding | mean age |
|---|---:|---:|---:|---:|
| C | 293 | 203 | 31 (10.6%) | 66.8 |
| D | 1608 | 1027 | 223 (13.9%) | 55.7 |
| G | 284 | 182 | 66 (23.2%) | 63.0 |
| N | 2873 | 1852 | 57 (2.0%) | 57.2 |

- **2148 patients contribute both eyes** - the mechanism behind patient-level leakage.
- 354 patients carry different diagnoses in each eye, so grouping is by patient but stratification is at eye level.
- 91 distinct resolutions, no EXIF: ODIR provenance cannot be labelled cleanly.

**Confounds to report, not hide.** Cataract patients average ~11 years older than diabetic-retinopathy patients; sex distribution differs by class; and 23.2% of glaucoma rows name a second condition in the annotation. ODIR is *less confounded*, not clean.

## 2. Provenance audit (RQ1)

The merged four-class benchmark is not one dataset. Acquisition source is assigned from file properties alone - never from the class folder - and it almost determines the label.

- Majority-class baseline: **26.0%**
- Source lookup table alone: **69.1%** (no model, no pixels)

| experiment | question | baseline | probe | lift |
|---|---|---:|---:|---:|
| E0a | class from file metadata only | 0.260 | **0.667** | +0.407 |
| E0b | source from FOV-normalised pixels | 0.398 | **0.749** | +0.351 |
| E0c-i | source from geometry only | 0.398 | **0.702** | +0.304 |
| E0c-ii[odir_jpg512] | predict class from pixels within source=odir_jpg512 | 0.640 | **0.644** | +0.004 |
| E0c-ii[crop_jpg256] | predict class from pixels within source=crop_jpg256 | 0.516 | **0.736** | +0.219 |
| E0c-ii[png_highres] | predict class from pixels within source=png_highres | 0.497 | **0.781** | +0.284 |

**The anti-circularity control (E0c-ii).** Within a single acquisition source, provenance is constant, so anything a probe achieves must come from retinal content:

- `odir_jpg512` (n=1678): 0.640 -> 0.644, lift **+0.004**
- `crop_jpg256` (n=1240): 0.516 -> 0.736, lift **+0.219**
- `png_highres` (n=201): 0.497 -> 0.781, lift **+0.284**

Every diabetic-retinopathy image is a `.jpeg` and no other class contains one, so file extension alone is a perfect DR detector on this benchmark. The cataract/glaucoma sources split roughly 50/50, so provenance explains the DR and normal classes almost entirely and contributes little to separating cataract from glaucoma. Both halves of that belong in the paper.

## 3. Deduplication

Detector: FOV-normalised 64x64 fingerprint, cosine similarity. Rejected alternative: ImageNet ResNet50 embedding cosine - see calibration table.

- Ground truth: **1654 known duplicate pairs** (a merged-benchmark image sharing a filename with an ODIR image), calibrated against the hardest available negatives.
- Operating threshold **0.999**: precision **0.9895**, recall **0.9728**.
- Within ODIR: **2 duplicate groups**, 2 images removed -> 5056 images, 2909 patients.
- **1626 merged-benchmark images are ODIR copies** and are therefore excluded from cross-source test data.

*Caveat carried into the paper:* recall is measured over ODIR<->merged ground-truth pairs; recall for within-ODIR duplicates is not directly measurable.

A methods note worth publishing on its own: ImageNet-embedding cosine similarity **fails** as a duplicate detector on fundus images. At threshold 0.999 it recovers 0% of known duplicates, while the FOV-normalised fingerprint recovers 97%. Fundus images are visually homogeneous and the copies differ by a large downscale, which ImageNet features are not invariant to.

## 4. Protocol ladder (RQ2)

Model held fixed, only the split varies. Frozen ResNet50 features with a logistic head.

| rung | n | accuracy | macro-F1 | 95% CI |
|---|---:|---:|---:|---|
| image_level | 5058 | 0.5975 | 0.5496 | [0.5308, 0.5677] |
| patient_disjoint | 5058 | 0.5951 | 0.5444 | [0.5248, 0.5622] |
| patient_disjoint_dedup | 5056 | 0.5949 | 0.5476 | [0.5282, 0.5675] |
| provenance_controlled | 1535 | 0.6098 | 0.5048 | [0.4735, 0.5383] |

| comparison | macro-F1 drop | McNemar p |
|---|---:|---:|
| image_level -> patient_disjoint | +0.0052 | 0.746733 |
| image_level -> patient_disjoint_dedup | +0.0019 | 0.750243 |
| image_level -> provenance_controlled | +0.0447 | 0.143081 |

**Result: H2 is falsified as run.** No rung differs significantly from any other; all confidence intervals overlap.

**The limitation that must accompany it.** A frozen linear head has almost no capacity to memorise, and leakage inflates a score by letting a model recall a near-copy it has already seen. This therefore measures a *lower bound* on protocol inflation, not the inflation a fine-tuned backbone would show. Settling that requires the GPU baselines in `colab/train_models.ipynb`.

## 4b. Does inflation depend on memorisation? (E1b)

E1's null invites an obvious objection: a frozen linear head cannot memorise, and leakage is exploited *by* memorisation. So rather than reaching for a GPU, sweep the variable that actually matters - capacity - holding features and splits fixed. 1-NN is a pure retriever: if a patient's fellow eye is in the training set, it will find it.

| model | image-level | patient-disjoint | inflation | McNemar p |
|---|---:|---:|---:|---:|
| logistic | 0.5496 | 0.5444 | **+0.0052** | 0.746733 |
| mlp | 0.5843 | 0.5569 | **+0.0274** | 0.200594 |
| knn_5 | 0.5250 | 0.5044 | **+0.0207** | 0.000646 |
| knn_1 | 0.5576 | 0.5124 | **+0.0453** | 0.0 |

**Verdict: inflation grows with memorisation capacity - E1's null was an artefact of using a low-capacity head.**

This corrects E1. Patient leakage on this dataset **is** real and **is** exploitable - by models that memorise. A pure retriever gains +0.045 macro-F1 from ignoring patient identity, at p ~ 0. The parametric models show smaller, non-significant gains. The practical implication for anyone using this benchmark is that patient-disjoint splitting matters more as model capacity rises, which is precisely the direction the field is moving.

## 4c. Transfer to unseen acquisition sources (E5)

Trained on 5056 ODIR images; tested on merged-benchmark images after removing 1626 confirmed ODIR copies, so nothing in the test set was seen in training.

| source | n | classes | majority-class acc | model acc | macro-F1 |
|---|---:|---|---:|---:|---:|
| crop_jpg256 | 1223 | C,G | 0.517 | **0.330** | 0.228 |
| eyepacs_jpeg | 1098 | D | 1.000 | **0.297** | 0.115 |
| odir_jpg512 | 69 | C,G,N | 0.449 | **0.507** | 0.412 |
| png_highres | 201 | C,G | 0.502 | **0.104** | 0.093 |

**The model does not transfer.** On three of four sources it scores *below* the accuracy of always guessing that source's majority class. Each source is class-restricted - itself a symptom of the provenance confound - so results are reported per source and never pooled.

Two caveats. The in-domain model is itself weak (macro-F1 0.55 on frozen features), so poor transfer is partly a weak model rather than purely a domain gap. And this measures transfer under a documented change of acquisition source, not *generalisation* in the unqualified sense.

## 4d. Is it a disease detector or a demographics detector? (E11)

ODIR is less confounded, not clean: cataract patients average 11 years older than diabetic-retinopathy patients, and the sex distribution differs by class. Networks read age and sex from fundus photographs accurately, so this is testable rather than merely a limitations-section caveat.

**Demographics alone, no image:** accuracy 0.5560 against a 0.5678 majority baseline (lift -0.0119), macro-F1 0.2531. Age and sex carry no usable class signal here - the classifier does *worse* than always guessing the majority class.

| condition | n | patients | macro-F1 | 95% CI | cataract AUROC |
|---|---:|---:|---:|---|---:|
| unrestricted | 5056 | 2909 | 0.5476 | [0.5283, 0.5668] | **0.9462** |
| age_matched | 964 | 779 | 0.5385 | [0.5030, 0.5718] | **0.9414** |
| sex_matched | 972 | 792 | 0.5302 | [0.4999, 0.5619] | **0.9406** |
| multi_finding_removed | 4679 | 2738 | 0.5478 | [0.5255, 0.5682] | **0.9560** |

- age_matched              macro-F1 -0.0091 vs unrestricted; CIs overlap
- sex_matched              macro-F1 -0.0174 vs unrestricted; CIs overlap
- multi_finding_removed    macro-F1 +0.0002 vs unrestricted; CIs overlap

**Image performance survives every control.** Cataract-vs-rest AUROC stays near 0.94 under age matching and sex matching, and every confidence interval overlaps the unrestricted one. The demographic confound is real in the data but is not what the model is using.

*Method:* matching is by downsampling within strata, so it closes the demographic route without reweighting or modelling.

## 5. Document extraction (E7)

Two extractors, developed on layouts A-C and evaluated on sealed layouts D-F which were opened once, after both were frozen.

| extractor | split | precision | recall | F1 | exact match | n |
|---|---|---:|---:|---:|---:|---:|
| E1 | dev | 0.696 | 0.534 | **0.604** | 0.568 | 1620 |
| E1 | sealed | 0.726 | 0.235 | **0.355** | 0.297 | 1620 |
| E2 | dev | 0.687 | 0.679 | **0.683** | 0.691 | 1620 |
| E2 | sealed | 0.563 | 0.430 | **0.488** | 0.453 | 1620 |

*synthetic documents; a technical evaluation under controlled degradation, not clinical validation.*

## 6. Reliability of the extraction signal (RQ3)

Is raw OCR confidence a usable estimate of whether an extracted clinical field is correct? Trained on development layouts, evaluated on sealed ones.

| extractor | signal | AUROC | Brier | ECE |
|---|---|---:|---:|---:|
| E1 | raw ocr confidence | **0.5214** | 0.2473 | 0.2254 |
| E1 | learned estimator | **0.9799** | 0.0542 | 0.0294 |
| E2 | raw ocr confidence | **0.6389** | 0.3806 | 0.3854 |
| E2 | learned estimator | **0.9871** | 0.0427 | 0.0151 |

## 7. Deferral (RQ4, RQ5)

**The document never changes the predicted class.** `argmax(p_img)` is the answer whenever the system answers; the document and its read-reliability only decide *whether* to answer.

| system | AURC | error-detection AUROC | acc@90% | acc@75% | acc@50% |
|---|---:|---:|---:|---:|---:|
| image_confidence_only | **0.3889** | 0.5875 | 0.5432 | 0.5481 | 0.5889 |
| system_A_rules | **0.1973** | 0.8817 | 0.5741 | 0.6593 | 0.8667 |
| system_B_learned | **0.2107** | 0.8692 | 0.5679 | 0.6444 | 0.8889 |
| system_B_without_c_hat | **0.2032** | 0.8705 | 0.5802 | 0.637 | 0.8667 |

Lower AURC is better. `system_B_without_c_hat` isolates what the learned extraction-reliability estimate contributes - the one architectural claim this project makes.

## 7b. Explainability (E12)

Model: frozen ImageNet ResNet50 + linear head on cached embeddings. Head fitted on 4044 training-fold embeddings; attribution over held-out fold 0 - the head never saw these images.

| measure | value |
|---|---:|
| FOV-mass ratio (observed) | **0.8364** |
| uniform-attribution baseline | 0.5348 |
| concentration ratio | **1.564x** |
| deletion AUC (mean) | 0.4687 |
| randomisation sanity check | **PASS** (r = -0.0943) |

Attribution concentrates inside the retinal field of view at 1.56x what uniform attribution would give, and the method passes the randomisation sanity check, so the maps do depend on the learned weights.

**What this permits us to say:** attribution falls on retinal tissue rather than on borders or letterbox artefacts. **What it does not permit:** any claim that the model is looking at a lesion. There are no lesion masks and no ophthalmologist. The FOV-mass ratio answers a shortcut question, not a clinical one.

*Caveat:* masking creates out-of-distribution inputs, so deletion AUC conflates attribution quality with behaviour on corrupted images. And not a fine-tuned diagnostic model; re-run after the Colab baselines.

## 9. Statistical validation (E13)

**Seed stability.** Every earlier result used split seed 42. The leakage comparison is re-run across [42, 7, 2024]; a finding that holds for only one random partition is not a finding.

| seed | model | image-level | patient-disjoint | inflation | p |
|---|---|---:|---:|---:|---:|
| 42 | logistic | 0.5514 | 0.5476 | **+0.0038** | 0.619208 |
| 42 | knn_1 | 0.5585 | 0.5101 | **+0.0484** | 0.0 |
| 7 | logistic | 0.5450 | 0.5576 | **-0.0126** | 0.036894 |
| 7 | knn_1 | 0.5531 | 0.5091 | **+0.0440** | 0.0 |
| 2024 | logistic | 0.5500 | 0.5505 | **-0.0005** | 0.840133 |
| 2024 | knn_1 | 0.5589 | 0.5156 | **+0.0434** | 0.0 |

| model | mean inflation | sd | min | max |
|---|---:|---:|---:|---:|
| knn_1 | +0.0453 | 0.0027 | +0.0434 | +0.0484 |
| logistic | -0.0031 | 0.0085 | -0.0126 | +0.0038 |

**Multiple comparisons.** The project reports 13 paired tests. At alpha = 0.05 one spurious result in twenty is expected by construction, so Holm-Bonferroni is applied across the whole family. **5 of 13 survive correction:**

- `leakage[knn_1,seed42]` - p = 0.0 (threshold 0.003846)
- `leakage[knn_1,seed7]` - p = 0.0 (threshold 0.004167)
- `leakage[knn_1,seed2024]` - p = 0.0 (threshold 0.004545)
- `E1b[knn_1]` - p = 0.0 (threshold 0.005)
- `E1b[knn_5]` - p = 0.000646 (threshold 0.005556)

*Holm rather than plain Bonferroni: uniformly more powerful at the same family-wise error rate, which matters when the smallest class has 182 patients.*

## 8. What is still outstanding

- **Fine-tuned baselines (E2/E3)** need a GPU: `colab/train_models.ipynb`. Until they run, the protocol-ladder result is a lower bound.
- **XAI battery (E12)** depends on a fine-tuned model to attribute over.
- **A real-document sanity check.** No suitable public de-identified ophthalmic referral corpus was found; the search is recorded, and the corpus remains synthetic.
