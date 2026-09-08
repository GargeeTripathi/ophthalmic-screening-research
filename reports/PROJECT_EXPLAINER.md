# Project Explainer

*One document, top to bottom: what this is, the gap it addresses, why that gap
matters, how every piece was built, what actually happened when it ran, and what
every result means. Numbers are copied from `reports/RESULTS.md` (itself
generated from `results/`, never typed by hand); process history is copied from
`STATUS.md`. Nothing here is a new claim — this is the connective narrative those
two documents don't provide on their own.*

---

## 0. The one-paragraph version

Public four-class fundus-photo benchmarks (cataract / diabetic retinopathy /
glaucoma / normal) are widely reported at 95–96% accuracy. This project asks how
much of that is retinal pathology and how much is an artefact of *how the
benchmark was built* — which file a patient's eye photo came from, and whether
the same patient's other eye leaked into both train and test. It re-evaluates
standard classifiers under a corrected protocol, finds both effects are real,
confirms the second one on an actual fine-tuned deployable model rather than a
toy, and then builds a working screening prototype whose *decision to defer a
case* is informed by a scanned referral document — while the disease prediction
itself is kept image-only, by construction, not by convention. It is an
empirical audit plus a working prototype, not a new architecture, and it says so
in its own README.

---

## 1. The gap

Three separate problems motivate this project. Each one is a specific, testable
claim about existing practice, not a vague "the field has issues."

### 1.1 Provenance confound

Four-class ODIR-style benchmarks are usually assembled by merging images from
several acquisition sources — different cameras, different export pipelines,
different downstream JPEG/PNG conversions — into one folder per disease class.
If acquisition source correlates with class (which it usually does, because
different sources were curated for different diseases), a classifier can score
well by learning "which pipeline produced this file" instead of "what disease is
in this eye." This is measurable, not hypothetical: file extension, resolution,
and pixel-level artefacts of the export pipeline are all detectable *without
looking at the retina at all.*

### 1.2 Patient-level leakage

Most patients contribute two eye photos (left and right). If a dataset is split
by *image* rather than by *patient*, one eye can land in the training set and
the fellow eye in the test set. A model with enough capacity to memorise — not
just to generalise — can then get credit for recognising *the same patient*
rather than *the disease pattern*. Whether this actually inflates reported
accuracy depends on whether the model in question has the capacity to memorise,
which is exactly the kind of thing that gets skipped when a paper only reports
one architecture.

### 1.3 The deployability gap

A classifier that must answer on every image is not a screening tool a clinician
can trust — it will confidently misclassify the images it's worst at, with no
signal that it should have been sent to a human instead. Building a system that
knows when to defer requires a second signal beyond image confidence, and a
referral document (symptoms, history, prior findings) is the natural place to
get one. But if that document is allowed to influence the diagnosis itself, the
system can be gamed or can silently drift into being a text classifier wearing
an image classifier's clothes. And once you introduce a machine-read document as
input, you inherit its own reliability problem: OCR confidence is not the same
thing as "was this clinical field extracted correctly."

These three gaps map directly onto the five research questions in the README,
and the falsification criterion for each was written into `configs/experiments.yaml`
*before* the corresponding experiment ran — the standing rule for this project,
recorded before any implementation started, was: audit first, define what would
prove the idea wrong before building it, and report the negative result if that's
what happens.

---

## 2. Why it matters

If provenance and leakage inflate reported numbers on the benchmarks the field
uses to compare methods, then architecture comparisons built on those numbers
are partly comparing "which model overfits the leak hardest," not "which model
understands the disease better." That's a methodological problem independent of
any single paper's conclusions. And if a screening prototype can't say "I'm not
sure, refer this one," it's not something a clinic could respons­ibly deploy,
regardless of how good its raw accuracy looks. Both problems are fixable with
existing techniques (patient-disjoint splitting, selective prediction) — this
project's contribution is measuring the size of the first problem precisely
enough to say whether it matters, and building the second as a real,
structurally-enforced system rather than an accuracy number with an asterisk.

---

## 3. What this project actually claims — and doesn't

From `README.md`, verbatim in spirit: **this is not a novel CNN architecture.**
The image model is a standard ResNet50, correctly evaluated. Every mechanism
used (selective prediction, reliability-weighted fusion, Grad-CAM++) is adapted
from prior work, cited as such. The five contributions are all labelled
*empirical* or *resource*, never *methods*:

| # | Contribution | Type |
|---|---|---|
| 1 | Acquisition-provenance and patient-leakage audit of a widely used four-class fundus benchmark | Empirical |
| 2 | Corrected patient-disjoint, deduplicated, provenance-controlled re-evaluation | Empirical |
| 3 | Calibration study of extraction-correctness estimators for clinical field extraction | Empirical |
| 4 | Controlled analysis mapping referral legibility and correctness onto triage quality | Empirical |
| 5 | Released synthetic ophthalmic referral extraction corpus with ground truth and sealed layouts | Resource |

Two structural rules hold the whole system honest, and both are enforced by
code, not by convention:

1. **The document never changes the diagnosis.** `ScreeningPipeline.predict_image()`
   takes only an image path — verified by a test that inspects the function's
   signature (`tests/test_pipeline_inference.py::test_predict_image_does_not_accept_a_document`),
   so a referral document *cannot* influence the class prediction even by accident.
   It can only influence whether the system defers.
2. **No patient appears in two splits.** Enforced by an assertion in
   `src/data/splits.py`, not by "we tried to be careful."

---

## 4. The dataset

ODIR-5K, four-class subset (cataract / diabetic retinopathy / glaucoma / normal):
**5,058 images from 2,910 patients.**

| class | images | patients | multi-finding | mean age |
|---|---:|---:|---:|---:|
| Cataract | 293 | 203 | 10.6% | 66.8 |
| Diabetic retinopathy | 1,608 | 1,027 | 13.9% | 55.7 |
| Glaucoma | 284 | 182 | 23.2% | 63.0 |
| Normal | 2,873 | 1,852 | 2.0% | 57.2 |

The two facts that make the whole audit possible: **2,148 patients contribute
both eyes** (the mechanism behind patient-level leakage), and the dataset
contains **91 distinct image resolutions with no EXIF data**, so provenance
cannot be read off a clean metadata field — it has to be *inferred*, which is
exactly what Experiment 0 does.

Confounds are stated up front rather than left for a limitations section:
cataract patients are ~11 years older on average than diabetic-retinopathy
patients, sex distribution differs by class, and 23.2% of glaucoma rows name a
second condition. ODIR is *less confounded than a naive merge*, not clean — and
Experiment 11 exists specifically to test whether the model is exploiting the
age/sex confound.

---

## 5. Pipeline overview — how the project is built

```
data/raw (read-only, outside repo)
   │
   ▼
manifest.py ──► odir_manifest.csv (5,058 rows, counts asserted)
   │
   ├─► fingerprint.py ──► FOV-normalised 64×64 fingerprints
   │        │
   │        ▼
   │   dedup.py ──► calibrated duplicate detector ──► duplicate_groups.csv
   │
   ├─► provenance/features.py ──► acquisition-source features (file-property based)
   │
   └─► splits.py ──► 4 protocol rungs, patient-disjointness asserted
                        │
                        ▼
        E0a/b/c, E1, E1b, E5, E11, E13  (CPU, frozen ResNet50 features)
        E3                              (GPU, Colab, fine-tuned ResNet50)
        E12                             (XAI, Grad-CAM++ on frozen model)

idp/generator.py ──► 360 synthetic referrals (layouts A–F, D–F sealed)
   │
   ▼
idp/ocr.py (RapidOCR) ──► idp/extract.py (2 extractors) ──► E7, RQ3 reliability estimator
                                                                    │
                                                                    ▼
deferral/systems.py ──► E9 (rules vs learned gate), E10 (robustness)

evaluation/metrics.py ──► McNemar, bootstrap CI, Holm-Bonferroni — used by every experiment above

app/pipeline.py + app/chatbot.py + app/render.py ──► working prototype
   │
   ▼
scripts/export_model.py ──► deployable checkpoint
scripts/serve.py (local) / docs/ (Vercel static site)
```

Every stage writes its output to disk and declares it; `scripts/run_all.py` is
the single entry point that runs whatever hasn't already run and skips the rest
— a crash costs only the stage that was in progress, and the whole local
pipeline (everything except the GPU stage) reproduces from cold in about 70
minutes on a 14-thread CPU.

---

## 6. Deep walkthrough of every experiment

### 6.1 Provenance audit — E0a / E0b / E0c (RQ1)

**Question:** how much of the reported accuracy on a merged benchmark comes from
*which file the image is*, not what's in it?

**Method:** acquisition source is labelled from file properties alone (never
from the class folder, which would make the label circular), then three probes
are run:

- **E0a** — predict class from file metadata only (extension, resolution). No
  pixels at all.
- **E0b** — predict *source* from FOV-normalised pixels. Tests whether source
  leaves a pixel-level signature even after cropping to the retinal field.
- **E0c** — two sub-probes: (i) predict source from image *geometry* only
  (aspect ratio, crop shape); (ii) the anti-circularity control — predict
  *class* from pixels **within a single source**, where source is held
  constant, so any success must come from retinal content, not provenance.

**Results:**

| probe | baseline | result | what it means |
|---|---:|---:|---|
| E0a: class from metadata | 0.260 | **0.667** | file extension/resolution alone gets two-thirds of the way to perfect |
| E0b: source from pixels | 0.398 | **0.749** | source leaves a detectable pixel signature even after FOV cropping |
| E0c-i: source from geometry | 0.398 | **0.702** | crop shape alone is a strong source detector |
| E0c-ii within `odir_jpg512` | 0.640 | 0.644 | signal **vanishes** — lift +0.004 |
| E0c-ii within `crop_jpg256` | 0.516 | 0.736 | signal **survives** — lift +0.219 |
| E0c-ii within `png_highres` | 0.497 | 0.781 | signal **survives** — lift +0.284 |

**Interpretation, both halves reported honestly:** every diabetic-retinopathy
image in this benchmark is a `.jpeg` and no other class contains one, so file
extension is a perfect DR detector with zero retinal information. That's why
E0a's lift is so large. But cataract and glaucoma are split roughly evenly
across sources, so within a fixed source the model still separates them from
pixels — the anti-circularity control (E0c-ii) shows the cataract/glaucoma
signal is *not* purely provenance. **Provenance explains the DR-vs-normal split
almost entirely, and explains little of cataract-vs-glaucoma.** A paper that
reported only the aggregate number would have missed this asymmetry.

### 6.2 Deduplication

**Method chosen, and the one that was rejected first:** the original approach —
cosine similarity between ImageNet ResNet50 embeddings — was tried first and
**failed outright**: at threshold 0.999 it recovered **0%** of a known set of
1,654 duplicate pairs (a merged-benchmark image sharing a filename with an ODIR
image). Fundus images are visually homogeneous to begin with, and the actual
duplicates differ by a large downscale that ImageNet features are not invariant
to. The fix was a domain-specific detector: an FOV-normalised 64×64 pixel
fingerprint compared by cosine similarity, which recovers **97%** of the same
ground truth at precision 0.99.

**A second miscalibration, caught before it did damage:** the fingerprint
threshold was first calibrated on a *balanced* synthetic sample and looked
perfect (precision 0.999) — but applied to the real search space (12.79 million
candidate pairs, positives roughly one in a million) it returned 26,634 false
pairs, chaining into duplicate clusters covering 37% of the dataset. The lesson,
recorded permanently in `STATUS.md`: **precision measured on a balanced sample
does not transfer to a search where positives are extremely rare** — you must
calibrate against hard negatives drawn from the true search space. Recalibrated
threshold: 0.999, giving 2 genuine within-ODIR duplicate pairs (5,058 → 5,056
images) and confirming that 1,626 merged-benchmark images are ODIR copies,
which are then excluded from the cross-source transfer test set (Section 6.5)
so nothing in that test set was seen during training.

### 6.3 Protocol ladder — E1 (RQ2, first pass)

**Question:** does patient-disjoint splitting change accuracy, holding the model
fixed?

**Method:** four rungs, same frozen-ResNet50 + logistic-head model, varying only
the split: `image_level` (patients can appear in both train and test),
`patient_disjoint`, `patient_disjoint_dedup`, and `provenance_controlled`
(restricted to a single acquisition source).

**Result:**

| rung | macro-F1 | 95% CI |
|---|---:|---|
| image_level | 0.5496 | [0.5308, 0.5677] |
| patient_disjoint | 0.5444 | [0.5248, 0.5622] |
| patient_disjoint_dedup | 0.5476 | [0.5282, 0.5675] |
| provenance_controlled | 0.5048 | [0.4735, 0.5383] |

No rung differs significantly from any other (all McNemar p > 0.14, all CIs
overlap). **Read literally, H2 (patient leakage inflates scores) is falsified.**

**But this was reported as a lower bound, not a conclusion**, because a frozen
linear head has almost no capacity to memorise a specific patient's fellow eye —
and leakage is exploited *by* memorisation. Reporting this null result as final
would have been the dishonest version of this finding; instead it became the
setup for the next experiment.

### 6.4 Capacity ladder — E1b (RQ2, the correction)

**Question:** does leakage inflation depend on how much the model *can*
memorise, holding features and splits fixed?

**Method:** sweep model capacity from a plain logistic head up to a pure 1-NN
retriever (which, by construction, will find a patient's fellow eye if it's in
the training set).

| model | capacity | image-level | patient-disjoint | inflation | McNemar p |
|---|---|---:|---:|---:|---:|
| logistic | none | 0.5496 | 0.5444 | +0.0052 | 0.75 |
| mlp | low | 0.5843 | 0.5569 | +0.0274 | 0.20 |
| knn_5 | moderate | 0.5250 | 0.5044 | +0.0207 | **0.00065** |
| knn_1 | pure memoriser | 0.5576 | 0.5124 | +0.0453 | **~0** |

**Verdict: inflation grows monotonically with memorisation capacity.** E1's
null was an artefact of testing a model that structurally cannot exploit the
leak — not evidence that the leak doesn't exist. This is the paper's central
methodological point: a protocol-ladder experiment run on only one model
architecture will silently understate leakage if that architecture happens to
have low capacity.

### 6.5 Fine-tuned confirmation — E3 (RQ2, closing the loop)

**Question:** does the E1b pattern hold for the model people would *actually
deploy* — a fine-tuned CNN, not a frozen-feature toy?

**Method:** `colab/train_models.ipynb`, a ResNet50 fine-tuned end-to-end for 8
epochs (seed 42, 384px images) on a Colab T4 GPU, run on the same two rungs
(`image_level`, `patient_disjoint_dedup`).

| rung | macro-F1 | 95% CI |
|---|---:|---|
| image-level (leaky) | 0.6797 | [0.6611, 0.6967] |
| patient-disjoint + dedup | 0.6579 | [0.6393, 0.6768] |

**Inflation: +0.0218 macro-F1, McNemar p = 0.0006825** (886 discordant
predictions out of 5,056 paired test cases).

Placed alongside the capacity sweep, the pattern is monotonic and now
significant at the model people would actually ship:

| model | capacity | inflation | p |
|---|---|---:|---:|
| logistic (E1) | none | +0.0052 | 0.75 |
| **fine-tuned ResNet50 (E3)** | **realistic** | **+0.0218** | **0.0007** |
| 1-NN (E1b) | pure | +0.0453 | ~0 |

**This confirms E1b rather than merely being consistent with it.** The claim in
the paper is upgraded from "a 1-NN toy model shows leakage" to "a deployable
fine-tuned CNN shows leakage, at p < 0.001." The honest caveat carried forward:
this is one seed, one backbone, 8 epochs — E13's seed-stability treatment
(Section 6.10) was run on the frozen-feature models only, so E3 should be read
as *confirmatory*, not independently seed-tested.

*(How this run actually happened — three real engineering problems, fixed live —
is in Section 8.1, since it's part of "how," not just "what.")*

### 6.6 Cross-source transfer — E5

**Question:** does a model trained on one acquisition source generalise to
others?

**Method:** train on the 5,056 ODIR images; test on merged-benchmark images
after removing the 1,626 confirmed ODIR copies (Section 6.2), so nothing in the
test set was seen during training.

| source | classes present | majority-class acc | model acc |
|---|---|---:|---:|
| crop_jpg256 | C, G | 0.517 | **0.330** |
| eyepacs_jpeg | D only | 1.000 | **0.297** |
| png_highres | C, G | 0.502 | **0.104** |
| odir_jpg512 | C, G, N | 0.449 | 0.507 |

**The model does not transfer.** On three of four unseen sources it scores
*below* the accuracy of simply guessing that source's majority class. Two
caveats reported alongside this, not hidden: the in-domain model is itself weak
(macro-F1 0.55, since this uses the frozen-feature model, not E3), so poor
transfer is partly "a weak model" and not purely a domain gap; and each source
is class-restricted, so results are reported per source and never pooled into
one misleading aggregate number.

### 6.7 Is it a disease detector or a demographics detector? — E11

**Question:** does the model's accuracy actually come from the documented
age/sex confound (Section 4), rather than retinal pathology?

**Method:** two tests. First, a classifier given *only* age and sex, no image at
all. Second, the image model re-evaluated under age-matched, sex-matched, and
multi-finding-removed subsets (matching by downsampling within strata, which
closes the confound route without needing to reweight or model anything).

**Result 1 — demographics alone:** accuracy 0.556 against a 0.568 majority
baseline. **Age and sex, with no image, score worse than always guessing the
majority class.** There is no usable class signal in demographics on this
dataset.

**Result 2 — image model under matching:**

| condition | n | macro-F1 | cataract AUROC |
|---|---:|---:|---:|
| unrestricted | 5,056 | 0.5476 | **0.9462** |
| age-matched | 964 | 0.5385 | **0.9414** |
| sex-matched | 972 | 0.5302 | **0.9406** |
| multi-finding removed | 4,679 | 0.5478 | **0.9560** |

Every confidence interval overlaps the unrestricted one; cataract-vs-rest AUROC
holds near 0.94 regardless of which confound is controlled for. **The 11-year
age gap between cataract and DR patients is real in the data, but it is not
what the model is using.** This is the strongest single piece of evidence in
the project that the image model is doing something real, not exploiting a
shortcut — precisely because it was tested rather than merely disclaimed in a
limitations paragraph.

### 6.8 Document extraction — E7 (foundation for RQ3/RQ4/RQ5)

**Method:** 360 synthetic referral documents across six layouts (A–F). Two
extraction methods (E1: simpler, E2: more robust) are developed and tuned only
on layouts A–C, then evaluated on layouts D–F, which were **sealed and opened
exactly once**, after both extractors were frozen — the IDP equivalent of a
held-out test set, chosen specifically so extraction methodology can't be
unconsciously fitted to the eval layouts.

| extractor | split | F1 | exact match |
|---|---|---:|---:|
| E1 | dev | 0.604 | 0.568 |
| E1 | sealed | **0.355** | 0.297 |
| E2 | dev | 0.683 | 0.691 |
| E2 | sealed | **0.488** | 0.453 |
| E2 | sealed | **0.488** | 0.453 |

Both extractors degrade substantially on sealed layouts, which is expected and
reported plainly — it's the reason a *reliability estimate* (Section 6.9) is
needed at all rather than trusting extraction output at face value.

### 6.9 Reliability of the extraction signal — RQ3

**Question:** is raw OCR confidence a usable signal for "was this clinical
field extracted correctly," or does something better need to be learned?

**Method:** compare raw OCR engine confidence against a small learned
classifier (`GradientBoostingClassifier`) trained on extraction features, both
evaluated on sealed layouts.

| extractor | signal | AUROC |
|---|---|---:|
| E1 | raw OCR confidence | 0.5214 |
| E1 | learned estimator | **0.9799** |
| E2 | raw OCR confidence | 0.6389 |
| E2 | learned estimator | **0.9871** |

**Raw OCR confidence is close to useless** (0.52 is barely above chance).
Feature-importance analysis explains why: `format_valid` and `value_length`
dominate the learned estimator's decision (a combined ~0.86 of feature weight),
while `ocr_confidence` itself contributes only ~0.08. In plain terms: OCR can be
"confident" about text it read perfectly that is nonetheless the wrong value for
the field (a semantic extraction error, not a legibility error) — which is
exactly the failure raw confidence cannot see and a format/length check can.

### 6.10 Deferral — E9 / E10 (RQ4, RQ5 — including the negative result)

**Question:** can a referral document improve the system's decision to defer,
without ever touching the diagnosis?

**Method:** two deferral systems compared against an image-confidence-only
baseline. **System A** is two hand-written rules combining image confidence and
document agreement. **System B** is a learned gate that additionally takes the
extraction-reliability estimate (`c_hat`, from Section 6.9) as an input. A
**kill rule was written before the run**: if System B does not beat System A, the
paper reports the rules and drops the learned-gate claim.

| system | AURC (lower = better) | error-detection AUROC | acc @ 50% coverage |
|---|---:|---:|---:|
| image confidence only | 0.389 | 0.588 | 0.589 |
| **System A (rules)** | **0.197** | **0.882** | 0.867 |
| System B (learned) | 0.211 | 0.869 | 0.889 |
| System B without `c_hat` | 0.203 | 0.871 | 0.867 |

**The learned gate lost, and removing `c_hat` from it made it *better*, not
worse.** Per the pre-registered kill rule, the paper reports this as a negative
result rather than quietly shipping the more sophisticated-sounding system:
document evidence roughly halves AURC and lifts error-detection AUROC from 0.59
to 0.88 — a real, large effect — but the *hand-written rules* deliver it, and
the learned gate adds nothing on top. `c_hat` itself is not wasted: it remains
an excellent *correctness* estimator (Section 6.9, AUROC 0.98–0.99); it simply
turned out not to help the *deferral* decision specifically. Caveat carried
forward: only 180 sealed rows back this comparison, so a 0.01 AURC gap is well
within noise — the honest statement is "the learned gate did not improve on the
rules," not "the rules are definitively better."

### 6.11 Explainability — E12

**Question:** does the model's attention (via Grad-CAM++) actually fall on the
retina, and can the maps be trusted at all?

**Method:** manual Grad-CAM++ (forward/backward hooks, no external XAI
dependency) over the frozen-feature model, computed only on held-out fold 0 —
images the classification head never saw during fitting (an earlier version
attributed in-sample, which was caught and fixed before it was reported).

| measure | value |
|---|---:|
| FOV-mass ratio (observed) | 0.8364 |
| uniform-attribution baseline | 0.5348 |
| concentration ratio | **1.56×** |
| randomisation sanity check | **PASS** (r = −0.094) |

Attribution concentrates inside the retinal field of view at 1.56× what
uniform attention would give, and the randomisation sanity check passes,
meaning the maps genuinely depend on the learned weights (not an artefact of the
attribution method itself). **What this licenses:** the model is looking at the
retina, not the border or letterboxing. **What it explicitly does not license:**
any claim that it's looking at a *lesion* — there are no lesion masks and no
ophthalmologist review, so that claim is off the table by design, not by
oversight.

### 6.12 Statistical validation — E13

**Question:** does the E1b leakage finding hold across more than one random
split, and does the project's overall family of hypothesis tests need multiple-
comparison correction?

**Method:** re-run the leakage comparison across three split seeds (42, 7,
2024), and apply Holm-Bonferroni across all 13 paired hypothesis tests reported
in the project.

| model | mean inflation | sd | range across seeds |
|---|---:|---:|---|
| **knn_1** | **+0.0453** | 0.0027 | [+0.0434, +0.0484] |
| logistic | −0.0031 | 0.0085 | [−0.0126, +0.0038] |

The memoriser's effect is large, tight, and significant at every seed. The
logistic head's effect **flips sign between seeds** — a concrete illustration of
"this is noise, not signal" rather than an abstract warning. After
Holm-Bonferroni correction across all 13 tests, **5 survive, and all five are
memorisation-based** (the three `knn_1` seeds plus `knn_5` and `knn_1` from
E1b). The instructive near-miss: `leakage[logistic, seed7]` had a raw p of
0.037 — it would have read as "significant" without correction, and its effect
was *negative*. That single row is the concrete argument for why the correction
was applied at all, rather than a boilerplate methods-section sentence.

---

## 7. The screening prototype

Beyond the audit, the project ships a working, runnable system — not a notebook.

- **`src/app/pipeline.py`** — `ScreeningPipeline.predict_image()` takes only an
  image path. Internally: ResNet50 backbone → **L2-normalise** → linear head →
  temperature scaling → softmax. (The L2-normalisation step is not decorative —
  see Section 8.2 for the bug it exists to prevent.)
- **`src/app/chatbot.py`** — a template-based chatbot, deliberately *not* an
  LLM, that reads only structured case data and always states the deferral
  status before any diagnosis, so it cannot narrate confidence the pipeline
  didn't actually produce.
- **UI** — designed first in Figma (file `u8Mu72qiltp4zVrJWTKHWP`, three frames:
  Case Review, Worklist, Design Intent; a shared "Screening tokens" variable
  collection for color), then implemented in `src/app/render.py` with CSS
  variables mirrored exactly from the Figma tokens, so the deployed HTML matches
  the design source of truth rather than approximating it.
- **Local demo** — `scripts/serve.py`, a stdlib-only `http.server`, deliberately
  avoiding a web framework for a project this size.
- **Deployment** — GitHub (private repo) for code, Vercel (public static site,
  `ophthalmic-screening.vercel.app`) for the findings, figures, and both app
  screens. The interactive upload form only works locally, because a static
  host cannot execute a PyTorch model — the deployed pages say so rather than
  offering a form that would fail silently.
- **Tests** — 137 passing, including a permanent regression test for the
  L2-normalisation bug (Section 8.2) that checks live inference features match
  cached training-time embeddings, and a signature-inspection test that
  literally asserts the diagnosis function cannot accept a document argument.

**One known, documented gap:** the deployed model is the frozen-feature
linear-head model (macro-F1 0.548), not the fine-tuned E3 model (macro-F1
0.658–0.680) that is the paper's headline result. The Colab notebook wrote out
prediction CSVs and summary statistics, not the fine-tuned weights themselves,
so there is nothing yet to re-export into the app. Upgrading this would mean
re-running Colab with the weights saved, then re-running `scripts/export_model.py`
against them — not started, since it was never requested.

---

## 8. Process — real problems, and how they were actually found and fixed

This section exists because "how" includes the debugging, not just the final
pipeline diagram.

### 8.1 The Colab run (E3)

Three separate real problems surfaced getting the fine-tuned baseline to run,
each fixed live rather than papered over:

1. **Avoiding a 7 GB upload.** `scripts/build_colab_bundle.py` resizes every
   image to 384px before zipping, cutting the upload to 77 MB — a 91× reduction,
   and almost all of the discarded resolution would have been thrown away by
   training anyway.
2. **A silent bundling bug, caught before it could fail mid-training.** The
   bundler initially packed images from the *deduplicated* manifest (5,056
   images) while two of the three protocol-ladder splits are defined against the
   *raw* manifest (5,058 images, since `image_level` is defined by *containing*
   the duplicates). This would have thrown a `FileNotFoundError` partway through
   a training fold, after the upload had already happened. Fixed by always
   bundling from the raw manifest, and by adding a self-verification step to the
   builder that checks every split's referenced filenames resolve to a bundled
   image *before* declaring success — it now exits non-zero rather than
   printing a false "done."
3. **Runtime-reset friction, twice**, both diagnosed live over chat as the user
   ran the notebook interactively: Google Drive reporting as mounted when it
   wasn't (fixed with a diagnostic-and-force-remount cell), and `DRIVE_ROOT`
   becoming undefined after a runtime reset wiped notebook-level variables
   (fixed by re-running the defining cell rather than guessing at the value).

### 8.2 The 100%-confidence bug — the most serious bug found in the project

While validating the deployed inference pipeline, every prediction was coming
back at essentially 100% confidence — which would have silently defeated the
entire deferral system, since a system that is always "certain" never defers.
**Diagnosis:** printing the L2 norm of the live backbone output next to the norm
of the cached training-time embeddings showed cosine similarity of 1.000000
(identical *direction*) but a roughly 12× difference in *magnitude* — the
classification head had been fitted on L2-normalised embeddings (norm exactly
1.0), but live inference was feeding the head raw, un-normalised backbone
output. That 12× magnitude difference saturated the softmax, making every
prediction read as maximally confident regardless of how uncertain the model
actually was. **Fix:** an explicit `torch.nn.functional.normalize(features,
p=2, dim=1)` call was added at inference time, and a permanent regression test
(`test_live_features_match_the_cached_embeddings`) now checks this can't
silently regress. This is why Section 7 calls out L2-normalisation explicitly
rather than treating it as an implementation detail — it was the difference
between a working safety mechanism and a broken one that looked fine on the
surface.

### 8.3 The Vercel deployment block

Every deployment attempt appeared to hang indefinitely at "Building…" with no
error, for about 40 minutes across six attempts. The cause, findable only by
querying the Vercel API directly rather than reading CLI output: deploying from
inside a git-tracked directory made Vercel attach commit metadata and try to
verify the committer's email against a GitHub account, which silently blocked
the deployment (`readyState: BLOCKED`) with no CLI-visible error. Fixed by
copying the site to a directory with no `.git` ancestry before deploying,
which sidesteps the check entirely. Documented permanently in `docs/DEPLOY.md`
so this isn't re-diagnosed from scratch on the next redeploy.

### 8.4 Smaller corrections, recorded so they aren't repeated

- The duplicate-detection threshold was first calibrated on a *balanced*
  sample and looked perfect, but failed badly at the real search-space base
  rate (Section 6.2) — recalibrated against hard negatives.
- ImageNet-embedding cosine similarity was tried and rejected as a duplicate
  detector before the FOV-fingerprint approach was built.
- OCR engine was switched from Tesseract to RapidOCR specifically to avoid a
  system-binary dependency, keeping the project reproducible on a clean machine.

---

## 9. What is done, and what is explicitly not

**Done and verified:**
- The full local pipeline (manifest → provenance → dedup → splits → 12
  experiments → figures → report), reproducible end-to-end via
  `scripts/run_all.py`.
- The GPU confirmation experiment (E3), integrated into every document.
- A working, tested screening prototype with a structurally-enforced safety
  rule, a Figma-designed UI, and a critical inference bug found and fixed
  before shipping.
- Deployment: private GitHub repo, public Vercel static site, both verified
  live and current.
- 137 tests passing.

**Explicitly not done, and not silently assumed:**
- The deployed demo runs the frozen-feature model, not the stronger fine-tuned
  E3 model (Section 7) — a known gap, not an oversight, pending only if
  requested.
- **The research paper itself.** `reports/PAPER_OUTLINE.md` maps every section
  to its supporting evidence and figure, but no prose exists yet — this
  document is the deep-explanation source material for writing it, not a
  substitute for it.
- E3 has not been independently seed-tested the way the frozen-feature models
  were in E13 (Section 6.12) — flagged as a caveat everywhere E3 is reported,
  not hidden.
- No real (non-synthetic) referral document corpus — searched for and not
  found; the search itself is recorded rather than papered over.

---

## 10. Where to find everything

| Question | File |
|---|---|
| Every generated number | `reports/RESULTS.md` |
| Section-by-section paper skeleton, reviewer objections pre-empted | `reports/PAPER_OUTLINE.md` |
| Engineering history, bugs, what's cached | `STATUS.md` |
| Project pitch, contributions, limitations | `README.md` |
| Vercel redeploy procedure and why the naive way fails | `docs/DEPLOY.md` |
| All 14 figures | `figures/` |
| Reproduce everything | `scripts/run_all.py --list` then `python scripts/run_all.py` |
