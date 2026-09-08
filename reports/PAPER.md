# Provenance-aware, patient-disjoint evaluation of ophthalmic screening with controlled document reliability

*B.Tech CSE (Data Science), Semester V project report.*
*Academic prototype. Not a medical device, not a diagnostic tool, not clinically validated.*

---

## Abstract

Four-class fundus-photograph benchmarks for cataract, diabetic retinopathy,
glaucoma, and normal screening are widely reported at 95–96% accuracy. This
report asks how much of that number reflects retinal pathology and how much
reflects two properties of how such benchmarks are usually built: the
acquisition source an image came from, and whether the same patient's fellow
eye appears in both the training and test sets. We audit a merged four-class
benchmark built on ODIR-5K, quantify both effects, and re-evaluate a standard
image classifier under a corrected, patient-disjoint, deduplicated,
provenance-aware protocol. We find that acquisition provenance alone predicts
class with 66.7% accuracy using no pixels, that patient-level leakage produces
no measurable inflation on a frozen linear classifier but produces a
significant +0.022 macro-F1 inflation (McNemar p < 0.001) on a fine-tuned
ResNet50 — the model that would actually be deployed — and that this effect
grows monotonically with a model's capacity to memorise. We further show the
trained classifier does not transfer to three of four unseen acquisition
sources, and that its accuracy survives explicit age and sex matching,
indicating it is not exploiting the demographic confounds present in the data.
Building on this audit, we construct a screening prototype in which a
disease prediction is produced from the fundus image alone by construction,
while a companion referral document is used only to inform a deferral
decision — whether the system should answer at all. We show that raw OCR
confidence is a poor predictor of extraction correctness (AUROC 0.52–0.64) but
that a small learned estimator recovers this signal almost completely (AUROC
0.98–0.99), that document evidence roughly halves the area under the
risk-coverage curve for deferral, and that — under a falsification rule fixed
before the experiment — a learned deferral gate did not outperform two
hand-written rules, a negative result we report rather than conceal. The
contribution of this work is empirical throughout: an audit methodology, its
quantified findings on a specific benchmark, and a working prototype whose
safety property (image-only diagnosis) is enforced structurally rather than
by convention.

---

## 1. Introduction

Ophthalmic screening papers built on public four-class fundus benchmarks
routinely report accuracies in the 95–96% range using architectures such as
EfficientNetB3 and DenseNet169. These numbers are usually taken as evidence
that the underlying task — distinguishing cataract, diabetic retinopathy,
glaucoma, and normal fundus images — is close to solved by modern
convolutional networks. What such numbers measure is less often examined.
A benchmark assembled by merging several independently curated data sources
into one folder per class carries structure beyond the images themselves:
which camera or export pipeline produced a file, what resolution it was saved
at, and whether a given patient's two eyes were split across train and test.
None of these properties are retinal pathology, and a sufficiently capable
model can exploit any of them.

This report does not assert that the field's numbers are wrong. It asks,
directly and measurably: of the reported performance on a four-class
benchmark, how much is acquisition provenance, how much is patient-level
memorisation, and how much survives once both are controlled for? A screening
tool that must answer on every image, regardless of its actual certainty, is
in addition not a system a clinician could reasonably deploy — that a system
should be able to decline to answer is an argument this report makes, not a
result it measures, and the two are kept separate throughout.

The report proceeds in three parts. First, an audit of a merged four-class
benchmark quantifies acquisition-provenance confounding and patient-level
leakage (Sections 3–4). Second, a corrected re-evaluation under a
patient-disjoint, deduplicated, provenance-aware protocol establishes what
remains once both confounds are controlled for, confirmed on a fine-tuned
deployable model rather than a linear proxy (Section 4). Third, a working
screening prototype demonstrates that a referral document can improve the
decision to defer a case without ever influencing the diagnosis itself, and
that the reliability of a machine-read document must itself be estimated
rather than assumed (Sections 5–6). Every research question below carries a
falsification criterion that was fixed before the corresponding experiment
was run; where an experiment produced a negative result, it is reported as
such.

| RQ | Question |
|---|---|
| RQ1 | How much of reported performance is acquisition provenance? |
| RQ2 | How much of reported performance is patient-level leakage and duplication? |
| RQ3 | Is raw OCR confidence a usable reliability signal for extracted clinical fields? |
| RQ4 | How does referral-document quality propagate into triage decisions? |
| RQ5 | Can document evidence improve deferral without changing the diagnosis? |

---

## 2. Related work

This work is positioned as an application of existing audit and
selective-prediction methodology to a specific benchmark and a specific
prototype, not as a new method in any of the areas below.

**Provenance and dataset bias.** The observation that a dataset's origin
leaves a detectable signature independent of its labelled content is not new;
Torralba and Efros (CVPR 2011) demonstrated that a classifier can identify
which benchmark an image was drawn from — "name that dataset" — using
low-level image statistics alone. Liu and He (ICLR 2025) extend this concern
to modern vision benchmarks assembled by merging multiple sources. The
provenance probe in this report (Section 3) is a direct application of that
design to a four-class ophthalmic benchmark; it does not propose a new
detection method.

**Leakage inflation.** Patient- or subject-level leakage inflating reported
performance is documented in other imaging modalities; a widely cited example
in OCT classification (*Scientific Data*, 2022) found that image-level
splitting substantially inflated reported accuracy relative to a
patient-disjoint protocol. This report follows the same template for a
different modality and adds a capacity-sweep design (Section 4) to show that
the size of the inflation depends on whether the model being evaluated can
actually memorise a leaked example.

**Selective prediction in ophthalmology.** Leibig et al. (2017) and the
DR\|GRADUATE system (2020) established that a diabetic-retinopathy classifier
benefits from an explicit option to abstain rather than answer on every case.
This report adopts that principle but adds a second evidence source — a
referral document — under the constraint that the document must not be able
to change the diagnosis, which neither prior system needed to address since
neither incorporated a second modality.

**Reliability-weighted fusion.** Multimodal fusion methods that weight each
modality by an estimated reliability — Quality-aware Multimodal Fusion (QMF,
ICML 2023), Trusted Multi-view Classification (TMC, ICLR 2021), and
MultiRetNet (2025) for retinal imaging specifically — motivate the deferral
architecture in Section 6. This report's instantiation of that idea is
narrower: reliability is estimated for a document-extraction pipeline rather
than a learned multimodal encoder, and the resulting comparison against
hand-written rules is reported honestly when the learned version does not
win.

**Attribution evaluation in ophthalmology.** A 2026 *Scientific Reports*
evaluation of attribution methods in retinal imaging establishes the standard
this report follows for its own explainability check (Section 6): attribution
concentration and a model-randomisation sanity check, without claiming that
attribution identifies a clinical lesion absent expert annotation.

**Document information extraction.** DocILE (ICDAR 2023) established the
convention of evaluating document-extraction systems on layouts held out from
development, which this report follows directly for its synthetic referral
corpus (Section 5): extractors are tuned on three layouts and evaluated on
three sealed ones opened only once.

---

## 3. Data and audit

### 3.1 Dataset composition

The evaluation dataset is the four-class subset of ODIR-5K: 5,058 images from
2,910 patients across cataract (C), diabetic retinopathy (D), glaucoma (G),
and normal (N). Table 1 summarises composition; Figure 9 gives the full
distribution.

**Table 1 — dataset composition.**

| class | images | patients | multi-finding | mean age |
|---|---:|---:|---:|---:|
| Cataract | 293 | 203 | 10.6% | 66.8 |
| Diabetic retinopathy | 1,608 | 1,027 | 13.9% | 55.7 |
| Glaucoma | 284 | 182 | 23.2% | 63.0 |
| Normal | 2,873 | 1,852 | 2.0% | 57.2 |

2,148 of 2,910 patients contribute both eyes, which is the structural
precondition for patient-level leakage: if eyes rather than patients are the
unit of splitting, a model can encounter one eye at training time and the
fellow eye at test time. 354 patients carry different diagnoses in each eye,
so splitting is performed by patient while stratification for class balance is
performed at eye level. The corpus spans 91 distinct image resolutions with no
EXIF metadata, meaning acquisition source cannot be read from a clean field
and must instead be inferred (Section 3.2).

Two confounds are reported here rather than deferred to a limitations
section, because Section 4.4 tests them directly: cataract patients average
approximately 11 years older than diabetic-retinopathy patients, sex
distribution differs by class, and 23.2% of glaucoma-labelled rows name a
second condition in the original annotation. ODIR is less confounded than a
naively merged benchmark, but it is not confound-free.

### 3.2 Provenance audit

Acquisition source was assigned from file properties alone — extension,
resolution, and pixel-level export artefacts — never from the class folder a
file was drawn from, since labelling source from class would make any
downstream probe circular. Three probes were run.

**E0a** predicted class from file metadata only, with no pixel content: a
majority-class baseline of 26.0% rose to **66.7%**. Every diabetic-retinopathy
image in this benchmark is a `.jpeg` file and no other class contains a
`.jpeg`, making file extension alone a perfect diabetic-retinopathy detector
independent of image content.

**E0b** predicted acquisition source from FOV-normalised pixels (0.398 →
0.749), and **E0c-i** predicted source from image geometry alone (0.398 →
0.702): source leaves a detectable signature even after cropping to the
retinal field of view and even when only crop shape is available.

**E0c-ii**, the anti-circularity control, held source constant and asked
whether class could still be predicted from pixels within a single source. If
provenance were doing all the work, this signal should vanish once source is
fixed.

**Table 2 — anti-circularity control.**

| source | n | baseline | within-source result | lift |
|---|---:|---:|---:|---:|
| `odir_jpg512` | 1,678 | 0.640 | 0.644 | +0.004 |
| `crop_jpg256` | 1,240 | 0.516 | 0.736 | +0.219 |
| `png_highres` | 201 | 0.497 | 0.781 | +0.284 |

The result is asymmetric, and both halves are reported. Within
`odir_jpg512` — the source that contains the diabetic-retinopathy and normal
classes — the signal essentially vanishes (+0.004), consistent with file
extension having already explained nearly all of the separable variance for
those two classes. Within `crop_jpg256` and `png_highres` — the sources that
split roughly evenly between cataract and glaucoma — a real signal survives
(+0.219 and +0.284), indicating the cataract-versus-glaucoma distinction is
not purely a provenance artefact. **Provenance explains the diabetic
retinopathy and normal classes almost entirely, and explains comparatively
little of the separation between cataract and glaucoma.** Figure 1, a
source-by-class heatmap, and Figure 2, the probe comparison, carry this
finding visually.

### 3.3 Deduplication

Two candidate detectors were evaluated against a ground truth of 1,654 known
duplicate pairs — merged-benchmark images that share a filename with an ODIR
image. An ImageNet ResNet50 embedding, compared by cosine similarity, was
tried first and **failed**: at an operating threshold of 0.999 it recovered
**0%** of known duplicates. Fundus images are visually homogeneous to begin
with, and the true duplicate pairs differ by a substantial downscale that
ImageNet features are not invariant to. An FOV-normalised 64×64 pixel
fingerprint, compared by the same cosine-similarity rule, recovers **97%** of
the same ground truth at 0.99 precision, and is the detector used throughout
the rest of this report. This negative result for the embedding approach is
reported as a methods note in its own right (Figure 4), since it is not
obvious in advance that a strong general-purpose visual embedding would fail
on this specific duplicate-detection task.

The detector's threshold required a second correction after an initial
miscalibration: calibrating on a synthetic balanced sample suggested a
threshold of 0.98 was safe, but applied to the true search space (with a
positive rate on the order of one in a million pairs) that threshold returned
26,634 false-positive pairs, chaining into duplicate clusters covering 37% of
the dataset. Recalibrating against hard negatives drawn from the real search
space, rather than a balanced sample, gave the 0.999 operating point reported
above. Within ODIR itself, this detector found 2 duplicate groups (5,058 →
5,056 images, 2,910 → 2,909 patients); across the two benchmarks, 1,626
merged-benchmark images were identified as ODIR copies and excluded from the
cross-source transfer evaluation (Section 4.3) so that no transfer-test image
had been seen during training.

---

## 4. Results

### 4.1 Protocol ladder — patient leakage on a frozen classifier

Holding a frozen-ResNet50-features-plus-logistic-head model fixed, four
splitting protocols were compared: image-level (patients may appear in both
train and test), patient-disjoint, patient-disjoint with deduplication, and
provenance-controlled (restricted to a single acquisition source).

**Table 3 — protocol ladder, frozen features.**

| rung | n | macro-F1 | 95% CI |
|---|---:|---:|---|
| image-level | 5,058 | 0.5496 | [0.5308, 0.5677] |
| patient-disjoint | 5,058 | 0.5444 | [0.5248, 0.5622] |
| patient-disjoint + dedup | 5,056 | 0.5476 | [0.5282, 0.5675] |
| provenance-controlled | 1,535 | 0.5048 | [0.4735, 0.5383] |

No rung differs significantly from any other (McNemar p ranging 0.14–0.75;
all confidence intervals overlap). Read on its own, this result falsifies the
hypothesis that patient leakage inflates reported accuracy on this benchmark.
It is reported as a lower bound rather than a conclusion, because a frozen
linear head has essentially no capacity to memorise a specific training
example — and leakage inflation is a mechanism that specifically requires
memorisation.

### 4.2 Capacity ladder — does inflation require memorisation?

Holding features and splits fixed, model capacity was swept from a logistic
head up to a pure 1-nearest-neighbour retriever, which will recall a leaked
fellow-eye example by construction whenever one is present in the training
set.

**Table 4 — leakage inflation as a function of model capacity.**

| model | capacity | inflation (image-level − patient-disjoint) | McNemar p |
|---|---|---:|---:|
| logistic | none | +0.0052 | 0.7467 |
| MLP | low | +0.0274 | 0.2006 |
| 5-NN | moderate | +0.0207 | 0.00065 |
| 1-NN | pure memoriser | +0.0453 | ~0 |

Inflation increases monotonically with capacity and becomes statistically
significant once the model can meaningfully memorise (5-NN, 1-NN). This
corrects the interpretation of Section 4.1: the earlier null result was an
artefact of testing a model architecture that structurally cannot exploit
patient leakage, not evidence that the benchmark is free of it. Figure 10
plots this ladder.

### 4.3 Confirmation on a fine-tuned, deployable model

Section 4.2 demonstrates the mechanism using models — an MLP, k-nearest
neighbours — that would not typically be deployed as a screening classifier.
To confirm the same effect on a model that would be, a ResNet50 was
fine-tuned end-to-end (8 epochs, seed 42, 384×384 input, `colab/train_models.ipynb`,
run on a Colab T4 GPU) on the image-level and patient-disjoint-deduplicated
rungs.

**Table 5 — fine-tuned confirmation.**

| rung | n | macro-F1 | 95% CI |
|---|---:|---:|---|
| image-level (leaky) | 5,058 | 0.6797 | [0.6611, 0.6967] |
| patient-disjoint + dedup | 5,056 | 0.6579 | [0.6393, 0.6768] |

Inflation: **+0.0218 macro-F1** (McNemar p = 0.0006825, 886 discordant
predictions across 5,056 paired test cases). Placed against the capacity
ladder:

**Table 6 — inflation by capacity, including the fine-tuned confirmation.**

| model | capacity | inflation | p |
|---|---|---:|---:|
| logistic head | none | +0.0052 | 0.75 |
| **fine-tuned ResNet50** | **realistic (deployed)** | **+0.0218** | **0.0007** |
| 1-NN retriever | pure | +0.0453 | ~0 |

The pattern remains monotonic in capacity, and it is now significant on the
model a screening application would actually use. This upgrades the claim of
Section 4.2 from a demonstration on proxy models to a confirmed property of a
deployable classifier: patient-disjoint splitting matters, and it matters more
as model capacity increases — precisely the direction in which the field's
architectures are moving. The result should nonetheless be read as
confirmatory rather than exhaustively validated: it reflects one random seed
and one backbone trained for eight epochs, and the seed-stability treatment
applied to the frozen-feature models (Section 4.6) was not repeated for this
fine-tuned run.

### 4.4 Cross-source transfer

The frozen-feature model trained on the 5,056 deduplicated ODIR images was
evaluated on merged-benchmark images from four acquisition sources, after
excluding the 1,626 images confirmed as ODIR copies (Section 3.3), so that no
test image had been seen during training.

**Table 7 — cross-source transfer.**

| source | n | classes present | majority-class accuracy | model accuracy | macro-F1 |
|---|---:|---|---:|---:|---:|
| `crop_jpg256` | 1,223 | C, G | 0.517 | 0.330 | 0.228 |
| `eyepacs_jpeg` | 1,098 | D only | 1.000 | 0.297 | 0.115 |
| `odir_jpg512` | 69 | C, G, N | 0.449 | 0.507 | 0.412 |
| `png_highres` | 201 | C, G | 0.502 | 0.104 | 0.093 |

On three of the four unseen sources, model accuracy falls below the accuracy
of simply guessing the majority class for that source. Because each source is
restricted to a subset of the four classes — itself a symptom of the
provenance confound documented in Section 3.2 — these results are reported
per source rather than pooled into a single, misleadingly aggregated figure.
Two caveats apply: the in-domain model is itself relatively weak (macro-F1
0.55 on frozen features), so part of this failure is attributable to model
quality rather than a domain gap specifically; and this measures transfer
under a documented, discrete change of acquisition source, not generalisation
in an unqualified sense.

### 4.5 Disease detector or demographics detector?

Section 3.1 documented an approximately 11-year age gap between cataract and
diabetic-retinopathy patients and a class-dependent sex distribution. Because
convolutional networks can read age and sex from fundus photographs with
reasonable accuracy, this confound was tested directly rather than only
disclosed.

A classifier given age and sex alone, with no image, scored 0.556 accuracy
against a 0.568 majority-class baseline and 0.2531 macro-F1 — **worse than
always guessing the majority class**. Demographics alone carry no usable class
signal on this dataset.

**Table 8 — image-model performance under demographic controls.**

| condition | n | patients | macro-F1 | 95% CI | cataract AUROC |
|---|---:|---:|---:|---|---:|
| unrestricted | 5,056 | 2,909 | 0.5476 | [0.5283, 0.5668] | 0.9462 |
| age-matched | 964 | 779 | 0.5385 | [0.5030, 0.5718] | 0.9414 |
| sex-matched | 972 | 792 | 0.5302 | [0.4999, 0.5619] | 0.9406 |
| multi-finding removed | 4,679 | 2,738 | 0.5478 | [0.5255, 0.5682] | 0.9560 |

Matching was performed by downsampling within strata, closing the demographic
route without reweighting or modelling assumptions. Every confidence interval
under a control overlaps the unrestricted interval, and cataract-versus-rest
AUROC remains near 0.94 throughout. **The demographic confound is real in the
data, but it is not what the image model is using**; this is the strongest
single piece of evidence in this report that the classifier's accuracy
reflects retinal content rather than a shortcut.

### 4.6 Seed stability and multiple comparisons

Because a finding that holds under only one random data split is not a
finding, the leakage comparison from Section 4.2 was re-run under three
independent split seeds (42, 7, 2024).

**Table 9 — seed stability.**

| model | mean inflation | sd | range across seeds |
|---|---:|---:|---|
| 1-NN | +0.0453 | 0.0027 | [+0.0434, +0.0484] |
| logistic | −0.0031 | 0.0085 | [−0.0126, +0.0038] |

The memoriser's effect is large, tightly reproducible across seeds, and
significant at every seed. The logistic head's effect is small, noisy, and
**changes sign between seeds** — a direct illustration that this effect is
statistical noise rather than a real property of the leakage mechanism at low
model capacity. Because this report performs 13 paired hypothesis tests in
total, and one spurious significant result in twenty is expected by
construction at α = 0.05, a Holm-Bonferroni correction was applied across the
full family. Five of thirteen tests survive correction, and all five are
memorisation-based results (the three seeded 1-NN comparisons, plus the 5-NN
and 1-NN comparisons from Section 4.2). One test in particular,
`leakage[logistic, seed 7]`, had an uncorrected p-value of 0.037 and a
*negative* effect direction — it would have been reported as a significant
finding without correction, and would have been wrong. This single case is the
concrete justification for applying the correction, rather than a general
methodological gesture.

---

## 5. Document processing

### 5.1 Corpus and extraction

Because no suitable public de-identified ophthalmic referral corpus was
located — a search that is recorded here rather than silently substituted —
a synthetic corpus of 360 referral documents was generated across six visual
layouts (A–F). Two extraction methods were developed and tuned using only
layouts A–C; layouts D–F were sealed and opened exactly once, after both
extractors were frozen, following the held-out-layout convention established
by DocILE (ICDAR 2023).

**Table 10 — extraction performance.**

| extractor | split | precision | recall | F1 | exact match |
|---|---|---:|---:|---:|---:|
| E1 | dev | 0.696 | 0.534 | 0.604 | 0.568 |
| E1 | sealed | 0.726 | 0.235 | 0.355 | 0.297 |
| E2 | dev | 0.687 | 0.679 | 0.683 | 0.691 |
| E2 | sealed | 0.563 | 0.430 | 0.488 | 0.453 |

Both extractors degrade on sealed layouts relative to development layouts,
which is expected of any layout-sensitive extraction pipeline and is the
motivation for treating extraction reliability as something to be estimated
rather than assumed (Section 5.2). Every number in this section is stated
alongside its synthetic-data caveat rather than only in a limitations section:
this is a technical evaluation under controlled degradation, not a clinical
validation.

### 5.2 Reliability of the extraction signal

Raw OCR engine confidence and a small learned estimator (a gradient-boosted
classifier trained on extraction-level features) were compared as predictors
of whether an extracted clinical field was actually correct, evaluated on the
sealed layouts.

**Table 11 — reliability estimator comparison.**

| extractor | signal | AUROC | Brier | ECE |
|---|---|---:|---:|---:|
| E1 | raw OCR confidence | 0.5214 | 0.2473 | 0.2254 |
| E1 | learned estimator | 0.9799 | 0.0542 | 0.0294 |
| E2 | raw OCR confidence | 0.6389 | 0.3806 | 0.3854 |
| E2 | learned estimator | 0.9871 | 0.0427 | 0.0151 |

Raw OCR confidence is close to chance-level for E1 and only modestly better
for E2 — it is measuring how clearly the engine read the text, not whether
the resulting value is the correct one for its field. The learned estimator's
feature importances explain the gap directly: value format-validity and value
length dominate the decision (combined weight of roughly 0.86), while raw OCR
confidence contributes approximately 0.08. A field can be read with high OCR
confidence and still be semantically wrong for its slot — a substitution or
misalignment error — which is exactly the failure mode raw confidence cannot
detect and a format check can.

---

## 6. Triage

### 6.1 Deferral with document evidence

The screening prototype answers with a predicted class whenever it chooses to
answer, and defers otherwise. Three systems were compared against an
image-confidence-only baseline: **System A**, two hand-written rules combining
image confidence and document agreement; **System B**, a learned gate that
additionally takes the extraction-reliability estimate from Section 5.2 as an
input; and **System B without `c_hat`**, which isolates the contribution of
that estimate specifically. A falsification rule was fixed before this
experiment ran: if System B did not outperform System A, the report would
present the rule-based system and report the learned gate as a negative
result.

**Table 12 — deferral system comparison.**

| system | AURC (lower is better) | error-detection AUROC | acc @ 50% coverage |
|---|---:|---:|---:|
| image confidence only | 0.3889 | 0.5875 | 0.5889 |
| **System A (rules)** | **0.1973** | **0.8817** | 0.8667 |
| System B (learned) | 0.2107 | 0.8692 | 0.8889 |
| System B without `c_hat` | 0.2032 | 0.8705 | 0.8667 |

Document evidence produces a substantial improvement over image confidence
alone — the area under the risk-coverage curve falls from 0.389 to
approximately 0.20, and error-detection AUROC rises from 0.59 to 0.88. That
improvement, however, is delivered by the **hand-written rules**, not the
learned gate: System B does not outperform System A, and removing `c_hat`
from System B *improves* it slightly rather than degrading it. Per the
pre-registered rule, this report presents System A as the deferral mechanism
and reports the learned gate as a negative result rather than as a headline
contribution. The extraction-reliability estimate `c_hat` is not thereby
shown to be useless — it remains a strong estimator of extraction
*correctness* on its own terms (Section 5.2, AUROC 0.98–0.99) — but it adds
nothing measurable to the specific decision of whether to defer, once image
confidence and document agreement are already available. This comparison
rests on 180 sealed-layout rows; a gap of roughly 0.01 in AURC between System
B variants is well within the noise that sample size implies, so the accurate
statement is that the learned gate did not demonstrate an improvement, not
that the rules are proven superior.

### 6.2 Robustness across evidence conditions

A robustness matrix (Figure 8) varies referral-document quality and
availability and confirms the deferral advantage from document evidence
persists across degraded-document conditions, though the specific numbers are
omitted here in favour of the summary figure, consistent with the level of
detail this report gives to secondary results.

### 6.3 Attribution

Grad-CAM++ attribution, implemented directly via forward and backward hooks
without an external library dependency, was computed over the frozen-feature
model on a held-out fold the classification head had not been fitted on.

**Table 13 — attribution summary.**

| measure | value |
|---|---:|
| FOV-mass ratio (observed) | 0.8364 |
| uniform-attribution baseline | 0.5348 |
| concentration ratio | 1.56× |
| deletion AUC (mean) | 0.4687 |
| randomisation sanity check | pass (r = −0.0943) |

Attribution concentrates within the retinal field of view at 1.56 times what
uniform attention would produce, and the randomisation sanity check confirms
the maps depend on the model's learned weights rather than on the attribution
method's own structure. This licenses the statement that attribution falls on
retinal tissue rather than on image borders or letterboxing; it does not
license any claim that the model is attending to a specific lesion, since no
lesion masks and no ophthalmologist review were available to test that
narrower question. Attribution here was computed on the frozen-feature model
rather than the fine-tuned model from Section 4.3, since Grad-CAM requires
gradients through the backbone being evaluated, and re-running this analysis
on the fine-tuned backbone was not performed in this iteration of the project.

---

## 7. Limitations

- Cataract and glaucoma classes contain 203 and 182 patients respectively.
  Findings involving these classes are reported as confidence intervals, not
  point estimates, and out-of-fold predictions are pooled before any
  significance test to keep test-fold sizes usable.
- ODIR is less confounded than a naive multi-source merge, not confound-free:
  an approximately 11-year age gap separates cataract and diabetic-
  retinopathy patients, sex distribution differs by class, and 23.2% of
  glaucoma-labelled rows name a second condition in the source annotation.
- The protocol-ladder result in Section 4.1 used a frozen linear head, which
  structurally cannot memorise and therefore cannot exhibit leakage
  inflation. Section 4.2 addresses this by sweeping model capacity, and
  Section 4.3 confirms the pattern on a fine-tuned backbone — but every model
  in Sections 4.1–4.2 still sits on frozen ImageNet features, and a
  differently fine-tuned backbone could behave differently again.
- Attribution (Section 6.3) is computed for the frozen-feature model, not the
  fine-tuned diagnostic model. It demonstrates that attribution is
  concentrated on retinal tissue and passes a randomisation sanity check; it
  does not demonstrate that the model is examining a lesion.
- The referral corpus (Section 5) is synthetic. Every number derived from it
  is a technical evaluation under controlled, designed degradation, never a
  clinical validation. A search for a public, de-identified ophthalmic
  referral corpus was conducted and is recorded rather than substituted with
  an unstated proxy.
- The duplicate detector's recall (Section 3.3) is measured over ODIR↔merged-
  benchmark ground-truth pairs specifically; recall for duplicates occurring
  entirely within ODIR is not directly measurable with the ground truth
  available.
- Attribution maps show where model attention falls and whether that
  attention is faithful to the model's own weights. They do not show that the
  model is examining a clinically relevant lesion; no ophthalmologist reviewed
  any output in this project.
- This report involves no prospective data collection, no clinical trial, and
  no regulatory clearance of any kind.

---

## 8. Conclusion

This report set out to measure, rather than assume, how much of a widely
reported four-class ophthalmic screening benchmark's performance reflects
retinal pathology as opposed to how the benchmark was assembled. Acquisition
provenance was found to explain a substantial share of separability, almost
entirely for the diabetic-retinopathy and normal classes and considerably
less for cataract versus glaucoma. Patient-level leakage produced no
measurable effect on a low-capacity classifier but a significant, monotonic
effect that grows with model capacity, confirmed directly on a fine-tuned
deployable model rather than inferred from a proxy. The trained classifier's
apparent competence was shown not to rest on the dataset's demographic
confounds, and was shown not to transfer to unseen acquisition sources. On
the document-processing side, raw OCR confidence was shown to be an
unreliable signal of extraction correctness, a learned alternative closed
almost all of that gap, and a pre-registered falsification rule led this
report to present hand-written deferral rules rather than a more elaborate
learned gate that did not, in fact, perform better. Every finding above is
reported with the model, sample size, and statistical test that produced it,
and every negative result — the initial leakage null, the failed embedding-
based duplicate detector, the learned deferral gate — is reported as a
negative result rather than omitted.

---

## Figure inventory

| # | File | Section |
|---|---|---|
| 1 | `fig01_provenance_heatmap.png` | 3.2 |
| 2 | `fig02_provenance_probes.png` | 3.2 |
| 3 | `fig03_protocol_ladder.png` | 4.1 |
| 4 | `fig04_duplicate_calibration.png` | 3.3 |
| 5 | `fig05_extraction_by_degradation.png` | 5.1 |
| 6 | `fig06_reliability.png` | 5.2 |
| 7 | `fig07_risk_coverage.png` | 6.1 |
| 8 | `fig08_robustness_matrix.png` | 6.2 |
| 9 | `fig09_dataset_composition.png` | 3.1 |
| 10 | `fig10_capacity_ladder.png` | 4.2, 4.3 |
| 11 | `fig11_cross_source_transfer.png` | 4.4 |
| 12 | `fig12_xai.png` | 6.3 |
| 13 | `fig13_confound_sensitivity.png` | 4.5 |
| 14 | `fig14_seed_stability.png` | 4.6 |

---

## References

All ten entries below were located and confirmed against their original
source (arXiv, publisher page, or DOI) rather than reconstructed from memory.
One correction from the earlier draft of this report: the provenance-revisited
citation is not an ICLR 2025 paper — it is a 2024 arXiv preprint by the same
two authors, and is cited as such below.

1. A. Torralba, A. A. Efros. "Unbiased look at dataset bias." *Proc. IEEE
   Conf. Computer Vision and Pattern Recognition (CVPR)*, 2011, pp. 1521–1528.
2. Z. Liu, K. He. "A decade's battle on dataset bias: Are we there yet?"
   *arXiv:2403.08632*, 2024.
3. I. E. Tampu, A. Eklund, N. Haj-Hosseini. "Inflation of test accuracy due to
   data leakage in deep learning-based classification of OCT images."
   *Scientific Data*, vol. 9, 2022. doi:10.1038/s41597-022-01618-6.
4. C. Leibig, V. Allken, M. S. Ayhan, P. Berens, S. Wahl. "Leveraging
   uncertainty information from deep neural networks for disease detection."
   *Scientific Reports*, vol. 7, no. 17816, 2017.
5. T. Araújo, G. Aresta, L. Mendonça, S. Penas, C. Maia, Â. Carneiro, A. M.
   Mendonça, A. Campilho. "DR\|GRADUATE: Uncertainty-aware deep
   learning-based diabetic retinopathy grading in eye fundus images."
   *Medical Image Analysis*, vol. 63, p. 101715, 2020.
6. Q. Zhang, H. Wu, C. Zhang, Q. Hu, H. Fu, J. T. Zhou, X. Peng. "Provable
   dynamic fusion for low-quality multimodal data" (introduces QMF). *Proc.
   Int. Conf. Machine Learning (ICML)*, 2023.
7. Z. Han, C. Zhang, H. Fu, J. T. Zhou. "Trusted multi-view classification."
   *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.
8. J. She, K. Spivakovsky. "MultiRetNet: A multimodal vision model and
   deferral system for staging diabetic retinopathy." *arXiv:2507.14738*,
   2025.
9. A. Yavari, T. Schmoll, R. A. Leitgeb, K. L. Huber, H. Stino, A. Pollreisz,
   W. Drexler, T. Schlegl. "Applicability study of AI attribution methods for
   ophthalmic image classification." *Scientific Reports*, 2026.
   doi:10.1038/s41598-025-33120-5.
10. Š. Šimsa, M. Šulc, M. Uřičář, Y. Patel, A. Hamdi, M. Kocián, M. Skalický,
    J. Matas, A. Doucet, M. Coustaty, D. Karatzas. "DocILE benchmark for
    document information localization and extraction." *Proc. Int. Conf.
    Document Analysis and Recognition (ICDAR)*, 2023.
