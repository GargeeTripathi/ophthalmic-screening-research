# Provenance-aware, patient-disjoint evaluation of ophthalmic screening with controlled document reliability

**Live site → [ophthalmic-screening.vercel.app](https://ophthalmic-screening.vercel.app)**

B.Tech CSE (Data Science), Semester V project.
Academic prototype. **Not a medical device, not a diagnostic tool, not clinically validated.**

---

## What this project is

An empirical study, plus a working screening prototype built around it.

Public four-class fundus benchmarks are widely reported at 95–96% accuracy. This project
asks how much of that is retinal pathology and how much is *where the image came from*,
re-evaluates standard classifiers under a corrected protocol, and builds a human-in-the-loop
screening workflow whose decision to defer a case is informed by a referral document — while
the disease prediction itself stays image-only.

**It is not a novel CNN architecture.** The image model is a standard backbone, correctly
evaluated. The contributions are empirical and are labelled honestly below.

### Contributions

| # | Contribution | Type |
|---|---|---|
| 1 | Acquisition-provenance and patient-leakage audit of a widely used four-class fundus benchmark | Empirical |
| 2 | Corrected patient-disjoint, deduplicated, provenance-controlled re-evaluation | Empirical |
| 3 | Calibration study of extraction-correctness estimators for clinical field extraction | Empirical |
| 4 | Controlled analysis mapping referral legibility and correctness onto triage quality | Empirical |
| 5 | Released synthetic ophthalmic referral extraction corpus with ground truth and sealed layouts | Resource |

The screening model adapts established selective-prediction and reliability-weighted fusion
methods (QMF, TMC, Leibig et al., MultiRetNet). **We claim its instantiation and evaluation,
not the mechanisms.**

### Two structural rules

1. **The document never changes the diagnosis.** `argmax(p_img)` comes from the fundus image
   alone. The referral and its extraction reliability only influence whether the system
   answers at all. This removes the text-dominance failure mode by construction rather than
   by penalty.
2. **No patient appears in two splits.** Enforced by assertion, not by convention.

---

## Live site

**[ophthalmic-screening.vercel.app](https://ophthalmic-screening.vercel.app)** — findings,
both app screens, all 14 figures and the full results.

The site is public; this repository is private. It carries the results and the disclaimers,
not the code. It is served with `X-Robots-Tag: noindex` so an unlisted prototype does not
end up in search results.

```bash
python scripts/build_site.py                  # rebuild docs/ from results/
npx vercel deploy docs --prod --yes           # redeploy
```

`docs/standalone.html` is the same site as one 1.3 MB self-contained file, for anywhere that
cannot host a directory. GitHub Pages is not used: Pages on a private repository needs a
paid plan.

The **upload form only works locally** (`python scripts/serve.py`) — screening runs a
PyTorch model that a static host cannot execute. The deployed pages say so rather than
offering a form that would fail silently.

## Results

See **[`reports/RESULTS.md`](reports/RESULTS.md)** — generated from `results/` by
`scripts/summarise_results.py`, so no number in it is transcribed by hand.
**[`reports/PAPER_OUTLINE.md`](reports/PAPER_OUTLINE.md)** maps every paper section onto the
experiment and figure that support it, plus the reviewer objections each one answers.

Three findings that tell one story — **performance on these benchmarks is substantially
source-bound**:

- **Provenance almost determines the label.** Majority-class baseline 26.0%; a four-row
  source lookup table alone reaches 69.1%; a metadata-only classifier reaches **66.7%**
  without seeing a single pixel. Every diabetic-retinopathy image is a `.jpeg` and no other
  class contains one. Within a single source that signal disappears (`odir_jpg512`: 0.640 →
  0.644), so provenance explains the DR and normal classes almost entirely and contributes
  little to separating cataract from glaucoma.
- **Patient leakage is real, and only memorisers exploit it.** The protocol ladder found
  nothing with a linear head (+0.005, p = 0.75) — but a linear head cannot memorise, and
  leakage is exploited *by* memorisation. Sweeping capacity instead of compute: 1-NN gains
  **+0.045 macro-F1** from ignoring patient identity, at p ≈ 0. Patient-disjoint splitting
  matters *more* as capacity rises.
- **The model does not transfer across sources.** Trained on ODIR, it scores *below*
  majority-class guessing on three of four unseen acquisition sources.
- **It is not a demographics detector.** Cataract-vs-rest AUROC holds at ~0.94 under age
  and sex matching, and age plus sex alone score *below* the majority baseline. The 11-year
  age gap between cataract and DR patients is real in the data but is not what the model uses.

Two methods notes worth publishing on their own:

- **ImageNet-embedding deduplication fails on fundus images.** At threshold 0.999 it
  recovers 0% of known duplicates; an FOV-normalised fingerprint recovers 97% at precision
  0.99.
- **Raw OCR confidence is near-useless for clinical field extraction** (AUROC 0.52 / 0.64).
  A learned correctness estimator reaches 0.98 / 0.99, and the feature importances explain
  why: value *format* and *length* carry the signal, OCR confidence contributes 0.08.

---

## Research questions

| RQ | Question | Experiment | Status |
|---|---|---|---|
| RQ1 | How much performance is acquisition provenance? | E0a / E0b / E0c, **E11** | done |
| RQ2 | How much is patient leakage and duplication? | E1, **E1b** | done |
| RQ3 | Is raw OCR confidence a useful reliability signal? | E8 | done |
| RQ4 | How does document quality propagate into triage? | E10 | done |
| RQ5 | Can document evidence improve deferral without changing the diagnosis? | E9 | done |

Every RQ has a falsification criterion in `configs/experiments.yaml`. Negative results are
reported, not hidden: RQ2 produced one that a follow-up experiment then overturned, and the
learned deferral gate was dropped by its own pre-registered kill rule.

---

## Layout

```
configs/     paths, data invariants, training, IDP corpus, experiment registry
src/
  data/        manifest, fingerprints, embeddings, dedup, splits
  provenance/  acquisition-source features
  idp/         document generator, OCR, extractors
  deferral/    deferral scoring and risk-coverage
  evaluation/  metrics, bootstrap, McNemar, calibration
  app/         grounded chatbot
scripts/     runnable entry points, one job each
colab/       train_models.ipynb — the GPU half
data/        manifests and splits committed; raw/processed/documents are not
results/     append-only experiment records
reports/     RESULTS.md, worklist.html
```

Source datasets are **read-only** and live outside this repo (`configs/paths.yaml`).

---

## Setup

Python 3.10+ (3.11 used here).

```bash
cd D:/ophthalmic-screening-research
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# or, for the exact versions this project was developed and measured on:
# .venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe scripts/health_check.py
.venv\Scripts\python.exe -m pytest
```

OCR uses **RapidOCR** (ONNX Runtime), not Tesseract — pip-installable, CPU, offline, no
system binary and no admin rights, which keeps the project reproducible in a clean
environment.

---

## Reproducing everything

One command. Every stage declares its outputs, so a stage that is already done is skipped —
safe to re-run, and a crash costs only the stage that was running.

```bash
python scripts/run_all.py            # everything not already done
python scripts/run_all.py --list     # show the plan and what is cached
python scripts/run_all.py --from E7  # resume from a named stage
python scripts/run_all.py --force    # ignore caches, redo everything
```

From cold: roughly **70 minutes** on a 14-thread CPU, dominated by fingerprints (~11 min),
embeddings (~18 min) and OCR (~35 min). Everything else is minutes.

<details>
<summary>The stages it runs, in order</summary>

```
manifest      build_manifest.py            5,058 images / 2,910 patients, counts asserted
merged        build_merged_manifest.py     provenance labels for the audit subject
fingerprints  build_fingerprints.py        FOV-normalised fingerprints
embeddings    build_embeddings.py          ResNet50 embeddings
dedup         find_duplicates.py           calibrated against free ground truth
splits        build_splits.py              four protocol rungs, disjointness asserted
E0            provenance_probe.py          E0a / E0b / E0c
E1            protocol_ladder.py           protocol ladder
E1b           capacity_ladder.py           does inflation need memorisation?
E5            cross_source_transfer.py     transfer to unseen sources
E11           confound_sensitivity.py      age / sex controls
documents     generate_documents.py        360 referrals, layouts D-F sealed
ocr           run_ocr.py                   OCR the corpus
E7            evaluate_extraction.py       extraction benchmark + reliability
E9            run_deferral.py              deferral + robustness matrix
E12           run_xai.py                   explainability battery
E13           statistical_validation.py    seed stability + Holm-Bonferroni
figures       make_figures.py              14 figures
report        summarise_results.py         reports/RESULTS.md
worklist      build_worklist.py            reports/worklist.html
```

</details>

The GPU half — E2/E3 fine-tuned baselines — is deliberately **not** in the orchestrator. It
runs in `colab/train_models.ipynb` and needs a one-time Drive upload. Nothing else depends
on it.

### Cost, measured

| Step | Time | Hardware |
|---|---:|---|
| Fingerprints (9,275 images) | 11.4 min | CPU, 14 threads |
| Embeddings (9,275 images) | 17.6 min | CPU |
| Everything else | < 10 min | CPU |
| E2/E3 fine-tuned baselines | ~6 h | Colab T4 |

---

## Local / Colab split

| Local (CPU) | Colab (GPU) |
|---|---|
| Manifests, splits, provenance, dedup | Fine-tuned backbone training (E2/E3) |
| Document generation, OCR, extraction | Batch inference over test folds |
| Reliability estimator, deferral, statistics | Grad-CAM batches (E12) |
| Figures, worklist, report | |

Every Colab run reads from Drive, checkpoints every epoch, writes predictions back, and
resumes from a fresh session. No experiment depends on a session staying alive.

---

## Limitations

- **Cataract and glaucoma have 203 and 182 patients.** Small-sample; intervals, not point
  estimates. A single test fold holds ~40 cataract patients, which is why out-of-fold
  predictions are pooled before any significance test.
- **ODIR is *less confounded*, not clean.** Cataract patients average ~11 years older than
  DR patients, sex distribution differs by class, and 23.2% of glaucoma rows name a second
  condition in the annotation.
- **The protocol ladder used a frozen linear head**, which cannot memorise and therefore
  cannot benefit from leakage. E1b addresses this by sweeping capacity, but every model in
  both experiments sits on frozen ImageNet features; a fine-tuned backbone could behave
  differently again.
- **Attribution (E12) is computed for the frozen-feature model**, not a fine-tuned
  diagnostic one. It shows attribution lands on retinal tissue (1.56x uniform) and passes a
  randomisation sanity check — it does not show the model is looking at a lesion.
- **The referral corpus is synthetic.** Every number from it is a technical evaluation under
  controlled degradation, never clinical validation. A search for a public de-identified
  ophthalmic referral corpus found none; that search is recorded rather than papered over.
- **The duplicate detector's recall is measured over ODIR↔merged ground-truth pairs.**
  Recall for within-ODIR duplicates is not directly measurable.
- **Attribution maps** show where attribution falls and whether it is faithful to the model.
  They do not show the model is looking at a lesion. No ophthalmologist reviewed anything.
- No prospective data, no clinical trial, no regulatory clearance.
