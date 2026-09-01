# Paper outline — every claim traced to an experiment

This maps the report's structure onto the experiments that support it. **No sentence in the
paper should make a claim that cannot point at a row here.** Numbers come from
`reports/RESULTS.md`, which is generated from `results/` rather than typed by hand.

---

## Title and framing

> Provenance-aware, patient-disjoint evaluation of ophthalmic screening with controlled
> document reliability

The paper is an **empirical audit plus a working prototype**. It is not a methods paper and
must not be submitted as one. The contribution statement in `README.md` is the honest version;
use it verbatim.

---

## 1. Introduction

| Claim | Evidence |
|---|---|
| Public four-class fundus benchmarks report 95–96% accuracy | Literature (EfficientNetB3, DenseNet169 studies, 2025) |
| Those numbers are of unverified provenance | Motivates the paper; do not assert, ask |
| A screening tool that answers on every image is not deployable | Argument, not a result — say so |

**Do not** open by claiming the field is wrong. Open by asking what the numbers measure.

## 2. Related work

Cite and position against, honestly:

- **Provenance / dataset bias** — Torralba & Efros (CVPR 2011) "name that dataset"; Liu & He
  (ICLR 2025). Our probe is an *application* of their design, not a new one.
- **Leakage inflation** — OCT leakage (Sci Data 2022) is the template for this paper type.
- **Selective prediction in ophthalmology** — Leibig et al. 2017; DR|GRADUATE 2020.
- **Reliability-weighted fusion** — QMF (ICML 2023); TMC (ICLR 2021); MultiRetNet 2025.
- **Attribution evaluation in ophthalmology** — Sci Rep 2026.
- **Document IE benchmarks** — DocILE (ICDAR 2023); held-out layouts is their convention.

## 3. Data and audit

| Section | Evidence | Figure/Table |
|---|---|---|
| Dataset composition | `manifest_summary.json` — 5,058 images / 2,910 patients, counts asserted | Fig 9, Tab 1 |
| Provenance structure of the merged benchmark | E0a/E0b/E0c-i | **Fig 1**, Fig 2 |
| Anti-circularity control | E0c-ii — within-source class prediction | Fig 2 |
| Deduplication | `deduplication_report.json` | Fig 4 |
| Splits | `splits_summary.json`, patient-disjointness asserted | Tab 2 |

**Signature figure: Fig 1** (source × class heatmap). It carries the paper.

## 4. Results

| Result | Experiment | Figure |
|---|---|---|
| Provenance predicts label without pixels | E0a | Fig 2 |
| Within a source the signal vanishes | E0c-ii | Fig 2 |
| Protocol ladder — no inflation with a linear head | E1 | Fig 3 |
| **Inflation appears with memorisation capacity** | **E1b** | **Fig 10** |
| Model does not transfer across sources | E5 | Fig 11 |
| Result survives demographic controls | E11 | Fig 13 |
| Seed stability + Holm-Bonferroni | E13 | Fig 14 |

**The narrative arc:** source predicts label → memorisers exploit patient leakage → models
collapse across sources. Three legs, one conclusion: performance on these benchmarks is
substantially source-bound.

**Report E1 and E1b together, in that order.** E1's null and its correction by E1b is the
most instructive sequence in the paper, and hiding the null would be dishonest.

## 5. Document processing

| Result | Experiment | Figure |
|---|---|---|
| Corpus design, sealed layouts | `corpus_summary.json` | — |
| Extraction degrades with legibility | E7 | Fig 5 |
| Raw OCR confidence is near-useless | E8 | **Fig 6** |
| A learned correctness estimator is calibrated | E8 | Fig 6 |

State the synthetic-data caveat **in the same paragraph as every number**, not only in
Limitations.

## 6. Triage

| Result | Experiment | Figure |
|---|---|---|
| Document evidence roughly halves AURC | E9 | **Fig 7** |
| The learned gate does not beat two rules | E9 | Fig 7 |
| Robustness across evidence conditions | E10 | Fig 8 |
| Attribution lands on retina; sanity check passes | E12 | Fig 12 |

**Report the negative result plainly.** The learned deferral gate and the `c_hat` input were
dropped by a kill rule written before the run. That is a feature of the methodology, not an
embarrassment — say so in one sentence and move on.

## 7. Limitations

Take these verbatim from `README.md`. The non-negotiable ones:

- 203 cataract and 182 glaucoma patients — intervals, not point estimates.
- Every model sits on frozen ImageNet features; a fine-tuned backbone may differ.
- The referral corpus is synthetic; technical evaluation, never clinical validation.
- Attribution shows *where*, not *what*. No lesion masks, no ophthalmologist.
- No prospective data, no clinical trial, no regulatory clearance.

## 8. Conclusion

One paragraph. No new claims. Do not use the words *novel*, *first*, *state-of-the-art*, or
*clinically validated* anywhere in the paper.

---

## Figure inventory

| # | File | Section |
|---|---|---|
| 1 | `fig01_provenance_heatmap.png` | 3 — signature figure |
| 2 | `fig02_provenance_probes.png` | 3, 4 |
| 3 | `fig03_protocol_ladder.png` | 4 |
| 4 | `fig04_duplicate_calibration.png` | 3 |
| 5 | `fig05_extraction_by_degradation.png` | 5 |
| 6 | `fig06_reliability.png` | 5 |
| 7 | `fig07_risk_coverage.png` | 6 |
| 8 | `fig08_robustness_matrix.png` | 6 |
| 9 | `fig09_dataset_composition.png` | 3 |
| 10 | `fig10_capacity_ladder.png` | 4 — the correction to E1 |
| 11 | `fig11_cross_source_transfer.png` | 4 |
| 12 | `fig12_xai.png` | 6 |
| 13 | `fig13_confound_sensitivity.png` | 4 |
| 14 | `fig14_seed_stability.png` | 4 |

If a figure does not appear in this table, it does not go in the paper.

---

## Reviewer objections to pre-empt, and where each is answered

| Objection | Answered by |
|---|---|
| "Every method here is off the shelf." | Conceded in the contribution statement. The claim is empirical. |
| "That a merged Kaggle set is confounded is obvious." | Obvious in principle, unmeasured in fact — E0a quantifies it. |
| "Your probe just detects file format." | E0b (normalised pixels) and E0c-i (geometry only). |
| "The probe may be reading pathology." | E0c-ii — within-source, the signal vanishes. |
| "ODIR is confounded too." | E11 — survives age and sex matching; demographics alone are useless. |
| "You used too weak a model to see leakage." | E1b — a pure memoriser shows +0.045 at p ≈ 0. |
| "One seed proves nothing." | E13 — three seeds. |
| "Twelve tests, no correction." | E13 — Holm-Bonferroni over the whole family. |
| "Synthetic documents prove nothing." | Conceded; framed as controlled robustness analysis throughout. |
| "No clinical validation." | Conceded entirely, in Limitations and on every UI screen. |
