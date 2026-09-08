---
title: Ophthalmic Screening Prototype
emoji: 👁️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Provenance-aware ophthalmic screening — working prototype

**Academic screening prototype. Not a medical device and not a diagnosis.** A qualified
clinician reviews every case and makes every decision. Trained on a public research dataset
with 203 cataract and 182 glaucoma patients. The referral documents used in development were
synthetic. There has been no prospective clinical validation.

Upload a fundus photograph, and optionally an image of a referral letter.

## What it demonstrates

**The referral document never changes the finding.** That is enforced in code rather than
asserted: `predict_image()` does not take the document as a parameter. The referral, and how
reliably it could be read, decide only *whether* the system answers — never *what* it
answers.

So a case is deferred to a clinician when any of these holds:

- image confidence falls below 60%
- the referral's stated diagnosis contradicts the image finding
- the referral's fields were read with low estimated reliability

That last one is a research result in its own right. Raw OCR confidence is close to useless
as a signal for whether a clinical field was read correctly — AUROC 0.52. A learned
correctness estimator over value format, length and layout position reaches 0.98.

## This runs the real system

Everything here is the same code that produced the reported results: the image model, the
OCR, the layout-aware extraction, the reliability estimator and the deferral rules. Nothing
is reimplemented for the web.

The queue starts empty. The ODIR images the study used are not ours to redistribute, so this
instance screens what you upload rather than shipping a pre-screened worklist.

## Findings, figures and full results

The research write-up lives at **https://ophthalmic-screening.vercel.app** — the provenance
audit, the capacity ladder, all 14 figures and the complete results. That site also screens
a fundus image in your browser, but it cannot read a referral: the OCR stack needs Python,
which is why this Space exists.
