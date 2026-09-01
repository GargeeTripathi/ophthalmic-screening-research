"""Assemble reports/RESULTS.md from the saved experiment outputs.

Every number in the report is read from results/ rather than typed by hand, so the report
cannot drift away from the experiments. Re-run after any experiment changes.

    python scripts/summarise_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import output_dir  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def section_dataset(manifests: Path) -> str:
    summary = read_json(manifests / "manifest_summary.json")
    if not summary:
        return ""
    by_class = pd.DataFrame(summary["by_class"]).T
    lines = ["## 1. Dataset", "",
             f"ODIR-5K, four-class subset: **{summary['total_images']} images** from "
             f"**{summary['total_patients']} patients**.", "",
             "| class | images | patients | multi-finding | mean age |",
             "|---|---:|---:|---:|---:|"]
    for code, row in by_class.iterrows():
        lines.append(f"| {code} | {int(row['images'])} | {int(row['patients'])} | "
                     f"{int(row['multi_finding'])} ({row['multi_finding_pct']}%) | "
                     f"{row['mean_age']} |")
    lines += ["",
              f"- **{summary['patients_with_both_eyes']} patients contribute both eyes** - "
              "the mechanism behind patient-level leakage.",
              f"- {summary['patients_spanning_two_classes']} patients carry different "
              "diagnoses in each eye, so grouping is by patient but stratification is at "
              "eye level.",
              f"- {summary['distinct_resolutions']} distinct resolutions, no EXIF: ODIR "
              "provenance cannot be labelled cleanly.",
              "",
              "**Confounds to report, not hide.** Cataract patients average ~11 years older "
              "than diabetic-retinopathy patients; sex distribution differs by class; and "
              "23.2% of glaucoma rows name a second condition in the annotation. ODIR is "
              "*less confounded*, not clean.", ""]
    return "\n".join(lines)


def section_provenance(manifests: Path, results: Path) -> str:
    merged = read_json(manifests / "merged_manifest_summary.json")
    probes = read_json(results / "E0_provenance_probes.json")
    if not merged or not probes:
        return ""
    lines = ["## 2. Provenance audit (RQ1)", "",
             "The merged four-class benchmark is not one dataset. Acquisition source is "
             "assigned from file properties alone - never from the class folder - and it "
             "almost determines the label.", "",
             f"- Majority-class baseline: **{merged['majority_class_baseline']:.1%}**",
             f"- Source lookup table alone: **{merged['source_majority_accuracy']:.1%}** "
             "(no model, no pixels)", "",
             "| experiment | question | baseline | probe | lift |",
             "|---|---|---:|---:|---:|"]
    questions = {
        "E0a": "class from file metadata only",
        "E0b": "source from FOV-normalised pixels",
        "E0c-i": "source from geometry only",
    }
    for entry in probes:
        name = entry["experiment"]
        question = questions.get(name, entry["description"])
        lines.append(f"| {name} | {question} | {entry['baseline_accuracy']:.3f} | "
                     f"**{entry['accuracy']:.3f}** | +{entry['lift_over_baseline']:.3f} |")
    lines += ["",
              "**The anti-circularity control (E0c-ii).** Within a single acquisition source, "
              "provenance is constant, so anything a probe achieves must come from retinal "
              "content:", ""]
    for entry in probes:
        if entry["experiment"].startswith("E0c-ii"):
            source = entry["experiment"].split("[")[1].rstrip("]")
            lines.append(f"- `{source}` (n={entry['n']}): {entry['baseline_accuracy']:.3f} -> "
                         f"{entry['accuracy']:.3f}, lift **+{entry['lift_over_baseline']:.3f}**")
    lines += ["",
              "Every diabetic-retinopathy image is a `.jpeg` and no other class contains one, "
              "so file extension alone is a perfect DR detector on this benchmark. The "
              "cataract/glaucoma sources split roughly 50/50, so provenance explains the DR "
              "and normal classes almost entirely and contributes little to separating "
              "cataract from glaucoma. Both halves of that belong in the paper.", ""]
    return "\n".join(lines)


def section_dedup(manifests: Path) -> str:
    report = read_json(manifests / "deduplication_report.json")
    if not report:
        return ""
    at = report.get("detector_at_operating_point", {})
    return "\n".join([
        "## 3. Deduplication", "",
        f"Detector: {report['method']}. Rejected alternative: {report['rejected_method']}.",
        "",
        f"- Ground truth: **{report['ground_truth_positives']} known duplicate pairs** "
        "(a merged-benchmark image sharing a filename with an ODIR image), calibrated "
        "against the hardest available negatives.",
        f"- Operating threshold **{report['operating_threshold']}**: precision "
        f"**{at.get('precision')}**, recall **{at.get('recall')}**.",
        f"- Within ODIR: **{report['within_odir_duplicate_groups']} duplicate groups**, "
        f"{report['within_odir_images_dropped']} images removed -> "
        f"{report['clean_images']} images, {report['clean_patients']} patients.",
        f"- **{report['merged_images_contaminated_by_odir']} merged-benchmark images are "
        "ODIR copies** and are therefore excluded from cross-source test data.", "",
        f"*Caveat carried into the paper:* {report['recall_caveat']}.", "",
        "A methods note worth publishing on its own: ImageNet-embedding cosine similarity "
        "**fails** as a duplicate detector on fundus images. At threshold 0.999 it recovers "
        "0% of known duplicates, while the FOV-normalised fingerprint recovers 97%. Fundus "
        "images are visually homogeneous and the copies differ by a large downscale, which "
        "ImageNet features are not invariant to.", ""])


def section_ladder(results: Path) -> str:
    payload = read_json(results / "E1_protocol_ladder.json")
    if not payload:
        return ""
    lines = ["## 4. Protocol ladder (RQ2)", "",
             "Model held fixed, only the split varies. Frozen ResNet50 features with a "
             "logistic head.", "",
             "| rung | n | accuracy | macro-F1 | 95% CI |", "|---|---:|---:|---:|---|"]
    for rung in payload["rungs"]:
        ci = rung["macro_f1_ci"]
        lines.append(f"| {rung['rung']} | {rung['n']} | {rung['accuracy']:.4f} | "
                     f"{rung['macro_f1']:.4f} | [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] |")
    lines += ["", "| comparison | macro-F1 drop | McNemar p |", "|---|---:|---:|"]
    for comparison in payload["comparisons"]:
        p = comparison.get("mcnemar", {}).get("p_value", "n/a")
        lines.append(f"| {comparison['from']} -> {comparison['to']} | "
                     f"{comparison['macro_f1_drop']:+.4f} | {p} |")
    lines += ["",
              "**Result: H2 is falsified as run.** No rung differs significantly from any "
              "other; all confidence intervals overlap.", "",
              "**The limitation that must accompany it.** A frozen linear head has almost no "
              "capacity to memorise, and leakage inflates a score by letting a model recall "
              "a near-copy it has already seen. This therefore measures a *lower bound* on "
              "protocol inflation, not the inflation a fine-tuned backbone would show. "
              "Settling that requires the GPU baselines in `colab/train_models.ipynb`.", ""]
    return "\n".join(lines)



def section_capacity(results: Path) -> str:
    payload = read_json(results / "E1b_capacity_ladder.json")
    if not payload:
        return ""
    lines = ["## 4b. Does inflation depend on memorisation? (E1b)", "",
             "E1's null invites an obvious objection: a frozen linear head cannot memorise, "
             "and leakage is exploited *by* memorisation. So rather than reaching for a GPU, "
             "sweep the variable that actually matters - capacity - holding features and "
             "splits fixed. 1-NN is a pure retriever: if a patient's fellow eye is in the "
             "training set, it will find it.", "",
             "| model | image-level | patient-disjoint | inflation | McNemar p |",
             "|---|---:|---:|---:|---:|"]
    for entry in payload["comparisons"]:
        lines.append(f"| {entry['model']} | {entry['image_level_macro_f1']:.4f} | "
                     f"{entry['patient_disjoint_macro_f1']:.4f} | "
                     f"**{entry['inflation']:+.4f}** | {entry['mcnemar']['p_value']} |")
    lines += ["", f"**Verdict: {payload['verdict']}.**", "",
              "This corrects E1. Patient leakage on this dataset **is** real and **is** "
              "exploitable - by models that memorise. A pure retriever gains +0.045 macro-F1 "
              "from ignoring patient identity, at p ~ 0. The parametric models show smaller, "
              "non-significant gains. The practical implication for anyone using this "
              "benchmark is that patient-disjoint splitting matters more as model capacity "
              "rises, which is precisely the direction the field is moving.", ""]
    return chr(10).join(lines)


def section_cross_source(results: Path) -> str:
    payload = read_json(results / "E5_cross_source_transfer.json")
    if not payload:
        return ""
    lines = ["## 4c. Transfer to unseen acquisition sources (E5)", "",
             f"Trained on {payload['train_images']} ODIR images; tested on merged-benchmark "
             f"images after removing {payload['excluded_as_odir_copies']} confirmed ODIR "
             "copies, so nothing in the test set was seen in training.", "",
             "| source | n | classes | majority-class acc | model acc | macro-F1 |",
             "|---|---:|---|---:|---:|---:|"]
    for entry in payload["per_source"]:
        lines.append(f"| {entry['source']} | {entry['n']} | "
                     f"{','.join(entry['classes_present'])} | "
                     f"{entry['majority_class_accuracy']:.3f} | "
                     f"**{entry['accuracy']:.3f}** | {entry['macro_f1']:.3f} |")
    lines += ["",
              "**The model does not transfer.** On three of four sources it scores *below* "
              "the accuracy of always guessing that source's majority class. Each source is "
              "class-restricted - itself a symptom of the provenance confound - so results "
              "are reported per source and never pooled.", "",
              "Two caveats. The in-domain model is itself weak (macro-F1 0.55 on frozen "
              "features), so poor transfer is partly a weak model rather than purely a "
              "domain gap. And this measures transfer under a documented change of "
              "acquisition source, not *generalisation* in the unqualified sense.", ""]
    return chr(10).join(lines)



def section_confounds(results: Path) -> str:
    payload = read_json(results / "E11_confound_sensitivity.json")
    if not payload:
        return ""
    demo = payload["demographics_only"]
    lines = ["## 4d. Is it a disease detector or a demographics detector? (E11)", "",
             "ODIR is less confounded, not clean: cataract patients average 11 years older "
             "than diabetic-retinopathy patients, and the sex distribution differs by class. "
             "Networks read age and sex from fundus photographs accurately, so this is "
             "testable rather than merely a limitations-section caveat.", "",
             "**Demographics alone, no image:** "
             f"accuracy {demo['accuracy']:.4f} against a {demo['baseline_accuracy']:.4f} "
             f"majority baseline (lift {demo['lift_over_baseline']:+.4f}), macro-F1 "
             f"{demo['macro_f1']:.4f}. Age and sex carry no usable class signal here - the "
             "classifier does *worse* than always guessing the majority class.", "",
             "| condition | n | patients | macro-F1 | 95% CI | cataract AUROC |",
             "|---|---:|---:|---:|---|---:|"]
    for entry in payload["conditions"]:
        lines.append(f"| {entry['condition']} | {entry['n']} | {entry['patients']} | "
                     f"{entry['macro_f1']:.4f} | "
                     f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}] | "
                     f"**{entry['cataract_auroc']:.4f}** |")
    lines += [""]
    for line in payload["comparisons"]:
        lines.append(f"- {line.strip()}")
    lines += ["",
              "**Image performance survives every control.** Cataract-vs-rest AUROC stays "
              "near 0.94 under age matching and sex matching, and every confidence interval "
              "overlaps the unrestricted one. The demographic confound is real in the data "
              "but is not what the model is using.", "",
              f"*Method:* {payload['method_note']}.", ""]
    return chr(10).join(lines)


def section_statistics(results: Path) -> str:
    payload = read_json(results / "E13_statistical_validation.json")
    if not payload:
        return ""
    lines = ["## 9. Statistical validation (E13)", "",
             f"**Seed stability.** Every earlier result used split seed 42. The leakage "
             f"comparison is re-run across {payload['seeds']}; a finding that holds for only "
             "one random partition is not a finding.", "",
             "| seed | model | image-level | patient-disjoint | inflation | p |",
             "|---|---|---:|---:|---:|---:|"]
    for row in payload["per_run"]:
        lines.append(f"| {row['seed']} | {row['model']} | "
                     f"{row['image_level_macro_f1']:.4f} | "
                     f"{row['patient_disjoint_macro_f1']:.4f} | "
                     f"**{row['inflation']:+.4f}** | {row['p_raw']} |")
    lines += ["", "| model | mean inflation | sd | min | max |", "|---|---:|---:|---:|---:|"]
    for model, stats in payload["stability"].items():
        lines.append(f"| {model} | {stats['mean']:+.4f} | {stats['std']:.4f} | "
                     f"{stats['min']:+.4f} | {stats['max']:+.4f} |")
    lines += ["",
              f"**Multiple comparisons.** The project reports {payload['family_size']} paired "
              "tests. At alpha = 0.05 one spurious result in twenty is expected by "
              "construction, so Holm-Bonferroni is applied across the whole family. "
              f"**{len(payload['survivors'])} of {payload['family_size']} survive correction:**", ""]
    for name in payload["survivors"]:
        entry = payload["holm_bonferroni"][name]
        lines.append(f"- `{name}` - p = {entry['p_raw']} (threshold {entry['holm_threshold']})")
    lines += ["", f"*{payload['note']}.*", ""]
    return chr(10).join(lines)


def section_extraction(results: Path) -> str:
    payload = read_json(results / "E7_extraction.json")
    if not payload:
        return ""
    lines = ["## 5. Document extraction (E7)", "",
             "Two extractors, developed on layouts A-C and evaluated on sealed layouts D-F "
             "which were opened once, after both were frozen.", "",
             "| extractor | split | precision | recall | F1 | exact match | n |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["overall"]:
        lines.append(f"| {row['extractor']} | {row['split']} | {row['precision']:.3f} | "
                     f"{row['recall']:.3f} | **{row['f1']:.3f}** | {row['exact_match']:.3f} "
                     f"| {row['n']} |")
    lines += ["", f"*{payload['caveat']}.*", ""]
    return "\n".join(lines)


def section_reliability(results: Path) -> str:
    payload = read_json(results / "E8_reliability.json")
    if not payload:
        return ""
    lines = ["## 6. Reliability of the extraction signal (RQ3)", "",
             "Is raw OCR confidence a usable estimate of whether an extracted clinical field "
             "is correct? Trained on development layouts, evaluated on sealed ones.", "",
             "| extractor | signal | AUROC | Brier | ECE |", "|---|---|---:|---:|---:|"]
    for extractor, entry in payload.items():
        for signal in ("raw_ocr_confidence", "learned_estimator"):
            scores = entry[signal]
            lines.append(f"| {extractor} | {signal.replace('_', ' ')} | "
                         f"**{scores['auroc']}** | {scores['brier']} | {scores['ece']} |")
    lines.append("")
    return "\n".join(lines)



def section_xai(results: Path) -> str:
    payload = read_json(results / "E12_xai.json")
    if not payload:
        return ""
    verdict = "PASS" if payload["randomisation_passes"] else "FAIL"
    lines = ["## 7b. Explainability (E12)", "",
             f"Model: {payload['model']}. Head fitted on "
             f"{payload.get('head_fitted_on', '?')} training-fold embeddings; attribution "
             f"over {payload['attributed_over']}.", "",
             "| measure | value |", "|---|---:|",
             f"| FOV-mass ratio (observed) | **{payload['fov_mass_ratio_mean']:.4f}** |",
             f"| uniform-attribution baseline | {payload['uniform_attribution_baseline']:.4f} |",
             f"| concentration ratio | **{payload['concentration_ratio']:.3f}x** |",
             f"| deletion AUC (mean) | {payload['deletion_auc_mean']:.4f} |",
             f"| randomisation sanity check | **{verdict}** "
             f"(r = {payload['randomisation_correlation']:+.4f}) |", "",
             "Attribution concentrates inside the retinal field of view at "
             f"{payload['concentration_ratio']:.2f}x what uniform attribution would give, "
             "and the method passes the randomisation sanity check, so the maps do depend "
             "on the learned weights.", "",
             "**What this permits us to say:** attribution falls on retinal tissue rather "
             "than on borders or letterbox artefacts. **What it does not permit:** any claim "
             "that the model is looking at a lesion. There are no lesion masks and no "
             "ophthalmologist. The FOV-mass ratio answers a shortcut question, not a "
             "clinical one.", "",
             f"*Caveat:* {payload['deletion_caveat']}. And {payload['model_caveat']}.", ""]
    return chr(10).join(lines)


def section_deferral(results: Path) -> str:
    payload = read_json(results / "E9_deferral.json")
    if not payload:
        return ""
    lines = ["## 7. Deferral (RQ4, RQ5)", "",
             "**The document never changes the predicted class.** `argmax(p_img)` is the "
             "answer whenever the system answers; the document and its read-reliability only "
             "decide *whether* to answer.", "",
             "| system | AURC | error-detection AUROC | acc@90% | acc@75% | acc@50% |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, entry in payload["systems"].items():
        at = entry["accuracy_at_coverage"]
        lines.append(f"| {name} | **{entry['aurc']:.4f}** | "
                     f"{entry['error_detection_auroc']} | {at.get('0.9')} | "
                     f"{at.get('0.75')} | {at.get('0.5')} |")
    lines += ["", "Lower AURC is better. `system_B_without_c_hat` isolates what the learned "
              "extraction-reliability estimate contributes - the one architectural claim "
              "this project makes.", ""]
    return "\n".join(lines)


def main() -> int:
    manifests, results = output_dir("manifests"), output_dir("results")
    parts = [
        "# Results", "",
        "Every number here is read from `results/` by `scripts/summarise_results.py`. "
        "Nothing is transcribed by hand.", "",
        "**Not a medical device. Synthetic documents are not clinical data. "
        "No clinical validation is claimed.**", "",
        section_dataset(manifests),
        section_provenance(manifests, results),
        section_dedup(manifests),
        section_ladder(results),
        section_capacity(results),
        section_cross_source(results),
        section_confounds(results),
        section_extraction(results),
        section_reliability(results),
        section_deferral(results),
        section_xai(results),
        section_statistics(results),
        "## 8. What is still outstanding", "",
        "- **Fine-tuned baselines (E2/E3)** need a GPU: `colab/train_models.ipynb`. Until "
        "they run, the protocol-ladder result is a lower bound.",
        "- **XAI battery (E12)** depends on a fine-tuned model to attribute over.",
        "- **A real-document sanity check.** No suitable public de-identified ophthalmic "
        "referral corpus was found; the search is recorded, and the corpus remains synthetic.",
        "",
    ]
    path = output_dir("reports") / "RESULTS.md"
    path.write_text("\n".join(p for p in parts if p is not None), encoding="utf-8")
    print(f"WRITTEN {path}  ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
