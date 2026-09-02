"""Generate every figure in the report from the saved results.

Each figure is regenerable from files in results/ and data/, so nothing in the report is a
screenshot of something that no longer exists.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import output_dir  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
INK, ACCENT, WARN = "#141D1C", "#0E5C58", "#B85A11"


def save(figure, name: str) -> None:
    path = output_dir("figures") / name
    figure.savefig(path)
    plt.close(figure)
    print(f"  {name}")


def fig_provenance_heatmap(manifests: Path) -> None:
    merged = pd.read_csv(manifests / "merged_manifest.csv", encoding="utf-8")
    table = pd.crosstab(merged["source"], merged["label"])
    share = table.div(table.sum(axis=1), axis=0) * 100

    figure, axis = plt.subplots(figsize=(5.4, 2.9))
    image = axis.imshow(share.values, cmap="BuPu", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(share.columns)), share.columns)
    axis.set_yticks(range(len(share.index)), share.index)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            value = share.values[i, j]
            axis.text(j, i, f"{value:.0f}%\n({table.values[i, j]})", ha="center", va="center",
                      fontsize=7.5, color="white" if value > 55 else INK)
    axis.set_title("Acquisition source almost determines the label", fontsize=10, pad=8)
    axis.set_xlabel("class"); axis.set_ylabel("source")
    axis.grid(False)
    figure.colorbar(image, ax=axis, label="% of source")
    save(figure, "fig01_provenance_heatmap.png")


def fig_provenance_probes(results: Path) -> None:
    frame = pd.read_csv(results / "E0_provenance_probes.csv")
    figure, axis = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(frame))
    axis.barh(y, frame["baseline_accuracy"], color="#C9D6D3", label="majority baseline")
    axis.barh(y, frame["accuracy"] - frame["baseline_accuracy"],
              left=frame["baseline_accuracy"], color=ACCENT, label="probe gain")
    axis.set_yticks(y, frame["experiment"])
    axis.invert_yaxis()
    axis.set_xlabel("accuracy"); axis.set_xlim(0, 1)
    axis.set_title("What can be predicted without looking at pathology", fontsize=10)
    for i, row in frame.iterrows():
        axis.text(row["accuracy"] + 0.012, i, f"{row['accuracy']:.3f}", va="center", fontsize=7.5)
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    save(figure, "fig02_provenance_probes.png")


def fig_protocol_ladder(results: Path) -> None:
    payload = json.loads((results / "E1_protocol_ladder.json").read_text(encoding="utf-8"))
    rungs = payload["rungs"]
    names = [r["rung"] for r in rungs]
    point = [r["macro_f1_ci"]["point"] for r in rungs]
    low = [p - r["macro_f1_ci"]["ci_low"] for p, r in zip(point, rungs)]
    high = [r["macro_f1_ci"]["ci_high"] - p for p, r in zip(point, rungs)]

    figure, axis = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(names))
    axis.errorbar(x, point, yerr=[low, high], fmt="o-", color=ACCENT,
                  capsize=4, linewidth=1.6, markersize=6)
    axis.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
    axis.set_ylabel("macro-F1 (patient-level 95% CI)")
    axis.set_title("Protocol ladder: no significant inflation on ODIR\n"
                   "(frozen linear head - a lower bound, see caption)", fontsize=9.5)
    save(figure, "fig03_protocol_ladder.png")


def fig_duplicate_calibration(manifests: Path) -> None:
    frame = pd.read_csv(manifests / "duplicate_detector_calibration.csv")
    figure, axis = plt.subplots(figsize=(5.6, 3.0))
    for method, style in (("fingerprint", "-"), ("imagenet_embedding", "--")):
        block = frame[frame["method"] == method]
        if block.empty:
            continue
        colour = ACCENT if method == "fingerprint" else WARN
        axis.plot(block["threshold"], block["recall"], style, color=colour,
                  label=f"{method} recall", linewidth=1.6)
        axis.plot(block["threshold"], block["precision"], style, color=colour,
                  alpha=0.45, label=f"{method} precision", linewidth=1.3)
    axis.set_xlabel("cosine threshold"); axis.set_ylabel("score")
    axis.set_title("Duplicate detector: FOV fingerprint vs ImageNet embedding", fontsize=10)
    axis.legend(fontsize=7.5, frameon=False)
    save(figure, "fig04_duplicate_calibration.png")


def fig_extraction_by_degradation(results: Path) -> None:
    payload = json.loads((results / "E7_extraction.json").read_text(encoding="utf-8"))
    by_level = payload["by_level"]
    figure, axis = plt.subplots(figsize=(5.8, 3.0))
    for (extractor, split), colour, style in (
        (("E1", "dev"), ACCENT, "--"), (("E1", "sealed"), ACCENT, "-"),
        (("E2", "dev"), WARN, "--"), (("E2", "sealed"), WARN, "-"),
    ):
        levels, scores = [], []
        for level, mapping in by_level.items():
            key = str((extractor, split))
            for k, v in mapping.items():
                if str(k) == key or (isinstance(k, str) and extractor in k and split in k):
                    levels.append(int(level)); scores.append(v)
        if levels:
            order = np.argsort(levels)
            axis.plot(np.array(levels)[order], np.array(scores)[order], style, color=colour,
                      marker="o", markersize=4, linewidth=1.5,
                      label=f"{extractor} {split}")
    axis.set_xlabel("degradation level"); axis.set_ylabel("field F1")
    axis.set_title("Extraction degrades with document legibility", fontsize=10)
    axis.legend(fontsize=8, frameon=False)
    save(figure, "fig05_extraction_by_degradation.png")


def fig_reliability(results: Path) -> None:
    payload = json.loads((results / "E8_reliability.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))

    extractors = list(payload)
    width = 0.35
    x = np.arange(len(extractors))
    for offset, (key, colour, label) in enumerate((
        ("raw_ocr_confidence", WARN, "raw OCR confidence"),
        ("learned_estimator", ACCENT, "learned estimator"),
    )):
        values = [payload[e][key]["auroc"] or 0 for e in extractors]
        axes[0].bar(x + offset * width, values, width, color=colour, label=label)
    axes[0].axhline(0.5, color=INK, linestyle=":", linewidth=1)
    axes[0].set_xticks(x + width / 2, extractors)
    axes[0].set_ylabel("AUROC for 'extraction is correct'")
    axes[0].set_title("Is OCR confidence a correctness signal?", fontsize=9.5)
    axes[0].legend(fontsize=7.5, frameon=False)

    for extractor in extractors:
        curve = payload[extractor]["learned_calibration_curve"]
        axes[1].plot(curve["mean_predicted"], curve["observed_fraction"], "o-",
                     markersize=4, linewidth=1.4, label=extractor)
    axes[1].plot([0, 1], [0, 1], ":", color=INK, linewidth=1)
    axes[1].set_xlabel("predicted P(correct)"); axes[1].set_ylabel("observed fraction correct")
    axes[1].set_title("Learned estimator calibration", fontsize=9.5)
    axes[1].legend(fontsize=7.5, frameon=False)
    save(figure, "fig06_reliability.png")


def fig_risk_coverage(results: Path) -> None:
    payload = json.loads((results / "E9_deferral.json").read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(5.8, 3.2))
    palette = {"image_confidence_only": "#9AA9A6", "system_A_rules": WARN,
               "system_B_learned": ACCENT, "system_B_without_c_hat": "#7FA8A4"}
    for name, curve in payload["curves"].items():
        coverage = [p["coverage"] for p in curve]
        accuracy = [p["accuracy"] for p in curve]
        aurc = payload["systems"][name]["aurc"]
        axis.plot(coverage, accuracy, "-", color=palette.get(name, INK), linewidth=1.7,
                  label=f"{name}  (AURC {aurc:.3f})")
    axis.set_xlabel("coverage (fraction answered)")
    axis.set_ylabel("accuracy on retained cases")
    axis.set_title("Risk-coverage: the document affects deferral, never the class", fontsize=9.5)
    axis.legend(fontsize=7.5, frameon=False, loc="lower left")
    save(figure, "fig07_risk_coverage.png")


def fig_robustness_matrix(results: Path) -> None:
    frame = pd.read_csv(results / "E10_robustness_matrix.csv")
    pivot = frame.pivot(index="condition", columns="degradation_level",
                        values="accuracy_at_75pct")
    figure, axis = plt.subplots(figsize=(5.8, 3.2))
    image = axis.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    axis.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5)
    axis.set_xlabel("document degradation level")
    axis.set_title("Triage accuracy at 75% coverage, by evidence condition", fontsize=9.5)
    axis.grid(False)
    figure.colorbar(image, ax=axis, label="accuracy")
    save(figure, "fig08_robustness_matrix.png")


def fig_dataset_composition(manifests: Path) -> None:
    odir = pd.read_csv(manifests / "odir_manifest.csv", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))

    counts = odir.groupby("label").agg(images=("image_id", "size"),
                                       patients=("patient_id", "nunique"))
    x = np.arange(len(counts))
    axes[0].bar(x - 0.2, counts["images"], 0.4, color=ACCENT, label="images")
    axes[0].bar(x + 0.2, counts["patients"], 0.4, color="#9AA9A6", label="patients")
    axes[0].set_xticks(x, counts.index)
    axes[0].set_title("ODIR class distribution", fontsize=9.5)
    axes[0].legend(fontsize=8, frameon=False)

    for code, block in odir.groupby("label"):
        axes[1].hist(block["age"].dropna(), bins=20, histtype="step", linewidth=1.5, label=code)
    axes[1].set_xlabel("age"); axes[1].set_ylabel("images")
    axes[1].set_title("Age differs by class - a confound to report", fontsize=9.5)
    axes[1].legend(fontsize=8, frameon=False)
    save(figure, "fig09_dataset_composition.png")



def fig_capacity_ladder(results: Path) -> None:
    payload = json.loads((results / "E1b_capacity_ladder.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["per_cell"])
    comparisons = pd.DataFrame(payload["comparisons"])

    # E3 (Colab, fine-tuned end-to-end) sits between the frozen-feature models and the 1-NN
    # retriever on the capacity axis - a real deployable model, not a proxy for one.
    finetuned_path = results / "E3_finetuned_ladder.json"
    if finetuned_path.is_file():
        e3 = json.loads(finetuned_path.read_text(encoding="utf-8"))
        model_list = list(comparisons["model"])
        insert_at = model_list.index("knn_5") + 1 if "knn_5" in model_list else len(model_list)
        backbone = e3["config"]["backbone"]
        finetuned_row = pd.DataFrame([{
            "model": "finetuned\n" + backbone,
            "inflation": e3["comparison"]["inflation"],
        }])
        comparisons = pd.concat([comparisons.iloc[:insert_at], finetuned_row,
                                 comparisons.iloc[insert_at:]], ignore_index=True)

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))

    for rung, colour, marker in (("image_level", WARN, "o"),
                                 ("patient_disjoint", ACCENT, "s"),
                                 ("patient_disjoint_dedup", "#7FA8A4", "^")):
        block = frame[frame["rung"] == rung]
        if block.empty:
            continue
        x = np.arange(len(block))
        axes[0].errorbar(x, block["macro_f1"],
                         yerr=[block["macro_f1"] - block["ci_low"],
                               block["ci_high"] - block["macro_f1"]],
                         fmt=marker + "-", color=colour, capsize=3, linewidth=1.5,
                         markersize=5, label=rung)
        axes[0].set_xticks(x, block["model"], fontsize=8)
    axes[0].set_ylabel("macro-F1 (patient-level 95% CI)")
    axes[0].set_xlabel("model capacity to memorise  →")
    axes[0].set_title("Performance by capacity and protocol", fontsize=9.5)
    axes[0].legend(fontsize=7.5, frameon=False)

    if not comparisons.empty:
        x = np.arange(len(comparisons))
        colours = [ACCENT if str(m).startswith("finetuned")
                   else (WARN if v > 0.02 else "#9AA9A6")
                   for m, v in zip(comparisons["model"], comparisons["inflation"])]
        axes[1].bar(x, comparisons["inflation"], color=colours)
        axes[1].axhline(0, color=INK, linewidth=1)
        axes[1].set_xticks(x, comparisons["model"], fontsize=7.5)
        axes[1].set_ylabel("macro-F1 gained by ignoring patient identity")
        axes[1].set_xlabel("model capacity to memorise  →")
        axes[1].set_title("Inflation from patient leakage\n(teal = fine-tuned, the deployable model)", fontsize=9)
        for i, value in enumerate(comparisons["inflation"]):
            axes[1].text(i, value, f"{value:+.3f}", ha="center", fontsize=7.5,
                         va="bottom" if value >= 0 else "top")
    save(figure, "fig10_capacity_ladder.png")


def fig_cross_source(results: Path) -> None:
    payload = json.loads((results / "E5_cross_source_transfer.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["per_source"])
    figure, axis = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(frame))
    axis.bar(x - 0.2, frame["majority_class_accuracy"], 0.4, color="#C9D6D3",
             label="always guess the source majority class")
    axis.bar(x + 0.2, frame["accuracy"], 0.4, color=ACCENT, label="ODIR-trained model")
    tick_labels = [f"{source}\n({','.join(classes)})"
                   for source, classes in zip(frame["source"], frame["classes_present"])]
    axis.set_xticks(x, tick_labels, fontsize=7.5)
    axis.set_ylabel("accuracy"); axis.set_ylim(0, 1)
    axis.set_title("Transfer to unseen acquisition sources", fontsize=10)
    axis.legend(fontsize=7.5, frameon=False)
    save(figure, "fig11_cross_source_transfer.png")



def fig_xai(results: Path) -> None:
    frame = pd.read_csv(results / "E12_xai_per_image.csv")
    payload = json.loads((results / "E12_xai.json").read_text(encoding="utf-8"))

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))

    axes[0].hist(frame["fov_mass_ratio"], bins=16, color=ACCENT, alpha=0.85)
    baseline = payload["uniform_attribution_baseline"]
    axes[0].axvline(baseline, color=WARN, linestyle="--", linewidth=1.6,
                    label=f"uniform attribution ({baseline:.2f})")
    axes[0].axvline(payload["fov_mass_ratio_mean"], color=INK, linewidth=1.6,
                    label=f"observed mean ({payload['fov_mass_ratio_mean']:.2f})")
    axes[0].set_xlabel("fraction of attribution inside the retinal FOV")
    axes[0].set_ylabel("images")
    axes[0].set_title("Attribution concentrates on retina, not borders", fontsize=9.5)
    axes[0].legend(fontsize=7.5, frameon=False)

    by_class = frame.groupby("true_label")["fov_mass_ratio"].mean()
    x = np.arange(len(by_class))
    axes[1].bar(x, by_class.values, color=ACCENT)
    axes[1].axhline(baseline, color=WARN, linestyle="--", linewidth=1.4)
    axes[1].set_xticks(x, by_class.index)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("mean FOV-mass ratio")
    verdict = "PASS" if payload["randomisation_passes"] else "FAIL"
    axes[1].set_title(f"By class - sanity check: {verdict} "
                      f"(r={payload['randomisation_correlation']:+.2f})", fontsize=9.5)
    save(figure, "fig12_xai.png")



def fig_confounds(results: Path) -> None:
    payload = json.loads((results / "E11_confound_sensitivity.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["conditions"])

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    x = np.arange(len(frame))
    axes[0].errorbar(x, frame["macro_f1"],
                     yerr=[frame["macro_f1"] - frame["ci_low"],
                           frame["ci_high"] - frame["macro_f1"]],
                     fmt="o", color=ACCENT, capsize=4, markersize=6)
    axes[0].axhline(frame["macro_f1"].iloc[0], color=INK, linestyle=":", linewidth=1)
    axes[0].set_xticks(x, [c.replace("_", chr(10)) for c in frame["condition"]], fontsize=7.5)
    axes[0].set_ylabel("macro-F1 (patient-level 95% CI)")
    axes[0].set_title("Image performance under demographic controls", fontsize=9.5)

    axes[1].bar(x, frame["cataract_auroc"], color=ACCENT)
    axes[1].set_xticks(x, [c.replace("_", chr(10)) for c in frame["condition"]], fontsize=7.5)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_ylabel("cataract-vs-rest AUROC")
    demo = payload["demographics_only"]
    axes[1].set_title(f"Cataract AUROC holds\n(demographics alone: "
                      f"{demo['accuracy']:.3f} vs {demo['baseline_accuracy']:.3f} baseline)",
                      fontsize=9)
    for i, value in enumerate(frame["cataract_auroc"]):
        axes[1].text(i, value + 0.008, f"{value:.3f}", ha="center", fontsize=7.5)
    save(figure, "fig13_confound_sensitivity.png")


def fig_seed_stability(results: Path) -> None:
    frame = pd.read_csv(results / "E13_seed_stability.csv")
    figure, axis = plt.subplots(figsize=(5.8, 3.0))
    for model, colour in (("logistic", "#9AA9A6"), ("knn_1", ACCENT)):
        block = frame[frame["model"] == model]
        if block.empty:
            continue
        axis.plot(block["seed"].astype(str), block["inflation"], "o-", color=colour,
                  linewidth=1.6, markersize=7, label=model)
    axis.axhline(0, color=INK, linewidth=1)
    axis.set_xlabel("split seed"); axis.set_ylabel("macro-F1 gained by ignoring patient identity")
    axis.set_title("Leakage inflation is stable across seeds", fontsize=10)
    axis.legend(fontsize=8, frameon=False)
    save(figure, "fig14_seed_stability.png")


FIGURES = [
    ("fig01", fig_provenance_heatmap, "manifests"),
    ("fig02", fig_provenance_probes, "results"),
    ("fig03", fig_protocol_ladder, "results"),
    ("fig04", fig_duplicate_calibration, "manifests"),
    ("fig05", fig_extraction_by_degradation, "results"),
    ("fig06", fig_reliability, "results"),
    ("fig07", fig_risk_coverage, "results"),
    ("fig08", fig_robustness_matrix, "results"),
    ("fig09", fig_dataset_composition, "manifests"),
    ("fig10", fig_capacity_ladder, "results"),
    ("fig11", fig_cross_source, "results"),
    ("fig12", fig_xai, "results"),
    ("fig13", fig_confounds, "results"),
    ("fig14", fig_seed_stability, "results"),
]


def main() -> int:
    directories = {"manifests": output_dir("manifests"), "results": output_dir("results")}
    print("=" * 72)
    print("GENERATE FIGURES")
    print("=" * 72)
    made, skipped = 0, []
    for name, builder, source in FIGURES:
        try:
            builder(directories[source])
            made += 1
        except FileNotFoundError as exc:
            skipped.append(f"{name}: missing input ({Path(str(exc.filename or exc)).name})")
        except Exception as exc:  # noqa: BLE001 - one bad figure must not stop the rest
            skipped.append(f"{name}: {type(exc).__name__}: {exc}")
    print(f"\n{made} figure(s) written to {output_dir('figures')}")
    for line in skipped:
        print(f"  skipped {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
