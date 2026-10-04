"""Evaluation of the trained ConvNeXt-Tiny checkpoints.

Two results are produced, and they answer different questions.

POOLED OUT-OF-FOLD (headline)
    Each of the five fold models predicts its own held-out test fold; the predictions are
    pooled, so every one of the 5,056 images is scored exactly once by a model that never saw
    its patient. This is the protocol of the original project and of the model comparison.

DEPLOYED MODEL
    The single checkpoint the app serves (model.checkpoint, the fold-0 model) is run through the
    real inference path - file -> preprocessing -> full network - on its held-out test fold.

Multiclass metrics: precision / recall / specificity / F1 are macro averages over the four
classes; AUROC and AUPRC are one-vs-rest per class, macro-averaged. Confidence intervals
resample patients, not images.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from src.data.dataset import stem_cache
from src.data.splits import load_split
from src.evaluation.metrics import (accuracy_of, expected_calibration_error, macro_f1_of,
                                    mcnemar_exact, patient_bootstrap_ci)
from src.inference.predict import Predictor
from src.models.convnext_tiny import ConvNeXtScreener, load_checkpoint
from src.utils.config import load_config, repo_path


def full_metrics(frame: pd.DataFrame, labels: list[str], class_names: dict) -> dict:
    """Everything in the brief: accuracy, P, R/sensitivity, specificity, F1, AUROC, AUPRC, TP/TN/FP/FN."""
    y_true, y_pred = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    scores = frame[[f"p_{c}" for c in labels]].to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for i, code in enumerate(labels):
        tp = int(matrix[i, i])
        fn = int(matrix[i].sum() - tp)
        fp = int(matrix[:, i].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        is_class = (y_true == code).astype(int)
        per_class[code] = {
            "name": class_names[code], "support": tp + fn, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": round(precision, 4), "recall_sensitivity": round(recall, 4),
            "specificity": round(tn / (tn + fp) if tn + fp else 0.0, 4),
            "f1": round(2 * precision * recall / (precision + recall) if precision + recall else 0.0, 4),
            "auroc": round(float(roc_auc_score(is_class, scores[:, i])), 4),
            "auprc": round(float(average_precision_score(is_class, scores[:, i])), 4),
            "prevalence": round(float(is_class.mean()), 4)}
    macro = lambda key: round(float(np.mean([v[key] for v in per_class.values()])), 4)  # noqa: E731
    return {
        "n_images": int(len(frame)), "n_patients": int(frame.patient_id.nunique()),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_precision": macro("precision"), "macro_recall_sensitivity": macro("recall_sensitivity"),
        "macro_specificity": macro("specificity"), "macro_f1": round(macro_f1_of(frame), 4),
        "macro_auroc_ovr": macro("auroc"), "macro_auprc_ovr": macro("auprc"),
        "macro_f1_ci95": patient_bootstrap_ci(frame, macro_f1_of),
        "accuracy_ci95": patient_bootstrap_ci(frame, accuracy_of),
        "per_class": per_class, "confusion_matrix": matrix.tolist(), "labels": labels,
        "averaging": "macro over 4 classes; AUROC/AUPRC one-vs-rest; CIs by patient-level bootstrap (2000)"}


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(matrix, labels, class_names, title, path):
    plt = _plt()
    matrix = np.asarray(matrix)
    rates = matrix / matrix.sum(axis=1, keepdims=True)
    names = [class_names[c].replace("_", " ") for c in labels]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]}\n{rates[i, j]:.1%}", ha="center", va="center", fontsize=11,
                    color="white" if rates[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), names, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of the true class (row-normalised)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curves(frame, labels, class_names, metrics, roc_path, pr_path):
    plt = _plt()
    for kind, path in (("roc", roc_path), ("pr", pr_path)):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        for i, code in enumerate(labels):
            is_class = (frame.y_true == code).astype(int)
            score = frame[f"p_{code}"]
            m = metrics["per_class"][code]
            name = class_names[code].replace("_", " ")
            if kind == "roc":
                x, y, _ = roc_curve(is_class, score)
                ax.plot(x, y, color=f"C{i}", label=f"{name} (AUROC {m['auroc']:.3f})")
            else:
                y, x, _ = precision_recall_curve(is_class, score)
                ax.plot(x, y, color=f"C{i}", label=f"{name} (AUPRC {m['auprc']:.3f}, prevalence {m['prevalence']:.3f})")
                ax.axhline(m["prevalence"], color=f"C{i}", ls=":", lw=.8)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=.8)
            ax.set_xlabel("False positive rate (1 - specificity)")
            ax.set_ylabel("True positive rate (sensitivity)")
            ax.set_title(f"ROC, one-vs-rest per class - macro AUROC {metrics['macro_auroc_ovr']:.3f}", fontsize=11)
        else:
            ax.set_xlabel("Recall (sensitivity)")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-recall, one-vs-rest - macro AUPRC {metrics['macro_auprc_ovr']:.3f}\n"
                         "(dotted lines: class prevalence, the no-skill level)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right" if kind == "roc" else "best")
        ax.grid(alpha=.3)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)


# ------------------------------------------------------------------ predictions
@torch.no_grad()
def out_of_fold_predictions(config, frame) -> pd.DataFrame:
    """Each saved fold checkpoint scores its own held-out fold (via the cached frozen stem)."""
    labels = config["data"]["labels"]
    features, index = stem_cache(ConvNeXtScreener(len(labels), pretrained=True), frame)
    blocks = []
    for k in config["training"]["folds"]:
        path = repo_path(config["model"]["checkpoint_dir"], f"fold_{k}.pt")
        if not path.is_file():
            print(f"  fold {k}: no checkpoint - skipped")
            continue
        model, meta = load_checkpoint(path)
        assert meta["test_fold"] == k
        test = frame[frame.fold == k].reset_index(drop=True)
        positions = np.array([index[i] for i in test.image_id])
        order = np.argsort(positions)
        logits = np.empty((len(test), len(labels)), dtype=np.float32)
        for start in range(0, len(test), 64):
            rows = order[start:start + 64]
            logits[rows] = model.forward_from_stem(torch.from_numpy(np.array(features[positions[rows]]))).numpy()
        probs = torch.softmax(torch.tensor(logits), 1).numpy()
        block = test[["image_id", "patient_id", "eye", "label"]].rename(columns={"label": "y_true"})
        block["y_pred"] = [labels[i] for i in probs.argmax(1)]
        for i, code in enumerate(labels):
            block[f"p_{code}"] = probs[:, i]
        block["test_fold"] = k
        blocks.append(block)
        print(f"  fold {k}: {len(block)} test images scored", flush=True)
    return pd.concat(blocks, ignore_index=True)


def deployed_predictions(config, frame) -> tuple[pd.DataFrame, dict]:
    """The served checkpoint through the real file -> preprocessing -> network path."""
    predictor = Predictor()
    labels, k = predictor.labels, predictor.meta["test_fold"]
    test = frame[frame.fold == k].reset_index(drop=True)
    rows, started = [], time.time()
    for n, row in enumerate(test.itertuples()):
        out = predictor.predict(row.image_path)
        rows.append({"image_id": row.image_id, "patient_id": row.patient_id, "eye": row.eye, "y_true": row.label,
                     "y_pred": out["predicted_class"], "confidence": out["confidence"],
                     **{f"p_{c}": out["uncalibrated_probabilities"][c] for c in labels},
                     **{f"cal_{c}": out["probabilities"][c] for c in labels}, "test_fold": k})
        if n % 200 == 0:
            print(f"  deployed model: {n}/{len(test)} images", flush=True)
    seconds = (time.time() - started) / len(test)
    return pd.DataFrame(rows), {"seconds_per_image_end_to_end_cpu": round(seconds, 4), "test_fold": k,
                                "temperature": predictor.temperature}


# ------------------------------------------------------------------ main
def main(skip_deployed: bool = False) -> int:
    config = load_config()
    labels, names = config["data"]["labels"], config["data"]["class_names"]
    results = repo_path(config["outputs"]["results"])
    metrics_dir = results / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame = load_split()
    display = config["model"]["display_name"]

    print("pooled out-of-fold evaluation ...", flush=True)
    pooled = out_of_fold_predictions(config, frame)
    folds_done = sorted(pooled.test_fold.unique().tolist())
    metrics = full_metrics(pooled, labels, names)
    metrics["folds"] = folds_done
    metrics["per_fold_macro_f1"] = {int(k): round(macro_f1_of(g), 4) for k, g in pooled.groupby("test_fold")}
    pooled.to_csv(metrics_dir / "predictions.csv", index=False)

    title = f"{display}, pooled out-of-fold test predictions\n({len(pooled)} images, folds {folds_done})"
    plot_confusion(metrics["confusion_matrix"], labels, names, title, results / "confusion_matrices" / "confusion_matrix.png")
    plot_curves(pooled, labels, names, metrics, results / "roc" / "roc_curve.png", results / "pr" / "precision_recall_curve.png")
    (metrics_dir / "classification_report.txt").write_text(
        title.replace("\n", " ") + "\n\n" + classification_report(pooled.y_true, pooled.y_pred, labels=labels,
                                               target_names=[names[c] for c in labels], digits=4, zero_division=0),
        encoding="utf-8")

    # paired comparison with the original project's ResNet50 on the same images, if its predictions are present
    reference = results / "comparison" / "resnet50_original_predictions.csv"
    if reference.is_file() and len(folds_done) == len(config["training"]["folds"]):
        old = pd.read_csv(reference).set_index("image_id")
        new = pooled.set_index("image_id")
        common = new.index.intersection(old.index)
        test = mcnemar_exact((new.loc[common].y_true == new.loc[common].y_pred).to_numpy(),
                             (old.loc[common].y_true == old.loc[common].y_pred).to_numpy())
        metrics["mcnemar_vs_original_resnet50"] = {
            **test, "n_paired_images": int(len(common)), "a": display, "b": "original project ResNet50 (partial fine-tuning)",
            "resnet50_macro_f1_same_images": round(macro_f1_of(old.loc[common].reset_index()), 4),
            "convnext_macro_f1_same_images": round(macro_f1_of(new.loc[common].reset_index()), 4)}

    # model size and speed
    model = ConvNeXtScreener(len(labels), pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t0 = time.time()
        for _ in range(20):
            model(x)
    metrics["parameters_total"] = sum(p.numel() for p in model.parameters())
    metrics["parameters_trained"] = sum(p.numel() for p in model.trainable_parameters())
    metrics["inference_ms_per_image_cpu_batch1"] = round((time.time() - t0) / 20 * 1000, 1)

    if not skip_deployed and repo_path(config["model"]["checkpoint"]).is_file():
        print("deployed checkpoint through the real inference path ...", flush=True)
        served, info = deployed_predictions(config, frame)
        served.to_csv(metrics_dir / "predictions_deployed_model.csv", index=False)
        deployed = full_metrics(served, labels, names)
        truth = served.y_true.to_numpy()
        deployed["ece_uncalibrated"] = expected_calibration_error(served[[f"p_{c}" for c in labels]].to_numpy(), truth, labels)
        deployed["ece_temperature_scaled"] = expected_calibration_error(served[[f"cal_{c}" for c in labels]].to_numpy(), truth, labels)
        deployed.update(info)
        # the two paths (cached stem vs file -> full network) must agree on the same checkpoint
        same = pooled[pooled.test_fold == info["test_fold"]].set_index("image_id").loc[served.image_id]
        deployed["max_abs_probability_difference_vs_cached_path"] = round(float(np.abs(
            same[[f"p_{c}" for c in labels]].to_numpy() - served[[f"p_{c}" for c in labels]].to_numpy()).max()), 6)
        deployed["prediction_agreement_vs_cached_path"] = round(float((same.y_pred.to_numpy() == served.y_pred.to_numpy()).mean()), 4)
        metrics["deployed_model"] = deployed
        plot_confusion(deployed["confusion_matrix"], labels, names,
                       f"{display}, deployed checkpoint\non its held-out fold {info['test_fold']} ({len(served)} images)",
                       results / "confusion_matrices" / "confusion_matrix_deployed_model.png")

    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    table = pd.DataFrame([{"class": f"{c} ({v['name']})", **{k: v[k] for k in
                           ("support", "TP", "TN", "FP", "FN", "precision", "recall_sensitivity", "specificity", "f1", "auroc", "auprc")}}
                          for c, v in metrics["per_class"].items()])
    table.loc[len(table)] = {"class": "macro average", "support": metrics["n_images"],
                             "precision": metrics["macro_precision"], "recall_sensitivity": metrics["macro_recall_sensitivity"],
                             "specificity": metrics["macro_specificity"], "f1": metrics["macro_f1"],
                             "auroc": metrics["macro_auroc_ovr"], "auprc": metrics["macro_auprc_ovr"]}
    table.to_csv(metrics_dir / "metrics.csv", index=False)

    ci = metrics["macro_f1_ci95"]
    final = pd.DataFrame([
        ("Accuracy", metrics["accuracy"]), ("Precision (macro)", metrics["macro_precision"]),
        ("Recall (macro)", metrics["macro_recall_sensitivity"]), ("Sensitivity (macro)", metrics["macro_recall_sensitivity"]),
        ("Specificity (macro)", metrics["macro_specificity"]),
        ("F1-score (macro)", f"{metrics['macro_f1']} [95% CI {ci['ci_low']}, {ci['ci_high']}]"),
        ("AUROC (macro, one-vs-rest)", metrics["macro_auroc_ovr"]), ("AUPRC (macro, one-vs-rest)", metrics["macro_auprc_ovr"]),
        ("Parameters", f"{metrics['parameters_total'] / 1e6:.2f} M ({metrics['parameters_trained'] / 1e6:.2f} M trained)"),
        ("Inference Time", f"{metrics['inference_ms_per_image_cpu_batch1']} ms per image (CPU, batch 1, network only)"),
        ("Test images / patients", f"{metrics['n_images']} / {metrics['n_patients']} (pooled out-of-fold, folds {folds_done})"),
    ], columns=["Metric", display])
    final.to_csv(results / "final_results.csv", index=False)
    final.to_excel(results / "final_results.xlsx", index=False)
    (results / "final_results.md").write_text(
        f"# Final results - {display}\n\nMeasured by `python scripts/evaluate.py`. Pooled out-of-fold test predictions on the "
        "patient-disjoint split; every image is scored once by a model that never saw its patient.\n\n"
        f"| Metric | {display} |\n|---|---:|\n" + "\n".join(f"| {m} | {v} |" for m, v in final.itertuples(index=False)) + "\n",
        encoding="utf-8")
    print(f"\n{display}: accuracy {metrics['accuracy']}, macro-F1 {metrics['macro_f1']} "
          f"[{ci['ci_low']}, {ci['ci_high']}], AUROC {metrics['macro_auroc_ovr']}, AUPRC {metrics['macro_auprc_ovr']}")
    return 0
