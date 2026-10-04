"""Partial fine-tuning of ImageNet ConvNeXt-Tiny over the patient-disjoint folds.

The recipe is the one validated in the model-comparison experiment (itself read from the
original project's ResNet50 script), unchanged:

    per test fold k:  validation = fold (k + 1) % 5, train = the other three folds
    phase 1   2 epochs, head only (LayerNorm + Linear), AdamW lr 1e-3
    phase 2   up to 8 epochs, stage 4 + head, AdamW lr 1e-4, cosine annealing,
              early stopping (patience 3) on VALIDATION macro-F1
    loss      cross-entropy with balanced class weights from the training folds
    test      the test fold is scored once, after training and model selection

The model served by the app is the fold fixed in config (model.deployed_fold), not the fold
that happened to score best.
"""

from __future__ import annotations

import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.data.dataset import Batches, stem_cache
from src.data.splits import fold_sets, leakage_report, load_split
from src.models.convnext_tiny import ConvNeXtScreener, save_checkpoint
from src.training.losses import balanced_class_weights, fit_temperature, weighted_cross_entropy
from src.utils.config import load_config, repo_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sanity_check(n_classes: int) -> dict:
    """Shape, finiteness, gradient flow and an optimiser step - before any long run."""
    torch.manual_seed(0)
    net = ConvNeXtScreener(n_classes, pretrained=True)
    net.train()
    x, y = torch.randn(2, 3, 224, 224), torch.tensor([0, n_classes - 1])
    out = net(x)
    assert tuple(out.shape) == (2, n_classes), f"output shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "non-finite outputs"
    loss = torch.nn.functional.cross_entropy(out, y)
    assert torch.isfinite(loss), "non-finite loss"
    optimiser = torch.optim.AdamW(net.trainable_parameters(), lr=1e-4)
    before = net.fc.weight.detach().clone()
    loss.backward()
    with_grad = [p for p in net.trainable_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(with_grad) == len(net.trainable_parameters()), "some trainable parameters got no gradient"
    assert all(p.grad is None for p in net.stem.parameters()), "the frozen stem received gradients"
    optimiser.step()
    assert not torch.equal(before, net.fc.weight), "optimiser step did not change the head"
    return {"input_shape": [2, 3, 224, 224], "stem_output_shape": list(net.stem_features(x).shape),
            "output_shape": list(out.shape), "loss": round(loss.item(), 4),
            "trainable_parameters": sum(p.numel() for p in net.trainable_parameters()),
            "frozen_parameters": sum(p.numel() for p in net.stem.parameters()), "status": "PASS"}


def run_epoch(net, batches, criterion, optimiser=None, train_tail=False):
    training = optimiser is not None
    net.tail.train(training and train_tail)
    net.norm.train(training)
    net.fc.train(training)
    total, n, ids, logits, trues = 0.0, 0, [], [], []
    with torch.set_grad_enabled(training):
        for x, y, batch_ids in batches:
            out = net.forward_from_stem(x)
            loss = criterion(out, y)
            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            total += loss.item() * len(y)
            n += len(y)
            ids += batch_ids
            logits.append(out.detach())
            trues.append(y)
    logits, y_true = torch.cat(logits).numpy(), torch.cat(trues).numpy()
    y_pred = logits.argmax(1)
    metrics = {"loss": round(total / max(n, 1), 4),
               "accuracy": round(float((y_true == y_pred).mean()), 4),
               "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)}
    return metrics, ids, logits, y_true


def train_fold(k: int, frame, features, index, config, smoke: bool, checkpoint_dir: Path):
    t, labels = config["training"], config["data"]["labels"]
    seed_everything(t["seed"] + k)
    train, val, test, val_fold = fold_sets(frame, k, len(t["folds"]))
    if smoke:
        train, val, test = (s.groupby("label").head(n) for s, n in ((train, 12), (val, 6), (test, 6)))

    net = ConvNeXtScreener(len(labels), pretrained=True)
    weights = balanced_class_weights(train.label, labels)
    criterion = weighted_cross_entropy(weights)
    batches = lambda f, tr: Batches(f, features, index, labels, t["batch_size"], train=tr,  # noqa: E731
                                    drop_last=t["drop_last_train"])
    history, started = [], time.time()

    def record(phase, epoch, tr, va, lr, t0):
        history.append({"fold": k, "phase": phase, "epoch": epoch, "learning_rate": lr,
                        "train_loss": tr["loss"], "train_accuracy": tr["accuracy"], "train_macro_f1": tr["macro_f1"],
                        "val_loss": va["loss"], "val_accuracy": va["accuracy"], "val_macro_f1": va["macro_f1"],
                        "seconds": round(time.time() - t0, 1)})
        print(f"  [fold {k}] {phase} epoch {epoch}: train loss {tr['loss']:.4f} acc {tr['accuracy']:.3f} | "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.3f} macro-F1 {va['macro_f1']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # phase 1: head only
    for p in net.tail.parameters():
        p.requires_grad_(False)
    optimiser = torch.optim.AdamW(net.head_parameters(), lr=t["phase1"]["lr"], weight_decay=t["weight_decay"])
    for e in range(1 if smoke else t["phase1"]["epochs"]):
        t0 = time.time()
        tr, *_ = run_epoch(net, batches(train, True), criterion, optimiser, train_tail=False)
        va, *_ = run_epoch(net, batches(val, False), criterion)
        record("1_head_only", e + 1, tr, va, t["phase1"]["lr"], t0)

    # phase 2: last stage + head; model selection on VALIDATION only
    for p in net.tail.parameters():
        p.requires_grad_(True)
    epochs = 1 if smoke else t["phase2"]["epochs"]
    optimiser = torch.optim.AdamW(net.trainable_parameters(), lr=t["phase2"]["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    best, best_epoch, bad, best_state = -1.0, None, 0, None
    for e in range(epochs):
        t0, lr = time.time(), optimiser.param_groups[0]["lr"]
        tr, *_ = run_epoch(net, batches(train, True), criterion, optimiser, train_tail=True)
        va, *_ = run_epoch(net, batches(val, False), criterion)
        scheduler.step()
        record("2_last_stage_and_head", e + 1, tr, va, round(lr, 8), t0)
        if va["macro_f1"] > best:
            best, best_epoch, bad, best_state = va["macro_f1"], e + 1, 0, net.trained_state()
        else:
            bad += 1
            if bad >= t["early_stopping_patience"]:
                print(f"  [fold {k}] early stop after epoch {e + 1}", flush=True)
                break
    last_state, last_epoch = net.trained_state(), history[-1]["epoch"]
    net.load_state_dict(best_state, strict=False)
    train_seconds = time.time() - started

    # temperature on the validation fold, then the ONLY use of the test fold
    _, _, val_logits, val_true = run_epoch(net, batches(val, False), criterion)
    temperature = fit_temperature(val_logits, val_true)
    test_metrics, test_ids, test_logits, _ = run_epoch(net, batches(test, False), criterion)
    probs = torch.softmax(torch.tensor(test_logits), 1).numpy()
    block = test.set_index("image_id").loc[test_ids, ["patient_id", "eye", "label"]].reset_index()
    block = block.rename(columns={"label": "y_true"})
    block["y_pred"] = [labels[i] for i in probs.argmax(1)]
    for i, code in enumerate(labels):
        block[f"p_{code}"] = probs[:, i]
    block["test_fold"] = k

    meta = {"architecture": config["model"]["name"], "weights": config["model"]["weights"],
            "labels": labels, "class_names": config["data"]["class_names"],
            "input_size": config["preprocessing"]["input_size"],
            "mean": config["preprocessing"]["mean"], "std": config["preprocessing"]["std"],
            "test_fold": k, "validation_fold": val_fold, "best_epoch_phase2": best_epoch,
            "validation_macro_f1": best, "temperature": temperature,
            "temperature_fitted_on": f"validation fold {val_fold}", "smoke_test": smoke}
    save_checkpoint(checkpoint_dir / f"fold_{k}.pt", best_state, meta)
    save_checkpoint(checkpoint_dir / f"fold_{k}_last.pt", last_state,
                    {**meta, "note": f"last epoch ({last_epoch}) of phase 2, not selected on validation"})

    log = {"test_fold": k, "validation_fold": val_fold,
           "n_images": {"train": len(train), "validation": len(val), "test": len(test)},
           "n_patients": {"train": train.patient_id.nunique(), "validation": val.patient_id.nunique(),
                          "test": test.patient_id.nunique()},
           "class_weights": {c: round(float(w), 4) for c, w in zip(labels, weights)},
           "best_epoch_phase2": best_epoch, "best_validation_macro_f1": best,
           "last_epoch_phase2": last_epoch, "temperature": temperature,
           "test_metrics_at_training_time": test_metrics, "train_seconds": round(train_seconds, 1)}
    return block, log, history


def plot_history(history: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = sorted(history.fold.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for k in folds:
        h = history[history.fold == k].reset_index(drop=True)
        x = np.arange(1, len(h) + 1)
        axes[0].plot(x, h.train_loss, "--", color=f"C{k}", alpha=.7)
        axes[0].plot(x, h.val_loss, "-", color=f"C{k}", label=f"fold {k}")
        axes[1].plot(x, h.train_accuracy, "--", color=f"C{k}", alpha=.7)
        axes[1].plot(x, h.val_accuracy, "-", color=f"C{k}", label=f"fold {k}")
        axes[2].plot(x, h.val_macro_f1, "-o", ms=3, color=f"C{k}", label=f"fold {k}")
    for ax, title in zip(axes, ("Loss (dashed = train, solid = validation)",
                                "Accuracy (dashed = train, solid = validation)", "Validation macro-F1")):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch (2 head-only epochs, then fine-tuning)")
        ax.axvline(2.5, color="grey", lw=.8, ls=":")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle("ConvNeXt-Tiny partial fine-tuning on patient-disjoint folds", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(folds: list[int] | None = None, smoke: bool = False, skip_completed: bool = False) -> int:
    config = load_config()
    t = config["training"]
    folds = [0] if smoke else (folds if folds else t["folds"])
    tag = "smoke" if smoke else "all"
    out = repo_path(config["outputs"]["results"], "_smoke" if smoke else "training")
    checkpoint_dir = repo_path(config["model"]["checkpoint_dir"]) / ("_smoke" if smoke else "")
    out.mkdir(parents=True, exist_ok=True)

    print("sanity check ...", flush=True)
    sanity = sanity_check(config["model"]["n_classes"])
    print(f"  {sanity}", flush=True)

    frame = load_split()
    leaks = leakage_report(frame)
    print(f"split: {leaks}", flush=True)
    assert leaks["patients_in_more_than_one_fold"] == 0 and leaks["duplicate_image_ids"] == 0, "leakage in split"
    if smoke:
        frame = frame[frame.image_id.isin(frame.groupby(["fold", "label"]).head(12).image_id)]

    print("caches (built once) ...", flush=True)
    features, index = stem_cache(ConvNeXtScreener(config["model"]["n_classes"], pretrained=True), frame, tag)

    for k in folds:
        log_path, pred_path, hist_path = (out / f"fold_{k}_log.json", out / f"predictions_fold_{k}.csv",
                                          out / f"history_fold_{k}.csv")
        if skip_completed and log_path.is_file() and pred_path.is_file() and hist_path.is_file() \
                and (checkpoint_dir / f"fold_{k}.pt").is_file():
            print(f"[fold {k}] already done - skipping", flush=True)
            continue
        block, log, history = train_fold(k, frame, features, index, config, smoke, checkpoint_dir)
        block.to_csv(pred_path, index=False)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"[fold {k}] best epoch {log['best_epoch_phase2']}, val macro-F1 {log['best_validation_macro_f1']}, "
              f"test macro-F1 {log['test_metrics_at_training_time']['macro_f1']}, T {log['temperature']}, "
              f"{log['train_seconds'] / 60:.1f} min", flush=True)

    done = sorted(int(p.stem.split("_")[1]) for p in out.glob("fold_*_log.json"))
    history = pd.concat([pd.read_csv(out / f"history_fold_{k}.csv") for k in done], ignore_index=True)
    history.to_csv(out / "training_history.csv", index=False)
    plot_history(history, out / "training_history.png")
    logs = [json.loads((out / f"fold_{k}_log.json").read_text()) for k in done]
    summary = {"model": config["model"]["display_name"], "weights": config["model"]["weights"],
               "frozen": config["model"]["frozen"], "trainable": config["model"]["trainable"],
               "training": t, "preprocessing": config["preprocessing"], "folds_completed": done,
               "leakage_checks": leaks, "sanity_check": sanity,
               "best_epochs": {log["test_fold"]: log["best_epoch_phase2"] for log in logs},
               "class_weights_fold0": logs[0]["class_weights"],
               "train_minutes_total": round(sum(log["train_seconds"] for log in logs) / 60, 1),
               "hardware": {"device": "cpu", "processor": platform.processor(), "torch": torch.__version__,
                            "threads": torch.get_num_threads(), "platform": platform.platform()},
               "smoke_test": smoke}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    deployed = config["model"]["deployed_fold"]
    if not smoke and deployed in done:
        shutil.copyfile(checkpoint_dir / f"fold_{deployed}.pt", repo_path(config["model"]["checkpoint"]))
        shutil.copyfile(checkpoint_dir / f"fold_{deployed}_last.pt", repo_path(config["model"]["last_checkpoint"]))
        print(f"deployed model: fold {deployed} -> {config['model']['checkpoint']}", flush=True)
    print(f"done: folds {done}, {summary['train_minutes_total']} min of training", flush=True)
    return 0
