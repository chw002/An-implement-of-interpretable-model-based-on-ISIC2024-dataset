"""Dissertation-grade plots.

Every figure exposed here is a deliberate piece of the dissertation — no
debug artefacts.  All figures use unique file names and captions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..pipelines import Pipeline, TransparentLRPipeline, TransparentTreePipeline
from ..utils import ensure_dir


# ------------------------------------------------------------- dataset figure
def plot_dataset_scale_distribution(
    cohort: pd.DataFrame,
    splits: Mapping[str, list],
    *,
    budgets: Iterable[Optional[int]],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    scales = []
    mals = []
    bens = []
    test_ids = set(splits["test"])
    val_ids = set(splits["val"])
    train_pool = cohort[cohort["image_id"].isin(splits["train"])]

    for b in budgets:
        if b is None:
            sub = train_pool
            label = "full"
        else:
            ben = train_pool[train_pool.label == 0].sample(
                n=min(int(b), int((train_pool.label == 0).sum())),
                random_state=1337,
            )
            mal = train_pool[train_pool.label == 1]
            sub = pd.concat([mal, ben], ignore_index=True)
            label = f"{b//1000}k" if b < 1_000_000 else f"{b}"
        scales.append(label)
        mals.append(int(sub.label.sum()))
        bens.append(int((sub.label == 0).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    idx = np.arange(len(scales))
    bar_w = 0.6
    ax.bar(idx, bens, bar_w, label="benign (train)", color="#6baed6")
    ax.bar(idx, mals, bar_w, bottom=bens, label="malignant (train)", color="#de2d26")
    ax.set_xticks(idx)
    ax.set_xticklabels(scales)
    ax.set_ylabel("images in training pool")
    ax.set_title("Training pool by benign budget (all malignant kept)")
    ax.legend()

    ax = axes[1]
    cats = ["train (full)", "val (fixed)", "test (fixed)"]
    counts_m = [
        int(cohort[cohort.image_id.isin(splits["train"])].label.sum()),
        int(cohort[cohort.image_id.isin(val_ids)].label.sum()),
        int(cohort[cohort.image_id.isin(test_ids)].label.sum()),
    ]
    counts_b = [
        int((cohort[cohort.image_id.isin(splits["train"])].label == 0).sum()),
        int((cohort[cohort.image_id.isin(val_ids)].label == 0).sum()),
        int((cohort[cohort.image_id.isin(test_ids)].label == 0).sum()),
    ]
    idx = np.arange(len(cats))
    ax.bar(idx, counts_b, bar_w, label="benign", color="#6baed6")
    ax.bar(idx, counts_m, bar_w, bottom=counts_b, label="malignant", color="#de2d26")
    ax.set_xticks(idx)
    ax.set_xticklabels(cats, rotation=0)
    ax.set_ylabel("images")
    ax.set_title("Patient-disjoint split composition")
    ax.legend()

    fig.suptitle("Figure 3.1 — Dataset scale + split distribution (ISIC 2024 image-only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------- metrics figures
def plot_metrics_vs_data_size(summary_df: pd.DataFrame, *, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    order = ["benign_40000", "benign_80000", "benign_160000", "benign_full"]
    summary_df = summary_df[summary_df["data_size"].isin(order)].copy()
    summary_df["data_size"] = pd.Categorical(summary_df["data_size"], categories=order, ordered=True)
    metrics = [("auc", "AUC"), ("balanced_accuracy", "Balanced accuracy"),
               ("recall", "Recall (sensitivity)"), ("specificity", "Specificity"),
               ("f1", "F1")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.8), sharex=True)
    for ax, (col, title) in zip(axes, metrics):
        for pipe, group in summary_df.groupby("pipeline"):
            group = group.sort_values("data_size")
            ax.plot(group["data_size"], group[col], marker="o", label=pipe, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("training benign budget")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("metric")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Figure 6.1 — Metrics vs. training data size (image-only pipelines)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_pipeline_comparison_full(summary_df: pd.DataFrame, *, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    full = summary_df[summary_df["data_size"] == "benign_full"].copy().sort_values("pipeline")
    if full.empty:
        # fall back to the largest available scale
        order = ["benign_full", "benign_160000", "benign_80000", "benign_40000"]
        for label in order:
            sub = summary_df[summary_df["data_size"] == label]
            if not sub.empty:
                full = sub.copy().sort_values("pipeline")
                break
    metrics = [("auc", "AUC"), ("balanced_accuracy", "Balanced Accuracy"),
               ("recall", "Recall"), ("specificity", "Specificity"),
               ("precision", "Precision"), ("f1", "F1")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(full["pipeline"], full[col], color="#4a90e2")
        ax.set_title(title)
        ax.set_ylim(0, max(1.0, full[col].max() * 1.05))
        ax.tick_params(axis="x", rotation=30)
        for bar, value in zip(bars, full[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 6.2 — Per-pipeline metrics at the largest training size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_confusion_matrices(
    predictions: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    n = len(predictions)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(pred.labels, pred.preds, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nτ = {thresholds.get(name, pred.threshold):.3f}")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["benign", "malignant"])
        ax.set_yticklabels(["benign", "malignant"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > vmax / 2 else "black",
                        fontsize=10)
    for ax in axes[len(predictions):]:
        ax.axis("off")
    fig.suptitle("Figure 6.3 — Confusion matrices by pipeline (test split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_transparent_explanation_example(
    per_pipeline: Mapping[str, Dict[str, Any]],
    artefacts: Mapping[str, Any],
    *,
    cohort: pd.DataFrame,
    out_path: str | Path,
) -> Optional[Path]:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    pack = per_pipeline.get("transparent_lr") or per_pipeline.get("transparent_tree")
    if pack is None:
        return None
    pipe: Pipeline = pack["pipe"]
    pred = pack["prediction"]
    df = pred.to_frame()
    candidates = df[(df.label == 1) & (df.pred == 1)]
    if candidates.empty:
        candidates = df[df.label == 1]
    if candidates.empty:
        candidates = df.iloc[:1]
    row = candidates.iloc[0].to_dict()
    image_id = row["image_id"]
    path = cohort.loc[cohort["image_id"] == image_id, "image_path"].iloc[0]

    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    explanation = pipe.explain(row, artefacts)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(img_small)
    axes[0].set_title(f"Raw image (id={image_id})")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("OpenCV Otsu + morphology mask")
    axes[1].axis("off")

    ax = axes[2]
    ax.axis("off")
    text_lines = explanation["text"].splitlines()[:16]
    ax.text(0.0, 1.0, "\n".join(text_lines), family="monospace", fontsize=8.5, va="top")
    ax.set_title(f"{pipe.name}: P(malignant)={explanation['prob']:.3f}  →  {explanation['verdict']}")

    fig.suptitle("Figure 7.1 — Fully-auditable decision trace (pixel → mask → formula/rule)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_feature_importance_or_shape_functions(
    per_pipeline: Mapping[str, Dict[str, Any]], *, out_path: str | Path
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    lr_pack = per_pipeline.get("transparent_lr")
    nam_pack = per_pipeline.get("glassbox_nam")
    tree_pack = per_pipeline.get("transparent_tree")

    # At most 3 panels: (i) LR coeffs, (ii) NAM shape curves, (iii) Tree feat importance.
    panels = []
    if lr_pack is not None:
        panels.append(("lr", lr_pack))
    if nam_pack is not None:
        panels.append(("nam", nam_pack))
    if tree_pack is not None:
        panels.append(("tree", tree_pack))
    if not panels:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No interpretable readout available.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.3))
    axes = np.atleast_1d(axes)

    for ax, (tag, pack) in zip(axes, panels):
        pipe = pack["pipe"]
        if tag == "lr":
            assert isinstance(pipe, TransparentLRPipeline)
            report = pipe.report()
            weights = sorted(report["weights"].items(), key=lambda kv: abs(kv[1]), reverse=True)
            names = [n for n, _ in weights]
            vals = [v for _, v in weights]
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
            ax.barh(range(len(names))[::-1], vals, color=colors)
            ax.set_yticks(range(len(names))[::-1])
            ax.set_yticklabels(names, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Transparent-LR coefficients")
            ax.set_xlabel("w (signed)")
        elif tag == "nam":
            vals = pipe.shape_values(100)
            x = vals["x"]
            for letter in "ABCD":
                ax.plot(x, vals[f"f_{letter}"], label=f"f_{letter}")
            ax.set_title("GlassBoxNAM shape functions")
            ax.set_xlabel("concept value (0..1)")
            ax.set_ylabel("contribution to logit")
            ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
            ax.legend(fontsize=8)
        elif tag == "tree":
            assert isinstance(pipe, TransparentTreePipeline)
            importances = pipe.model.feature_importances_
            names = pipe.feature_columns
            order = np.argsort(importances)[::-1]
            names = [names[i] for i in order]
            vals = [importances[i] for i in order]
            ax.barh(range(len(names))[::-1], vals, color="#2ca02c")
            ax.set_yticks(range(len(names))[::-1])
            ax.set_yticklabels(names, fontsize=8)
            ax.set_title("Transparent-Tree feature importance")
            ax.set_xlabel("Gini importance")

    fig.suptitle("Figure 7.2 — Interpretable readouts per pipeline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_transparent_lr_vs_tree_tradeoff(summary_df: pd.DataFrame, *, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    lr_df = summary_df[summary_df["pipeline"] == "transparent_lr"].copy()
    tree_df = summary_df[summary_df["pipeline"] == "transparent_tree"].copy()
    metrics = [("auc", "AUC"), ("balanced_accuracy", "Balanced accuracy"),
               ("recall", "Recall"), ("specificity", "Specificity"),
               ("f1", "F1")]
    order = ["benign_40000", "benign_80000", "benign_160000", "benign_full"]
    lr_df["data_size"] = pd.Categorical(lr_df["data_size"], categories=order, ordered=True)
    tree_df["data_size"] = pd.Categorical(tree_df["data_size"], categories=order, ordered=True)

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.8), sharex=True)
    for ax, (col, title) in zip(axes, metrics):
        if not lr_df.empty:
            ax.plot(lr_df["data_size"], lr_df[col], marker="o", label="LR", linewidth=1.8)
        if not tree_df.empty:
            ax.plot(tree_df["data_size"], tree_df[col], marker="s", label="Tree", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("training benign budget")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("metric")
    axes[-1].legend()
    fig.suptitle("Figure 7.3 — Transparent-LR vs. Transparent-Tree tradeoff")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_correct_prediction_features(
    summary_csv: str | Path, *, out_path: str | Path
) -> Optional[Path]:
    summary_csv = Path(summary_csv)
    out_path = Path(out_path)
    if not summary_csv.exists():
        return None
    df = pd.read_csv(summary_csv)
    if df.empty:
        return None
    top = df[df["rank"] <= 5]
    pipelines = top["pipeline"].unique().tolist()
    classes = ["TP_malignant", "TN_benign"]
    fig, axes = plt.subplots(len(pipelines), len(classes),
                             figsize=(5.5 * len(classes), 2.4 * max(1, len(pipelines))),
                             squeeze=False)
    for i, pipe in enumerate(pipelines):
        for j, cls in enumerate(classes):
            sub = top[(top["pipeline"] == pipe) & (top["class"] == cls)].sort_values("rank")
            ax = axes[i][j]
            if sub.empty:
                ax.axis("off")
                ax.set_title(f"{pipe} / {cls}: no samples", fontsize=9)
                continue
            ax.barh(sub["feature"][::-1], sub["median"][::-1], color="#7ba9c8" if cls.startswith("TN") else "#d2826f")
            ax.set_xlim(0, 1.0)
            ax.set_title(f"{pipe} · {cls}", fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Figure 7.4 — Top image-derived features for correct predictions (median value)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
