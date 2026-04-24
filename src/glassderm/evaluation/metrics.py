"""Binary-classification metrics computed with one consistent threshold.

We deliberately centralise all metric calls here so that every pipeline is
scored with exactly the same code — no per-pipeline metric drift.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
    }


def _balanced_accuracy(y_true, y_pred) -> float:
    from sklearn.metrics import balanced_accuracy_score

    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except ValueError:
        return float("nan")


def metrics_to_row(name: str, transparency: str, m: Dict[str, float]) -> Dict:
    return {
        "pipeline": name,
        "transparency": transparency,
        "threshold": m["threshold"],
        "accuracy": m["accuracy"],
        "auc": m["auc"],
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "specificity": m["specificity"],
        "balanced_accuracy": m["balanced_accuracy"],
        "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
    }


def roc_points(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return fpr, tpr, thr
