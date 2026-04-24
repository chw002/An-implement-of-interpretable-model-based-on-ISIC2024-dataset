"""Threshold selection strategies — chosen on validation, used everywhere.

We centralise this so (a) each pipeline is tuned with the same rule, and (b)
the single tuned τ is what drives both ``predict`` and ``explain``.  If those
two ever disagree, case-study reports would contradict themselves.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_curve


def select_threshold(
    probs: Sequence[float],
    labels: Sequence[int],
    strategy: str = "youden",
    fixed: float = 0.5,
) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    strategy = strategy.lower()

    if strategy == "fixed":
        return float(fixed)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float(fixed)

    if strategy == "youden":
        fpr, tpr, thr = roc_curve(labels, probs)
        j = tpr - fpr
        return float(np.clip(thr[int(np.argmax(j))], 1e-4, 1.0 - 1e-4))
    if strategy == "f1":
        grid = np.linspace(0.01, 0.99, 99)
        scores = [f1_score(labels, (probs >= t).astype(int), zero_division=0) for t in grid]
        return float(grid[int(np.argmax(scores))])
    raise ValueError(f"Unknown threshold strategy {strategy!r}")
