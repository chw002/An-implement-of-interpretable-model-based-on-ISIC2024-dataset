"""Common abstractions for all five pipelines.

Every pipeline advertises a ``feature_manifest`` — the list of columns that
will enter its model input — so the top-level driver can route it through
:func:`glassderm.data.audit.audit_feature_columns` before any weights are
trained.

Every ``explain`` verdict **must** match the ``predict`` verdict at the same
threshold — :func:`Pipeline._apply_threshold` is the single source of truth.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PipelinePrediction:
    image_ids: Sequence[str]
    labels: np.ndarray
    probs: np.ndarray
    preds: np.ndarray
    concepts: Optional[np.ndarray] = None      # (N, 4) pixel-derived ABCD proxy
    threshold: float = 0.5
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        cols: Dict[str, Any] = {
            "image_id": list(self.image_ids),
            "label": self.labels,
            "prob": self.probs,
            "pred": self.preds,
        }
        if self.concepts is not None:
            for i, letter in enumerate("ABCD"):
                cols[f"concept_{letter}"] = self.concepts[:, i]
        return pd.DataFrame(cols)


class Pipeline(ABC):
    name: str = "base"
    transparency: str = "unspecified"
    # A short tag used in docs & the dissertation:
    #   "fully_auditable"          — LR / Tree on pixel-derived features
    #   "interpretable_partial"    — CBM / NAM (perception opaque, readout transparent)
    #   "image_only_baseline"      — MultiTaskCNN (black-box head)
    transparency_tag: str = "unspecified"

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.threshold: float = float(cfg.evaluate.fixed_threshold)
        self.feature_manifest: List[str] = []       # populated during fit()

    @abstractmethod
    def fit(self, artefacts: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def predict(self, split: str, artefacts: Mapping[str, Any]) -> PipelinePrediction: ...

    @abstractmethod
    def explain(self, row: Mapping[str, Any], artefacts: Mapping[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> None: ...

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def _apply_threshold(self, probs: np.ndarray) -> np.ndarray:
        return (probs >= self.threshold).astype(int)

    def describe(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "transparency": self.transparency,
            "transparency_tag": self.transparency_tag,
        }
