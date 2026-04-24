"""Transparent pipelines — strictly image-only, fully auditable.

Two sibling pipelines share the same pixel-derived feature table produced by
:mod:`glassderm.data.features`:

* :class:`TransparentLRPipeline`   — logistic regression (formula-transparent)
* :class:`TransparentTreePipeline` — shallow decision tree (rule-transparent)

Neither pipeline consumes a single ISIC/TBP metadata column — every input is
a function of the image pixels computed by OpenCV.  That is verified at fit
time via :func:`glassderm.data.audit.audit_feature_columns`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

from ..data.audit import audit_feature_columns
from ..data.features import FEATURE_NAMES
from ..utils import dump_json
from .base import Pipeline, PipelinePrediction


@dataclass
class _MinMax:
    mins: Dict[str, float]
    maxs: Dict[str, float]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, lo in self.mins.items():
            hi = self.maxs[c]
            span = max(hi - lo, 1e-8)
            out[c] = ((df[c].astype(float) - lo) / span).clip(0.0, 1.0)
        return out

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: Sequence[str]) -> "_MinMax":
        return cls(
            mins={c: float(df[c].min()) for c in columns},
            maxs={c: float(df[c].max()) for c in columns},
        )

    def to_dict(self) -> Dict:
        return {"mins": self.mins, "maxs": self.maxs}


class _TransparentBase(Pipeline):
    """Shared fit/predict/audit logic for the LR and Tree pipelines."""

    transparency_tag = "fully_auditable"

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.feature_columns: List[str] = []
        self.scaler: Optional[_MinMax] = None
        self.model = None

    # ------------------------------------------------------------ shared fit
    def _resolve_feature_columns(self, train_df: pd.DataFrame) -> List[str]:
        cols = [c for c in FEATURE_NAMES if c in train_df.columns]
        if not cols:
            raise ValueError(
                "Transparent pipeline found no pixel-derived feature columns. "
                "Re-run prepare-data to build data/features/transparent_features.*"
            )
        return cols

    def _audit(self, train_df: pd.DataFrame) -> None:
        audit_feature_columns(
            pipeline=self.name,
            feature_set="image_only",
            all_columns=list(train_df.columns),
            model_input_columns=self.feature_columns,
        )

    def _fit_scaler(self, train_df: pd.DataFrame) -> np.ndarray:
        self.scaler = _MinMax.fit(train_df, self.feature_columns)
        return self.scaler.transform(train_df)[self.feature_columns].to_numpy()

    def predict(self, split: str, artefacts: Mapping[str, Any]) -> PipelinePrediction:
        assert self.model is not None and self.scaler is not None
        df = artefacts["transparent_features"][split]
        X = self.scaler.transform(df)[self.feature_columns].to_numpy()
        probs = self.model.predict_proba(X)[:, 1]
        labels = df["label"].astype(int).to_numpy()
        preds = self._apply_threshold(probs)
        concept_cols = ["concept_A_asymmetry", "concept_B_border", "concept_C_color", "concept_D_diameter"]
        if all(c in df.columns for c in concept_cols):
            concepts = df[concept_cols].to_numpy(dtype=float)
        else:
            concepts = None
        return PipelinePrediction(
            image_ids=df["image_id"].tolist(),
            labels=labels,
            probs=probs,
            preds=preds,
            concepts=concepts,
            threshold=self.threshold,
            extras={"feature_columns": list(self.feature_columns)},
        )


# --------------------------------------------------------------- LR pipeline
class TransparentLRPipeline(_TransparentBase):
    name = "transparent_lr"
    transparency = "fully_auditable_logistic_on_pixel_features"

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self._c = float(cfg.pipelines.transparent_lr.logistic_c)

    def fit(self, artefacts: Mapping[str, Any]) -> None:
        train_df = artefacts["transparent_features"]["train"]
        self.feature_columns = self._resolve_feature_columns(train_df)
        self.feature_manifest = list(self.feature_columns)
        self._audit(train_df)
        X = self._fit_scaler(train_df)
        y = train_df["label"].astype(int).to_numpy()
        self.model = LogisticRegression(
            C=self._c,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=int(self.cfg.project.seed),
        )
        self.model.fit(X, y)
        self.logger.info(
            "Transparent-LR fitted on %d pixel-derived features, %d samples",
            X.shape[1], X.shape[0],
        )

    def explain(self, row: Mapping[str, Any], artefacts: Mapping[str, Any]) -> dict:
        assert self.model is not None and self.scaler is not None
        image_id = row["image_id"]
        feat_row = artefacts["transparent_features_by_id"][image_id]
        raw = {c: float(feat_row[c]) for c in self.feature_columns}
        scaled = {
            c: float(np.clip((v - self.scaler.mins[c]) /
                             max(self.scaler.maxs[c] - self.scaler.mins[c], 1e-8),
                             0.0, 1.0))
            for c, v in raw.items()
        }
        w = self.model.coef_.flatten()
        b = float(self.model.intercept_[0])
        contributions = {c: float(w[i] * scaled[c]) for i, c in enumerate(self.feature_columns)}
        logit = float(sum(contributions.values()) + b)
        prob = 1.0 / (1.0 + math.exp(-logit))
        verdict = "MALIGNANT" if prob >= self.threshold else "BENIGN"

        top = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        lines = [
            "Transparent-LR — logistic regression on pixel-derived features.",
            f"  Validation-tuned threshold τ = {self.threshold:.3f}",
            "  Top signed contributions (w · scaled_x) driving this decision:",
        ]
        for name, contrib in top:
            idx = self.feature_columns.index(name)
            lines.append(
                f"    {name:<26}  raw={raw[name]:+.4f}  scaled={scaled[name]:.3f}  "
                f"w={w[idx]:+.3f}  contrib={contrib:+.3f}"
            )
        lines.append(f"  + bias = {b:+.3f}")
        lines.append(
            f"  = logit {logit:+.3f}  →  σ(logit) = P(malignant) = {prob:.3f}  →  {verdict}"
        )
        return {
            "pipeline": self.name,
            "transparency": self.transparency,
            "image_id": image_id,
            "threshold": self.threshold,
            "prob": prob,
            "verdict": verdict,
            "logit": logit,
            "bias": b,
            "weights": {c: float(w[i]) for i, c in enumerate(self.feature_columns)},
            "feature_values_raw": raw,
            "feature_values_scaled": scaled,
            "contributions": contributions,
            "text": "\n".join(lines),
        }

    def report(self) -> Dict[str, Any]:
        assert self.model is not None and self.scaler is not None
        return {
            "classifier": "logistic",
            "feature_set": "image_only",
            "threshold": self.threshold,
            "feature_columns": list(self.feature_columns),
            "scaler": self.scaler.to_dict(),
            "weights": {c: float(self.model.coef_[0, i]) for i, c in enumerate(self.feature_columns)},
            "bias": float(self.model.intercept_[0]),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "scaler": self.scaler.to_dict() if self.scaler else None,
                "threshold": self.threshold,
                "model": self.model,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        state = joblib.load(path)
        self.feature_columns = list(state["feature_columns"])
        self.feature_manifest = list(self.feature_columns)
        self.scaler = _MinMax(**state["scaler"]) if state["scaler"] else None
        self.threshold = float(state["threshold"])
        self.model = state["model"]


# ------------------------------------------------------------- Tree pipeline
class TransparentTreePipeline(_TransparentBase):
    name = "transparent_tree"
    transparency = "fully_auditable_tree_on_pixel_features"

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        tree_cfg = cfg.pipelines.transparent_tree
        self._max_depth = int(tree_cfg.max_depth)
        self._min_samples_leaf = int(tree_cfg.min_samples_leaf)
        self._tree_rules: Optional[str] = None

    def fit(self, artefacts: Mapping[str, Any]) -> None:
        train_df = artefacts["transparent_features"]["train"]
        self.feature_columns = self._resolve_feature_columns(train_df)
        self.feature_manifest = list(self.feature_columns)
        self._audit(train_df)
        X = self._fit_scaler(train_df)
        y = train_df["label"].astype(int).to_numpy()
        self.model = DecisionTreeClassifier(
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            class_weight="balanced",
            random_state=int(self.cfg.project.seed),
        )
        self.model.fit(X, y)
        self._tree_rules = export_text(self.model, feature_names=list(self.feature_columns))
        self.logger.info(
            "Transparent-Tree fitted (max_depth=%d, min_samples_leaf=%d) on "
            "%d pixel-derived features",
            self._max_depth, self._min_samples_leaf, X.shape[1],
        )

    def explain(self, row: Mapping[str, Any], artefacts: Mapping[str, Any]) -> dict:
        assert self.model is not None and self.scaler is not None
        image_id = row["image_id"]
        feat_row = artefacts["transparent_features_by_id"][image_id]
        raw = {c: float(feat_row[c]) for c in self.feature_columns}
        scaled = {
            c: float(np.clip((v - self.scaler.mins[c]) /
                             max(self.scaler.maxs[c] - self.scaler.mins[c], 1e-8),
                             0.0, 1.0))
            for c, v in raw.items()
        }
        tree = self.model.tree_
        X = np.asarray([[scaled[c] for c in self.feature_columns]], dtype=np.float32)
        path = []
        node = 0
        while tree.children_left[node] != -1:
            feat_idx = tree.feature[node]
            threshold = float(tree.threshold[node])
            feat_name = self.feature_columns[feat_idx]
            x_val = float(X[0, feat_idx])
            if x_val <= threshold:
                direction, next_node = "<=", int(tree.children_left[node])
            else:
                direction, next_node = ">", int(tree.children_right[node])
            path.append(
                {
                    "node": int(node),
                    "feature": feat_name,
                    "scaled_value": x_val,
                    "raw_value": raw[feat_name],
                    "direction": direction,
                    "threshold_scaled": threshold,
                }
            )
            node = next_node

        leaf_counts = tree.value[node][0]
        total = float(leaf_counts.sum()) or 1.0
        prob = float(leaf_counts[1] / total) if tree.n_classes[0] > 1 else float(leaf_counts[0])
        verdict = "MALIGNANT" if prob >= self.threshold else "BENIGN"

        lines = [
            "Transparent-Tree — shallow decision tree on pixel-derived features.",
            f"  Validation-tuned threshold τ = {self.threshold:.3f}",
            f"  Tree max depth = {self._max_depth}   min samples / leaf = {self._min_samples_leaf}",
            "  Rule path taken:",
        ]
        for step in path:
            lines.append(
                f"    {step['feature']:<26} scaled={step['scaled_value']:.3f} "
                f"{step['direction']} {step['threshold_scaled']:.3f}"
            )
        lines.append(
            f"  → leaf class counts [benign,malignant] = {leaf_counts.tolist()}"
        )
        lines.append(
            f"  → leaf P(malignant) = {prob:.3f}  →  {verdict}"
        )
        return {
            "pipeline": self.name,
            "transparency": self.transparency,
            "image_id": image_id,
            "threshold": self.threshold,
            "prob": prob,
            "verdict": verdict,
            "path": path,
            "leaf_counts": leaf_counts.tolist(),
            "feature_values_raw": raw,
            "feature_values_scaled": scaled,
            "text": "\n".join(lines),
        }

    def report(self) -> Dict[str, Any]:
        assert self.model is not None and self.scaler is not None
        leaves = []
        tree = self.model.tree_
        for node in range(tree.node_count):
            if tree.children_left[node] == -1:
                counts = tree.value[node][0]
                total = float(counts.sum()) or 1.0
                leaves.append({
                    "node": int(node),
                    "class_counts": counts.tolist(),
                    "p_malignant": float(counts[1] / total),
                })
        return {
            "classifier": "tree",
            "feature_set": "image_only",
            "threshold": self.threshold,
            "feature_columns": list(self.feature_columns),
            "scaler": self.scaler.to_dict(),
            "max_depth": self._max_depth,
            "min_samples_leaf": self._min_samples_leaf,
            "tree_rules": self._tree_rules,
            "leaves": leaves,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "scaler": self.scaler.to_dict() if self.scaler else None,
                "threshold": self.threshold,
                "model": self.model,
                "tree_rules": self._tree_rules,
                "max_depth": self._max_depth,
                "min_samples_leaf": self._min_samples_leaf,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        state = joblib.load(path)
        self.feature_columns = list(state["feature_columns"])
        self.feature_manifest = list(self.feature_columns)
        self.scaler = _MinMax(**state["scaler"]) if state["scaler"] else None
        self.threshold = float(state["threshold"])
        self.model = state["model"]
        self._tree_rules = state.get("tree_rules")
        self._max_depth = int(state.get("max_depth", self._max_depth))
        self._min_samples_leaf = int(state.get("min_samples_leaf", self._min_samples_leaf))
