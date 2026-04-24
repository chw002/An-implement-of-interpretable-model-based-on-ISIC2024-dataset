"""Summarise the pixel-derived features of *correctly* predicted test images.

For each pipeline we ask: "when this pipeline got the answer right, which
image-derived features were systematically high?"  The output feeds
:func:`glassderm.analysis.plots.plot_correct_prediction_features`.

Two files are emitted side by side:

* ``correct_prediction_feature_summary.csv`` — machine-readable, full ranking
* ``correct_prediction_feature_summary.md``  — human-readable top-10 per bucket

Schema of ``correct_prediction_feature_summary.csv``
-----------------------------------------------------
- ``pipeline``          pipeline name
- ``class``             TP_malignant | TN_benign | FP_benign | FN_malignant
- ``n_images``          number of test images in this bucket for this pipeline
- ``feature``           image-derived feature name (``FEATURE_NAMES`` ∪ ``CONCEPT_NAMES``)
- ``median``            median feature value (all features already ∈ [0, 1])
- ``mean``              mean feature value
- ``rank``              1 = largest median within (pipeline, class)

Because each feature value already lies in [0, 1] (see
:mod:`glassderm.data.features`), medians are directly comparable across
features.  Using the median makes the ranking robust to the few outlier
lesions in each bucket.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data.features import CONCEPT_NAMES, FEATURE_NAMES
from ..utils import ensure_dir, get_logger

logger = get_logger("glassderm.correct_features")

_CLASS_BUCKETS = (
    ("TP_malignant", 1, 1),
    ("TN_benign",    0, 0),
    ("FP_benign",    0, 1),
    ("FN_malignant", 1, 0),
)

_BUCKET_DESCRIPTIONS = {
    "TP_malignant":  "True positives — malignant lesions the pipeline correctly flagged.",
    "TN_benign":     "True negatives — benign lesions the pipeline correctly passed.",
    "FP_benign":     "False positives — benign lesions mistakenly flagged as malignant.",
    "FN_malignant":  "False negatives — malignant lesions the pipeline missed.",
}

_TOP_K_IN_MD = 10


def write_correct_prediction_summary(
    per_pipeline: Mapping[str, Mapping[str, Any]],
    artefacts: Mapping[str, Any],
    *,
    out_dir: str | Path,
) -> Path:
    """Emit ``<out_dir>/correct_prediction_feature_summary.csv``.

    Parameters
    ----------
    per_pipeline
        ``{name: {"pipe": Pipeline, "prediction": PipelinePrediction, ...}}``.
    artefacts
        Must expose ``transparent_features_by_id`` — i.e. the pixel-derived
        feature lookup built in :func:`glassderm.artefacts.build_artefacts`.
    out_dir
        Per-scale output directory.
    """
    out_dir = ensure_dir(out_dir)
    feats_by_id = artefacts.get("transparent_features_by_id", {}) or {}
    if not feats_by_id:
        logger.warning(
            "transparent_features_by_id missing → correct_prediction_feature_summary.{csv,md} empty"
        )
        empty_csv = out_dir / "correct_prediction_feature_summary.csv"
        pd.DataFrame(
            columns=["pipeline", "class", "n_images", "feature", "median", "mean", "rank"]
        ).to_csv(empty_csv, index=False)
        (out_dir / "correct_prediction_feature_summary.md").write_text(
            "# Correct-Prediction Feature Summary\n\n"
            "_No transparent features available — nothing to summarise._\n",
            encoding="utf-8",
        )
        return empty_csv

    feature_cols = [
        c for c in (*FEATURE_NAMES, *CONCEPT_NAMES)
        if c in next(iter(feats_by_id.values()), {})
    ]

    rows = []
    for pname, pack in per_pipeline.items():
        pred = pack["prediction"]
        ids = np.asarray(pred.image_ids)
        y = pred.labels.astype(int)
        yhat = pred.preds.astype(int)

        for bucket_name, lbl, prd in _CLASS_BUCKETS:
            mask = (y == lbl) & (yhat == prd)
            if not mask.any():
                continue
            bucket_ids = ids[mask]
            feat_rows = [feats_by_id[i] for i in bucket_ids if i in feats_by_id]
            if not feat_rows:
                continue
            bucket_df = pd.DataFrame(feat_rows)[feature_cols].astype(float)
            medians = bucket_df.median(axis=0)
            means = bucket_df.mean(axis=0)
            ranked = medians.sort_values(ascending=False)
            for rank, (feat, median) in enumerate(ranked.items(), start=1):
                rows.append(
                    {
                        "pipeline": pname,
                        "class": bucket_name,
                        "n_images": int(len(bucket_df)),
                        "feature": feat,
                        "median": float(median),
                        "mean": float(means[feat]),
                        "rank": rank,
                    }
                )

    summary = pd.DataFrame(rows, columns=[
        "pipeline", "class", "n_images", "feature", "median", "mean", "rank",
    ])
    out_csv = out_dir / "correct_prediction_feature_summary.csv"
    summary.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(summary))

    out_md = out_dir / "correct_prediction_feature_summary.md"
    out_md.write_text(_render_markdown(summary), encoding="utf-8")
    logger.info("Wrote %s", out_md)
    return out_csv


def _render_markdown(summary: pd.DataFrame) -> str:
    """Render ``summary`` as a per-pipeline, per-bucket top-K Markdown report."""
    lines: list[str] = []
    lines.append("# Correct-Prediction Feature Summary")
    lines.append("")
    lines.append(
        "For each pipeline and each of the four prediction buckets, this table "
        "lists the pixel-derived image features that were highest (by median) "
        "across the bucket's test images. All feature values are normalised to "
        "[0, 1] by `glassderm.data.features`, so medians are directly comparable."
    )
    lines.append("")
    lines.append(
        f"Only the top {_TOP_K_IN_MD} features per bucket are shown; see "
        "`correct_prediction_feature_summary.csv` for the full ranking."
    )
    lines.append("")
    lines.append("**Bucket legend**")
    lines.append("")
    for name, desc in _BUCKET_DESCRIPTIONS.items():
        lines.append(f"- `{name}` — {desc}")
    lines.append("")

    if summary.empty:
        lines.append("_No correct-prediction rows available for this scale._")
        lines.append("")
        return "\n".join(lines)

    for pname in sorted(summary["pipeline"].unique()):
        lines.append(f"## Pipeline: `{pname}`")
        lines.append("")
        sub = summary[summary["pipeline"] == pname]
        for bucket_name, _, _ in _CLASS_BUCKETS:
            bucket_sub = sub[sub["class"] == bucket_name]
            if bucket_sub.empty:
                lines.append(f"### {bucket_name}")
                lines.append("")
                lines.append("_No test images in this bucket._")
                lines.append("")
                continue
            n_images = int(bucket_sub["n_images"].iloc[0])
            lines.append(f"### {bucket_name}  (n = {n_images})")
            lines.append("")
            lines.append("| rank | feature | median | mean |")
            lines.append("|---:|:---|---:|---:|")
            top = bucket_sub.sort_values("rank").head(_TOP_K_IN_MD)
            for _, r in top.iterrows():
                lines.append(
                    f"| {int(r['rank'])} | `{r['feature']}` | "
                    f"{float(r['median']):.3f} | {float(r['mean']):.3f} |"
                )
            lines.append("")

    return "\n".join(lines)
