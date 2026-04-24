"""Per-pipeline and per-scale textual readouts.

All four helpers are pure *output-formatting* routines — they accept
already-fitted pipelines and evaluation predictions and write human-readable
artefacts into the scale's output directory.  No training, no metric
recomputation.

Files written
-------------
* ``transparent_lr_readout.json``     full LR weights + contribution chain per case
* ``transparent_lr_readout.txt``      same, as reviewer-friendly prose
* ``transparent_tree_readout.json``   tree report (scaler, leaves, full rules)
* ``transparent_tree_rules.txt``      ``export_text(tree)`` output verbatim
* ``transparent_tree_cases.txt``      per-case decision-path trace
* ``hard_cbm_readout.txt``            ``logit = w_A·A + w_B·B + w_C·C + w_D·D + bias``
* ``README_RESULTS.md``               one-page scale summary for humans
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils import dump_json

if TYPE_CHECKING:  # avoid import cycles at runtime
    from ..pipelines.base import Pipeline, PipelinePrediction
    from ..scale import ScaleRunConfig


_CASE_BUCKETS = (
    ("true_positive_malignant", 1, 1),   # (label, pred)
    ("false_negative_malignant", 1, 0),
    ("false_positive_benign",    0, 1),
    ("true_negative_benign",     0, 0),
)


def _pick_cases(
    pred: "PipelinePrediction",
    per_bucket: int = 2,
) -> dict:
    """Select representative (label, pred) cases per bucket for narrative output.

    We prefer the highest-confidence members of each bucket (furthest from τ)
    because they are the most illustrative for the dissertation.
    """
    ids = np.asarray(pred.image_ids)
    y = pred.labels.astype(int)
    yhat = pred.preds.astype(int)
    probs = pred.probs.astype(float)
    order = np.argsort(-np.abs(probs - pred.threshold))
    picks = {}
    for name, lbl, prd in _CASE_BUCKETS:
        mask = (y == lbl) & (yhat == prd)
        chosen = [int(i) for i in order if bool(mask[i])][:per_bucket]
        picks[name] = [(str(ids[i]), float(probs[i])) for i in chosen]
    return picks


# ----------------------------------------------------------- Transparent LR --
def write_transparent_lr_readouts(
    pipe: "Pipeline",
    artefacts: Mapping[str, Any],
    pred: "PipelinePrediction",
    out_dir: Path,
    cohort: pd.DataFrame,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = pipe.report()

    weights = report["weights"]
    ranked = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_features = [{"feature": k, "weight": float(v)} for k, v in ranked]

    picked = _pick_cases(pred, per_bucket=2)
    case_traces = []
    for bucket, cases in picked.items():
        for image_id, prob in cases:
            explanation = pipe.explain({"image_id": image_id}, artefacts)
            contribs = sorted(
                explanation["contributions"].items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:6]
            case_traces.append(
                {
                    "bucket": bucket,
                    "image_id": image_id,
                    "prob": prob,
                    "verdict": explanation["verdict"],
                    "logit": explanation["logit"],
                    "bias": explanation["bias"],
                    "top_contributions": [
                        {"feature": k, "contribution": float(v)} for k, v in contribs
                    ],
                }
            )

    payload = {
        "pipeline": pipe.name,
        "transparency": pipe.transparency,
        "transparency_tag": pipe.transparency_tag,
        "threshold": report["threshold"],
        "bias": report["bias"],
        "n_features": len(report["feature_columns"]),
        "weights_ranked": top_features,
        "scaler": report["scaler"],
        "cases": case_traces,
    }
    dump_json(payload, out_dir / "transparent_lr_readout.json")

    lines: list[str] = []
    lines.append("# Transparent-LR readout")
    lines.append("")
    lines.append("Formula: logit = Σ wᵢ · scaled(xᵢ) + bias")
    lines.append(f"  bias        = {report['bias']:+.4f}")
    lines.append(f"  threshold τ = {report['threshold']:.3f}")
    lines.append(f"  n features  = {len(report['feature_columns'])}")
    lines.append("")
    lines.append("## Weights (ranked by |w|)")
    for item in top_features:
        lines.append(f"  {item['feature']:<26}  w = {item['weight']:+.4f}")
    lines.append("")
    lines.append("## Representative cases (validation-tuned threshold)")
    for trace in case_traces:
        lines.append(
            f"### {trace['bucket']} · image_id={trace['image_id']} · "
            f"prob={trace['prob']:.3f} → {trace['verdict']}"
        )
        lines.append(f"  logit = {trace['logit']:+.3f}  (bias = {trace['bias']:+.3f})")
        for c in trace["top_contributions"]:
            lines.append(f"    {c['feature']:<26}  contrib = {c['contribution']:+.3f}")
        lines.append("")
    (out_dir / "transparent_lr_readout.txt").write_text("\n".join(lines))


# --------------------------------------------------------- Transparent Tree --
def write_transparent_tree_readouts(
    pipe: "Pipeline",
    artefacts: Mapping[str, Any],
    pred: "PipelinePrediction",
    out_dir: Path,
    cohort: pd.DataFrame,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = pipe.report()

    header = (
        f"# Decision tree rules (max_depth={report['max_depth']}, "
        f"min_samples_leaf={report['min_samples_leaf']}, "
        f"threshold τ={report['threshold']:.3f})\n"
        "# Feature values are min–max scaled to [0,1] using train-fit ranges.\n\n"
    )
    (out_dir / "transparent_tree_rules.txt").write_text(
        header + (report.get("tree_rules") or "(no rules)")
    )

    picked = _pick_cases(pred, per_bucket=2)
    case_lines: list[str] = []
    case_payload = []
    for bucket, cases in picked.items():
        case_lines.append(f"=== {bucket} ===")
        for image_id, prob in cases:
            explanation = pipe.explain({"image_id": image_id}, artefacts)
            case_lines.append(explanation["text"])
            case_lines.append("")
            case_payload.append(
                {
                    "bucket": bucket,
                    "image_id": image_id,
                    "prob": prob,
                    "verdict": explanation["verdict"],
                    "path": explanation["path"],
                    "leaf_counts": explanation["leaf_counts"],
                }
            )
    (out_dir / "transparent_tree_cases.txt").write_text("\n".join(case_lines))

    dump_json(
        {
            "pipeline": pipe.name,
            "transparency": pipe.transparency,
            "transparency_tag": pipe.transparency_tag,
            "threshold": report["threshold"],
            "max_depth": report["max_depth"],
            "min_samples_leaf": report["min_samples_leaf"],
            "feature_columns": report["feature_columns"],
            "scaler": report["scaler"],
            "leaves": report["leaves"],
            "tree_rules": report.get("tree_rules"),
            "cases": case_payload,
        },
        out_dir / "transparent_tree_readout.json",
    )


# ------------------------------------------------------------- Hard CBM ------
def write_hard_cbm_report(pipe: "Pipeline", out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(pipe, "weights_readout"):
        out_path.write_text("(HardCBM pipeline did not expose weights_readout)\n")
        return out_path
    wb = pipe.weights_readout()
    lines = [
        "# HardCBM linear readout",
        "",
        "logit = w_A · A + w_B · B + w_C · C + w_D · D + bias",
        "",
        f"  w_A   = {wb['w_A']:+.4f}    (Asymmetry)",
        f"  w_B   = {wb['w_B']:+.4f}    (Border)",
        f"  w_C   = {wb['w_C']:+.4f}    (Color)",
        f"  w_D   = {wb['w_D']:+.4f}    (Diameter)",
        f"  bias  = {wb['bias']:+.4f}",
        f"  τ     = {float(pipe.threshold):.3f}   (validation-tuned)",
        "",
        "CNN concepts are supervised against pixel-derived ABCD proxies.",
        "No ISIC/TBP metadata enters this pipeline.",
    ]
    out_path.write_text("\n".join(lines))
    dump_json(
        {**wb, "threshold": float(pipe.threshold)},
        out_path.with_suffix(".json"),
    )
    return out_path


# ------------------------------------------------------------- Scale README --
def write_readme_results(
    scale_out: Path,
    scale_cfg: "ScaleRunConfig",
    main_metrics_df: pd.DataFrame,
) -> None:
    scale_out = Path(scale_out)
    budget = scale_cfg.benign_budget
    size_str = "full (all benign)" if budget is None else f"{budget} benign"

    lines: list[str] = []
    lines.append(f"# Scale: {scale_cfg.label}")
    lines.append("")
    lines.append(f"- Training budget: **{size_str}**, every malignant kept.")
    lines.append("- Val/test split: patient-disjoint, fixed across all scales.")
    lines.append("- Features: pixel-derived only (see `feature_audit.csv`).")
    lines.append("- Thresholds tuned on val set only (`tables/thresholds.csv`).")
    lines.append("")
    lines.append("## Test metrics")
    lines.append("")
    if main_metrics_df is not None and len(main_metrics_df) > 0:
        cols = [
            "pipeline", "transparency_tag", "auc",
            "balanced_accuracy", "recall", "specificity",
            "precision", "f1", "threshold",
            "tn", "fp", "fn", "tp",
        ]
        present = [c for c in cols if c in main_metrics_df.columns]
        df = main_metrics_df[present].copy()
        for c in ("auc", "balanced_accuracy", "recall", "specificity",
                  "precision", "f1", "threshold"):
            if c in df.columns:
                df[c] = df[c].astype(float).round(4)
        try:
            lines.append(df.to_markdown(index=False))
        except ImportError:
            lines.append("```")
            lines.append(df.to_string(index=False))
            lines.append("```")
    else:
        lines.append("(no pipelines produced metrics)")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `cohort_provenance.json` – row counts per split + malignant/benign.")
    lines.append("- `splits.json` – exact image_id lists per split.")
    lines.append("- `config_snapshot.yaml` – full config used for this scale.")
    lines.append("- `tables/main_metrics.csv`, `tables/thresholds.csv` – numeric results.")
    lines.append("- `<pipeline>/feature_audit.csv` + `feature_audit.json` – no-metadata contract.")
    lines.append("- `<pipeline>/predictions_test.csv` – per-image test probabilities.")
    lines.append("- `case_studies/` – human-readable case narratives.")
    lines.append("- `figures/` – per-scale figures for dissertation.")
    (scale_out / "README_RESULTS.md").write_text("\n".join(lines))
