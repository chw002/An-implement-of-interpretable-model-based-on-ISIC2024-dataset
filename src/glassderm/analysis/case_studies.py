"""Curated case studies — one markdown per scale, no scattered files.

For each scale we publish a single ``case_studies.md`` holding eight
narratives (by default):

* 2 × true-positive malignant
* 2 × false-negative malignant
* 2 × false-positive benign
* 2 × true-negative benign

Each narrative stitches together every pipeline's ``explain()`` output for
the same test image so the reader can see the transparent and the opaque
pipelines side-by-side on the same lesion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from ..pipelines import TRANSPARENCY_TAGS
from ..utils import dump_json, ensure_dir, get_logger

logger = get_logger("glassderm.cases")


_BUCKETS = (
    ("true_positive_malignant", "TP · malignant correctly flagged", 1, 1),
    ("false_negative_malignant", "FN · malignant missed", 1, 0),
    ("false_positive_benign",    "FP · benign wrongly flagged", 0, 1),
    ("true_negative_benign",     "TN · benign correctly cleared", 0, 0),
)


def build_case_studies(
    per_pipeline: Mapping[str, Mapping[str, Any]],
    artefacts: Mapping[str, Any],
    *,
    n_per_bucket: int = 2,
    out_dir: str | Path,
    cohort: pd.DataFrame,
    seed: int = 1337,
) -> List[Path]:
    """Write ``<out_dir>/case_studies.md`` + a manifest json.

    Parameters
    ----------
    per_pipeline
        Mapping ``{name: {"pipe": Pipeline, "prediction": PipelinePrediction,
        "metrics": dict}}``.
    artefacts
        Shared artefact dict (holds ``transparent_features_by_id`` so explain()
        can look up pixel features).
    n_per_bucket
        Number of narratives per confusion-matrix bucket.  2 ⇒ 8 total.
    cohort
        Master cohort frame; used to surface patient_id + image_path for context.
    """
    out_dir = ensure_dir(out_dir)
    if not per_pipeline:
        (out_dir / "case_studies.md").write_text("# Case studies\n\n(No pipelines produced predictions for this scale.)\n")
        return [out_dir / "case_studies.md"]

    anchor_name = _choose_anchor(per_pipeline)
    anchor_pred = per_pipeline[anchor_name]["prediction"]

    picks = _pick_cases(anchor_pred, n_per_bucket=n_per_bucket, seed=seed)

    cohort_by_id = cohort.set_index("image_id") if "image_id" in cohort.columns else pd.DataFrame()

    markdown_lines: List[str] = []
    markdown_lines.append("# Case studies")
    markdown_lines.append("")
    markdown_lines.append(
        f"_Anchor pipeline: **{anchor_name}**. All narratives share the same "
        f"test image across every pipeline so the reader can compare "
        f"explanation styles side-by-side._"
    )
    markdown_lines.append("")

    manifest: Dict[str, Any] = {"anchor_pipeline": anchor_name, "buckets": {}}

    for bucket_key, bucket_title, _, _ in _BUCKETS:
        entries = picks.get(bucket_key, [])
        manifest["buckets"][bucket_key] = [e["image_id"] for e in entries]
        markdown_lines.append(f"## {bucket_title}")
        markdown_lines.append("")
        if not entries:
            markdown_lines.append("_No samples in this bucket at test time._")
            markdown_lines.append("")
            continue

        for entry in entries:
            image_id = entry["image_id"]
            anchor_prob = entry["prob"]
            patient_id = (
                str(cohort_by_id.loc[image_id, "patient_id"])
                if image_id in cohort_by_id.index else "(unknown)"
            )
            image_path = (
                str(cohort_by_id.loc[image_id, "image_path"])
                if image_id in cohort_by_id.index else "(unknown)"
            )

            markdown_lines.append(f"### image_id = `{image_id}`")
            markdown_lines.append("")
            markdown_lines.append(f"- patient_id = `{patient_id}`")
            markdown_lines.append(f"- image_path = `{image_path}`")
            markdown_lines.append(
                f"- anchor ({anchor_name}) P(malignant) = {anchor_prob:.3f}"
            )
            markdown_lines.append("")

            for pname, pack in per_pipeline.items():
                pipe = pack["pipe"]
                pred = pack["prediction"]
                idx = _find_index(pred, image_id)
                if idx is None:
                    markdown_lines.append(
                        f"**{pname}** (`{TRANSPARENCY_TAGS.get(pname, pipe.transparency_tag)}`): "
                        "this image did not appear in this pipeline's test predictions."
                    )
                    markdown_lines.append("")
                    continue

                prob = float(pred.probs[idx])
                prd = int(pred.preds[idx])
                label = int(pred.labels[idx])
                row = {
                    "image_id": image_id,
                    "label": label,
                    "prob": prob,
                    "pred": prd,
                }
                if pred.concepts is not None:
                    for j, letter in enumerate("ABCD"):
                        row[f"concept_{letter}"] = float(pred.concepts[idx, j])

                try:
                    explanation = pipe.explain(row, artefacts)
                    narrative = explanation.get("text", "(no explain text)")
                    verdict = explanation.get("verdict", "?")
                except Exception as exc:
                    narrative = f"(explain raised: {exc!r})"
                    verdict = "?"

                markdown_lines.append(
                    f"**{pname}** "
                    f"(`{TRANSPARENCY_TAGS.get(pname, pipe.transparency_tag)}`) — "
                    f"P(mal)={prob:.3f} τ={pipe.threshold:.3f} "
                    f"→ pred={'M' if prd else 'B'} / verdict={verdict}"
                )
                markdown_lines.append("")
                markdown_lines.append("```")
                markdown_lines.append(narrative)
                markdown_lines.append("```")
                markdown_lines.append("")

            markdown_lines.append("---")
            markdown_lines.append("")

    out_path = out_dir / "case_studies.md"
    out_path.write_text("\n".join(markdown_lines))
    dump_json(manifest, out_dir / "case_studies_manifest.json")
    logger.info("Wrote curated case studies → %s", out_path)
    return [out_path]


# ------------------------------------------------------------------- helpers
def _choose_anchor(per_pipeline: Mapping[str, Mapping[str, Any]]) -> str:
    """Prefer the most-transparent pipeline as anchor (LR → Tree → others)."""
    preference = ["transparent_lr", "transparent_tree", "hard_cbm", "glassbox_nam", "multitask_cnn"]
    for name in preference:
        if name in per_pipeline:
            return name
    return next(iter(per_pipeline))


def _pick_cases(pred, *, n_per_bucket: int, seed: int) -> Dict[str, list]:
    rng = np.random.default_rng(seed)
    ids = np.asarray(pred.image_ids)
    y = pred.labels.astype(int)
    yhat = pred.preds.astype(int)
    probs = pred.probs.astype(float)
    picks: Dict[str, list] = {}
    for key, _, lbl, prd in _BUCKETS:
        mask = (y == lbl) & (yhat == prd)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            picks[key] = []
            continue
        # prefer the most decisive (far from τ) — tie-broken with the RNG
        order = sorted(
            idxs.tolist(),
            key=lambda i: (-abs(float(probs[i]) - pred.threshold), int(rng.integers(1 << 30))),
        )
        chosen = order[:n_per_bucket]
        picks[key] = [
            {"image_id": str(ids[i]), "prob": float(probs[i])} for i in chosen
        ]
    return picks


def _find_index(pred, image_id: str) -> int | None:
    ids = list(pred.image_ids)
    try:
        return ids.index(image_id)
    except ValueError:
        return None
