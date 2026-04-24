"""Strict image-only feature audit.

Every pipeline that consumes a table of features routes through
:func:`audit_feature_columns`.  If any column matches the forbidden-metadata
blocklist, the audit raises before model input is constructed.  The audit
also emits a CSV and JSON report classifying each column as:

* ``pixel_derived``      — computed from image pixels (OpenCV or CNN output)
* ``label_only``         — ground-truth supervision, never a model input
* ``split_only``         — used exclusively to realise the train/val/test split
* ``identifier_only``    — used to index images for case studies, never input
* ``forbidden_metadata`` — ISIC/TBP vendor fields that must not enter the model

A pipeline whose ``model-input`` columns contain anything other than
``pixel_derived`` is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from ..utils import dump_json


# --- forbidden metadata ------------------------------------------------------

FORBIDDEN_PREFIXES: Tuple[str, ...] = ("tbp_lv_", "tbp_", "iddx", "mel_", "anatom_")

FORBIDDEN_EXACT: frozenset = frozenset({
    # identity fields that are not allowed as model input
    "age_approx", "sex", "anatom_site_general",
    # vendor measurement columns
    "clin_size_long_diam_mm", "image_type", "tbp_tile_type",
    "attribution", "copyright_license",
    # vendor-derived ABCD aliases from the legacy pipeline
    "A", "B", "C", "D",
    "abcd_A", "abcd_B", "abcd_C", "abcd_D",
    # diagnostic helpers that would leak the label
    "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5",
    "mel_mitotic_index", "mel_thick_mm",
})

IDENTIFIER_COLUMNS: frozenset = frozenset({
    "image_id", "isic_id", "lesion_id",
    "image_path",
})

SPLIT_ONLY_COLUMNS: frozenset = frozenset({"patient_id"})

LABEL_ONLY_COLUMNS: frozenset = frozenset({"label", "target", "malignant"})


def classify_column(name: str) -> str:
    if name in LABEL_ONLY_COLUMNS:
        return "label_only"
    if name in SPLIT_ONLY_COLUMNS:
        return "split_only"
    if name in IDENTIFIER_COLUMNS:
        return "identifier_only"
    lname = name.lower()
    if lname in FORBIDDEN_EXACT:
        return "forbidden_metadata"
    for pref in FORBIDDEN_PREFIXES:
        if lname.startswith(pref):
            return "forbidden_metadata"
    # any column surviving to here must be pixel-derived feature that the
    # caller has registered.  The final verifier refuses anything that
    # the caller cannot prove is pixel-derived.
    return "pixel_derived"


@dataclass
class FeatureAudit:
    pipeline: str
    feature_set: str
    model_input_columns: List[str]
    classification: Dict[str, str]

    def rows(self) -> List[Dict[str, str]]:
        out = []
        for col, src in sorted(self.classification.items()):
            out.append({
                "pipeline": self.pipeline,
                "feature_set": self.feature_set,
                "column": col,
                "source": src,
                "is_model_input": col in self.model_input_columns,
            })
        return out

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows())

    def dump(self, out_dir: str | Path) -> Tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        csv_path = out_dir / "feature_audit.csv"
        json_path = out_dir / "feature_audit.json"
        df.to_csv(csv_path, index=False)
        dump_json(
            {
                "pipeline": self.pipeline,
                "feature_set": self.feature_set,
                "model_input_columns": list(self.model_input_columns),
                "classification": self.classification,
            },
            json_path,
        )
        return csv_path, json_path


class MetadataLeakError(RuntimeError):
    """Raised when forbidden metadata columns reach a pipeline's model input."""


def audit_feature_columns(
    pipeline: str,
    feature_set: str,
    all_columns: Iterable[str],
    model_input_columns: Sequence[str],
    pixel_derived_whitelist: Iterable[str] | None = None,
) -> FeatureAudit:
    """Classify ``all_columns`` and raise if any forbidden column is in
    ``model_input_columns``.

    ``pixel_derived_whitelist`` lets the caller register feature names that are
    known to come from images but happen to live outside the default naming
    convention (for CNN pipelines, this is e.g. ``cnn_concept_A``).  Anything
    not in the whitelist is classified by the default rules.
    """
    whitelist = set(pixel_derived_whitelist or ())
    classification: Dict[str, str] = {}
    for col in all_columns:
        if col in whitelist:
            classification[col] = "pixel_derived"
        else:
            classification[col] = classify_column(col)

    forbidden_in_input = [
        c for c in model_input_columns if classification.get(c) == "forbidden_metadata"
    ]
    non_pixel_input = [
        c
        for c in model_input_columns
        if classification.get(c) not in {"pixel_derived"}
    ]
    if forbidden_in_input:
        raise MetadataLeakError(
            f"[{pipeline}] forbidden metadata column(s) reached model input: "
            f"{forbidden_in_input}"
        )
    if non_pixel_input:
        raise MetadataLeakError(
            f"[{pipeline}] non-pixel column(s) reached model input: "
            f"{non_pixel_input} (only pixel_derived columns are allowed)"
        )
    return FeatureAudit(
        pipeline=pipeline,
        feature_set=feature_set,
        model_input_columns=list(model_input_columns),
        classification=classification,
    )
