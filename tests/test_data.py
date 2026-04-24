"""Cohort construction, patient-disjoint split, image-only feature extraction.

The important invariants verified here:

* The cohort written to disk contains *only* `image_id, patient_id, label,
  image_path` — no ISIC/TBP metadata columns.
* The patient-disjoint split puts each patient in exactly one of train/val/test.
* The transparent features table exposes every pixel-derived feature name, and
  also the four ABCD concept proxies.
"""
from __future__ import annotations

import pandas as pd

from glassderm.artefacts import build_artefacts, prepare_data
from glassderm.data import CONCEPT_NAMES, FEATURE_NAMES
from glassderm.data.audit import FORBIDDEN_EXACT, FORBIDDEN_PREFIXES


def test_prepare_data_pipeline(tiny_cfg):
    prepare_data(tiny_cfg)

    cohort_path = pd.read_csv(f"{tiny_cfg.data.processed_dir}/cohort.csv")
    assert len(cohort_path) > 0
    assert set(cohort_path.columns) == {"image_id", "patient_id", "label", "image_path"}, (
        f"cohort leaked non-image columns: {set(cohort_path.columns)}"
    )
    for col in cohort_path.columns:
        assert col not in FORBIDDEN_EXACT
        assert not any(col.startswith(pref) for pref in FORBIDDEN_PREFIXES)

    arte = build_artefacts(tiny_cfg, need_images=False)
    tf = arte["transparent_features"]
    assert set(tf).issuperset({"train", "val", "test"})
    patients = {k: set(arte[f"{k}_df"]["patient_id"]) for k in ("train", "val", "test")}
    assert patients["train"].isdisjoint(patients["val"])
    assert patients["train"].isdisjoint(patients["test"])
    assert patients["val"].isdisjoint(patients["test"])

    feats = tf["train"]
    assert set(FEATURE_NAMES).issubset(feats.columns)
    assert set(CONCEPT_NAMES).issubset(feats.columns)
    for col in feats.columns:
        assert col not in FORBIDDEN_EXACT, f"forbidden metadata column leaked: {col}"
        assert not any(col.startswith(pref) for pref in FORBIDDEN_PREFIXES), (
            f"forbidden metadata prefix leaked: {col}"
        )
