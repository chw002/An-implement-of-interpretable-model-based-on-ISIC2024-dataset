"""Image-only cohort construction.

The cohort CSV contains *only* what downstream pipelines need:

* ``image_id``   — used solely to index images / write case studies
* ``patient_id`` — used solely to realise a patient-disjoint split
* ``label``      — ground-truth supervision
* ``image_path`` — resolved path to the JPEG on disk

**Nothing else.**  No ``age_approx``, no ``sex``, no ``tbp_lv_*``, no vendor
ABCD columns — every one of those is flagged in
:mod:`glassderm.data.audit` and forbidden from model input.

If the caller asks for a cohort of ``n_benign=40000`` we sample benigns
(patient-aware) to that count and keep all malignant.  Passing ``None`` keeps
every benign image (the "full" scale).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils import dump_json, get_logger
from .download import DatasetLocator

logger = get_logger("glassderm.sample")


@dataclass
class CohortSpec:
    keep_all_malignant: bool = True
    n_benign: Optional[int] = None       # None → keep all benign
    seed: int = 1337


def build_cohort(
    loc: DatasetLocator,
    processed_dir: str | Path,
    spec: CohortSpec,
    out_csv: str = "cohort.csv",
) -> pd.DataFrame:
    """Merge GT + supplement + metadata strictly for identity/split/label columns."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading ISIC 2024 source CSVs from %s", loc.raw_dir)
    gt = pd.read_csv(loc.gt_csv)
    sup = pd.read_csv(loc.supplement_csv)
    meta = pd.read_csv(loc.metadata_csv, low_memory=False)

    gt = gt.rename(columns={"isic_id": "image_id"})
    gt["label"] = (gt["malignant"].astype(float) >= 0.5).astype(int)
    gt = gt[["image_id", "label"]]

    sup = sup.rename(columns={"isic_id": "image_id"})
    if "lesion_id" in sup.columns:
        sup = sup[["image_id", "lesion_id"]]
    else:
        sup = sup[["image_id"]]

    meta = meta.rename(columns={"isic_id": "image_id"})
    if "patient_id" not in meta.columns:
        raise KeyError(
            "metadata.csv has no patient_id column — cannot build patient-disjoint split"
        )
    meta = meta[["image_id", "patient_id"]].copy()

    merged = gt.merge(meta, on="image_id", how="inner").merge(
        sup, on="image_id", how="left"
    )
    merged["patient_id"] = (
        merged["patient_id"]
        .fillna(merged.get("lesion_id"))
        .fillna(merged["image_id"])
        .astype(str)
    )
    merged["image_path"] = merged["image_id"].apply(
        lambda x: str((loc.images_dir / f"{x}.jpg").resolve())
    )

    # Strictly drop anything we don't need — safer than relying on downstream
    # selection.
    merged = merged[["image_id", "patient_id", "label", "image_path"]].copy()

    n_mal = int(merged.label.sum())
    n_ben = int((merged.label == 0).sum())
    logger.info(
        "Source pool: %d images, %d malignant / %d benign, %d unique patients",
        len(merged),
        n_mal,
        n_ben,
        merged["patient_id"].nunique(),
    )

    cohort = _sample(merged, spec)
    out_path = processed_dir / out_csv
    cohort.to_csv(out_path, index=False)
    logger.info("Saved cohort → %s (%d rows)", out_path, len(cohort))

    dump_json(
        {
            "source_pool_rows": int(len(merged)),
            "source_pool_malignant": n_mal,
            "source_pool_benign": n_ben,
            "cohort_rows": int(len(cohort)),
            "cohort_malignant": int(cohort.label.sum()),
            "cohort_benign": int((cohort.label == 0).sum()),
            "spec": {
                "keep_all_malignant": spec.keep_all_malignant,
                "n_benign": spec.n_benign,
                "seed": spec.seed,
            },
            "schema": {
                "image_id": "identifier_only",
                "patient_id": "split_only",
                "label": "label_only",
                "image_path": "identifier_only",
            },
            "metadata_policy": (
                "No ISIC/TBP metadata columns are retained in the cohort — "
                "every model input must be computed from image pixels."
            ),
        },
        processed_dir / "cohort_provenance.json",
    )
    return cohort


def _sample(df: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    rng_seed = int(spec.seed)
    mal = df[df.label == 1]
    ben = df[df.label == 0]

    mal_keep = mal if spec.keep_all_malignant else mal.sample(
        n=min(len(mal), spec.n_benign or len(mal)),
        random_state=rng_seed,
    )

    if spec.n_benign is None or spec.n_benign >= len(ben):
        ben_keep = ben
    else:
        ben_keep = ben.sample(n=int(spec.n_benign), random_state=rng_seed)

    cohort = pd.concat([mal_keep, ben_keep], ignore_index=True)
    cohort = cohort.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
    logger.info(
        "Cohort sampled: %d malignant (kept %s) + %d benign (target %s)",
        len(mal_keep),
        "all" if spec.keep_all_malignant else "subsampled",
        len(ben_keep),
        "all" if spec.n_benign is None else spec.n_benign,
    )
    return cohort
