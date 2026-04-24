"""Build the shared artefact dict consumed by every pipeline.

Keys:

* ``cohort``                     — pd.DataFrame with image_id, patient_id, label, image_path
* ``splits``                     — ``{"train": [...], "val": [...], "test": [...]}``
* ``cnn_loaders``                — ``{split: DataLoader}`` for CNN pipelines
* ``transparent_features``       — ``{split: pd.DataFrame of pixel-derived features}``
* ``transparent_features_by_id`` — ``{image_id: row-dict}``

**No ISIC/TBP metadata columns are ever added to this dict.**
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .config import Config
from .data import (
    CohortSpec,
    apply_splits,
    build_cohort,
    build_dataloaders,
    extract_features_for_cohort,
    load_cohort_splits,
    patient_disjoint_split,
)
from .data.datasets import concepts_from_features
from .data.download import locate_or_download
from .utils import get_logger

logger = get_logger("glassderm.artefacts")


def prepare_data(cfg: Config) -> Dict[str, Any]:
    locator = locate_or_download(cfg, allow_download=True)
    spec = CohortSpec(
        keep_all_malignant=bool(cfg.data.sample.keep_all_malignant),
        n_benign=_maybe_int(cfg.data.sample.get("n_benign")),
        seed=int(cfg.data.sample.seed),
    )
    cohort = build_cohort(locator, processed_dir=cfg.data.processed_dir, spec=spec)
    splits = patient_disjoint_split(
        cohort,
        val_fraction=float(cfg.data.split.val_fraction),
        test_fraction=float(cfg.data.split.test_fraction),
        seed=int(cfg.data.split.seed),
        processed_dir=cfg.data.processed_dir,
    )
    feats_path = Path(cfg.data.features_cache)
    extract_features_for_cohort(cohort, feats_path)
    return {"cohort": cohort, "splits": splits, "feature_cache": feats_path}


def build_artefacts(
    cfg: Config,
    *,
    need_images: bool = True,
    cohort_override: Optional[pd.DataFrame] = None,
    splits_override: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    locator = locate_or_download(cfg, allow_download=False)
    if cohort_override is not None and splits_override is not None:
        cohort = cohort_override
        splits = splits_override
        train_df, val_df, test_df = apply_splits(cohort, splits)
    else:
        train_df, val_df, test_df = load_cohort_splits(cfg.data.processed_dir)
        cohort = pd.concat([train_df, val_df, test_df], ignore_index=True)
        splits = {
            "train": train_df["image_id"].tolist(),
            "val": val_df["image_id"].tolist(),
            "test": test_df["image_id"].tolist(),
        }

    artefacts: Dict[str, Any] = {
        "locator": locator,
        "cohort": cohort,
        "splits": splits,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }

    feats = extract_features_for_cohort(cohort=cohort, out_parquet=cfg.data.features_cache)
    feats_by_id = feats.set_index("image_id")

    split_feats = {}
    for split, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        ids = df["image_id"].tolist()
        split_feats[split] = feats[feats["image_id"].isin(ids)].reset_index(drop=True)
    artefacts["transparent_features"] = split_feats
    artefacts["transparent_features_by_id"] = feats_by_id.to_dict(orient="index")

    if need_images:
        concepts_by_id = concepts_from_features(feats)
        artefacts["cnn_loaders"] = build_dataloaders(
            train_df,
            val_df,
            test_df,
            image_size=int(cfg.image.size),
            mean=list(cfg.image.mean),
            std=list(cfg.image.std),
            batch_size=int(cfg.train.batch_size),
            num_workers=int(cfg.train.num_workers),
            concepts_by_id=concepts_by_id,
            seed=int(cfg.project.seed),
        )
    return artefacts


def _maybe_int(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, str) and x.lower() in {"null", "none", ""}:
        return None
    return int(x)
