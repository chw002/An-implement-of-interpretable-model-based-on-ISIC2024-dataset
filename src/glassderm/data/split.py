"""Patient-disjoint train/val/test splitting.

We shuffle unique ``patient_id`` values (not rows), then allocate by fraction.
This guarantees no patient appears in two splits — the single most important
anti-leakage guard on ISIC 2024, since many patients contribute dozens of
lesions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils import dump_json, get_logger

logger = get_logger("glassderm.split")


def patient_disjoint_split(
    cohort: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 1337,
    processed_dir: str | Path | None = None,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)

    patients = cohort["patient_id"].astype(str).unique().tolist()
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))

    test_p = set(patients[:n_test])
    val_p = set(patients[n_test : n_test + n_val])
    train_p = set(patients[n_test + n_val :])

    def ids(subset: set) -> List[str]:
        sub = cohort[cohort.patient_id.astype(str).isin(subset)]
        return sub["image_id"].tolist()

    splits = {
        "train": ids(train_p),
        "val": ids(val_p),
        "test": ids(test_p),
    }
    meta = {
        "method": "patient_disjoint",
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "n_patients_total": n,
        "n_patients_train": len(train_p),
        "n_patients_val": len(val_p),
        "n_patients_test": len(test_p),
        "n_images_train": len(splits["train"]),
        "n_images_val": len(splits["val"]),
        "n_images_test": len(splits["test"]),
    }

    if processed_dir is not None:
        processed_dir = Path(processed_dir)
        dump_json({**splits, "_meta": meta}, processed_dir / "splits.json")
        logger.info("Saved splits → %s", processed_dir / "splits.json")

    logger.info(
        "Patient-disjoint split: %d train / %d val / %d test  "
        "(images: %d / %d / %d)",
        meta["n_patients_train"],
        meta["n_patients_val"],
        meta["n_patients_test"],
        meta["n_images_train"],
        meta["n_images_val"],
        meta["n_images_test"],
    )
    return {**splits, "_meta": meta}


def apply_splits(
    cohort: pd.DataFrame, splits: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def sub(key: str) -> pd.DataFrame:
        return cohort[cohort.image_id.isin(splits[key])].reset_index(drop=True)

    return sub("train"), sub("val"), sub("test")
