"""Pytest fixtures — synthesize a tiny dataset so the suite runs anywhere.

The tiny cohort mirrors the real ISIC 2024 release layout:
    <raw>/ISIC_2024_Permissive_Training_GroundTruth.csv
    <raw>/ISIC_2024_Permissive_Training_Supplement.csv
    <raw>/ISIC_2024_Permissive_Training_Input/*.jpg
    <raw>/ISIC_2024_Permissive_Training_Input/metadata.csv

We deliberately include `tbp_lv_*` / `clin_size_long_diam_mm` columns in the
synthetic `metadata.csv` so we can verify that the image-only pipeline
refuses to let them reach any model input (see :mod:`glassderm.data.audit`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    import cv2
    import numpy as np
    import pandas as pd

    root = tmp_path_factory.mktemp("tiny_isic")
    raw = root / "raw" / "isic2024_official"
    img_dir = raw / "ISIC_2024_Permissive_Training_Input"
    img_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows_gt, rows_sup, rows_meta = [], [], []
    n = 40
    for i in range(n):
        iid = f"SYN_{i:04d}"
        label = int(i < n // 4)
        patient = f"P{i // 5}"
        img = (np.ones((96, 96, 3), dtype=np.uint8) * 220)
        center = (48, 48)
        axes = (20 + (5 if label else 0), 18 + (9 if label else 0))
        angle = int(rng.integers(0, 180))
        color = (60, 40, 40) if label else (80, 70, 100)
        cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
        if label:
            cv2.circle(img, (55, 50), 4, (30, 30, 30), -1)
        cv2.imwrite(str(img_dir / f"{iid}.jpg"), img)

        rows_gt.append({"isic_id": iid, "malignant": label})
        rows_sup.append({"isic_id": iid, "lesion_id": patient})
        rows_meta.append(
            {
                "isic_id": iid,
                "patient_id": patient,
                "tbp_lv_symm_2axis": float(label) + rng.normal(0, 0.1),
                "tbp_lv_norm_border": float(label) * 5 + rng.normal(0, 0.5),
                "tbp_lv_norm_color": float(label) * 0.8 + rng.normal(0, 0.2),
                "clin_size_long_diam_mm": 3 + float(label) * 4 + rng.normal(0, 0.3),
            }
        )

    pd.DataFrame(rows_gt).to_csv(
        raw / "ISIC_2024_Permissive_Training_GroundTruth.csv", index=False
    )
    pd.DataFrame(rows_sup).to_csv(
        raw / "ISIC_2024_Permissive_Training_Supplement.csv", index=False
    )
    pd.DataFrame(rows_meta).to_csv(img_dir / "metadata.csv", index=False)
    return raw


@pytest.fixture()
def tiny_cfg(tmp_path, tiny_dataset):
    """Config pointed at the tiny synthetic cohort.

    Note: `cfg.pipelines` only holds the five current pipelines
    (multitask_cnn, hard_cbm, glassbox_nam, transparent_lr, transparent_tree).
    There is **no** legacy ``pipelines.transparent`` key.
    """
    from glassderm.config import load_config

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.set_path("data.raw_dir", str(tiny_dataset))
    cfg.set_path(
        "data.images_dir", str(tiny_dataset / "ISIC_2024_Permissive_Training_Input")
    )
    cfg.set_path(
        "data.gt_csv",
        str(tiny_dataset / "ISIC_2024_Permissive_Training_GroundTruth.csv"),
    )
    cfg.set_path(
        "data.supplement_csv",
        str(tiny_dataset / "ISIC_2024_Permissive_Training_Supplement.csv"),
    )
    cfg.set_path(
        "data.metadata_csv",
        str(tiny_dataset / "ISIC_2024_Permissive_Training_Input" / "metadata.csv"),
    )
    cfg.set_path("data.processed_dir", str(tmp_path / "processed"))
    cfg.set_path("data.features_dir", str(tmp_path / "features"))
    cfg.set_path("data.cache_dir", str(tmp_path / "cache"))
    cfg.set_path(
        "data.features_cache",
        str(tmp_path / "features" / "transparent_features.parquet"),
    )
    cfg.set_path("outputs.root", str(tmp_path / "outputs"))
    cfg.set_path("outputs.checkpoints", str(tmp_path / "outputs" / "checkpoints"))
    cfg.set_path("outputs.tables", str(tmp_path / "outputs" / "tables"))
    cfg.set_path("outputs.figures", str(tmp_path / "outputs" / "figures"))
    cfg.set_path("outputs.reports", str(tmp_path / "outputs" / "reports"))
    cfg.set_path("outputs.case_studies", str(tmp_path / "outputs" / "case_studies"))
    cfg.set_path("outputs.logs", str(tmp_path / "outputs" / "logs"))
    cfg.set_path("data.sample.n_benign", 30)
    cfg.set_path("train.batch_size", 8)
    cfg.set_path("train.num_workers", 0)
    cfg.set_path("train.epochs", 1)
    cfg.set_path("train.amp", False)
    cfg.set_path("image.size", 64)
    cfg.set_path("project.device", "cpu")
    cfg.set_path("analysis.n_case_studies_per_bucket", 1)
    return cfg
