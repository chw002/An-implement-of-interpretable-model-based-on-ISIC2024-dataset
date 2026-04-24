"""Config loader: interpolation, overrides, attribute access."""
from __future__ import annotations

from pathlib import Path

import pytest

from glassderm.config import load_config


def test_default_config_parses():
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
    assert cfg.project.seed == 1337
    # interpolation resolved
    assert cfg.data.images_dir.endswith("ISIC_2024_Permissive_Training_Input")
    assert cfg.outputs.figures == "outputs/figures"


def test_override_applies():
    cfg = load_config(
        Path(__file__).resolve().parent.parent / "configs" / "default.yaml",
        overrides=["train.epochs=2", "project.seed=7"],
    )
    assert cfg.train.epochs == 2
    assert cfg.project.seed == 7
