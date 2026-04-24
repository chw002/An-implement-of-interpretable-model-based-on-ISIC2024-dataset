"""Locate or download the ISIC 2024 permissive training release.

We *prefer* an already-present copy (because the release is ~40 GB of JPEGs and
re-downloading is unfriendly to bandwidth).  The locator checks, in order:

1. ``config.data.raw_dir`` — what the user has configured;
2. a small list of known fallback paths (see :data:`KNOWN_FALLBACKS`);
3. a fresh Kaggle download, if credentials are present.

If none of the three succeed we raise :class:`DatasetNotFound` with a human
readable error that tells the user exactly what files we expect and where to
put them.  There is no silent "succeed with garbage" path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..utils import get_logger

logger = get_logger("glassderm.download")


REQUIRED_FILES = (
    "ISIC_2024_Permissive_Training_GroundTruth.csv",
    "ISIC_2024_Permissive_Training_Supplement.csv",
    "ISIC_2024_Permissive_Training_Input",   # directory with images + metadata.csv
)


KNOWN_FALLBACKS: tuple[str, ...] = (
    "/home/chw/222/data/raw/isic2024_official",
    "./data/raw/isic2024_official",
    os.path.expanduser("~/datasets/isic2024_official"),
)


class DatasetNotFound(RuntimeError):
    pass


@dataclass
class DatasetLocator:
    raw_dir: Path
    images_dir: Path
    gt_csv: Path
    supplement_csv: Path
    metadata_csv: Path

    @classmethod
    def from_config(cls, cfg) -> "DatasetLocator":
        raw = Path(cfg.data.raw_dir)
        loc = cls(
            raw_dir=raw,
            images_dir=Path(cfg.data.images_dir),
            gt_csv=Path(cfg.data.gt_csv),
            supplement_csv=Path(cfg.data.supplement_csv),
            metadata_csv=Path(cfg.data.metadata_csv),
        )
        return loc

    def exists(self) -> bool:
        return (
            self.gt_csv.is_file()
            and self.supplement_csv.is_file()
            and self.metadata_csv.is_file()
            and self.images_dir.is_dir()
        )

    def describe(self) -> str:
        return (
            f"raw_dir       = {self.raw_dir}\n"
            f"images_dir    = {self.images_dir}\n"
            f"gt_csv        = {self.gt_csv}\n"
            f"supplement    = {self.supplement_csv}\n"
            f"metadata      = {self.metadata_csv}\n"
        )


def locate_or_download(cfg, allow_download: bool = True) -> DatasetLocator:
    """Return a :class:`DatasetLocator` pointing at a usable ISIC 2024 copy."""
    primary = DatasetLocator.from_config(cfg)
    if primary.exists():
        logger.info("Using ISIC 2024 dataset at %s", primary.raw_dir)
        return primary

    for candidate in KNOWN_FALLBACKS:
        cand = Path(candidate)
        if _candidate_complete(cand):
            logger.warning(
                "Config raw_dir (%s) was incomplete; falling back to %s",
                primary.raw_dir,
                cand,
            )
            return _locator_from_raw(cand)

    if allow_download and _kaggle_available():
        target = primary.raw_dir
        target.mkdir(parents=True, exist_ok=True)
        try:
            _kaggle_download(cfg.data.kaggle.dataset, target)
        except Exception as e:  # pragma: no cover — only runs when credentials exist
            raise DatasetNotFound(
                f"Kaggle download failed: {e}\n\n"
                + _manual_instructions(primary)
            ) from e
        if primary.exists():
            return primary

    raise DatasetNotFound(_manual_instructions(primary))


def _candidate_complete(raw: Path) -> bool:
    return all((raw / f).exists() for f in REQUIRED_FILES) and (
        raw / "ISIC_2024_Permissive_Training_Input" / "metadata.csv"
    ).exists()


def _locator_from_raw(raw: Path) -> DatasetLocator:
    img = raw / "ISIC_2024_Permissive_Training_Input"
    return DatasetLocator(
        raw_dir=raw,
        images_dir=img,
        gt_csv=raw / "ISIC_2024_Permissive_Training_GroundTruth.csv",
        supplement_csv=raw / "ISIC_2024_Permissive_Training_Supplement.csv",
        metadata_csv=img / "metadata.csv",
    )


def _kaggle_available() -> bool:
    if shutil.which("kaggle") is None:
        return False
    token = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return token.is_file()


def _kaggle_download(slug: str, target: Path) -> None:
    logger.info("Attempting `kaggle datasets download %s` into %s", slug, target)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"],
        check=True,
    )


def _manual_instructions(loc: DatasetLocator) -> str:
    return (
        "ISIC 2024 dataset not found.  GlassDerm expects the following layout "
        "(any of these paths):\n\n"
        "  <raw_dir>/ISIC_2024_Permissive_Training_GroundTruth.csv\n"
        "  <raw_dir>/ISIC_2024_Permissive_Training_Supplement.csv\n"
        "  <raw_dir>/ISIC_2024_Permissive_Training_Input/metadata.csv\n"
        "  <raw_dir>/ISIC_2024_Permissive_Training_Input/ISIC_*.jpg\n\n"
        f"Current configuration expected:\n{loc.describe()}\n"
        "Fix by either:\n"
        "  (1) editing configs/default.yaml so `data.raw_dir` points to your copy;\n"
        "  (2) placing the files at one of the fallback paths: "
        f"{list(KNOWN_FALLBACKS)};\n"
        "  (3) installing a Kaggle API token at ~/.kaggle/kaggle.json and\n"
        "      re-running `glassderm prepare-data` (the script will try to\n"
        "      download automatically).\n"
    )
