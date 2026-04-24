"""Torch Datasets + DataLoaders for the CNN-based image-only pipelines.

The CNN pipelines receive (image, label, pixel-derived ABCD concept proxy)
triples.  The concept proxy is supervised with a target that is itself
computed from pixels (see :mod:`glassderm.data.features`), *never* with
ISIC/TBP metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..utils import get_logger, load_json
from .features import CONCEPT_NAMES
from .split import apply_splits

logger = get_logger("glassderm.datasets")


def load_cohort_splits(
    processed_dir: str | Path,
    cohort_csv: str = "cohort.csv",
    splits_json: str = "splits.json",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    df = pd.read_csv(processed_dir / cohort_csv)
    splits = load_json(processed_dir / splits_json)
    return apply_splits(df, splits)


class ISICImageDataset(Dataset):
    """Pairs (image, label, pixel-derived ABCD concept target, image_id).

    ``concepts_by_id`` is a ``{image_id: (A, B, C, D)}`` dict with values in
    [0, 1] coming from :mod:`glassderm.data.features` — never from vendor
    metadata.  Set to ``None`` to return label-only triples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        concepts_by_id: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.concepts_by_id = concepts_by_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row.image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(int(row.label), dtype=torch.float32)
        if self.concepts_by_id is None:
            concepts = torch.zeros(4, dtype=torch.float32)
        else:
            vals = self.concepts_by_id.get(str(row.image_id))
            if vals is None:
                concepts = torch.full((4,), 0.5, dtype=torch.float32)
            else:
                concepts = torch.tensor(list(vals), dtype=torch.float32)
        return img, label, concepts, row.image_id


def build_transforms(size: int, mean, std, train: bool):
    base = [transforms.Resize((size, size))]
    if train:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(base)


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    image_size: int,
    mean,
    std,
    batch_size: int,
    num_workers: int,
    concepts_by_id: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    seed: int = 1337,
) -> Dict[str, DataLoader]:
    tfm_train = build_transforms(image_size, mean, std, train=True)
    tfm_eval = build_transforms(image_size, mean, std, train=False)

    def _make(df, tfm, shuffle):
        ds = ISICImageDataset(df, transform=tfm, concepts_by_id=concepts_by_id)
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            generator=g if shuffle else None,
        )

    return {
        "train": _make(train_df, tfm_train, shuffle=True),
        "val": _make(val_df, tfm_eval, shuffle=False),
        "test": _make(test_df, tfm_eval, shuffle=False),
    }


def concepts_from_features(
    features_df: pd.DataFrame,
) -> Dict[str, Tuple[float, float, float, float]]:
    cols = list(CONCEPT_NAMES)
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for _, row in features_df.iterrows():
        out[str(row["image_id"])] = tuple(float(row[c]) for c in cols)  # type: ignore[assignment]
    return out
