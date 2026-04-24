"""Strictly auditable, pixel-derived image features.

Every entry in :data:`FEATURE_NAMES` maps to a closed-form formula that a
clinician (or a reviewer with a calculator) can reproduce from the raw image.
**There is no learned representation anywhere in this file** — no CNN, no
latent vector, no embedding — and **no ISIC/TBP metadata is consulted**.

Stages
------
1.  Resize to a fixed 256×256 canvas so formulas are scale-comparable.
2.  Otsu threshold + morphology → binary lesion mask.
3.  Extract geometric / colour / border features from the mask + image.

The final four "ABCD concept proxies" (``concept_A_asymmetry``,
``concept_B_border``, ``concept_C_color``, ``concept_D_diameter``) are
deterministic weighted aggregates of the lower-level features.  They are
**not** the vendor ``tbp_lv_*`` columns — see
:mod:`glassderm.data.audit` for the forbidden-metadata contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import get_logger

logger = get_logger("glassderm.features")


FEATURE_NAMES: Tuple[str, ...] = (
    # geometry / shape
    "geom_area_ratio",
    "geom_perim_ratio",
    "geom_compactness",
    "geom_solidity",
    "geom_convexity_defects",
    "geom_eccentricity",
    "geom_aspect_ratio",
    # asymmetry
    "asym_horizontal",
    "asym_vertical",
    "asym_diagonal",
    "asym_centroid_offset",
    # border
    "border_radial_variance",
    "border_gradient",
    # colour
    "color_hue_std",
    "color_sat_std",
    "color_val_std",
    "color_rgb_std_mean",
    "color_darkness",
    "color_n_regions",
    "color_variegation",
)


CONCEPT_NAMES: Tuple[str, ...] = (
    "concept_A_asymmetry",
    "concept_B_border",
    "concept_C_color",
    "concept_D_diameter",
)


# Canonical dermoscopy colours (RGB thresholds) used in the "number-of-colours"
# feature.  Simple but widely cited.
_COLOR_RULES = [
    ("white",     lambda r, g, b: (r > 200) & (g > 200) & (b > 200)),
    ("red",       lambda r, g, b: (r > 150) & (g < 100) & (b < 100)),
    ("lt_brown",  lambda r, g, b: (r > 150) & (g > 100) & (g < 180) & (b < 120)),
    ("dk_brown",  lambda r, g, b: (r > 80) & (r < 160) & (g > 40) & (g < 120) & (b < 80)),
    ("blue_gray", lambda r, g, b: (b > r) & (b > g) & (b > 80)),
    ("black",     lambda r, g, b: (r < 60) & (g < 60) & (b < 60)),
]


@dataclass
class TransparentFeatureExtractor:
    image_size: int = 256

    def __call__(self, image_path: str | Path) -> Dict[str, float]:
        img = cv2.imread(str(image_path))
        if img is None:
            return self._zero_features(reason="unreadable_image")
        return self.features_from_bgr(img)

    def features_from_bgr(self, img_bgr: np.ndarray) -> Dict[str, float]:
        img = cv2.resize(img_bgr, (self.image_size, self.image_size))
        mask, thresh = self._segment(img)
        if not mask.any():
            return self._zero_features(reason="empty_mask", thresh=thresh)

        feats: Dict[str, float] = {}
        feats.update(self._geometry_and_asymmetry(mask, img.shape))
        feats.update(self._border(mask, img))
        feats.update(self._color(img, mask))
        feats.update(_derive_abcd(feats))
        feats["_otsu_threshold"] = float(thresh)
        return feats

    # ------------------------------------------------------------ segmentation
    def _segment(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_lab <= 1:
            return np.zeros_like(binary), float(thresh)
        best_idx = 0
        best_area = 0
        h, w = binary.shape
        for i in range(1, n_lab):
            x, y, ww, hh, area = stats[i]
            touches_border = (x == 0) or (y == 0) or (x + ww == w) or (y + hh == h)
            if touches_border and area < 0.5 * h * w:
                continue
            if area > best_area:
                best_area = area
                best_idx = i
        if best_idx == 0:
            best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(binary)
        out[lab == best_idx] = 255
        return out, float(thresh)

    # ------------------------------------------------------------ geometry
    def _geometry_and_asymmetry(
        self, mask: np.ndarray, shape: Tuple[int, int, int]
    ) -> Dict[str, float]:
        H, W = shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _zero_geom()
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        perim = float(cv2.arcLength(cnt, True))
        if area < 50.0 or perim < 10.0:
            return _zero_geom()

        M = cv2.moments(cnt)
        cx = M["m10"] / (M["m00"] + 1e-8)
        cy = M["m01"] / (M["m00"] + 1e-8)

        compactness = (perim ** 2) / (4.0 * np.pi * area)
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1e-8)

        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        try:
            defects = (
                cv2.convexityDefects(cnt, hull_idx)
                if hull_idx is not None and len(hull_idx) > 2
                else None
            )
        except cv2.error:
            defects = None
        n_defects = 0 if defects is None else int(np.sum(defects[:, 0, 3] > 2 * 256))

        if len(cnt) >= 5:
            (_, _), (a_axis, b_axis), _ = cv2.fitEllipse(cnt)
            maj = max(a_axis, b_axis)
            mn = min(a_axis, b_axis)
            ecc = float(np.sqrt(max(0.0, 1.0 - (mn ** 2) / max(maj, 1e-8) ** 2)))
        else:
            ecc = 0.0
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = max(ww, hh) / max(min(ww, hh), 1)

        asym_lr = _mirror_disagreement(mask, axis="vertical", center=(cx, cy))
        asym_tb = _mirror_disagreement(mask, axis="horizontal", center=(cx, cy))
        asym_d1 = _mirror_disagreement(mask, axis="diag1", center=(cx, cy))
        asym_d2 = _mirror_disagreement(mask, axis="diag2", center=(cx, cy))
        asym_diag = 0.5 * (asym_d1 + asym_d2)

        diag = float(np.sqrt(H * H + W * W))
        offset = float(np.sqrt((cx - W / 2) ** 2 + (cy - H / 2) ** 2) / diag)

        return {
            "geom_area_ratio": area / (H * W),
            "geom_perim_ratio": perim / diag,
            "geom_compactness": float(np.clip(compactness, 1.0, 10.0)) - 1.0,
            "geom_solidity": float(np.clip(solidity, 0.0, 1.0)),
            "geom_convexity_defects": float(min(n_defects, 25) / 25.0),
            "geom_eccentricity": float(np.clip(ecc, 0.0, 1.0)),
            "geom_aspect_ratio": float(min(aspect, 10.0) / 10.0),
            "asym_horizontal": asym_lr,
            "asym_vertical": asym_tb,
            "asym_diagonal": asym_diag,
            "asym_centroid_offset": offset,
        }

    # ------------------------------------------------------------ border
    def _border(self, mask: np.ndarray, img_bgr: np.ndarray) -> Dict[str, float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"border_radial_variance": 0.0, "border_gradient": 0.0}
        cnt = max(contours, key=cv2.contourArea)

        M = cv2.moments(cnt)
        cx = M["m10"] / (M["m00"] + 1e-8)
        cy = M["m01"] / (M["m00"] + 1e-8)
        pts = cnt.reshape(-1, 2).astype(float)
        d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        radial = float(np.std(d) / max(np.mean(d), 1e-6))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        border_pixels = mag[edge > 0]
        border_grad = float(border_pixels.mean() / 255.0) if border_pixels.size else 0.0

        return {
            "border_radial_variance": float(np.clip(radial, 0.0, 1.0)),
            "border_gradient": float(np.clip(border_grad, 0.0, 1.0)),
        }

    # ------------------------------------------------------------ colour
    def _color(self, img_bgr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        pixels = img_bgr[mask > 0]
        if len(pixels) < 50:
            return {
                "color_hue_std": 0.0,
                "color_sat_std": 0.0,
                "color_val_std": 0.0,
                "color_rgb_std_mean": 0.0,
                "color_darkness": 0.0,
                "color_n_regions": 0.0,
                "color_variegation": 0.0,
            }

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[mask > 0]
        h_std = float(np.std(hsv[:, 0].astype(float))) / 180.0
        s_std = float(np.std(hsv[:, 1].astype(float))) / 255.0
        v_std = float(np.std(hsv[:, 2].astype(float))) / 255.0
        v_mean = float(np.mean(hsv[:, 2].astype(float))) / 255.0

        r = pixels[:, 2]
        g = pixels[:, 1]
        b = pixels[:, 0]

        color_proportions = []
        for _, rule in _COLOR_RULES:
            color_proportions.append(float(np.mean(rule(r, g, b))))
        color_proportions = np.asarray(color_proportions)

        present = (color_proportions > 0.02).sum()
        norm = color_proportions / max(color_proportions.sum(), 1e-8)
        entropy = float(-np.sum(norm * np.log(norm + 1e-8)) / np.log(len(_COLOR_RULES)))

        rgb_std_mean = float((np.std(r) + np.std(g) + np.std(b)) / (3.0 * 255.0))

        return {
            "color_hue_std": float(np.clip(h_std, 0.0, 1.0)),
            "color_sat_std": float(np.clip(s_std, 0.0, 1.0)),
            "color_val_std": float(np.clip(v_std, 0.0, 1.0)),
            "color_rgb_std_mean": float(np.clip(rgb_std_mean, 0.0, 1.0)),
            "color_darkness": float(np.clip(1.0 - v_mean, 0.0, 1.0)),
            "color_n_regions": float(present) / len(_COLOR_RULES),
            "color_variegation": entropy,
        }

    def _zero_features(self, reason: str, thresh: float | None = None) -> Dict[str, float]:
        feats = {name: 0.5 for name in FEATURE_NAMES}
        feats.update({name: 0.5 for name in CONCEPT_NAMES})
        feats["_extraction_error"] = reason  # type: ignore[assignment]
        if thresh is not None:
            feats["_otsu_threshold"] = float(thresh)
        return feats


# ---------------------------------------------------------------- helpers ---
def _zero_geom() -> Dict[str, float]:
    return {
        "geom_area_ratio": 0.0,
        "geom_perim_ratio": 0.0,
        "geom_compactness": 0.0,
        "geom_solidity": 0.0,
        "geom_convexity_defects": 0.0,
        "geom_eccentricity": 0.0,
        "geom_aspect_ratio": 0.0,
        "asym_horizontal": 0.0,
        "asym_vertical": 0.0,
        "asym_diagonal": 0.0,
        "asym_centroid_offset": 0.0,
    }


def _mirror_disagreement(mask: np.ndarray, axis: str, center) -> float:
    cx, cy = center
    H, W = mask.shape
    if axis == "vertical":
        left = mask[:, : int(cx)]
        right = cv2.flip(mask[:, int(cx):], 1)
        return _compare_halves(left, right)
    if axis == "horizontal":
        top = mask[: int(cy), :]
        bot = cv2.flip(mask[int(cy):, :], 0)
        return _compare_halves(top, bot)
    angle = 45 if axis == "diag1" else 135
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    rot = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST)
    left = rot[:, : W // 2]
    right = cv2.flip(rot[:, W // 2 :], 1)
    return _compare_halves(left, right)


def _compare_halves(a: np.ndarray, b: np.ndarray) -> float:
    m = min(a.shape[1], b.shape[1])
    n = min(a.shape[0], b.shape[0])
    if m <= 1 or n <= 1:
        return 0.5
    a = a[-n:, -m:].astype(float)
    b = b[-n:, -m:].astype(float)
    total = (a.sum() + b.sum()) / 255.0 + 1e-8
    return float(np.sum(np.abs(a - b)) / 255.0 / total)


def _derive_abcd(feats: Dict[str, float]) -> Dict[str, float]:
    """Pixel-derived ABCD concept proxies.

    These are weighted averages of already-computed OpenCV features.  They are
    deterministic, auditable and independent of ISIC/TBP metadata.
    """
    A = 0.40 * feats["asym_horizontal"] + 0.40 * feats["asym_vertical"] + 0.20 * feats["asym_diagonal"]
    B = (
        0.40 * feats["border_radial_variance"]
        + 0.30 * feats["border_gradient"]
        + 0.20 * feats["geom_compactness"]
        + 0.10 * feats["geom_convexity_defects"]
    )
    C = (
        0.30 * feats["color_variegation"]
        + 0.20 * feats["color_hue_std"]
        + 0.15 * feats["color_sat_std"]
        + 0.15 * feats["color_val_std"]
        + 0.10 * feats["color_rgb_std_mean"]
        + 0.10 * feats["color_n_regions"]
    )
    D = 0.60 * feats["geom_area_ratio"] + 0.30 * feats["geom_aspect_ratio"] + 0.10 * feats["geom_eccentricity"]
    return {
        "concept_A_asymmetry": float(np.clip(A, 0.0, 1.0)),
        "concept_B_border": float(np.clip(B, 0.0, 1.0)),
        "concept_C_color": float(np.clip(C, 0.0, 1.0)),
        "concept_D_diameter": float(np.clip(D, 0.0, 1.0)),
    }


# ---------------------------------------------------------------- batch API --
def extract_features_for_cohort(
    cohort: pd.DataFrame,
    out_parquet: str | Path,
    *,
    recompute: bool = False,
    image_size: int = 256,
) -> pd.DataFrame:
    """Extract pixel-derived features for every row in ``cohort``.

    The output table contains: image_id, patient_id, label, the 20 raw
    features, and the 4 pixel-derived ABCD concept proxies.  **Nothing from
    ISIC/TBP metadata is allowed to ride along.**
    """
    out_parquet = Path(out_parquet)
    csv_fallback = out_parquet.with_suffix(".csv")
    if (out_parquet.exists() or csv_fallback.exists()) and not recompute:
        src = out_parquet if out_parquet.exists() else csv_fallback
        cached = _read_features(src)
        if _covers(cached, cohort):
            logger.info("Using cached features at %s", src)
            return _sanitise(cached, cohort)
        logger.info(
            "Cached features at %s do not cover every cohort row — recomputing",
            src,
        )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    extractor = TransparentFeatureExtractor(image_size=image_size)

    rows = []
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="features"):
        feats = extractor(row.image_path)
        rec = {
            "image_id": row.image_id,
            "patient_id": row.patient_id,
            "label": int(row.label),
        }
        rec.update({k: v for k, v in feats.items() if not k.startswith("_")})
        rows.append(rec)

    df = pd.DataFrame(rows)
    _write_features(df, out_parquet)
    logger.info("Saved %d feature rows → %s", len(df), out_parquet)
    return df


def _covers(cached: pd.DataFrame, cohort: pd.DataFrame) -> bool:
    have = set(cached["image_id"].astype(str).tolist())
    want = set(cohort["image_id"].astype(str).tolist())
    cols_needed = set(FEATURE_NAMES) | set(CONCEPT_NAMES) | {"image_id", "patient_id", "label"}
    return want.issubset(have) and cols_needed.issubset(cached.columns)


def _sanitise(cached: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    cohort_ids = cohort["image_id"].astype(str)
    out = cached[cached["image_id"].astype(str).isin(cohort_ids)].copy()
    keep = ["image_id", "patient_id", "label"] + list(FEATURE_NAMES) + list(CONCEPT_NAMES)
    out = out[[c for c in keep if c in out.columns]].reset_index(drop=True)
    return out


def _read_features(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    alt = path.with_suffix(".csv")
    if alt.exists():
        return pd.read_csv(alt)
    return pd.read_csv(path)


def _write_features(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.warning(
            "Parquet backend unavailable; wrote CSV fallback at %s", csv_path
        )
