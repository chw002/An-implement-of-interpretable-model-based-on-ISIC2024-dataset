"""HardCBM — image-only concept bottleneck with a single linear readout.

Input is the raw image.  A CNN backbone predicts four concept scalars
supervised with pixel-derived ABCD proxies from
:mod:`glassderm.data.features`.  The *only* reasoning layer is a 4→1 linear
head, so the final decision is ``logit = w_A·A + w_B·B + w_C·C + w_D·D + bias``.

Perception (the CNN) remains opaque, so this pipeline is labelled
*partially interpretable*.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .._training_utils import predict_loader, train_loop
from .base import Pipeline, PipelinePrediction
from ._cnn_shared import ConceptHead, build_backbone, resolve_device


class _HardCBMModel(nn.Module):
    def __init__(self, backbone: str, pretrained: bool):
        super().__init__()
        self.features, self.pool, dim = build_backbone(backbone, pretrained)
        self.concept_head = ConceptHead(dim, n_concepts=4)
        self.label_head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor):
        feat = self.pool(self.features(x)).flatten(1)
        concepts = self.concept_head(feat)
        logit = self.label_head(concepts)
        return logit, concepts


class HardCBMPipeline(Pipeline):
    name = "hard_cbm"
    transparency = "image_only_linear_readout_over_cnn_concepts"
    transparency_tag = "interpretable_partial"

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.device = resolve_device(cfg)
        self.model = _HardCBMModel(cfg.train.backbone, cfg.train.pretrained).to(self.device)
        self._concept_sup_weight = float(cfg.pipelines.hard_cbm.get("concept_sup_weight", 2.0))
        # model-input feature manifest — these concepts are CNN outputs trained
        # against pixel-derived targets, registered as pixel_derived in the audit.
        self.feature_manifest = [
            "cnn_concept_A_asymmetry",
            "cnn_concept_B_border",
            "cnn_concept_C_color",
            "cnn_concept_D_diameter",
        ]

    def fit(self, artefacts: Mapping[str, Any]) -> None:
        train_loop(
            model=self.model,
            loaders=artefacts["cnn_loaders"],
            device=self.device,
            cfg=self.cfg,
            logger=self.logger,
            concept_sup_weight=self._concept_sup_weight,
            tag=self.name,
        )

    def predict(self, split: str, artefacts: Mapping[str, Any]) -> PipelinePrediction:
        loader: DataLoader = artefacts["cnn_loaders"][split]
        probs, labels, concepts, image_ids = predict_loader(self.model, loader, self.device)
        return PipelinePrediction(
            image_ids=image_ids,
            labels=labels,
            probs=probs,
            preds=self._apply_threshold(probs),
            concepts=concepts,
            threshold=self.threshold,
        )

    def explain(self, row: Mapping[str, Any], artefacts: Mapping[str, Any]) -> dict:
        w, bias = self._weights_bias()
        concepts = np.array([float(row[f"concept_{k}"]) for k in "ABCD"])
        contributions = w * concepts
        logit = float(contributions.sum() + bias)
        prob = 1.0 / (1.0 + math.exp(-logit))
        verdict = "MALIGNANT" if prob >= self.threshold else "BENIGN"

        letters = ["A(Asymmetry)", "B(Border)", "C(Color)", "D(Diameter)"]
        lines = [
            "HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.",
            f"  Validation-tuned threshold τ = {self.threshold:.3f}",
            "  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:",
        ]
        for i, n in enumerate(letters):
            lines.append(
                f"    {n:<14}  c={concepts[i]:+.3f}  w={w[i]:+.3f}  "
                f"contribution={contributions[i]:+.3f}"
            )
        lines.append(f"  + bias = {bias:+.3f}")
        lines.append(
            f"  = logit {logit:+.3f}  →  σ(logit) = P(malignant) = {prob:.3f}  →  {verdict}"
        )
        return {
            "pipeline": self.name,
            "transparency": self.transparency,
            "threshold": self.threshold,
            "prob": prob,
            "verdict": verdict,
            "concepts_abcd": concepts.tolist(),
            "weights": w.tolist(),
            "bias": bias,
            "contributions": contributions.tolist(),
            "logit": logit,
            "text": "\n".join(lines),
        }

    def _weights_bias(self):
        w = self.model.label_head.weight.detach().cpu().numpy().squeeze().astype(float)
        b = float(self.model.label_head.bias.detach().cpu().item())
        return w, b

    def weights_readout(self) -> dict:
        w, b = self._weights_bias()
        return {
            "w_A": float(w[0]),
            "w_B": float(w[1]),
            "w_C": float(w[2]),
            "w_D": float(w[3]),
            "bias": b,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "threshold": self.threshold,
                "weights_readout": self.weights_readout(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.threshold = float(state.get("threshold", 0.5))
