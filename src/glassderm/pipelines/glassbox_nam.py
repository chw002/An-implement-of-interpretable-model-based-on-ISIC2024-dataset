"""GlassBoxNAM — image-only Neural Additive Model.

``logit = f_A(A) + f_B(B) + f_C(C) + f_D(D) + bias``.  The four concept
inputs are CNN predictions supervised with pixel-derived ABCD proxies.  Each
``f_i`` is a tiny one-input MLP, so every contribution can be read as a 1-D
shape curve.  Perception is still opaque → *partially interpretable*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .._training_utils import predict_loader, train_loop
from .base import Pipeline, PipelinePrediction
from ._cnn_shared import ConceptHead, build_backbone, resolve_device


class _ShapeFunction(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GlassBoxNAMModel(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, hidden: int):
        super().__init__()
        self.features, self.pool, dim = build_backbone(backbone, pretrained)
        self.concept_head = ConceptHead(dim, n_concepts=4)
        self.shape_fns = nn.ModuleList([_ShapeFunction(hidden) for _ in range(4)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        feat = self.pool(self.features(x)).flatten(1)
        concepts = self.concept_head(feat)
        logits = sum(self.shape_fns[i](concepts[:, i : i + 1]) for i in range(4))
        return logits + self.bias, concepts


class GlassBoxNAMPipeline(Pipeline):
    name = "glassbox_nam"
    transparency = "image_only_additive_readout_over_cnn_concepts"
    transparency_tag = "interpretable_partial"

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.device = resolve_device(cfg)
        hidden = int(cfg.pipelines.glassbox_nam.get("hidden", 64))
        self.model = _GlassBoxNAMModel(cfg.train.backbone, cfg.train.pretrained, hidden=hidden).to(self.device)
        self._concept_sup_weight = float(cfg.pipelines.glassbox_nam.get("concept_sup_weight", 2.0))
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
        concepts = np.array([float(row[f"concept_{k}"]) for k in "ABCD"])
        contribs = self._contributions(concepts)
        bias = float(self.model.bias.detach().cpu().item())
        logit = float(contribs.sum() + bias)
        prob = 1.0 / (1.0 + float(np.exp(-logit)))
        verdict = "MALIGNANT" if prob >= self.threshold else "BENIGN"

        letters = ["A(Asymmetry)", "B(Border)", "C(Color)", "D(Diameter)"]
        lines = [
            "GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.",
            f"  Validation-tuned threshold τ = {self.threshold:.3f}",
        ]
        for i, n in enumerate(letters):
            lines.append(f"    f_{'ABCD'[i]}({concepts[i]:.3f}) = {contribs[i]:+.3f}   ({n})")
        lines.append(f"  + bias = {bias:+.3f}")
        lines.append(f"  = logit {logit:+.3f}  →  P(malignant) = {prob:.3f}  →  {verdict}")
        return {
            "pipeline": self.name,
            "transparency": self.transparency,
            "threshold": self.threshold,
            "prob": prob,
            "verdict": verdict,
            "concepts_abcd": concepts.tolist(),
            "contributions": contribs.tolist(),
            "bias": bias,
            "logit": logit,
            "text": "\n".join(lines),
        }

    def shape_values(self, n_points: int) -> Dict[str, np.ndarray]:
        x = torch.linspace(0, 1, n_points, device=self.device).unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            out = {
                f"f_{c}": self.model.shape_fns[i](x).cpu().numpy().flatten()
                for i, c in enumerate("ABCD")
            }
        out["x"] = x.cpu().numpy().flatten()
        out["bias"] = float(self.model.bias.detach().cpu().item())
        return out

    def _contributions(self, concepts: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(concepts, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return np.array([
                float(self.model.shape_fns[i](x[i : i + 1].view(1, 1)).item())
                for i in range(4)
            ])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "threshold": self.threshold},
            path,
        )

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.threshold = float(state.get("threshold", 0.5))
