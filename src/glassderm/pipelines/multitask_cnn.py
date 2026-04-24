"""MultiTaskCNN — image-only baseline.

Input is the raw image.  The auxiliary ABCD head is trained against
pixel-derived concept proxies from :mod:`glassderm.data.features`.  No ISIC /
TBP metadata is consulted at any point.

The final decision is still produced by an MLP on top of the CNN features,
so this pipeline is labelled as an *image-only baseline* — it is **not**
claimed to be fully interpretable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .._training_utils import predict_loader, train_loop
from .base import Pipeline, PipelinePrediction
from ._cnn_shared import build_backbone, resolve_device


class _MultiTaskModel(nn.Module):
    def __init__(self, backbone: str, pretrained: bool):
        super().__init__()
        self.features, self.pool, dim = build_backbone(backbone, pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.abcd_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        feat = self.pool(self.features(x)).flatten(1)
        return self.classifier(feat), self.abcd_head(feat)


class MultiTaskCNNPipeline(Pipeline):
    name = "multitask_cnn"
    transparency = "image_only_black_box_head"
    transparency_tag = "image_only_baseline"

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.device = resolve_device(cfg)
        self.model = _MultiTaskModel(cfg.train.backbone, cfg.train.pretrained).to(self.device)
        self.feature_manifest = ["image_pixel_tensor"]   # image input, no tabular features

    def fit(self, artefacts: Mapping[str, Any]) -> None:
        loaders = artefacts["cnn_loaders"]
        train_loop(
            model=self.model,
            loaders=loaders,
            device=self.device,
            cfg=self.cfg,
            logger=self.logger,
            label_loss_weight=self.cfg.train.label_loss_weight,
            concept_loss_weight=self.cfg.train.concept_loss_weight,
            concept_sup_weight=1.0,
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
        prob = float(row["prob"])
        verdict = "MALIGNANT" if prob >= self.threshold else "BENIGN"
        concepts = [float(row.get(f"concept_{k}", float("nan"))) for k in "ABCD"]
        text = (
            "MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced\n"
            f"P(malignant) = {prob:.3f}; with validation-tuned threshold τ={self.threshold:.3f}\n"
            f"this is classified as {verdict}.  The auxiliary ABCD head reported\n"
            f"A={concepts[0]:.3f}  B={concepts[1]:.3f}  C={concepts[2]:.3f}  D={concepts[3]:.3f}.\n"
            "These concepts are associated image-derived observations; they did not\n"
            "participate in the final decision (the decision is produced by the MLP head)."
        )
        return {
            "pipeline": self.name,
            "transparency": self.transparency,
            "prob": prob,
            "threshold": self.threshold,
            "verdict": verdict,
            "concepts_abcd": concepts,
            "text": text,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "threshold": self.threshold}, path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.threshold = float(state.get("threshold", 0.5))
