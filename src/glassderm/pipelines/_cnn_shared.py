"""Shared CNN building blocks used by the 3 CNN-based pipelines.

Keeping the backbone and concept-predictor head in one place means the three
"neural" pipelines differ *only* in their reasoning head — exactly the axis
we want to ablate.
"""
from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, nn.Module, int]:
    name = name.lower()
    if name == "efficientnet_b0":
        w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.efficientnet_b0(weights=w)
        return net.features, nn.AdaptiveAvgPool2d(1), 1280
    if name == "resnet18":
        w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet18(weights=w)
        features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        return features, nn.AdaptiveAvgPool2d(1), 512
    raise ValueError(f"Unknown backbone {name!r}")


class ConceptHead(nn.Module):
    """Tiny MLP that projects pooled features → N concepts in [0, 1]."""

    def __init__(self, feature_dim: int, n_concepts: int = 4, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden, n_concepts),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(cfg) -> str:
    from ..config import pick_device

    return pick_device(cfg.project.device)
