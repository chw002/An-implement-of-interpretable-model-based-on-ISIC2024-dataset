"""GlassDerm pipelines, all strictly image-only.

================    ========================    ====================
Pipeline            Classifier / readout         Transparency tag
================    ========================    ====================
multitask_cnn       CNN + MLP                   image_only_baseline
hard_cbm            CNN + 4→1 linear head       interpretable_partial
glassbox_nam        CNN + additive shapes       interpretable_partial
transparent_lr      pixel features + LR         fully_auditable
transparent_tree    pixel features + tree       fully_auditable
================    ========================    ====================
"""
from .base import Pipeline, PipelinePrediction
from .glassbox_nam import GlassBoxNAMPipeline
from .hard_cbm import HardCBMPipeline
from .multitask_cnn import MultiTaskCNNPipeline
from .transparent import TransparentLRPipeline, TransparentTreePipeline


PIPELINE_REGISTRY = {
    "multitask_cnn": MultiTaskCNNPipeline,
    "hard_cbm": HardCBMPipeline,
    "glassbox_nam": GlassBoxNAMPipeline,
    "transparent_lr": TransparentLRPipeline,
    "transparent_tree": TransparentTreePipeline,
}

CNN_PIPELINES = ("multitask_cnn", "hard_cbm", "glassbox_nam")
TRANSPARENT_PIPELINES = ("transparent_lr", "transparent_tree")

TRANSPARENCY_TAGS = {
    "multitask_cnn": "image_only_baseline",
    "hard_cbm": "interpretable_partial",
    "glassbox_nam": "interpretable_partial",
    "transparent_lr": "fully_auditable",
    "transparent_tree": "fully_auditable",
}

__all__ = [
    "CNN_PIPELINES",
    "GlassBoxNAMPipeline",
    "HardCBMPipeline",
    "MultiTaskCNNPipeline",
    "PIPELINE_REGISTRY",
    "Pipeline",
    "PipelinePrediction",
    "TRANSPARENCY_TAGS",
    "TRANSPARENT_PIPELINES",
    "TransparentLRPipeline",
    "TransparentTreePipeline",
]
