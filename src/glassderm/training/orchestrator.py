"""Train orchestrator — fit all four pipelines, tune thresholds, save state.

Keeps the top-level CLI thin: the orchestrator is the one place that knows
about the artefact dict every pipeline consumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from ..evaluation.thresholds import select_threshold
from ..pipelines import PIPELINE_REGISTRY, Pipeline
from ..utils import dump_json, get_logger

logger = get_logger("glassderm.train")


class TrainOrchestrator:
    def __init__(self, cfg, artefacts: Mapping[str, object]):
        self.cfg = cfg
        self.artefacts = artefacts
        self.pipelines: Dict[str, Pipeline] = {}

    def run(self, only: Optional[Iterable[str]] = None) -> Dict[str, Pipeline]:
        wanted = set(only or PIPELINE_REGISTRY)
        for name, cls in PIPELINE_REGISTRY.items():
            if name not in wanted:
                continue
            if not self.cfg.pipelines.get(name, {}).get("enabled", True):
                logger.info("Skipping %s (disabled in config)", name)
                continue
            logger.info("============== training pipeline: %s ==============", name)
            pipe = cls(self.cfg, logger)
            pipe.fit(self.artefacts)
            self._tune_threshold(pipe)
            self._save(pipe)
            self.pipelines[name] = pipe
        return self.pipelines

    # ------------------------------------------------------------- internals
    def _tune_threshold(self, pipe: Pipeline) -> None:
        val_pred = pipe.predict("val", self.artefacts)
        chosen = select_threshold(
            probs=val_pred.probs,
            labels=val_pred.labels,
            strategy=self.cfg.evaluate.threshold_strategy,
            fixed=self.cfg.evaluate.fixed_threshold,
        )
        pipe.set_threshold(chosen)
        logger.info("[%s] threshold tuned on val: %.4f", pipe.name, chosen)

    def _save(self, pipe: Pipeline) -> None:
        ckpt_dir = Path(self.cfg.outputs.checkpoints)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".joblib" if pipe.name.startswith("transparent") else ".pth"
        ckpt_path = ckpt_dir / f"{pipe.name}{suffix}"
        pipe.save(ckpt_path)
        logger.info("[%s] checkpoint saved → %s", pipe.name, ckpt_path)

        meta_path = ckpt_dir / f"{pipe.name}_meta.json"
        dump_json(
            {
                "name": pipe.name,
                "transparency": pipe.transparency,
                "threshold": pipe.threshold,
                "checkpoint": str(ckpt_path),
            },
            meta_path,
        )
