"""GlassDerm CLI — thin wrappers around :mod:`glassderm.scale`.

Sub-commands
------------
prepare-data      Locate ISIC 2024, build cohort + patient-disjoint splits, cache
                  the pixel-derived feature parquet.
inspect           Print the resolved config + dataset locator; no side effects.
scale-experiment  The main entry point for the dissertation: run every pipeline
                  across a sequence of training-benign budgets and emit the
                  summary tables + figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .artefacts import prepare_data
from .config import Config, load_config, pick_device
from .pipelines import PIPELINE_REGISTRY
from .scale import SCALE_LABELS, run_all_scales
from .utils import get_logger, set_seed, setup_file_logging

logger = get_logger("glassderm")


def _load_cfg(args) -> Config:
    cfg = load_config(args.config, overrides=args.override or [])
    device = pick_device(cfg.project.device)
    cfg.project.device = device
    set_seed(int(cfg.project.seed))
    setup_file_logging(logger, Path(cfg.outputs.logs) / "glassderm.log")
    logger.info("Using device: %s", device)
    return cfg


# --------------------------------------------------------------- commands ---
def cmd_inspect(args) -> int:
    cfg = _load_cfg(args)
    from .data.download import DatasetLocator

    import yaml

    loc = DatasetLocator.from_config(cfg)
    logger.info("resolved config:\n%s", yaml.safe_dump(dict(cfg), sort_keys=False))
    logger.info("dataset locator:\n%s", loc.describe())
    return 0


def cmd_prepare_data(args) -> int:
    cfg = _load_cfg(args)
    prepare_data(cfg)
    return 0


def cmd_scale_experiment(args) -> int:
    cfg = _load_cfg(args)
    budgets = _parse_budgets(args.benign_budgets)
    outputs_root = Path(args.outputs_root or "outputs_scale")
    run_all_scales(
        cfg,
        budgets=budgets,
        outputs_root=outputs_root,
        skip_cnn=bool(args.skip_cnn),
        only=args.only,
        epochs_override=args.epochs,
    )
    return 0


# ---------------------------------------------------------------- helpers ---
def _parse_budgets(items: Optional[List[str]]):
    if not items:
        return [40000, 80000, 160000, None]
    out = []
    for item in items:
        low = item.strip().lower()
        if low in {"full", "all", "none", "null"}:
            out.append(None)
        else:
            out.append(int(low))
    return out


# ---------------------------------------------------------------- argparse --
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("glassderm")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    p.add_argument(
        "-o",
        "--override",
        action="append",
        help="Dotted overrides, e.g. `-o train.epochs=2`",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp_prep = sub.add_parser("prepare-data", help="Build cohort + split + features cache")
    sp_prep.set_defaults(func=cmd_prepare_data)

    sp_inspect = sub.add_parser("inspect", help="Print resolved config & dataset locator")
    sp_inspect.set_defaults(func=cmd_inspect)

    sp_scale = sub.add_parser(
        "scale-experiment",
        help="Run every pipeline across benign budgets; writes outputs_scale/",
    )
    sp_scale.set_defaults(func=cmd_scale_experiment)
    sp_scale.add_argument(
        "--benign-budgets",
        nargs="+",
        default=None,
        help="e.g. --benign-budgets 40000 80000 160000 full  (default: all four)",
    )
    sp_scale.add_argument(
        "--outputs-root",
        default="outputs_scale",
        help="Root for per-scale output directories (default: outputs_scale)",
    )
    sp_scale.add_argument(
        "--only",
        nargs="+",
        choices=list(PIPELINE_REGISTRY),
        help="Subset of pipelines to fit (default: every pipeline in the registry)",
    )
    sp_scale.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Skip CNN-based pipelines (multitask_cnn, hard_cbm, glassbox_nam)",
    )
    sp_scale.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override cfg.train.epochs for this run",
    )

    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
