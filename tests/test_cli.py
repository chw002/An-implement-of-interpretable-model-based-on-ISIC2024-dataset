"""CLI smoke test — the parser exposes only the current three subcommands,
and a direct scale-experiment smoke run fits both transparent pipelines
end-to-end on the tiny synthetic cohort.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_parser_has_current_subcommands():
    from glassderm.cli import build_parser

    p = build_parser()
    # inspect / prepare-data / scale-experiment
    for command in ("inspect", "prepare-data", "scale-experiment"):
        ns = p.parse_args([command] if command != "scale-experiment" else [command])
        assert ns.command == command

    ns = p.parse_args(["scale-experiment", "--skip-cnn", "--only", "transparent_lr"])
    assert ns.skip_cnn is True
    assert ns.only == ["transparent_lr"]


def test_scale_experiment_smoke(tiny_cfg, tmp_path):
    """Run the transparent pipelines through the same driver scale-experiment uses."""
    from glassderm.artefacts import prepare_data
    from glassderm.data import CohortSpec, build_cohort
    from glassderm.data.download import locate_or_download
    from glassderm.scale import (
        ScaleRunConfig,
        build_fixed_splits,
        run_scale_experiment,
    )

    prepare_data(tiny_cfg)

    locator = locate_or_download(tiny_cfg, allow_download=False)
    cohort = build_cohort(
        locator,
        processed_dir=tiny_cfg.data.processed_dir,
        spec=CohortSpec(keep_all_malignant=True, n_benign=None, seed=1337),
    )
    fixed = build_fixed_splits(
        cohort,
        val_fraction=float(tiny_cfg.data.split.val_fraction),
        test_fraction=float(tiny_cfg.data.split.test_fraction),
        seed=int(tiny_cfg.data.split.seed),
    )

    out_root = Path(tmp_path) / "outputs_scale"
    scale_cfg = ScaleRunConfig(
        benign_budget=None,
        pipelines=["transparent_lr", "transparent_tree"],
        skip_cnn=True,
        tag="smoke",
    )
    result = run_scale_experiment(
        tiny_cfg, scale_cfg,
        cohort=cohort,
        fixed_splits=fixed,
        outputs_root=out_root,
    )

    scale_dir = result["output_dir"]
    assert (scale_dir / "tables" / "main_metrics.csv").exists()
    assert (scale_dir / "tables" / "thresholds.csv").exists()
    assert (scale_dir / "README_RESULTS.md").exists()
    assert (scale_dir / "case_studies" / "case_studies.md").exists()
    assert (scale_dir / "correct_prediction_feature_summary.csv").exists()
    md_summary = scale_dir / "correct_prediction_feature_summary.md"
    assert md_summary.exists()
    md_text = md_summary.read_text(encoding="utf-8")
    assert "# Correct-Prediction Feature Summary" in md_text
    assert "Bucket legend" in md_text
    for pipe_name in ("transparent_lr", "transparent_tree"):
        pipe_dir = scale_dir / pipe_name
        assert (pipe_dir / "feature_audit.csv").exists()
        assert (pipe_dir / "feature_audit.json").exists()
        assert (pipe_dir / "predictions_test.csv").exists()
        assert (pipe_dir / "pipeline_meta.json").exists()
    assert (scale_dir / "transparent_lr" / "transparent_lr_readout.json").exists()
    assert (scale_dir / "transparent_lr" / "transparent_lr_readout.txt").exists()
    assert (scale_dir / "transparent_tree" / "transparent_tree_rules.txt").exists()

    # Feature audit must list every model input column as pixel_derived.
    audit = pd.read_csv(scale_dir / "transparent_lr" / "feature_audit.csv")
    model_inputs = audit[audit["is_model_input"].astype(bool)]
    assert len(model_inputs) > 0
    assert (model_inputs["source"] == "pixel_derived").all(), (
        f"Non-pixel columns reached model input: "
        f"{model_inputs[model_inputs['source'] != 'pixel_derived']}"
    )
