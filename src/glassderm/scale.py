"""Scale-experiment driver.

We fix a patient-disjoint val/test split **once** (the "full" split) and then,
for each training-benign budget (40 000 / 80 000 / 160 000 / full), we
re-sample the train pool keeping every malignant.  This lets us study
"does more benign data help an image-only pipeline?" without contaminating
val or test.

Outputs land at ``outputs_scale/benign_{size}/{pipeline}/``.
"""
from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from .analysis.case_studies import build_case_studies
from .analysis.correct_features import write_correct_prediction_summary
from .analysis.plots import (
    plot_confusion_matrices,
    plot_correct_prediction_features,
    plot_dataset_scale_distribution,
    plot_feature_importance_or_shape_functions,
    plot_metrics_vs_data_size,
    plot_pipeline_comparison_full,
    plot_transparent_explanation_example,
    plot_transparent_lr_vs_tree_tradeoff,
)
from .analysis.reports import (
    write_hard_cbm_report,
    write_readme_results,
    write_transparent_lr_readouts,
    write_transparent_tree_readouts,
)
from .artefacts import build_artefacts
from .config import Config
from .data import MetadataLeakError, audit_feature_columns
from .data.features import FEATURE_NAMES
from .evaluation.metrics import compute_metrics
from .evaluation.thresholds import select_threshold
from .pipelines import (
    CNN_PIPELINES,
    PIPELINE_REGISTRY,
    TRANSPARENCY_TAGS,
    TRANSPARENT_PIPELINES,
    Pipeline,
)
from .utils import dump_json, ensure_dir, get_logger, load_json

logger = get_logger("glassderm.scale")

SCALE_LABELS = {
    40000: "benign_40000",
    80000: "benign_80000",
    160000: "benign_160000",
    None: "benign_full",
}


@dataclass
class ScaleRunConfig:
    benign_budget: Optional[int]             # None → full
    pipelines: List[str] = field(default_factory=lambda: list(PIPELINE_REGISTRY))
    skip_cnn: bool = False
    epochs: Optional[int] = None             # override cfg.train.epochs
    tag: Optional[str] = None

    @property
    def label(self) -> str:
        if self.tag:
            return self.tag
        if self.benign_budget in SCALE_LABELS:
            return SCALE_LABELS[self.benign_budget]
        return (
            "benign_full"
            if self.benign_budget is None
            else f"benign_{int(self.benign_budget)}"
        )


# ------------------------------------------------------------------- split helpers
def build_fixed_splits(
    cohort: pd.DataFrame,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, list]:
    """Patient-disjoint split with every patient in the cohort assigned once."""
    rng = np.random.default_rng(seed)
    patients = cohort["patient_id"].astype(str).unique().tolist()
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test : n_test + n_val])
    train_p = set(patients[n_test + n_val :])

    def ids(subset):
        sub = cohort[cohort.patient_id.astype(str).isin(subset)]
        return sub["image_id"].tolist()

    return {
        "train": ids(train_p),
        "val": ids(val_p),
        "test": ids(test_p),
    }


def subsample_train_benign(
    cohort: pd.DataFrame,
    train_ids: List[str],
    benign_budget: Optional[int],
    seed: int,
) -> List[str]:
    train = cohort[cohort["image_id"].isin(train_ids)]
    mal = train[train.label == 1]
    ben = train[train.label == 0]
    if benign_budget is None or benign_budget >= len(ben):
        keep = ben
    else:
        keep = ben.sample(n=int(benign_budget), random_state=seed)
    subset = pd.concat([mal, keep], ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=seed)
    return subset["image_id"].tolist()


# ------------------------------------------------------------------- pipeline exec
def _tune_and_save(pipe: Pipeline, artefacts: Mapping[str, Any], out_dir: Path, cfg: Config) -> float:
    ensure_dir(out_dir)
    val_pred = pipe.predict("val", artefacts)
    threshold = select_threshold(
        probs=val_pred.probs,
        labels=val_pred.labels,
        strategy=cfg.evaluate.threshold_strategy,
        fixed=cfg.evaluate.fixed_threshold,
    )
    pipe.set_threshold(threshold)
    suffix = ".joblib" if pipe.name.startswith("transparent") else ".pth"
    pipe.save(out_dir / f"model{suffix}")
    dump_json(
        {
            "name": pipe.name,
            "transparency": pipe.transparency,
            "transparency_tag": pipe.transparency_tag,
            "threshold": pipe.threshold,
            "feature_manifest": list(pipe.feature_manifest),
        },
        out_dir / "pipeline_meta.json",
    )
    return threshold


def _evaluate_pipeline(
    pipe: Pipeline, artefacts: Mapping[str, Any], out_dir: Path
) -> Dict[str, Any]:
    pred = pipe.predict("test", artefacts)
    metrics = compute_metrics(pred.labels, pred.probs, pred.threshold)
    metrics["pipeline"] = pipe.name
    metrics["transparency_tag"] = pipe.transparency_tag
    pred.to_frame().to_csv(out_dir / "predictions_test.csv", index=False)
    return {"metrics": metrics, "prediction": pred}


def _verify_audit(pipe: Pipeline, artefacts: Mapping[str, Any], out_dir: Path) -> None:
    if pipe.name in CNN_PIPELINES:
        # CNN input is image pixels; the manifest records that.  We generate a
        # tiny feature_audit that documents the whole pipeline so reviewers can
        # see there is nothing tabular being fed in.
        classification = {col: "pixel_derived" for col in pipe.feature_manifest}
        rows = pd.DataFrame([
            {
                "pipeline": pipe.name,
                "feature_set": "image_only",
                "column": col,
                "source": "pixel_derived",
                "is_model_input": True,
            }
            for col in pipe.feature_manifest
        ])
        ensure_dir(out_dir)
        rows.to_csv(out_dir / "feature_audit.csv", index=False)
        dump_json(
            {
                "pipeline": pipe.name,
                "feature_set": "image_only",
                "model_input_columns": list(pipe.feature_manifest),
                "classification": classification,
            },
            out_dir / "feature_audit.json",
        )
        return
    train_df = artefacts["transparent_features"]["train"]
    audit = audit_feature_columns(
        pipeline=pipe.name,
        feature_set="image_only",
        all_columns=list(train_df.columns),
        model_input_columns=list(pipe.feature_manifest),
    )
    audit.dump(out_dir)


# ------------------------------------------------------------------- main entry
def run_scale_experiment(
    cfg: Config,
    scale_cfg: ScaleRunConfig,
    *,
    cohort: pd.DataFrame,
    fixed_splits: Dict[str, list],
    outputs_root: Path,
) -> Dict[str, Any]:
    scale_out = ensure_dir(outputs_root / scale_cfg.label)
    run_cfg = copy.deepcopy(cfg)
    if scale_cfg.epochs is not None:
        run_cfg.set_path("train.epochs", scale_cfg.epochs)
    run_cfg.set_path("outputs.root", str(scale_out))
    run_cfg.set_path("outputs.checkpoints", str(scale_out / "checkpoints"))
    run_cfg.set_path("outputs.tables", str(scale_out / "tables"))
    run_cfg.set_path("outputs.figures", str(scale_out / "figures"))
    run_cfg.set_path("outputs.reports", str(scale_out / "reports"))
    run_cfg.set_path("outputs.case_studies", str(scale_out / "case_studies"))
    run_cfg.set_path("outputs.logs", str(scale_out / "logs"))

    # Build the scale's splits by subsampling training benigns only.
    train_ids = subsample_train_benign(
        cohort,
        fixed_splits["train"],
        scale_cfg.benign_budget,
        seed=int(cfg.project.seed),
    )
    splits = {
        "train": train_ids,
        "val": list(fixed_splits["val"]),
        "test": list(fixed_splits["test"]),
    }

    # Persist provenance early so we can debug post-mortem even if training fails.
    dump_json(
        {
            "scale_label": scale_cfg.label,
            "benign_budget": scale_cfg.benign_budget,
            "pipelines_requested": list(scale_cfg.pipelines),
            "skip_cnn": bool(scale_cfg.skip_cnn),
            "epochs_override": scale_cfg.epochs,
            "counts": {
                split: {
                    "images": len(ids),
                    "malignant": int(
                        cohort[cohort.image_id.isin(ids)].label.sum()
                    ),
                    "benign": int(
                        (cohort[cohort.image_id.isin(ids)].label == 0).sum()
                    ),
                    "patients": int(
                        cohort[cohort.image_id.isin(ids)].patient_id.nunique()
                    ),
                }
                for split, ids in splits.items()
            },
        },
        scale_out / "cohort_provenance.json",
    )
    dump_json(splits, scale_out / "splits.json")
    with open(scale_out / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(dict(run_cfg), f, sort_keys=False)

    # Which pipelines actually run?
    to_run = [
        p for p in scale_cfg.pipelines
        if not (scale_cfg.skip_cnn and p in CNN_PIPELINES)
    ]
    need_images = any(p in CNN_PIPELINES for p in to_run)

    artefacts = build_artefacts(
        run_cfg,
        need_images=need_images,
        cohort_override=cohort,
        splits_override=splits,
    )

    per_pipeline: Dict[str, Dict[str, Any]] = {}
    thresholds_rows = []
    metrics_rows = []
    for name in to_run:
        logger.info("\n=== scale=%s  pipeline=%s ===", scale_cfg.label, name)
        pipe_out = ensure_dir(scale_out / name)
        cls = PIPELINE_REGISTRY[name]
        pipe = cls(run_cfg, logger)

        # ---- fit + audit ------------------------------------------------
        try:
            pipe.fit(artefacts)
        except MetadataLeakError as e:
            logger.error("[%s] aborted due to metadata leak: %s", name, e)
            (pipe_out / "FAILED.txt").write_text(
                f"Training aborted: feature audit refused the input.\n\n{e}"
            )
            continue

        _verify_audit(pipe, artefacts, pipe_out)
        _tune_and_save(pipe, artefacts, pipe_out, run_cfg)
        result = _evaluate_pipeline(pipe, artefacts, pipe_out)
        metrics = result["metrics"]
        pred = result["prediction"]
        per_pipeline[name] = {"pipe": pipe, "prediction": pred, "metrics": metrics}

        thresholds_rows.append(
            {
                "pipeline": name,
                "transparency_tag": TRANSPARENCY_TAGS.get(name, pipe.transparency_tag),
                "threshold": float(pipe.threshold),
                "strategy": str(run_cfg.evaluate.threshold_strategy),
            }
        )
        metrics_rows.append(
            {
                "pipeline": name,
                "transparency_tag": TRANSPARENCY_TAGS.get(name, pipe.transparency_tag),
                "auc": metrics["auc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "threshold": metrics["threshold"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )

        # ---- per-pipeline reports ---------------------------------------
        if name == "transparent_lr":
            write_transparent_lr_readouts(pipe, artefacts, pred, pipe_out, cohort)
        elif name == "transparent_tree":
            write_transparent_tree_readouts(pipe, artefacts, pred, pipe_out, cohort)
        elif name == "hard_cbm":
            write_hard_cbm_report(pipe, pipe_out / "hard_cbm_readout.txt")
        dump_json(metrics, pipe_out / "test_metrics.json")

    # ---- scale-level tables -------------------------------------------------
    tables_dir = ensure_dir(scale_out / "tables")
    thresholds_df = pd.DataFrame(thresholds_rows)
    thresholds_df.to_csv(tables_dir / "thresholds.csv", index=False)
    main_metrics_df = pd.DataFrame(metrics_rows)
    main_metrics_df.to_csv(tables_dir / "main_metrics.csv", index=False)

    # ---- case studies + correct-prediction summary + figures ---------------
    figures_dir = ensure_dir(scale_out / "figures")
    if per_pipeline:
        build_case_studies(
            per_pipeline,
            artefacts,
            n_per_bucket=int(run_cfg.analysis.n_case_studies_per_bucket),
            out_dir=scale_out / "case_studies",
            cohort=cohort,
            seed=int(cfg.project.seed),
        )
        plot_confusion_matrices(
            {name: pack["prediction"] for name, pack in per_pipeline.items()},
            thresholds={name: pack["pipe"].threshold for name, pack in per_pipeline.items()},
            out_path=figures_dir / "fig_confusion_matrices.png",
        )
        plot_feature_importance_or_shape_functions(
            per_pipeline,
            out_path=figures_dir / "fig_feature_importance_or_shape_functions.png",
        )
        plot_transparent_explanation_example(
            per_pipeline,
            artefacts,
            cohort=cohort,
            out_path=figures_dir / "fig_transparent_explanation_example.png",
        )
        write_correct_prediction_summary(
            per_pipeline,
            artefacts,
            out_dir=scale_out,
        )
        plot_correct_prediction_features(
            scale_out / "correct_prediction_feature_summary.csv",
            out_path=figures_dir / "fig_correct_prediction_features.png",
        )

    write_readme_results(scale_out, scale_cfg, main_metrics_df)
    return {
        "scale": scale_cfg.label,
        "output_dir": scale_out,
        "metrics": main_metrics_df,
        "thresholds": thresholds_df,
    }


# ------------------------------------------------------------- top-level entry
def run_all_scales(
    cfg: Config,
    *,
    budgets: Iterable[Optional[int]],
    outputs_root: Path,
    skip_cnn: bool = False,
    only: Optional[List[str]] = None,
    epochs_override: Optional[int] = None,
) -> pd.DataFrame:
    outputs_root = ensure_dir(outputs_root)
    # Build the master cohort (full ISIC), then pin val/test once.
    full_spec_cfg = copy.deepcopy(cfg)
    full_spec_cfg.set_path("data.sample.n_benign", None)
    from .data import CohortSpec, build_cohort
    from .data.download import locate_or_download

    locator = locate_or_download(full_spec_cfg, allow_download=False)
    cohort = build_cohort(
        locator,
        processed_dir=cfg.data.processed_dir,
        spec=CohortSpec(keep_all_malignant=True, n_benign=None, seed=int(cfg.project.seed)),
    )
    fixed = build_fixed_splits(
        cohort,
        val_fraction=float(cfg.data.split.val_fraction),
        test_fraction=float(cfg.data.split.test_fraction),
        seed=int(cfg.data.split.seed),
    )
    dump_json(fixed, outputs_root / "fixed_splits.json")

    pipelines = only or list(PIPELINE_REGISTRY)
    summary_rows = []
    for budget in budgets:
        scale_cfg = ScaleRunConfig(
            benign_budget=budget,
            pipelines=pipelines,
            skip_cnn=skip_cnn,
            epochs=epochs_override,
        )
        result = run_scale_experiment(
            cfg,
            scale_cfg,
            cohort=cohort,
            fixed_splits=fixed,
            outputs_root=outputs_root,
        )
        for _, row in result["metrics"].iterrows():
            r = dict(row)
            r["data_size"] = scale_cfg.label
            r["benign_budget"] = (
                "full" if scale_cfg.benign_budget is None else scale_cfg.benign_budget
            )
            summary_rows.append(r)
    summary_df = pd.DataFrame(summary_rows)
    out_csv = outputs_root / "summary_metrics_all_scales.csv"
    # Column order for dissertation tables
    preferred = [
        "data_size", "benign_budget", "pipeline", "transparency_tag",
        "auc", "balanced_accuracy", "recall", "specificity",
        "precision", "f1", "threshold", "tn", "fp", "fn", "tp",
    ]
    cols = [c for c in preferred if c in summary_df.columns] + [
        c for c in summary_df.columns if c not in preferred
    ]
    summary_df = summary_df[cols]
    summary_df.to_csv(out_csv, index=False)
    logger.info("Scale-experiment summary → %s", out_csv)

    # cross-scale figures
    figs_dir = ensure_dir(outputs_root / "figures")
    plot_dataset_scale_distribution(cohort, fixed, budgets=list(budgets), out_path=figs_dir / "fig_dataset_scale_distribution.png")
    plot_metrics_vs_data_size(summary_df, out_path=figs_dir / "fig_metrics_vs_data_size.png")
    plot_pipeline_comparison_full(summary_df, out_path=figs_dir / "fig_pipeline_comparison_full.png")
    plot_transparent_lr_vs_tree_tradeoff(summary_df, out_path=figs_dir / "fig_transparent_lr_vs_tree_tradeoff.png")
    return summary_df
