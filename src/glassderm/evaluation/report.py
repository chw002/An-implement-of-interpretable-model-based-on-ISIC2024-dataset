"""Cross-pipeline evaluation — writes the tables the README points at."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from ..pipelines import Pipeline, PIPELINE_REGISTRY, TRANSPARENCY_TAGS
from ..utils import dump_json, get_logger
from .metrics import compute_metrics, metrics_to_row, roc_points

logger = get_logger("glassderm.eval")


@dataclass
class EvaluationArtefacts:
    main_table_csv: Path
    main_table_json: Path
    predictions_dir: Path
    per_pipeline_json: Dict[str, Path]


def evaluate_all(
    pipelines: Mapping[str, Pipeline],
    artefacts: Mapping[str, Any],
    *,
    outputs_root: str | Path,
    tables_dir: str | Path,
    reports_dir: str | Path,
) -> EvaluationArtefacts:
    outputs_root = Path(outputs_root)
    tables_dir = Path(tables_dir)
    reports_dir = Path(reports_dir)
    predictions_dir = outputs_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_pipeline_json: Dict[str, Path] = {}

    for name, pipe in pipelines.items():
        logger.info("Evaluating %s on test split", name)
        pred = pipe.predict("test", artefacts)
        metrics = compute_metrics(pred.labels, pred.probs, pred.threshold)
        transparency = TRANSPARENCY_TAGS.get(name, pipe.transparency_tag)
        rows.append(metrics_to_row(name, transparency, metrics))

        # Persist per-pipeline artefacts
        fpr, tpr, _ = roc_points(pred.labels, pred.probs)
        per_pipeline_json[name] = reports_dir / f"{name}_test_metrics.json"
        dump_json(
            {
                "metrics": metrics,
                "threshold": pred.threshold,
                "transparency": transparency,
                "roc_fpr": fpr.tolist(),
                "roc_tpr": tpr.tolist(),
            },
            per_pipeline_json[name],
        )
        pred.to_frame().to_csv(
            predictions_dir / f"{name}_test_predictions.csv", index=False
        )

    df = pd.DataFrame(rows).sort_values("pipeline")
    main_csv = tables_dir / "main_metrics.csv"
    main_json = tables_dir / "main_metrics.json"
    df.to_csv(main_csv, index=False)
    dump_json(df.to_dict(orient="records"), main_json)
    logger.info("Main metrics table → %s", main_csv)

    return EvaluationArtefacts(
        main_table_csv=main_csv,
        main_table_json=main_json,
        predictions_dir=predictions_dir,
        per_pipeline_json=per_pipeline_json,
    )
