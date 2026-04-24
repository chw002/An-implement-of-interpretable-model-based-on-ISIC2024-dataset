"""Explanation-versus-prediction contract for every transparent pipeline.

Every pipeline's ``explain()`` verdict must match its ``predict()`` verdict at
the same threshold — if those ever disagree the case studies would lie to
the reader.  We check both the logistic-regression pipeline
(``TransparentLRPipeline``) and the decision-tree pipeline
(``TransparentTreePipeline``).
"""
from __future__ import annotations

import math

import numpy as np


def test_transparent_lr_explain_matches_predict(tiny_cfg):
    from glassderm.artefacts import build_artefacts, prepare_data
    from glassderm.pipelines.transparent import TransparentLRPipeline
    from glassderm.utils import get_logger

    prepare_data(tiny_cfg)
    arte = build_artefacts(tiny_cfg, need_images=False)

    pipe = TransparentLRPipeline(tiny_cfg, get_logger("glassderm.test"))
    pipe.fit(arte)
    pipe.set_threshold(0.42)

    pred = pipe.predict("test", arte)
    frame = pred.to_frame()
    for _, row in frame.iterrows():
        explanation = pipe.explain(row.to_dict(), arte)
        expected = "MALIGNANT" if row.pred == 1 else "BENIGN"
        assert explanation["verdict"] == expected, (
            f"image {row.image_id}: predict→{expected} but explain→{explanation['verdict']}"
        )
        assert abs(explanation["prob"] - row.prob) < 1e-6


def test_transparent_tree_explain_matches_predict(tiny_cfg):
    from glassderm.artefacts import build_artefacts, prepare_data
    from glassderm.pipelines.transparent import TransparentTreePipeline
    from glassderm.utils import get_logger

    prepare_data(tiny_cfg)
    arte = build_artefacts(tiny_cfg, need_images=False)

    pipe = TransparentTreePipeline(tiny_cfg, get_logger("glassderm.test"))
    pipe.fit(arte)
    pipe.set_threshold(0.5)

    pred = pipe.predict("test", arte)
    frame = pred.to_frame()
    for _, row in frame.iterrows():
        explanation = pipe.explain(row.to_dict(), arte)
        expected = "MALIGNANT" if row.pred == 1 else "BENIGN"
        assert explanation["verdict"] == expected, (
            f"image {row.image_id}: predict→{expected} but explain→{explanation['verdict']}"
        )
        # Tree probabilities are leaf frequencies — equal to `row.prob`.
        assert abs(explanation["prob"] - row.prob) < 1e-6


def test_hard_cbm_explanation_is_linear(tiny_cfg):
    """HardCBM's explanation must recompute logit = Σ wᵢcᵢ + b exactly."""
    from glassderm.pipelines.hard_cbm import HardCBMPipeline
    from glassderm.utils import get_logger

    pipe = HardCBMPipeline(tiny_cfg, get_logger("glassderm.test"))
    pipe.set_threshold(0.5)

    row = {
        "concept_A": 0.2,
        "concept_B": 0.4,
        "concept_C": 0.6,
        "concept_D": 0.8,
        "prob": 0.0,
        "pred": 0,
    }
    explanation = pipe.explain(row, artefacts={})

    w, b = pipe._weights_bias()
    manual_logit = float(np.dot(w, [0.2, 0.4, 0.6, 0.8]) + b)
    manual_prob = 1.0 / (1.0 + math.exp(-manual_logit))
    assert abs(explanation["logit"] - manual_logit) < 1e-5
    assert abs(explanation["prob"] - manual_prob) < 1e-5


def test_threshold_strategy_selection():
    from glassderm.evaluation.thresholds import select_threshold

    probs = np.array([0.1, 0.2, 0.4, 0.6, 0.9, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t_youden = select_threshold(probs, labels, "youden")
    assert 0.0 < t_youden < 1.0
    t_f1 = select_threshold(probs, labels, "f1")
    assert 0.0 < t_f1 < 1.0
    t_fixed = select_threshold(probs, labels, "fixed", fixed=0.42)
    assert t_fixed == 0.42
