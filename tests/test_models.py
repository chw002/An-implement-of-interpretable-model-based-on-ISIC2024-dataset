"""Forward-pass + fit/predict smoke tests for every pipeline model."""
from __future__ import annotations

import torch


def test_multitask_forward(tiny_cfg):
    from glassderm.pipelines.multitask_cnn import _MultiTaskModel

    m = _MultiTaskModel(tiny_cfg.train.backbone, pretrained=False)
    x = torch.randn(2, 3, tiny_cfg.image.size, tiny_cfg.image.size)
    logit, abcd = m(x)
    assert logit.shape == (2, 1)
    assert abcd.shape == (2, 4)
    assert float(abcd.min()) >= 0.0 and float(abcd.max()) <= 1.0


def test_hard_cbm_forward(tiny_cfg):
    from glassderm.pipelines.hard_cbm import _HardCBMModel

    m = _HardCBMModel(tiny_cfg.train.backbone, pretrained=False)
    x = torch.randn(2, 3, tiny_cfg.image.size, tiny_cfg.image.size)
    logit, concepts = m(x)
    assert logit.shape == (2, 1)
    assert concepts.shape == (2, 4)


def test_glassbox_nam_forward(tiny_cfg):
    from glassderm.pipelines.glassbox_nam import _GlassBoxNAMModel

    m = _GlassBoxNAMModel(tiny_cfg.train.backbone, pretrained=False, hidden=16)
    x = torch.randn(2, 3, tiny_cfg.image.size, tiny_cfg.image.size)
    logit, concepts = m(x)
    assert logit.shape == (2, 1)
    assert concepts.shape == (2, 4)


def test_transparent_lr_fit_predict_save_load(tiny_cfg):
    import os
    from glassderm.artefacts import build_artefacts, prepare_data
    from glassderm.pipelines.transparent import TransparentLRPipeline
    from glassderm.utils import get_logger

    prepare_data(tiny_cfg)
    arte = build_artefacts(tiny_cfg, need_images=False)

    pipe = TransparentLRPipeline(tiny_cfg, get_logger("glassderm.test"))
    pipe.fit(arte)
    pred = pipe.predict("test", arte)
    assert len(pred.probs) == len(arte["transparent_features"]["test"])
    assert pred.probs.min() >= 0.0 and pred.probs.max() <= 1.0

    ck = tiny_cfg.outputs.checkpoints
    os.makedirs(ck, exist_ok=True)
    pipe.save(f"{ck}/transparent_lr.joblib")
    pipe2 = TransparentLRPipeline(tiny_cfg, get_logger("glassderm.test"))
    pipe2.load(f"{ck}/transparent_lr.joblib")
    pred2 = pipe2.predict("test", arte)
    assert pred.probs.tolist() == pred2.probs.tolist()


def test_transparent_tree_fit_predict_save_load(tiny_cfg):
    import os
    from glassderm.artefacts import build_artefacts, prepare_data
    from glassderm.pipelines.transparent import TransparentTreePipeline
    from glassderm.utils import get_logger

    prepare_data(tiny_cfg)
    arte = build_artefacts(tiny_cfg, need_images=False)

    pipe = TransparentTreePipeline(tiny_cfg, get_logger("glassderm.test"))
    pipe.fit(arte)
    pred = pipe.predict("test", arte)
    assert len(pred.probs) == len(arte["transparent_features"]["test"])
    assert pred.probs.min() >= 0.0 and pred.probs.max() <= 1.0

    ck = tiny_cfg.outputs.checkpoints
    os.makedirs(ck, exist_ok=True)
    pipe.save(f"{ck}/transparent_tree.joblib")
    pipe2 = TransparentTreePipeline(tiny_cfg, get_logger("glassderm.test"))
    pipe2.load(f"{ck}/transparent_tree.joblib")
    pred2 = pipe2.predict("test", arte)
    assert pred.probs.tolist() == pred2.probs.tolist()
