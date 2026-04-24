"""GlassDerm — four-pipeline interpretable dermatology benchmark on ISIC 2024.

Pipelines (see README for a detailed per-pipeline transparency ladder):

* ``multitask_cnn``   — black-box CNN baseline (NOT end-to-end interpretable).
* ``hard_cbm``        — CNN perception + single linear concept readout.
* ``glassbox_nam``    — CNN perception + additive shape-function readout.
* ``transparent``     — OpenCV features + logistic regression / small tree.
                        The only pipeline that is *strictly* end-to-end
                        interpretable: every number between input and output is
                        a hand-computable formula.
"""
__version__ = "1.0.0"
