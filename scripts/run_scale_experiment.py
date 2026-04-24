#!/usr/bin/env python3
"""Thin launcher for the dissertation's scale experiment.

Usage
-----
    python scripts/run_scale_experiment.py                      # all 4 budgets, all 5 pipelines
    python scripts/run_scale_experiment.py --skip-cnn           # transparent-only
    python scripts/run_scale_experiment.py --benign-budgets 40000 full --only transparent_lr transparent_tree
    python scripts/run_scale_experiment.py --epochs 1           # smoke run

Equivalent to ``python -m glassderm.cli scale-experiment ...``; kept as a
standalone script so collaborators can reproduce the dissertation figures in
one command without setting up ``PYTHONPATH``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.chdir(ROOT)

from glassderm.cli import main  # noqa: E402

if __name__ == "__main__":
    argv = ["scale-experiment", *sys.argv[1:]]
    raise SystemExit(main(argv))
