#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Small smoke sweep — exercises the image-only scale-experiment end-to-end on
# the smallest benign budget (40k) with only the two transparent pipelines.
# On a laptop CPU this completes in a few minutes; on GPU it is faster still.
# Useful for sanity checking that the repo wires together cleanly before
# launching the full 40k/80k/160k/full sweep.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python -m glassderm.cli inspect
python -m glassderm.cli prepare-data
python -m glassderm.cli scale-experiment \
    --benign-budgets 40000 \
    --only transparent_lr transparent_tree \
    --skip-cnn

echo
echo "Smoke run complete.  Inspect:"
echo "  outputs_scale/benign_40000/README_RESULTS.md"
echo "  outputs_scale/benign_40000/tables/main_metrics.csv"
echo "  outputs_scale/benign_40000/correct_prediction_feature_summary.md"
