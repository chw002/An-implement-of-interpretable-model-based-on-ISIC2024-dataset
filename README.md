# GlassDerm — image-only interpretable dermatology benchmark on ISIC 2024

GlassDerm is a final-year research project that compares **five** ways to do
binary skin-lesion classification on the official **ISIC 2024** permissive
training release.  Every pipeline consumes *only the dermoscopic image*: the
ISIC/TBP tabular metadata (`tbp_lv_*`, `clin_size_long_diam_mm`, `age_approx`,
`sex`, `iddx_*`, …) is explicitly forbidden from reaching any model input.
This is enforced by an audit step that writes `feature_audit.csv` next to every
fitted model — a pipeline that tries to read a forbidden column aborts with
`MetadataLeakError`.

The project's central contribution is the *transparency ladder* visible in
every table we produce:

| # | Pipeline           | Perception                       | Reasoning head              | Transparency tag |
|---|--------------------|----------------------------------|-----------------------------|------------------|
| 1 | `multitask_cnn`    | EfficientNet-B0                  | MLP classifier + ABCD head  | `image_only_baseline` (black-box reference) |
| 2 | `hard_cbm`         | EfficientNet-B0 → ABCD           | `Linear(4 → 1)`             | `interpretable_partial` — linear readout only; CNN perception still opaque |
| 3 | `glassbox_nam`     | EfficientNet-B0 → ABCD           | `Σᵢ fᵢ(cᵢ)` (NAM)           | `interpretable_partial` — additive readout; CNN perception still opaque |
| 4 | `transparent_lr`   | OpenCV formulas → pixel features | Logistic regression         | `fully_auditable_image_only` — every step is a closed-form formula a human can replay |
| 5 | `transparent_tree` | OpenCV formulas → pixel features | Depth-4 decision tree       | `fully_auditable_image_only` — every leaf is a short rule a human can read |

Only `transparent_lr` / `transparent_tree` are labelled *fully auditable*.
`multitask_cnn` is explicitly a black-box baseline and everything — the code,
the config, the tables, the README — says so in those exact words.

## Why GlassDerm?

Most "interpretable medicine" papers graft Grad-CAM/SHAP/LIME onto a black-box
CNN and call it a day.  For a deployed clinical tool that is not enough: the
reviewer wants a decision path they can *audit*, not an after-the-fact
explanation of a decision made elsewhere.  GlassDerm therefore:

1. **Compares three interpretable heads on the same CNN backbone**, so
   "what does interpretable reasoning cost us?" is a single variable;
2. **Adds two fully image-only pipelines** that replace the CNN itself with a
   documented chain of OpenCV formulas, then a logistic regression
   (`transparent_lr`) or a shallow decision tree (`transparent_tree`) whose
   complete rule set fits on one page;
3. **Shares the same threshold** between `predict()` and `explain()` in every
   pipeline, so case-study narratives never contradict the main metrics
   table — enforced by `tests/test_explanations.py`.

## Dataset

We use the **ISIC 2024** permissive-training release exactly as distributed:

* the full malignant pool is kept (≈294 images);
* the training pool's benign count is **swept across four budgets**:
  `40 000 / 80 000 / 160 000 / full` (≈217 k total), with every malignant
  retained in each budget;
* the **val and test splits are fixed once** across all four budgets —
  patient-disjoint, 15 % + 15 % of patients — so AUC differences between
  scales reflect more training data, not a different test set.

### Getting the data

If the official release is already on disk, point `data.raw_dir` in
`configs/default.yaml` at it; otherwise `glassderm inspect` prints every path
it is going to look in.  `scripts/download_isic2024.sh` is a Kaggle helper.

## Installing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .    # optional — installs the `glassderm` console script
```

## Running

The CLI has **exactly three** sub-commands:

```bash
# 0. (optional) show the resolved config and dataset locator, no side effects
python -m glassderm.cli inspect

# 1. locate ISIC 2024, build the cohort + patient-disjoint split,
#    cache the pixel-derived feature parquet
python -m glassderm.cli prepare-data

# 2. run every pipeline across every benign budget; writes outputs_scale/
python -m glassderm.cli scale-experiment
```

`scale-experiment` accepts the following useful flags:

| flag | default | effect |
|------|---------|--------|
| `--benign-budgets 40000 80000 160000 full` | all four | which training-benign sizes to sweep; `full`/`none` means "keep every benign" |
| `--outputs-root PATH`                      | `outputs_scale` | where per-scale directories are written |
| `--only transparent_lr transparent_tree`   | every pipeline | restrict to a subset |
| `--skip-cnn`                               | off     | skip `multitask_cnn / hard_cbm / glassbox_nam` — useful on machines without CUDA |
| `--epochs N`                               | from config | override `train.epochs` (for smoke tests) |
| `-o key.subkey=value`                      | —       | arbitrary config override, parsed before the subcommand |

A thin launcher script is also provided:

```bash
python scripts/run_scale_experiment.py --benign-budgets 40000 full
```

### Reproducing the dissertation run

```bash
# full run — four budgets, every pipeline, GPU recommended for CNNs
python -m glassderm.cli scale-experiment \
    --benign-budgets 40000 80000 160000 full

# transparent-only run (fast, CPU OK, ~10 min on a laptop)
python -m glassderm.cli scale-experiment \
    --benign-budgets 40000 80000 160000 full \
    --only transparent_lr transparent_tree --skip-cnn
```

## Outputs

After `scale-experiment` completes, `outputs_scale/` has this shape:

```
outputs_scale/
├── fixed_splits.json                       # the patient-disjoint val/test, shared across budgets
├── summary_metrics_all_scales.csv          # every (pipeline × budget) row
├── figures/
│   ├── fig_dataset_scale_distribution.png
│   ├── fig_metrics_vs_data_size.png
│   ├── fig_pipeline_comparison_full.png
│   └── fig_transparent_lr_vs_tree_tradeoff.png
└── benign_{40000,80000,160000,full}/
    ├── README_RESULTS.md                   # a one-page summary of this scale
    ├── splits.json                         # exact ids used at this budget
    ├── cohort_provenance.json              # budget, counts, per-split patient tallies
    ├── config_snapshot.yaml                # resolved config used at this budget
    ├── tables/
    │   ├── main_metrics.csv                # one row per pipeline (AUC, F1, recall, …)
    │   └── thresholds.csv                  # val-tuned τ per pipeline
    ├── case_studies/
    │   └── case_studies.md                 # TP + FP + FN + TN narratives (one MD, not N files)
    ├── correct_prediction_feature_summary.csv   # machine-readable full ranking
    ├── correct_prediction_feature_summary.md    # human-readable top-10 per bucket
    ├── figures/
    │   ├── fig_confusion_matrices.png
    │   ├── fig_feature_importance_or_shape_functions.png
    │   ├── fig_transparent_explanation_example.png
    │   └── fig_correct_prediction_features.png
    └── {pipeline}/                         # one subdir per pipeline that ran
        ├── model.{joblib|pth}              # .joblib for transparent_*, .pth for CNNs
        ├── pipeline_meta.json              # name, transparency, τ, feature manifest
        ├── feature_audit.{csv,json}        # every column → model_input? + source ∈ {pixel_derived, …}
        ├── predictions_test.csv            # image_id, prob, pred, label
        ├── test_metrics.json               # AUC, balanced accuracy, recall, …
        └── (one of)
            ├── transparent_lr_readout.{txt,json}     # logit = Σ wᵢ·xᵢ + b, weights + per-case chain
            ├── transparent_tree_rules.txt            # every leaf as plain-English rules
            ├── transparent_tree_readout.json
            └── hard_cbm_readout.txt                  # 4 weights + bias of the linear head
```

The feature audit (`feature_audit.csv`) is the load-bearing file: every
dissertation claim about "image-only" is directly verifiable by inspecting it.
`source` must be `pixel_derived` for every row where `is_model_input=True`.

## Pipeline deep dives

### 1. `multitask_cnn` — black-box baseline

```
image → EfficientNet-B0 → pooled_features ─┬─ MLP → logit
                                           └─ MLP → A,B,C,D (auxiliary)
```

The ABCD head is an auxiliary task; the final prediction comes from the MLP
classifier and is **not** auditable from the input.  Present as the
interpretability-cost reference.

### 2. `hard_cbm` — concept bottleneck + linear readout

```
image → EfficientNet-B0 → ConceptHead → (A,B,C,D) ──► Linear(4,1) → logit
```

Final decision: `logit = w_A·A + w_B·B + w_C·C + w_D·D + b`.  Four weights and
a bias, dumped verbatim to `{pipeline_dir}/hard_cbm_readout.txt`.  The *linear
readout* is auditable; the ABCD predictor is still a CNN, so we tag the
pipeline `interpretable_partial`.

### 3. `glassbox_nam` — concept bottleneck + additive shape functions

```
image → EfficientNet-B0 → ConceptHead → (A,B,C,D)
        logit = f_A(A) + f_B(B) + f_C(C) + f_D(D) + b
```

Each `f_i` is a tiny one-input MLP — plotted as a 1-D curve in
`fig_feature_importance_or_shape_functions.png`.  No cross-concept
interactions, but the concept predictor is still a CNN — hence also
`interpretable_partial`.

### 4/5. `transparent_lr` & `transparent_tree` — **fully auditable, image-only**

These are the pipelines the dissertation is re-built around.  They replace the
CNN perception stage with a closed-form chain of OpenCV operations, and the
reasoning stage with either a logistic regression or a shallow decision tree.

```
raw image
  1. cv2.resize(image, 256×256)
  2. Otsu threshold + morphology     → binary lesion mask
  3. Closed-form feature formulas    → 20 pixel-derived numbers (geometry,
                                       asymmetry, border, color)  +  four
                                       ABCD concept proxies (closed-form
                                       weighted aggregates — no learned weights)
  4. Min-max scaling (fit on TRAIN only)
  5. Classifier:
        (a) transparent_lr   logit = Σ wᵢ·xᵢ + b → σ(logit)
        (b) transparent_tree rule path → leaf frequency
  6. Threshold τ (tuned on VAL)      → {benign, malignant}
```

Every feature's formula is defined in
[`src/glassderm/data/features.py`](src/glassderm/data/features.py); the scaler
persists its min/max, so scaling is reversible on paper.  `transparent_lr`
dumps every weight; `transparent_tree` dumps every rule.  A test
(`tests/test_explanations.py`) asserts that `predict()` and `explain()` never
disagree at the shared threshold.

#### Why this qualifies as *fully auditable, image-only*

1. **No learned perception.**  No CNN, no MLP, no embedding, no latent vector
   anywhere in the input→output chain.
2. **No metadata.**  `feature_audit.csv` shows every model-input column has
   `source == pixel_derived`; vendor columns like `tbp_lv_*` are rejected by
   `src/glassderm/data/audit.py` before fit is even reached.
3. **Every formula is closed form and documented.**  OpenCV calls that are
   themselves deterministic algorithms (Otsu, Sobel, `findContours`) are
   treated as *glass*: their behaviour is public and their parameters live in
   the config.
4. **Same τ in predict/explain.**  The val-tuned threshold is stored in
   `pipeline_meta.json` and reused by the explain path.
5. **Reproduction by hand is feasible.**  Armed with `transparent_lr_readout.txt`
   (weights + scaler) and a raw image, a reviewer can reproduce any prediction
   in a spreadsheet.

#### Honest limitations

1. **Segmentation is Otsu + morphology.**  On heavily vignetted dermoscopy
   frames Otsu can latch onto the vignette; the extractor detects this
   (`_extraction_error = empty_mask`) and backs off to neutral feature values,
   but those frames do cost accuracy.
2. **20 pixel features are a design choice.**  They cover the clinical ABCD
   signals but are not exhaustive — pigment networks and streaks would need a
   richer pipeline.
3. **Raw AUC vs. CNN baselines.**  On ISIC 2024's extreme class imbalance the
   transparent pipelines typically have lower AUC than the CNN baselines —
   that is the whole trade-off the project is set up to measure and document.

## Most recommended pipeline for a supervisor demo

`transparent_lr` — the one the project is built around, and the only one whose
decision chain is auditable in full without any post-hoc magic.  Pair it with
`multitask_cnn` (black-box baseline) in every table so the trade-off is
explicit.

## Repo layout

```
.1project/
├── configs/default.yaml             # every experiment knob in one file
├── data/                            # raw / processed / features / cache
├── dissertation_bristol_fyp.md      # long-form write-up
├── outputs_scale/                   # created by `scale-experiment`
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── download_isic2024.sh         # Kaggle helper
│   ├── make_dissertation_docx.py    # markdown → docx
│   ├── make_report_docx.py          # README_RESULTS.md → docx
│   ├── run_demo.sh                  # small smoke sweep
│   └── run_scale_experiment.py      # thin launcher
├── src/glassderm/
│   ├── analysis/                    # case studies, plots, reports, correct-feature summary
│   ├── data/                        # download, sample, split, datasets, features, audit
│   ├── evaluation/                  # metrics, threshold selection, report
│   ├── pipelines/                   # multitask_cnn / hard_cbm / glassbox_nam / transparent
│   ├── training/                    # orchestrator (used internally by scale.py)
│   ├── utils/                       # seeding, logging, IO
│   ├── artefacts.py                 # build_artefacts + prepare_data entry points
│   ├── cli.py                       # three subcommands
│   ├── config.py                    # YAML loader with ${…} interpolation + overrides
│   └── scale.py                     # scale-experiment driver
└── tests/                           # conftest, config, data, models, explanations, CLI
```

## Reproducibility cheat-sheet

* Fixed seeds across `random`, `numpy`, `torch`, `cuDNN` via
  `glassderm.utils.set_seed`.
* Val/test patient-disjoint splits are computed once and persisted to
  `outputs_scale/fixed_splits.json`, so every benign budget trains against the
  same test set.
* Scalers are fit on **train only**; the fitted parameters go into the
  pipeline's `.joblib` / `.pth`.
* Every pipeline writes `pipeline_meta.json` capturing name, transparency tag,
  val-tuned threshold, and the exact feature manifest fed to the model.
* `feature_audit.csv` is the single auditable file for the image-only claim;
  CI could gate on it and refuse a merge if any row has `source != pixel_derived`.
* Every scale snapshots the resolved config to `config_snapshot.yaml`, so the
  experiment is reconstructable from the output directory alone.
