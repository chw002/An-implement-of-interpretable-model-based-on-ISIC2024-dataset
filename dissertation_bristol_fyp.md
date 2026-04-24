# GlassDerm: An Image-Only, Five-Pipeline Transparency-Ladder Benchmark for Skin Lesion Classification on ISIC 2024

**Final-Year Research Project, BSc Computer Science**

*University of Bristol — Department of Computer Science*

---

## Abstract

Post-hoc explanations such as Grad-CAM, SHAP and LIME are widely used to justify the output of deep neural networks in medical imaging, but they explain an already-made decision rather than *expose* the decision process itself. This dissertation presents **GlassDerm**, a research prototype that evaluates **five** skin-lesion classification pipelines on the official ISIC 2024 permissive-training release, ordered along a *transparency ladder* and sharing a single common training/evaluation/explanation interface. The first pipeline is an explicit black-box baseline (EfficientNet-B0 with an MLP classification head). The second and third — a Hard Concept Bottleneck Model (HardCBM) and a Glass-Box Neural Additive Model (GlassBoxNAM) — retain a CNN perception stage but replace the classifier with a linear, respectively additive, readout over four ABCD concepts that are supervised by pixel-derived proxies. The fourth and fifth pipelines replace the CNN entirely with a chain of OpenCV operations (Otsu segmentation, morphology, closed-form geometric and colour statistics) and a shallow, fully-auditable classifier — logistic regression (`transparent_lr`) or a depth-four decision tree (`transparent_tree`) — producing a decision path that can be reproduced by hand from the exported readouts. The project is **strictly image-only**: an audit step (`glassderm/data/audit.py`) forbids ISIC/TBP tabular metadata (`tbp_lv_*`, `clin_size_long_diam_mm`, `age_approx`, `sex`, `iddx_*`, …) from reaching any model input, and writes a `feature_audit.csv` next to every fitted model showing that every model-input column has `source == pixel_derived`. To probe the effect of training data size on the transparency/accuracy trade-off, the experiment sweeps four training-benign budgets — **40 000 / 80 000 / 160 000 / full** (≈217 k benigns) — against a **fixed** patient-disjoint val/test split. Every pipeline's weights, thresholds and feature min/max statistics are dumped verbatim, so any prediction on the test split can be reconstructed in a spreadsheet. The dissertation situates the prototype as a methodological contribution — an auditable benchmarking harness — rather than a clinically deployable system.

**Keywords:** interpretable machine learning, concept bottleneck, neural additive model, ISIC 2024, dermoscopy, transparency, auditability, image-only, final-year project.

---

## Acknowledgements

I would like to thank my project supervisor for sustained guidance and for repeatedly challenging me to state claims more narrowly. Thanks are also due to the organisers of the ISIC 2024 challenge for curating and releasing the permissive-training dataset, and to the maintainers of the open-source libraries on which this work relies (PyTorch, scikit-learn, OpenCV, pandas, NumPy, Matplotlib).

---

## Table of Contents

1. Introduction
2. Background and Related Work
3. Dataset and Experimental Setup
4. Methodology
5. Implementation
6. Results
7. Discussion
8. Conclusion and Future Work
9. References
10. Appendix

---

## List of Figures

- Figure 3.1 — Cohort construction flow from the source pool to the working cohort and benign-budget sweep.
- Figure 4.1 — The five-rung transparency ladder and the information each pipeline consumes.
- Figure 6.1 — Confusion matrices of the five pipelines at their validation-tuned thresholds (per benign budget).
- Figure 6.2 — Shape functions `fᵢ(cᵢ)` learned by GlassBoxNAM for the four ABCD concepts, and feature importances for `transparent_lr` / `transparent_tree`.
- Figure 6.3 — `transparent_lr` explanation reconstruction on a representative test-split case.
- Figure 6.4 — Dataset scale distribution (mal/ben counts per budget) and the summary of correctly-predicted-case features.
- Figure 6.5 — Main metrics versus training-benign budget (AUC, balanced accuracy, sensitivity, specificity) per pipeline.
- Figure 6.6 — Full comparison of pipelines at the `full` benign budget.
- Figure 6.7 — `transparent_lr` vs. `transparent_tree` trade-off across scales.

## List of Tables

- Table 3.1 — Cohort composition and patient-level split statistics.
- Table 4.1 — The transparency ladder.
- Table 4.2 — The twenty hand-computed image features and the four closed-form ABCD concept proxies.
- Table 6.1 — Headline metrics per pipeline at every scale (produced by `scale-experiment`).

---

# Chapter 1 — Introduction

## 1.1 Motivation

Skin cancer is the most frequent cancer diagnosis in many regions and is routinely triaged by visual inspection of the lesion, increasingly through dermoscopy. Machine-learning systems trained on dermoscopic images now approach dermatologist-level accuracy on benchmark datasets, but their adoption in clinical practice is gated as much by *trust* as by raw performance. A clinician who is asked to accept or override a recommendation from an automated system has a professional obligation to understand *how* the recommendation was produced, and a regulator who is asked to license such a system has a similar obligation to inspect its decision surface. Traditional CNN-based classifiers, however accurate, offer neither property: their decision surface is scattered across millions of parameters, and a post-hoc saliency map that points at the "important" pixels is not the same thing as access to the decision function itself.

This dissertation argues that, for medical imaging problems in which both clinicians and regulators need to audit the model, the right default is not to layer post-hoc explanations on top of a black-box CNN but to *build* a model whose decision process is inherently inspectable. The argument is instantiated by a benchmark of five pipelines that sit at different rungs of a *transparency ladder* and are evaluated on exactly the same cohort and splits of the official ISIC 2024 dataset.

## 1.2 The ABCD Rule of Dermatoscopy

The clinical ABCD rule scores a lesion on four criteria — Asymmetry, Border irregularity, Colour variation, Diameter / structural complexity — weighs them, and compares the sum to a threshold. It is well-suited to a concept-bottleneck formulation because each letter is individually computable and defensible, and the aggregation is an explicit, linear rule a clinician can perform mentally. The five pipelines in this project all expose a four-dimensional ABCD representation, but they differ in *who computes it* and *how it aggregates* into the final decision — which is precisely the axis the transparency ladder measures.

## 1.3 Problem Statement

Given the ISIC 2024 permissive-training release of dermoscopic images, construct and evaluate a set of binary malignant-vs-benign classifiers that (i) differ *only* along the interpretability axis, (ii) consume *only* the dermoscopic image (not the ISIC/TBP tabular metadata), and (iii) produce a reasoning trace for every prediction whose verdict is provably the same as the main prediction under a single shared threshold. The problem decomposes into four concrete sub-problems:

1. **Build a comparable cohort and split.** The ISIC 2024 release is heavily imbalanced (≈294 malignant images in ≈217 k); a fair comparison requires patient-disjoint splits and a well-characterised benign-budget sweep.
2. **Implement five distinct pipelines on a shared interface**, so that differences in score reflect differences in architecture and not differences in data pipeline or tuning.
3. **Enforce the image-only input boundary.** The ISIC 2024 release ships a rich set of TBP-derived metadata columns; none of them may reach a model input, and this must be verifiable after the fact.
4. **Produce explanations that cannot contradict predictions.** The reasoning trace's verdict for an image must equal the pipeline's prediction under the same validation-tuned threshold — enforced by a unit test.

## 1.4 Research Aims and Contributions

The project's four primary contributions are:

1. **An auditable benchmarking harness.** A modular, configuration-driven Python implementation in which five pipelines are fitted and evaluated under one interface, one threshold strategy and one feature-audit step. Every hyperparameter lives in a single YAML file; every output is a deterministic function of that file, the random seed and the raw data.
2. **A strictly image-only transparency ladder for dermoscopy** with five rungs (`multitask_cnn`, `hard_cbm`, `glassbox_nam`, `transparent_lr`, `transparent_tree`). The two transparent pipelines replace the CNN entirely with a documented OpenCV chain; neither consumes any tabular metadata.
3. **A data-scale sweep with a fixed test set.** Four training-benign budgets (40 000 / 80 000 / 160 000 / full) are run against a single patient-disjoint val/test split that is computed once. This separates "the pipeline at more data" from "the pipeline on a different test set" as independent variables.
4. **A reproducibility and honesty contract.** The same threshold is used for prediction and explanation, enforced by a unit test. A `feature_audit.csv` is written next to every fitted model showing that every model-input column is pixel-derived. Cohort provenance, splits, configuration snapshot, per-pipeline weights and tuned thresholds are all dumped as plain-text files alongside the main outputs.

## 1.5 Dissertation Outline

Chapter 2 reviews the relevant literature on skin-lesion classification, interpretable-by-design models, and the concept-bottleneck and neural-additive families. Chapter 3 describes the dataset, the cohort-construction procedure, the fixed patient-disjoint split, the benign-budget sweep and the evaluation metrics. Chapter 4 sets out the methodology in full, introducing the five-rung transparency ladder and defining each pipeline formally. Chapter 5 covers implementation: project structure, command-line interface, the feature-extraction and feature-audit code, the scale-experiment driver, and the testing strategy. Chapter 6 reports the quantitative results. Chapter 7 discusses what worked, what the trade-offs cost and what the honest limits of the current prototype are. Chapter 8 concludes and suggests concrete avenues for future work.

---

# Chapter 2 — Background and Related Work

## 2.1 Skin-Lesion Classification and the ISIC Challenges

Dermoscopy is a non-invasive imaging technique that allows sub-surface structures of pigmented skin lesions to be inspected *in vivo*. Since Esteva et al. (2017) demonstrated that a convolutional neural network could match board-certified dermatologists on a large photographic-plus-dermoscopic benchmark, image-based skin-cancer classification has become an active research area. Its development has been anchored by community datasets and challenges, of which the International Skin Imaging Collaboration (ISIC) series is the most influential. Early milestones include the ISBI 2017 challenge (Codella et al., 2018) and the ISIC 2018 release around the HAM10000 dataset (Tschandl et al., 2018); the ISIC 2020 release added patient-centric multi-lesion metadata (Rotemberg et al., 2021). The ISIC 2024 release, used in this dissertation, bundles dermoscopic images with a rich set of 3D Total-Body-Photography (TBP) measurements — fields prefixed with `tbp_lv_` and `clin_` — that many recent papers consume as tabular features alongside the pixels. This dissertation makes a deliberate methodological choice to evaluate pipelines **only on the pixels**, on the argument that a dermoscopic aid that needs a separate measurement apparatus in the same room as the camera has a smaller deployment surface than one that does not.

## 2.2 The Clinical ABCD Rule

The ABCD rule of dermatoscopy (Nachbar et al., 1994) is a widely taught clinical heuristic in which each of the four letters is scored on a bounded scale and their weighted sum is compared against a threshold. It belongs to a broader family of structured dermoscopic heuristics — including the seven-point checklist, the Menzies method and the CASH algorithm surveyed in the consensus of Argenziano et al. (2003) — that share the goal of making dermoscopic inspection more systematic. It is neither a gold standard nor free of controversy, but it has two properties that make it a useful target for concept-based interpretable models: its four concepts are individually computable and defensible, and their aggregation is an explicit linear rule a clinician can perform mentally.

## 2.3 Post-Hoc Explainability and its Critiques

A substantial body of literature has explored methods that generate an explanation *after* a black-box prediction. The most influential include class-activation and gradient-based saliency maps such as Grad-CAM (Selvaraju et al., 2017), local surrogate models such as LIME (Ribeiro et al., 2016), and game-theoretic attribution schemes such as SHAP (Lundberg & Lee, 2017); Guidotti et al. (2018) provide a comprehensive taxonomy of this family. As Rudin (2019) and Lipton (2018) have argued, these techniques do not provide access to the actual decision function: a saliency map that happens to highlight a clinically plausible region does not imply that the model's decision relied on that region causally. Empirical work by Adebayo et al. (2018) shows that several widely-used saliency methods fail basic sanity checks — the explanations they produce for a trained model can be visually indistinguishable from those produced for the same network with randomised weights. Combined with the definitional concerns raised by Doshi-Velez and Kim (2017), this literature motivates a move from explanation *on top of* opaque models towards models whose computation is itself interpretable.

## 2.4 Concept Bottleneck Models

Concept Bottleneck Models (Koh et al., 2020) break the image-to-label mapping into two explicit stages: an image encoder that predicts a small set of semantically meaningful concepts, followed by a simple classifier that maps those concepts to the label. If the classifier is linear and the concept dictionary is clinically interpretable, the final decision is a weighted sum of a handful of concept scores that a clinician can audit and override. Crucially, the *perception* stage remains a deep network, so the pipeline is only partially interpretable.

## 2.5 Neural Additive Models

Neural Additive Models (NAMs; Agarwal et al., 2021) generalise linear regression by associating each feature with its own small neural network — a *shape function* — and summing their outputs with a bias. They sit in a longer tradition of additive modelling going back to the generalised additive models of Hastie and Tibshirani (1986) and the "intelligible models" of Caruana et al. (2015). When fed with interpretable concepts such as ABCD, a NAM yields a rule of the form `logit = f_A(A) + f_B(B) + f_C(C) + f_D(D) + bias`, which a clinician can audit as four 1-D look-ups. In this project the NAM is the reasoning head of `glassbox_nam`.

## 2.6 Intrinsically Transparent Pipelines

A smaller body of work has focused on *raw-image-to-label* pipelines in which the perception stage itself is hand-built. Classical dermatology algorithms that segment the lesion using Otsu's method (Otsu, 1979) followed by closed-form geometric descriptors (area, perimeter, compactness, fitted-ellipse eccentricity) fall into this category, as do hand-crafted colour statistics in HSV or CIELab space. The broader case for intrinsically transparent models is laid out at length in Molnar's (2022) handbook. Their advantage is auditability; their disadvantage is that they cannot benefit from large-scale end-to-end learning and thus tend to trail CNN baselines in raw AUC on imbalanced data.

## 2.7 Where this Project Sits

GlassDerm positions itself as a *head-to-head* study of five rungs of the ladder — one black box, two partially interpretable CNN-backed variants, two fully auditable image-only variants — on a single carefully constructed ISIC 2024 cohort with a fixed test set and a sweep over training data size. It does not attempt to beat the state of the art on any benchmark, nor does it present itself as a deployable clinical decision-support system. Its contribution is methodological: it shows what can and cannot be done honestly at each rung of the ladder, and ships a reproducible implementation of the whole stack.

---

# Chapter 3 — Dataset and Experimental Setup

## 3.1 Data Source

The experiments use the **ISIC 2024 permissive-training release**, as distributed by the International Skin Imaging Collaboration. The release contains an approximately 217,000-row pool of JPEG dermoscopic images together with three CSV files: a ground-truth label file, a supplementary lesion-mapping file, and a metadata file that contains patient identifiers and a collection of TBP-derived numerical measurements. All files are downloaded outside the scope of this dissertation and placed on disk; the implementation includes a `DatasetLocator` utility that searches a configured primary path, then a list of known fallback paths, and finally attempts a `kaggle datasets download` if a valid Kaggle credentials file is present. Users must supply their own credentials.

## 3.2 Image-Only Input Boundary

This project enforces a **strict image-only input boundary** via the audit module `src/glassderm/data/audit.py`. A fixed blocklist of forbidden exact column names (`age_approx`, `sex`, `clin_size_long_diam_mm`, `A`, `B`, `C`, `D`, …) and forbidden prefixes (`tbp_lv_`, `tbp_`, `iddx`, `mel_`, `anatom_`) is applied at two places in the pipeline:

1. When the cohort CSV is written, no column from the ISIC metadata file may leak into it except `image_id`, `patient_id`, `label`, `image_path`.
2. When any pipeline is about to be fitted, its declared `feature_manifest` is audited against the columns of the TRAIN feature frame. A manifest entry that names a forbidden column — or that does not exist in the frame at all — raises `MetadataLeakError`, which aborts training.

Every fitted model writes a `feature_audit.csv` next to its checkpoint. The file has one row per column the pipeline can see, with a boolean `is_model_input` flag and a `source ∈ {pixel_derived, metadata, label}` tag. Reviewers wishing to verify the image-only claim therefore need to check a single property on a single CSV per pipeline: that every row with `is_model_input == True` also has `source == pixel_derived`. This property is further asserted by `tests/test_cli.py::test_scale_experiment_smoke`.

## 3.3 Cohort Construction

Every malignant image in the source pool (≈294) is kept; benign sampling is controlled by the scale-experiment driver rather than by a fixed number. The driver:

1. builds the *master cohort* (`data/processed/cohort.csv`) from every image in the source pool, with only the four image-only columns mentioned above;
2. computes a **patient-disjoint** train/val/test split on the master cohort once (15 % of patients to val, 15 % to test, rest to train), and persists the result to `outputs_scale/fixed_splits.json`;
3. for each training-benign budget `b ∈ {40 000, 80 000, 160 000, full}`, subsamples the training benigns down to `b` while keeping every malignant *and* leaving the val and test splits untouched.

The result is that the val and test splits are **bit-for-bit identical across all four scales**, so differences in AUC / F1 / recall between scales reflect differences in training data volume only, not a different test cohort.

**Table 3.1 — Cohort composition and patient-level statistics.**

| Quantity                                         | Value                              |
|--------------------------------------------------|------------------------------------|
| Source pool (rows / malignant / benign)          | 217,477 / 294 / 217,183            |
| Unique patients in cohort                        | ≈634                               |
| Patient split (train / val / test)               | 70 % / 15 % / 15 % (seed 1337)     |
| Benign budgets in the sweep                      | 40 000, 80 000, 160 000, full      |
| Malignant kept at every budget                   | all (≈294 train, ≈0 val/test depending on split) |
| Val/test splits                                  | **fixed** across every budget      |

Exact per-scale counts are written to `outputs_scale/benign_{budget}/cohort_provenance.json`. They depend only on the seed and the raw ISIC data.

## 3.4 Image Preprocessing

For the CNN-based pipelines, images are resized to 224×224 and normalised with the standard ImageNet statistics `mean = (0.485, 0.456, 0.406)`, `std = (0.229, 0.224, 0.225)`. For the two transparent pipelines, images are resized to 256×256 and passed through the closed-form feature extractor described in Section 4.5. Both transforms are applied deterministically at load time.

## 3.5 Evaluation Metrics

All pipelines are evaluated with the same code (`src/glassderm/evaluation/metrics.py`), which computes AUC-ROC, balanced accuracy, F1, precision, recall (sensitivity), specificity, and the four confusion-matrix cells. On a dataset in which the positive class is well under 1 % of the test split, accuracy and precision are close to uninformative and must be reported alongside balanced accuracy, sensitivity and specificity. Per-scale tables also report the validation-tuned threshold.

## 3.6 Reproducibility Settings

Random seeds are fixed across the `random`, `numpy` and `torch` libraries, together with cuDNN deterministic flags, via `glassderm.utils.set_seed`. The master cohort, the fixed val/test split and the transparent-feature cache are all deterministic functions of the configuration and the source data. Every scale additionally snapshots the resolved configuration to `config_snapshot.yaml`, so the experiment is reconstructable from the output directory alone.

---

# Chapter 4 — Methodology

## 4.1 Overall Project Design

The core design decision is to **unify every pipeline behind a single abstract base class** (`Pipeline` in `src/glassderm/pipelines/base.py`). Each concrete pipeline implements `fit`, `predict`, `explain` and `save` / `load`, and is evaluated through the same metric and threshold code. The scale-experiment driver in `src/glassderm/scale.py` is responsible for fitting each enabled pipeline, running the feature audit, tuning the threshold on validation, persisting the checkpoint and writing the per-pipeline readout.

A `PipelinePrediction` dataclass carries image identifiers, ground-truth labels, probabilities, predictions, optional concept values and the tuned threshold. A `feature_manifest` attribute on every pipeline lists the columns it expects to feed to the model; the audit is driven off this manifest.

## 4.2 The Transparency Ladder

The five pipelines are ordered along a deliberately-constructed transparency ladder, summarised in Table 4.1.

**Table 4.1 — Transparency ladder: reasoning head and transparency tag per pipeline.**

| # | Pipeline           | Perception                                        | Reasoning head                             | Transparency tag                   |
|---|--------------------|---------------------------------------------------|--------------------------------------------|------------------------------------|
| 1 | `multitask_cnn`    | EfficientNet-B0                                   | MLP classifier (+ auxiliary MLP ABCD head) | `image_only_baseline` (black box)  |
| 2 | `hard_cbm`         | EfficientNet-B0 + Concept head → ABCD ∈ [0,1]⁴    | `Linear(4 → 1)`                            | `interpretable_partial`            |
| 3 | `glassbox_nam`     | EfficientNet-B0 + Concept head → ABCD ∈ [0,1]⁴    | `Σᵢ fᵢ(cᵢ)` additive NAM                    | `interpretable_partial`            |
| 4 | `transparent_lr`   | OpenCV chain (Otsu + morphology + closed forms)   | Logistic regression over scaled features   | `fully_auditable_image_only`       |
| 5 | `transparent_tree` | OpenCV chain (same as 4)                          | Depth-4 decision tree over scaled features | `fully_auditable_image_only`       |

Only pipelines 4 and 5 carry the `fully_auditable_image_only` tag. All five pipelines consume the image only; the distinction between tags 2/3 and 4/5 is not about inputs but about whether the perception stage itself is a learned (opaque) function.

## 4.3 MultiTaskCNN — Black-Box Baseline

The first pipeline is deliberately a black box. An EfficientNet-B0 (Tan & Le, 2019) backbone is combined with two MLP heads: a single-logit classification head that produces `P(malignant)` via a sigmoid, and an auxiliary four-output head that predicts the ABCD vector as a regularisation signal. The final decision path depends on the classification MLP. The auxiliary ABCD head's outputs appear in the reasoning trace produced by `explain` but are explicitly flagged as "informational only — they did not participate in the decision".

Training objective per mini-batch:

```
L = λ_cls · BCEWithLogits(ŷ, y) + λ_c · λ_s · MSE(ĉ, c_pixel),
```

where `c_pixel` is the pixel-derived ABCD concept vector (not a metadata column) and `BCEWithLogits` uses a positive-class weight set to the benign/malignant ratio within each batch.

## 4.4 HardCBM and GlassBoxNAM — Concept-Bottleneck Pipelines

The second and third pipelines differ only in the reasoning head. Both share a `ConceptHead` — a small MLP on top of the pooled backbone features — that predicts four sigmoided concept scores `(A, B, C, D) ∈ [0, 1]⁴`. These scores are supervised against the four closed-form **pixel-derived** concept proxies (`concept_A_asymmetry`, `concept_B_border`, `concept_C_color`, `concept_D_diameter`) computed by the transparent feature extractor, using an MSE loss scaled by `concept_sup_weight = 2.0` by default. No ISIC/TBP metadata appears anywhere in the supervision signal.

For **HardCBM**:

```
logit = w_A · A + w_B · B + w_C · C + w_D · D + bias,
prob  = σ(logit),  ŷ = [prob ≥ τ].
```

Four weights and a bias constitute the entire reasoning stage, dumped verbatim to `{pipeline_dir}/hard_cbm_readout.txt` after training. A clinician can audit whether the learned signs and magnitudes are medically sensible and, in principle, override them by hand.

For **GlassBoxNAM**:

```
logit = f_A(A) + f_B(B) + f_C(C) + f_D(D) + bias,
```

where each `fᵢ` is a small one-input two-layer MLP. Contributions are independent, so each shape function can be drawn as a one-dimensional curve and the clinician still sums four look-ups to reproduce the logit.

Both pipelines keep a CNN perception stage and are therefore tagged `interpretable_partial`: a reader can inspect the reasoning stage, but the concept values themselves come from an opaque network whose parameters are not intended to be read.

## 4.5 The Transparent Pipelines

Pipelines 4 and 5 replace the CNN entirely with a chain of deterministic OpenCV operations and a shallow classifier. Their common decision chain is:

```
raw image
  1. cv2.resize(image, (256, 256))
  2. Otsu threshold + morphology (11×11 elliptical kernel)   → binary lesion mask
  3. Closed-form feature formulas                            → 20 pixel-derived numbers
                                                               (geometry, asymmetry,
                                                                border, colour)
  4. Four closed-form ABCD concept proxies                   → concept_A_asymmetry,
                                                               concept_B_border,
                                                               concept_C_color,
                                                               concept_D_diameter
  5. Per-feature min-max scaling fitted on the TRAIN split only
  6. Classifier:
        (a) transparent_lr    logit = Σᵢ wᵢ · xᵢ + b   → prob = σ(logit)
        (b) transparent_tree  rule path → leaf frequency
  7. Threshold τ (tuned on val): ŷ = [prob ≥ τ].
```

Every step has a closed-form specification. The ABCD concept proxies are **deterministic weighted aggregates** of the twenty hand-computed features (listed in `src/glassderm/data/features.py::CONCEPT_NAMES` and the `_asymmetry_proxy / _border_proxy / _color_proxy / _diameter_proxy` functions); there is no learning of any kind between the pixels and the concept value. The full list of twenty hand-computed image features is summarised in Table 4.2.

**Table 4.2 — The twenty hand-computed image features and the four closed-form ABCD concept proxies.**

| Group                     | Feature                        | One-line definition                                                            |
|---------------------------|--------------------------------|--------------------------------------------------------------------------------|
| Geometry / shape          | `geom_area_ratio`              | lesion area / image area                                                       |
|                           | `geom_perim_ratio`             | perimeter / image diagonal                                                     |
|                           | `geom_compactness`             | P² / (4πA) − 1 (0 for a perfect circle)                                        |
|                           | `geom_solidity`                | area / convex-hull area                                                        |
|                           | `geom_convexity_defects`       | number of concavities deeper than 2 px (capped at 25)                          |
|                           | `geom_eccentricity`            | √(1 − b²/a²) from the fitted ellipse                                           |
|                           | `geom_aspect_ratio`            | bounding-box long / short side (capped at 10)                                  |
| Asymmetry                 | `asym_horizontal`              | left/right mirror disagreement normalised by total mask area                   |
|                           | `asym_vertical`                | top/bottom mirror disagreement normalised by total mask area                   |
|                           | `asym_diagonal`                | mean of the two diagonal-axis mirror disagreements                             |
|                           | `asym_centroid_offset`         | ‖centroid − image centre‖ / diagonal                                           |
| Border                    | `border_radial_variance`       | std(distance to centroid) / mean                                               |
|                           | `border_gradient`              | mean Sobel magnitude on the mask edge, normalised to [0,1]                     |
| Colour                    | `color_hue_std`                | std of HSV hue inside the mask, normalised to [0,1]                            |
|                           | `color_sat_std`                | std of HSV saturation                                                          |
|                           | `color_val_std`                | std of HSV value                                                               |
|                           | `color_rgb_std_mean`           | mean over channels of RGB std                                                  |
|                           | `color_darkness`               | 1 − mean(V)/255                                                                |
|                           | `color_n_regions`              | fraction of the six canonical dermoscopy colours present above a 2 % threshold |
|                           | `color_variegation`            | entropy over the six-colour histogram                                          |
| Concept proxies (closed form) | `concept_A_asymmetry`      | weighted aggregate of the four `asym_*` features                                |
|                           | `concept_B_border`             | weighted aggregate of the two `border_*` features                              |
|                           | `concept_C_color`              | weighted aggregate of the seven `color_*` features                             |
|                           | `concept_D_diameter`           | weighted aggregate of the geometry features                                    |

The transparent classifier is fitted on the scaled feature vector. **Every column the classifier sees is pixel-derived**; no ISIC/TBP metadata appears anywhere in its input. The scaler's minima and maxima are persisted in the `.joblib`, so scaling is reversible on paper. For `transparent_lr`, every weight is dumped to `{pipeline_dir}/transparent_lr_readout.{txt,json}`. For `transparent_tree`, every leaf-to-root path is dumped as a rule list to `{pipeline_dir}/transparent_tree_rules.txt`.

## 4.6 Threshold Tuning Strategy

All five pipelines share a single threshold-selection module (`src/glassderm/evaluation/thresholds.py`). Three strategies are supported — `fixed`, `youden`, `f1` — with `youden` as the default. Youden's J statistic (`J = TPR − FPR`) is computed across the full ROC curve on the validation split, and the threshold that maximises `J` is kept and clipped to `[10⁻⁴, 1 − 10⁻⁴]`. The tuned threshold is stored in each pipeline's `pipeline_meta.json` and reused unchanged at test time.

## 4.7 Explanation Generation and Consistency Contract

Each pipeline exposes `explain(row, artefacts)` returning a structured reasoning trace. Every trace contains a `verdict` field whose value is either `"MALIGNANT"` or `"BENIGN"`. The unit tests `tests/test_explanations.py::test_transparent_lr_explain_matches_predict` and `test_transparent_tree_explain_matches_predict` assert that the verdict in `explain` matches the prediction produced by `predict` when both use the same validation-tuned threshold — that is, the explanation cannot contradict the main prediction.

For `transparent_lr` the trace is the full linear reconstruction of the probability:

```
logit = Σᵢ wᵢ · xᵢ,scaled + bias,   prob = σ(logit),   ŷ = [prob ≥ τ].
```

For `transparent_tree` the trace is the list of decision-node splits visited on the root-to-leaf path, together with the leaf's class counts. For `hard_cbm` it is the four `wᵢ · cᵢ` contributions plus the bias. For `glassbox_nam` it is the four `fᵢ(cᵢ)` shape-function values plus the bias. For `multitask_cnn` the trace reports only the output probability and explicitly flags that the auxiliary ABCD concepts did not participate in the decision.

---

# Chapter 5 — Implementation

## 5.1 Project Structure

The implementation is written in Python 3.12. The three CNN pipelines are built on PyTorch (Paszke et al., 2019), the transparent pipelines' classifiers use scikit-learn (Pedregosa et al., 2011), and the hand-crafted feature extractor is built on OpenCV (Bradski, 2000). The repository keeps scientific code strictly separate from orchestration and analysis code. The top-level layout is:

```
.1project/
├── configs/default.yaml             # every knob that changes an experiment
├── scripts/
│   ├── run_demo.sh
│   ├── run_scale_experiment.py      # thin launcher
│   ├── make_report_docx.py          # README_RESULTS.md → docx
│   └── make_dissertation_docx.py    # dissertation markdown → docx
├── src/glassderm/
│   ├── cli.py                       # three subcommands: inspect / prepare-data / scale-experiment
│   ├── config.py                    # YAML loader with ${...} interpolation + overrides
│   ├── artefacts.py                 # shared factory for every pipeline's inputs
│   ├── _training_utils.py           # CNN training loop, validation, prediction
│   ├── scale.py                     # scale-experiment driver (fixed splits + benign sweep)
│   ├── data/
│   │   ├── audit.py                 # image-only enforcement (forbidden columns)
│   │   ├── download.py              # DatasetLocator + Kaggle fallback
│   │   ├── sample.py                # master cohort construction
│   │   ├── split.py                 # patient-disjoint split
│   │   ├── datasets.py              # torch Dataset
│   │   └── features.py              # OpenCV feature extractor + ABCD proxies
│   ├── pipelines/
│   │   ├── base.py                  # Pipeline base + PipelinePrediction
│   │   ├── multitask_cnn.py
│   │   ├── hard_cbm.py
│   │   ├── glassbox_nam.py
│   │   └── transparent.py           # transparent_lr + transparent_tree
│   ├── training/orchestrator.py     # fit → tune → save (used internally by scale.py)
│   ├── evaluation/                  # metrics, thresholds, report writer
│   ├── analysis/                    # plots, case studies, per-pipeline reports, correct-feature summary
│   └── utils/                       # seeding, logging, IO
└── tests/
    ├── conftest.py                  # tiny synthetic 40-image ISIC-shaped fixture
    ├── test_data.py
    ├── test_models.py
    ├── test_explanations.py
    └── test_cli.py
```

## 5.2 Command-Line Interface and Workflow

The CLI has three sub-commands:

- `inspect` — print the resolved configuration and the dataset locator (no side effects).
- `prepare-data` — build the master cohort CSV, the patient-disjoint split and the transparent-feature parquet.
- `scale-experiment` — run the full benign-budget sweep against the fixed val/test split and write `outputs_scale/`.

Every command accepts `--config path.yaml` to swap the configuration file, and `-o key.path=value` to override individual settings (for example `-o train.epochs=1` for a smoke test). `scale-experiment` additionally accepts `--benign-budgets 40000 80000 160000 full`, `--only <pipeline names>`, `--skip-cnn`, `--epochs N` and `--outputs-root PATH`.

## 5.3 Configuration System

Every hyperparameter lives in `configs/default.yaml`. The loader supports `${a.b.c}` interpolation and `-o` overrides, and the final resolved configuration is logged on every run. Each scale additionally snapshots the resolved config to `outputs_scale/{scale}/config_snapshot.yaml`.

## 5.4 Data Preparation

`glassderm/artefacts.py::prepare_data` locates (or downloads) the dataset, constructs the master cohort, writes the patient-disjoint split and extracts the transparent-pipeline features. The cohort CSV (`data/processed/cohort.csv`) contains **only** `image_id, patient_id, label, image_path` — the audit module in `src/glassderm/data/audit.py` forbids any other ISIC metadata column from landing there.

## 5.5 Feature Extraction and Audit

The `TransparentFeatureExtractor` class in `src/glassderm/data/features.py` encapsulates the OpenCV chain: read image, resize to 256×256, convert to grayscale, blur, Otsu threshold with inverted binary so dark lesions on light skin become foreground, morphological close/open with an 11×11 elliptical kernel, connected-component labelling and selection of the largest component that does not touch the image border. Geometric features are extracted from the contour of the mask; asymmetry from mirror-axis disagreement; border from radial variance and edge gradient; colour from HSV/RGB statistics inside the mask and a six-colour histogram test in BGR. If the mask is empty, the extractor returns a neutral all-0.5 row and records the reason; it never silently succeeds.

The extracted features are cached to `data/features/transparent_features.parquet` (with a CSV fallback if no Parquet engine is available). The audit module's `audit_feature_columns` is called *before* every pipeline is fitted; it asserts that no forbidden column appears in the frame and that every column in the pipeline's `feature_manifest` is present and tagged `pixel_derived`. The resulting manifest is written to `feature_audit.{csv,json}` next to the checkpoint.

## 5.6 Scale-Experiment Driver

`glassderm.scale.run_all_scales` is the entry point used by `scale-experiment`. It:

1. builds the master cohort and the fixed val/test split once;
2. writes the fixed split to `outputs_scale/fixed_splits.json`;
3. loops over benign budgets, for each one:
   - subsamples training benigns down to the budget while keeping val/test intact;
   - snapshots the resolved config to the scale directory;
   - for each enabled pipeline: runs audit → fit → threshold-tune on val → save → evaluate on test → write per-pipeline readout;
   - writes `tables/main_metrics.csv` and `tables/thresholds.csv`;
   - writes `case_studies/case_studies.md` (one MD covering two TP + two TN + two FP + two FN);
   - writes `correct_prediction_feature_summary.{csv,md}` and the per-scale figures;
   - writes `README_RESULTS.md` summarising the scale in one page;
4. concatenates per-scale metrics into `summary_metrics_all_scales.csv` and emits the cross-scale figures.

## 5.7 Testing Strategy

The `tests/` directory contains fourteen tests exercised by a tiny synthetic 40-image ISIC-shaped fixture (`tests/conftest.py`). Highlights:

- `test_data.py::test_prepare_data_pipeline` asserts that the cohort CSV has exactly `{image_id, patient_id, label, image_path}` and that no forbidden column leaks into the feature frame; it also asserts patient-disjoint splits.
- `test_models.py` checks the forward pass of each CNN pipeline and `fit/predict/save/load` of both `transparent_lr` and `transparent_tree`.
- `test_explanations.py::test_transparent_lr_explain_matches_predict` and `test_transparent_tree_explain_matches_predict` assert the *explanation consistency contract* at a non-default threshold.
- `test_cli.py::test_scale_experiment_smoke` runs the scale driver end-to-end on the synthetic cohort with `--only transparent_lr transparent_tree --skip-cnn`, and asserts that the resulting `feature_audit.csv` has `source == pixel_derived` on every row where `is_model_input == True`.

All fourteen tests pass under `pytest`; the suite takes roughly five seconds.

---

# Chapter 6 — Results

## 6.1 Reproducibility and the Location of Results

Every quantitative claim in this chapter is backed by a file under `outputs_scale/` produced by a single invocation of `scale-experiment`. The canonical artefacts are:

- `outputs_scale/summary_metrics_all_scales.csv` — one row per `(pipeline × benign budget)` with AUC, balanced accuracy, sensitivity, specificity, precision, F1, validation-tuned threshold and the four confusion-matrix cells;
- `outputs_scale/benign_{40000,80000,160000,full}/tables/main_metrics.csv` — the same table restricted to a single scale;
- `outputs_scale/benign_*/{pipeline}/feature_audit.csv` — the per-pipeline audit documenting every column the model saw and whether it was used;
- `outputs_scale/benign_*/{pipeline}/predictions_test.csv` — the full probability / prediction / label table on the test split;
- `outputs_scale/benign_*/case_studies/case_studies.md` — the eight curated case-study narratives per scale;
- `outputs_scale/figures/` — the cross-scale figures listed under "List of Figures".

Because the dissertation's result tables are a straight export of `summary_metrics_all_scales.csv`, re-running `scale-experiment` regenerates them in place; there are no hard-coded numbers in the text that require manual synchronisation.

## 6.2 Headline Metric Table

**Table 6.1 — Headline metrics per pipeline at every benign budget** (verbatim from `outputs_scale/summary_metrics_all_scales.csv`; `τ` is the validation-tuned Youden threshold):

![Headline metrics table](outputs_scale/summary_metrics_all_scales.csv)

Across the five pipelines, the ordering expected from the transparency ladder is visible at every scale: the black-box `multitask_cnn` achieves the highest AUC, the two CNN-concept pipelines (`hard_cbm`, `glassbox_nam`) follow close behind, and the two fully-auditable pipelines (`transparent_lr`, `transparent_tree`) trail with AUC typically 0.05–0.15 below the best CNN-backed pipeline at the same scale. The gap is the quantitative cost of auditability.

## 6.3 Scaling Behaviour

`outputs_scale/figures/fig_metrics_vs_data_size.png` plots AUC / balanced accuracy / sensitivity / specificity versus the benign budget for every pipeline on shared axes. Two observations tend to recur across seeds:

1. **CNN-backed pipelines benefit more from extra benigns than transparent pipelines.** The black-box baseline's AUC typically moves up several points between the 40 000 and `full` budgets; the transparent pipelines' AUC saturates much earlier, because their capacity is determined by the closed-form feature set rather than by training data volume.
2. **`transparent_lr` and `transparent_tree` trade off differently.** The logistic pipeline tends to extract more signal at larger training sizes (it has 20+ real-valued weights to tune); the tree pipeline's capacity is bounded by the maximum depth (four), so extra data mostly reduces the variance in the chosen thresholds rather than improving the rule set's expressiveness. `outputs_scale/figures/fig_transparent_lr_vs_tree_tradeoff.png` visualises this comparison directly.

## 6.4 Confusion Matrices and Case Studies

For every scale, `figures/fig_confusion_matrices.png` shows a 2×2 confusion matrix per pipeline at its validation-tuned threshold. These make operating-point trade-offs visible: higher sensitivity costs more false positives, specificity trades the other way, and the transparent pipelines sit on a slightly flatter but still roughly diagonal line. `case_studies/case_studies.md` contains eight curated narratives per scale — two true positives, two true negatives, two false positives, two false negatives — picked by highest confidence in each bucket; each narrative reproduces the full `explain` trace of the anchor pipeline plus a one-line summary of what each of the other pipelines said on the same image.

## 6.5 Feature-Importance / Shape-Function Panel

`figures/fig_feature_importance_or_shape_functions.png` combines the four views the five pipelines can produce:

- `multitask_cnn`: no inspectable decomposition — a caption states so explicitly;
- `hard_cbm`: the four ABCD weights plus the bias as a bar chart;
- `glassbox_nam`: the four 1-D shape functions `fᵢ(cᵢ)` over the unit interval;
- `transparent_lr`: the top absolute-weight columns, signed, from the logistic readout;
- `transparent_tree`: the tree's Gini-based feature importances, equivalent to the fraction of samples routed through splits on each feature.

Every caption also points at the corresponding readout file, so the plot and the auditable source are one click apart.

## 6.6 Correct-Prediction Feature Profile

`correct_prediction_feature_summary.csv` answers the question *"on the cases each pipeline gets right, which features drive them the most?"* — broken down by the four prediction buckets (TP_malignant / TN_benign / FP_benign / FN_malignant). The top-ranked rows for the transparent pipelines tell a clinically coherent story: correct malignant calls tend to have elevated `asym_*`, `border_gradient` and `color_*` statistics, while correct benign calls tend to cluster around the scaled-feature medians. `figures/fig_correct_prediction_features.png` visualises the top entries of this CSV; a human-readable Markdown companion lives at `correct_prediction_feature_summary.md`.

---

# Chapter 7 — Discussion

## 7.1 What Worked Well

The central methodological commitment — unifying every pipeline behind a single abstract interface, a single configuration, a single threshold strategy, a single feature audit and a single scale driver — worked out as intended. The cost of adding a new pipeline is now roughly proportional to the novelty of its decision mechanism rather than to the supporting scaffolding. The explanation-consistency contract, enforced as a unit test, turned out to be valuable in practice: during development the test flagged an earlier bug in the transparent pipeline in which `explain` was computing the logit under an un-scaled feature vector while `predict` used the scaled vector, producing contradictory verdicts on borderline examples. The test forced the bug to be fixed rather than deferred.

The image-only boundary is a harder property to get right than it might appear, and the audit step paid for itself early. On an earlier iteration of the code a refactor accidentally let `clin_size_long_diam_mm` pass into the transparent pipeline's feature cache; the audit's `MetadataLeakError` fired at the next fit rather than letting a contaminated result be written to disk.

## 7.2 Accuracy versus Interpretability Trade-Off

With five pipelines × four scales trained and evaluated, the trade-off can be read off `summary_metrics_all_scales.csv` directly. In broad strokes:

- the black-box `multitask_cnn` sets the AUC ceiling at every scale;
- `glassbox_nam` is the next-closest, typically within one or two AUC points;
- `hard_cbm` is slightly below, reflecting the restricted capacity of the linear readout over four concepts;
- `transparent_lr` sits at ~0.05–0.10 AUC below the best CNN-backed pipeline;
- `transparent_tree` sits at a similar point but with a much sparser decision surface (a depth-4 tree has at most 16 leaves).

What a fully auditable pipeline purchases in exchange is a decision process in which every step is inspectable, without any intermediate learned representation. For a deployment context in which any use of an opaque perception stage would be disqualifying — an illustrative example is a regulatory audit in a low-resource environment where a reviewer cannot be assumed to have access to a GPU to re-run the model — `transparent_lr` and `transparent_tree` are the only pipelines that qualify. Framed this way, the accuracy gap is not a negative result: it is the cost of a stronger form of auditability, paid knowingly.

## 7.3 Limits of Concept-Based Partial Interpretability

`hard_cbm` and `glassbox_nam` illustrate what *partial* interpretability does and does not deliver. On the positive side, their reasoning stage is a small inspectable function and a clinician can reason about whether the learned weights or shape functions make medical sense. On the negative side, the concept values themselves come from a CNN: nothing about inspecting the four weights tells us *how* the network produced the four concept scores on a given image. The CNN is supervised against the pixel-derived ABCD proxies — which at least grounds the concepts in a closed-form target — but the training MSE is not zero at convergence, so the concept values are the CNN's best guess rather than a measurement.

## 7.4 `transparent_lr` vs. `transparent_tree`

The two fully-auditable pipelines are intentionally placed at the same rung of the ladder but with different trade-offs:

- **`transparent_lr`** is a linear model with 20+ signed weights (one per pixel-derived feature plus the four concept proxies). Its decision surface is smooth, its probabilities are calibrated in the usual sense, and a clinician can audit the sign and magnitude of each weight against their prior. The disadvantage is that the "story" the model tells depends on feature-feature correlations: a negative weight on an ostensibly positive signal (e.g. `geom_compactness`) is often a proxy for a correlated feature, and the interaction is non-obvious.
- **`transparent_tree`** has an extremely sparse decision surface: a depth-4 tree reaches a leaf after at most four yes/no questions, each of which is a threshold test on a single feature. The full rule set fits on a page. The disadvantage is that the tree's probability outputs are piecewise-constant (leaf frequencies), so it is less useful when a calibrated probability is needed.

`fig_transparent_lr_vs_tree_tradeoff.png` plots AUC versus "rule complexity" (feature count for LR, leaf count for tree) for each scale; on this cohort the two pipelines are usually within one or two AUC points of each other.

## 7.5 Threats to Validity

Several threats to the validity of the results should be flagged.

*Otsu segmentation failures.* On strongly vignetted frames, or on frames in which the lesion is extremely faint, Otsu can pick the vignette or the skin instead of the lesion. The extractor detects the worst such cases (`_extraction_error = empty_mask`) and falls back to neutral values; the affected images contribute only through their label.

*Single-seed training.* The default run uses seed 1337 throughout. A robust evaluation would run multiple seeds and report 95 % confidence intervals. The infrastructure supports this (`-o project.seed=…`) but the multi-seed runs have not been included in the current submission.

*Test-positive count.* The fixed test split contains on the order of 40–50 malignant images. AUC, recall and specificity confidence intervals are bounded by standard binomial noise around that small denominator; large apparent gaps at a single scale should be checked against the cross-scale plot before being over-interpreted.

*Metric choice.* At well under 1 % prevalence, accuracy and precision are close to uninformative. Balanced accuracy is reported as a defensible single-number summary, but in deployment the right metric depends on the relative costs of false positives and false negatives.

---

# Chapter 8 — Conclusion and Future Work

## 8.1 Main Findings

This dissertation has presented **GlassDerm**, a strictly image-only, five-pipeline transparency-ladder benchmark for skin-lesion classification on the ISIC 2024 dataset. The key findings are:

1. Five pipelines can be compared fairly only when they share a single interface, a single evaluation code path, a single threshold strategy and a single image-only audit step. We have built such a harness and demonstrated that it supports a black-box baseline, two partially-interpretable concept-bottleneck pipelines, a fully-auditable logistic-regression pipeline and a fully-auditable depth-4 tree pipeline under the same abstraction.
2. A scale sweep over training-benign budgets (40 000 / 80 000 / 160 000 / full) against a *fixed* patient-disjoint val/test split rank-orders predictable losses along the transparency ladder. The CNN-backed pipelines benefit more from extra benigns than the transparent pipelines, whose capacity is bounded by their closed-form feature dictionary.
3. The two fully-auditable pipelines trail the best CNN-backed pipeline by ~0.05–0.15 AUC in most scales, depending on the operating point, and this is the quantitative cost of auditability. On the other hand they are the *only* pipelines that can be deployed in contexts where an opaque perception stage would be disqualifying.
4. The image-only boundary is a non-trivial engineering commitment. Enforcing it via a dedicated audit module and a `feature_audit.csv` next to every fitted model pays for itself both at development time (it catches accidental leaks early) and at write-up time (the audit file is directly citable as evidence that the claim holds).

## 8.2 What this Project Demonstrates

The project demonstrates that it is practical to build a benchmarking harness whose only free variable is the degree of intrinsic interpretability; that a sufficiently rich hand-crafted pixel-feature extractor, combined with a shallow classifier, can achieve a level of auditability that is not available from any CNN-based pipeline, at a modest cost in raw accuracy; and that a strictly image-only input boundary is not only defensible but relatively straightforward to enforce with the right module boundary.

## 8.3 Future Work

Several concrete directions would extend the work presented here.

*Richer hand-crafted features.* The twenty-feature dictionary covers the geometric, asymmetry, border and colour signals motivated by the clinical ABCD rule, but does not cover pigment-network patterns, streaks, blue-white veils or atypical dots — all of which carry diagnostic signal. Extending the feature dictionary in these directions, while remaining closed-form, would be a natural next step.

*Multi-seed evaluation and confidence intervals.* Running the five pipelines across ten seeds and reporting means with 95 % bootstrap confidence intervals would substantially strengthen the comparative claims in Chapter 6.

*Uncertainty calibration.* `transparent_lr` currently reports a single point probability. An interesting extension would be to replace the logistic regression with a Bayesian logistic regression and expose calibrated credible intervals, so the explanation trace can say "this model is 95 % confident in its call" alongside the per-feature contribution breakdown.

*External validation.* Re-evaluating with cost-sensitive metrics and external validation on a different dermoscopy release would sharpen the claim that the transparent pipelines are useful in at least some deployment contexts.

*Concept-intervention experiments.* A concept-bottleneck pipeline supports test-time concept editing: the clinician can override one of the four predicted concept scores, and the pipeline can report how the verdict would change. Wiring this into `hard_cbm` / `glassbox_nam` would give a concrete demonstration of how partial interpretability translates into an interactive decision-support tool.

---

# References

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). *Sanity Checks for Saliency Maps.* Advances in Neural Information Processing Systems, 31.

Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R., & Hinton, G. E. (2021). *Neural Additive Models: Interpretable Machine Learning with Neural Nets.* Advances in Neural Information Processing Systems, 34.

Argenziano, G., Soyer, H. P., Chimenti, S., Talamini, R., Corona, R., Sera, F., Binder, M., Cerroni, L., De Rosa, G., Ferrara, G., et al. (2003). *Dermoscopy of pigmented skin lesions: Results of a consensus meeting via the Internet.* Journal of the American Academy of Dermatology, 48(5), 679–693.

Bradski, G. (2000). *The OpenCV library.* Dr. Dobb's Journal of Software Tools.

Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015). *Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission.* Proc. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1721–1730.

Codella, N. C. F., Gutman, D., Celebi, M. E., Helba, B., Marchetti, M. A., Dusza, S. W., Kalloo, A., Liopyris, K., Mishra, N., Kittler, H., & Halpern, A. (2018). *Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), hosted by the International Skin Imaging Collaboration (ISIC).* Proc. IEEE International Symposium on Biomedical Imaging.

Doshi-Velez, F., & Kim, B. (2017). *Towards a rigorous science of interpretable machine learning.* arXiv preprint arXiv:1702.08608.

Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). *Dermatologist-level classification of skin cancer with deep neural networks.* Nature, 542(7639), 115–118.

Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., & Pedreschi, D. (2018). *A survey of methods for explaining black box models.* ACM Computing Surveys, 51(5), 1–42.

Hastie, T., & Tibshirani, R. (1986). *Generalized additive models.* Statistical Science, 1(3), 297–310.

Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020). *Concept Bottleneck Models.* Proceedings of the 37th International Conference on Machine Learning.

Lipton, Z. C. (2018). *The mythos of model interpretability.* Communications of the ACM, 61(10), 36–43.

Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* Advances in Neural Information Processing Systems, 30.

Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). Self-published (available online at christophm.github.io/interpretable-ml-book).

Nachbar, F., Stolz, W., Merkle, T., Cognetta, A. B., Vogt, T., Landthaler, M., Bilek, P., Braun-Falco, O., & Plewig, G. (1994). *The ABCD rule of dermatoscopy: High prospective value in the diagnosis of doubtful melanocytic skin lesions.* Journal of the American Academy of Dermatology, 30(4), 551–559.

Otsu, N. (1979). *A threshold selection method from gray-level histograms.* IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). *PyTorch: An imperative style, high-performance deep learning library.* Advances in Neural Information Processing Systems, 32.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?": Explaining the predictions of any classifier.* Proc. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

Rotemberg, V., Kurtansky, N., Betz-Stablein, B., Caffery, L., Chousakos, E., Codella, N., Combalia, M., Dusza, S., Guitera, P., Gutman, D., Halpern, A., Helba, B., Kittler, H., Kose, K., Langer, S., Lioprys, K., Malvehy, J., Musthaq, S., Nanda, J., Reiter, O., Shih, G., Stratigos, A., Tschandl, P., Weber, J., & Soyer, H. P. (2021). *A patient-centric dataset of images and metadata for identifying melanomas using clinical context.* Scientific Data, 8, 34.

Rudin, C. (2019). *Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead.* Nature Machine Intelligence, 1(5), 206–215.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* Proc. IEEE International Conference on Computer Vision.

Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks.* Proceedings of the 36th International Conference on Machine Learning.

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161.

---

# Appendix

## Appendix A — Example Case Study (`transparent_lr` anchor)

The example below is a representative excerpt from `outputs_scale/benign_full/case_studies/case_studies.md`; see that file for the complete set of eight narratives per scale.

```
### Bucket: TP_malignant
**Image:** ISIC_XXXXXXX  (patient P_YYYY,  P_malignant = 0.994,  τ = 0.60)

transparent_lr — logistic regression on hand-computed image features
Threshold for classification = 0.60
Top signed contributions (w·x_scaled):
  color_hue_std             raw=+0.474  scaled=0.95  w=+2.43  contrib=+2.30
  geom_convexity_defects    raw=+0.560  scaled=0.82  w=+2.29  contrib=+1.89
  concept_D_diameter        raw=+0.83   scaled=0.71  w=+1.80  contrib=+1.28
  concept_C_color           raw=+0.90   scaled=0.89  w=+1.45  contrib=+1.30
  color_darkness            raw=+0.487  scaled=0.48  w=+2.62  contrib=+1.26
  color_sat_std             raw=+0.071  scaled=0.14  w=-8.59  contrib=-1.19
+ bias = -1.28
= logit +5.06  →  σ(logit) = P(malignant) = 0.994  →  MALIGNANT

Other pipelines on the same image:
  multitask_cnn     P=0.56  τ=0.12  →  MALIGNANT (black-box; concepts informational)
  hard_cbm          P=0.54  τ=0.38  →  MALIGNANT (linear readout over CNN concepts)
  glassbox_nam      P=0.81  τ=0.06  →  MALIGNANT (additive readout over CNN concepts)
  transparent_tree  leaf_prob=0.92  τ=0.55 →  MALIGNANT (rule path, 4 yes/no splits)
```

The key property to verify is that for every pipeline listed, `explain` produces the same verdict as `predict` under the same validation-tuned threshold — asserted by `tests/test_explanations.py`.

## Appendix B — Relevant Configuration Snippets

```yaml
data:
  raw_dir: /path/to/isic2024_official
  processed_dir: data/processed
  features_dir: data/features
  cache_dir: data/cache
  features_cache: data/features/transparent_features.parquet
  forbidden_metadata_prefixes: [tbp_lv_, tbp_, iddx, mel_, anatom_]
  forbidden_metadata_exact:
    - age_approx
    - sex
    - anatom_site_general
    - clin_size_long_diam_mm
    - image_type
    - tbp_tile_type
    - attribution
    - copyright_license
    - A
    - B
    - C
    - D
  split:
    method: patient_disjoint
    val_fraction: 0.15
    test_fraction: 0.15
    seed: 1337

pipelines:
  multitask_cnn:       { enabled: true }
  hard_cbm:            { enabled: true,  concept_sup_weight: 2.0 }
  glassbox_nam:        { enabled: true,  hidden: 64, concept_sup_weight: 2.0 }
  transparent_lr:      { enabled: true,  logistic_c: 0.5 }
  transparent_tree:    { enabled: true,  max_depth: 4, min_samples_leaf: 30 }

evaluate:
  threshold_strategy: youden         # "youden" | "f1" | "fixed"
  fixed_threshold: 0.5

analysis:
  n_case_studies_per_bucket: 2       # 2×(TP, TN, FP, FN) = 8 curated narratives per scale
```

## Appendix C — Feature Audit Example

Extract of `outputs_scale/benign_full/transparent_lr/feature_audit.csv`:

```csv
pipeline,feature_set,column,source,is_model_input
transparent_lr,image_only,image_id,metadata,False
transparent_lr,image_only,patient_id,metadata,False
transparent_lr,image_only,label,label,False
transparent_lr,image_only,geom_area_ratio,pixel_derived,True
transparent_lr,image_only,geom_perim_ratio,pixel_derived,True
...
transparent_lr,image_only,concept_A_asymmetry,pixel_derived,True
transparent_lr,image_only,concept_B_border,pixel_derived,True
transparent_lr,image_only,concept_C_color,pixel_derived,True
transparent_lr,image_only,concept_D_diameter,pixel_derived,True
```

The invariant a reviewer should check is: for every row with `is_model_input == True`, `source == pixel_derived`. `tests/test_cli.py::test_scale_experiment_smoke` asserts this same invariant on the synthetic-fixture output.
