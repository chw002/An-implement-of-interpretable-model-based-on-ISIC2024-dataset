# Scale: benign_80000

- Training budget: **80000 benign**, every malignant kept.
- Val/test split: patient-disjoint, fixed across all scales.
- Features: pixel-derived only (see `feature_audit.csv`).
- Thresholds tuned on val set only (`tables/thresholds.csv`).

## Test metrics

```
        pipeline      transparency_tag    auc  balanced_accuracy  recall  specificity  precision     f1  threshold    tn   fp  fn  tp
   multitask_cnn   image_only_baseline 0.9054             0.7864  0.6552       0.9176     0.0082 0.0161     0.0161 25733 2311  10  19
        hard_cbm interpretable_partial 0.8544             0.7634  0.5862       0.9405     0.0101 0.0198     0.2889 26376 1668  12  17
    glassbox_nam interpretable_partial 0.8865             0.8108  0.7241       0.8974     0.0072 0.0143     0.0290 25166 2878   8  21
  transparent_lr       fully_auditable 0.7323             0.6666  0.4483       0.8849     0.0040 0.0080     0.5862 24817 3227  16  13
transparent_tree       fully_auditable 0.7741             0.6629  0.3793       0.9464     0.0073 0.0143     0.8474 26541 1503  18  11
```

## Files

- `cohort_provenance.json` – row counts per split + malignant/benign.
- `splits.json` – exact image_id lists per split.
- `config_snapshot.yaml` – full config used for this scale.
- `tables/main_metrics.csv`, `tables/thresholds.csv` – numeric results.
- `<pipeline>/feature_audit.csv` + `feature_audit.json` – no-metadata contract.
- `<pipeline>/predictions_test.csv` – per-image test probabilities.
- `case_studies/` – human-readable case narratives.
- `figures/` – per-scale figures for dissertation.