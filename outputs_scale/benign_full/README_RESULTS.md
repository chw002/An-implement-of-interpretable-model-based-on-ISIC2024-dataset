# Scale: benign_full

- Training budget: **full (all benign)**, every malignant kept.
- Val/test split: patient-disjoint, fixed across all scales.
- Features: pixel-derived only (see `feature_audit.csv`).
- Thresholds tuned on val set only (`tables/thresholds.csv`).

## Test metrics

```
        pipeline      transparency_tag    auc  balanced_accuracy  recall  specificity  precision     f1  threshold    tn    fp  fn  tp
   multitask_cnn   image_only_baseline 0.8351             0.7148  0.5172       0.9124     0.0061 0.0120     0.0111 25587  2457  14  15
        hard_cbm interpretable_partial 0.6964             0.6439  0.6552       0.6327     0.0018 0.0037     0.1770 17744 10300  10  19
    glassbox_nam interpretable_partial 0.8604             0.7318  0.5517       0.9119     0.0064 0.0127     0.0179 25572  2472  13  16
  transparent_lr       fully_auditable 0.7380             0.7141  0.6207       0.8074     0.0033 0.0066     0.5265 22644  5400  11  18
transparent_tree       fully_auditable 0.7741             0.6629  0.3793       0.9464     0.0073 0.0143     0.8462 26542  1502  18  11
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