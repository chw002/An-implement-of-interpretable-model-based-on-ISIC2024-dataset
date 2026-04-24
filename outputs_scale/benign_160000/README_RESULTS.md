# Scale: benign_160000

- Training budget: **160000 benign**, every malignant kept.
- Val/test split: patient-disjoint, fixed across all scales.
- Features: pixel-derived only (see `feature_audit.csv`).
- Thresholds tuned on val set only (`tables/thresholds.csv`).

## Test metrics

```
        pipeline      transparency_tag    auc  balanced_accuracy  recall  specificity  precision     f1  threshold    tn   fp  fn  tp
   multitask_cnn   image_only_baseline 0.9238             0.8504  0.9310       0.7697     0.0042 0.0083     0.0007 21585 6459   2  27
        hard_cbm interpretable_partial 0.8711             0.8089  0.6897       0.9281     0.0098 0.0194     0.2036 26027 2017   9  20
    glassbox_nam interpretable_partial 0.7880             0.6719  0.4483       0.8956     0.0044 0.0088     0.0204 25116 2928  16  13
  transparent_lr       fully_auditable 0.7380             0.7141  0.6207       0.8074     0.0033 0.0066     0.5265 22644 5400  11  18
transparent_tree       fully_auditable 0.7741             0.6629  0.3793       0.9464     0.0073 0.0143     0.8462 26542 1502  18  11
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