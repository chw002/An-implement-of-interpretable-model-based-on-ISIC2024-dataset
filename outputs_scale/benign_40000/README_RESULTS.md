# Scale: benign_40000

- Training budget: **40000 benign**, every malignant kept.
- Val/test split: patient-disjoint, fixed across all scales.
- Features: pixel-derived only (see `feature_audit.csv`).
- Thresholds tuned on val set only (`tables/thresholds.csv`).

## Test metrics

```
        pipeline      transparency_tag    auc  balanced_accuracy  recall  specificity  precision     f1  threshold    tn   fp  fn  tp
   multitask_cnn   image_only_baseline 0.8624             0.7557  0.6552       0.8563     0.0047 0.0093     0.0099 24013 4031  10  19
        hard_cbm interpretable_partial 0.8586             0.7354  0.6897       0.7812     0.0032 0.0065     0.3040 21908 6136   9  20
    glassbox_nam interpretable_partial 0.8924             0.7966  0.6897       0.9036     0.0073 0.0145     0.1088 25341 2703   9  20
  transparent_lr       fully_auditable 0.7229             0.6521  0.4138       0.8904     0.0039 0.0077     0.5902 24971 3073  17  12
transparent_tree       fully_auditable 0.7302             0.6620  0.3793       0.9446     0.0070 0.0138     0.8406 26491 1553  18  11
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