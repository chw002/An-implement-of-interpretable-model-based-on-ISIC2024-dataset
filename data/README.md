# `data/` — what lives here

| folder       | lifecycle                 | contents                                                         |
|--------------|---------------------------|------------------------------------------------------------------|
| `raw/`       | external (not in git)     | the official ISIC 2024 permissive release                         |
| `processed/` | built by `prepare-data`   | `cohort.csv`, `splits.json`, `concept_scaler.json`, provenance    |
| `features/`  | built by `prepare-data`   | cached transparent-pipeline features (parquet or CSV fallback)    |
| `cache/`     | scratch                   | transient intermediaries (safe to delete)                         |

The code in `src/glassderm/data/download.py` locates the raw data by, in
order:

1. the path configured via `configs/default.yaml` (`data.raw_dir`);
2. a short list of known fallback paths;
3. a fresh Kaggle download if credentials are available.

If none of those produces all four of

```
<raw>/ISIC_2024_Permissive_Training_GroundTruth.csv
<raw>/ISIC_2024_Permissive_Training_Supplement.csv
<raw>/ISIC_2024_Permissive_Training_Input/metadata.csv
<raw>/ISIC_2024_Permissive_Training_Input/ISIC_*.jpg
```

you will receive an error with detailed remediation steps, rather than a
silent failure.
