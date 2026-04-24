# Correct-Prediction Feature Summary

For each pipeline and each of the four prediction buckets, this table lists the pixel-derived image features that were highest (by median) across the bucket's test images. All feature values are normalised to [0, 1] by `glassderm.data.features`, so medians are directly comparable.

Only the top 10 features per bucket are shown; see `correct_prediction_feature_summary.csv` for the full ranking.

**Bucket legend**

- `TP_malignant` — True positives — malignant lesions the pipeline correctly flagged.
- `TN_benign` — True negatives — benign lesions the pipeline correctly passed.
- `FP_benign` — False positives — benign lesions mistakenly flagged as malignant.
- `FN_malignant` — False negatives — malignant lesions the pipeline missed.

## Pipeline: `glassbox_nam`

### TP_malignant  (n = 16)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.499 | 1.945 |
| 2 | `geom_solidity` | 0.879 | 0.852 |
| 3 | `geom_compactness` | 0.860 | 1.731 |
| 4 | `geom_eccentricity` | 0.640 | 0.653 |
| 5 | `color_darkness` | 0.471 | 0.450 |
| 6 | `asym_diagonal` | 0.359 | 0.445 |
| 7 | `concept_B_border` | 0.320 | 0.426 |
| 8 | `geom_convexity_defects` | 0.260 | 0.270 |
| 9 | `concept_A_asymmetry` | 0.252 | 0.269 |
| 10 | `concept_D_diameter` | 0.225 | 0.250 |

### TN_benign  (n = 25572)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.891 | 0.858 |
| 2 | `geom_eccentricity` | 0.760 | 0.729 |
| 3 | `geom_compactness` | 0.615 | 1.140 |
| 4 | `geom_perim_ratio` | 0.530 | 0.830 |
| 5 | `asym_diagonal` | 0.492 | 0.573 |
| 6 | `color_darkness` | 0.359 | 0.375 |
| 7 | `concept_A_asymmetry` | 0.332 | 0.337 |
| 8 | `concept_B_border` | 0.262 | 0.352 |
| 9 | `asym_vertical` | 0.248 | 0.278 |
| 10 | `asym_horizontal` | 0.246 | 0.277 |

### FP_benign  (n = 2472)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 2.686 | 2.497 |
| 2 | `geom_compactness` | 1.281 | 1.722 |
| 3 | `geom_solidity` | 0.871 | 0.862 |
| 4 | `geom_eccentricity` | 0.758 | 0.730 |
| 5 | `geom_area_ratio` | 0.528 | 0.414 |
| 6 | `concept_D_diameter` | 0.427 | 0.361 |
| 7 | `color_darkness` | 0.408 | 0.420 |
| 8 | `concept_B_border` | 0.404 | 0.465 |
| 9 | `asym_diagonal` | 0.370 | 0.409 |
| 10 | `concept_A_asymmetry` | 0.223 | 0.241 |

### FN_malignant  (n = 13)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.922 | 0.886 |
| 2 | `asym_diagonal` | 0.768 | 0.669 |
| 3 | `geom_eccentricity` | 0.715 | 0.696 |
| 4 | `geom_compactness` | 0.429 | 0.638 |
| 5 | `color_darkness` | 0.393 | 0.389 |
| 6 | `concept_A_asymmetry` | 0.343 | 0.350 |
| 7 | `geom_perim_ratio` | 0.340 | 0.495 |
| 8 | `color_n_regions` | 0.333 | 0.333 |
| 9 | `color_variegation` | 0.321 | 0.244 |
| 10 | `asym_vertical` | 0.241 | 0.280 |

## Pipeline: `hard_cbm`

### TP_malignant  (n = 19)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.921 | 0.874 |
| 2 | `geom_perim_ratio` | 0.828 | 1.038 |
| 3 | `geom_eccentricity` | 0.770 | 0.729 |
| 4 | `geom_compactness` | 0.482 | 0.900 |
| 5 | `asym_diagonal` | 0.392 | 0.510 |
| 6 | `color_darkness` | 0.388 | 0.391 |
| 7 | `concept_A_asymmetry` | 0.343 | 0.307 |
| 8 | `color_n_regions` | 0.333 | 0.307 |
| 9 | `concept_B_border` | 0.240 | 0.321 |
| 10 | `asym_vertical` | 0.217 | 0.259 |

### TN_benign  (n = 17744)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_compactness` | 0.950 | 1.510 |
| 2 | `geom_solidity` | 0.846 | 0.828 |
| 3 | `geom_eccentricity` | 0.776 | 0.739 |
| 4 | `asym_diagonal` | 0.774 | 0.690 |
| 5 | `geom_perim_ratio` | 0.557 | 1.093 |
| 6 | `color_darkness` | 0.376 | 0.397 |
| 7 | `concept_A_asymmetry` | 0.367 | 0.381 |
| 8 | `concept_B_border` | 0.352 | 0.426 |
| 9 | `asym_vertical` | 0.277 | 0.304 |
| 10 | `asym_horizontal` | 0.275 | 0.303 |

### FP_benign  (n = 10300)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.940 | 0.911 |
| 2 | `geom_eccentricity` | 0.735 | 0.712 |
| 3 | `geom_perim_ratio` | 0.575 | 0.778 |
| 4 | `geom_compactness` | 0.390 | 0.640 |
| 5 | `color_darkness` | 0.339 | 0.347 |
| 6 | `color_n_regions` | 0.333 | 0.311 |
| 7 | `asym_diagonal` | 0.234 | 0.332 |
| 8 | `concept_A_asymmetry` | 0.207 | 0.238 |
| 9 | `concept_B_border` | 0.203 | 0.251 |
| 10 | `asym_vertical` | 0.184 | 0.214 |

### FN_malignant  (n = 10)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.879 | 0.855 |
| 2 | `geom_compactness` | 0.751 | 1.888 |
| 3 | `asym_diagonal` | 0.620 | 0.613 |
| 4 | `geom_eccentricity` | 0.535 | 0.565 |
| 5 | `color_darkness` | 0.466 | 0.482 |
| 6 | `geom_perim_ratio` | 0.407 | 1.782 |
| 7 | `concept_A_asymmetry` | 0.316 | 0.302 |
| 8 | `concept_B_border` | 0.281 | 0.397 |
| 9 | `asym_vertical` | 0.188 | 0.245 |
| 10 | `color_n_regions` | 0.167 | 0.250 |

## Pipeline: `multitask_cnn`

### TP_malignant  (n = 15)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 0.950 | 1.366 |
| 2 | `geom_compactness` | 0.898 | 1.151 |
| 3 | `geom_solidity` | 0.880 | 0.852 |
| 4 | `geom_eccentricity` | 0.721 | 0.666 |
| 5 | `color_darkness` | 0.435 | 0.416 |
| 6 | `asym_diagonal` | 0.392 | 0.496 |
| 7 | `concept_A_asymmetry` | 0.343 | 0.300 |
| 8 | `concept_B_border` | 0.314 | 0.375 |
| 9 | `border_radial_variance` | 0.227 | 0.215 |
| 10 | `asym_horizontal` | 0.203 | 0.239 |

### TN_benign  (n = 25587)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.890 | 0.859 |
| 2 | `geom_eccentricity` | 0.761 | 0.729 |
| 3 | `geom_compactness` | 0.650 | 1.178 |
| 4 | `geom_perim_ratio` | 0.561 | 0.953 |
| 5 | `asym_diagonal` | 0.458 | 0.559 |
| 6 | `color_darkness` | 0.360 | 0.377 |
| 7 | `concept_A_asymmetry` | 0.322 | 0.328 |
| 8 | `concept_B_border` | 0.272 | 0.360 |
| 9 | `asym_vertical` | 0.240 | 0.271 |
| 10 | `asym_horizontal` | 0.238 | 0.270 |

### FP_benign  (n = 2457)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.875 | 0.853 |
| 2 | `geom_compactness` | 0.757 | 1.321 |
| 3 | `geom_eccentricity` | 0.750 | 0.725 |
| 4 | `geom_perim_ratio` | 0.697 | 1.227 |
| 5 | `asym_diagonal` | 0.430 | 0.550 |
| 6 | `color_darkness` | 0.388 | 0.397 |
| 7 | `color_n_regions` | 0.333 | 0.294 |
| 8 | `concept_A_asymmetry` | 0.319 | 0.326 |
| 9 | `concept_B_border` | 0.303 | 0.387 |
| 10 | `asym_vertical` | 0.243 | 0.272 |

### FN_malignant  (n = 14)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.922 | 0.884 |
| 2 | `geom_eccentricity` | 0.663 | 0.679 |
| 3 | `asym_diagonal` | 0.558 | 0.598 |
| 4 | `geom_perim_ratio` | 0.486 | 1.218 |
| 5 | `geom_compactness` | 0.442 | 1.338 |
| 6 | `color_darkness` | 0.432 | 0.429 |
| 7 | `concept_A_asymmetry` | 0.336 | 0.311 |
| 8 | `color_n_regions` | 0.250 | 0.298 |
| 9 | `concept_B_border` | 0.213 | 0.318 |
| 10 | `asym_vertical` | 0.207 | 0.246 |

## Pipeline: `transparent_lr`

### TP_malignant  (n = 18)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.378 | 1.517 |
| 2 | `geom_solidity` | 0.879 | 0.860 |
| 3 | `geom_compactness` | 0.860 | 1.199 |
| 4 | `geom_eccentricity` | 0.594 | 0.620 |
| 5 | `color_darkness` | 0.468 | 0.481 |
| 6 | `asym_diagonal` | 0.343 | 0.438 |
| 7 | `concept_B_border` | 0.296 | 0.377 |
| 8 | `concept_A_asymmetry` | 0.255 | 0.271 |
| 9 | `geom_convexity_defects` | 0.240 | 0.256 |
| 10 | `concept_D_diameter` | 0.188 | 0.220 |

### TN_benign  (n = 22644)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.892 | 0.857 |
| 2 | `geom_eccentricity` | 0.776 | 0.748 |
| 3 | `geom_compactness` | 0.619 | 1.140 |
| 4 | `geom_perim_ratio` | 0.530 | 0.841 |
| 5 | `asym_diagonal` | 0.480 | 0.574 |
| 6 | `color_darkness` | 0.346 | 0.361 |
| 7 | `concept_A_asymmetry` | 0.334 | 0.340 |
| 8 | `concept_B_border` | 0.265 | 0.352 |
| 9 | `asym_vertical` | 0.250 | 0.282 |
| 10 | `asym_horizontal` | 0.249 | 0.282 |

### FP_benign  (n = 5400)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 0.972 | 1.551 |
| 2 | `geom_solidity` | 0.879 | 0.865 |
| 3 | `geom_compactness` | 0.855 | 1.402 |
| 4 | `geom_eccentricity` | 0.683 | 0.650 |
| 5 | `color_darkness` | 0.421 | 0.454 |
| 6 | `asym_diagonal` | 0.407 | 0.493 |
| 7 | `color_n_regions` | 0.333 | 0.301 |
| 8 | `concept_B_border` | 0.322 | 0.404 |
| 9 | `concept_A_asymmetry` | 0.260 | 0.278 |
| 10 | `asym_vertical` | 0.207 | 0.226 |

### FN_malignant  (n = 11)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `asym_diagonal` | 1.000 | 0.721 |
| 2 | `geom_solidity` | 0.932 | 0.880 |
| 3 | `geom_eccentricity` | 0.770 | 0.758 |
| 4 | `geom_compactness` | 0.438 | 1.309 |
| 5 | `concept_A_asymmetry` | 0.361 | 0.361 |
| 6 | `geom_perim_ratio` | 0.340 | 0.931 |
| 7 | `color_darkness` | 0.328 | 0.327 |
| 8 | `asym_vertical` | 0.270 | 0.280 |
| 9 | `border_radial_variance` | 0.227 | 0.233 |
| 10 | `concept_B_border` | 0.217 | 0.299 |

## Pipeline: `transparent_tree`

### TP_malignant  (n = 11)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.682 | 2.171 |
| 2 | `geom_compactness` | 1.578 | 1.644 |
| 3 | `geom_solidity` | 0.850 | 0.842 |
| 4 | `geom_eccentricity` | 0.649 | 0.686 |
| 5 | `concept_B_border` | 0.493 | 0.486 |
| 6 | `color_darkness` | 0.438 | 0.456 |
| 7 | `geom_convexity_defects` | 0.360 | 0.327 |
| 8 | `asym_diagonal` | 0.336 | 0.338 |
| 9 | `concept_D_diameter` | 0.256 | 0.296 |
| 10 | `geom_area_ratio` | 0.247 | 0.308 |

### TN_benign  (n = 26542)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.890 | 0.859 |
| 2 | `geom_eccentricity` | 0.760 | 0.729 |
| 3 | `geom_compactness` | 0.630 | 1.147 |
| 4 | `geom_perim_ratio` | 0.537 | 0.918 |
| 5 | `asym_diagonal` | 0.465 | 0.566 |
| 6 | `color_darkness` | 0.361 | 0.373 |
| 7 | `concept_A_asymmetry` | 0.326 | 0.331 |
| 8 | `concept_B_border` | 0.266 | 0.351 |
| 9 | `asym_vertical` | 0.242 | 0.273 |
| 10 | `asym_horizontal` | 0.239 | 0.272 |

### FP_benign  (n = 1502)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.646 | 2.032 |
| 2 | `geom_compactness` | 1.499 | 1.971 |
| 3 | `geom_solidity` | 0.853 | 0.838 |
| 4 | `geom_eccentricity` | 0.762 | 0.723 |
| 5 | `concept_B_border` | 0.522 | 0.557 |
| 6 | `color_darkness` | 0.408 | 0.487 |
| 7 | `asym_diagonal` | 0.379 | 0.434 |
| 8 | `color_n_regions` | 0.333 | 0.294 |
| 9 | `border_radial_variance` | 0.246 | 0.259 |
| 10 | `concept_D_diameter` | 0.244 | 0.286 |

### FN_malignant  (n = 18)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.922 | 0.883 |
| 2 | `asym_diagonal` | 0.842 | 0.672 |
| 3 | `geom_eccentricity` | 0.705 | 0.664 |
| 4 | `geom_compactness` | 0.424 | 0.995 |
| 5 | `color_darkness` | 0.407 | 0.402 |
| 6 | `geom_perim_ratio` | 0.391 | 0.759 |
| 7 | `concept_A_asymmetry` | 0.350 | 0.337 |
| 8 | `color_n_regions` | 0.250 | 0.296 |
| 9 | `asym_vertical` | 0.229 | 0.263 |
| 10 | `asym_horizontal` | 0.209 | 0.244 |
