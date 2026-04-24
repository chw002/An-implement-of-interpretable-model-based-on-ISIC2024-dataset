# Correct-Prediction Feature Summary

For each pipeline and each of the four prediction buckets, this table lists the pixel-derived image features that were highest (by median) across the bucket's test images. All feature values are normalised to [0, 1] by `glassderm.data.features`, so medians are directly comparable.

Only the top 10 features per bucket are shown; see `correct_prediction_feature_summary.csv` for the full ranking.

**Bucket legend**

- `TP_malignant` — True positives — malignant lesions the pipeline correctly flagged.
- `TN_benign` — True negatives — benign lesions the pipeline correctly passed.
- `FP_benign` — False positives — benign lesions mistakenly flagged as malignant.
- `FN_malignant` — False negatives — malignant lesions the pipeline missed.

## Pipeline: `glassbox_nam`

### TP_malignant  (n = 13)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.488 | 1.677 |
| 2 | `geom_solidity` | 0.894 | 0.878 |
| 3 | `geom_compactness` | 0.721 | 1.196 |
| 4 | `geom_eccentricity` | 0.632 | 0.642 |
| 5 | `color_darkness` | 0.445 | 0.441 |
| 6 | `geom_convexity_defects` | 0.320 | 0.295 |
| 7 | `asym_diagonal` | 0.290 | 0.385 |
| 8 | `concept_B_border` | 0.270 | 0.370 |
| 9 | `concept_D_diameter` | 0.203 | 0.239 |
| 10 | `concept_A_asymmetry` | 0.196 | 0.230 |

### TN_benign  (n = 25116)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.880 | 0.852 |
| 2 | `geom_eccentricity` | 0.766 | 0.733 |
| 3 | `geom_compactness` | 0.682 | 1.239 |
| 4 | `geom_perim_ratio` | 0.523 | 0.917 |
| 5 | `asym_diagonal` | 0.515 | 0.590 |
| 6 | `color_darkness` | 0.363 | 0.379 |
| 7 | `concept_A_asymmetry` | 0.338 | 0.345 |
| 8 | `concept_B_border` | 0.282 | 0.372 |
| 9 | `asym_vertical` | 0.255 | 0.284 |
| 10 | `asym_horizontal` | 0.253 | 0.283 |

### FP_benign  (n = 2928)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.097 | 1.498 |
| 2 | `geom_solidity` | 0.928 | 0.910 |
| 3 | `geom_eccentricity` | 0.713 | 0.693 |
| 4 | `geom_compactness` | 0.516 | 0.780 |
| 5 | `color_darkness` | 0.360 | 0.375 |
| 6 | `color_n_regions` | 0.333 | 0.298 |
| 7 | `asym_diagonal` | 0.238 | 0.292 |
| 8 | `concept_B_border` | 0.236 | 0.278 |
| 9 | `geom_convexity_defects` | 0.200 | 0.200 |
| 10 | `concept_D_diameter` | 0.185 | 0.252 |

### FN_malignant  (n = 16)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.907 | 0.859 |
| 2 | `geom_eccentricity` | 0.706 | 0.697 |
| 3 | `asym_diagonal` | 0.706 | 0.676 |
| 4 | `geom_compactness` | 0.464 | 1.278 |
| 5 | `color_darkness` | 0.394 | 0.408 |
| 6 | `geom_perim_ratio` | 0.374 | 0.984 |
| 7 | `concept_A_asymmetry` | 0.367 | 0.366 |
| 8 | `asym_vertical` | 0.287 | 0.300 |
| 9 | `color_n_regions` | 0.250 | 0.292 |
| 10 | `border_radial_variance` | 0.240 | 0.237 |

## Pipeline: `hard_cbm`

### TP_malignant  (n = 20)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 0.935 | 1.367 |
| 2 | `geom_solidity` | 0.887 | 0.874 |
| 3 | `geom_compactness` | 0.700 | 1.071 |
| 4 | `geom_eccentricity` | 0.682 | 0.661 |
| 5 | `color_darkness` | 0.436 | 0.431 |
| 6 | `asym_diagonal` | 0.344 | 0.454 |
| 7 | `concept_B_border` | 0.274 | 0.349 |
| 8 | `concept_A_asymmetry` | 0.272 | 0.275 |
| 9 | `geom_convexity_defects` | 0.220 | 0.236 |
| 10 | `asym_horizontal` | 0.195 | 0.223 |

### TN_benign  (n = 26027)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.883 | 0.854 |
| 2 | `geom_eccentricity` | 0.765 | 0.733 |
| 3 | `geom_compactness` | 0.692 | 1.229 |
| 4 | `geom_perim_ratio` | 0.542 | 0.969 |
| 5 | `asym_diagonal` | 0.486 | 0.579 |
| 6 | `color_darkness` | 0.363 | 0.378 |
| 7 | `concept_A_asymmetry` | 0.331 | 0.338 |
| 8 | `concept_B_border` | 0.284 | 0.370 |
| 9 | `asym_vertical` | 0.248 | 0.278 |
| 10 | `asym_horizontal` | 0.246 | 0.277 |

### FP_benign  (n = 2017)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.932 | 0.909 |
| 2 | `geom_perim_ratio` | 0.895 | 1.083 |
| 3 | `geom_eccentricity` | 0.684 | 0.669 |
| 4 | `geom_compactness` | 0.429 | 0.695 |
| 5 | `color_darkness` | 0.366 | 0.381 |
| 6 | `color_n_regions` | 0.333 | 0.298 |
| 7 | `concept_B_border` | 0.208 | 0.259 |
| 8 | `asym_diagonal` | 0.203 | 0.299 |
| 9 | `geom_convexity_defects` | 0.200 | 0.210 |
| 10 | `concept_A_asymmetry` | 0.170 | 0.205 |

### FN_malignant  (n = 9)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.921 | 0.852 |
| 2 | `asym_diagonal` | 0.915 | 0.748 |
| 3 | `geom_eccentricity` | 0.694 | 0.699 |
| 4 | `geom_compactness` | 0.418 | 1.619 |
| 5 | `geom_perim_ratio` | 0.406 | 1.135 |
| 6 | `color_darkness` | 0.394 | 0.404 |
| 7 | `concept_A_asymmetry` | 0.361 | 0.372 |
| 8 | `color_n_regions` | 0.333 | 0.278 |
| 9 | `asym_vertical` | 0.270 | 0.289 |
| 10 | `asym_horizontal` | 0.237 | 0.267 |

## Pipeline: `multitask_cnn`

### TP_malignant  (n = 27)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.893 | 0.862 |
| 2 | `geom_perim_ratio` | 0.816 | 1.354 |
| 3 | `geom_eccentricity` | 0.694 | 0.670 |
| 4 | `geom_compactness` | 0.679 | 1.311 |
| 5 | `color_darkness` | 0.438 | 0.426 |
| 6 | `asym_diagonal` | 0.418 | 0.541 |
| 7 | `concept_A_asymmetry` | 0.340 | 0.305 |
| 8 | `concept_B_border` | 0.270 | 0.362 |
| 9 | `asym_horizontal` | 0.203 | 0.237 |
| 10 | `geom_convexity_defects` | 0.200 | 0.204 |

### TN_benign  (n = 21585)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.885 | 0.855 |
| 2 | `geom_eccentricity` | 0.763 | 0.731 |
| 3 | `geom_compactness` | 0.661 | 1.198 |
| 4 | `geom_perim_ratio` | 0.540 | 0.920 |
| 5 | `asym_diagonal` | 0.478 | 0.573 |
| 6 | `color_darkness` | 0.354 | 0.369 |
| 7 | `concept_A_asymmetry` | 0.330 | 0.336 |
| 8 | `concept_B_border` | 0.274 | 0.363 |
| 9 | `asym_vertical` | 0.246 | 0.276 |
| 10 | `asym_horizontal` | 0.244 | 0.276 |

### FP_benign  (n = 6459)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.898 | 0.870 |
| 2 | `geom_eccentricity` | 0.751 | 0.722 |
| 3 | `geom_perim_ratio` | 0.730 | 1.169 |
| 4 | `geom_compactness` | 0.659 | 1.167 |
| 5 | `color_darkness` | 0.396 | 0.411 |
| 6 | `asym_diagonal` | 0.392 | 0.511 |
| 7 | `color_n_regions` | 0.333 | 0.296 |
| 8 | `concept_A_asymmetry` | 0.293 | 0.303 |
| 9 | `concept_B_border` | 0.278 | 0.359 |
| 10 | `asym_vertical` | 0.226 | 0.253 |

### FN_malignant  (n = 2)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.948 | 0.948 |
| 2 | `geom_eccentricity` | 0.710 | 0.710 |
| 3 | `asym_diagonal` | 0.609 | 0.609 |
| 4 | `geom_perim_ratio` | 0.496 | 0.496 |
| 5 | `color_n_regions` | 0.417 | 0.417 |
| 6 | `color_darkness` | 0.379 | 0.379 |
| 7 | `color_variegation` | 0.317 | 0.317 |
| 8 | `concept_A_asymmetry` | 0.315 | 0.315 |
| 9 | `geom_compactness` | 0.296 | 0.296 |
| 10 | `asym_vertical` | 0.250 | 0.250 |

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
