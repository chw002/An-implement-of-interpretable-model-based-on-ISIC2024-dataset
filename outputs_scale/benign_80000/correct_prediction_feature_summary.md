# Correct-Prediction Feature Summary

For each pipeline and each of the four prediction buckets, this table lists the pixel-derived image features that were highest (by median) across the bucket's test images. All feature values are normalised to [0, 1] by `glassderm.data.features`, so medians are directly comparable.

Only the top 10 features per bucket are shown; see `correct_prediction_feature_summary.csv` for the full ranking.

**Bucket legend**

- `TP_malignant` — True positives — malignant lesions the pipeline correctly flagged.
- `TN_benign` — True negatives — benign lesions the pipeline correctly passed.
- `FP_benign` — False positives — benign lesions mistakenly flagged as malignant.
- `FN_malignant` — False negatives — malignant lesions the pipeline missed.

## Pipeline: `glassbox_nam`

### TP_malignant  (n = 21)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.270 | 1.650 |
| 2 | `geom_solidity` | 0.880 | 0.854 |
| 3 | `geom_compactness` | 0.823 | 1.506 |
| 4 | `geom_eccentricity` | 0.632 | 0.645 |
| 5 | `color_darkness` | 0.445 | 0.442 |
| 6 | `asym_diagonal` | 0.369 | 0.466 |
| 7 | `concept_A_asymmetry` | 0.297 | 0.280 |
| 8 | `concept_B_border` | 0.277 | 0.395 |
| 9 | `geom_convexity_defects` | 0.240 | 0.246 |
| 10 | `asym_horizontal` | 0.203 | 0.222 |

### TN_benign  (n = 25166)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.887 | 0.856 |
| 2 | `geom_eccentricity` | 0.760 | 0.728 |
| 3 | `geom_compactness` | 0.650 | 1.186 |
| 4 | `geom_perim_ratio` | 0.540 | 0.910 |
| 5 | `asym_diagonal` | 0.479 | 0.572 |
| 6 | `color_darkness` | 0.359 | 0.376 |
| 7 | `concept_A_asymmetry` | 0.328 | 0.334 |
| 8 | `concept_B_border` | 0.271 | 0.360 |
| 9 | `asym_vertical` | 0.244 | 0.275 |
| 10 | `asym_horizontal` | 0.242 | 0.274 |

### FP_benign  (n = 2878)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.148 | 1.563 |
| 2 | `geom_solidity` | 0.900 | 0.876 |
| 3 | `geom_eccentricity` | 0.761 | 0.734 |
| 4 | `geom_compactness` | 0.732 | 1.237 |
| 5 | `color_darkness` | 0.392 | 0.402 |
| 6 | `asym_diagonal` | 0.345 | 0.442 |
| 7 | `color_n_regions` | 0.333 | 0.319 |
| 8 | `concept_B_border` | 0.298 | 0.377 |
| 9 | `concept_A_asymmetry` | 0.257 | 0.276 |
| 10 | `asym_vertical` | 0.212 | 0.237 |

### FN_malignant  (n = 8)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `asym_diagonal` | 1.000 | 0.755 |
| 2 | `geom_solidity` | 0.933 | 0.902 |
| 3 | `geom_eccentricity` | 0.744 | 0.743 |
| 4 | `color_darkness` | 0.362 | 0.371 |
| 5 | `concept_A_asymmetry` | 0.352 | 0.372 |
| 6 | `geom_compactness` | 0.352 | 0.546 |
| 7 | `asym_vertical` | 0.243 | 0.280 |
| 8 | `geom_perim_ratio` | 0.227 | 0.363 |
| 9 | `asym_horizontal` | 0.211 | 0.273 |
| 10 | `asym_centroid_offset` | 0.211 | 0.167 |

## Pipeline: `hard_cbm`

### TP_malignant  (n = 17)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 0.950 | 1.338 |
| 2 | `geom_solidity` | 0.894 | 0.870 |
| 3 | `geom_compactness` | 0.721 | 1.043 |
| 4 | `geom_eccentricity` | 0.649 | 0.656 |
| 5 | `color_darkness` | 0.451 | 0.447 |
| 6 | `asym_diagonal` | 0.349 | 0.440 |
| 7 | `geom_convexity_defects` | 0.280 | 0.266 |
| 8 | `concept_B_border` | 0.270 | 0.343 |
| 9 | `concept_A_asymmetry` | 0.213 | 0.269 |
| 10 | `asym_horizontal` | 0.203 | 0.217 |

### TN_benign  (n = 26376)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.886 | 0.856 |
| 2 | `geom_eccentricity` | 0.763 | 0.731 |
| 3 | `geom_compactness` | 0.670 | 1.210 |
| 4 | `geom_perim_ratio` | 0.552 | 0.966 |
| 5 | `asym_diagonal` | 0.471 | 0.569 |
| 6 | `color_darkness` | 0.360 | 0.376 |
| 7 | `concept_A_asymmetry` | 0.327 | 0.333 |
| 8 | `concept_B_border` | 0.278 | 0.365 |
| 9 | `asym_vertical` | 0.244 | 0.274 |
| 10 | `asym_horizontal` | 0.242 | 0.274 |

### FP_benign  (n = 1668)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 0.969 | 1.161 |
| 2 | `geom_solidity` | 0.916 | 0.888 |
| 3 | `geom_eccentricity` | 0.718 | 0.698 |
| 4 | `geom_compactness` | 0.544 | 0.893 |
| 5 | `color_darkness` | 0.404 | 0.415 |
| 6 | `color_n_regions` | 0.333 | 0.301 |
| 7 | `asym_diagonal` | 0.269 | 0.391 |
| 8 | `concept_B_border` | 0.248 | 0.306 |
| 9 | `concept_A_asymmetry` | 0.216 | 0.248 |
| 10 | `geom_convexity_defects` | 0.200 | 0.211 |

### FN_malignant  (n = 12)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.907 | 0.863 |
| 2 | `asym_diagonal` | 0.822 | 0.694 |
| 3 | `geom_eccentricity` | 0.706 | 0.696 |
| 4 | `geom_compactness` | 0.451 | 1.521 |
| 5 | `color_darkness` | 0.386 | 0.388 |
| 6 | `concept_A_asymmetry` | 0.352 | 0.357 |
| 7 | `geom_perim_ratio` | 0.320 | 1.234 |
| 8 | `color_n_regions` | 0.250 | 0.278 |
| 9 | `asym_vertical` | 0.243 | 0.281 |
| 10 | `concept_B_border` | 0.218 | 0.353 |

## Pipeline: `multitask_cnn`

### TP_malignant  (n = 19)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 0.950 | 1.438 |
| 2 | `geom_solidity` | 0.893 | 0.872 |
| 3 | `geom_compactness` | 0.721 | 1.099 |
| 4 | `geom_eccentricity` | 0.632 | 0.636 |
| 5 | `color_darkness` | 0.445 | 0.445 |
| 6 | `asym_diagonal` | 0.349 | 0.456 |
| 7 | `concept_B_border` | 0.270 | 0.354 |
| 8 | `geom_convexity_defects` | 0.240 | 0.253 |
| 9 | `concept_A_asymmetry` | 0.213 | 0.265 |
| 10 | `asym_vertical` | 0.188 | 0.228 |

### TN_benign  (n = 25733)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.887 | 0.857 |
| 2 | `geom_eccentricity` | 0.761 | 0.729 |
| 3 | `geom_compactness` | 0.656 | 1.193 |
| 4 | `geom_perim_ratio` | 0.550 | 0.949 |
| 5 | `asym_diagonal` | 0.463 | 0.564 |
| 6 | `color_darkness` | 0.360 | 0.375 |
| 7 | `concept_A_asymmetry` | 0.325 | 0.331 |
| 8 | `concept_B_border` | 0.273 | 0.362 |
| 9 | `asym_vertical` | 0.242 | 0.273 |
| 10 | `asym_horizontal` | 0.240 | 0.272 |

### FP_benign  (n = 2311)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.053 | 1.297 |
| 2 | `geom_solidity` | 0.900 | 0.873 |
| 3 | `geom_eccentricity` | 0.753 | 0.724 |
| 4 | `geom_compactness` | 0.699 | 1.162 |
| 5 | `color_darkness` | 0.400 | 0.422 |
| 6 | `asym_diagonal` | 0.381 | 0.502 |
| 7 | `concept_A_asymmetry` | 0.289 | 0.299 |
| 8 | `concept_B_border` | 0.288 | 0.365 |
| 9 | `asym_vertical` | 0.227 | 0.250 |
| 10 | `asym_horizontal` | 0.220 | 0.246 |

### FN_malignant  (n = 10)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.927 | 0.859 |
| 2 | `asym_diagonal` | 0.822 | 0.715 |
| 3 | `geom_eccentricity` | 0.744 | 0.742 |
| 4 | `geom_compactness` | 0.445 | 1.511 |
| 5 | `color_darkness` | 0.371 | 0.380 |
| 6 | `concept_A_asymmetry` | 0.367 | 0.381 |
| 7 | `geom_perim_ratio` | 0.297 | 1.023 |
| 8 | `asym_vertical` | 0.277 | 0.304 |
| 9 | `asym_horizontal` | 0.258 | 0.290 |
| 10 | `color_n_regions` | 0.250 | 0.267 |

## Pipeline: `transparent_lr`

### TP_malignant  (n = 13)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.510 | 1.823 |
| 2 | `geom_compactness` | 0.928 | 1.381 |
| 3 | `geom_solidity` | 0.878 | 0.852 |
| 4 | `geom_eccentricity` | 0.609 | 0.605 |
| 5 | `color_darkness` | 0.445 | 0.470 |
| 6 | `geom_convexity_defects` | 0.360 | 0.305 |
| 7 | `asym_diagonal` | 0.336 | 0.417 |
| 8 | `concept_B_border` | 0.314 | 0.413 |
| 9 | `concept_D_diameter` | 0.247 | 0.249 |
| 10 | `geom_area_ratio` | 0.229 | 0.257 |

### TN_benign  (n = 24817)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.888 | 0.856 |
| 2 | `geom_eccentricity` | 0.770 | 0.742 |
| 3 | `geom_compactness` | 0.636 | 1.158 |
| 4 | `geom_perim_ratio` | 0.539 | 0.886 |
| 5 | `asym_diagonal` | 0.481 | 0.575 |
| 6 | `color_darkness` | 0.352 | 0.366 |
| 7 | `concept_A_asymmetry` | 0.333 | 0.338 |
| 8 | `concept_B_border` | 0.269 | 0.355 |
| 9 | `asym_vertical` | 0.249 | 0.279 |
| 10 | `asym_horizontal` | 0.247 | 0.279 |

### FP_benign  (n = 3227)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.252 | 1.683 |
| 2 | `geom_solidity` | 0.889 | 0.874 |
| 3 | `geom_compactness` | 0.884 | 1.446 |
| 4 | `geom_eccentricity` | 0.664 | 0.630 |
| 5 | `color_darkness` | 0.428 | 0.472 |
| 6 | `asym_diagonal` | 0.371 | 0.435 |
| 7 | `color_n_regions` | 0.333 | 0.309 |
| 8 | `concept_B_border` | 0.333 | 0.416 |
| 9 | `concept_A_asymmetry` | 0.229 | 0.252 |
| 10 | `geom_convexity_defects` | 0.200 | 0.201 |

### FN_malignant  (n = 16)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.927 | 0.880 |
| 2 | `geom_eccentricity` | 0.745 | 0.727 |
| 3 | `asym_diagonal` | 0.562 | 0.650 |
| 4 | `geom_perim_ratio` | 0.504 | 0.866 |
| 5 | `geom_compactness` | 0.446 | 1.127 |
| 6 | `color_darkness` | 0.371 | 0.384 |
| 7 | `concept_A_asymmetry` | 0.359 | 0.341 |
| 8 | `color_n_regions` | 0.250 | 0.281 |
| 9 | `asym_vertical` | 0.243 | 0.269 |
| 10 | `concept_B_border` | 0.213 | 0.294 |

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

### TN_benign  (n = 26541)

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

### FP_benign  (n = 1503)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.646 | 2.032 |
| 2 | `geom_compactness` | 1.499 | 1.972 |
| 3 | `geom_solidity` | 0.853 | 0.838 |
| 4 | `geom_eccentricity` | 0.762 | 0.723 |
| 5 | `concept_B_border` | 0.523 | 0.557 |
| 6 | `color_darkness` | 0.408 | 0.487 |
| 7 | `asym_diagonal` | 0.379 | 0.434 |
| 8 | `color_n_regions` | 0.333 | 0.294 |
| 9 | `border_radial_variance` | 0.246 | 0.259 |
| 10 | `concept_D_diameter` | 0.243 | 0.285 |

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
