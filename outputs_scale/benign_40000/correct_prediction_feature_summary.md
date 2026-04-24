# Correct-Prediction Feature Summary

For each pipeline and each of the four prediction buckets, this table lists the pixel-derived image features that were highest (by median) across the bucket's test images. All feature values are normalised to [0, 1] by `glassderm.data.features`, so medians are directly comparable.

Only the top 10 features per bucket are shown; see `correct_prediction_feature_summary.csv` for the full ranking.

**Bucket legend**

- `TP_malignant` — True positives — malignant lesions the pipeline correctly flagged.
- `TN_benign` — True negatives — benign lesions the pipeline correctly passed.
- `FP_benign` — False positives — benign lesions mistakenly flagged as malignant.
- `FN_malignant` — False negatives — malignant lesions the pipeline missed.

## Pipeline: `glassbox_nam`

### TP_malignant  (n = 20)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 0.935 | 1.374 |
| 2 | `geom_solidity` | 0.894 | 0.877 |
| 3 | `geom_eccentricity` | 0.640 | 0.640 |
| 4 | `geom_compactness` | 0.627 | 1.055 |
| 5 | `color_darkness` | 0.442 | 0.441 |
| 6 | `asym_diagonal` | 0.359 | 0.483 |
| 7 | `concept_B_border` | 0.260 | 0.342 |
| 8 | `concept_A_asymmetry` | 0.255 | 0.269 |
| 9 | `geom_convexity_defects` | 0.220 | 0.240 |
| 10 | `asym_horizontal` | 0.184 | 0.207 |

### TN_benign  (n = 25341)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.882 | 0.854 |
| 2 | `geom_eccentricity` | 0.766 | 0.733 |
| 3 | `geom_compactness` | 0.695 | 1.233 |
| 4 | `geom_perim_ratio` | 0.546 | 0.971 |
| 5 | `asym_diagonal` | 0.488 | 0.580 |
| 6 | `color_darkness` | 0.360 | 0.377 |
| 7 | `concept_A_asymmetry` | 0.332 | 0.338 |
| 8 | `concept_B_border` | 0.284 | 0.370 |
| 9 | `asym_vertical` | 0.248 | 0.278 |
| 10 | `asym_horizontal` | 0.246 | 0.278 |

### FP_benign  (n = 2703)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.931 | 0.902 |
| 2 | `geom_perim_ratio` | 0.776 | 1.037 |
| 3 | `geom_eccentricity` | 0.710 | 0.688 |
| 4 | `geom_compactness` | 0.435 | 0.795 |
| 5 | `color_darkness` | 0.388 | 0.396 |
| 6 | `color_n_regions` | 0.333 | 0.346 |
| 7 | `color_variegation` | 0.263 | 0.254 |
| 8 | `asym_diagonal` | 0.241 | 0.357 |
| 9 | `concept_B_border` | 0.219 | 0.285 |
| 10 | `concept_A_asymmetry` | 0.195 | 0.232 |

### FN_malignant  (n = 9)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.921 | 0.847 |
| 2 | `geom_eccentricity` | 0.770 | 0.745 |
| 3 | `asym_diagonal` | 0.644 | 0.684 |
| 4 | `geom_compactness` | 0.473 | 1.654 |
| 5 | `color_darkness` | 0.379 | 0.382 |
| 6 | `concept_A_asymmetry` | 0.374 | 0.385 |
| 7 | `geom_perim_ratio` | 0.340 | 1.118 |
| 8 | `color_n_regions` | 0.333 | 0.278 |
| 9 | `asym_vertical` | 0.283 | 0.320 |
| 10 | `asym_horizontal` | 0.279 | 0.302 |

## Pipeline: `hard_cbm`

### TP_malignant  (n = 20)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 0.935 | 1.458 |
| 2 | `geom_solidity` | 0.897 | 0.864 |
| 3 | `geom_eccentricity` | 0.640 | 0.647 |
| 4 | `geom_compactness` | 0.627 | 1.362 |
| 5 | `color_darkness` | 0.466 | 0.460 |
| 6 | `asym_diagonal` | 0.381 | 0.498 |
| 7 | `concept_A_asymmetry` | 0.299 | 0.282 |
| 8 | `concept_B_border` | 0.260 | 0.354 |
| 9 | `geom_convexity_defects` | 0.240 | 0.238 |
| 10 | `asym_horizontal` | 0.205 | 0.219 |

### TN_benign  (n = 21908)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.867 | 0.843 |
| 2 | `geom_compactness` | 0.792 | 1.334 |
| 3 | `geom_eccentricity` | 0.776 | 0.741 |
| 4 | `asym_diagonal` | 0.584 | 0.621 |
| 5 | `geom_perim_ratio` | 0.564 | 1.011 |
| 6 | `color_darkness` | 0.359 | 0.376 |
| 7 | `concept_A_asymmetry` | 0.348 | 0.358 |
| 8 | `concept_B_border` | 0.311 | 0.392 |
| 9 | `asym_vertical` | 0.266 | 0.293 |
| 10 | `asym_horizontal` | 0.264 | 0.292 |

### FP_benign  (n = 6136)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.945 | 0.914 |
| 2 | `geom_eccentricity` | 0.707 | 0.684 |
| 3 | `geom_perim_ratio` | 0.576 | 0.856 |
| 4 | `color_darkness` | 0.378 | 0.388 |
| 5 | `geom_compactness` | 0.346 | 0.679 |
| 6 | `color_n_regions` | 0.333 | 0.324 |
| 7 | `asym_diagonal` | 0.222 | 0.335 |
| 8 | `color_variegation` | 0.210 | 0.230 |
| 9 | `concept_B_border` | 0.191 | 0.254 |
| 10 | `concept_A_asymmetry` | 0.184 | 0.220 |

### FN_malignant  (n = 9)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.893 | 0.875 |
| 2 | `geom_eccentricity` | 0.770 | 0.730 |
| 3 | `asym_diagonal` | 0.644 | 0.651 |
| 4 | `geom_compactness` | 0.473 | 0.971 |
| 5 | `color_darkness` | 0.363 | 0.338 |
| 6 | `concept_A_asymmetry` | 0.343 | 0.356 |
| 7 | `geom_perim_ratio` | 0.340 | 0.932 |
| 8 | `color_n_regions` | 0.333 | 0.296 |
| 9 | `concept_B_border` | 0.219 | 0.332 |
| 10 | `asym_vertical` | 0.217 | 0.290 |

## Pipeline: `multitask_cnn`

### TP_malignant  (n = 19)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.270 | 1.472 |
| 2 | `geom_solidity` | 0.880 | 0.864 |
| 3 | `geom_compactness` | 0.823 | 1.167 |
| 4 | `geom_eccentricity` | 0.649 | 0.664 |
| 5 | `color_darkness` | 0.438 | 0.441 |
| 6 | `asym_diagonal` | 0.369 | 0.481 |
| 7 | `concept_A_asymmetry` | 0.297 | 0.280 |
| 8 | `concept_B_border` | 0.277 | 0.373 |
| 9 | `geom_convexity_defects` | 0.240 | 0.251 |
| 10 | `border_radial_variance` | 0.196 | 0.203 |

### TN_benign  (n = 24013)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.889 | 0.858 |
| 2 | `geom_eccentricity` | 0.761 | 0.730 |
| 3 | `geom_compactness` | 0.638 | 1.169 |
| 4 | `geom_perim_ratio` | 0.546 | 0.915 |
| 5 | `asym_diagonal` | 0.459 | 0.559 |
| 6 | `color_darkness` | 0.355 | 0.370 |
| 7 | `concept_A_asymmetry` | 0.324 | 0.330 |
| 8 | `concept_B_border` | 0.268 | 0.357 |
| 9 | `asym_vertical` | 0.241 | 0.273 |
| 10 | `asym_horizontal` | 0.240 | 0.272 |

### FP_benign  (n = 4031)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 0.938 | 1.351 |
| 2 | `geom_solidity` | 0.883 | 0.861 |
| 3 | `geom_compactness` | 0.773 | 1.323 |
| 4 | `geom_eccentricity` | 0.756 | 0.724 |
| 5 | `asym_diagonal` | 0.433 | 0.557 |
| 6 | `color_darkness` | 0.406 | 0.429 |
| 7 | `concept_A_asymmetry` | 0.311 | 0.318 |
| 8 | `concept_B_border` | 0.310 | 0.391 |
| 9 | `asym_vertical` | 0.235 | 0.261 |
| 10 | `asym_horizontal` | 0.231 | 0.257 |

### FN_malignant  (n = 10)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.927 | 0.874 |
| 2 | `asym_diagonal` | 0.740 | 0.667 |
| 3 | `geom_eccentricity` | 0.706 | 0.688 |
| 4 | `geom_compactness` | 0.436 | 1.382 |
| 5 | `color_darkness` | 0.379 | 0.387 |
| 6 | `concept_A_asymmetry` | 0.352 | 0.353 |
| 7 | `geom_perim_ratio` | 0.297 | 0.959 |
| 8 | `color_n_regions` | 0.250 | 0.267 |
| 9 | `asym_vertical` | 0.243 | 0.278 |
| 10 | `asym_horizontal` | 0.222 | 0.271 |

## Pipeline: `transparent_lr`

### TP_malignant  (n = 12)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.596 | 1.941 |
| 2 | `geom_compactness` | 1.073 | 1.418 |
| 3 | `geom_solidity` | 0.879 | 0.858 |
| 4 | `geom_eccentricity` | 0.620 | 0.618 |
| 5 | `color_darkness` | 0.466 | 0.477 |
| 6 | `geom_convexity_defects` | 0.360 | 0.320 |
| 7 | `concept_B_border` | 0.335 | 0.422 |
| 8 | `asym_diagonal` | 0.313 | 0.388 |
| 9 | `concept_D_diameter` | 0.251 | 0.263 |
| 10 | `geom_area_ratio` | 0.238 | 0.277 |

### TN_benign  (n = 24971)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.888 | 0.856 |
| 2 | `geom_eccentricity` | 0.770 | 0.741 |
| 3 | `geom_compactness` | 0.636 | 1.161 |
| 4 | `geom_perim_ratio` | 0.539 | 0.888 |
| 5 | `asym_diagonal` | 0.479 | 0.574 |
| 6 | `color_darkness` | 0.353 | 0.367 |
| 7 | `concept_A_asymmetry` | 0.332 | 0.338 |
| 8 | `concept_B_border` | 0.269 | 0.355 |
| 9 | `asym_vertical` | 0.249 | 0.279 |
| 10 | `asym_horizontal` | 0.247 | 0.279 |

### FP_benign  (n = 3073)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.318 | 1.699 |
| 2 | `geom_solidity` | 0.891 | 0.876 |
| 3 | `geom_compactness` | 0.889 | 1.437 |
| 4 | `geom_eccentricity` | 0.666 | 0.629 |
| 5 | `color_darkness` | 0.430 | 0.475 |
| 6 | `asym_diagonal` | 0.369 | 0.435 |
| 7 | `concept_B_border` | 0.334 | 0.416 |
| 8 | `color_n_regions` | 0.333 | 0.309 |
| 9 | `concept_A_asymmetry` | 0.227 | 0.249 |
| 10 | `geom_convexity_defects` | 0.200 | 0.205 |

### FN_malignant  (n = 17)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.922 | 0.874 |
| 2 | `geom_eccentricity` | 0.721 | 0.710 |
| 3 | `asym_diagonal` | 0.644 | 0.657 |
| 4 | `geom_compactness` | 0.455 | 1.116 |
| 5 | `geom_perim_ratio` | 0.408 | 0.839 |
| 6 | `color_darkness` | 0.379 | 0.384 |
| 7 | `concept_A_asymmetry` | 0.361 | 0.344 |
| 8 | `color_n_regions` | 0.333 | 0.294 |
| 9 | `asym_vertical` | 0.270 | 0.276 |
| 10 | `border_radial_variance` | 0.227 | 0.229 |

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

### TN_benign  (n = 26491)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_solidity` | 0.890 | 0.859 |
| 2 | `geom_eccentricity` | 0.760 | 0.729 |
| 3 | `geom_compactness` | 0.630 | 1.147 |
| 4 | `geom_perim_ratio` | 0.537 | 0.918 |
| 5 | `asym_diagonal` | 0.466 | 0.566 |
| 6 | `color_darkness` | 0.360 | 0.372 |
| 7 | `concept_A_asymmetry` | 0.326 | 0.332 |
| 8 | `concept_B_border` | 0.266 | 0.351 |
| 9 | `asym_vertical` | 0.242 | 0.273 |
| 10 | `asym_horizontal` | 0.240 | 0.272 |

### FP_benign  (n = 1553)

| rank | feature | median | mean |
|---:|:---|---:|---:|
| 1 | `geom_perim_ratio` | 1.626 | 1.987 |
| 2 | `geom_compactness` | 1.458 | 1.932 |
| 3 | `geom_solidity` | 0.856 | 0.840 |
| 4 | `geom_eccentricity` | 0.762 | 0.724 |
| 5 | `concept_B_border` | 0.510 | 0.548 |
| 6 | `color_darkness` | 0.409 | 0.485 |
| 7 | `asym_diagonal` | 0.375 | 0.429 |
| 8 | `color_n_regions` | 0.333 | 0.295 |
| 9 | `border_radial_variance` | 0.245 | 0.257 |
| 10 | `concept_A_asymmetry` | 0.242 | 0.270 |

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
