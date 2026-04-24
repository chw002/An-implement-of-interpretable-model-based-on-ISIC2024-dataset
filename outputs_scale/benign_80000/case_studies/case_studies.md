# Case studies

_Anchor pipeline: **transparent_lr**. All narratives share the same test image across every pipeline so the reader can compare explanation styles side-by-side._

## TP · malignant correctly flagged

### image_id = `ISIC_0664841`

- patient_id = `IP_4549819`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_0664841.jpg`
- anchor (transparent_lr) P(malignant) = 0.968

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.018 τ=0.016 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.018; with validation-tuned threshold τ=0.016
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.358  B=0.408  C=0.184  D=0.184.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.336 τ=0.289 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.289
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.225  w=-0.631  contribution=-0.142
    B(Border)       c=+0.368  w=-0.610  contribution=-0.225
    C(Color)        c=+0.055  w=-0.671  contribution=-0.037
    D(Diameter)     c=+0.150  w=-0.679  contribution=-0.102
  + bias = -0.178
  = logit -0.683  →  σ(logit) = P(malignant) = 0.336  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.123 τ=0.029 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.029
    f_A(0.319) = -1.039   (A(Asymmetry))
    f_B(0.497) = -0.874   (B(Border))
    f_C(0.686) = +0.245   (C(Color))
    f_D(0.656) = -0.268   (D(Diameter))
  + bias = -0.027
  = logit -1.962  →  P(malignant) = 0.123  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.968 τ=0.586 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.586
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.3466  scaled=0.693  w=+2.886  contrib=+2.001
    geom_convexity_defects      raw=+0.4400  scaled=0.647  w=+2.935  contrib=+1.899
    color_val_std               raw=+0.0704  scaled=0.135  w=+11.384  contrib=+1.541
    geom_perim_ratio            raw=+3.2010  scaled=0.315  w=+4.875  contrib=+1.534
    color_darkness              raw=+0.6053  scaled=0.600  w=+2.055  contrib=+1.233
    color_rgb_std_mean          raw=+0.0566  scaled=0.111  w=-11.036  contrib=-1.229
    color_sat_std               raw=+0.0738  scaled=0.144  w=-6.934  contrib=-0.999
    geom_compactness            raw=+2.2902  scaled=0.254  w=-3.260  contrib=-0.829
  + bias = -1.170
  = logit +3.396  →  σ(logit) = P(malignant) = 0.968  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.880 τ=0.847 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.847
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.693 > 0.018
    geom_area_ratio            scaled=0.500 > 0.081
    asym_horizontal            scaled=0.361 > 0.000
    border_gradient            scaled=0.153 > 0.032
  → leaf class counts [benign,malignant] = [0.11990769918581569, 0.8800923008141599]
  → leaf P(malignant) = 0.880  →  MALIGNANT
```

---

### image_id = `ISIC_5366444`

- patient_id = `IP_2878659`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_5366444.jpg`
- anchor (transparent_lr) P(malignant) = 0.958

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.299 τ=0.016 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.299; with validation-tuned threshold τ=0.016
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.238  B=0.341  C=0.060  D=0.223.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.324 τ=0.289 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.289
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.257  w=-0.631  contribution=-0.162
    B(Border)       c=+0.385  w=-0.610  contribution=-0.235
    C(Color)        c=+0.044  w=-0.671  contribution=-0.030
    D(Diameter)     c=+0.193  w=-0.679  contribution=-0.131
  + bias = -0.178
  = logit -0.736  →  σ(logit) = P(malignant) = 0.324  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.066 τ=0.029 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.029
    f_A(0.317) = -1.039   (A(Asymmetry))
    f_B(0.460) = -0.879   (B(Border))
    f_C(0.376) = -0.259   (C(Color))
    f_D(0.489) = -0.440   (D(Diameter))
  + bias = -0.027
  = logit -2.644  →  P(malignant) = 0.066  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.958 τ=0.586 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.586
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.3596  scaled=0.719  w=+2.886  contrib=+2.076
    geom_perim_ratio            raw=+4.0827  scaled=0.401  w=+4.875  contrib=+1.956
    geom_convexity_defects      raw=+0.3200  scaled=0.471  w=+2.935  contrib=+1.381
    geom_compactness            raw=+3.4361  scaled=0.382  w=-3.260  contrib=-1.244
    color_darkness              raw=+0.5684  scaled=0.563  w=+2.055  contrib=+1.156
    geom_eccentricity           raw=+0.5309  scaled=0.531  w=-1.008  contrib=-0.535
    color_sat_std               raw=+0.0347  scaled=0.066  w=-6.934  contrib=-0.455
    asym_centroid_offset        raw=+0.1267  scaled=0.253  w=+1.669  contrib=+0.423
  + bias = -1.170
  = logit +3.116  →  σ(logit) = P(malignant) = 0.958  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.880 τ=0.847 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.847
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.719 > 0.018
    geom_area_ratio            scaled=0.603 > 0.081
    asym_horizontal            scaled=0.126 > 0.000
    border_gradient            scaled=0.044 > 0.032
  → leaf class counts [benign,malignant] = [0.11990769918581569, 0.8800923008141599]
  → leaf P(malignant) = 0.880  →  MALIGNANT
```

---

## FN · malignant missed

### image_id = `ISIC_9420821`

- patient_id = `IP_5611762`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_9420821.jpg`
- anchor (transparent_lr) P(malignant) = 0.158

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.002 τ=0.016 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.002; with validation-tuned threshold τ=0.016
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.273  B=0.254  C=0.135  D=0.106.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.271 τ=0.289 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.289
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.387  w=-0.631  contribution=-0.244
    B(Border)       c=+0.349  w=-0.610  contribution=-0.213
    C(Color)        c=+0.262  w=-0.671  contribution=-0.176
    D(Diameter)     c=+0.264  w=-0.679  contribution=-0.179
  + bias = -0.178
  = logit -0.990  →  σ(logit) = P(malignant) = 0.271  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.021 τ=0.029 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.029
    f_A(0.284) = -1.051   (A(Asymmetry))
    f_B(0.285) = -0.953   (B(Border))
    f_C(0.119) = -0.940   (C(Color))
    f_D(0.165) = -0.861   (D(Diameter))
  + bias = -0.027
  = logit -3.831  →  P(malignant) = 0.021  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.158 τ=0.586 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.586
  Top signed contributions (w · scaled_x) driving this decision:
    color_rgb_std_mean          raw=+0.1220  scaled=0.242  w=-11.036  contrib=-2.676
    color_val_std               raw=+0.1181  scaled=0.231  w=+11.384  contrib=+2.634
    color_sat_std               raw=+0.0989  scaled=0.194  w=-6.934  contrib=-1.348
    geom_convexity_defects      raw=+0.2000  scaled=0.294  w=+2.935  contrib=+0.863
    geom_eccentricity           raw=+0.7700  scaled=0.770  w=-1.008  contrib=-0.776
    geom_compactness            raw=+1.8920  scaled=0.210  w=-3.260  contrib=-0.685
    color_darkness              raw=+0.2829  scaled=0.273  w=+2.055  contrib=+0.561
    geom_perim_ratio            raw=+0.7913  scaled=0.078  w=+4.875  contrib=+0.379
  + bias = -1.170
  = logit -1.670  →  σ(logit) = P(malignant) = 0.158  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.240 τ=0.847 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.847
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.017 <= 0.018
    color_sat_std              scaled=0.194 > 0.060
    color_val_std              scaled=0.231 > 0.128
    geom_area_ratio            scaled=0.035 <= 0.064
  → leaf class counts [benign,malignant] = [0.7601846578134401, 0.23981534218650558]
  → leaf P(malignant) = 0.240  →  BENIGN
```

---

### image_id = `ISIC_2538516`

- patient_id = `IP_6141958`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_2538516.jpg`
- anchor (transparent_lr) P(malignant) = 0.222

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.001 τ=0.016 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.001; with validation-tuned threshold τ=0.016
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.327  B=0.428  C=0.055  D=0.209.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.236 τ=0.289 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.289
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.456  w=-0.631  contribution=-0.288
    B(Border)       c=+0.553  w=-0.610  contribution=-0.337
    C(Color)        c=+0.179  w=-0.671  contribution=-0.120
    D(Diameter)     c=+0.373  w=-0.679  contribution=-0.254
  + bias = -0.178
  = logit -1.176  →  σ(logit) = P(malignant) = 0.236  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.027 τ=0.029 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.029
    f_A(0.336) = -1.039   (A(Asymmetry))
    f_B(0.394) = -0.902   (B(Border))
    f_C(0.170) = -0.792   (C(Color))
    f_D(0.205) = -0.816   (D(Diameter))
  + bias = -0.027
  = logit -3.576  →  P(malignant) = 0.027  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.222 τ=0.586 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.586
  Top signed contributions (w · scaled_x) driving this decision:
    geom_eccentricity           raw=+0.8756  scaled=0.876  w=-1.008  contrib=-0.882
    color_rgb_std_mean          raw=+0.0303  scaled=0.059  w=-11.036  contrib=-0.648
    color_val_std               raw=+0.0310  scaled=0.056  w=+11.384  contrib=+0.637
    color_darkness              raw=+0.2926  scaled=0.283  w=+2.055  contrib=+0.581
    geom_convexity_defects      raw=+0.0800  scaled=0.118  w=+2.935  contrib=+0.345
    border_radial_variance      raw=+0.3030  scaled=0.303  w=-0.838  contrib=-0.254
    geom_compactness            raw=+0.6785  scaled=0.075  w=-3.260  contrib=-0.246
    asym_vertical               raw=+0.4434  scaled=0.466  w=+0.505  contrib=+0.235
  + bias = -1.170
  = logit -1.257  →  σ(logit) = P(malignant) = 0.222  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.483 τ=0.847 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.847
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.006 <= 0.018
    color_sat_std              scaled=0.033 <= 0.060
    asym_diagonal              scaled=0.338 > 0.307
    color_darkness             scaled=0.283 > 0.241
  → leaf class counts [benign,malignant] = [0.5170304001428218, 0.48296959985689275]
  → leaf P(malignant) = 0.483  →  BENIGN
```

---

## FP · benign wrongly flagged

### image_id = `ISIC_6569920`

- patient_id = `IP_8494850`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_6569920.jpg`
- anchor (transparent_lr) P(malignant) = 0.993

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.494 τ=0.016 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.494; with validation-tuned threshold τ=0.016
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.227  B=0.280  C=0.111  D=0.200.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.455 τ=0.289 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.289
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.000  w=-0.631  contribution=-0.000
    B(Border)       c=+0.000  w=-0.610  contribution=-0.000
    C(Color)        c=+0.000  w=-0.671  contribution=-0.000
    D(Diameter)     c=+0.000  w=-0.679  contribution=-0.000
  + bias = -0.178
  = logit -0.179  →  σ(logit) = P(malignant) = 0.455  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.141 τ=0.029 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.029
    f_A(0.320) = -1.039   (A(Asymmetry))
    f_B(0.497) = -0.874   (B(Border))
    f_C(0.747) = +0.315   (C(Color))
    f_D(0.742) = -0.180   (D(Diameter))
  + bias = -0.027
  = logit -1.804  →  P(malignant) = 0.141  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.993 τ=0.586 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.586
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.4749  scaled=0.950  w=+2.886  contrib=+2.741
    geom_convexity_defects      raw=+0.5200  scaled=0.765  w=+2.935  contrib=+2.244
    color_val_std               raw=+0.0656  scaled=0.126  w=+11.384  contrib=+1.431
    color_rgb_std_mean          raw=+0.0451  scaled=0.088  w=-11.036  contrib=-0.975
    geom_perim_ratio            raw=+2.0008  scaled=0.197  w=+4.875  contrib=+0.959
    color_darkness              raw=+0.4540  scaled=0.446  w=+2.055  contrib=+0.917
    geom_eccentricity           raw=+0.8068  scaled=0.807  w=-1.008  contrib=-0.813
    color_sat_std               raw=+0.0564  scaled=0.109  w=-6.934  contrib=-0.756
  + bias = -1.170
  = logit +4.908  →  σ(logit) = P(malignant) = 0.993  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.880 τ=0.847 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.847
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.950 > 0.018
    geom_area_ratio            scaled=0.368 > 0.081
    asym_horizontal            scaled=0.358 > 0.000
    border_gradient            scaled=0.068 > 0.032
  → leaf class counts [benign,malignant] = [0.11990769918581569, 0.8800923008141599]
  → leaf P(malignant) = 0.880  →  MALIGNANT
```

---

### image_id = `ISIC_2759370`

- patient_id = `IP_7272992`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_2759370.jpg`
- anchor (transparent_lr) P(malignant) = 0.986

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.173 τ=0.016 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.173; with validation-tuned threshold τ=0.016
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.297  B=0.435  C=0.139  D=0.332.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.248 τ=0.289 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.289
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.365  w=-0.631  contribution=-0.230
    B(Border)       c=+0.460  w=-0.610  contribution=-0.281
    C(Color)        c=+0.233  w=-0.671  contribution=-0.156
    D(Diameter)     c=+0.387  w=-0.679  contribution=-0.263
  + bias = -0.178
  = logit -1.108  →  σ(logit) = P(malignant) = 0.248  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.056 τ=0.029 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.029
    f_A(0.302) = -1.043   (A(Asymmetry))
    f_B(0.477) = -0.877   (B(Border))
    f_C(0.343) = -0.330   (C(Color))
    f_D(0.410) = -0.547   (D(Diameter))
  + bias = -0.027
  = logit -2.823  →  P(malignant) = 0.056  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.986 τ=0.586 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.586
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.3716  scaled=0.743  w=+2.886  contrib=+2.145
    border_gradient             raw=+0.7096  scaled=0.704  w=+2.650  contrib=+1.866
    color_darkness              raw=+0.7486  scaled=0.745  w=+2.055  contrib=+1.532
    color_sat_std               raw=+0.0769  scaled=0.150  w=-6.934  contrib=-1.042
    geom_perim_ratio            raw=+1.8270  scaled=0.180  w=+4.875  contrib=+0.875
    asym_centroid_offset        raw=+0.2536  scaled=0.507  w=+1.669  contrib=+0.847
    geom_eccentricity           raw=+0.5834  scaled=0.583  w=-1.008  contrib=-0.588
    color_val_std               raw=+0.0284  scaled=0.051  w=+11.384  contrib=+0.579
  + bias = -1.170
  = logit +4.281  →  σ(logit) = P(malignant) = 0.986  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.880 τ=0.847 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.847
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.743 > 0.018
    geom_area_ratio            scaled=0.270 > 0.081
    asym_horizontal            scaled=0.282 > 0.000
    border_gradient            scaled=0.704 > 0.032
  → leaf class counts [benign,malignant] = [0.11990769918581569, 0.8800923008141599]
  → leaf P(malignant) = 0.880  →  MALIGNANT
```

---

## TN · benign correctly cleared

### image_id = `ISIC_5313485`

- patient_id = `IP_4549819`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_5313485.jpg`
- anchor (transparent_lr) P(malignant) = 0.011

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.004 τ=0.016 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.004; with validation-tuned threshold τ=0.016
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.454  B=0.472  C=0.071  D=0.165.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.188 τ=0.289 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.289
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.349  w=-0.631  contribution=-0.220
    B(Border)       c=+0.912  w=-0.610  contribution=-0.556
    C(Color)        c=+0.256  w=-0.671  contribution=-0.172
    D(Diameter)     c=+0.498  w=-0.679  contribution=-0.338
  + bias = -0.178
  = logit -1.464  →  σ(logit) = P(malignant) = 0.188  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.018 τ=0.029 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.029
    f_A(0.151) = -1.116   (A(Asymmetry))
    f_B(0.986) = -1.044   (B(Border))
    f_C(0.059) = -1.114   (C(Color))
    f_D(0.318) = -0.682   (D(Diameter))
  + bias = -0.027
  = logit -3.983  →  P(malignant) = 0.018  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.011 τ=0.586 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.586
  Top signed contributions (w · scaled_x) driving this decision:
    color_sat_std               raw=+0.4741  scaled=0.948  w=-6.934  contrib=-6.573
    border_gradient             raw=+1.0000  scaled=1.000  w=+2.650  contrib=+2.650
    geom_compactness            raw=+6.0891  scaled=0.677  w=-3.260  contrib=-2.205
    color_darkness              raw=+0.9723  scaled=0.972  w=+2.055  contrib=+1.998
    color_val_std               raw=+0.0756  scaled=0.146  w=+11.384  contrib=+1.660
    color_rgb_std_mean          raw=+0.0637  scaled=0.126  w=-11.036  contrib=-1.386
    asym_centroid_offset        raw=+0.3331  scaled=0.666  w=+1.669  contrib=+1.112
    geom_eccentricity           raw=+0.9999  scaled=1.000  w=-1.008  contrib=-1.008
  + bias = -1.170
  = logit -4.476  →  σ(logit) = P(malignant) = 0.011  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.000 τ=0.847 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.847
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.111 > 0.018
    geom_area_ratio            scaled=0.050 <= 0.081
    geom_compactness           scaled=0.677 > 0.031
    geom_perim_ratio           scaled=0.146 > 0.087
  → leaf class counts [benign,malignant] = [1.0, 0.0]
  → leaf P(malignant) = 0.000  →  BENIGN
```

---

### image_id = `ISIC_3115416`

- patient_id = `IP_4414342`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_3115416.jpg`
- anchor (transparent_lr) P(malignant) = 0.013

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.004 τ=0.016 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.004; with validation-tuned threshold τ=0.016
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.268  B=0.283  C=0.069  D=0.151.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.191 τ=0.289 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.289
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.313  w=-0.631  contribution=-0.198
    B(Border)       c=+0.929  w=-0.610  contribution=-0.566
    C(Color)        c=+0.229  w=-0.671  contribution=-0.154
    D(Diameter)     c=+0.512  w=-0.679  contribution=-0.348
  + bias = -0.178
  = logit -1.443  →  σ(logit) = P(malignant) = 0.191  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.021 τ=0.029 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.029
    f_A(0.186) = -1.099   (A(Asymmetry))
    f_B(0.967) = -1.034   (B(Border))
    f_C(0.089) = -1.027   (C(Color))
    f_D(0.337) = -0.654   (D(Diameter))
  + bias = -0.027
  = logit -3.840  →  P(malignant) = 0.021  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.013 τ=0.586 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.586
  Top signed contributions (w · scaled_x) driving this decision:
    color_sat_std               raw=+0.4616  scaled=0.923  w=-6.934  contrib=-6.400
    border_gradient             raw=+1.0000  scaled=1.000  w=+2.650  contrib=+2.650
    geom_compactness            raw=+7.0832  scaled=0.787  w=-3.260  contrib=-2.565
    color_darkness              raw=+0.9684  scaled=0.968  w=+2.055  contrib=+1.990
    color_val_std               raw=+0.0735  scaled=0.142  w=+11.384  contrib=+1.612
    color_rgb_std_mean          raw=+0.0563  scaled=0.111  w=-11.036  contrib=-1.221
    asym_centroid_offset        raw=+0.3357  scaled=0.671  w=+1.669  contrib=+1.121
    geom_eccentricity           raw=+0.9997  scaled=1.000  w=-1.008  contrib=-1.007
  + bias = -1.170
  = logit -4.296  →  σ(logit) = P(malignant) = 0.013  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.000 τ=0.847 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.847
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.025 > 0.018
    geom_area_ratio            scaled=0.043 <= 0.081
    geom_compactness           scaled=0.787 > 0.031
    geom_perim_ratio           scaled=0.145 > 0.087
  → leaf class counts [benign,malignant] = [1.0, 0.0]
  → leaf P(malignant) = 0.000  →  BENIGN
```

---
