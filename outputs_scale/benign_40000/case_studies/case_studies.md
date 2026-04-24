# Case studies

_Anchor pipeline: **transparent_lr**. All narratives share the same test image across every pipeline so the reader can compare explanation styles side-by-side._

## TP · malignant correctly flagged

### image_id = `ISIC_0664841`

- patient_id = `IP_4549819`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_0664841.jpg`
- anchor (transparent_lr) P(malignant) = 0.962

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.182 τ=0.010 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.182; with validation-tuned threshold τ=0.010
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.371  B=0.516  C=0.172  D=0.281.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.342 τ=0.304 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.304
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.227  w=-0.459  contribution=-0.104
    B(Border)       c=+0.353  w=-0.624  contribution=-0.220
    C(Color)        c=+0.049  w=-0.392  contribution=-0.019
    D(Diameter)     c=+0.270  w=-0.135  contribution=-0.036
  + bias = -0.275
  = logit -0.655  →  σ(logit) = P(malignant) = 0.342  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.171 τ=0.109 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.109
    f_A(0.009) = -0.208   (A(Asymmetry))
    f_B(0.037) = -0.449   (B(Border))
    f_C(0.006) = -0.679   (C(Color))
    f_D(0.016) = -0.213   (D(Diameter))
  + bias = -0.032
  = logit -1.582  →  P(malignant) = 0.171  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.962 τ=0.590 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.590
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.3466  scaled=0.693  w=+2.791  contrib=+1.935
    geom_convexity_defects      raw=+0.4400  scaled=0.647  w=+2.966  contrib=+1.919
    color_darkness              raw=+0.6053  scaled=0.600  w=+2.184  contrib=+1.311
    geom_perim_ratio            raw=+3.2010  scaled=0.315  w=+3.490  contrib=+1.098
    color_val_std               raw=+0.0704  scaled=0.135  w=+7.054  contrib=+0.954
    color_sat_std               raw=+0.0738  scaled=0.144  w=-6.392  contrib=-0.921
    geom_compactness            raw=+2.2902  scaled=0.254  w=-2.704  contrib=-0.688
    color_rgb_std_mean          raw=+0.0566  scaled=0.111  w=-6.133  contrib=-0.682
  + bias = -1.407
  = logit +3.242  →  σ(logit) = P(malignant) = 0.962  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.881 τ=0.841 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.841
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.693 > 0.018
    geom_area_ratio            scaled=0.500 > 0.081
    asym_horizontal            scaled=0.361 > 0.000
    border_gradient            scaled=0.152 > 0.032
  → leaf class counts [benign,malignant] = [0.11869958217728521, 0.8813004178227047]
  → leaf P(malignant) = 0.881  →  MALIGNANT
```

---

### image_id = `ISIC_5366444`

- patient_id = `IP_2878659`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_5366444.jpg`
- anchor (transparent_lr) P(malignant) = 0.955

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.290 τ=0.010 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.290; with validation-tuned threshold τ=0.010
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.340  B=0.417  C=0.107  D=0.265.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.431 τ=0.304 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.304
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.001  w=-0.459  contribution=-0.001
    B(Border)       c=+0.002  w=-0.624  contribution=-0.001
    C(Color)        c=+0.000  w=-0.392  contribution=-0.000
    D(Diameter)     c=+0.010  w=-0.135  contribution=-0.001
  + bias = -0.275
  = logit -0.278  →  σ(logit) = P(malignant) = 0.431  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.155 τ=0.109 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.109
    f_A(0.033) = -0.244   (A(Asymmetry))
    f_B(0.081) = -0.495   (B(Border))
    f_C(0.014) = -0.681   (C(Color))
    f_D(0.058) = -0.248   (D(Diameter))
  + bias = -0.032
  = logit -1.700  →  P(malignant) = 0.155  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.955 τ=0.590 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.590
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.3596  scaled=0.719  w=+2.791  contrib=+2.007
    geom_perim_ratio            raw=+4.0827  scaled=0.401  w=+3.490  contrib=+1.401
    geom_convexity_defects      raw=+0.3200  scaled=0.471  w=+2.966  contrib=+1.396
    color_darkness              raw=+0.5684  scaled=0.563  w=+2.184  contrib=+1.229
    geom_compactness            raw=+3.4361  scaled=0.382  w=-2.704  contrib=-1.032
    geom_eccentricity           raw=+0.5309  scaled=0.531  w=-0.971  contrib=-0.515
    asym_centroid_offset        raw=+0.1267  scaled=0.253  w=+1.672  contrib=+0.424
    color_sat_std               raw=+0.0347  scaled=0.066  w=-6.392  contrib=-0.419
  + bias = -1.407
  = logit +3.066  →  σ(logit) = P(malignant) = 0.955  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.881 τ=0.841 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.841
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.719 > 0.018
    geom_area_ratio            scaled=0.603 > 0.081
    asym_horizontal            scaled=0.126 > 0.000
    border_gradient            scaled=0.044 > 0.032
  → leaf class counts [benign,malignant] = [0.11869958217728521, 0.8813004178227047]
  → leaf P(malignant) = 0.881  →  MALIGNANT
```

---

## FN · malignant missed

### image_id = `ISIC_9420821`

- patient_id = `IP_5611762`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_9420821.jpg`
- anchor (transparent_lr) P(malignant) = 0.184

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.004 τ=0.010 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.004; with validation-tuned threshold τ=0.010
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.360  B=0.394  C=0.109  D=0.197.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.296 τ=0.304 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.304
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.384  w=-0.459  contribution=-0.176
    B(Border)       c=+0.431  w=-0.624  contribution=-0.269
    C(Color)        c=+0.312  w=-0.392  contribution=-0.122
    D(Diameter)     c=+0.160  w=-0.135  contribution=-0.022
  + bias = -0.275
  = logit -0.864  →  σ(logit) = P(malignant) = 0.296  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.054 τ=0.109 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.109
    f_A(0.387) = -0.829   (A(Asymmetry))
    f_B(0.371) = -0.944   (B(Border))
    f_C(0.185) = -0.715   (C(Color))
    f_D(0.188) = -0.353   (D(Diameter))
  + bias = -0.032
  = logit -2.873  →  P(malignant) = 0.054  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.184 τ=0.590 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.590
  Top signed contributions (w · scaled_x) driving this decision:
    color_val_std               raw=+0.1181  scaled=0.231  w=+7.054  contrib=+1.632
    color_rgb_std_mean          raw=+0.1220  scaled=0.242  w=-6.133  contrib=-1.487
    color_sat_std               raw=+0.0989  scaled=0.194  w=-6.392  contrib=-1.243
    geom_convexity_defects      raw=+0.2000  scaled=0.294  w=+2.966  contrib=+0.872
    geom_eccentricity           raw=+0.7700  scaled=0.770  w=-0.971  contrib=-0.747
    color_darkness              raw=+0.2829  scaled=0.273  w=+2.184  contrib=+0.596
    geom_compactness            raw=+1.8920  scaled=0.210  w=-2.704  contrib=-0.568
    color_variegation           raw=+0.4225  scaled=0.689  w=+0.576  contrib=+0.397
  + bias = -1.407
  = logit -1.489  →  σ(logit) = P(malignant) = 0.184  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.244 τ=0.841 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.841
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.017 <= 0.018
    color_sat_std              scaled=0.194 > 0.060
    color_val_std              scaled=0.231 > 0.128
    geom_area_ratio            scaled=0.035 <= 0.064
  → leaf class counts [benign,malignant] = [0.7556714758664721, 0.2443285241335527]
  → leaf P(malignant) = 0.244  →  BENIGN
```

---

### image_id = `ISIC_2538516`

- patient_id = `IP_6141958`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_2538516.jpg`
- anchor (transparent_lr) P(malignant) = 0.214

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.006 τ=0.010 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.006; with validation-tuned threshold τ=0.010
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.380  B=0.470  C=0.054  D=0.227.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.273 τ=0.304 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.304
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.483  w=-0.459  contribution=-0.221
    B(Border)       c=+0.620  w=-0.624  contribution=-0.387
    C(Color)        c=+0.153  w=-0.392  contribution=-0.060
    D(Diameter)     c=+0.266  w=-0.135  contribution=-0.036
  + bias = -0.275
  = logit -0.979  →  σ(logit) = P(malignant) = 0.273  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.052 τ=0.109 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.109
    f_A(0.330) = -0.741   (A(Asymmetry))
    f_B(0.420) = -1.050   (B(Border))
    f_C(0.085) = -0.698   (C(Color))
    f_D(0.234) = -0.385   (D(Diameter))
  + bias = -0.032
  = logit -2.906  →  P(malignant) = 0.052  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.214 τ=0.590 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.590
  Top signed contributions (w · scaled_x) driving this decision:
    geom_eccentricity           raw=+0.8756  scaled=0.876  w=-0.971  contrib=-0.850
    color_darkness              raw=+0.2926  scaled=0.283  w=+2.184  contrib=+0.618
    color_val_std               raw=+0.0310  scaled=0.056  w=+7.054  contrib=+0.394
    color_rgb_std_mean          raw=+0.0303  scaled=0.059  w=-6.133  contrib=-0.360
    geom_convexity_defects      raw=+0.0800  scaled=0.118  w=+2.966  contrib=+0.349
    border_radial_variance      raw=+0.3030  scaled=0.322  w=-0.726  contrib=-0.234
    asym_vertical               raw=+0.4434  scaled=0.469  w=+0.473  contrib=+0.222
    color_sat_std               raw=+0.0185  scaled=0.033  w=-6.392  contrib=-0.210
  + bias = -1.407
  = logit -1.299  →  σ(logit) = P(malignant) = 0.214  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.481 τ=0.841 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.841
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.006 <= 0.018
    color_sat_std              scaled=0.033 <= 0.060
    asym_diagonal              scaled=0.338 > 0.307
    color_darkness             scaled=0.283 > 0.241
  → leaf class counts [benign,malignant] = [0.5185007983363604, 0.4814992016634821]
  → leaf P(malignant) = 0.481  →  BENIGN
```

---

## FP · benign wrongly flagged

### image_id = `ISIC_6569920`

- patient_id = `IP_8494850`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_6569920.jpg`
- anchor (transparent_lr) P(malignant) = 0.991

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.658 τ=0.010 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.658; with validation-tuned threshold τ=0.010
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.230  B=0.326  C=0.141  D=0.184.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.432 τ=0.304 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.304
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.000  w=-0.459  contribution=-0.000
    B(Border)       c=+0.000  w=-0.624  contribution=-0.000
    C(Color)        c=+0.000  w=-0.392  contribution=-0.000
    D(Diameter)     c=+0.000  w=-0.135  contribution=-0.000
  + bias = -0.275
  = logit -0.275  →  σ(logit) = P(malignant) = 0.432  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.179 τ=0.109 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.109
    f_A(0.000) = -0.196   (A(Asymmetry))
    f_B(0.000) = -0.413   (B(Border))
    f_C(0.000) = -0.678   (C(Color))
    f_D(0.000) = -0.201   (D(Diameter))
  + bias = -0.032
  = logit -1.520  →  P(malignant) = 0.179  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.991 τ=0.590 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.590
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.4749  scaled=0.950  w=+2.791  contrib=+2.651
    geom_convexity_defects      raw=+0.5200  scaled=0.765  w=+2.966  contrib=+2.268
    color_darkness              raw=+0.4540  scaled=0.447  w=+2.184  contrib=+0.975
    color_val_std               raw=+0.0656  scaled=0.126  w=+7.054  contrib=+0.886
    geom_eccentricity           raw=+0.8068  scaled=0.807  w=-0.971  contrib=-0.783
    color_sat_std               raw=+0.0564  scaled=0.109  w=-6.392  contrib=-0.697
    geom_perim_ratio            raw=+2.0008  scaled=0.197  w=+3.490  contrib=+0.686
    color_rgb_std_mean          raw=+0.0451  scaled=0.088  w=-6.133  contrib=-0.542
  + bias = -1.407
  = logit +4.669  →  σ(logit) = P(malignant) = 0.991  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.881 τ=0.841 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.841
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.950 > 0.018
    geom_area_ratio            scaled=0.368 > 0.081
    asym_horizontal            scaled=0.358 > 0.000
    border_gradient            scaled=0.067 > 0.032
  → leaf class counts [benign,malignant] = [0.11869958217728521, 0.8813004178227047]
  → leaf P(malignant) = 0.881  →  MALIGNANT
```

---

### image_id = `ISIC_2759370`

- patient_id = `IP_7272992`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_2759370.jpg`
- anchor (transparent_lr) P(malignant) = 0.983

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.029 τ=0.010 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.029; with validation-tuned threshold τ=0.010
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.296  B=0.410  C=0.132  D=0.343.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.277 τ=0.304 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.304
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.406  w=-0.459  contribution=-0.186
    B(Border)       c=+0.536  w=-0.624  contribution=-0.335
    C(Color)        c=+0.297  w=-0.392  contribution=-0.116
    D(Diameter)     c=+0.360  w=-0.135  contribution=-0.049
  + bias = -0.275
  = logit -0.961  →  σ(logit) = P(malignant) = 0.277  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.053 τ=0.109 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.109
    f_A(0.232) = -0.580   (A(Asymmetry))
    f_B(0.482) = -1.172   (B(Border))
    f_C(0.055) = -0.691   (C(Color))
    f_D(0.273) = -0.407   (D(Diameter))
  + bias = -0.032
  = logit -2.882  →  P(malignant) = 0.053  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.983 τ=0.590 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.590
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.3716  scaled=0.743  w=+2.791  contrib=+2.074
    border_gradient             raw=+0.7096  scaled=0.704  w=+2.362  contrib=+1.663
    color_darkness              raw=+0.7486  scaled=0.746  w=+2.184  contrib=+1.629
    color_sat_std               raw=+0.0769  scaled=0.150  w=-6.392  contrib=-0.961
    asym_centroid_offset        raw=+0.2536  scaled=0.507  w=+1.672  contrib=+0.848
    geom_perim_ratio            raw=+1.8270  scaled=0.180  w=+3.490  contrib=+0.627
    geom_eccentricity           raw=+0.5834  scaled=0.583  w=-0.971  contrib=-0.566
    color_val_std               raw=+0.0284  scaled=0.051  w=+7.054  contrib=+0.358
  + bias = -1.407
  = logit +4.075  →  σ(logit) = P(malignant) = 0.983  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.881 τ=0.841 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.841
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.743 > 0.018
    geom_area_ratio            scaled=0.270 > 0.081
    asym_horizontal            scaled=0.282 > 0.000
    border_gradient            scaled=0.704 > 0.032
  → leaf class counts [benign,malignant] = [0.11869958217728521, 0.8813004178227047]
  → leaf P(malignant) = 0.881  →  MALIGNANT
```

---

## TN · benign correctly cleared

### image_id = `ISIC_5313485`

- patient_id = `IP_4549819`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_5313485.jpg`
- anchor (transparent_lr) P(malignant) = 0.015

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.005 τ=0.010 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.005; with validation-tuned threshold τ=0.010
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.435  B=0.465  C=0.088  D=0.199.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.266 τ=0.304 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.304
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.448  w=-0.459  contribution=-0.206
    B(Border)       c=+0.715  w=-0.624  contribution=-0.446
    C(Color)        c=+0.141  w=-0.392  contribution=-0.055
    D(Diameter)     c=+0.237  w=-0.135  contribution=-0.032
  + bias = -0.275
  = logit -1.014  →  σ(logit) = P(malignant) = 0.266  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.038 τ=0.109 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.109
    f_A(0.326) = -0.734   (A(Asymmetry))
    f_B(0.606) = -1.377   (B(Border))
    f_C(0.049) = -0.690   (C(Color))
    f_D(0.244) = -0.392   (D(Diameter))
  + bias = -0.032
  = logit -3.224  →  P(malignant) = 0.038  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.015 τ=0.590 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.590
  Top signed contributions (w · scaled_x) driving this decision:
    color_sat_std               raw=+0.4741  scaled=0.948  w=-6.392  contrib=-6.059
    border_gradient             raw=+1.0000  scaled=1.000  w=+2.362  contrib=+2.362
    color_darkness              raw=+0.9723  scaled=0.972  w=+2.184  contrib=+2.124
    geom_compactness            raw=+6.0891  scaled=0.677  w=-2.704  contrib=-1.829
    asym_centroid_offset        raw=+0.3331  scaled=0.666  w=+1.672  contrib=+1.114
    color_val_std               raw=+0.0756  scaled=0.146  w=+7.054  contrib=+1.028
    geom_eccentricity           raw=+0.9999  scaled=1.000  w=-0.971  contrib=-0.970
    color_rgb_std_mean          raw=+0.0637  scaled=0.126  w=-6.133  contrib=-0.770
  + bias = -1.407
  = logit -4.161  →  σ(logit) = P(malignant) = 0.015  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.000 τ=0.841 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.841
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.111 > 0.018
    geom_area_ratio            scaled=0.050 <= 0.081
    asym_diagonal              scaled=1.000 > 0.467
    geom_perim_ratio           scaled=0.146 > 0.087
  → leaf class counts [benign,malignant] = [1.0, 0.0]
  → leaf P(malignant) = 0.000  →  BENIGN
```

---

### image_id = `ISIC_3115416`

- patient_id = `IP_4414342`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_3115416.jpg`
- anchor (transparent_lr) P(malignant) = 0.018

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.004 τ=0.010 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.004; with validation-tuned threshold τ=0.010
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.282  B=0.307  C=0.105  D=0.220.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.281 τ=0.304 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.304
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.365  w=-0.459  contribution=-0.167
    B(Border)       c=+0.649  w=-0.624  contribution=-0.405
    C(Color)        c=+0.147  w=-0.392  contribution=-0.057
    D(Diameter)     c=+0.272  w=-0.135  contribution=-0.037
  + bias = -0.275
  = logit -0.942  →  σ(logit) = P(malignant) = 0.281  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.061 τ=0.109 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.109
    f_A(0.213) = -0.545   (A(Asymmetry))
    f_B(0.431) = -1.073   (B(Border))
    f_C(0.082) = -0.697   (C(Color))
    f_D(0.243) = -0.392   (D(Diameter))
  + bias = -0.032
  = logit -2.740  →  P(malignant) = 0.061  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.018 τ=0.590 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.590
  Top signed contributions (w · scaled_x) driving this decision:
    color_sat_std               raw=+0.4616  scaled=0.923  w=-6.392  contrib=-5.899
    border_gradient             raw=+1.0000  scaled=1.000  w=+2.362  contrib=+2.362
    geom_compactness            raw=+7.0832  scaled=0.787  w=-2.704  contrib=-2.128
    color_darkness              raw=+0.9684  scaled=0.969  w=+2.184  contrib=+2.116
    asym_centroid_offset        raw=+0.3357  scaled=0.671  w=+1.672  contrib=+1.123
    color_val_std               raw=+0.0735  scaled=0.141  w=+7.054  contrib=+0.998
    geom_eccentricity           raw=+0.9997  scaled=1.000  w=-0.971  contrib=-0.970
    color_rgb_std_mean          raw=+0.0563  scaled=0.111  w=-6.133  contrib=-0.678
  + bias = -1.407
  = logit -4.014  →  σ(logit) = P(malignant) = 0.018  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.000 τ=0.841 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.841
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.025 > 0.018
    geom_area_ratio            scaled=0.043 <= 0.081
    asym_diagonal              scaled=1.000 > 0.467
    geom_perim_ratio           scaled=0.145 > 0.087
  → leaf class counts [benign,malignant] = [1.0, 0.0]
  → leaf P(malignant) = 0.000  →  BENIGN
```

---
