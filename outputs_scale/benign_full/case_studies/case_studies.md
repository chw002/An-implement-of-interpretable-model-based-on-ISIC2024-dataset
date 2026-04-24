# Case studies

_Anchor pipeline: **transparent_lr**. All narratives share the same test image across every pipeline so the reader can compare explanation styles side-by-side._

## TP · malignant correctly flagged

### image_id = `ISIC_0664841`

- patient_id = `IP_4549819`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_0664841.jpg`
- anchor (transparent_lr) P(malignant) = 0.973

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.025 τ=0.011 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.025; with validation-tuned threshold τ=0.011
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.422  B=0.463  C=0.112  D=0.214.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.240 τ=0.177 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.177
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.222  w=-0.310  contribution=-0.069
    B(Border)       c=+0.235  w=-0.874  contribution=-0.205
    C(Color)        c=+0.072  w=-0.466  contribution=-0.034
    D(Diameter)     c=+0.197  w=-0.590  contribution=-0.116
  + bias = -0.730
  = logit -1.153  →  σ(logit) = P(malignant) = 0.240  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.035 τ=0.018 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.018
    f_A(0.278) = -1.281   (A(Asymmetry))
    f_B(0.462) = -1.185   (B(Border))
    f_C(0.126) = -0.765   (C(Color))
    f_D(0.545) = -0.050   (D(Diameter))
  + bias = -0.027
  = logit -3.308  →  P(malignant) = 0.035  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.973 τ=0.526 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.526
  Top signed contributions (w · scaled_x) driving this decision:
    color_val_std               raw=+0.0704  scaled=0.138  w=+17.130  contrib=+2.366
    color_hue_std               raw=+0.3466  scaled=0.693  w=+2.925  contrib=+2.028
    geom_perim_ratio            raw=+3.2010  scaled=0.315  w=+6.188  contrib=+1.947
    color_rgb_std_mean          raw=+0.0566  scaled=0.112  w=-17.373  contrib=-1.940
    geom_convexity_defects      raw=+0.4400  scaled=0.579  w=+3.269  contrib=+1.893
    color_darkness              raw=+0.6053  scaled=0.600  w=+1.798  contrib=+1.079
    color_sat_std               raw=+0.0738  scaled=0.144  w=-6.978  contrib=-1.006
    geom_compactness            raw=+2.2902  scaled=0.254  w=-3.898  contrib=-0.992
  + bias = -0.820
  = logit +3.577  →  σ(logit) = P(malignant) = 0.973  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.879 τ=0.846 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.846
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.693 > 0.018
    geom_area_ratio            scaled=0.500 > 0.081
    asym_horizontal            scaled=0.355 > 0.000
    border_gradient            scaled=0.161 > 0.042
  → leaf class counts [benign,malignant] = [0.1210598316769907, 0.8789401683228204]
  → leaf P(malignant) = 0.879  →  MALIGNANT
```

---

### image_id = `ISIC_7748442`

- patient_id = `IP_0273153`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_7748442.jpg`
- anchor (transparent_lr) P(malignant) = 0.961

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.043 τ=0.011 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.043; with validation-tuned threshold τ=0.011
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.347  B=0.326  C=0.076  D=0.273.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.205 τ=0.177 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.177
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.296  w=-0.310  contribution=-0.092
    B(Border)       c=+0.403  w=-0.874  contribution=-0.352
    C(Color)        c=+0.069  w=-0.466  contribution=-0.032
    D(Diameter)     c=+0.253  w=-0.590  contribution=-0.149
  + bias = -0.730
  = logit -1.355  →  σ(logit) = P(malignant) = 0.205  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.059 τ=0.018 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.018
    f_A(0.107) = -1.227   (A(Asymmetry))
    f_B(0.382) = -1.167   (B(Border))
    f_C(0.039) = -0.755   (C(Color))
    f_D(0.847) = +0.401   (D(Diameter))
  + bias = -0.027
  = logit -2.775  →  P(malignant) = 0.059  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.961 τ=0.526 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.526
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.4678  scaled=0.936  w=+2.925  contrib=+2.737
    geom_convexity_defects      raw=+0.4400  scaled=0.579  w=+3.269  contrib=+1.893
    geom_perim_ratio            raw=+2.1256  scaled=0.209  w=+6.188  contrib=+1.293
    color_rgb_std_mean          raw=+0.0326  scaled=0.064  w=-17.373  contrib=-1.105
    color_val_std               raw=+0.0279  scaled=0.053  w=+17.130  contrib=+0.906
    geom_eccentricity           raw=+0.8439  scaled=0.844  w=-1.032  contrib=-0.871
    geom_compactness            raw=+1.5781  scaled=0.175  w=-3.898  contrib=-0.683
    color_darkness              raw=+0.2764  scaled=0.266  w=+1.798  contrib=+0.479
  + bias = -0.820
  = logit +3.195  →  σ(logit) = P(malignant) = 0.961  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.879 τ=0.846 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.846
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.936 > 0.018
    geom_area_ratio            scaled=0.281 > 0.081
    asym_horizontal            scaled=0.400 > 0.000
    border_gradient            scaled=0.072 > 0.042
  → leaf class counts [benign,malignant] = [0.1210598316769907, 0.8789401683228204]
  → leaf P(malignant) = 0.879  →  MALIGNANT
```

---

## FN · malignant missed

### image_id = `ISIC_9420821`

- patient_id = `IP_5611762`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_9420821.jpg`
- anchor (transparent_lr) P(malignant) = 0.147

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.011 τ=0.011 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.011; with validation-tuned threshold τ=0.011
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.318  B=0.289  C=0.095  D=0.150.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.186 τ=0.177 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.177
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.360  w=-0.310  contribution=-0.112
    B(Border)       c=+0.439  w=-0.874  contribution=-0.384
    C(Color)        c=+0.189  w=-0.466  contribution=-0.088
    D(Diameter)     c=+0.278  w=-0.590  contribution=-0.164
  + bias = -0.730
  = logit -1.477  →  σ(logit) = P(malignant) = 0.186  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.008 τ=0.018 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.018
    f_A(0.324) = -1.297   (A(Asymmetry))
    f_B(0.258) = -1.193   (B(Border))
    f_C(0.116) = -0.764   (C(Color))
    f_D(0.114) = -1.538   (D(Diameter))
  + bias = -0.027
  = logit -4.819  →  P(malignant) = 0.008  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.147 τ=0.526 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.526
  Top signed contributions (w · scaled_x) driving this decision:
    color_rgb_std_mean          raw=+0.1220  scaled=0.243  w=-17.373  contrib=-4.218
    color_val_std               raw=+0.1181  scaled=0.234  w=+17.130  contrib=+4.006
    color_sat_std               raw=+0.0989  scaled=0.194  w=-6.978  contrib=-1.357
    geom_convexity_defects      raw=+0.2000  scaled=0.263  w=+3.269  contrib=+0.860
    geom_compactness            raw=+1.8920  scaled=0.210  w=-3.898  contrib=-0.819
    geom_eccentricity           raw=+0.7700  scaled=0.770  w=-1.032  contrib=-0.795
    color_darkness              raw=+0.2829  scaled=0.273  w=+1.798  contrib=+0.491
    geom_perim_ratio            raw=+0.7913  scaled=0.078  w=+6.188  contrib=+0.481
  + bias = -0.820
  = logit -1.757  →  σ(logit) = P(malignant) = 0.147  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.242 τ=0.846 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.846
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.017 <= 0.018
    color_sat_std              scaled=0.194 > 0.060
    color_val_std              scaled=0.234 > 0.130
    geom_area_ratio            scaled=0.035 <= 0.064
  → leaf class counts [benign,malignant] = [0.7578400681537809, 0.2421599318462191]
  → leaf P(malignant) = 0.242  →  BENIGN
```

---

### image_id = `ISIC_2538516`

- patient_id = `IP_6141958`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_2538516.jpg`
- anchor (transparent_lr) P(malignant) = 0.232

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.005 τ=0.011 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.005; with validation-tuned threshold τ=0.011
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.402  B=0.452  C=0.046  D=0.216.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.175 τ=0.177 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.177
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.404  w=-0.310  contribution=-0.125
    B(Border)       c=+0.529  w=-0.874  contribution=-0.462
    C(Color)        c=+0.092  w=-0.466  contribution=-0.043
    D(Diameter)     c=+0.323  w=-0.590  contribution=-0.190
  + bias = -0.730
  = logit -1.551  →  σ(logit) = P(malignant) = 0.175  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.012 τ=0.018 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.018
    f_A(0.296) = -1.287   (A(Asymmetry))
    f_B(0.272) = -1.190   (B(Border))
    f_C(0.086) = -0.761   (C(Color))
    f_D(0.210) = -1.174   (D(Diameter))
  + bias = -0.027
  = logit -4.438  →  P(malignant) = 0.012  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.232 τ=0.526 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.526
  Top signed contributions (w · scaled_x) driving this decision:
    color_rgb_std_mean          raw=+0.0303  scaled=0.059  w=-17.373  contrib=-1.026
    color_val_std               raw=+0.0310  scaled=0.059  w=+17.130  contrib=+1.010
    geom_eccentricity           raw=+0.8756  scaled=0.876  w=-1.032  contrib=-0.903
    color_darkness              raw=+0.2926  scaled=0.283  w=+1.798  contrib=+0.508
    geom_convexity_defects      raw=+0.0800  scaled=0.105  w=+3.269  contrib=+0.344
    geom_compactness            raw=+0.6785  scaled=0.075  w=-3.898  contrib=-0.294
    asym_vertical               raw=+0.4434  scaled=0.459  w=+0.602  contrib=+0.277
    border_radial_variance      raw=+0.3030  scaled=0.303  w=-0.863  contrib=-0.262
  + bias = -0.820
  = logit -1.196  →  σ(logit) = P(malignant) = 0.232  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.483 τ=0.846 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.846
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.006 <= 0.018
    color_sat_std              scaled=0.033 <= 0.060
    asym_diagonal              scaled=0.338 > 0.307
    color_darkness             scaled=0.283 > 0.241
  → leaf class counts [benign,malignant] = [0.5165794366788962, 0.48342056332093297]
  → leaf P(malignant) = 0.483  →  BENIGN
```

---

## FP · benign wrongly flagged

### image_id = `ISIC_6569920`

- patient_id = `IP_8494850`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_6569920.jpg`
- anchor (transparent_lr) P(malignant) = 0.995

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.146 τ=0.011 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.146; with validation-tuned threshold τ=0.011
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.304  B=0.388  C=0.137  D=0.269.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.240 τ=0.177 → pred=M / verdict=MALIGNANT

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.177
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.147  w=-0.310  contribution=-0.046
    B(Border)       c=+0.231  w=-0.874  contribution=-0.202
    C(Color)        c=+0.124  w=-0.466  contribution=-0.058
    D(Diameter)     c=+0.199  w=-0.590  contribution=-0.118
  + bias = -0.730
  = logit -1.152  →  σ(logit) = P(malignant) = 0.240  →  MALIGNANT
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.049 τ=0.018 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.018
    f_A(0.118) = -1.229   (A(Asymmetry))
    f_B(0.319) = -1.180   (B(Border))
    f_C(0.073) = -0.759   (C(Color))
    f_D(0.709) = +0.221   (D(Diameter))
  + bias = -0.027
  = logit -2.975  →  P(malignant) = 0.049  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.995 τ=0.526 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.526
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.4749  scaled=0.950  w=+2.925  contrib=+2.778
    geom_convexity_defects      raw=+0.5200  scaled=0.684  w=+3.269  contrib=+2.237
    color_val_std               raw=+0.0656  scaled=0.129  w=+17.130  contrib=+2.202
    color_rgb_std_mean          raw=+0.0451  scaled=0.089  w=-17.373  contrib=-1.542
    geom_perim_ratio            raw=+2.0008  scaled=0.197  w=+6.188  contrib=+1.217
    geom_eccentricity           raw=+0.8068  scaled=0.807  w=-1.032  contrib=-0.832
    color_darkness              raw=+0.4540  scaled=0.446  w=+1.798  contrib=+0.803
    color_sat_std               raw=+0.0564  scaled=0.109  w=-6.978  contrib=-0.761
  + bias = -0.820
  = logit +5.199  →  σ(logit) = P(malignant) = 0.995  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.879 τ=0.846 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.846
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.950 > 0.018
    geom_area_ratio            scaled=0.368 > 0.081
    asym_horizontal            scaled=0.352 > 0.000
    border_gradient            scaled=0.077 > 0.042
  → leaf class counts [benign,malignant] = [0.1210598316769907, 0.8789401683228204]
  → leaf P(malignant) = 0.879  →  MALIGNANT
```

---

### image_id = `ISIC_2759370`

- patient_id = `IP_7272992`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_2759370.jpg`
- anchor (transparent_lr) P(malignant) = 0.987

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.012 τ=0.011 → pred=M / verdict=MALIGNANT

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.012; with validation-tuned threshold τ=0.011
this is classified as MALIGNANT.  The auxiliary ABCD head reported
A=0.238  B=0.556  C=0.118  D=0.366.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.174 τ=0.177 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.177
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.300  w=-0.310  contribution=-0.093
    B(Border)       c=+0.473  w=-0.874  contribution=-0.413
    C(Color)        c=+0.158  w=-0.466  contribution=-0.073
    D(Diameter)     c=+0.423  w=-0.590  contribution=-0.249
  + bias = -0.730
  = logit -1.559  →  σ(logit) = P(malignant) = 0.174  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.027 τ=0.018 → pred=M / verdict=MALIGNANT

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.018
    f_A(0.262) = -1.275   (A(Asymmetry))
    f_B(0.411) = -1.166   (B(Border))
    f_C(0.148) = -0.768   (C(Color))
    f_D(0.439) = -0.331   (D(Diameter))
  + bias = -0.027
  = logit -3.567  →  P(malignant) = 0.027  →  MALIGNANT
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.987 τ=0.526 → pred=M / verdict=MALIGNANT

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.526
  Top signed contributions (w · scaled_x) driving this decision:
    color_hue_std               raw=+0.3716  scaled=0.743  w=+2.925  contrib=+2.174
    border_gradient             raw=+0.7096  scaled=0.707  w=+2.784  contrib=+1.968
    color_darkness              raw=+0.7486  scaled=0.745  w=+1.798  contrib=+1.340
    geom_perim_ratio            raw=+1.8270  scaled=0.180  w=+6.188  contrib=+1.111
    color_sat_std               raw=+0.0769  scaled=0.150  w=-6.978  contrib=-1.050
    color_val_std               raw=+0.0284  scaled=0.054  w=+17.130  contrib=+0.923
    color_rgb_std_mean          raw=+0.0266  scaled=0.052  w=-17.373  contrib=-0.896
    asym_centroid_offset        raw=+0.2536  scaled=0.507  w=+1.648  contrib=+0.836
  + bias = -0.820
  = logit +4.328  →  σ(logit) = P(malignant) = 0.987  →  MALIGNANT
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.879 τ=0.846 → pred=M / verdict=MALIGNANT

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.846
  Tree max depth = 4   min samples / leaf = 30
  Rule path taken:
    color_hue_std              scaled=0.743 > 0.018
    geom_area_ratio            scaled=0.270 > 0.081
    asym_horizontal            scaled=0.278 > 0.000
    border_gradient            scaled=0.707 > 0.042
  → leaf class counts [benign,malignant] = [0.1210598316769907, 0.8789401683228204]
  → leaf P(malignant) = 0.879  →  MALIGNANT
```

---

## TN · benign correctly cleared

### image_id = `ISIC_5313485`

- patient_id = `IP_4549819`
- image_path = `/home/chw/222/data/raw/isic2024_official/ISIC_2024_Permissive_Training_Input/ISIC_5313485.jpg`
- anchor (transparent_lr) P(malignant) = 0.009

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.006 τ=0.011 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.006; with validation-tuned threshold τ=0.011
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.432  B=0.491  C=0.075  D=0.214.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.119 τ=0.177 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.177
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.310  w=-0.310  contribution=-0.096
    B(Border)       c=+0.930  w=-0.874  contribution=-0.813
    C(Color)        c=+0.194  w=-0.466  contribution=-0.090
    D(Diameter)     c=+0.457  w=-0.590  contribution=-0.270
  + bias = -0.730
  = logit -1.999  →  σ(logit) = P(malignant) = 0.119  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.013 τ=0.018 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.018
    f_A(0.237) = -1.266   (A(Asymmetry))
    f_B(0.965) = -1.640   (B(Border))
    f_C(0.136) = -0.766   (C(Color))
    f_D(0.344) = -0.659   (D(Diameter))
  + bias = -0.027
  = logit -4.358  →  P(malignant) = 0.013  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.009 τ=0.526 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.526
  Top signed contributions (w · scaled_x) driving this decision:
    color_sat_std               raw=+0.4741  scaled=0.948  w=-6.978  contrib=-6.615
    border_gradient             raw=+1.0000  scaled=1.000  w=+2.784  contrib=+2.784
    geom_compactness            raw=+6.0891  scaled=0.677  w=-3.898  contrib=-2.637
    color_val_std               raw=+0.0756  scaled=0.149  w=+17.130  contrib=+2.544
    color_rgb_std_mean          raw=+0.0637  scaled=0.126  w=-17.373  contrib=-2.188
    color_darkness              raw=+0.9723  scaled=0.972  w=+1.798  contrib=+1.748
    asym_centroid_offset        raw=+0.3331  scaled=0.666  w=+1.648  contrib=+1.098
    geom_eccentricity           raw=+0.9999  scaled=1.000  w=-1.032  contrib=-1.032
  + bias = -0.820
  = logit -4.713  →  σ(logit) = P(malignant) = 0.009  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.000 τ=0.846 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.846
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
- anchor (transparent_lr) P(malignant) = 0.011

**multitask_cnn** (`image_only_baseline`) — P(mal)=0.001 τ=0.011 → pred=B / verdict=BENIGN

```
MultiTaskCNN — image-only baseline.  The CNN's MLP classifier produced
P(malignant) = 0.001; with validation-tuned threshold τ=0.011
this is classified as BENIGN.  The auxiliary ABCD head reported
A=0.294  B=0.328  C=0.065  D=0.208.
These concepts are associated image-derived observations; they did not
participate in the final decision (the decision is produced by the MLP head).
```

**hard_cbm** (`interpretable_partial`) — P(mal)=0.123 τ=0.177 → pred=B / verdict=BENIGN

```
HardCBM — linear readout: logit = Σ wᵢ·cᵢ + bias.
  Validation-tuned threshold τ = 0.177
  CNN-predicted concepts, supervised by pixel-derived proxies ∈ [0,1]:
    A(Asymmetry)    c=+0.298  w=-0.310  contribution=-0.093
    B(Border)       c=+0.895  w=-0.874  contribution=-0.782
    C(Color)        c=+0.179  w=-0.466  contribution=-0.084
    D(Diameter)     c=+0.464  w=-0.590  contribution=-0.273
  + bias = -0.730
  = logit -1.962  →  σ(logit) = P(malignant) = 0.123  →  BENIGN
```

**glassbox_nam** (`interpretable_partial`) — P(mal)=0.012 τ=0.018 → pred=B / verdict=BENIGN

```
GlassBoxNAM — additive readout: logit = Σ fᵢ(cᵢ) + bias.
  Validation-tuned threshold τ = 0.018
    f_A(0.263) = -1.275   (A(Asymmetry))
    f_B(0.918) = -1.593   (B(Border))
    f_C(0.147) = -0.767   (C(Color))
    f_D(0.321) = -0.750   (D(Diameter))
  + bias = -0.027
  = logit -4.412  →  P(malignant) = 0.012  →  BENIGN
```

**transparent_lr** (`fully_auditable`) — P(mal)=0.011 τ=0.526 → pred=B / verdict=BENIGN

```
Transparent-LR — logistic regression on pixel-derived features.
  Validation-tuned threshold τ = 0.526
  Top signed contributions (w · scaled_x) driving this decision:
    color_sat_std               raw=+0.4616  scaled=0.923  w=-6.978  contrib=-6.440
    geom_compactness            raw=+7.0832  scaled=0.787  w=-3.898  contrib=-3.067
    border_gradient             raw=+1.0000  scaled=1.000  w=+2.784  contrib=+2.784
    color_val_std               raw=+0.0735  scaled=0.144  w=+17.130  contrib=+2.473
    color_rgb_std_mean          raw=+0.0563  scaled=0.111  w=-17.373  contrib=-1.929
    color_darkness              raw=+0.9684  scaled=0.968  w=+1.798  contrib=+1.741
    asym_centroid_offset        raw=+0.3357  scaled=0.671  w=+1.648  contrib=+1.106
    geom_eccentricity           raw=+0.9997  scaled=1.000  w=-1.032  contrib=-1.032
  + bias = -0.820
  = logit -4.521  →  σ(logit) = P(malignant) = 0.011  →  BENIGN
```

**transparent_tree** (`fully_auditable`) — P(mal)=0.000 τ=0.846 → pred=B / verdict=BENIGN

```
Transparent-Tree — shallow decision tree on pixel-derived features.
  Validation-tuned threshold τ = 0.846
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
