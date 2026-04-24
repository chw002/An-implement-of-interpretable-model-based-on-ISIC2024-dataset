```text
I want you to build my final-year project from scratch as a clean, research-grade Python project.

Project goal:
Build a 100% interpretable neuro-symbolic reasoning system for skin lesion classification using the official ISIC 2024 dataset. The system must combine neural perception with symbolic reasoning, but the final decision path must remain fully human-readable and reproducible.

Core principle:
The neural network is only allowed to do perception and concept extraction. It must not be the final black-box decision-maker.
The final prediction must be made by transparent reasoning modules such as:
1. a symbolic rule engine,
2. a linear concept bottleneck classifier,
3. a single decision tree arbitrator.
No opaque ensemble, no XGBoost, and no black-box MLP is allowed in the final decision path.

Concrete task:
Use dermoscopic skin lesion images from the official ISIC 2024 dataset and map them to clinically meaningful ABCD concepts:
- A: Asymmetry
- B: Border irregularity
- C: Color variation
- D: Diameter / structural complexity

Then use these concepts for transparent diagnosis.

What I want the project to include:
1. A clean repo structure, not a single large script.
2. Python + PyTorch implementation.
3. Modular files for:
   - data loading
   - preprocessing
   - ABCD concept extraction
   - neural models
   - symbolic reasoning
   - arbitration / meta-learner
   - training
   - evaluation
   - analysis / plotting
4. A main entry script or CLI to reproduce the full experiment pipeline.
5. Proper dependency file: requirements.txt or pyproject.toml.
6. A clear README explaining:
   - project motivation
   - architecture
   - why the project is based on ISIC 2024 official data
   - how to run training
   - how to run evaluation
   - how interpretability is guaranteed
7. Saved outputs:
   - checkpoints
   - tables
   - figures
   - case studies
   - interpretable explanation reports

Required architecture:
1. Perception model:
   - a CNN backbone that predicts ABCD concept scores from images
   - optionally also train a multitask CNN baseline for comparison
2. Hard Concept Bottleneck Model:
   - image -> ABCD concepts -> single linear layer -> label
   - final formula must be readable as:
     logit = w_A*A + w_B*B + w_C*C + w_D*D + bias
3. Causal Concept Bottleneck Model:
   - same transparent linear label layer
   - add concept intervention and decorrelation during training
4. Glass-box additive model:
   - concept-based additive model with independent shape functions
   - still fully interpretable
5. Symbolic rule engine:
   - implement ABCD rule scoring and threshold-based reasoning
   - output both prediction and natural-language reasoning trace
6. Transparent arbitration:
   - if neural and symbolic predictions disagree, use only a single decision tree
   - every rule path must be exportable as readable text

Interpretability constraints:
- Every final prediction must be explainable step by step.
- A human should be able to reconstruct the final decision from concept values and rules.
- Do not rely on post-hoc explanations like Grad-CAM as the main explanation method.
- Interpretability must be intrinsic, not decorative.

Evaluation requirements:
1. Standard metrics:
   - accuracy
   - AUC
   - F1
   - precision
   - recall
   - confusion matrix
2. Interpretability-related analysis:
   - concept prediction quality
   - concept completeness
   - faithfulness
   - concept separation
3. Ablation studies:
   - remove each concept one by one
   - compare arbitration strategies
   - compare interpretable baselines
4. Case studies:
   - export several examples with full reasoning chains
5. Threshold optimization:
   - tune thresholds on validation set only
6. Data leakage prevention:
   - use patient-disjoint train/val/test split if patient IDs exist
   - fit scalers only on training split
7. Dataset requirements:
   - inspect the current workspace and reuse the official ISIC 2024 dataset files if already present
   - preserve the distinction between raw data, processed data, checkpoints, and results
   - avoid assumptions that conflict with the official dataset structure

Engineering requirements:
- Keep the code modular and readable.
- Avoid giant monolithic scripts.
- Add concise comments only where needed.
- Prefer reproducibility and clarity over cleverness.
- Make reasonable assumptions and proceed unless blocked.
- If some dataset details are missing, inspect the workspace and adapt.
- If the workspace already contains partial code or data, reuse what is useful but refactor aggressively into a cleaner structure.

Execution plan:
1. First inspect the current workspace and summarize what already exists.
2. Then propose a clean target structure briefly.
3. Then implement the full project step by step.
4. Run sanity checks or lightweight verification where possible.
5. Finish by summarizing:
   - what was built
   - how to run it
   - which files are most important
   - any remaining risks or limitations

Important:
I care more about true end-to-end interpretability than maximizing raw benchmark performance.
Optimize for a credible final-year research project that is technically clean, academically defensible, and easy for a supervisor to understand.
```
