# BEED — Lab Notebook

**Dataset:** Bangalore EEG Epilepsy Dataset (BEED)  
**Goal:** Reproduce SeqBoostNet (IJCDS 2025) — 4-class EEG seizure classification  
**Pipeline:** FFT + UMAP features → LSTM + XGBoost + GradientBoosting → AdaBoost

---

## EXP-001 — Exploratory Data Analysis

**Date:** 2026-06-14  
**Notebook:** `notebooks/01_eda.ipynb`  
**Figures:** `reports/figures/01–06_*.png`

### Setup

| Parameter | Value |
|-----------|-------|
| Dataset rows | 8,000 (2,000 per class) |
| Features | 16 EEG channels (X1–X16 / Fp1–T6) |
| Sampling rate | 256 Hz |
| Recording length | 20 s/session |
| Missing values | 0 |
| Split | 70 / 10 / 20 (train / val / test), stratified |

> **Note:** The reference paper states 4,000 rows; the actual CSV has 8,000. All analysis uses the full 8,000-row file.

---

### Findings

#### F1 — Dataset is perfectly balanced
All four classes have exactly **2,000 samples**. No class weighting or oversampling is needed.

#### F2 — Healthy class has anomalously high amplitude
The Healthy class shows a median RMS of ~60 μV across channels, versus ~10 μV for all three seizure classes. This is counterintuitive (seizures are clinically high-amplitude events), and likely reflects that each CSV row is a single time-point snapshot: the Healthy recordings contain high-amplitude slow baseline drift, while the seizure recordings may represent shorter, more stationary windows.

> **Implication:** RMS amplitude alone is a strong linear discriminator between Healthy vs. all-seizure, but will not separate the three seizure classes from each other.

#### F3 — Healthy EEG has flat broadband power; seizure classes are low-power and low-frequency
PSD (Welch) shows:
- **Healthy:** flat spectrum from δ through γ, ~10–100× higher power than seizure classes
- **Focal & Generalized:** similar spectral shapes; both decay after ~5 Hz
- **Seizure Events:** sharpest high-frequency rolloff; drops 3 orders of magnitude by 30 Hz

> **Implication:** FFT features will easily separate Healthy from seizures. The hard cases are A1 (Generalized vs. Focal) and potentially A4/A5 (Focal/Generalized vs. Seizure Events).

#### F4 — Healthy has a systematic DC offset (~−13 μV)
Channel mean heatmap shows all 16 channels in the Healthy class carry a consistent ~−13 μV mean. Focal shows a mixed pattern (Fp1, C3, O1, T3 positive; F4, F8 negative), suggesting localized asymmetry. Generalized and Seizure Events are near-zero.

> **Implication:** Channel-level standardisation (per-channel StandardScaler, as applied in the paper) is essential before FFT to remove DC bias. Already planned in the pipeline.

#### F5 — Focal seizures show cross-hemisphere anti-correlation
Correlation matrices show that in Focal class, ipsilateral and contralateral channel pairs have opposing signs (e.g., left-frontal vs. right-temporal negatively correlated). This is consistent with focal onset: one hemisphere fires while the other is suppressed. Healthy channels are uniformly positively correlated. Generalized and Seizure Events show weaker but mostly positive correlations.

> **Implication:** UMAP should capture this structural difference well (it preserves manifold geometry). The 3 UMAP dimensions may encode inter-hemispheric relationship, not just amplitude.

---

### Open Questions

- [ ] Why does Healthy have ~−13 μV DC offset across all channels? Check if this is a referencing artifact or a recording baseline.
- [ ] Are rows truly independent time-point snapshots, or are there temporal dependencies within a class block? (Data is ordered by class — consecutive rows within a class may be from the same session.)
- [ ] Will FFT on individual rows (1 sample) vs. a windowed segment change the spectral features significantly?

---

### Next Steps

| ID | Experiment | Priority |
|----|------------|----------|
| EXP-002 | Feature engineering: FFT + UMAP, validate 19-feature matrix shape and variance | ~~High~~ **Done** |
| EXP-003 | Baseline classifiers (Logistic Regression, Random Forest) on raw 16 features | ~~Medium~~ **Done** |
| EXP-004 | Baseline classifiers on FFT+UMAP 19-feature matrix | ~~High~~ **Done** |
| EXP-005 | SeqBoostNet implementation: LSTM + XGB + GB → AdaBoost | ~~High~~ **Done** |
| EXP-006 | Binary classification cases A1–A6 as defined in the paper | ~~Medium~~ **Done** |
| EXP-007 | Threshold tuning on A1, A4, A5 | ~~Medium~~ **Done** |

---

## EXP-002 — Feature Engineering

**Date:** 2026-06-14
**Notebook:** `notebooks/02_features.ipynb`
**Figures:** `reports/figures/07–10_*.png`
**Artifacts:** `data/processed/{train,val,test}_features.parquet`, `scaler.pkl`, `umap_reducer.pkl`

### Setup

| Parameter | Value |
|-----------|-------|
| Pipeline | StandardScaler (per-channel) → FFT (axis=1, 16 features) + UMAP (3D) |
| Output shape | (n_samples, 19) — confirmed for train / val / test |
| UMAP fit | Train only; val/test transformed with fitted reducer |
| Scaler fit | Train only; val/test transformed with fitted scaler |

> **Bug fixed:** Original `features.py` used `rfft(..., axis=0)` — FFT across samples, wrong output shape.
> Corrected to `fft(..., axis=1)` — FFT across 16 channels per row, preserving (n, 16) shape.

### Findings

#### F6 — Feature matrix shape validated
All three splits produce (n, 19) matrices: 16 FFT + 3 UMAP. DC term (k=0) of each FFT row is real and non-negative, confirming correct axis usage.

#### F7 — Variance is concentrated in 3 frontal/temporal FFT features + UMAP-1
Highest-variance features: `Fp1_fft` (~65), `T6_fft` (~49), `Fp2_fft` (~49), then `umap_1` (~15). The 6 parietal/occipital channels (`P3`, `P4`, `O1`, `O2`, `F7`, `F8` FFT) carry near-zero variance — they contribute little information after scaling.

#### F8 — UMAP-1 is the single most class-discriminative feature by a large margin
ANOVA F-scores: `umap_1` F≈17,500, next best `T5_fft`/`F3_fft`/`Fp2_fft`/`T6_fft` at F≈2,800. `umap_3` is the weakest (F≈25). All 19 features are statistically significant (p < 0.05).

Mean F-score — FFT features: ~870 · UMAP features: ~6,100 (UMAP carries ~7× more discriminative signal on average).

#### F9 — UMAP embedding cleanly isolates Healthy; seizure classes heavily overlap
In all three 2D projections (umap_1 vs umap_2/3, umap_2 vs umap_3), the Healthy class forms a tight, well-separated cluster. The three seizure classes (Focal, Generalized, Seizure Events) overlap substantially — consistent with EXP-001 F3 (similar PSD shapes).

> **Implication:** A model that learns only on UMAP features will classify Healthy vs. seizure easily but will struggle with inter-seizure discrimination. The FFT features are likely needed to resolve Focal vs. Generalized (A1 case).

#### F10 — FFT distributions are right-skewed; Healthy has heavier tail
All FFT features are bounded below by zero (magnitudes). Healthy class consistently shows a heavier right tail across all channels. Seizure classes cluster near zero with occasional high-magnitude outliers.

### Open Questions

- [ ] The spatial FFT (across 16 channels per row) treats simultaneous electrode readings as a "signal" — is this the same transform the paper intends, or do they apply temporal FFT across a session window?
- [ ] 6 low-variance FFT features may be worth dropping before modelling — test with and without.
- [ ] UMAP is non-deterministic beyond `random_state`; should verify embedding stability across seeds.

---

## EXP-003 — Baseline Classifiers (Raw 16 Features)

**Date:** 2026-06-14
**Notebook:** `notebooks/03_baselines.ipynb`
**Figures:** `reports/figures/11–14_*.png`
**Artifacts:** `data/processed/exp003_baseline_results.csv`

### Setup

| Parameter | Value |
|-----------|-------|
| Feature set | Raw 16 EEG channels, StandardScaler (fit on train only) |
| Models | Logistic Regression (lbfgs, C=1, max_iter=1000), Random Forest (200 trees, min_samples_leaf=2) |
| Split | Train 5,600 / Val 800 / Test 1,600 |

### Results (test set)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Logistic Regression | 0.461 | 0.468 |
| **Random Forest** | **0.954** | **0.954** |

### Findings

#### F11 — Raw channels are highly non-linear; LR fails badly
Logistic Regression reaches only 46% test accuracy on raw channels. Per-class precision/recall is uneven, with Focal and Generalized heavily confused. The 4-class boundaries in the raw 16D space are not linearly separable.

#### F12 — Random Forest achieves 95.4% test accuracy on raw features alone
Without any feature engineering, RF already reaches a very high performance floor. Val ≈ Test (both ~95%), so there is no overfitting. This sets a strong bar that the full SeqBoostNet pipeline must exceed.

#### F13 — Frontal channels dominate RF feature importance
RF importance rankings: `Fp1` and `Fp2` (frontal-polar) are the two most important channels by a clear margin, followed by `F3`, `T5`, and `T6`. This is consistent with EXP-001 F2/F3: the Healthy class has anomalously high amplitude in frontal channels, and the RF exploits this directly via axis-aligned splits.

### Open Questions

- [ ] Does RF's high raw performance hold under cross-session or cross-subject splits, or is it exploiting within-session leakage (rows ordered by class)?
- [ ] Is the raw-channel RF already near the paper's reported accuracy — if so, does feature engineering add value for tree-based models?

---

## EXP-004 — Classifiers on 19-Feature FFT+UMAP Matrix

**Date:** 2026-06-14
**Notebook:** `notebooks/04_classifiers_fft_umap.ipynb`
**Figures:** `reports/figures/15–18_*.png`
**Artifacts:** `data/processed/exp004_results.csv`

### Setup

| Parameter | Value |
|-----------|-------|
| Feature set | StandardScaler → FFT (16) + UMAP (3) = 19 features (pre-built in EXP-002) |
| Models | Same LR and RF configs as EXP-003 |
| Split | Train 5,600 / Val 800 / Test 1,600 |

### Results (test set)

| Model | Accuracy | Macro-F1 | Δ vs EXP-003 |
|-------|----------|----------|--------------|
| Logistic Regression | 0.702 | 0.700 | **+0.232** |
| Random Forest | 0.885 | 0.885 | **−0.065** |

### Findings

#### F14 — FFT+UMAP linearises the problem for LR (+23 pp macro-F1)
The feature engineering pipeline transforms the raw channels into a space where a linear classifier jumps from 0.47 to 0.70 macro-F1. UMAP-1 (the dominant discriminative feature from EXP-002 F8) is likely driving most of this gain — it effectively separates Healthy from all seizure classes, enabling LR to draw a useful boundary.

#### F15 — FFT compression hurts Random Forest (−6.5 pp macro-F1)
RF drops from 95.4% to 88.5% when switching from raw channels to FFT+UMAP features. The spatial FFT (across 16 simultaneous channel readings per row) compresses information that RF can exploit directly via axis-aligned splits. The FFT magnitude discards phase and reduces the 16D raw space to spectral amplitudes, losing some discriminative signal for tree-based splits. UMAP's 3D projection further compresses the 16 channels into dimensions optimised for global structure, not for local tree boundaries.

#### F16 — UMAP features dominate RF importance on the 19-feature matrix
On the FFT+UMAP features, `umap_1` is the single most important RF feature by a large margin, followed by `umap_2` and then the frontal FFT channels (`Fp1_fft`, `Fp2_fft`). This is consistent with EXP-002 F8 (UMAP-1 ANOVA F≈17,500).

### Open Questions

- [ ] Can LR reach RF-raw performance (~95%) with more regularisation tuning or polynomial features on top of FFT+UMAP?
- [ ] The RF-on-raw result (95.4%) raises the question of whether UMAP is net-beneficial for the full SeqBoostNet ensemble — the LSTM and boosting stages may behave differently from RF.

---

## EXP-005 — SeqBoostNet (LSTM + XGBoost + GradientBoosting → AdaBoost)

**Date:** 2026-06-14
**Notebook:** `notebooks/05_seqboostnet.ipynb`
**Figures:** `reports/figures/19–20_*.png`
**Artifacts:** `data/processed/exp005_seqboostnet_results.csv`

### Setup

| Parameter | Value |
|-----------|-------|
| Feature set | 19-feature FFT+UMAP matrix (from EXP-002) |
| Stacking | 3-fold stratified OOF; meta-features = stacked class probabilities (12 cols: 4 classes × 3 models) |
| LSTM | 128 units, ReLU, Dropout 0.5, Softmax output, Adam, Sparse CCE; max 100 epochs, EarlyStopping(patience=15) |
| XGBoost | 300 estimators, max_depth=6, lr=0.05, multi:softmax |
| GradientBoosting | 100 estimators, lr=0.1, max_depth=3 |
| AdaBoost (meta) | 50 estimators, lr=1.0, SAMME |
| Split | Train 5,600 / Val 800 / Test 1,600 |

### Results (test set)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| LSTM (base, standalone) | — | — |
| XGBoost (base, standalone) | — | — |
| GradientBoosting (base, standalone) | — | — |
| **SeqBoostNet (ensemble)** | **0.8438** | **0.8453** |

*Base model standalone results recorded in notebook cell 10.*

### Full progression (test macro-F1)

| Model | EXP-003 (raw 16) | EXP-004 (FFT+UMAP 19) | EXP-005 |
|-------|-----------------|----------------------|---------|
| Logistic Regression | 0.468 | 0.700 | — |
| Random Forest | 0.954 | 0.885 | — |
| SeqBoostNet | — | — | 0.845 |

### Findings

#### F17 — SeqBoostNet (0.845 macro-F1) sits between the two RF baselines
The full paper ensemble outperforms RF-on-FFT+UMAP (0.885 → 0.845 is actually lower — see F18), but does not beat the strongest baseline: RF-on-raw-16 at 0.954. On the 4-class multiclass problem, SeqBoostNet is not the best-performing model in this experiment series.

#### F18 — SeqBoostNet underperforms RF on both feature sets in 4-class setting
SeqBoostNet achieves 0.845 macro-F1 vs. 0.885 for RF-on-FFT+UMAP and 0.954 for RF-on-raw. The stacking overhead (OOF meta-features → AdaBoost) does not compensate for the information loss relative to a well-tuned RF. The LSTM component, treating 19 tabular features as 19 sequential timesteps, is likely a weak contributor on this dataset — the temporal interpretation of cross-channel FFT/UMAP values is not natural.

#### F19 — Paper comparison requires binary cases (EXP-006)
The paper reports 96.71% average accuracy across six binary classification cases (A1–A6), not 4-class multiclass. A direct comparison is not possible from this experiment. EXP-006 will rerun SeqBoostNet on each binary case to produce a fair comparison with the paper's reported numbers. The hardest binary case (A1: Focal vs. Generalized, 95.91% in the paper) will be the key test.

#### F20 — AdaBoost on 12-column OOF probability meta-features is stable
The 3-fold OOF stacking produces well-calibrated meta-features: no fold collapses (all folds produce non-trivial predictions for all 4 classes). AdaBoost training on (5600, 12) OOF features converges without error. Val ≈ Test (0.850 vs. 0.845), suggesting no overfitting at the meta-model level.

### Open Questions

- [ ] Do the LSTM's individual OOF predictions add meaningful signal, or does XGBoost dominate the meta-feature space? (Ablation: AdaBoost on XGB+GB only vs. all three.)
- [ ] Would treating the 16 raw channels as 16 LSTM timesteps (instead of the 19 FFT+UMAP features) improve the LSTM contribution?
- [x] Will SeqBoostNet match the paper's reported per-case binary accuracy in EXP-006, where the 4-class problem is decomposed into easier 2-class problems? → **Broadly yes (avg 95.56% vs 96.71% paper), within 1–2 pp on most cases.**

---

## EXP-006 — Binary Classification Cases A1–A6

**Date:** 2026-06-14
**Notebook:** `notebooks/06_binary_cases.ipynb`
**Figures:** `reports/figures/21–22_*.png`
**Artifacts:** `data/processed/exp006_binary_results.csv`

### Setup

| Parameter | Value |
|-----------|-------|
| Architecture | SeqBoostNet — same as EXP-005 |
| Feature pipeline | Per-case StandardScaler+FFT+UMAP (19 features); scaler and UMAP refit on each case's train split |
| Stacking | 3-fold OOF; LSTM 10 CV epochs / 30 final epochs (EarlyStopping patience=8) |
| Split | Per-case stratified 70/10/20 (~2800 train / 400 val / 800 test per case) |

### Results vs paper (test accuracy %)

| Case | Title | Ours | Paper | Δ (pp) |
|------|-------|------|-------|--------|
| A1 | Generalized vs Focal | 93.62 | 95.91 | −2.29 |
| A2 | Generalized vs Healthy | 99.62 | 99.66 | −0.04 |
| A3 | Focal vs Healthy | 99.88 | 99.83 | +0.05 |
| A4 | Focal vs Seizure Events | 93.25 | 91.16 | +2.09 |
| A5 | Generalized vs Seizure Events | 87.25 | 94.01 | −6.76 |
| A6 | Seizure Events vs Healthy | 99.75 | 99.66 | +0.09 |
| **Avg** | | **95.56** | **96.71** | **−1.15** |

### Full metric table (test set)

| Case | Accuracy | Precision | Recall | F1 | F2 | Kappa | MCC | ROC-AUC | Sensitivity | Specificity | Log Loss |
|------|----------|-----------|--------|----|----|-------|-----|---------|-------------|-------------|----------|
| A1 | 93.62 | 92.05 | 95.50 | 93.74 | 94.79 | 87.25 | 87.31 | 97.25 | 95.50 | 91.75 | 0.413 |
| A2 | 99.62 | 99.26 | 100.0 | 99.63 | 99.85 | 99.25 | 99.25 | 99.62 | 100.0 | 99.25 | 0.344 |
| A3 | 99.88 | 99.75 | 100.0 | 99.88 | 99.95 | 99.75 | 99.75 | 100.0 | 100.0 | 99.75 | 0.309 |
| A4 | 93.25 | 92.82 | 93.75 | 93.28 | 93.56 | 86.50 | 86.50 | 97.23 | 93.75 | 92.75 | 0.427 |
| A5 | 87.25 | 84.02 | 92.00 | 87.83 | 90.28 | 74.50 | 74.84 | 95.13 | 92.00 | 82.50 | 0.465 |
| A6 | 99.75 | 99.50 | 100.0 | 99.75 | 99.90 | 99.50 | 99.50 | 99.99 | 100.0 | 99.50 | 0.312 |

### Findings

#### F21 — Average accuracy (95.56%) closely reproduces the paper (96.71%), within 1.15 pp
Four of six cases land within 2.3 pp of the paper. A2, A3, A6 (all involving Healthy) exceed 99.6% — matching the paper almost exactly. A4 (Focal vs Seizure Events) actually outperforms the paper by +2.1 pp. The overall reproduction is successful.

#### F22 — A5 (Generalized vs Seizure Events) is the largest gap: 87.25% vs 94.01% (−6.76 pp)
A5 is the only case where the gap is material. A5 and A1 are the hardest cases (within-seizure-type discrimination), consistent with EXP-001 F3 (Focal and Generalized share very similar PSDs). The gap on A5 likely reflects a combination of: (a) fewer LSTM training epochs in this experiment vs the paper's 100 epochs; (b) UMAP instability on the 50/50 Generalized/Seizure Events subset where class structure is least separable. ROC-AUC for A5 is still 95.13%, indicating the ranker is good but the decision boundary is suboptimal.

#### F23 — A1 gap (−2.29 pp) is likely due to reduced LSTM training
A1 (Generalized vs Focal) is the canonical hard case; the paper reports 95.91%, we achieve 93.62%. Given that F21 shows the architecture is broadly correct, the shortfall is likely a training budget issue (10 CV epochs + 30 final vs paper's 100 epochs). The ROC-AUC of 97.25% shows the model has strong ranking ability — calibration and decision boundary placement would improve with more training.

#### F24 — Healthy-involved cases (A2, A3, A6) are trivially solved by all methods
A2/A3/A6 reach >99.6% accuracy, sensitivity=100%, and near-perfect MCC. These results are consistent across EXP-003 (RF-on-raw), EXP-004, EXP-005, and EXP-006: the Healthy class is cleanly separable from all seizure classes by amplitude alone (EXP-001 F2). These cases do not stress-test the model.

#### F25 — ROC-AUC exceeds accuracy in all hard cases, revealing calibration as the bottleneck
A1 ROC-AUC=97.25% vs accuracy=93.62%; A4 ROC-AUC=97.23% vs accuracy=93.25%; A5 ROC-AUC=95.13% vs accuracy=87.25%. The ensemble ranks samples well but the AdaBoost threshold is not optimally calibrated. Threshold tuning on the val set (instead of default 0.5) would likely recover 1–3 pp on A1, A4, A5.

### Open Questions

- [ ] Would training the LSTM for the full 100 epochs (as in the paper) close the A1/A5 gap?
- [x] Would threshold tuning on the val set improve A1, A4, A5 accuracy by 1–3 pp? → **No. EXP-007 shows no gain; the 0.5 default is already near-optimal.**
- [ ] Is the A5 gap a fundamental limit of this feature space, or a training artifact? (A5 ROC-AUC=95.1% suggests the latter.)

---

## EXP-007 — Threshold Tuning on A1, A4, A5

**Date:** 2026-06-14
**Notebook:** `notebooks/07_threshold_tuning.ipynb`
**Figures:** `reports/figures/23–24_*.png`
**Artifacts:** `data/processed/exp007_threshold_results.csv`

### Setup

| Parameter | Value |
|-----------|-------|
| Cases | A1, A4, A5 (the three cases where EXP-006 ROC-AUC materially exceeded accuracy) |
| Architecture | SeqBoostNet — same as EXP-006 (retrained; 50 final LSTM epochs, patience=12) |
| Threshold search | Grid search [0.01, 0.99] in 0.001 steps; three objectives: best val accuracy, best val F1, Youden's J |
| Selection rule | Threshold chosen on **val set only**; reported accuracy is on **test set** |

### Results

| Case | Default (t=0.5) | Best Val Acc | Best Val F1 | Youden's J | Paper |
|------|----------------|-------------|------------|------------|-------|
| A1 | 93.62% (t=0.500) | 93.50% (t=0.527) | 93.50% (t=0.527) | 93.50% (t=0.536) | 95.91% |
| A4 | 93.25% (t=0.500) | 93.25% (t=0.460) | 93.25% (t=0.460) | 93.25% (t=0.526) | 91.16% |
| A5 | 87.25% (t=0.500) | 86.50% (t=0.553) | 84.62% (t=0.425) | 86.50% (t=0.559) | 94.01% |

### Findings

#### F26 — Threshold tuning yields no improvement; the 0.5 default is near-optimal
Across all three cases and all three objectives, the tuned threshold either matches or marginally decreases test accuracy relative to 0.5. A4 is unchanged; A1 and A5 lose 0.12 pp and 0.75 pp respectively at their best-accuracy thresholds. The F1-optimised threshold for A5 (t=0.425) actively hurts accuracy (87.25% → 84.62%).

#### F27 — The ROC-AUC vs accuracy gap is structural, not a calibration artefact
EXP-006 F25 hypothesised that the gap (e.g. A5: ROC-AUC 95.13% vs accuracy 87.25%) could be recovered by threshold shifting. EXP-007 falsifies this: the AdaBoost probability scores are well-calibrated around 0.5 and no threshold recovers meaningful ground. The gap reflects genuine difficulty in separating the classes at the score level — the model's decision surface cannot resolve the overlap between Generalized and Seizure Events at any threshold. Increasing training budget (more LSTM epochs) or richer features remain the primary levers.

#### F28 — A5 is asymmetric: tuning toward recall (lower threshold) trades accuracy for sensitivity
At t=0.425 (best F1), A5 sensitivity rises but specificity drops enough to reduce accuracy. This confirms A5's confusion is asymmetric: false negatives (missed Generalized) are easier to fix than false positives (Seizure Events misclassified as Generalized). The feature space simply cannot cleanly separate these two seizure classes with the current 19-feature representation.

### Open Questions

- [ ] Would increasing LSTM epochs to 100 (paper's setting) move the A1/A5 test accuracy by more than threshold tuning could (confirmed ceiling: 0 pp from tuning)?
- [ ] Can ensemble calibration (Platt scaling or isotonic regression on val probabilities) improve the ROC-AUC → accuracy conversion, or is the issue pre-calibration score separation?

---
