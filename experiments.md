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
| EXP-005 | SeqBoostNet implementation: LSTM + XGB + GB → AdaBoost | High |
| EXP-006 | Binary classification cases A1–A6 as defined in the paper | Medium |

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
