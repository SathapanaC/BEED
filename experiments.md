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
| EXP-002 | Feature engineering: FFT + UMAP, validate 19-feature matrix shape and variance | High |
| EXP-003 | Baseline classifiers (Logistic Regression, Random Forest) on raw 16 features | Medium |
| EXP-004 | Baseline classifiers on FFT+UMAP 19-feature matrix | High |
| EXP-005 | SeqBoostNet implementation: LSTM + XGB + GB → AdaBoost | High |
| EXP-006 | Binary classification cases A1–A6 as defined in the paper | Medium |

---
