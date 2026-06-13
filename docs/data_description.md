# BEED Dataset Description

**Source:** Bangalore EEG Epilepsy Dataset — collected at an EEG clinic in Bangalore, India  
**Reference:** Najmusseher & Nizar Banu P K, *Feature Engineering for Epileptic Seizure Classification Using SeqBoostNet*, IJCDS 2025, Vol. 17, No. 1  
**Recording protocol:** 10-20 International System for electrode placement

---

## Overview

The BEED dataset contains raw EEG waveform recordings from four subject groups — healthy individuals, focal seizure patients, generalized seizure patients, and patients recorded during physical movement with seizure activity. All recordings were acquired under clinical conditions at 256 Hz sampling rate and last 20 seconds per session.

```text
Dataset Snapshot
─────────────────────────────────────────────
  Rows (samples)    :  4,000
  Columns (features):     16  (EEG channels)
  Label column      :      1  (y)
  Total columns     :     17
  Sampling rate     :    256 Hz
  Recording length  :     20 seconds/session
  Electrode system  :  10-20 International
─────────────────────────────────────────────
```

---

## File Structure

```text
BEED_Data.csv
│
├── X1  … X16   — Raw EEG channel amplitudes (16 columns)
└── y           — Class label (1 column)
```

Each **row** is one time-point sample: a simultaneous snapshot of the brain's electrical state measured across all 16 electrodes. Amplitude values are in **microvolts (μV)**, consistent with typical EEG ranges observed in the data (approximately −200 μV to +300 μV).

---

## Columns

### EEG Signal Channels — `X1` to `X16`

Each column corresponds to one electrode channel recorded under the 10-20 International System. The 10-20 system is a standardized method for electrode placement in which electrode positions are defined as proportional distances (10% and 20%) between skull landmarks, ensuring reproducibility across subjects and clinical sites.

The table below shows the probable channel-to-electrode mapping for a standard 16-channel 10-20 montage. The exact mapping should be confirmed against the original clinic documentation.

| Column | Electrode | Brain Region |
|--------|-----------|--------------|
| `X1`  | Fp1 | Left prefrontal |
| `X2`  | Fp2 | Right prefrontal |
| `X3`  | F3  | Left frontal |
| `X4`  | F4  | Right frontal |
| `X5`  | C3  | Left central |
| `X6`  | C4  | Right central |
| `X7`  | P3  | Left parietal |
| `X8`  | P4  | Right parietal |
| `X9`  | O1  | Left occipital |
| `X10` | O2  | Right occipital |
| `X11` | F7  | Left anterior temporal |
| `X12` | F8  | Right anterior temporal |
| `X13` | T3  | Left mid-temporal |
| `X14` | T4  | Right mid-temporal |
| `X15` | T5  | Left posterior temporal |
| `X16` | T6  | Right posterior temporal |

The 16 electrodes cover all four major lobes bilaterally, enabling detection of both **focal** seizure activity (localised to one hemisphere/region) and **generalized** seizure activity (simultaneous in both hemispheres).

```text
Electrode Coverage — 10-20 System (16-channel)
──────────────────────────────────────────────────────────────────
                          FRONTAL
              Fp1 (X1) ─────────── Fp2 (X2)
          F7 (X11) ─ F3 (X3) ─ F4 (X4) ─ F8 (X12)
                          │           │
                      CENTRAL
                   C3 (X5) ─────── C4 (X6)
                          │           │
                      PARIETAL
          T5 (X15) ─ P3 (X7) ─ P4 (X8) ─ T6 (X16)
              T3 (X13) ─────────── T4 (X14)
                          │           │
                      OCCIPITAL
                   O1 (X9) ──────── O2 (X10)
──────────────────────────────────────────────────────────────────
  Left hemisphere : X1, X3, X5, X7, X9, X11, X13, X15
  Right hemisphere: X2, X4, X6, X8, X10, X12, X14, X16
```

> [!NOTE]
> The column names `X1`–`X16` are generic labels in the CSV. The electrode-to-column correspondence above assumes a standard left-to-right, front-to-back ordering. Always verify against the original acquisition metadata before drawing region-specific conclusions.

---

### Target Label — `y`

The `y` column encodes the subject/recording class as an integer. Four classes are present, corresponding to the four clinical categories described in the paper (Table I):

| Value | Class | Clinical Description |
|-------|-------|----------------------|
| `0` | Healthy subject | EEG from seizure-free participants — baseline/control |
| `1` | Focal | Seizure onset localised to a specific brain area (one hemisphere) |
| `2` | Generalized | Seizure onset propagating simultaneously across both hemispheres |
| `3` | Seizure Events | Seizure activity recorded during physical movement |

> [!NOTE]
> The head of the CSV shows `y = 0` across all visible rows, suggesting the data is ordered by class. Rows are not shuffled in the raw file — stratified splitting is recommended before modelling.

The four classes support six distinct binary classification cases studied in the paper:

```text
Binary Classification Cases derived from y
│
├── A1 — y ∈ {2} vs. y ∈ {1}   →  Generalized  vs. Focal
├── A2 — y ∈ {2} vs. y ∈ {0}   →  Generalized  vs. Healthy
├── A3 — y ∈ {1} vs. y ∈ {0}   →  Focal        vs. Healthy
├── A4 — y ∈ {1} vs. y ∈ {3}   →  Focal        vs. Seizure Events
├── A5 — y ∈ {2} vs. y ∈ {3}   →  Generalized  vs. Seizure Events
└── A6 — y ∈ {3} vs. y ∈ {0}   →  Seizure Events vs. Healthy
```

---

## Feature Engineering Transformations

The raw 16-channel signals undergo two parallel transformations before classification, expanding the feature space:

```text
Raw CSV  (4,000 × 16)   ←── X1…X16, one row per time sample
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
  FFT (Fast Fourier Transform)            UMAP (dim. reduction)
  Frequency-domain decomposition          Manifold-preserving
  Reveals δ/θ/α/β/γ band power            compression to 3D
         │                                          │
         ▼                                          ▼
  Spectral features  (4,000 × 16)    Temporal features (4,000 × 3)
         │                                          │
         └────────────────────┬─────────────────────┘
                              ▼
                   Combined feature matrix
                       (4,000 × 19)
                              │
                              ▼
                        SeqBoostNet
                  (LSTM + XGB + GB → AdaBoost)
```

| Stage | Input shape | Output shape | Method |
|-------|-------------|--------------|--------|
| Raw data | 4,000 × 16 | — | As acquired |
| Spectral features | 4,000 × 16 | 4,000 × 16 | FFT |
| Temporal features | 4,000 × 16 | 4,000 × 3 | UMAP |
| **Combined** | — | **4,000 × 19** | Concatenation |

The combined 19-feature vector is the actual model input: 16 FFT-derived spectral features (one per channel, frequency-domain amplitude) plus 3 UMAP embedding dimensions (capturing global temporal structure across all channels jointly).

---

## EEG Signal Characteristics

EEG seizure signals are distinguished by their spectral composition. The five canonical brainwave bands captured by FFT are:

| Band | Frequency | Associated state |
|------|-----------|-----------------|
| Delta (δ) | 0.5 – 4 Hz | Deep sleep, severe pathology |
| Theta (θ) | 4 – 8 Hz | Drowsiness, early seizure activity |
| Alpha (α) | 8 – 13 Hz | Relaxed wakefulness |
| Beta (β) | 13 – 30 Hz | Active cognition, alert state |
| Gamma (γ) | 30 – 100 Hz | High cognitive processing, ictal activity |

Seizure signals typically exhibit **high-amplitude repetitive activity** combining slow waves and spike patterns — features that become clearly separable in the frequency domain after FFT transformation.

---

## Key Observations from Raw Data Head

Inspecting the first 19 data rows (all `y = 0`, Healthy class):

- Amplitude values range approximately from **−142 μV to +258 μV**, within the expected physiological EEG range
- Consecutive rows show **smooth temporal transitions** — successive time-point amplitudes change gradually rather than abruptly, consistent with continuous EEG recording
- Channels vary **independently** across the same row, reflecting spatially distributed brain activity recorded simultaneously at different scalp locations
- No missing values are present in the visible head

> [!TIP]
> Because amplitude scales may differ across electrode positions (due to proximity to signal sources, impedance variation, or referencing scheme), **standardisation per channel** is recommended during preprocessing — which the paper confirms is applied before feature extraction.
