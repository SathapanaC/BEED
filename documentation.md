# BEED — EEG Epilepsy Classification

Reproduction and extension of the **SeqBoostNet** paper (Najmusseher & Nizar Banu P K, IJCDS 2025)
on the **Bangalore EEG Epilepsy Dataset**. The goal is to classify 16-channel EEG recordings into
four clinical categories — Healthy, Focal seizure, Generalized seizure, and Seizure Events — using
a pipeline of FFT + UMAP features fed into an LSTM + XGBoost + Gradient Boosting → AdaBoost
stacking ensemble.

**Reference paper:** *Feature Engineering for Epileptic Seizure Classification Using SeqBoostNet*,
International Journal of Computing and Digital Systems, Vol. 17, No. 1, 2025.
DOI: [10.12785/ijcds/1571020131](http://dx.doi.org/10.12785/ijcds/1571020131)

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Feature Pipeline](#2-feature-pipeline)
3. [SeqBoostNet Architecture](#3-seqboostnet-architecture)
4. [Experiment Cases](#4-experiment-cases)
5. [Results](#5-results)
6. [Codebase Layout](#6-codebase-layout)
7. [Development Setup](#7-development-setup)
8. [Running Notebooks](#8-running-notebooks)
9. [Testing](#9-testing)
10. [Dependencies](#10-dependencies)

---

## 1. Dataset

### Overview

The **Bangalore EEG Epilepsy Dataset (BEED)** contains raw waveform EEG recordings from 16 channels
collected at a clinic in Bangalore, India, under the 10-20 International electrode placement system.

```text
Dataset Snapshot
─────────────────────────────────────────────
  Rows (samples)    :  4,000
  EEG channels      :     16  (X1–X16 in CSV)
  Label column      :      1  (y)
  Total columns     :     17
  Sampling rate     :    256 Hz
  Recording length  :     20 seconds/session
  Electrode system  :  10-20 International
  Amplitude range   :  ≈ −200 μV to +300 μV
─────────────────────────────────────────────
```

> [!IMPORTANT]
> The raw CSV is ordered by class — rows are **not shuffled**. Always use stratified splitting
> before modelling. `beed.data.split()` handles this automatically.

### Class Labels

| Value | Class | Clinical Description |
|-------|-------|----------------------|
| `0` | Healthy | EEG from seizure-free participants — baseline/control |
| `1` | Focal | Seizure onset localised to a specific brain area (one hemisphere) |
| `2` | Generalized | Seizure onset propagating simultaneously across both hemispheres |
| `3` | Seizure Events | Seizure activity recorded during physical movement |

### Electrode Mapping

The 16 CSV columns (`X1`–`X16`) correspond to the following 10-20 system electrodes:

| Column | Electrode | Brain Region |
|--------|-----------|--------------|
| `X1` | Fp1 | Left prefrontal |
| `X2` | Fp2 | Right prefrontal |
| `X3` | F3 | Left frontal |
| `X4` | F4 | Right frontal |
| `X5` | C3 | Left central |
| `X6` | C4 | Right central |
| `X7` | P3 | Left parietal |
| `X8` | P4 | Right parietal |
| `X9` | O1 | Left occipital |
| `X10` | O2 | Right occipital |
| `X11` | F7 | Left anterior temporal |
| `X12` | F8 | Right anterior temporal |
| `X13` | T3 | Left mid-temporal |
| `X14` | T4 | Right mid-temporal |
| `X15` | T5 | Left posterior temporal |
| `X16` | T6 | Right posterior temporal |

The 16 electrodes span all four major lobes bilaterally, enabling detection of both focal
(lateralised) and generalized (bilateral) seizure activity.

```text
Electrode Layout — 10-20 System (16-channel)
──────────────────────────────────────────────────────
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
──────────────────────────────────────────────────────
  Left hemisphere : X1, X3, X5, X7, X9, X11, X13, X15
  Right hemisphere: X2, X4, X6, X8, X10, X12, X14, X16
```

### EEG Frequency Bands

FFT decomposes raw EEG amplitudes into five canonical brainwave bands. Seizure signals typically
exhibit high-amplitude repetitive activity that becomes clearly separable in the frequency domain.

| Band | Frequency Range | Associated State |
|------|----------------|-----------------|
| Delta (δ) | 0.5 – 4 Hz | Deep sleep, severe pathology |
| Theta (θ) | 4 – 8 Hz | Drowsiness, early seizure activity |
| Alpha (α) | 8 – 13 Hz | Relaxed wakefulness |
| Beta (β) | 13 – 30 Hz | Active cognition, alert state |
| Gamma (γ) | 30 – 100 Hz | High cognitive processing, ictal activity |

---

## 2. Feature Pipeline

Raw 16-channel signals are transformed through two parallel paths — spectral (FFT) and temporal
(UMAP) — then concatenated into a 19-feature matrix fed to SeqBoostNet.

```text
Raw CSV  (4,000 × 16)   ←── X1…X16, one row per time sample
         │
         ▼
  StandardScaler (per channel)
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

| Stage | Input Shape | Output Shape | Method |
|-------|-------------|--------------|--------|
| Raw data | 4,000 × 16 | — | As acquired |
| StandardScaler | 4,000 × 16 | 4,000 × 16 | Per-channel z-score |
| Spectral features | 4,000 × 16 | 4,000 × 16 | FFT magnitude |
| Temporal features | 4,000 × 16 | 4,000 × 3 | UMAP (3D embedding) |
| **Combined** | — | **4,000 × 19** | Concatenation |

### 2.1 Spectral Features (FFT)

FFT transforms EEG from the time domain to the frequency domain using:

$$X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-\frac{2\pi i n k}{N}}$$

Where `Xₖ` is the frequency-domain output, `N` is the total sample count, and `k` indexes
frequency bins 0 to N−1. One spectral amplitude feature is extracted per channel, yielding
16 spectral features total.

### 2.2 Temporal Features (UMAP)

UMAP (Uniform Manifold Approximation and Projection) compresses the 16-channel EEG into 3
dimensions while preserving the non-linear manifold structure of inter-channel relationships.
UMAP is **fit on train only** and `.transform()` applied to val/test to prevent data leakage.

```text
UMAP Pipeline
│
├── Step 1 — Build local neighborhood graph
│
├── Step 2 — Compute fuzzy simplicial set
│            φ(dᵢⱼ, σᵢ) = exp(−dᵢⱼ² / 2σᵢ²)
│
├── Step 3 — Gradient descent to minimise
│            cross-entropy between high-D and low-D similarities
│
└── Step 4 — Output 3D embedding
```

> [!WARNING]
> Always fit the UMAP reducer on training data only. Fitting on the full dataset leaks
> validation/test distribution information into the embedding.

---

## 3. SeqBoostNet Architecture

SeqBoostNet is a two-level stacking ensemble that combines deep learning with gradient
boosting for robust seizure classification.

```text
SeqBoostNet Architecture
─────────────────────────────────────────────────────────

Input (4,000 × 19 feature matrix)
         │
         ├───────────────┬───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐   ┌──────────┐
    │  LSTM   │    │ XGBoost  │   │ Gradient │
    │ 128 cells│   │ 300 trees│   │ Boosting │
    │ ReLU    │    │ depth 6  │   │ 100 trees│
    │ dropout │    │ lr 0.05  │   │ depth 3  │
    │  0.5    │    │          │   │ lr 0.1   │
    └────┬────┘    └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
               LEVEL 0 predictions
               (stacked as new features)
                        │
                        ▼
               ┌────────────────┐
               │    AdaBoost    │
               │  50 estimators │
               │   lr  1.0      │
               └───────┬────────┘
                        │
                        ▼
                 Final Prediction
```

### Base Models (Level 0)

| Model | Role | Key Hyperparameters |
|-------|------|---------------------|
| LSTM | Captures sequential/temporal EEG patterns across channels | 128 cells, ReLU, dropout 0.5, Adam, 100 epochs, batch 32 |
| XGBoost | Regularised tree boosting with high accuracy | 300 estimators, max depth 6, lr 0.05, Multi-Softmax |
| Gradient Boosting | Sequential residual correction, robust to outliers | 100 estimators, max depth 3, lr 0.1 |

### Meta-Model (Level 1)

**AdaBoost** combines the three base model predictions by iteratively reweighting misclassified
samples. Final prediction: `H(x) = sign(Σ αₜ hₜ(x))` where `αₜ = ½ log((1−errₜ)/errₜ)`.

### Time Complexity

| Component | Complexity |
|-----------|-----------|
| FFT | O(N log N) |
| UMAP | O(N × D) |
| LSTM | O(N) |
| XGBoost / GB / AdaBoost | O(M × T) |
| **SeqBoostNet total** | **O(N log N + N×D + M×T)** |

Where N = samples, D = dimensions, M = features, T = boosting iterations.

---

## 4. Experiment Cases

Six binary classification cases are derived from the four BEED classes, testing different
clinically meaningful discrimination tasks.

```text
Binary Classification Cases
│
├── A1 — Generalized  vs. Focal          (within-seizure discrimination)
├── A2 — Generalized  vs. Healthy        (seizure vs. normal)
├── A3 — Focal        vs. Healthy        (seizure vs. normal)
├── A4 — Focal        vs. Seizure Events (within-seizure discrimination)
├── A5 — Generalized  vs. Seizure Events (within-seizure discrimination)
└── A6 — Seizure Events vs. Healthy      (seizure vs. normal)
```

Cases A2, A3, and A6 (seizure vs. healthy) are the easiest discrimination tasks.
Cases A1, A4, and A5 (within-seizure-type) are the hardest.

---

## 5. Results

### BEED Performance

| Case | Accuracy | Precision | Recall | F1 | Kappa | MCC | ROC-AUC | Time (s) |
|------|----------|-----------|--------|----|-------|-----|---------|---------|
| A1 — Gen vs. Focal | 95.91 | 96.01 | 95.91 | 95.91 | 91.83 | 91.91 | 95.98 | 28 |
| A2 — Gen vs. Healthy | 99.66 | 99.66 | 99.66 | 99.66 | 99.33 | 99.33 | 99.65 | 28 |
| A3 — Focal vs. Healthy | 99.83 | 99.83 | 99.83 | 99.83 | 99.66 | 99.66 | 99.82 | 28 |
| A4 — Focal vs. Seizure | 91.16 | 91.25 | 91.16 | 91.15 | 82.27 | 82.38 | 91.06 | 30 |
| A5 — Gen vs. Seizure | 94.01 | 94.01 | 94.01 | 94.01 | 87.98 | 87.99 | 94.01 | 31 |
| A6 — Seizure vs. Healthy | 99.66 | 99.66 | 99.66 | 99.66 | 99.33 | 99.33 | 99.65 | 48 |
| **Average** | **96.71** | | | | | | | |

### Comparison vs. Other Feature Extraction Methods (BEED, Accuracy %)

| Case | WT | STATS | STFT | PCA | ICA | HHT | EMD | **Proposed** |
|------|----|-------|------|-----|-----|-----|-----|----|
| A1 | 89.91 | 83.83 | 81.42 | 85.83 | 85.67 | 78.75 | 83.58 | **93.33** |
| A2 | 89.83 | 92.19 | 91.99 | 90.83 | 90.99 | 94.75 | 93.58 | **99.58** |
| A3 | 92.30 | 91.42 | 92.51 | 93.67 | 92.75 | 94.42 | 94.33 | **99.83** |
| A4 | 85.08 | 68.75 | 67.83 | 83.58 | 82.33 | 59.17 | 70.92 | **89.91** |
| A5 | 89.91 | 76.42 | 75.42 | 90.33 | 90.92 | 66.25 | 78.08 | **92.25** |
| A6 | 91.83 | 93.33 | 93.75 | 94.67 | 94.75 | 92.08 | 93.99 | **99.66** |
| **Avg** | 89.81 | 84.32 | 83.82 | 89.82 | 89.57 | 80.90 | 85.75 | **95.76** |

SeqBoostNet outperforms all seven competing feature extraction methods on every binary case,
with the largest gains on the hardest cases (A4, A5).

### Comparison vs. Other Stacking Classifiers (BEED, Accuracy %)

| Case | Stack 1 | Stack 2 | **SeqBoostNet** |
|------|---------|---------|---------|
| A1 | 83.00 | 86.58 | **95.91** |
| A2 | 91.75 | 90.92 | **99.66** |
| A3 | 90.67 | 89.91 | **99.83** |
| A4 | 71.67 | 75.58 | **91.16** |
| A5 | 78.42 | 82.67 | **94.01** |
| A6 | 99.58 | 99.75 | **99.66** |
| **Avg** | 85.85 | 87.57 | **96.71** |

*Stack 1: XGBoost + LightGBM → Bagging. Stack 2: RF + LightGBM + GB → XGBoost.*

---

## 6. Codebase Layout

```text
BEED/
│
├── src/beed/
│   ├── config.py        — paths, constants, channel names, class labels
│   └── data.py          — load_raw(), split() (stratified train/val/test)
│
├── notebooks/
│   ├── 01_eda.ipynb               — exploratory data analysis
│   ├── 02_features.ipynb          — FFT & UMAP feature extraction
│   ├── 03_baselines.ipynb         — baseline model comparisons
│   ├── 04_classifiers_fft_umap.ipynb — full classifier sweep on combined features
│   ├── 05_seqboostnet.ipynb       — SeqBoostNet training and evaluation
│   ├── 06_binary_cases.ipynb      — all 6 BEED binary classification cases
│   └── 07_threshold_tuning.ipynb  — decision threshold tuning for A1, A4, A5
│
├── tests/
│   └── test_data.py     — smoke tests for loading and splitting
│
├── docs/
│   ├── data_description.md        — dataset details and feature pipeline
│   └── seqboostnet_article.md     — reference paper summary
│
├── data/
│   ├── raw/             — BEED_Data.csv (gitignored, keep locally)
│   └── processed/       — serialised feature matrices (gitignored)
│
├── reports/
│   └── figures/         — saved plots (gitignored)
│
├── pyproject.toml
└── CLAUDE.md
```

### Key Modules

#### `src/beed/config.py`

Central constants — import from here, never hard-code paths or numbers elsewhere.

```python
from beed.config import (
    RAW_FILE,          # Path → data/raw/BEED_Data.csv
    DATA_PROCESSED,    # Path → data/processed/
    FIGURES,           # Path → reports/figures/
    SAMPLING_RATE,     # 256  (Hz)
    N_CHANNELS,        # 16
    N_UMAP_COMPONENTS, # 3
    RANDOM_STATE,      # 42   — pass to every stochastic operation
    CHANNEL_NAMES,     # ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',
                       #  'P3',  'P4',  'O1', 'O2', 'F7', 'F8',
                       #  'T3',  'T4',  'T5', 'T6']
    CLASS_LABELS,      # {0: 'Healthy', 1: 'Focal',
                       #  2: 'Generalized', 3: 'Seizure Events'}
)
```

All paths are resolved relative to the repository root via `Path(__file__).resolve().parents[2]`,
so the package works regardless of where it is imported from.

---

#### `src/beed/data.py`

Two functions that cover the full data ingestion and splitting workflow.

##### `load_raw() → pd.DataFrame`

Loads `data/raw/BEED_Data.csv` and renames the generic `X1`–`X16` column names to the
corresponding 10-20 electrode labels (`Fp1`, `Fp2`, …, `T6`). Returns a 4,000 × 17 DataFrame
with columns `[Fp1, Fp2, ..., T6, y]`.

```python
from beed.data import load_raw

df = load_raw()
# df.columns → ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',
#                'P3',  'P4',  'O1', 'O2', 'F7', 'F8',
#                'T3',  'T4',  'T5', 'T6', 'y']
# df.shape   → (4000, 17)
```

##### `split(df, test_size=0.2, val_size=0.1) → tuple[DataFrame, DataFrame, DataFrame]`

Performs a **stratified two-stage split** into train / val / test. The default split is
70 / 10 / 20. Returns `(train, val, test)` — each DataFrame retains the `y` column.

```python
from beed.data import split

train, val, test = split(df, test_size=0.2, val_size=0.1)
# len(train) ≈ 2800  (70%)
# len(val)   ≈  400  (10%)
# len(test)  ≈  800  (20%)
```

`val_size` is expressed as a fraction of the **full dataset**, not the remaining training set.
The function converts it internally (`val_relative = val_size / (1 - test_size)`) so callers
always think in terms of the total dataset. Both splits use `RANDOM_STATE = 42`.

> [!IMPORTANT]
> The raw CSV is ordered by class. Skip `split()` and you will train on only class 0 (Healthy).
> Always use this function — never `iloc` a plain slice.

---

#### `src/beed/features.py`

The full feature engineering pipeline: per-channel scaling, FFT spectral features, UMAP
temporal embedding, and a convenience function that runs all three in order.

All functions follow the **fit-on-train, transform-on-val/test** pattern via a `fit: bool`
parameter. Fitted objects (`StandardScaler`, `UMAP`) are returned so they can be reused.

##### `scale(X, scaler=None, fit=True) → tuple[DataFrame, StandardScaler]`

Applies `StandardScaler` independently to each of the 16 EEG channels (zero mean, unit
variance). Returns the scaled DataFrame and the fitted scaler.

```python
from beed.features import scale

X_train_scaled, scaler = scale(X_train, fit=True)          # fits on train
X_val_scaled,   _      = scale(X_val,   scaler=scaler, fit=False)  # reuses
X_test_scaled,  _      = scale(X_test,  scaler=scaler, fit=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `pd.DataFrame` | — | Input with the 16 channel columns |
| `scaler` | `StandardScaler \| None` | `None` | Pre-fitted scaler; required when `fit=False` |
| `fit` | `bool` | `True` | Fit a new scaler (`True`) or apply an existing one (`False`) |

##### `fft_features(X_scaled) → pd.DataFrame`

Applies `numpy.fft.fft` along **axis=1** (across the 16 channels per row), then takes the
absolute value to get spectral amplitudes. This treats the 16 simultaneously recorded channel
readings as a spatial signal and extracts one amplitude feature per channel.

Returns a DataFrame of shape `(n_samples, 16)` with columns `Fp1_fft`, `Fp2_fft`, …, `T6_fft`.

```python
from beed.features import fft_features

fft = fft_features(X_train_scaled)
# fft.shape   → (2800, 16)
# fft.columns → ['Fp1_fft', 'Fp2_fft', ..., 'T6_fft']
```

> [!NOTE]
> FFT is applied row-wise across the 16 channel amplitudes at each time point — not along the
> time axis. This extracts **inter-channel spatial frequency** structure rather than temporal
> band power. The output shape is identical to the input.

##### `umap_features(X_scaled, reducer=None, fit=True) → tuple[DataFrame, UMAP]`

Reduces the 16 scaled channels jointly to `N_UMAP_COMPONENTS = 3` dimensions using UMAP,
capturing non-linear manifold structure across channels. Returns the embedding DataFrame and
the fitted `UMAP` reducer.

```python
from beed.features import umap_features

umap_train, reducer = umap_features(X_train_scaled, fit=True)
umap_val,   _       = umap_features(X_val_scaled, reducer=reducer, fit=False)
# umap_train.shape   → (2800, 3)
# umap_train.columns → ['umap_1', 'umap_2', 'umap_3']
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_scaled` | `pd.DataFrame` | — | Scaled 16-channel input |
| `reducer` | `UMAP \| None` | `None` | Pre-fitted reducer; required when `fit=False` |
| `fit` | `bool` | `True` | Fit a new UMAP (`True`) or transform with an existing one (`False`) |

##### `build_features(X, scaler=None, umap_reducer=None, fit=True) → tuple[DataFrame, StandardScaler, UMAP]`

Convenience function that chains `scale → fft_features + umap_features → concat` into one
call. Returns the 19-feature matrix, the fitted scaler, and the fitted UMAP reducer.

```python
from beed.features import build_features

# Training set — fit everything
X_train_feats, scaler, reducer = build_features(X_train, fit=True)

# Val / test — reuse fitted objects
X_val_feats,  _, _ = build_features(X_val,  scaler=scaler, umap_reducer=reducer, fit=False)
X_test_feats, _, _ = build_features(X_test, scaler=scaler, umap_reducer=reducer, fit=False)

# X_train_feats.shape   → (2800, 19)
# X_train_feats.columns → ['Fp1_fft', ..., 'T6_fft', 'umap_1', 'umap_2', 'umap_3']
```

The internal execution order is:

```text
X (raw channels)
  │
  ▼
scale()                → X_scaled  (n × 16)
  │
  ├──► fft_features()  → fft       (n × 16)   columns: *_fft
  │
  └──► umap_features() → umap      (n ×  3)   columns: umap_1…3
                │
                ▼
         pd.concat([fft, umap], axis=1)
                │
                ▼
         combined       (n × 19)
```

---

## 7. Development Setup

### Prerequisites

- Python 3.12 at `C:\Python312\python.exe`
- [`uv`](https://github.com/astral-sh/uv) (manages the virtual environment)

### Installation

```bash
# Install all dependencies including dev tools
uv sync --extra dev
```

### Running commands

All commands must run through `uv` so they use the managed virtual environment:

```bash
uv run python <script.py>
uv run pytest
uv run jupyter lab
uv run ruff check src/
```

> [!NOTE]
> Do not activate the `.venv` manually or run `python` directly — always prefix with `uv run`
> to ensure the correct interpreter and packages are used.

---

## 8. Running Notebooks

```bash
uv run jupyter lab
```

Open notebooks in order for a full end-to-end walkthrough:

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Explore raw signal distributions, class balance, channel correlations |
| `02_features.ipynb` | Build and visualise FFT + UMAP features |
| `03_baselines.ipynb` | Train simple classifiers as a benchmark |
| `04_classifiers_fft_umap.ipynb` | Evaluate the full feature set across classifier types |
| `05_seqboostnet.ipynb` | Full SeqBoostNet training, stacking, and evaluation |
| `06_binary_cases.ipynb` | Reproduce all six A1–A6 binary case results |
| `07_threshold_tuning.ipynb` | Tune decision thresholds for hard cases A1, A4, A5 |

---

## 9. Testing

```bash
uv run pytest
```

Tests run with coverage reporting (`--cov=beed --cov-report=term-missing`) and cover:

| Test | What it checks |
|------|---------------|
| `test_load_shape` | CSV loads as 4,000 × 17 with named columns |
| `test_no_nulls` | No missing values in raw data |
| `test_class_balance` | All four classes present (`y` ∈ {0, 1, 2, 3}) |
| `test_split_sizes` | Train + val + test = full dataset; test ≈ 20% |
| `test_split_stratification` | All four classes present in every split |

> [!NOTE]
> Tests require `data/raw/BEED_Data.csv` to be present locally. The file is gitignored —
> obtain it separately and place it at that path before running.

---

## 10. Dependencies

### Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.26 | Numerical arrays |
| `pandas` | ≥ 2.2 | DataFrame I/O and manipulation |
| `scikit-learn` | ≥ 1.5 | StandardScaler, train/test split, baselines |
| `scipy` | ≥ 1.13 | FFT (`scipy.fft`) |
| `umap-learn` | ≥ 0.5 | UMAP dimensionality reduction |
| `xgboost` | ≥ 2.0 | XGBoost base model |
| `tensorflow` | ≥ 2.16 | LSTM implementation |
| `matplotlib` | ≥ 3.9 | Plotting |
| `seaborn` | ≥ 0.13 | Statistical visualisations |
| `pyarrow` | ≥ 24.0 | Parquet I/O for processed features |

### Dev

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥ 8.2 | Test runner |
| `pytest-cov` | ≥ 5.0 | Coverage reporting |
| `ruff` | ≥ 0.4 | Linting and import sorting |
