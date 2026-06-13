# Feature Engineering for Epileptic Seizure Classification Using SeqBoostNet

**Authors:** Najmusseher and Nizar Banu P K  
**Affiliation:** Department of Computer Science, CHRIST (Deemed to be University) Central Campus, Bangalore-560029, India  
**Journal:** International Journal of Computing and Digital Systems, 2025, Vol. 17, No. 1, pp. 1–15  
**DOI:** [10.12785/ijcds/1571020131](http://dx.doi.org/10.12785/ijcds/1571020131)  
**Received:** 16 April 2024 · **Revised:** 15 October 2024 · **Accepted:** 25 October 2024

---

## Abstract

Epileptic seizure is a severe neurological condition that profoundly impacts patients' social lives, necessitating precise diagnosis for classification and prediction. This study addresses the need for reliable automated seizure detection in epilepsy by employing AI-driven analysis of EEG signals. Key innovations include:

- Combining **spectral and temporal features** using **UMAP** (Uniform Manifold Approximation and Projection) with **FFT** (Fast Fourier Transformation)
- Introduction of **SeqBoostNet** (Sequential Boosting Network), a robust stacking model integrating ML and deep learning for effective seizure classification

Validated on the **BONN** (UCI repository) and **BEED** (Bangalore EEG Epilepsy Dataset) benchmark datasets, achieving:

- **95.91%** accuracy distinguishing Focal vs. Generalized seizure onsets
- **96.71%** average accuracy on BEED
- **97.11%** average accuracy on BONN

**Keywords:** Epileptic Seizure · UMAP · Machine Learning · Deep Learning · FFT · LSTM · AdaBoost

---

## 1. Introduction

An epileptic seizure is a neurological condition caused by abnormality in the brain's electrical activity. Seizures are classified into **Focal**, **Generalized**, or **Unknown** types, affecting approximately 1% of the world population.

- **Focal seizures** begin from one area on one side of the brain
- **Generalized seizures** occur simultaneously in both hemispheres

EEG (Electroencephalography) is an integral tool for diagnosing brain seizure disorders, recording electrical signals as wavy lines via electrodes placed using the **10-20 International System**. Manual EEG analysis is laborious and time-consuming; automated detection methods are essential to assist neurologists.

This research aims to enhance seizure detection using spectral and temporal features combined with advanced classification via the novel **SeqBoostNet** stacking model, which supports both binary and multiclass classification.

**Key contributions:**

- Enhanced seizure detection reliability, improving patient quality of life
- Novel approach combining spectral and temporal domain features with a stacking meta-model
- New benchmark for seizure classification accuracy across multiple datasets
- Foundation for broader applications in computational neuroscience and brain-computer interfaces (BCI)

---

## 2. Related Work

Modern EEG-based seizure detection approaches and their limitations:

| Approach | Method | Limitation |
|---|---|---|
| CNN + FST / DCB | Feature extraction & classification | Dataset diversity constraints |
| CNN + traditional ML | Mutual information-based features | Dependency on image preprocessing |
| Three-step pipeline | Notch filtering + statistical + CNN features (CHB-MIT) | Scalability |
| DRSN + GRU | Patient-specific personalized models | Reduced generalizability |
| Multichannel EEG (5 features + MODWT) | Variance, Pearson, Hoeffding's D, Shannon entropy, IQR | Single clinical dataset |
| Savitzky–Golay + DWT + SVM | Preprocessing + feature extraction + SVM | Limited cross-population validation |
| Random Neural Network (RNN) | Seizure classification on BONN | Breadth of validation |
| Time-frequency + Relief selection | BONN dataset | Adaptability to varied sources |
| Stacking ensemble DNN | Ensemble learning | High computational demands |
| Wavelet + Fractal Dimension + SVD Entropy | BONN dataset | Complex transformations; real-time limits |
| Sliding window + DWT | Temporal element via sliding windows | Generalizability across datasets |
| DWT + Moth Flame Optimization ELM | BONN dataset | Limited adaptability to larger datasets |

The proposed model addresses these gaps through **robust feature engineering** (UMAP + FFT) and a **stacking approach** (SeqBoostNet), reducing preprocessing dependency while improving computational efficiency and seizure classification accuracy.

---

## 3. Methodology

The framework consists of four sequential stages:

```text
┌─────────────────┐
│  Data           │
│  Acquisition    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data           │
│  Preprocessing  │
│  (EDA +         │
│  Standardize)   │
└────────┬────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐   ┌─────────────────┐
│  UMAP           │   │  FFT            │
│  (Temporal      │   │  (Spectral      │
│   Features)     │   │   Features)     │
└────────┬────────┘   └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  Combined            │
         │  Spectral + Temporal │
         │  Feature Set         │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  Classification      │
         │  (SeqBoostNet)       │
         └──────────────────────┘
```

### 3.1 Data Acquisition

Two distinct datasets were employed:

#### BEED — Bangalore EEG Epilepsy Dataset

Raw waveform signals from **16 EEG channels** at **256 Hz** sampling rate, recorded with the 10–20 electrode placement system. Each recording lasts **20 seconds**.

```text
BEED Dataset Categories
│
├── Seizure Events   — Seizure recording during physical movement
├── Healthy Subject  — Recordings from seizure-free participants
├── Generalized      — Seizure recording in both brain hemispheres
└── Focal            — Seizure recording in a specific brain area
```

#### BONN — University of Bonn Dataset (UCI repository)

Five subsets of **100 individual channel recordings** from **500 subjects**, lasting **23.6 seconds** each, sampled at **173.61 Hz** (frequency range: 0.53–40 Hz). Dataset: 11,500 rows × 179 columns.

```text
BONN Class Labels
│
├── Class 1 — Seizure activity recordings
├── Class 2 — Tumor location recordings
├── Class 3 — Healthy brain recordings
├── Class 4 — Recordings with eyes closed
└── Class 5 — Recordings with eyes opened
```

### 3.2 Data Preprocessing

Initial preprocessing involves:

- **Exploratory Data Analysis (EDA):** Identify anomalies, understand EEG data attributes, and inform feature extraction decisions
- **Data Standardization:** Ensures consistent scales across EEG channels and subjects for clearer model interpretation

### 3.3 Temporal Features Using UMAP

UMAP (Uniform Manifold Approximation and Projection) reduces data dimensionality while preserving structural integrity, capturing intricate patterns through manifold learning and topological data analysis. It proceeds in four steps:

```text
UMAP Pipeline
│
├── Step 1 — Local neighborhood graph
│            Examine relationships between each data point
│            and its nearest neighbors
│
├── Step 2 — Fuzzy simplicial set
│            Compute pairwise fuzzy memberships (Eq. 1, 2)
│
├── Step 3 — Gradient descent
│            Optimize the UMAP objective function (Eq. 3)
│            to minimize cross-entropy between high-D
│            and low-D similarity distributions
│
└── Step 4 — Low-dimensional embedding
             Output the compressed representation
             for visualization and analysis
```

**Mathematical formulation:**

*Fuzzy Set Membership Function (Eq. 1):*

$$\phi(d_{ij}, \sigma_i) = \exp\!\left(-\frac{d_{ij}^2}{2\sigma_i^2}\right)$$

*Fuzzy Simplicial Set (Eq. 2):*

$$S_{ij} = \phi(d_{ij}, \sigma_i) \cdot \phi(d_{ij}, \sigma_j) \cdot \text{Mutual\_knn}(i,j)$$

*Objective Function (Eq. 3):*

$$L = \sum_{(i)} \sum_{(j)} \cdot S_{ij} \cdot \log\!\left(\frac{S_{ij}}{Q_{ij}}\right)$$

Where `σᵢ` is the scaling parameter, `Sᵢⱼ` is pairwise high-dimensional similarity, and `Qᵢⱼ` is the low-dimensional similarity target.

**Dimensionality reduction outcomes:**

| Dataset | Input shape | Output shape (temporal) |
|---|---|---|
| BEED | 4000 × 16 | 4000 × 3 |
| BONN | 4600 × 178 | 4600 × 3 |

### 3.4 Spectral Features Using FFT

FFT transforms EEG signals from the **time domain** to the **frequency domain**, enabling analysis of brain rhythms (delta, theta, alpha, beta, gamma waves).

**Mathematical formulation (Eq. 4):**

$$X_k = \sum_{n=0}^{N-1} x_j \cdot e^{-\frac{2\pi i j k}{N}}$$

Where `Xₖ` is the frequency-domain output, `N` is the total number of samples, `k` indexes frequency bins (0 to N−1), and the exponential term encodes the phase shift.

**Combined feature dimensions after FFT + UMAP fusion:**

| Dataset | After FFT (spectral) | Combined (spectral + temporal) |
|---|---|---|
| BEED | 4000 × 16 | 4000 × 19 |
| BONN | 4600 × 178 | 4600 × 181 |

### 3.5 Model Selection Rationale

The complementarity of FFT and UMAP addresses the **non-stationarity** of EEG signals:

```text
EEG Signal Characteristics vs. Feature Type
─────────────────────────────────────────────────────────
  Rhythmic oscillations              Temporal dynamics
  (power, frequency bands)           (spike-like onset,
                                      transient patterns)
         │                                   │
         ▼                                   ▼
        FFT                               UMAP
  (Spectral features)            (Temporal features)
  frequency-domain               preserves non-linear
  decomposition                  manifold structure
         │                                   │
         └────────────────┬──────────────────┘
                          ▼
               Combined Feature Set
        (comprehensive seizure characterization,
         robust to noise, generalizes across
         subjects and seizure types)
```

### 3.6 SeqBoostNet — Sequential Boosting Network

SeqBoostNet employs a **two-level stacking ensemble** to improve predictive performance:

```text
SeqBoostNet Architecture
─────────────────────────────────────────────────────────

Input Data (Spectral + Temporal Features)
         │
         ├─────────────────────────────────┐
         │                                 │
         │         LEVEL 0 (Base Models)   │
         │                                 │
         ▼          ▼          ▼           │
    ┌────────┐ ┌────────┐ ┌────────┐      │
    │  LSTM  │ │ XGBoost│ │  Grad. │      │
    │        │ │ (XGB)  │ │Boosting│      │
    │Temporal│ │ Robust │ │ (GB)   │      │
    │patterns│ │accuracy│ │Handles │      │
    │        │ │        │ │outliers│      │
    └───┬────┘ └───┬────┘ └───┬────┘      │
        │          │          │            │
        │   Y_LSTM │Y_XGB Y_GB│            │
        └──────────┴──────────┘            │
                   │                       │
                   ▼                       │
         New Training Data                 │
         (stacked predictions)             │
                   │                       │
         LEVEL 1 (Meta-Model)             │
                   │                       │
                   ▼                       │
           ┌──────────────┐                │
           │   AdaBoost   │◄───────────────┘
           │  (meta-model)│
           └──────┬───────┘
                  │
                  ▼
          Final Prediction
```

#### 3.6.1 Long Short-Term Memory (LSTM)

LSTM processes EEG data sequentially across all channels. Its three gates and cell state manage information flow:

| Gate | Role | Equation |
|---|---|---|
| Forget gate | Decides what to discard from previous cell state | `fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)` |
| Input gate | Decides what new information to add | `iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)` |
| Candidate cell | Computes potential updates (tanh) | `C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)` |
| Cell state update | Combines old state with new candidate | `Cₜ = fₜ·Cₜ₋₁ + iₜ·C̃ₜ` |
| Output gate | Regulates how much of cell state passes forward | `oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)` |
| Hidden state | Used for prediction or passed to next step | `hₜ = oₜ·tanh(Cₜ)` |

#### 3.6.2 Extreme Gradient Boosting (XGBoost)

XGBoost builds an ensemble of weak learners (decision trees) with explicit regularization:

*Loss function (Eq. 11):*

$$L = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

*Regularization (Eq. 12):*

$$\Phi(f) = \gamma T + \frac{1}{2}\lambda \sum \omega_j^2$$

Where `T` is the number of tree leaves, `ωⱼ` are leaf weights, and `γ`, `λ` are regularization parameters.

#### 3.6.3 Gradient Boosting (GB)

GB builds trees sequentially, correcting residual errors from previous iterations:

$$F_m(x) = F_{m-1}(x) + \alpha h_m(x) \quad \text{(Eq. 13)}$$

Where `α` is the learning rate controlling each new tree's influence.

#### 3.6.4 Adaptive Boosting (AdaBoost — Meta-Model)

AdaBoost combines the base model predictions by iteratively reweighting samples:

| Component | Equation |
|---|---|
| Weighted loss | `Lₜ = Σ ωᵢᵗ⁻¹ · l(yᵢ, hₜ(xᵢ))` (Eq. 15) |
| Classifier weight | `αₜ = ½ log((1 − errₜ) / errₜ)` (Eq. 16) |
| Weight update | `wᵢᵗ = wᵢᵗ⁻¹ · exp(−αₜ · yᵢ · hₜ(xᵢ))` (Eq. 17) |
| Final prediction | `H(x) = sign(Σ αₜ hₜ(x))` (Eq. 18) |

#### 3.6.5 Hyperparameters

| Model | Key Hyperparameters |
|---|---|
| LSTM | 128 cells, ReLU activation, dropout 0.5, Sigmoid output, Adam optimizer, Sparse Categorical Cross-Entropy, 100 epochs, batch size 32 |
| XGBoost | 300 estimators, max depth 6, learning rate 0.05, Multi-Softmax objective |
| Gradient Boosting | 100 estimators, learning rate 0.1, max depth 3 |
| AdaBoost | 50 weak learners, learning rate 1.0 |

### 3.7 Performance Evaluation Metrics

| Metric | Formula |
|---|---|
| Accuracy (A) | `(TS + TH) / TP` |
| F1-Score | `2·P·R / (P + R)` |
| F2-Score | `5·P·R / (4·P + R)` |
| Kappa (K) | `(A − CA) / (1 − CA)` |
| Log Loss | `−(1/N) Σ [y log(p) + (1−y) log(1−p)]` |
| MCC | `(TS·TH − FS·FH) / √((TS+FS)(TS+FH)(TH+FS)(TH+FH)·N)` |
| Precision (P) | `TS / (TS + FS)` |
| Recall (R) | `TS / (TS + FH)` |
| Sensitivity | `TS / (TS + FH)` |
| Specificity | `TH / (TH + FS)` |

*Abbreviations: TS = True Seizures, TH = True Healthy, TP = Total Positives, FS = False Seizures, FH = False Healthy*

---

## 4. Results and Discussion

Experiments were run on Windows 10 (64-bit, 8 GB RAM, Intel Core i3-6006U @ 2.00 GHz) in Python.

### 4.1 BEED Test Cases

```text
BEED Classification Cases
│
├── A1 — Generalized  vs. Focal
├── A2 — Generalized  vs. Healthy
├── A3 — Focal        vs. Healthy
├── A4 — Focal        vs. Seizure Events
├── A5 — Generalized  vs. Seizure Events
└── A6 — Seizure Events vs. Healthy
```

### 4.2 BONN Test Cases

```text
BONN Classification Cases
│
├── B1 — Seizure      vs. Healthy
├── B2 — Seizure      vs. Tumor
├── B3 — Seizure      vs. Eye Closed
├── B4 — Seizure      vs. Eye Opened
└── B5 — Eye Closed   vs. Eye Opened
```

### 4.3 Performance on BEED

| Metric | A1 | A2 | A3 | A4 | A5 | A6 |
|---|---|---|---|---|---|---|
| Accuracy | 95.91 | 99.66 | 99.83 | 91.16 | 94.01 | 99.66 |
| Precision | 96.01 | 99.66 | 99.83 | 91.25 | 94.01 | 99.66 |
| Recall | 95.91 | 99.66 | 99.83 | 91.16 | 94.01 | 99.66 |
| F1-score | 95.91 | 99.66 | 99.83 | 91.15 | 94.01 | 99.66 |
| Kappa | 91.83 | 99.33 | 99.66 | 82.27 | 87.98 | 99.33 |
| MCC | 91.91 | 99.33 | 99.66 | 82.38 | 87.99 | 99.33 |
| ROC-AUC | 95.98 | 99.65 | 99.82 | 91.06 | 94.01 | 99.65 |
| Sensitivity | 94.05 | 1.00 | 1.00 | 93.89 | 93.56 | 1.00 |
| Specificity | 97.92 | 99.30 | 99.65 | 88.23 | 94.46 | 99.30 |
| F2-score | 95.93 | 99.66 | 99.83 | 91.18 | 94.01 | 99.66 |
| Log Loss | 1.47 | 0.12 | 0.06 | 0.61 | 2.16 | 0.12 |
| Time (s) | 28 | 28 | 28 | 30 | 31 | 48 |

**Top performers:** A2, A3, A6 (>99% across all metrics, perfect sensitivity). A4 and A5 are the hardest cases (within-seizure-type discrimination) but still exceed 91%.

ROC-AUC highlights:
- AUC = 1.00: A3, A4, A6
- AUC = 0.96: A5
- AUC = 0.94: A1
- AUC = 0.92: A2

### 4.4 Performance on BONN

| Metric | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|
| Accuracy | 97.39 | 98.40 | 99.34 | 99.63 | 90.79 |
| Precision | 97.40 | 98.40 | 99.35 | 99.63 | 90.85 |
| Recall | 97.39 | 98.40 | 99.34 | 99.63 | 90.79 |
| F1-score | 97.39 | 98.40 | 99.34 | 99.63 | 90.79 |
| Kappa | 94.77 | 96.80 | 98.69 | 99.27 | 81.59 |
| MCC | 94.78 | 96.81 | 98.69 | 99.27 | 81.63 |
| ROC-AUC | 97.41 | 98.38 | 98.69 | 99.64 | 90.84 |
| Sensitivity | 96.77 | 98.87 | 99.01 | 99.43 | 89.49 |
| Specificity | 98.04 | 97.89 | 99.69 | 99.84 | 92.19 |
| F2-score | 97.39 | 98.40 | 99.34 | 99.63 | 90.80 |
| Log Loss | 0.94 | 0.57 | 0.23 | 0.13 | 3.31 |
| Time | 2m 58s | 2m 19s | 2m 53s | 2m 56s | 2m 32s |

**Top performers:** B3 and B4 (>99%), followed by B2 (98.40%). B5 (Eye Closed vs. Eye Opened) is the hardest case at 90.79%.

ROC-AUC highlights: B1 = 1.00, B2 = 0.99, B3 = 0.98, B4 = 0.97, B5 = 0.91.

### 4.5 Time Complexity

| Technique | Complexity |
|---|---|
| FFT | O(N log N) |
| UMAP | O(N × D) |
| LSTM | O(N) |
| XGBoost | O(M × T) |
| Gradient Boosting | O(M × T) |
| AdaBoost | O(M × T) |
| **SeqBoostNet** | **O(N log N + N×D + M×T)** |

Where `N` = number of samples, `D` = number of dimensions, `M` = number of features, `T` = boosting iterations.

### 4.6 Comparison with Feature Extraction Techniques (BEED)

| Case | WT | STATS | STFT | PCA | ICA | HHT | EMD | **Proposed** |
|---|---|---|---|---|---|---|---|---|
| A1 | 89.91 | 83.83 | 81.42 | 85.83 | 85.67 | 78.75 | 83.58 | **93.33** |
| A2 | 89.83 | 92.19 | 91.99 | 90.83 | 90.99 | 94.75 | 93.58 | **99.58** |
| A3 | 92.30 | 91.42 | 92.51 | 93.67 | 92.75 | 94.42 | 94.33 | **99.83** |
| A4 | 85.08 | 68.75 | 67.83 | 83.58 | 82.33 | 59.17 | 70.92 | **89.91** |
| A5 | 89.91 | 76.42 | 75.42 | 90.33 | 90.92 | 66.25 | 78.08 | **92.25** |
| A6 | 91.83 | 93.33 | 93.75 | 94.67 | 94.75 | 92.08 | 93.99 | **99.66** |
| **Avg** | 89.81 | 84.32 | 83.82 | 89.82 | 89.57 | 80.90 | 85.75 | **95.76** |

### 4.7 Comparison with Feature Extraction Techniques (BONN)

| Case | WT | STATS | STFT | PCA | ICA | HHT | EMD | **Proposed** |
|---|---|---|---|---|---|---|---|---|
| B1 | 93.90 | 93.72 | 91.22 | 92.13 | 93.26 | 93.33 | 93.43 | **97.10** |
| B2 | 90.41 | 91.46 | 91.25 | 90.58 | 90.22 | 90.23 | 91.26 | **98.47** |
| B3 | 92.26 | 92.96 | 93.71 | 93.68 | 93.19 | 93.29 | 94.19 | **99.42** |
| B4 | 93.64 | 94.93 | 94.28 | 91.99 | 91.55 | 93.71 | 94.78 | **99.56** |
| B5 | 86.74 | 76.67 | 75.80 | 76.23 | 76.80 | 75.58 | 82.03 | **90.28** |
| **Avg** | 91.39 | 89.95 | 89.25 | 88.92 | 89.00 | 89.23 | 91.14 | **96.97** |

### 4.8 Comparison with Stacking Classifiers (BEED)

| Case | Stack 1 | Stack 2 | **SeqBoostNet** |
|---|---|---|---|
| A1 | 83.00 | 86.58 | **95.91** |
| A2 | 91.75 | 90.92 | **99.66** |
| A3 | 90.67 | 89.91 | **99.83** |
| A4 | 71.67 | 75.58 | **91.16** |
| A5 | 78.42 | 82.67 | **94.01** |
| A6 | 99.58 | 99.75 | **99.66** |
| **Avg** | 85.85 | 87.57 | **96.71** |

*Stack 1: XGBoost + LightGBM → Bagging Classifier. Stack 2: RF + LightGBM + GB → XGBoost.*

### 4.9 Comparison with Stacking Classifiers (BONN)

| Case | Stack 1 | Stack 2 | **SeqBoostNet** |
|---|---|---|---|
| B1 | 94.88 | 93.96 | **97.39** |
| B2 | 96.26 | 95.04 | **98.40** |
| B3 | 97.28 | 96.35 | **99.34** |
| B4 | 95.49 | 96.71 | **99.63** |
| B5 | 87.90 | 89.42 | **90.79** |
| **Avg** | 94.36 | 94.30 | **97.11** |

---

## 5. Conclusions and Future Work

SeqBoostNet achieves state-of-the-art seizure classification through its integrated spectral-temporal feature engineering and stacking ensemble design:

```text
Performance Summary
─────────────────────────────────────────────────────
  Focal vs. Generalized (A1)   →  95.91% accuracy
  Best binary BEED (A3)        →  99.83% accuracy
  Average BEED                 →  96.71% accuracy

  Seizure vs. Tumor (B2)       →  98.40% accuracy
  Best binary BONN (B4)        →  99.63% accuracy
  Average BONN                 →  97.11% accuracy

  vs. best competing stack     →  +10.86 pp (BEED avg)
                                →  + 2.75 pp (BONN avg)
─────────────────────────────────────────────────────
```

**Strengths:**
- Handles variability in seizure presentations better than existing models
- Reduces dependency on extensive preprocessing/augmentation
- Flexible architecture — additional base models can be incorporated
- Computationally scalable: `O(N log N + N×D + M×T)`

**Limitations:**
- Feature relevance can differ significantly across datasets
- Stacking complexity may impact performance on very heterogeneous clinical datasets
- Generalizability of stacked models across diverse contexts remains a challenge

**Future directions:**
- Improve adaptability of the stacking approach across diverse clinical contexts
- Integration with brain-computer interfaces (BCI)
- Exploration of personalized medicine applications
- Real-time clinical deployment

---

## References

1. N. Kumari et al., "Leveraging wearable sensors and supervised learning paradigm as a configurable solution for epileptic patients," *Int. J. Computing and Digital Systems*, vol. 14, no. 1, pp. 10243–10250, 2023.
2. S. M. Usman, S. Khalid, M. H. Aslam, "Epileptic seizures prediction using deep learning techniques," *IEEE Access*, vol. 8, pp. 39998–40007, 2020.
3. E. Giourou et al., "Introduction to epilepsy and related brain disorders," *Cyberphysical Systems for Epilepsy and Related Brain Disorders*, pp. 11–38, 2015.
4. M. D. Holmes, M. Brown, D. M. Tucker, "Are 'generalized' seizures truly generalized? Evidence of localized mesial frontal and frontopolar discharges in absence," *Epilepsia*, vol. 45, no. 12, pp. 1568–1579, 2004.
5. R. K. Maganti and P. Rutecki, "EEG and epilepsy monitoring," *CONTINUUM: Lifelong Learning in Neurology*, vol. 19, no. 3, pp. 598–622, 2013.
6. M. Gaik et al., "Functional divergence of the two elongator subcomplexes during neurodevelopment," *EMBO Molecular Medicine*, vol. 14, no. 7, p. e15608, 2022.
7. S. Ashokkumar et al., "Implementation of deep neural networks for classifying EEG signals using FST for epileptic seizure detection," *Int. J. Imaging Systems and Technology*, vol. 31, no. 2, pp. 895–908, 2021.
8. M. S. Islam, K. Thapa, S.-H. Yang, "Epileptic-net: an improved epileptic seizure detection system using dense convolutional block with attention network from EEG," *Sensors*, vol. 22, no. 3, p. 728, 2022.
9. F. Hassan, S. F. Hussain, S. M. Qaisar, "Epileptic seizure detection using a hybrid 1D CNN-ML approach from EEG data," *J. Healthcare Engineering*, vol. 2022, p. 9579422, 2022.
10. T. S. Cleatus and M. Thungamani, "Epileptic seizure detection using spectral transformation and CNNs," *J. Institution of Engineers (India): Series B*, vol. 103, no. 4, pp. 1115–1125, 2022.
11. M. H. Aslam et al., "Classification of EEG signals for prediction of epileptic seizures," *Applied Sciences*, vol. 12, no. 14, p. 7251, 2022.
12. A. Altameem et al., "Performance analysis of ML algorithms for classifying hand motion-based EEG brain signals," *Computer Systems Science & Engineering*, vol. 42, no. 3, 2022.
13. X. Xu et al., "Patient-specific method for predicting epileptic seizures based on DRSN-GRU," *Biomedical Signal Processing and Control*, vol. 81, p. 104449, 2023.
14. Y. Gao et al., "Automatic epileptic seizure classification in multichannel EEG time series with linear discriminant analysis," *Technology and Health Care*, vol. 28, no. 1, pp. 23–33, 2020.
15. S. Urbina Fredes et al., "Enhanced epileptic seizure detection through wavelet-based analysis of EEG signal processing," *Applied Sciences*, vol. 14, no. 13, p. 5783, 2024.
16. S. Y. Shah et al., "Epileptic seizure classification based on random neural networks using DWT for EEG signal decomposition," *Applied Sciences*, vol. 14, no. 2, p. 599, 2024.
17. D. Hernandez et al., "Detecting epilepsy in EEG signals using time, frequency and time-frequency domain features," *Computer Science and Engineering—Theory and Applications*, pp. 167–182, 2018.
18. M. G. Tsipouras, "Spectral information of EEG signals with respect to epilepsy classification," *EURASIP J. Advances in Signal Processing*, vol. 2019, no. 1, pp. 1–17, 2019.
19. K. Akyol, "Stacking ensemble based deep neural networks modeling for effective epileptic seizure detection," *Expert Systems with Applications*, vol. 148, p. 113239, 2020.
20. M. K. M. Rabby et al., "Wavelet transform-based feature extraction approach for epileptic seizure classification," *Proc. ACM Southeast Conf.*, pp. 164–169, 2021.
21. J. Jing et al., "Classification and identification of epileptic EEG signals based on signal enhancement," *Biomedical Signal Processing and Control*, vol. 71, p. 103248, 2022.
22. S. Mishra et al., "A DM-ELM based classifier for EEG brain signal classification for epileptic seizure detection," *Communicative & Integrative Biology*, vol. 16, no. 1, p. 2153648, 2023.
23. R. G. Andrzejak et al., "Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity," *Physical Review E*, vol. 64, no. 6, p. 061907, 2001.
24. D. Thara, B. PremaSudha, F. Xiong, "Auto-detection of epileptic seizure events using deep neural network with different feature scaling techniques," *Pattern Recognition Letters*, vol. 128, pp. 544–550, 2019.
25. T. Liu et al., "Unsupervised feature representation based on deep Boltzmann machine for seizure detection," *IEEE Trans. Neural Systems and Rehabilitation Engineering*, vol. 31, pp. 1624–1634, 2023.
26. Y. Pan et al., "Epileptic seizure detection with hybrid time-frequency EEG input: A deep learning approach," *Computational and Mathematical Methods in Medicine*, vol. 2022, p. 8724536, 2022.
27. K. Singh and J. Malhotra, "Two-layer LSTM network-based prediction of epileptic seizures using EEG spectral features," *Complex & Intelligent Systems*, vol. 8, no. 3, pp. 2405–2418, 2022.
28. Aayesha et al., "Machine learning-based EEG signals classification model for epileptic seizure detection," *Multimedia Tools and Applications*, vol. 80, no. 12, pp. 17849–17877, 2021.
29. T. Islam et al., "Performance investigation of epilepsy detection from noisy EEG signals using base-2-meta stacking classifier," *Scientific Reports*, vol. 14, no. 1, p. 10792, 2024.
30. S. Chatterjee and Y.-C. Byun, "EEG-based emotion classification using stacking ensemble approach," *Sensors*, vol. 22, no. 21, p. 8550, 2022.
