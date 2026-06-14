# BEED — EEG Epilepsy Classification

Reproduce and extend the **SeqBoostNet** paper ([Najmusseher & Nizar Banu, IJCDS 2025](docs/seqboostnet_article.md)) on the Bangalore EEG Epilepsy Dataset.

**Goal:** classify EEG recordings into 4 classes using FFT + UMAP features fed into an LSTM + XGBoost + GradientBoosting → AdaBoost ensemble.

## Dataset

| Property | Value |
|----------|-------|
| Samples | 8,000 (2,000 per class) |
| Channels | 16 (10-20 electrode system) |
| Sampling rate | 256 Hz |
| Recording length | 20 s per session |
| File | `data/raw/BEED_Data.csv` |

> **Note:** The paper states 4,000 rows; the actual CSV contains 8,000. All experiments use the full 8,000-row file.

| Label | Class | Description |
|-------|-------|-------------|
| `0` | Healthy | Seizure-free baseline |
| `1` | Focal | Seizure localised to one hemisphere |
| `2` | Generalized | Seizure across both hemispheres |
| `3` | Seizure Events | Seizure activity during physical movement |

> Data is ordered by class — always use stratified splits.

## Feature Pipeline

```
Raw EEG (8,000 × 16)
        │
        ├── StandardScaler (per channel, fit on train only)
        │
        ├── FFT  → 16 spectral amplitude features
        └── UMAP → 3 embedding dimensions (3D, fit on train only)
                │
                └── Combined feature matrix (8,000 × 19) → SeqBoostNet
```

## Results

### 4-class multiclass (train 5,600 / val 800 / test 1,600)

| Model | Features | Test Accuracy | Test Macro-F1 |
|-------|----------|--------------|---------------|
| Logistic Regression | Raw 16 | 46.1% | 0.468 |
| Logistic Regression | FFT+UMAP 19 | 70.2% | 0.700 |
| Random Forest | FFT+UMAP 19 | 88.5% | 0.885 |
| **Random Forest** | **Raw 16** | **95.4%** | **0.954** |
| SeqBoostNet | FFT+UMAP 19 | 84.4% | 0.845 |

### Binary cases A1–A6 (SeqBoostNet, test accuracy %)

| Case | Task | Ours | Paper |
|------|------|------|-------|
| A1 | Generalized vs Focal | 93.62% | 95.91% |
| A2 | Generalized vs Healthy | 99.62% | 99.66% |
| A3 | Focal vs Healthy | 99.88% | 99.83% |
| A4 | Focal vs Seizure Events | 93.25% | 91.16% |
| A5 | Generalized vs Seizure Events | 87.25% | 94.01% |
| A6 | Seizure Events vs Healthy | 99.75% | 99.66% |
| **Avg** | | **95.56%** | **96.71%** |

## Experiments

| ID | Notebook | Description |
|----|----------|-------------|
| EXP-001 | `notebooks/01_eda.ipynb` | Exploratory data analysis |
| EXP-002 | `notebooks/02_features.ipynb` | FFT + UMAP feature engineering |
| EXP-003 | `notebooks/03_baselines.ipynb` | LR + RF on raw 16 features |
| EXP-004 | `notebooks/04_classifiers_fft_umap.ipynb` | LR + RF on 19-feature FFT+UMAP matrix |
| EXP-005 | `notebooks/05_seqboostnet.ipynb` | SeqBoostNet 4-class implementation |
| EXP-006 | `notebooks/06_binary_cases.ipynb` | SeqBoostNet on binary cases A1–A6 |
| EXP-007 | `notebooks/07_threshold_tuning.ipynb` | Threshold tuning on A1, A4, A5 — no gain; 0.5 is near-optimal |

See [`experiments.md`](experiments.md) for detailed findings, open questions, and the lab notebook.

## Project Layout

```
src/beed/
  config.py      — paths, constants, channel names, class labels
  data.py        — load_raw(), split() (stratified train/val/test)
  features.py    — scale(), fft_features(), umap_features(), build_features()
data/raw/        — raw CSVs (gitignored)
data/processed/  — serialised feature matrices (gitignored)
notebooks/       — exploratory analysis and experiments
tests/           — pytest suite
docs/            — dataset description and reference paper
reports/figures/ — saved plots (gitignored)
```

## Quickstart

```bash
uv sync --extra dev       # install all deps (includes TensorFlow for LSTM)
uv run pytest             # run tests
uv run jupyter lab        # open notebooks
```

Requires Python 3.12.

## Reference

Najmusseher & Nizar Banu P K, *Feature Engineering for Epileptic Seizure Classification Using SeqBoostNet*, IJCDS 2025, Vol. 17, No. 1.
