# BEED — EEG Epilepsy Classification

## Project summary
Reproduce and extend the SeqBoostNet paper (IJCDS 2025) on the **Bangalore EEG Epilepsy Dataset**.
Goal: classify EEG recordings into 4 classes (Healthy / Focal / Generalized / Seizure Events)
using FFT + UMAP features fed into an LSTM + XGBoost + GradientBoosting → AdaBoost ensemble.

## Environment
```
uv run <command>          # always run through uv — it manages the venv
uv sync --extra dev       # install all deps including dev
uv run pytest             # run tests
uv run jupyter lab        # open notebooks
```
Python 3.12 at `C:\Python312\python.exe`.

## Layout
```
src/beed/
  config.py    — paths, constants, channel names, class labels
  data.py      — load_raw(), split() (stratified train/val/test)
  features.py  — fft_features(), umap_features(), build_features()
data/raw/      — raw CSVs (gitignored, keep locally)
data/processed/— serialised feature matrices (gitignored)
notebooks/     — exploratory analysis
tests/         — pytest tests against src/beed
reports/figures/— saved plots (gitignored)
docs/          — dataset description and reference article
```

## Data
- `data/raw/BEED_Data.csv` — 4,000 × 17 (16 EEG channels + label `y`)
- Sampling rate: 256 Hz, 20 s per recording, 10-20 electrode system
- Classes: 0=Healthy, 1=Focal, 2=Generalized, 3=Seizure Events
- Data is **ordered by class** — always use stratified splits

## Feature pipeline (paper)
1. **StandardScaler** per channel
2. **FFT** → 16 spectral amplitude features (one per channel)
3. **UMAP** (3D) on all 16 channels jointly → 3 temporal features
4. Concatenate → 19-feature matrix fed to SeqBoostNet

## Key decisions
- Raw data is gitignored; only code and notebooks go to git
- UMAP reducer is fit on train only, then `.transform()` on val/test
- Use `RANDOM_STATE = 42` everywhere for reproducibility
