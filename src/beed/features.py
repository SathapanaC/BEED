"""Feature engineering: FFT spectral features + UMAP temporal embedding."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from beed.config import CHANNEL_NAMES, N_UMAP_COMPONENTS, RANDOM_STATE


def scale(
    X: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Per-channel StandardScaler. Fit on train, transform val/test."""
    if fit:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X[CHANNEL_NAMES])
    else:
        assert scaler is not None, "scaler must be provided when fit=False"
        scaled = scaler.transform(X[CHANNEL_NAMES])
    return pd.DataFrame(scaled, columns=CHANNEL_NAMES, index=X.index), scaler


def fft_features(X_scaled: pd.DataFrame) -> pd.DataFrame:
    """Per-row FFT across 16 channels → 16 spatial-frequency amplitudes per sample.

    axis=1 treats the 16 simultaneous channel readings as a spatial signal,
    preserving the (n_samples, 16) shape through the transform.
    """
    amplitudes = np.abs(np.fft.fft(X_scaled[CHANNEL_NAMES].values, axis=1))
    cols = [f"{ch}_fft" for ch in CHANNEL_NAMES]
    return pd.DataFrame(amplitudes, columns=cols, index=X_scaled.index)


def umap_features(
    X_scaled: pd.DataFrame,
    reducer: UMAP | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, UMAP]:
    """Reduce all 16 scaled channels jointly to N_UMAP_COMPONENTS dimensions.

    Pass a pre-fitted reducer with fit=False to transform val/test sets.
    Returns (embedding_df, reducer).
    """
    if fit:
        reducer = UMAP(n_components=N_UMAP_COMPONENTS, random_state=RANDOM_STATE)
        embedding = reducer.fit_transform(X_scaled[CHANNEL_NAMES].values)
    else:
        assert reducer is not None, "reducer must be provided when fit=False"
        embedding = reducer.transform(X_scaled[CHANNEL_NAMES].values)

    cols = [f"umap_{i+1}" for i in range(N_UMAP_COMPONENTS)]
    return pd.DataFrame(embedding, columns=cols, index=X_scaled.index), reducer


def build_features(
    X: pd.DataFrame,
    scaler: StandardScaler | None = None,
    umap_reducer: UMAP | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler, UMAP]:
    """Full pipeline: scale → FFT (16) + UMAP (3) → 19-feature matrix.

    On train: fit=True (fits scaler and UMAP, returns them for reuse).
    On val/test: fit=False, pass fitted scaler and umap_reducer.
    Returns (features_df, scaler, umap_reducer).
    """
    X_scaled, scaler = scale(X, scaler=scaler, fit=fit)
    fft = fft_features(X_scaled)
    umap, reducer = umap_features(X_scaled, reducer=umap_reducer, fit=fit)
    combined = pd.concat([fft, umap], axis=1)
    return combined, scaler, reducer
