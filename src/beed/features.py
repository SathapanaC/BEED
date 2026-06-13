"""Feature engineering: FFT spectral features + UMAP temporal embedding."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from beed.config import CHANNEL_NAMES, N_UMAP_COMPONENTS, RANDOM_STATE


def fft_features(X: pd.DataFrame) -> pd.DataFrame:
    """Compute per-channel FFT magnitude and return as a DataFrame."""
    amplitudes = np.abs(np.fft.rfft(X[CHANNEL_NAMES].values, axis=0))
    cols = [f"{ch}_fft" for ch in CHANNEL_NAMES]
    return pd.DataFrame(amplitudes, columns=cols, index=X.index)


def umap_features(
    X: pd.DataFrame,
    reducer: UMAP | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, UMAP]:
    """Reduce all channels jointly to N_UMAP_COMPONENTS dimensions.

    Pass a pre-fitted reducer with fit=False to transform val/test sets.
    Returns (embedding_df, reducer).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X[CHANNEL_NAMES])

    if fit:
        reducer = UMAP(n_components=N_UMAP_COMPONENTS, random_state=RANDOM_STATE)
        embedding = reducer.fit_transform(scaled)
    else:
        assert reducer is not None, "reducer must be provided when fit=False"
        embedding = reducer.transform(scaled)

    cols = [f"umap_{i+1}" for i in range(N_UMAP_COMPONENTS)]
    return pd.DataFrame(embedding, columns=cols, index=X.index), reducer


def build_features(
    X: pd.DataFrame,
    umap_reducer: UMAP | None = None,
    fit_umap: bool = True,
) -> tuple[pd.DataFrame, UMAP]:
    """Combine FFT and UMAP features into the 19-column feature matrix."""
    fft = fft_features(X)
    umap, reducer = umap_features(X, reducer=umap_reducer, fit=fit_umap)
    combined = pd.concat([fft, umap], axis=1)
    return combined, reducer
