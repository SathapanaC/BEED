"""Data loading and train/test splitting."""

import pandas as pd
from sklearn.model_selection import train_test_split

from beed.config import CHANNEL_NAMES, N_CHANNELS, RAW_FILE, RANDOM_STATE


def load_raw() -> pd.DataFrame:
    """Load the raw BEED CSV and attach readable channel names."""
    df = pd.read_csv(RAW_FILE)
    rename = {f"X{i+1}": CHANNEL_NAMES[i] for i in range(N_CHANNELS)}
    return df.rename(columns=rename)


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split.

    Val is carved from the training portion so the test set stays untouched.
    Returns (train, val, test).
    """
    X, y = df.drop(columns=["y"]), df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    # val_size is relative to the *remaining* training data
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_relative, stratify=y_train, random_state=RANDOM_STATE
    )

    train = X_train.assign(y=y_train)
    val = X_val.assign(y=y_val)
    test = X_test.assign(y=y_test)
    return train, val, test
