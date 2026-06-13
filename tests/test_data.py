"""Smoke tests for data loading and splitting."""

import pandas as pd
import pytest

from beed.config import N_CHANNELS
from beed.data import load_raw, split


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


def test_load_shape(raw_df):
    n_rows, n_cols = raw_df.shape
    assert n_cols == N_CHANNELS + 1
    assert n_rows > 0


def test_no_nulls(raw_df):
    assert raw_df.isnull().sum().sum() == 0


def test_class_balance(raw_df):
    counts = raw_df["y"].value_counts()
    assert set(counts.index) == {0, 1, 2, 3}


def test_split_sizes(raw_df):
    train, val, test = split(raw_df)
    total = len(train) + len(val) + len(test)
    assert total == len(raw_df)
    assert abs(len(test) / total - 0.2) < 0.01


def test_split_stratification(raw_df):
    train, val, test = split(raw_df)
    for split_df in [train, val, test]:
        assert set(split_df["y"].unique()) == {0, 1, 2, 3}
