"""
Sanity-check tests for the preprocessing and modeling pipeline.
Run with: pytest tests/test.py -v
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_raw_data,
    clean_data,
    create_target,
    add_features,
    get_feature_columns,
    time_split,
    build_dataset,
    HORIZON,
)


def test_load_raw_data():
    """Data loads without error and has expected shape."""
    df = load_raw_data()
    assert len(df) > 10000, f"Expected >10k rows, got {len(df)}"
    assert "Close" in df.columns
    assert "DateTime" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["DateTime"])


def test_clean_data_removes_zero_volume():
    """Cleaning should remove zero-volume rows."""
    df = load_raw_data()
    n_before = len(df)
    df_clean = clean_data(df)
    assert len(df_clean) < n_before, "Should remove some zero-volume rows"
    assert (df_clean["Volume"] > 0).all(), "No zero-volume rows should remain"


def test_clean_data_sorted():
    """Cleaned data should be sorted by time."""
    df = clean_data(load_raw_data())
    assert df["DateTime"].is_monotonic_increasing, "Data must be sorted by time"


def test_create_target_binary():
    """Target should be binary (0 or 1)."""
    df = clean_data(load_raw_data())
    df = create_target(df, horizon=1)
    assert set(df["target"].unique()).issubset({0, 1})


def test_create_target_no_nan():
    """Target column should have no NaN values."""
    df = clean_data(load_raw_data())
    df = create_target(df, horizon=1)
    assert df["target"].isna().sum() == 0


def test_create_target_removes_last_rows():
    """Target creation drops last `horizon` rows."""
    df = clean_data(load_raw_data())
    n = len(df)
    df_t = create_target(df, horizon=3)
    assert len(df_t) == n - 3


def test_features_no_leakage():
    """Feature columns should not include raw OHLCV or target."""
    df = clean_data(load_raw_data())
    df = add_features(df)
    df = create_target(df, horizon=1)
    feat_cols = get_feature_columns(df)
    forbidden = {"Open", "High", "Low", "Close", "Volume", "target"}
    assert forbidden.isdisjoint(set(feat_cols)), f"Features contain forbidden: {forbidden & set(feat_cols)}"


def test_time_split_no_overlap():
    """Train period must end before test period begins."""
    df = clean_data(load_raw_data())
    df = add_features(df)
    df = create_target(df, horizon=1)
    df = df.dropna(subset=get_feature_columns(df)).reset_index(drop=True)

    train, test = time_split(df, test_frac=0.2)
    assert train["DateTime"].max() < test["DateTime"].min(), "Train must end before test"


def test_build_dataset_shapes():
    """build_dataset returns consistent shapes."""
    X_train, y_train, X_test, y_test, feat_cols, df = build_dataset()
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == len(feat_cols)
    assert X_test.shape[1] == len(feat_cols)
    assert X_train.shape[0] > X_test.shape[0], "Train should be larger than test"


def test_no_nan_in_features():
    """Final feature matrices should have no NaN."""
    X_train, _, X_test, _, _, _ = build_dataset()
    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert not np.isnan(X_test).any(), "NaN in X_test"
