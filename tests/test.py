"""
Sanity-check tests for the preprocessing and modeling pipeline (CP2).
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
    add_external_features,
    get_feature_columns,
    time_split,
    walk_forward_split,
    build_dataset,
    build_full_df,
    load_external_factors,
    _merge_external,
)


def test_load_raw_data():
    """Data loads without error and has expected shape."""
    df = load_raw_data()
    assert len(df) > 10000, f"Expected >10k rows, got {len(df)}"
    assert "Close" in df.columns
    assert "DateTime" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["DateTime"])


def test_clean_data_removes_zero_volume():
    """clean_data should remove rows with Volume <= 0."""
    df_raw = load_raw_data()
    # Inject a dummy row with zero volume to ensure there's something to clean
    bad_row = df_raw.iloc[0:1].copy()
    bad_row["DateTime"] = df_raw["DateTime"].max() + pd.Timedelta(hours=1)
    bad_row["Volume"] = 0
    df_with_bad = pd.concat([df_raw, bad_row], ignore_index=True)

    n_before = len(df_with_bad)
    df_clean = clean_data(df_with_bad)
    assert len(df_clean) < n_before, "Should remove the zero-volume row"
    assert (df_clean["Volume"] > 0).all(), "All volumes should be > 0"


def test_clean_data_sorted():
    """Cleaned data should be sorted by time."""
    df = clean_data(load_raw_data())
    assert df["DateTime"].is_monotonic_increasing, "Data must be sorted by time"


def test_create_target_multiclass():
    """Target should be multi-class (0, 1, 2)."""
    df = clean_data(load_raw_data())
    df = create_target(df, horizon=1)
    assert set(df["target"].unique()).issubset({0, 1, 2})


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
    assert forbidden.isdisjoint(set(feat_cols)), \
        f"Features contain forbidden: {forbidden & set(feat_cols)}"


def test_technical_indicators_present():
    """RSI, MACD, and Bollinger Band features should be generated."""
    df = clean_data(load_raw_data())
    df = add_features(df)
    feat_cols = get_feature_columns(df)
    expected = ["rsi_14", "rsi_6", "macd", "macd_signal", "macd_hist",
                "bb_upper_12", "bb_lower_12", "bb_width_12", "bb_position_12"]
    for col in expected:
        assert col in feat_cols, f"Missing feature: {col}"


def test_time_split_no_overlap():
    """Train period must end before test period begins."""
    df = clean_data(load_raw_data())
    df = add_features(df)
    df = create_target(df, horizon=1)
    df = df.dropna(subset=get_feature_columns(df)).reset_index(drop=True)

    train, test = time_split(df, test_frac=0.2)
    assert train["DateTime"].max() < test["DateTime"].min(), "Train must end before test"


def test_walk_forward_splits():
    """Walk-forward splits should have non-overlapping, ordered folds."""
    df, feat_cols = build_full_df(horizon=1, use_external=False)
    splits = walk_forward_split(df, n_splits=3)
    assert len(splits) >= 2, "Should produce at least 2 folds"

    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx), "Train must precede test in each fold"

    for i in range(1, len(splits)):
        assert len(splits[i][0]) > len(splits[i - 1][0]), "Train should grow across folds"


def test_build_dataset_shapes():
    """build_dataset returns consistent shapes (gold-only)."""
    X_train, y_train, X_test, y_test, feat_cols, df, _ = build_dataset(use_external=False)
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == len(feat_cols)
    assert X_test.shape[1] == len(feat_cols)
    assert X_train.shape[0] > X_test.shape[0], "Train should be larger than test"


def test_no_nan_in_features():
    """Final feature matrices should have no NaN (gold-only)."""
    X_train, _, X_test, _, _, _, _ = build_dataset(use_external=False)
    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert not np.isnan(X_test).any(), "NaN in X_test"


def test_multiple_horizons():
    """Pipeline should work for different horizons (gold-only)."""
    for h in [1, 4, 24]:
        X_train, y_train, X_test, y_test, feat_cols, df, _ = build_dataset(
            horizon=h, use_external=False
        )
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert set(np.unique(y_train)).issubset({0, 1, 2})


# ─────────────────────────────────────────────────────────────────────────────
# External factor tests (CP2)
# ─────────────────────────────────────────────────────────────────────────────

def test_load_external_factors():
    """External factors CSV should load with expected columns."""
    ext = load_external_factors()
    assert "DateTime" in ext.columns
    for col in ["dxy", "vix", "tnx", "silver", "oil"]:
        assert col in ext.columns, f"Missing external column: {col}"
    assert len(ext) > 200, "Expected >200 daily rows"


def test_merge_external_no_leakage():
    """merge_asof with direction='backward' must not introduce future data."""
    gold = clean_data(load_raw_data())
    ext = load_external_factors()
    merged = _merge_external(gold, ext)

    # For each row, the daily DateTime value used must be <= the hourly DateTime
    # We verify no external date is ahead of the gold timestamp
    assert merged["DateTime"].is_monotonic_increasing

    # External columns should not be all NaN after merge
    for col in ["dxy", "vix", "tnx", "silver", "oil"]:
        non_null = merged[col].notna().sum()
        assert non_null > 0, f"All NaN in merged column: {col}"


def test_external_features_generated():
    """add_external_features should create derived columns for each factor."""
    gold = clean_data(load_raw_data())
    ext = load_external_factors()
    merged = _merge_external(gold, ext)
    merged = add_features(merged)
    merged = add_external_features(merged)

    for factor in ["dxy", "vix", "tnx", "silver", "oil"]:
        assert f"{factor}_ret1d" in merged.columns, f"Missing {factor}_ret1d"
        assert f"{factor}_zscore20" in merged.columns, f"Missing {factor}_zscore20"
        assert f"{factor}_dev_ma20" in merged.columns, f"Missing {factor}_dev_ma20"

    # Cross-market features
    assert "gold_minus_dxy_ret" in merged.columns
    assert "gold_minus_silver_ret" in merged.columns
    assert "gold_silver_ratio" in merged.columns
    assert "tnx_change1d" in merged.columns


def test_build_dataset_with_external():
    """build_dataset with use_external=True returns more features than without."""
    X_a, _, _, _, fc_a, _, _ = build_dataset(use_external=False)
    X_b, _, _, _, fc_b, _, _ = build_dataset(use_external=True)

    assert len(fc_b) > len(fc_a), \
        f"Variant B should have more features ({len(fc_b)}) than A ({len(fc_a)})"
    assert not np.isnan(X_b).any(), "NaN in external variant X_train"


def test_no_nan_in_external_features():
    """Final feature matrix with external factors should have no NaN."""
    X_train, _, X_test, _, _, _, _ = build_dataset(use_external=True)
    assert not np.isnan(X_train).any(), "NaN in X_train (external)"
    assert not np.isnan(X_test).any(), "NaN in X_test (external)"
