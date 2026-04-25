"""
Preprocessing module for gold futures price direction prediction.

Handles data loading, cleaning, feature engineering, target creation,
and time-based train/test splitting.
All features are computed using only past information to prevent lookahead leakage.
"""

import numpy as np
import pandas as pd
from pathlib import Path


RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

HORIZON = 1


def load_raw_data(filename: str = "gold_data_1h_cleaned.csv") -> pd.DataFrame:
    """Load raw CSV and parse datetime."""
    path = RAW_DIR / filename
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, rows with zero volume, and NaN prices."""
    df = df.drop_duplicates(subset="DateTime", keep="first")
    df = df[df["Volume"] > 0].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def create_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Create binary target: 1 if Close price `horizon` bars ahead > current Close, else 0.
    Rows where target cannot be computed (last `horizon` rows) are dropped.
    """
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    valid_mask = future_close.notna()
    df = df[valid_mask].copy()
    future_close = future_close[valid_mask]
    df["target"] = (future_close > df["Close"]).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features using only past information.
    Groups: price lags, return lags, rolling statistics,
    high-low range, volume features, deviation from moving averages,
    hour-of-day / day-of-week.
    """
    df = df.copy()
    close = df["Close"]

    df["return_1"] = close.pct_change(1)
    df["return_2"] = close.pct_change(2)
    df["return_4"] = close.pct_change(4)
    df["return_8"] = close.pct_change(8)
    df["return_24"] = close.pct_change(24)

    for lag in [1, 2, 3, 4, 8]:
        df[f"close_lag_{lag}_ratio"] = close / close.shift(lag)

    for window in [6, 12, 24, 48]:
        df[f"rolling_mean_{window}"] = close.rolling(window).mean()
        df[f"rolling_std_{window}"] = close.rolling(window).std()
        df[f"dev_from_ma_{window}"] = (close - df[f"rolling_mean_{window}"]) / df[f"rolling_mean_{window}"]

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["hl_range_rolling_6"] = df["hl_range"].rolling(6).mean()

    df["volume_ma_6"] = df["Volume"].rolling(6).mean()
    df["volume_ma_24"] = df["Volume"].rolling(24).mean()
    df["volume_ratio_6"] = df["Volume"] / df["volume_ma_6"].replace(0, np.nan)
    df["volume_ratio_24"] = df["Volume"] / df["volume_ma_24"].replace(0, np.nan)

    df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]

    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (everything except meta/target)."""
    exclude = {
        "DateTime", "Date", "Time", "Year", "Month", "Day", "Hour",
        "Open", "High", "Low", "Close", "Volume",
        "Price_Change", "Price_Change_Percent",
        "target",
        "rolling_mean_6", "rolling_mean_12", "rolling_mean_24", "rolling_mean_48",
        "volume_ma_6", "volume_ma_24",
    }
    return [c for c in df.columns if c not in exclude]


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Chronological split: first (1-test_frac) rows for training,
    last test_frac rows for testing. No shuffling.
    """
    n = len(df)
    split_idx = int(n * (1 - test_frac))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def build_dataset(horizon: int = HORIZON, test_frac: float = 0.2):
    """
    End-to-end pipeline: load → clean → features → target → split.
    Returns (X_train, y_train, X_test, y_test, feature_cols, df_full).
    """
    df = load_raw_data()
    df = clean_data(df)
    df = add_features(df)
    df = create_target(df, horizon=horizon)

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    train, test = time_split(df, test_frac=test_frac)

    X_train = train[feature_cols].values
    y_train = train["target"].values
    X_test = test[feature_cols].values
    y_test = test["target"].values

    return X_train, y_train, X_test, y_test, feature_cols, df
