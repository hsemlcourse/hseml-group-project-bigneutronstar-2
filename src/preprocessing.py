# Обработка данных: загрузка, очистка, признаки, таргет.
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HORIZON = 1


def load_raw_data(filename: str = "gold_data_1h_cleaned.csv") -> pd.DataFrame:
    """Загрузка данных."""
    path = RAW_DIR / filename
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка данных."""
    df = df.drop_duplicates(subset="DateTime", keep="first")
    df = df[df["Volume"] > 0].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def create_target(df: pd.DataFrame, horizon: int = HORIZON, threshold: float = 0.001) -> pd.DataFrame:
    """
    Создание таргета:
      2 (up) if future_return > threshold
      0 (down) if future_return < -threshold
      1 (flat) otherwise
    Drops last `horizon` rows where future is unknown.
    Also keeps `future_return` for backtesting.
    """
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    valid_mask = future_close.notna()
    df = df[valid_mask].copy()
    future_close = future_close[valid_mask]
    df["future_return"] = (future_close / df["Close"]) - 1
    conditions = [
        df["future_return"] > threshold,
        df["future_return"] < -threshold
    ]
    choices = [2, 0]
    df["target"] = np.select(conditions, choices, default=1)
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Расчет RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Расчет MACD."""
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерация признаков.
    Groups: price lags, return lags, rolling statistics,
    high-low range, volume features, deviation from moving averages,
    technical indicators (RSI, MACD, Bollinger Bands),
    hour-of-day / day-of-week.
    """
    df = df.copy()
    close = df["Close"]

    df["return_1"] = close.pct_change(1)
    df["return_2"] = close.pct_change(2)
    df["return_4"] = close.pct_change(4)
    df["return_8"] = close.pct_change(8)
    df["return_24"] = close.pct_change(24)
    df["return_1_diff"] = df["return_1"].diff()

    pos_ret = (df["return_1"] > 0).astype(int)
    df["pos_bars_last_4"] = pos_ret.rolling(4).sum()

    tr1 = df["High"] - df["Low"]
    df["body_to_range"] = (df["Close"] - df["Open"]).abs() / tr1.replace(0, np.nan)
    tr2 = (df["High"] - close.shift(1)).abs()
    tr3 = (df["Low"] - close.shift(1)).abs()
    df["tr"] = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean() / close

    for lag in [1, 2, 3, 4, 8]:
        df[f"close_lag_{lag}_ratio"] = close / close.shift(lag)

    for window in [6, 12, 24, 48]:
        df[f"rolling_mean_{window}"] = close.rolling(window).mean()
        df[f"rolling_std_{window}"] = close.rolling(window).std()
        df[f"dev_from_ma_{window}"] = (close - df[f"rolling_mean_{window}"]) / df[f"rolling_mean_{window}"]
        if window in [12, 24]:
            df[f"zscore_price_{window}"] = (close - df[f"rolling_mean_{window}"]) / df[f"rolling_std_{window}"].replace(0, np.nan)
            vol_mean = df["Volume"].rolling(window).mean()
            vol_std = df["Volume"].rolling(window).std()
            df[f"zscore_vol_{window}"] = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    df["hl_range"] = tr1 / close
    df["hl_range_rolling_6"] = df["hl_range"].rolling(6).mean()

    df["volume_ma_6"] = df["Volume"].rolling(6).mean()
    df["volume_ma_24"] = df["Volume"].rolling(24).mean()
    df["volume_ratio_6"] = df["Volume"] / df["volume_ma_6"].replace(0, np.nan)
    df["volume_ratio_24"] = df["Volume"] / df["volume_ma_24"].replace(0, np.nan)

    df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]

    df["rsi_14"] = _compute_rsi(close, period=14)
    df["rsi_6"] = _compute_rsi(close, period=6)

    macd_line, signal_line, macd_hist = _compute_macd(close)
    df["macd"] = macd_line / close
    df["macd_signal"] = signal_line / close
    df["macd_hist"] = macd_hist / close

    for window in [12, 24]:
        bb_mean = close.rolling(window).mean()
        bb_std = close.rolling(window).std()
        df[f"bb_upper_{window}"] = (bb_mean + 2 * bb_std - close) / close
        df[f"bb_lower_{window}"] = (close - (bb_mean - 2 * bb_std)) / close
        df[f"bb_width_{window}"] = (4 * bb_std) / close
        df[f"bb_position_{window}"] = (close - (bb_mean - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)

    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek

    return df


def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build external factor features for each of the 5 macro/cross-market series:
    dxy, vix, tnx, silver, oil.

    For each series (where it makes sense) we add:
      - return_1d:     1-day log return (prev daily value)
      - return_5d:     5-day log return
      - return_20d:    20-day log return
      - zscore_20d:    z-score of the price over a 20-day rolling window
      - dev_from_ma20: deviation from 20-day moving average (normalized)

    These are computed on the already-merged (hourly-granularity) columns,
    which carry the last known daily close via merge_asof.  Since multiple
    hourly rows share the same daily value, we compute the rolling statistics
    on the deduplicated daily series and then re-merge, so the windows count
    actual trading days, not hours.

    Additionally we add cross-market (inter-market) features:
      - gold_minus_dxy_return:   gold return_1h minus dxy daily return
      - gold_minus_silver_return: gold return_1h minus silver daily return
      - gold_silver_ratio:       gold Close / silver price
      - vix_zscore_20d:          20-day z-score of VIX (convenience alias)
      - tnx_change_1d:           1-day change in TNX yield (daily)
    """
    df = df.copy()
    external_names = ["dxy", "vix", "tnx", "silver", "oil"]
    present = [n for n in external_names if n in df.columns]

    if not present:
        return df

    for name in present:
        col = df[name]
        # Daily returns (shift(1) avoids lookahead; each row already has yesterday's close)
        df[f"{name}_ret1d"]      = col.pct_change(1)
        df[f"{name}_ret5d"]      = col.pct_change(5)
        df[f"{name}_ret20d"]     = col.pct_change(20)
        # Rolling stats
        ma20 = col.rolling(20, min_periods=5).mean()
        std20 = col.rolling(20, min_periods=5).std()
        df[f"{name}_zscore20"]   = (col - ma20) / std20.replace(0, np.nan)
        df[f"{name}_dev_ma20"]   = (col - ma20) / ma20.replace(0, np.nan)

    # Cross-market / inter-market features
    gold_ret = df["return_1"]   # hourly gold return, already computed in add_features

    if "dxy" in df.columns:
        df["gold_minus_dxy_ret"]  = gold_ret - df["dxy_ret1d"]

    if "silver" in df.columns:
        df["gold_minus_silver_ret"] = gold_ret - df["silver_ret1d"]
        df["gold_silver_ratio"]   = df["Close"] / df["silver"].replace(0, np.nan)

    if "vix" in df.columns:
        # vix_zscore20 already computed above; expose as explicit cross-market feature
        df["vix_level"] = df["vix"]   # raw VIX level is itself informative

    if "tnx" in df.columns:
        df["tnx_change1d"] = df["tnx"].diff(1)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (everything except meta/target)."""
    exclude = {
        "DateTime", "Date", "Time", "Year", "Month", "Day", "Hour",
        "Open", "High", "Low", "Close", "Volume",
        "Price_Change", "Price_Change_Percent",
        "target", "future_return", "tr",
        "rolling_mean_6", "rolling_mean_12", "rolling_mean_24", "rolling_mean_48",
        "volume_ma_6", "volume_ma_24",
        # raw external columns (we use derived features instead)
        "dxy", "vix", "tnx", "silver", "oil",
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


def walk_forward_split(df: pd.DataFrame, n_splits: int = 5, test_size: int = None):
    """
    Walk-forward (expanding window) cross-validation splits.
    Returns list of (train_idx, test_idx) tuples.
    Each subsequent fold uses a larger training set.
    """
    n = len(df)
    if test_size is None:
        test_size = n // (n_splits + 1)
    min_train_size = n - n_splits * test_size
    if min_train_size < test_size:
        min_train_size = test_size
    splits = []
    for i in range(n_splits):
        test_end = n - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start
        if train_end < min_train_size:
            continue
        splits.append((list(range(0, train_end)), list(range(test_start, test_end))))
    return splits


def prepare_features(df: pd.DataFrame):
    """
    Apply feature engineering, drop NaN rows, return cleaned df and feature column names.
    Does NOT create target - call create_target separately before or after.
    """
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


def load_external_factors() -> pd.DataFrame:
    """Load combined external factors CSV."""
    path = RAW_DIR / "external" / "external_factors_combined.csv"
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def _merge_external(gold_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily external factors into hourly gold data without lookahead leakage.
    Uses merge_asof with direction='backward': each hourly row gets the most recent
    daily value that is strictly before or at that hour's timestamp.
    Then forward-fills any remaining gaps.
    """
    gold_sorted = gold_df.sort_values("DateTime").copy()
    ext_sorted = external_df.sort_values("DateTime").copy()
    merged = pd.merge_asof(
        gold_sorted, ext_sorted,
        on="DateTime", direction="backward"
    )
    ext_cols = [c for c in external_df.columns if c != "DateTime"]
    merged[ext_cols] = merged[ext_cols].ffill()
    return merged


def build_dataset(horizon: int = HORIZON, threshold: float = 0.001,
                  test_frac: float = 0.2, use_external: bool = True):
    """
    End-to-end pipeline: load → clean → (optional external merge) → features → target → split.
    Returns (X_train, y_train, X_test, y_test, feature_cols, df_full, future_rets_test).
    Set use_external=False to build the gold-only baseline dataset.
    """
    df = load_raw_data()
    df = clean_data(df)

    if use_external:
        external_df = load_external_factors()
        df = _merge_external(df, external_df)

    df = add_features(df)

    if use_external:
        df = add_external_features(df)

    df = create_target(df, horizon=horizon, threshold=threshold)
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    train, test = time_split(df, test_frac=test_frac)
    X_train = train[feature_cols].values
    y_train = train["target"].values
    X_test = test[feature_cols].values
    y_test = test["target"].values
    future_rets_test = test["future_return"]
    return X_train, y_train, X_test, y_test, feature_cols, df, future_rets_test


def build_full_df(horizon: int = HORIZON, threshold: float = 0.001,
                  use_external: bool = True):
    """
    Load → clean → (optional external) → features → target → drop NaN.
    Returns (df, feature_cols) without splitting.
    """
    df = load_raw_data()
    df = clean_data(df)

    if use_external:
        external_df = load_external_factors()
        df = _merge_external(df, external_df)

    df = add_features(df)

    if use_external:
        df = add_external_features(df)

    df = create_target(df, horizon=horizon, threshold=threshold)
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols
