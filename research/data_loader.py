"""
Iron Ore CTA Research — Data Loader

Loads I9999.parquet, resamples to various timeframes, tags sessions.
"""

import pandas as pd
import numpy as np
from .config import (
    PARQUET_PATH, TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    SESSION_DAY1, SESSION_DAY2, SESSION_NIGHT,
)


def load_raw(path: str = PARQUET_PATH) -> pd.DataFrame:
    """Load raw 1-min parquet and set datetime index."""
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    # Drop duplicates (some rollovers may have overlapping bars)
    df = df[~df.index.duplicated(keep="first")]
    return df


def tag_session(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'session' column: 'day1', 'day2', 'night', or 'unknown'."""
    hour = df.index.hour
    minute = df.index.minute
    time_val = hour * 100 + minute  # e.g., 930 for 09:30

    conditions = [
        (time_val >= 900) & (time_val <= 1130),
        (time_val >= 1330) & (time_val <= 1500),
        (time_val >= 2100) & (time_val <= 2300),
    ]
    choices = ["day1", "day2", "night"]
    df["session"] = np.select(conditions, choices, default="unknown")
    return df


def tag_trading_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'tday' column — the trading day each bar belongs to.
    Night session bars (21:00-23:00) belong to the NEXT calendar day's trading day.
    """
    date = df.index.date
    hour = df.index.hour
    # Night session belongs to next trading day
    is_night = hour >= 21
    tday = pd.Series(df.index.date, index=df.index)
    tday[is_night] = pd.to_datetime(tday[is_night]) + pd.Timedelta(days=1)
    tday[~is_night] = pd.to_datetime(tday[~is_night])
    df["tday"] = pd.to_datetime(tday).dt.date
    return df


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.

    Args:
        df: DataFrame with OHLCV columns and datetime index.
        freq: Pandas frequency string, e.g., '2min', '3min', '5min', '15min', '30min', '1h'.

    Returns:
        Resampled DataFrame with clean OHLCV + oi columns.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "oi": "last",
    }
    # Only aggregate columns that exist
    agg = {k: v for k, v in agg.items() if k in df.columns}
    resampled = df.resample(freq).agg(agg).dropna(subset=["open"])
    return resampled


def split_train_test(df: pd.DataFrame):
    """Split into train and test sets based on config dates."""
    train = df[TRAIN_START:TRAIN_END]
    test = df[TEST_START:TEST_END]
    return train, test


def load_and_prepare(freq: str = "1min", with_session: bool = True) -> tuple:
    """
    Full pipeline: load → tag → resample → split.

    Args:
        freq: Target timeframe ('1min', '2min', '5min', '15min', '30min', '1h').
        with_session: Whether to tag sessions.

    Returns:
        (train_df, test_df) tuple.
    """
    df = load_raw()

    if with_session:
        df = tag_session(df)
        df = tag_trading_day(df)

    if freq != "1min":
        # Keep session/tday tags by forward-filling after resample
        session_cols = []
        if "session" in df.columns:
            session_cols = ["session", "tday"]
            session_data = df[session_cols]
            df = df.drop(columns=session_cols)

        df = resample(df, freq)

        if session_cols:
            # Re-tag on resampled data
            df = tag_session(df)
            df = tag_trading_day(df)

    return split_train_test(df)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute intraday VWAP (resets each trading day)."""
    if "tday" not in df.columns:
        df = tag_trading_day(df)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).groupby(df["tday"]).cumsum()
    cum_vol = df["volume"].groupby(df["tday"]).cumsum()
    return (cum_tp_vol / cum_vol.replace(0, np.nan)).ffill()
