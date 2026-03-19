"""
Strategy #7 — Vol Regime + Donchian Channel (15-min bars)

Iron ore futures strategy that adapts between trend-following and
mean-reversion based on the current volatility regime.

High vol → Donchian breakout (trend-following)
Low vol  → Donchian reversion (fade the extremes)

Params (2): donch_period, vol_ratio — kept coarse for anti-overfitting.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class VolRegimeDonchian(BaseResearchStrategy):
    name = "Vol Regime + Donchian"
    freq = "15min"

    def param_grid(self) -> dict:
        return {
            "donch_period": [20, 30, 50, 80],
            "vol_ratio": [0.8, 1.0, 1.2, 1.5],
        }

    def generate_signals(
        self, df: pd.DataFrame, donch_period: int = 30, vol_ratio: float = 1.0
    ) -> np.ndarray:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)

        # --- Donchian Channel (shifted by 1 to exclude current bar) ---
        upper = high_s.rolling(donch_period).max().shift(1).values
        lower = low_s.rolling(donch_period).min().shift(1).values

        # --- Volatility regime detection ---
        returns = close_s.pct_change().values
        returns_s = pd.Series(returns)

        current_vol = returns_s.rolling(donch_period).std().values
        long_term_vol = returns_s.rolling(donch_period * 4).std().values

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_regime = np.where(long_term_vol > 0, current_vol / long_term_vol, np.nan)

        # --- Entry logic ---
        signals = np.zeros(len(df), dtype=np.int8)

        # High vol regime: Donchian breakout (trend-following)
        high_vol = vol_regime > vol_ratio
        signals[(high_vol) & (close > upper)] = 1
        signals[(high_vol) & (close < lower)] = -1

        # Low vol regime: Donchian reversion (fade extremes)
        low_vol = vol_regime < (1.0 / vol_ratio)
        signals[(low_vol) & (close < lower)] = 1   # buy the dip
        signals[(low_vol) & (close > upper)] = -1   # sell the top

        # NaN safety — any position where indicators are undefined → 0
        nan_mask = (
            np.isnan(upper)
            | np.isnan(lower)
            | np.isnan(vol_regime)
            | np.isnan(close)
        )
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=2.0, trailing_pct=1.2, tp1_pct=1.5, tp2_pct=3.0, max_lots=3)
