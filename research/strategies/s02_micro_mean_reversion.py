"""
Strategy #2 — Micro Mean-Reversion (2-min bars)

Iron ore futures mean-reversion strategy using Bollinger Bands
on 2-minute bars. Enters when price crosses beyond the bands
expecting a snap-back toward the mean.

Params (2): bb_period, bb_std — kept coarse for anti-overfitting.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class MicroMeanReversion(BaseResearchStrategy):
    name = "Micro Mean-Reversion"
    freq = "2min"

    def param_grid(self) -> dict:
        return {
            "bb_period": [15, 20, 30, 40],
            "bb_std": [1.5, 2.0, 2.5, 3.0],
        }

    def generate_signals(
        self, df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0
    ) -> np.ndarray:
        close = pd.Series(df["close"].values, dtype=np.float64)

        # Bollinger Bands
        mid = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        upper = mid + bb_std * std
        lower = mid - bb_std * std

        # Detect crosses: previous bar was inside band, current bar is outside
        prev_close = close.shift(1)

        # Long: close crosses below lower band (mean-reversion buy)
        cross_below_lower = (prev_close >= lower.shift(1)) & (close < lower)

        # Short: close crosses above upper band (mean-reversion sell)
        cross_above_upper = (prev_close <= upper.shift(1)) & (close > upper)

        # Build signal array
        signals = np.zeros(len(df), dtype=np.int8)
        signals[cross_below_lower.values] = 1
        signals[cross_above_upper.values] = -1

        # NaN safety — warm-up period and any NaN positions default to 0
        nan_mask = mid.isna().values | std.isna().values
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=0.7, trailing_pct=0.5, tp1_pct=0.5, tp2_pct=1.0)
