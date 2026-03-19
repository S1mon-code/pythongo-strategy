"""
Strategy #5 — KAMA + ATR Trend (15-min bars)

Iron ore futures trend-following strategy using Kaufman Adaptive
Moving Average with ATR-based bands.  Entries trigger on breakouts
above/below the adaptive channel.

Params (2): kama_period, atr_mult — kept coarse for anti-overfitting.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class KamaAtrTrend(BaseResearchStrategy):
    name = "KAMA + ATR Trend"
    freq = "15min"

    def param_grid(self) -> dict:
        return {
            "kama_period": [10, 15, 20, 30],
            "atr_mult": [1.5, 2.0, 2.5, 3.0],
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_kama(close: np.ndarray, period: int) -> np.ndarray:
        """Kaufman Adaptive Moving Average."""
        n = len(close)
        kama = np.full(n, np.nan)

        # smoothing constants
        fast_sc = 2.0 / (2 + 1)     # fast EMA period = 2
        slow_sc = 2.0 / (30 + 1)    # slow EMA period = 30

        # seed KAMA at the end of the first look-back window
        kama[period] = close[period]

        for i in range(period + 1, n):
            direction = abs(close[i] - close[i - period])
            volatility = np.sum(np.abs(np.diff(close[i - period : i + 1])))

            er = direction / volatility if volatility != 0 else 0.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])

        return kama

    @staticmethod
    def _compute_atr(high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, period: int) -> np.ndarray:
        """Average True Range over *period* bars."""
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
        atr = pd.Series(tr).rolling(period).mean().values
        return atr

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        kama_period: int = 15,
        atr_mult: float = 2.0,
    ) -> np.ndarray:
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        kama = self._compute_kama(close, kama_period)
        atr = self._compute_atr(high, low, close, kama_period)

        upper = kama + atr_mult * atr
        lower = kama - atr_mult * atr

        # Cross detection: close crosses above upper / below lower
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        cross_above = (prev_close <= upper) & (close > upper)
        cross_below = (prev_close >= lower) & (close < lower)

        signals = np.zeros(len(df), dtype=np.int8)
        signals[cross_above] = 1
        signals[cross_below] = -1

        # NaN safety — set to 0 wherever any indicator is NaN
        nan_mask = np.isnan(kama) | np.isnan(atr) | np.isnan(prev_close)
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=2.0, trailing_pct=1.2, tp1_pct=1.5, tp2_pct=3.0, max_lots=3)
