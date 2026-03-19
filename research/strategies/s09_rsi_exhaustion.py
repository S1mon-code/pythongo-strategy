"""
Strategy #9 — RSI Exhaustion Reversal (15-min bars)

Iron ore futures strategy that fades RSI extremes when price
confirms via VWAP direction — exhausted selling near support
(long) or exhausted buying near resistance (short).

Params (3): rsi_period, oversold, overbought — kept coarse for robustness.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class RsiExhaustionReversal(BaseResearchStrategy):
    name = "RSI Exhaustion Reversal"
    freq = "15min"

    def param_grid(self) -> dict:
        return {
            "rsi_period": [10, 14, 20, 30],
            "oversold": [20, 25, 30],
            "overbought": [70, 75, 80],
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        rsi_period: int = 14,
        oversold: int = 25,
        overbought: int = 75,
    ) -> np.ndarray:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        close_s = pd.Series(close)

        # ---- RSI (EWM-smoothed Wilder style) ----
        delta = close_s.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(span=rsi_period, adjust=False).mean().values
        avg_loss = loss.ewm(span=rsi_period, adjust=False).mean().values

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

        # ---- VWAP (reset each trading day) ----
        typical_price = (high + low + close) / 3.0

        # Determine day boundaries for VWAP reset
        if "tday" in df.columns:
            day_labels = df["tday"].values
        else:
            day_labels = pd.Series(df.index).dt.date.values

        vwap = np.full(len(df), np.nan)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = None

        for i in range(len(df)):
            cur_day = day_labels[i]
            if cur_day != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = cur_day

            cum_tp_vol += typical_price[i] * volume[i]
            cum_vol += volume[i]

            if cum_vol > 0:
                vwap[i] = cum_tp_vol / cum_vol

        # ---- Entry signals ----
        signals = np.zeros(len(df), dtype=np.int8)

        # Long: RSI oversold AND price above VWAP (exhausted selling, not freefall)
        long_mask = (rsi < oversold) & (close > vwap)
        signals[long_mask] = 1

        # Short: RSI overbought AND price below VWAP (exhausted buying, not melt-up)
        short_mask = (rsi > overbought) & (close < vwap)
        signals[short_mask] = -1

        # ---- Cooldown: 3 bars after any signal ----
        cooldown = 0
        for i in range(len(signals)):
            if cooldown > 0:
                signals[i] = 0
                cooldown -= 1
            elif signals[i] != 0:
                cooldown = 3

        # ---- NaN safety ----
        nan_mask = np.isnan(rsi) | np.isnan(vwap) | np.isnan(close)
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=1.5, trailing_pct=1.0, tp1_pct=1.0, tp2_pct=2.0, max_lots=2)
