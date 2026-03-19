"""
Strategy #1 — Tick Momentum Scalper (1-min bars)

Iron ore futures momentum scalper using price momentum
with volume confirmation on 1-minute bars.

Params (2): mom_bars, threshold — kept coarse for anti-overfitting.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class TickMomentumScalper(BaseResearchStrategy):
    name = "Tick Momentum Scalper"
    freq = "1min"

    def param_grid(self) -> dict:
        return {
            "mom_bars": [3, 5, 8, 12, 15],
            "threshold": [0.002, 0.003, 0.005, 0.008],
        }

    def generate_signals(
        self, df: pd.DataFrame, mom_bars: int = 5, threshold: float = 0.003
    ) -> np.ndarray:
        close = df["close"].values
        volume = df["volume"].values

        # Momentum: pct change over mom_bars periods
        momentum = pd.Series(close).pct_change(periods=mom_bars).values

        # Volume confirmation: current volume > 20-bar rolling mean
        vol_ma = pd.Series(volume).rolling(20).mean().values
        vol_confirmed = volume > vol_ma

        # Signals
        signals = np.zeros(len(df), dtype=np.int8)
        signals[(momentum > threshold) & vol_confirmed] = 1
        signals[(momentum < -threshold) & vol_confirmed] = -1

        # NaN safety — any remaining NaN positions default to 0
        signals[np.isnan(momentum) | np.isnan(vol_ma)] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=0.5, trailing_pct=0.4, tp1_pct=0.4, tp2_pct=0.8)
