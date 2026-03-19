"""
Strategy #3: Order Flow Imbalance (3-min bars)

Idea: When open interest increases significantly alongside high volume,
institutional players are building positions.  Follow their direction —
bullish bar → long, bearish bar → short.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class OrderFlowImbalance(BaseResearchStrategy):
    name = "Order Flow Imbalance"
    freq = "3min"

    def param_grid(self) -> dict:
        return {
            "oi_threshold": [0.005, 0.01, 0.02, 0.03],
            "vol_ratio": [1.5, 2.0, 2.5, 3.0],
        }

    def generate_signals(
        self, df: pd.DataFrame, oi_threshold: float = 0.01, vol_ratio: float = 2.0
    ) -> np.ndarray:
        signals = np.zeros(len(df), dtype=int)

        # OI change pct (row-over-row)
        oi_chg = df["oi"].pct_change().values

        # Volume ratio vs 20-bar rolling mean
        vol_mean = df["volume"].rolling(20).mean().values
        v_ratio = np.where(vol_mean > 0, df["volume"].values / vol_mean, 0.0)

        # Bar direction
        bullish = (df["close"].values > df["open"].values)
        bearish = (df["close"].values < df["open"].values)

        # Core condition: OI surging + volume spike
        flow = (oi_chg > oi_threshold) & (v_ratio > vol_ratio)

        signals[flow & bullish] = 1
        signals[flow & bearish] = -1

        # Replace any NaN-originated positions with 0
        nan_mask = np.isnan(oi_chg) | np.isnan(v_ratio)
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=0.8, trailing_pct=0.6, tp1_pct=0.5, tp2_pct=1.2)
