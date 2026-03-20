"""
Strategy #13 — Realized Semivariance Asymmetry + Bollinger Band (5-min bars)
Core: RS+ / (RS+ + RS-) asymmetry ratio as regime filter for BB mean-reversion entries
Research: Liu et al. JEF 2023 - semivariance asymmetry on 31 Chinese commodity futures
Params: sv_window [20,30,50], bb_period [15,20,30], asym_threshold [0.50,0.55,0.60,0.65]
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class SemivarianceBB(BaseResearchStrategy):
    """
    Realized Semivariance Asymmetry + Bollinger Band mean-reversion strategy.

    Regime filter: compute the asymmetry ratio of realized upside vs downside
    semivariance over a rolling window of 5-min bars.

      RS_pos = sum(max(ret, 0)^2)  over sv_window bars
      RS_neg = sum(max(-ret, 0)^2) over sv_window bars
      asym   = RS_pos / (RS_pos + RS_neg)   [0.5 when total == 0]

    Entry rules:
      LONG  — price crosses below BB lower band AND asym > asym_threshold
              (upside vol dominates → dip-buying has higher success rate)
      SHORT — price crosses above BB upper band AND asym < (1 - asym_threshold)
              (downside vol dominates → selling rips works better)

    Exit rules:
      Reverse signal OR hard stop.

    Research basis:
      Liu et al. (JEF 2023) "Time series momentum and reversal: Intraday
      information from realized semivariance" — tested on 31 Chinese commodity
      futures including iron ore. When upside RS exceeds downside RS the market
      is in a mean-reversion-favorable state; combining this regime filter with
      Bollinger Band entries produces statistically higher Sharpe than standard BB.
    """

    name = "SemivarianceBB"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "sv_window":      [20, 30, 50],
            "bb_period":      [15, 20, 30],
            "asym_threshold": [0.50, 0.55, 0.60, 0.65],
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        sv_window: int = 20,
        bb_period: int = 20,
        asym_threshold: float = 0.55,
    ) -> np.ndarray:
        """
        Generate entry signals using realized semivariance asymmetry + BB crossover.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at least a 'close' column.
        sv_window : int
            Rolling window (bars) for semivariance calculation.
        bb_period : int
            Bollinger Band lookback period.
        asym_threshold : float
            Minimum asymmetry ratio required to enter a long position.
            Mirror threshold (1 - asym_threshold) used for shorts.

        Returns
        -------
        np.ndarray of np.int8
            +1 for long entry, -1 for short entry, 0 for no signal.
        """
        close = pd.Series(df["close"].values, dtype=np.float64)

        # --- Returns ---
        ret = close.pct_change()

        # --- Rolling realized semivariance ---
        ret_pos = ret.clip(lower=0)
        ret_neg = (-ret).clip(lower=0)
        rs_pos = (ret_pos ** 2).rolling(sv_window).sum()
        rs_neg = (ret_neg ** 2).rolling(sv_window).sum()
        rs_total = rs_pos + rs_neg

        # Asymmetry ratio; fall back to 0.5 (neutral) when total variance is zero
        asym = (rs_pos / rs_total).where(rs_total > 0, 0.5)

        # --- Bollinger Bands (fixed multiplier = 2.0) ---
        mid = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        upper = mid + 2.0 * std
        lower = mid - 2.0 * std

        # --- Crossover detection (previous bar inside band, current bar outside) ---
        prev_close = close.shift(1)
        cross_below = (prev_close >= lower.shift(1)) & (close < lower)
        cross_above = (prev_close <= upper.shift(1)) & (close > upper)

        # --- Apply asymmetry regime filter ---
        long_signal  = cross_below & (asym > asym_threshold)
        short_signal = cross_above & (asym < (1.0 - asym_threshold))

        signals = np.zeros(len(df), dtype=np.int8)
        signals[long_signal.values]  = 1
        signals[short_signal.values] = -1

        # --- NaN safety: zero out any bar where indicators are not yet valid ---
        nan_mask = mid.isna() | std.isna() | asym.isna()
        signals[nan_mask.values] = 0

        return signals

    def position_params(self) -> PositionParams:
        """
        Mean-reversion position sizing and risk parameters.

        hard_stop_pct : 0.7% — wider stop to tolerate BB overshoot noise
        trailing_pct  : 0.5% — lock in gains as price reverts toward mid
        tp1_pct       : 0.5% — first partial profit target (half position)
        tp2_pct       : 1.0% — second profit target / full exit
        max_lots      : 1    — single lot; mean-reversion sizing
        """
        return PositionParams(
            hard_stop_pct=0.7,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
