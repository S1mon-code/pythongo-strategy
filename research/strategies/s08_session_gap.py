"""
Strategy #8 -- Session-Gap Reversion (5-min bars)

Iron ore futures mean-reversion strategy that fades overnight gaps.
When the day session opens significantly above/below the previous
session's close, we bet on the gap closing within the first 30 minutes.

Params (2): gap_threshold, atr_mult -- kept coarse for anti-overfitting.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class SessionGapReversion(BaseResearchStrategy):
    name = "Session-Gap Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "gap_threshold": [0.003, 0.005, 0.008, 0.01],
            "atr_mult": [1.0, 1.5, 2.0, 2.5],
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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

    @staticmethod
    def _derive_session(idx: pd.DatetimeIndex) -> pd.Series:
        """Derive session labels from bar timestamps.

        hour >= 21        -> 'night'
        9:00 - 11:30      -> 'day1'
        13:30 - 15:00     -> 'day2'
        everything else   -> 'unknown'
        """
        hour = idx.hour
        minute = idx.minute
        time_float = hour + minute / 60.0

        session = pd.Series("unknown", index=idx)
        session[hour >= 21] = "night"
        session[(time_float >= 9.0) & (time_float < 11.5)] = "day1"
        session[(time_float >= 13.5) & (time_float < 15.0)] = "day2"
        return session

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        gap_threshold: float = 0.005,
        atr_mult: float = 1.5,
    ) -> np.ndarray:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        close = df["close"].values.astype(float)
        open_ = df["open"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        # --- session column ---
        if "session" in df.columns:
            session = df["session"].values
        else:
            session = self._derive_session(df.index).values

        # --- ATR (20-bar rolling) ---
        atr = self._compute_atr(high, low, close, 20)

        # --- detect session gaps and generate signals ---
        # Track the last close of the previous session and gap bar info.
        prev_session_close = np.nan
        prev_session_label: str | None = None
        gap_bar_idx: int = -999      # index of the bar that started the gap
        gap_direction: int = 0       # 1 = gap-up (short), -1 = gap-down (long)

        for i in range(n):
            cur_session = session[i]

            # Detect start of a new day session (day1) after a different session
            is_day1 = cur_session == "day1"
            was_different = prev_session_label is not None and prev_session_label != "day1"
            is_first_day1_bar = is_day1 and was_different

            if is_first_day1_bar and not np.isnan(prev_session_close):
                gap_pct = (open_[i] - prev_session_close) / prev_session_close

                if not np.isnan(gap_pct) and not np.isnan(atr[i]):
                    if gap_pct > gap_threshold:
                        # Gap up -> short (fade)
                        gap_bar_idx = i
                        gap_direction = -1
                        signals[i] = -1
                    elif gap_pct < -gap_threshold:
                        # Gap down -> long (fade)
                        gap_bar_idx = i
                        gap_direction = 1
                        signals[i] = 1
                    else:
                        gap_direction = 0
                else:
                    gap_direction = 0

            # Allow signals in the first 30 min window (6 bars of 5-min)
            elif (
                gap_direction != 0
                and 0 < (i - gap_bar_idx) < 6
                and is_day1
            ):
                # Continue the fade signal for the first 30 min
                signals[i] = gap_direction

            # Update previous-session tracking
            if cur_session in ("day1", "day2", "night"):
                if cur_session != prev_session_label and prev_session_label is not None:
                    # Session changed -- we already captured prev_session_close
                    pass
                prev_session_close = close[i]
                prev_session_label = cur_session

        # NaN safety -- zero out wherever ATR is NaN
        nan_mask = np.isnan(atr)
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=1.0, trailing_pct=0.7, tp1_pct=0.6, tp2_pct=1.2, max_lots=2)
