"""
Strategy #6: Opening Range Breakout (5-min bars)

Idea: Identify the high/low of the first N minutes of the day session,
then trade breakouts with volume confirmation.  DCE iron ore day session
starts at 09:00.

- opening_min controls how many minutes define the opening range.
- vol_mult requires the breakout bar's volume to exceed the average
  volume of the opening bars by this multiple.
- Only one signal per direction per trading day.
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class OpeningRangeBreakout(BaseResearchStrategy):
    name = "Opening Range Breakout"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "opening_min": [15, 30, 45, 60],
            "vol_mult": [1.2, 1.5, 2.0, 2.5],
        }

    def generate_signals(
        self, df: pd.DataFrame, opening_min: int = 30, vol_mult: float = 1.5
    ) -> np.ndarray:
        signals = np.zeros(len(df), dtype=int)
        n_bars = max(1, opening_min // 5)  # number of opening-range bars

        # Derive trading day from index (date portion)
        # DCE day session starts at 09:00; night session belongs to next tday.
        # Use 'tday' column if available, otherwise derive from index.
        if "tday" in df.columns:
            tday = df["tday"].values
        else:
            idx = df.index
            hour = idx.hour
            # Night session (21:00-23:00, 00:00-02:30) belongs to next calendar day
            dates = idx.date
            tday = np.array(dates, dtype="datetime64[D]")
            # Shift: bars before 09:00 belong to the previous tday assignment
            # but for opening range we only care about day session (>= 09:00)
            # so tday grouping by calendar date is fine for day-session logic.

        # Unique trading days
        unique_days = np.unique(tday)

        for day in unique_days:
            day_mask = tday == day
            day_idx = np.where(day_mask)[0]
            if len(day_idx) == 0:
                continue

            # Filter to day-session bars only (hour >= 9 and hour < 15:15)
            day_session_idx = []
            for i in day_idx:
                h = df.index[i].hour
                m = df.index[i].minute
                # DCE day session: 09:00 - 11:30, 13:30 - 15:00
                if 9 <= h < 16:
                    day_session_idx.append(i)

            if len(day_session_idx) < n_bars + 1:
                continue  # not enough bars to form opening range + trade

            day_session_idx = np.array(day_session_idx)

            # Opening range: first n_bars of the day session
            or_idx = day_session_idx[:n_bars]
            or_high = df["high"].iloc[or_idx].max()
            or_low = df["low"].iloc[or_idx].min()
            or_avg_vol = df["volume"].iloc[or_idx].mean()

            # Skip if opening range values are invalid
            if np.isnan(or_high) or np.isnan(or_low) or np.isnan(or_avg_vol):
                continue
            if or_avg_vol <= 0:
                continue

            # Scan bars after the opening range
            post_or_idx = day_session_idx[n_bars:]
            long_triggered = False
            short_triggered = False

            for i in post_or_idx:
                close_val = df["close"].iat[i]
                vol_val = df["volume"].iat[i]

                if np.isnan(close_val) or np.isnan(vol_val):
                    continue

                vol_ok = vol_val > vol_mult * or_avg_vol

                if not long_triggered and close_val > or_high and vol_ok:
                    signals[i] = 1
                    long_triggered = True

                if not short_triggered and close_val < or_low and vol_ok:
                    signals[i] = -1
                    short_triggered = True

                if long_triggered and short_triggered:
                    break

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=1.0, trailing_pct=0.7, tp1_pct=0.7, tp2_pct=1.5, max_lots=3)
