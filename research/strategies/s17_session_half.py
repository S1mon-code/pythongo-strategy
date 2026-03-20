"""
Strategy #17 — Intraday Session Half Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: First-half session move (40-80 min) predicts second-half reversal

  Research: Pacific-Basin Finance Journal 2024 confirms significant intraday
  reversal in Chinese commodity futures: the first-half session move is a
  reliable predictor of second-half mean reversion. Effect is strongest for
  DCE liquid contracts (iron ore qualifies). arXiv 2501.16772 further shows
  that sub-30-min price dynamics are dominated by mean reversion rather than
  momentum.

  DCE Session Windows:
  - Day1 session: 09:00 - 11:30  (150 minutes = 30 bars at 5-min)
  - Day2 session: 13:30 - 15:00  (90 minutes = 18 bars at 5-min)

  Signal Logic:
  - At bar `half_bar` from each session start, compute first-half return:
      ret = (close[half_bar] - session_open) / session_open
  - If ret > +threshold  → SHORT  (first half surged, fade the move)
  - If ret < -threshold  → LONG   (first half dropped, fade the move)
  - Target: price returns to session_open (full reversion)
  - Exit 1: close crosses back to session_open (target_price)
  - Exit 2: time limit — bar_in_session > half_bar + max_bars  (signal = 2)
  - Exit 3: forced at session boundary (last bar before gap, signal = 2)

  Session Detection (time_float = hour + minute/60.0):
  - is_day1 : [9.0, 11.5)
  - is_day2 : [13.5, 15.0)

  Loop-based implementation is required because each session has its own
  open price and the entry bar is session-relative (not absolute index).
  A vectorised approach would require pre-computing session labels and
  session-relative bar counters — the loop is simpler and correct.

  参数设计 (3个，18组合):
  - half_bar  : [8, 12, 16]          — session bar to evaluate (8=40min, 12=60min, 16=80min)
  - threshold : [0.002, 0.004, 0.006] — first-half return magnitude to trigger fade
  - max_bars  : [6, 12]              — max bars to hold the second-half fade trade

  关键设计决策:
  - session_open is the open price of the FIRST bar of each session
  - target_price is always session_open (symmetric reversion target)
  - signal = +1 / -1 maintained for every bar inside the hold window so that
    the backtest engine knows the trade is still open
  - signal = 2 (force close) is written on the last bar where we want to exit
  - Force-exit on session boundary uses i-1 (last bar still inside session)
    to avoid writing signal on a non-session bar
  - NaN/zero session_open guards prevent division errors in sparse data

  适用环境: 日内交易、震荡收敛行情
  风险提示: 如果上午行情单边趋势延续至下午，逆势信号会持续亏损；
            half_bar=16 在 day2 (仅18根) 时入场窗口极窄，需注意样本量
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class SessionHalfReversion(BaseResearchStrategy):
    name = "Session Half Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "half_bar":  [8, 12, 16],
            "threshold": [0.002, 0.004, 0.006],
            "max_bars":  [6, 12],
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        half_bar: int = 12,
        threshold: float = 0.004,
        max_bars: int = 6,
    ) -> np.ndarray:
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        close = df["close"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)

        # ------------------------------------------------------------------
        # Session detection — time_float = hour + minute / 60.0
        # ------------------------------------------------------------------
        hour = df.index.hour
        minute = df.index.minute
        time_float = hour + minute / 60.0

        is_day1 = (time_float >= 9.0) & (time_float < 11.5)
        is_day2 = (time_float >= 13.5) & (time_float < 15.0)
        is_day = is_day1 | is_day2

        # ------------------------------------------------------------------
        # Loop-based session state tracking
        # ------------------------------------------------------------------
        session_start_idx: int = -1
        session_open_price: float = np.nan
        signal_direction: int = 0   # +1 long, -1 short, 0 flat
        target_price: float = np.nan
        in_day: bool = False

        for i in range(n):
            cur_is_day = bool(is_day[i])

            # ---- Detect session transition --------------------------------
            if cur_is_day and not in_day:
                # New session begins at bar i
                session_start_idx = i
                session_open_price = open_[i]
                signal_direction = 0
                target_price = session_open_price
                in_day = True

            elif not cur_is_day:
                # Session just ended — force exit on last in-session bar
                if in_day and signal_direction != 0 and i > 0:
                    signals[i - 1] = 2  # i-1 is the last bar still in session
                in_day = False
                signal_direction = 0
                continue

            # ---- Guard: unusable session open ----------------------------
            if np.isnan(session_open_price) or session_open_price == 0.0:
                continue

            bar_in_session = i - session_start_idx

            # ---- Reversion target hit — exit -----------------------------
            if signal_direction != 0 and not np.isnan(target_price):
                if signal_direction == 1 and close[i] >= target_price:
                    signals[i] = 2
                    signal_direction = 0
                    continue
                elif signal_direction == -1 and close[i] <= target_price:
                    signals[i] = 2
                    signal_direction = 0
                    continue

            # ---- Time-limit exit -----------------------------------------
            if signal_direction != 0 and bar_in_session > half_bar + max_bars:
                signals[i] = 2
                signal_direction = 0
                continue

            # ---- Entry: at half_bar from session start -------------------
            if bar_in_session == half_bar and signal_direction == 0:
                ret = (close[i] - session_open_price) / session_open_price
                if ret > threshold:
                    signals[i] = -1          # fade the surge
                    signal_direction = -1
                elif ret < -threshold:
                    signals[i] = 1           # fade the drop
                    signal_direction = 1
                # target_price remains session_open (already set at session start)

            # ---- Maintain signal within hold window ----------------------
            elif half_bar < bar_in_session <= half_bar + max_bars and signal_direction != 0:
                signals[i] = signal_direction

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.4,
            tp1_pct=0.4,
            tp2_pct=0.8,
            max_lots=1,
        )
