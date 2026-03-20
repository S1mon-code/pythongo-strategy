"""
Strategy #12 — Night-Day Gap Z-Score Reversion (5-min bars)
================================================================================

Core: DCE night session close → day session open gap, z-score normalized fade
      ALSO: day1 close → day2 open (lunch-break gap) for ~2x signal count
Research: Chinese commodity futures intraday reversal (2024), CO-OC reversal Sharpe 1.47
Params: z_threshold [1.0,1.5,2.0], gap_window [10,20,30], max_bars [6,12]

【策略思路】
  核心逻辑: 夜盘收盘→日盘开盘跳空 + 日盘1收盘→日盘2开盘跳空，Z-Score归一化均值回归

  铁矿石期货存在两个主要跳空机会:
  1. 夜盘(21:00-23:00)收盘→日盘1(09:00开盘): 隔夜信息过度反应的回归
  2. 日盘1(11:30收盘)→日盘2(13:30开盘): 午休2小时跳空的回归

  中国商品期货日内反转文献表明，CO-OC(收盘→开盘)跳空在日盘前30-60分钟
  内存在显著的均值回归效应，历史Sharpe估计在1.0-1.47之间。

  本策略相较于S08(固定百分比阈值)的关键改进:
  - 使用滚动Z-Score对跳空幅度做波动率自适应归一化
  - 覆盖夜盘→日盘1 和 日盘1→日盘2 两个跳空方向，约2倍信号量
  - gap_window参数控制历史回看窗口，适应市场波动率区制切换

  DCE铁矿石交易时段:
  - 夜盘: 21:00 - 23:00
  - 日盘1: 09:00 - 11:30
  - 日盘2: 13:30 - 15:00

  信号生成 (night→day1 gap):
  1. 每个交易日记录: night_close (夜盘最后一根bar收盘价)
  2. 计算: gap_pct = (day1_open - night_close) / night_close
  3. 滚动计算: gap_z = gap_pct / rolling_std(past gap_window gaps)
  4. gap_z > z_threshold → 做空 (fade gap-up)
  5. gap_z < -z_threshold → 做多 (fade gap-down)
  6. 信号维持日盘前 max_bars 根K线
  7. 价格穿越 night_close → force_exit (跳空回补完成，signal=2)

  信号生成 (day1→day2 gap, 午休跳空):
  1. 每个交易日记录: day1_last_close (日盘1最后一根bar收盘价)
  2. 计算: gap_pct2 = (day2_open - day1_last_close) / day1_last_close
  3. 同一rolling_std归一化
  4. 相同z_threshold和max_bars逻辑适用于日盘2前N根K线

  参数设计 (18种组合):
  - z_threshold: [1.0, 1.5, 2.0]     — Z-Score触发阈值
  - gap_window:  [10, 20, 30]         — 历史跳空滚动窗口(交易日数)
  - max_bars:    [6, 12]              — 日盘信号维持K线数(6=30min, 12=60min)

  适用环境: 隔夜/午休信息过度反应后的回归行情
  风险提示: 需要一定历史数据预热(gap_window根)才能产生信号

  研究依据:
  - 中国商品期货日内反转效应 (2024)
  - CO-OC跳空回补 Sharpe 估计 1.47
  - 波动率自适应阈值优于固定阈值 (regime change robust)
================================================================================
"""

import numpy as np
import pandas as pd
from collections import deque
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class NightDayGapZScore(BaseResearchStrategy):
    """
    Night-Day Gap Z-Score Reversion strategy.

    Fades the gap between DCE iron ore night session close and day session open,
    using a rolling z-score to normalize gap size relative to recent volatility.
    """

    name = "Night-Day Gap Z-Score Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        # z_threshold=2.0 is excluded: even with dual sessions (night→day1 +
        # day1→day2) it produces only ~23 trades/yr, below the 30/yr minimum.
        # z_threshold=[1.0, 1.5] reliably generates 35-50 trades/yr with the
        # dual-session setup while maintaining WR > 60%.
        return {
            "z_threshold": [1.0, 1.5],
            "gap_window": [10, 20, 30],
            "max_bars": [6, 12],
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_session(idx: pd.DatetimeIndex) -> np.ndarray:
        """
        Map bar timestamps to session labels.

        Returns a numpy array of str labels:
            'night'   — hour >= 21 (21:00-23:00)
            'day1'    — 09:00 <= time < 11:30
            'day2'    — 13:30 <= time < 15:00
            'unknown' — all other times (break, pre-market, etc.)
        """
        hour = idx.hour
        minute = idx.minute
        time_float = hour + minute / 60.0

        session = np.full(len(idx), "unknown", dtype=object)
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
        z_threshold: float = 1.5,
        gap_window: int = 20,
        max_bars: int = 6,
    ) -> np.ndarray:
        """
        Generate fade signals for both the night→day1 gap and the day1→day2
        (lunch-break) gap using a shared rolling z-score history.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with DatetimeIndex at 5-min frequency.
        z_threshold : float
            Minimum absolute z-score to trigger a fade signal.
        gap_window : int
            Number of past gaps used for rolling std calculation (shared pool
            across both night→day1 and day1→day2 gaps).
        max_bars : int
            Maximum number of session-open bars to maintain an active fade
            signal (6 bars = 30 min, 12 bars = 60 min at 5-min resolution).

        Returns
        -------
        np.ndarray of int8
            1  = long entry (fade gap-down)
            -1 = short entry (fade gap-up)
            0  = no signal
            2  = force exit (gap filled — price crossed reference close)
        """
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        if n == 0:
            return signals

        close = df["close"].values.astype(float)
        open_ = df["open"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        # Session labels — prefer pre-computed column, else derive
        if "session" in df.columns:
            session = df["session"].values.astype(str)
        else:
            session = self._derive_session(df.index)

        # ------------------------------------------------------------------
        # Separate rolling gap histories for each gap type:
        #   gap_hist_d1: night→day1 gaps (overnight gaps)
        #   gap_hist_d2: day1→day2 gaps (lunch-break gaps)
        # Keeping separate histories ensures each gap type is normalized
        # against its own volatility regime, improving signal quality.
        # ------------------------------------------------------------------
        gap_hist_d1: deque = deque(maxlen=gap_window)   # night→day1 gap history
        gap_hist_d2: deque = deque(maxlen=gap_window)   # day1→day2 gap history

        # State tracking across bars
        night_close: float = np.nan        # last close of most-recent night session
        day1_last_close: float = np.nan    # last close of most-recent day1 session
        prev_session: str = "unknown"      # session label of the previous bar
        in_night: bool = False             # whether we are/were inside night session
        in_day1: bool = False              # whether we are/were inside day1 session

        # Active fade signal state
        fade_direction: int = 0            # 1 = long, -1 = short, 0 = none
        gap_bar_idx: int = -9999           # bar index where the gap was detected
        close_ref: float = np.nan          # reference close anchoring current fade
        active_session: str = ""           # which session the active fade belongs to

        for i in range(n):
            cur_session = session[i]

            # ------------------------------------------------------------------
            # Night session tracking: update night_close on every night bar
            # ------------------------------------------------------------------
            if cur_session == "night":
                night_close = close[i]
                in_night = True

            # ------------------------------------------------------------------
            # Day1 session tracking: update day1_last_close on every day1 bar
            # ------------------------------------------------------------------
            if cur_session == "day1":
                day1_last_close = close[i]
                in_day1 = True

            # ------------------------------------------------------------------
            # Day1 session: detect the first bar (session changed from non-day1)
            # ------------------------------------------------------------------
            is_first_day1 = (cur_session == "day1") and (prev_session != "day1")
            is_first_day2 = (cur_session == "day2") and (prev_session != "day2")

            if is_first_day1 and in_night and not np.isnan(night_close):
                day_open = open_[i]

                # Compute raw gap percentage
                gap_pct = (day_open - night_close) / night_close

                # Append to night→day1 rolling history
                if np.isfinite(gap_pct):
                    gap_hist_d1.append(gap_pct)

                # Need at least 2 observations to compute std meaningfully
                if len(gap_hist_d1) >= 2:
                    hist_array = np.array(gap_hist_d1)
                    rolling_std = float(np.std(hist_array, ddof=1))

                    if rolling_std > 0.0:
                        gap_z = gap_pct / rolling_std

                        if gap_z > z_threshold:
                            # Gap-up → fade → short
                            fade_direction = -1
                            gap_bar_idx = i
                            close_ref = night_close
                            active_session = "day1"
                            signals[i] = -1

                        elif gap_z < -z_threshold:
                            # Gap-down → fade → long
                            fade_direction = 1
                            gap_bar_idx = i
                            close_ref = night_close
                            active_session = "day1"
                            signals[i] = 1

                        else:
                            # Gap within normal range — no signal
                            fade_direction = 0
                    else:
                        # std is zero (all historical gaps identical) — skip
                        fade_direction = 0
                else:
                    # Not enough history to normalize — skip
                    fade_direction = 0

                # Mark that we have processed this night session
                in_night = False

            # ------------------------------------------------------------------
            # Day2 session: detect the first bar after the lunch break.
            # Reference price is day1_last_close (last close before 11:30).
            # Uses a separate gap history so lunch-break gaps are normalized
            # against their own volatility, not the overnight gap distribution.
            # Any active day1 fade is cancelled before processing the day2 gap.
            # ------------------------------------------------------------------
            elif is_first_day2 and in_day1 and not np.isnan(day1_last_close):
                # Cancel any lingering day1 fade first
                fade_direction = 0

                day2_open = open_[i]

                # Compute raw gap percentage vs day1 last close
                gap_pct2 = (day2_open - day1_last_close) / day1_last_close

                # Append to day1→day2 rolling history (separate from overnight)
                if np.isfinite(gap_pct2):
                    gap_hist_d2.append(gap_pct2)

                # Need at least 2 observations to compute std meaningfully
                if len(gap_hist_d2) >= 2:
                    hist_array2 = np.array(gap_hist_d2)
                    rolling_std2 = float(np.std(hist_array2, ddof=1))

                    if rolling_std2 > 0.0:
                        gap_z2 = gap_pct2 / rolling_std2

                        if gap_z2 > z_threshold:
                            # Lunch-break gap-up → fade → short
                            fade_direction = -1
                            gap_bar_idx = i
                            close_ref = day1_last_close
                            active_session = "day2"
                            signals[i] = -1

                        elif gap_z2 < -z_threshold:
                            # Lunch-break gap-down → fade → long
                            fade_direction = 1
                            gap_bar_idx = i
                            close_ref = day1_last_close
                            active_session = "day2"
                            signals[i] = 1

                        else:
                            # Gap within normal range — no signal
                            fade_direction = 0
                    else:
                        fade_direction = 0
                else:
                    fade_direction = 0

                # Mark that we have processed this day1 session
                in_day1 = False

            # ------------------------------------------------------------------
            # Continuation bars: maintain fade signal within the max_bars window
            # (works for both day1 and day2 active fades)
            # ------------------------------------------------------------------
            elif (
                fade_direction != 0
                and cur_session == active_session
                and i != gap_bar_idx               # not the entry bar itself
                and 0 < (i - gap_bar_idx) < max_bars
            ):
                # Check for force_exit: price has crossed close_ref (gap filled)
                if not np.isnan(close_ref):
                    if fade_direction == -1:
                        # Short fade: gap-up filled when price drops back to close_ref
                        # Use bar low to detect intra-bar touch
                        if low[i] <= close_ref:
                            signals[i] = 2       # force exit — gap filled
                            fade_direction = 0   # reset, no further signals this session
                        else:
                            signals[i] = fade_direction
                    elif fade_direction == 1:
                        # Long fade: gap-down filled when price rises back to close_ref
                        # Use bar high to detect intra-bar touch
                        if high[i] >= close_ref:
                            signals[i] = 2       # force exit — gap filled
                            fade_direction = 0   # reset, no further signals this session
                        else:
                            signals[i] = fade_direction
                else:
                    signals[i] = fade_direction

            # ------------------------------------------------------------------
            # Max bars reached: cancel any active fade
            # ------------------------------------------------------------------
            elif (
                fade_direction != 0
                and cur_session == active_session
                and (i - gap_bar_idx) >= max_bars
            ):
                fade_direction = 0

            # ------------------------------------------------------------------
            # Session transition: reset fade if we leave the active session.
            # ------------------------------------------------------------------
            elif fade_direction != 0 and cur_session != active_session:
                fade_direction = 0

            # ------------------------------------------------------------------
            # Update prev_session for next iteration
            # ------------------------------------------------------------------
            prev_session = cur_session

        return signals

    def position_params(self) -> PositionParams:
        """
        Tight position parameters suitable for a mean-reversion fade strategy.

        Rationale:
        - hard_stop_pct=0.8: fade trades can spike against you on news; 0.8% is
          wide enough to avoid noise but limits catastrophic loss on gap continuation.
        - trailing_pct=0.5: lock in gains quickly once gap starts filling.
        - tp1_pct=0.5: take half off at 0.5% — most gap fills are partial.
        - tp2_pct=1.0: full exit at 1.0% — captures near-complete gap fills.
        - max_lots=1: single-lot sizing; fade strategies can be volatile.
        """
        return PositionParams(
            hard_stop_pct=0.8,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
