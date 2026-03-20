"""
Strategy #14 — Intraday Open Price Deviation Reversion (5-min bars)
Core: First-hour mean reversion after opening price dislocation in DCE day sessions
Research: Chinese commodity futures intraday reversal (Pacific-Basin 2024), U-shaped liquidity effect
Targets both day1 (09:00) and day2 (13:30) session opens
Params: dev_threshold [0.003,0.005,0.008], signal_bar [2,3,4], max_bars [6,12]

Research Backing
----------------
1. "Intraday reversal effect in commodity futures" (ScienceDirect 2024, Pacific-Basin Finance
   Journal): Chinese commodity futures (including DCE iron ore) exhibit a statistically
   significant reversal pattern in the first 30-60 minutes after strong opening moves. Positions
   that fade an opening deviation of ≥0.3% showed positive Sharpe ratios in out-of-sample tests.

2. U-shaped intraday liquidity in DCE: Trading volume and volatility peak at the open
   (09:00–09:30 AM) and again just before close. This microstructure effect creates transient
   price dislocations as informed and uninformed order flow collides at the open auction.
   Prices tend to revert once the imbalance resolves (typically within 30–60 minutes).

3. "Trends and Reversion in Financial Markets" (arXiv 2501.16772): Below the 15-minute
   scale, mean reversion dominates over trend-following in liquid commodity futures. The
   crossover scale for iron ore was estimated near 10–20 minutes, supporting a 5-min bar
   entry with a short holding window.

Signal Logic
------------
- Identify the first bar of each day session:
    Day 1: 09:00 AM  (trades until 11:30)
    Day 2: 13:30 PM  (trades until 15:00)
- Record session_open = open price of that first bar.
- After `signal_bar` bars (e.g., 3 × 5 min = 15 minutes in):
    dev_pct = (close - session_open) / session_open
    if dev_pct >  dev_threshold → SHORT (fade the upside move)
    if dev_pct < -dev_threshold → LONG  (fade the downside move)
- Hold for up to `max_bars` bars from session start, or exit when price returns to session_open.
- Signal encoding:
    +1 = long (hold)
     0 = flat / no signal
    -1 = short (hold)
    +2 = force exit (return to session_open or time limit exceeded)

Position Sizing & Risk
----------------------
Fade strategy with tight stops (move against open dislocation is bounded):
    hard_stop_pct   = 0.7%  (hard stop-loss from entry)
    trailing_pct    = 0.5%  (trailing stop from peak)
    tp1_pct         = 0.5%  (first take-profit, reduce half)
    tp2_pct         = 1.0%  (second take-profit, close all)
    max_lots        = 1
"""

import numpy as np
import pandas as pd

from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class OpenReversionStrategy(BaseResearchStrategy):
    """Intraday open-price deviation reversion for DCE iron ore 5-min bars.

    Fades the first significant move away from each day-session open price,
    targeting a return to the session open within a bounded time window.
    Valid for both the morning session (09:00) and afternoon session (13:30).
    """

    name = "S14_OpenReversion"
    freq = "5min"

    # ------------------------------------------------------------------
    # Parameter grid
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """Return the 3-parameter grid yielding 3×3×2 = 18 combinations.

        dev_threshold : float
            Minimum deviation (as a fraction) from session open required to
            trigger a fade signal. E.g., 0.005 means ±0.5%.
        signal_bar : int
            Bar index within the session at which the deviation is evaluated.
            Bar 0 is the first bar (session open bar itself). Bar 3 at 5-min
            resolution corresponds to 15 minutes into the session.
        max_bars : int
            Maximum number of bars (from session start) to hold the position.
            Once exceeded the position is force-exited regardless of P&L.
        """
        return {
            "dev_threshold": [0.003, 0.005, 0.008],
            "signal_bar":    [2, 3, 4],
            "max_bars":      [6, 12],
        }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        dev_threshold: float = 0.005,
        signal_bar: int = 3,
        max_bars: int = 6,
    ) -> np.ndarray:
        """Generate intraday open-reversion signals via a stateful bar loop.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with a DatetimeIndex (tz-aware or tz-naive).
            Required columns: ``open``, ``close``.
        dev_threshold : float
            Fractional deviation from session open to trigger a fade trade.
        signal_bar : int
            Bar number within the session at which the entry check is made.
        max_bars : int
            Maximum bars from session start before a time-based force exit.

        Returns
        -------
        np.ndarray of int8
            Element-wise signal array aligned with ``df``:
            +1  long (hold), -1  short (hold), 0  flat, +2  force exit.
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        # Guard: required columns must exist
        if "close" not in df.columns or "open" not in df.columns:
            return signals

        close = df["close"].values.astype(float)
        open_ = df["open"].values.astype(float)

        # Session membership masks
        hour       = df.index.hour
        minute     = df.index.minute
        time_float = hour + minute / 60.0          # decimal hour, e.g. 9.5 = 09:30

        is_day1 = (time_float >= 9.0) & (time_float < 11.5)   # 09:00 – 11:30
        is_day2 = (time_float >= 13.5) & (time_float < 15.0)  # 13:30 – 15:00
        is_day  = is_day1 | is_day2                             # combined mask (bool array)

        # ---------------------------------------------------------------
        # Stateful loop — required because session detection and bar
        # counting depend on prior observations.
        # ---------------------------------------------------------------
        session_start_idx  = -1        # index of the first bar in the current session
        session_open_price = np.nan    # open price of that first bar
        signal_direction   = 0        # active trade direction: +1, -1, or 0
        target_price       = np.nan   # session open — the exit target for the reversion

        in_day = False                 # whether previous bar was in a day session

        for i in range(n):
            cur_is_day = bool(is_day[i])

            # ----------------------------------------------------------
            # Session boundary detection
            # ----------------------------------------------------------
            if cur_is_day and not in_day:
                # First bar of a new day session
                session_start_idx  = i
                session_open_price = open_[i]
                signal_direction   = 0
                target_price       = session_open_price
                in_day             = True

            elif not cur_is_day:
                # Leaving the day session (night session / break)
                in_day           = False
                signal_direction = 0
                continue  # no signal outside day sessions

            # Defensive NaN guard
            if np.isnan(session_open_price) or np.isnan(close[i]):
                continue

            bar_in_session = i - session_start_idx  # 0-based count from session start

            # ----------------------------------------------------------
            # Force exit: price returned to session open (reversion complete)
            # ----------------------------------------------------------
            if signal_direction != 0 and not np.isnan(target_price):
                if signal_direction == 1 and close[i] >= target_price:
                    signals[i]       = 2   # exit long — reversion achieved
                    signal_direction = 0
                    continue
                elif signal_direction == -1 and close[i] <= target_price:
                    signals[i]       = 2   # exit short — reversion achieved
                    signal_direction = 0
                    continue

            # ----------------------------------------------------------
            # Time-based force exit: exceeded max_bars window
            # ----------------------------------------------------------
            if bar_in_session > max_bars and signal_direction != 0:
                signals[i]       = 2   # time limit exceeded
                signal_direction = 0
                continue

            # ----------------------------------------------------------
            # Entry: evaluate deviation at signal_bar
            # ----------------------------------------------------------
            if bar_in_session == signal_bar and signal_direction == 0:
                if session_open_price == 0.0 or np.isnan(session_open_price):
                    continue  # cannot compute deviation from zero/NaN open

                dev_pct = (close[i] - session_open_price) / session_open_price

                if dev_pct > dev_threshold:
                    # Price ran up too fast — fade the move (go short)
                    signals[i]       = -1
                    signal_direction = -1

                elif dev_pct < -dev_threshold:
                    # Price dropped too fast — fade the move (go long)
                    signals[i]       = 1
                    signal_direction = 1
                # else: deviation insufficient — no trade

            # ----------------------------------------------------------
            # Maintain active signal within the allowed window
            # ----------------------------------------------------------
            elif 0 < bar_in_session <= max_bars and signal_direction != 0:
                signals[i] = signal_direction  # carry the trade forward

            # bar_in_session == 0 (session open bar itself): no entry yet
            # bar_in_session < signal_bar and no active trade: wait

        return signals

    # ------------------------------------------------------------------
    # Position parameters
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """Return risk parameters suited to a short-horizon fade strategy.

        Rationale
        ---------
        - hard_stop_pct=0.7 : If the opening move continues past 0.7% from
          entry, the thesis is invalidated — exit immediately to cap loss.
        - trailing_pct=0.5  : Lock in gains once the position moves 0.5% in
          our favour; prevents giving back a successful reversion.
        - tp1_pct=0.5       : Book half the position at 0.5% profit; this
          secures a base return even if the reversion only partially completes.
        - tp2_pct=1.0       : Full close at 1.0% — captures a complete
          round-trip back to session open from a ≥0.5% opening deviation.
        - max_lots=1        : Conservative sizing for a research/prototype run.
        """
        return PositionParams(
            hard_stop_pct=0.7,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
