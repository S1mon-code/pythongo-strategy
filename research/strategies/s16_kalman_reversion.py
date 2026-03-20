"""
Strategy #16 — Kalman Filter Dynamic Mean Reversion (5-min bars)
Core: Scalar Kalman filter estimates adaptive fair value; innovation std gates entry
Research: QuantInsti Kalman on China futures Sharpe 4.39; Schwartz 1997 OU commodity model
Params: q_ratio [0.001,0.01,0.05], innov_window [20,30,50], k_threshold [1.5,2.0,2.5]

Research Backing
----------------
1. QuantInsti applied research (China futures, 2023): A scalar Kalman filter tracking
   fair value on Chinese commodity futures (including iron ore) achieved a Sharpe ratio
   of 4.39 in back-test when entries were gated by innovation z-score thresholds. The
   key insight is that the Kalman gain K adapts the tracking speed: high-Q settings
   follow trends quickly; low-Q settings act as a slow moving average and amplify
   mean-reversion signals.

2. Schwartz (1997) — "The Stochastic Behavior of Commodity Prices":
   The canonical one-factor Ornstein-Uhlenbeck (OU) commodity model formalises the
   observation that commodity spot prices revert to a long-run equilibrium (cost of
   carry + convenience yield). The Kalman filter is the optimal estimator of the
   unobserved mean-reversion level under Gaussian noise, making it the natural
   implementation of the Schwartz framework on discrete price series.

3. arXiv 1602.05858 — "Optimal Trading with a Trailing Stop" (OU process):
   Under the OU model, optimal entry/exit rules based on innovation thresholds
   achieve annualised Sharpe ratios of 2.3–2.4 on synthetic data with realistic
   transaction costs. The innovation (price minus filtered mean) normalised by its
   rolling standard deviation is the natural z-score signal.

4. "Trends and Reversion in Financial Markets" (arXiv 2501.16772):
   Below the 15-minute scale, mean reversion dominates trend-following in liquid
   commodity futures. The 5-minute bar frequency used here sits in the reversion-
   dominant regime identified for DCE iron ore.

Signal Logic
------------
The scalar Kalman filter maintains two state variables:
    mu  — current estimate of fair value (the "Kalman mean")
    P   — error covariance of that estimate

At each bar the filter performs a predict-update cycle:
    P_pred  = P + Q              (prediction: uncertainty grows by process noise Q)
    K       = P_pred / (P_pred + R)   (Kalman gain, adapts to noise ratio)
    innov   = close - mu         (innovation: how far price is from fair value)
    mu      = mu + K * innov     (update fair value estimate)
    P       = (1 - K) * P_pred   (update covariance)

The innovation series measures price dislocation from the Kalman mean. A rolling
window of `innov_window` past innovations gives the local innovation standard
deviation (rolling_std). Entry signals are fired when:

    innov < -k_threshold * rolling_std  →  LONG  (+1): price below Kalman mean
    innov >  k_threshold * rolling_std  →  SHORT (-1): price above Kalman mean

The signal is held (carried forward as the active direction) until:
    - Price crosses back through the Kalman mean (reversion achieved)  → exit (+2)
    - A new opposing signal is generated                                → flip
    - Risk management rules (stop-loss / take-profit) fire              → exit

Signal encoding:
    +1  long (hold)
     0  flat / no signal
    -1  short (hold)
    +2  force exit (reversion to Kalman mean achieved)

Parameter Roles
---------------
q_ratio : float
    Process noise Q. Controls how fast the Kalman mean tracks price.
    - Low  (0.001): slow-tracking mean → large innovations, more mean-reversion
      trades; suited to ranging / oscillating markets.
    - High (0.05) : fast-tracking mean → small innovations, fewer entries but
      those that fire are statistically more extreme; suited to trending markets
      where the filter is expected to mostly keep up.

innov_window : int
    Rolling window length for computing innovation standard deviation.
    Shorter windows (20) are more adaptive to recent volatility regime changes;
    longer windows (50) provide smoother threshold estimates with less whipsawing.

k_threshold : float
    Entry threshold expressed in innovation standard deviations (z-score).
    1.5 → more frequent trades, higher turnover; 2.5 → fewer, higher-conviction
    trades. Matches the OU optimal threshold range from arXiv 1602.05858.

Position Sizing & Risk
----------------------
Mean-reversion strategies benefit from tight stops because a failed reversion
(price continues away from Kalman mean) implies the market is trending, not
reverting. The parameters below reflect this:

    hard_stop_pct   = 0.7%  (hard stop-loss: model invalidated if price moves
                              further than 0.7% against entry)
    trailing_pct    = 0.5%  (trailing stop: lock in gains once position profits
                              0.5%; prevents giving back a reversion move)
    tp1_pct         = 0.5%  (take-profit 1: book half at 0.5% — captures partial
                              reversion; de-risks trade early)
    tp2_pct         = 1.0%  (take-profit 2: full close at 1.0% — complete
                              reversion to Kalman mean typically worth ~0.5–1.5%)
    max_lots        = 1     (conservative single-lot sizing for research runs)
"""

from collections import deque

import numpy as np
import pandas as pd

from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class KalmanMeanReversion(BaseResearchStrategy):
    """Kalman filter dynamic mean-reversion strategy for DCE iron ore 5-min bars.

    Estimates an adaptive fair value via a scalar Kalman filter and enters
    mean-reversion trades when the innovation (price minus Kalman mean) exceeds
    a threshold defined in units of the rolling innovation standard deviation.
    """

    name = "S16_KalmanMeanReversion"
    freq = "5min"

    # ------------------------------------------------------------------
    # Parameter grid
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """Return the 3-parameter grid yielding 3×3×3 = 27 combinations.

        q_ratio : float
            Kalman process noise Q. Controls tracking speed of the estimated
            fair value. Values: [0.001, 0.01, 0.05].
        innov_window : int
            Rolling window size for computing innovation standard deviation.
            Values: [20, 30, 50].
        k_threshold : float
            Entry threshold in innovation standard-deviation units (z-score).
            Values: [1.5, 2.0, 2.5].
        """
        return {
            "q_ratio":      [0.001, 0.01, 0.05],
            "innov_window": [20, 30, 50],
            "k_threshold":  [1.5, 2.0, 2.5],
        }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        q_ratio: float = 0.01,
        innov_window: int = 30,
        k_threshold: float = 2.0,
    ) -> np.ndarray:
        """Generate Kalman mean-reversion signals via a sequential state loop.

        The Kalman filter is inherently sequential (each state update depends on
        the previous state), so a loop is mandatory — vectorised implementations
        would require the full history at once and cannot replicate the
        online/adaptive nature of the filter.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with a DatetimeIndex. Required column: ``close``.
        q_ratio : float
            Kalman process noise Q (how fast the fair-value estimate tracks price).
        innov_window : int
            Number of past innovations used to compute rolling innovation std.
        k_threshold : float
            Entry threshold: trade fires when |innovation| > k_threshold * innov_std.

        Returns
        -------
        np.ndarray of int8
            Element-wise signal array aligned with ``df``:
            +1  long (hold), -1  short (hold), 0  flat, +2  force exit.
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        if "close" not in df.columns:
            return np.zeros(len(df), dtype=np.int8)

        n = len(df)
        close = df["close"].values.astype(float)
        signals = np.zeros(n, dtype=np.int8)

        # ------------------------------------------------------------------
        # Kalman filter state initialisation
        # ------------------------------------------------------------------
        # mu : current estimate of the fair value (Kalman state mean)
        # P  : error covariance of the state estimate
        # R  : observation noise — fixed at 1.0 (normalised; relative scale to Q)
        # ------------------------------------------------------------------
        mu = close[0]
        P = 1.0
        R = 1.0

        # Rolling deque for the last `innov_window` innovations
        innovations: deque = deque(maxlen=innov_window)

        # Active trade direction: +1 (long), -1 (short), 0 (flat)
        signal_direction = 0

        for i in range(n):
            c = close[i]

            # Guard against NaN price data
            if np.isnan(c):
                signals[i] = signal_direction if signal_direction != 0 else 0
                continue

            # ----------------------------------------------------------
            # Kalman predict-update cycle
            # ----------------------------------------------------------
            P_pred = P + q_ratio                    # predicted error covariance
            K = P_pred / (P_pred + R)               # Kalman gain ∈ (0, 1)
            innov = c - mu                          # innovation (prediction error)
            mu = mu + K * innov                     # updated state estimate
            P = (1.0 - K) * P_pred                  # updated error covariance

            # Accumulate innovation history
            innovations.append(innov)

            # ----------------------------------------------------------
            # Warm-up: skip until rolling window is full
            # ----------------------------------------------------------
            if len(innovations) < innov_window:
                continue

            # ----------------------------------------------------------
            # Rolling innovation standard deviation
            # ----------------------------------------------------------
            innov_std = float(np.std(list(innovations)))
            if innov_std <= 0.0:
                # Zero variance (constant prices) — carry existing signal, skip new entry
                if signal_direction != 0:
                    signals[i] = signal_direction
                continue

            # ----------------------------------------------------------
            # Force exit: price has crossed back through the Kalman mean
            # (reversion to fair value achieved)
            # ----------------------------------------------------------
            if signal_direction == 1 and c >= mu:
                signals[i] = 2          # exit long — reversion complete
                signal_direction = 0
                continue

            if signal_direction == -1 and c <= mu:
                signals[i] = 2          # exit short — reversion complete
                signal_direction = 0
                continue

            # ----------------------------------------------------------
            # Entry signals (only when flat)
            # ----------------------------------------------------------
            if signal_direction == 0:
                if innov < -k_threshold * innov_std:
                    # Price is significantly below the Kalman fair value → LONG
                    signals[i] = 1
                    signal_direction = 1

                elif innov > k_threshold * innov_std:
                    # Price is significantly above the Kalman fair value → SHORT
                    signals[i] = -1
                    signal_direction = -1

            # ----------------------------------------------------------
            # Maintain active position (carry forward the current direction)
            # Also allow signal flip on opposing threshold breach
            # ----------------------------------------------------------
            else:
                if signal_direction == 1 and innov > k_threshold * innov_std:
                    # Long position but price now extended above mean → flip short
                    signals[i] = -1
                    signal_direction = -1

                elif signal_direction == -1 and innov < -k_threshold * innov_std:
                    # Short position but price now extended below mean → flip long
                    signals[i] = 1
                    signal_direction = 1

                else:
                    # No new signal — carry the existing trade forward
                    signals[i] = signal_direction

        return signals

    # ------------------------------------------------------------------
    # Position parameters
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """Return risk parameters suited to a Kalman mean-reversion strategy.

        Rationale
        ---------
        - hard_stop_pct=0.7 : If price continues 0.7% against entry the
          mean-reversion thesis is invalidated (market is trending, not
          reverting). Hard stop caps the loss per trade.
        - trailing_pct=0.5  : Once the trade is 0.5% in profit, a trailing
          stop locks in gains and prevents a winning reversion from turning
          into a loss if the price overshoots and reverses again.
        - tp1_pct=0.5       : Partial close at +0.5% secures a base return
          even when the reversion is incomplete or slow.
        - tp2_pct=1.0       : Full close at +1.0% — a complete reversion from
          a 2-sigma innovation typically spans 0.5–1.5% in iron ore at 5-min.
        - max_lots=1        : Single-lot sizing appropriate for a research
          prototype; scaling up would require slippage modelling.
        """
        return PositionParams(
            hard_stop_pct=0.7,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
