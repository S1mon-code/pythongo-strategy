"""
Strategy #10 — VWAP Z-Score Reversion (5-min bars) ⭐ WINNER #2
================================================================================

【策略思路】
  核心逻辑: 价格偏离VWAP的Z-Score → 均值回归

  VWAP (成交量加权平均价) 是日内的"公允价格"。当价格显著偏离VWAP
  时，意味着市场短期过度反应。用Z-Score标准化偏离程度: Z>3表示价格
  高于VWAP超过3个标准差，统计上极不可能持续，回归概率大。

  VWAP计算 (每日重置):
  - typical_price = (high + low + close) / 3
  - VWAP = cumsum(typical_price × volume) / cumsum(volume)

  Z-Score计算:
  - deviation = close - VWAP
  - rolling_std = std(deviation, min_bars)
  - Z = deviation / rolling_std

  信号生成:
  - 做多: Z < -z_threshold (价格远低于VWAP，期待回归上行)
  - 做空: Z > z_threshold (价格远高于VWAP，期待回归下行)
  - 出场: Z回到0附近 (回归完成) 或 硬止损5%
  - 最少等 min_bars 根K线后才开始交易 (确保VWAP有意义)

  参数设计 (2个):
  - z_threshold: Z-Score阈值 [1.5,2.0,2.5,3.0] — 偏离灵敏度
  - min_bars: 最少K线数 [10,20,30,50] — VWAP预热期

  适用环境: 日内有明确价值中枢的震荡行情
  风险提示: 趋势市中VWAP持续偏离，Z-Score可能长期处于极端

  回测表现:
  - 测试集 (2023-2026): Sharpe 0.65 | 年化 1.03% | 最大回撤 2.12%
  - 逐年: -0.41%, +1.98%, +1.49%, +0.10% — 3/4年正收益
  - PythonGo: strategies/composite/VwapZscoreReversion_PythonGo.py
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class VwapZscoreReversion(BaseResearchStrategy):
    name = "VWAP Z-Score Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "z_threshold": [1.5, 2.0, 2.5, 3.0],
            "min_bars": [10, 20, 30, 50],
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_trading_day(df: pd.DataFrame) -> np.ndarray:
        """
        Return an array of trading-day labels (one per row).

        If df already has a 'tday' column, use it directly.
        Otherwise derive: bars whose hour >= 21 belong to the *next*
        calendar date (night session is part of the following trading day).
        """
        if "tday" in df.columns:
            return df["tday"].values

        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
        dates = idx.date
        hours = idx.hour

        tday = pd.Series(dates, index=df.index)
        night_mask = hours >= 21
        # Shift night-session bars to the next calendar date
        tday[night_mask] = tday[night_mask].apply(
            lambda d: d + pd.Timedelta(days=1)
        )
        return tday.values

    @staticmethod
    def _compute_vwap(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        tday: np.ndarray,
    ) -> np.ndarray:
        """Intraday VWAP that resets each trading day."""
        typical_price = (high + low + close) / 3.0
        tp_vol = typical_price * volume

        n = len(close)
        vwap = np.empty(n, dtype=np.float64)

        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = None

        for i in range(n):
            day = tday[i]
            if day != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = day

            cum_tp_vol += tp_vol[i]
            cum_vol += volume[i]

            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else np.nan

        return vwap

    @staticmethod
    def _bars_into_day(tday: np.ndarray) -> np.ndarray:
        """Return 1-based count of how many bars into the trading day each row is."""
        n = len(tday)
        count = np.empty(n, dtype=np.int64)
        prev_day = None
        c = 0
        for i in range(n):
            day = tday[i]
            if day != prev_day:
                c = 0
                prev_day = day
            c += 1
            count[i] = c
        return count

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        z_threshold: float = 2.0,
        min_bars: int = 20,
    ) -> np.ndarray:
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float)

        tday = self._assign_trading_day(df)

        # Intraday VWAP (resets each trading day)
        vwap = self._compute_vwap(high, low, close, volume, tday)

        # Deviation and rolling Z-score
        deviation = close - vwap
        rolling_std = pd.Series(deviation).rolling(min_bars).std().values
        z_score = np.where(rolling_std > 0, deviation / rolling_std, 0.0)

        # Bar count within each trading day
        bar_count = self._bars_into_day(tday)

        # Entry signals — mean reversion toward VWAP
        signals = np.zeros(len(df), dtype=np.int8)
        signals[z_score < -z_threshold] = 1    # price far below VWAP → long
        signals[z_score > z_threshold] = -1    # price far above VWAP → short

        # Suppress signals before min_bars into the trading day
        signals[bar_count < min_bars] = 0

        # NaN safety
        nan_mask = np.isnan(vwap) | np.isnan(rolling_std) | np.isnan(z_score)
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=1.0, trailing_pct=0.7, tp1_pct=0.6, tp2_pct=1.2, max_lots=2)
