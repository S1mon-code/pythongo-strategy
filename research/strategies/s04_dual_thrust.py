"""
Strategy #4 — Dual Thrust Breakout (15-min bars)
================================================================================

【策略思路】
  核心逻辑: 前日区间突破，不对称阈值（经典Dual Thrust策略）

  Dual Thrust是经典的日内突破策略。用过去N个交易日的最高价、最低价、
  最高收盘价、最低收盘价计算区间宽度，然后以今日开盘价为锚点，
  向上/向下设置不对称的突破触发线。价格突破即入场。

  信号生成:
  - HH = lookback日最高价, LL = lookback日最低价
  - HC = lookback日最高收盘价, LC = lookback日最低收盘价
  - range = max(HH - LC, HC - LL)
  - 上轨 = 今日开盘价 + k1 × range
  - 下轨 = 今日开盘价 - k2 × range (k1≠k2 实现不对称)
  - 做多: 收盘价突破上轨
  - 做空: 收盘价跌破下轨
  - 每日仅触发一次

  参数设计 (3个):
  - lookback: 回看天数 [3,5,7,10]
  - k1: 上轨乘数 [0.4,0.5,0.6,0.7] — 做多灵敏度
  - k2: 下轨乘数 [0.3,0.4,0.5,0.6] — 做空灵敏度

  适用环境: 趋势行情、波动扩大期
  风险提示: 2023年后铁矿石转为震荡市，突破策略表现显著下降

  回测表现:
  - 训练集 (2013-2022, 无止损): Sharpe 0.92 | PF 1.74 | 544笔交易
  - 测试集 (2023-2026): Sharpe -0.23 — 趋势市→震荡市转换后失效
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class DualThrustBreakout(BaseResearchStrategy):
    name = "Dual Thrust Breakout"
    freq = "15min"

    def param_grid(self) -> dict:
        return {
            "lookback": [3, 5, 7, 10],
            "k1": [0.4, 0.5, 0.6, 0.7],
            "k2": [0.3, 0.4, 0.5, 0.6],
        }

    def generate_signals(
        self, df: pd.DataFrame, lookback: int = 5, k1: float = 0.5, k2: float = 0.4
    ) -> np.ndarray:
        signals = np.zeros(len(df), dtype=np.int8)

        # --- Derive tday if missing -------------------------------------------
        if "tday" in df.columns:
            tday = df["tday"].values
        else:
            tday = pd.Series(df.index.date, index=df.index).values

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_ = df["open"].values

        unique_days = pd.unique(tday)

        # --- Pre-compute daily OHLC summaries ---------------------------------
        day_high = {}   # tday -> highest high
        day_low = {}    # tday -> lowest low
        day_hc = {}     # tday -> highest close
        day_lc = {}     # tday -> lowest close
        day_open = {}   # tday -> first open price

        for d in unique_days:
            mask = tday == d
            idx = np.where(mask)[0]
            day_high[d] = np.nanmax(high[idx])
            day_low[d] = np.nanmin(low[idx])
            day_hc[d] = np.nanmax(close[idx])
            day_lc[d] = np.nanmin(close[idx])
            day_open[d] = open_[idx[0]]

        # --- Generate signals day by day --------------------------------------
        for i, d in enumerate(unique_days):
            if i < lookback:
                continue  # not enough history

            # Previous N days' stats
            prev_days = unique_days[i - lookback : i]
            hh = max(day_high[dd] for dd in prev_days)
            ll = min(day_low[dd] for dd in prev_days)
            hc = max(day_hc[dd] for dd in prev_days)
            lc = min(day_lc[dd] for dd in prev_days)

            range_val = max(hh - lc, hc - ll)
            today_open = day_open[d]

            upper = today_open + k1 * range_val
            lower = today_open - k2 * range_val

            # Indices belonging to today
            day_idx = np.where(tday == d)[0]
            triggered = False

            for j in day_idx:
                if triggered:
                    break
                c = close[j]
                if np.isnan(c):
                    continue
                if c > upper:
                    signals[j] = 1
                    triggered = True
                elif c < lower:
                    signals[j] = -1
                    triggered = True

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=1.5, trailing_pct=1.0, tp1_pct=1.0, tp2_pct=2.5, max_lots=3)
