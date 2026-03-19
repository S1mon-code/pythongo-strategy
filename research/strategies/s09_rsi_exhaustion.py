"""
Strategy #9 — RSI Exhaustion Reversal (15-min bars)
================================================================================

【策略思路】
  核心逻辑: RSI超卖超买反转 + VWAP方向过滤

  RSI到达极端区域(超卖/超买)时，意味着短期价格运动已经"疲惫"，
  反转概率增加。但单纯的RSI超卖不足以判断——可能是趋势中的正常
  回调。加入VWAP方向过滤: RSI超卖+价格在VWAP上方=不是下跌趋势
  中的超卖，而是暂时的过度反应，可以做多。

  信号生成:
  - RSI: EWM平滑(Wilder风格), 期间 rsi_period
  - VWAP: 日内成交量加权平均价，每日重置
  - 做多: RSI < oversold 且 close > VWAP (超卖但价格仍在均价上方)
  - 做空: RSI > overbought 且 close < VWAP (超买但价格在均价下方)
  - 冷却期: 信号后等待3根K线再产生下一个信号

  参数设计 (3个):
  - rsi_period: RSI周期 [10,14,20,30]
  - oversold: 超卖阈值 [20,25,30]
  - overbought: 超买阈值 [70,75,80]

  适用环境: 震荡市中的超买超卖反转
  风险提示: 交易频率低，VWAP过滤可能过于严格

  回测表现:
  - 训练集 (2013-2022, 无止损): Sharpe 0.51 | PF 1.71 | 147笔交易
  - 测试集 (2023-2026): Sharpe -0.13 — 信号过少不稳定
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class RsiExhaustionReversal(BaseResearchStrategy):
    name = "RSI Exhaustion Reversal"
    freq = "15min"

    def param_grid(self) -> dict:
        return {
            "rsi_period": [10, 14, 20, 30],
            "oversold": [20, 25, 30],
            "overbought": [70, 75, 80],
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        rsi_period: int = 14,
        oversold: int = 25,
        overbought: int = 75,
    ) -> np.ndarray:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        close_s = pd.Series(close)

        # ---- RSI (EWM-smoothed Wilder style) ----
        delta = close_s.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(span=rsi_period, adjust=False).mean().values
        avg_loss = loss.ewm(span=rsi_period, adjust=False).mean().values

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

        # ---- VWAP (reset each trading day) ----
        typical_price = (high + low + close) / 3.0

        # Determine day boundaries for VWAP reset
        if "tday" in df.columns:
            day_labels = df["tday"].values
        else:
            day_labels = pd.Series(df.index).dt.date.values

        vwap = np.full(len(df), np.nan)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = None

        for i in range(len(df)):
            cur_day = day_labels[i]
            if cur_day != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = cur_day

            cum_tp_vol += typical_price[i] * volume[i]
            cum_vol += volume[i]

            if cum_vol > 0:
                vwap[i] = cum_tp_vol / cum_vol

        # ---- Entry signals ----
        signals = np.zeros(len(df), dtype=np.int8)

        # Long: RSI oversold AND price above VWAP (exhausted selling, not freefall)
        long_mask = (rsi < oversold) & (close > vwap)
        signals[long_mask] = 1

        # Short: RSI overbought AND price below VWAP (exhausted buying, not melt-up)
        short_mask = (rsi > overbought) & (close < vwap)
        signals[short_mask] = -1

        # ---- Cooldown: 3 bars after any signal ----
        cooldown = 0
        for i in range(len(signals)):
            if cooldown > 0:
                signals[i] = 0
                cooldown -= 1
            elif signals[i] != 0:
                cooldown = 3

        # ---- NaN safety ----
        nan_mask = np.isnan(rsi) | np.isnan(vwap) | np.isnan(close)
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=1.5, trailing_pct=1.0, tp1_pct=1.0, tp2_pct=2.0, max_lots=2)
