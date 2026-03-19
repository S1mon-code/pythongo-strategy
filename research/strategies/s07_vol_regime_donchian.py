"""
Strategy #7 — Vol Regime + Donchian Channel (15-min bars)
================================================================================

【策略思路】
  核心逻辑: 波动率分区 + 唐奇安通道 (自适应策略)

  市场在不同波动率环境下表现不同: 高波动率时趋势明确，适合突破策略；
  低波动率时价格在区间内震荡，适合均值回归。本策略根据当前波动率
  相对于长期波动率的比值来判断市场状态，自动切换逻辑。

  波动率判断:
  - 短期波动率 = donch_period 根K线收益率标准差
  - 长期波动率 = donch_period×4 根K线收益率标准差
  - vol_regime = 短期 / 长期 (>1 高波动, <1 低波动)

  唐奇安通道:
  - 上轨 = donch_period 根K线最高价 (shift 1, 不含当前bar)
  - 下轨 = donch_period 根K线最低价 (shift 1)

  信号生成:
  - 高波动 (vol_regime > vol_ratio): 趋势跟随
    - 做多: 收盘价 > 上轨 | 做空: 收盘价 < 下轨
  - 低波动 (vol_regime < 1/vol_ratio): 均值回归
    - 做多: 收盘价 < 下轨 (买入) | 做空: 收盘价 > 上轨 (卖出)

  参数设计 (2个):
  - donch_period: 唐奇安周期 [20,30,50,80]
  - vol_ratio: 波动率切换阈值 [0.8,1.0,1.2,1.5]

  适用环境: 全天候（自适应切换）
  风险提示: 波动率分区判断滞后，切换点附近信号混乱

  回测表现:
  - 训练集 (2013-2022, 无止损): Sharpe 0.60 | PF 1.27 | 1046笔交易
  - 测试集 (2023-2026): Sharpe -0.40 — 切换逻辑在近期不稳定
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class VolRegimeDonchian(BaseResearchStrategy):
    name = "Vol Regime + Donchian"
    freq = "15min"

    def param_grid(self) -> dict:
        return {
            "donch_period": [20, 30, 50, 80],
            "vol_ratio": [0.8, 1.0, 1.2, 1.5],
        }

    def generate_signals(
        self, df: pd.DataFrame, donch_period: int = 30, vol_ratio: float = 1.0
    ) -> np.ndarray:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)

        # --- Donchian Channel (shifted by 1 to exclude current bar) ---
        upper = high_s.rolling(donch_period).max().shift(1).values
        lower = low_s.rolling(donch_period).min().shift(1).values

        # --- Volatility regime detection ---
        returns = close_s.pct_change().values
        returns_s = pd.Series(returns)

        current_vol = returns_s.rolling(donch_period).std().values
        long_term_vol = returns_s.rolling(donch_period * 4).std().values

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_regime = np.where(long_term_vol > 0, current_vol / long_term_vol, np.nan)

        # --- Entry logic ---
        signals = np.zeros(len(df), dtype=np.int8)

        # High vol regime: Donchian breakout (trend-following)
        high_vol = vol_regime > vol_ratio
        signals[(high_vol) & (close > upper)] = 1
        signals[(high_vol) & (close < lower)] = -1

        # Low vol regime: Donchian reversion (fade extremes)
        low_vol = vol_regime < (1.0 / vol_ratio)
        signals[(low_vol) & (close < lower)] = 1   # buy the dip
        signals[(low_vol) & (close > upper)] = -1   # sell the top

        # NaN safety — any position where indicators are undefined → 0
        nan_mask = (
            np.isnan(upper)
            | np.isnan(lower)
            | np.isnan(vol_regime)
            | np.isnan(close)
        )
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=2.0, trailing_pct=1.2, tp1_pct=1.5, tp2_pct=3.0, max_lots=3)
