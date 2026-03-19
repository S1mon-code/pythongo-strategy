"""
Strategy #3 — Order Flow Imbalance (3-min bars)
================================================================================

【策略思路】
  核心逻辑: 持仓量(OI)变化 + 量比 → 跟随主力方向

  期货市场中，持仓量(OI)的显著增加意味着新资金入场建仓。当OI增幅
  超过阈值，且成交量远超均量时，说明机构正在集中建仓。此时跟随K线
  方向（阳线做多、阴线做空）即跟随主力资金方向。

  信号生成:
  - OI变化率 = oi.pct_change()
  - 量比 = 当前成交量 / 20根K线均量
  - 做多: OI增幅 > oi_threshold 且 量比 > vol_ratio 且 收阳 (close > open)
  - 做空: OI增幅 > oi_threshold 且 量比 > vol_ratio 且 收阴 (close < open)

  参数设计 (2个):
  - oi_threshold: OI变化阈值 [0.005,0.01,0.02,0.03]
  - vol_ratio: 量比阈值 [1.5,2.0,2.5,3.0]

  适用环境: 主力建仓期、趋势启动前
  风险提示: OI数据噪声大，信号极其稀少(仅75笔/9年)，不具统计可靠性

  回测表现:
  - 训练集 (无止损): Sharpe 1.05 | PF 3.25 — 但仅75笔交易
  - 测试集: 仅1笔交易 — 信号过于稀少，不可用
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class OrderFlowImbalance(BaseResearchStrategy):
    name = "Order Flow Imbalance"
    freq = "3min"

    def param_grid(self) -> dict:
        return {
            "oi_threshold": [0.005, 0.01, 0.02, 0.03],
            "vol_ratio": [1.5, 2.0, 2.5, 3.0],
        }

    def generate_signals(
        self, df: pd.DataFrame, oi_threshold: float = 0.01, vol_ratio: float = 2.0
    ) -> np.ndarray:
        signals = np.zeros(len(df), dtype=int)

        # OI change pct (row-over-row)
        oi_chg = df["oi"].pct_change().values

        # Volume ratio vs 20-bar rolling mean
        vol_mean = df["volume"].rolling(20).mean().values
        v_ratio = np.where(vol_mean > 0, df["volume"].values / vol_mean, 0.0)

        # Bar direction
        bullish = (df["close"].values > df["open"].values)
        bearish = (df["close"].values < df["open"].values)

        # Core condition: OI surging + volume spike
        flow = (oi_chg > oi_threshold) & (v_ratio > vol_ratio)

        signals[flow & bullish] = 1
        signals[flow & bearish] = -1

        # Replace any NaN-originated positions with 0
        nan_mask = np.isnan(oi_chg) | np.isnan(v_ratio)
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=0.8, trailing_pct=0.6, tp1_pct=0.5, tp2_pct=1.2)
