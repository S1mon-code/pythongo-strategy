"""
Strategy #15 — Variance Ratio Adaptive Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: VR < threshold 作为政体门控; Z-score偏离EMA触发信号

  方差比检验 (Lo & MacKinlay 1988) 是检验均值回复的经典学术工具。
  VR = Var(k步收益) / (k × Var(1步收益)):
    - VR < 1: 均值回复 — 短期波动后向均值靠拢
    - VR > 1: 趋势延续 — 价格继续延伸
    - VR ≈ 1: 随机游走

  Firoozye 2025 研究发现，VR从1.0降至0.7时，均值回复策略的Sharpe
  近似翻倍。本策略以VR < vr_threshold作为政体门，仅在均值回复政体下
  开仓，避免在趋势市中逆势操作。

  信号生成:
  - 计算 vr_window 根K线的方差比 VR(k=5)
  - 计算 close 对 EMA(z_period) 的Z-score偏离
  - Z = (close - EMA) / rolling_std(close - EMA, z_period)
  - 做多: Z < -1.5 且 VR < vr_threshold → 超卖回归买入
  - 做空: Z > +1.5 且 VR < vr_threshold → 超买回归卖出

  参数设计 (3个，27组合):
  - vr_window:    [30, 50, 80]     — VR估算滚动窗口
  - vr_threshold: [0.75, 0.85, 0.95] — VR政体门 (低于此值视为均值回复)
  - z_period:     [15, 25, 40]     — EMA / Z-score 周期
  - z_threshold 固定为 1.5

  关键设计决策:
  - k=5 固定，对应5根5分钟K线 = 25分钟跨度的方差比
  - 滚动方差使用 pandas .var() (ddof=1 无偏)
  - NaN安全: vr/z_score/dev_std任一为NaN时信号归零

  适用环境: 震荡市、区间震荡
  风险提示: VR门控虽能过滤趋势市，但VR本身有滞后性
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams

_Z_THRESHOLD = 1.5  # fixed entry threshold — not part of the grid
_VR_K = 5           # fixed k for k-step returns in VR computation


class VarianceRatioReversion(BaseResearchStrategy):
    name = "Variance Ratio Mean-Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "vr_window":    [30, 50, 80],
            "vr_threshold": [0.75, 0.85, 0.95],
            "z_period":     [15, 25, 40],
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        vr_window: int = 50,
        vr_threshold: float = 0.85,
        z_period: int = 25,
    ) -> np.ndarray:
        close = pd.Series(df["close"].values, dtype=np.float64)

        # ------------------------------------------------------------------
        # Step 1: Variance Ratio  VR = Var(k-step returns) / (k * Var(1-step))
        # ------------------------------------------------------------------
        k = _VR_K
        ret1 = close.pct_change(1)           # 1-step returns
        retk = close.pct_change(k)           # k-step returns

        var1 = ret1.rolling(vr_window).var()  # rolling variance of 1-step
        vark = retk.rolling(vr_window).var()  # rolling variance of k-step

        # Avoid divide-by-zero: where var1==0, vr stays NaN
        vr = vark / (k * var1)

        # ------------------------------------------------------------------
        # Step 2: Z-score of price deviation from EMA
        # ------------------------------------------------------------------
        ema = close.ewm(span=z_period, adjust=False).mean()
        dev = close - ema
        dev_std = dev.rolling(z_period).std()

        # z_score is NaN where dev_std == 0 or rolling window not yet full
        z_score = dev / dev_std

        # ------------------------------------------------------------------
        # Step 3: Signal generation — regime gate + Z-score trigger
        # ------------------------------------------------------------------
        mean_reverting = vr < vr_threshold          # VR regime gate
        long_signal  = (z_score < -_Z_THRESHOLD) & mean_reverting
        short_signal = (z_score >  _Z_THRESHOLD) & mean_reverting

        signals = np.zeros(len(df), dtype=np.int8)
        signals[long_signal.values]  =  1
        signals[short_signal.values] = -1

        # ------------------------------------------------------------------
        # NaN safety — zero out any bar where computation is incomplete
        # ------------------------------------------------------------------
        nan_mask = vr.isna().values | z_score.isna().values | dev_std.isna().values
        signals[nan_mask] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(
            hard_stop_pct=0.7,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
