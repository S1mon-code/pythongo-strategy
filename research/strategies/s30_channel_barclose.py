"""
Strategy #30 — Channel Position + Bar Close Strength Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 价格处于滚动通道极端区间 + 当前K线收盘强度确认 → 均值回归入场

  设计思路:
  - 继承S11的通道相对位置方法 (Sharpe 1.02验证有效): 价格处于N根K线通道的上/下X%区间
    即 channel_position = (close - channel_low) / (channel_high - channel_low)
  - 新增"K线收盘位置"指标: (close - low) / (high - low)
    反映当前K线内部的多空力量对比，无前瞻问题（使用当前已完成K线的OHLC）

  信号逻辑 (两个条件同时满足):
  ┌────────────────────────────────────────────────────────────┐
  │ 做多: 价格在通道底部极端区 + 当前K线强势收盘（收于K线高位）  │
  │       → 极端低位出现买方力量 → 买方在极值处介入 → 均值回归  │
  │ 做空: 价格在通道顶部极端区 + 当前K线弱势收盘（收于K线低位）  │
  │       → 极端高位出现卖方力量 → 卖方在极值处介入 → 均值回归  │
  └────────────────────────────────────────────────────────────┘

  与S11对比:
  - S11 使用明确的裸K形态（吞噬/Pin Bar/Inside Bar）作为反转确认，信号精准但频率低
  - S30 使用K线收盘位置作为确认，逻辑更简单，信号频率更高（任何强势/弱势收盘均满足）
  - 两者通道极值检测方式完全一致

  出场逻辑 (有状态):
  - 价格回归至通道中轨 (ch_low + 0.5 * ch_range) 时平仓
  - 在状态机中持续持有，直至回归或风控触发

  参数设计 (3个，27种组合，WFO可承受):
  - channel_period:  通道周期 [10, 20, 30]       — 约50min到2.5h的通道
  - extreme_pct:     极端区间比例 [0.15, 0.20, 0.25] — 通道两端多少%视为极端
  - close_threshold: 收盘强度阈值 [0.55, 0.65, 0.75] — K线内收盘位置的强弱分界

  研究依据:
  - 铁矿石2023+均值回归特性已由3个胜出策略验证
  - S11通道位置方法已证明有效 (Sharpe 1.02)
  - K线收盘位置是成熟技术分析指标，无参数过拟合风险
  - 5min周期: 信号充足，与VwapZscore/SessionGap同频
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class ChannelBarCloseReversion(BaseResearchStrategy):
    name = "Channel-BarClose Mean-Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "channel_period":  [10, 20, 30],           # 滚动通道周期
            "extreme_pct":     [0.15, 0.20, 0.25],     # 极端区间比例
            "close_threshold": [0.55, 0.65, 0.75],     # K线收盘位置阈值
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        channel_period: int = 20,
        extreme_pct: float = 0.20,
        close_threshold: float = 0.65,
    ) -> np.ndarray:
        """
        Generate mean-reversion signals using channel position + bar close strength.

        Args:
            df:               OHLCV DataFrame.
            channel_period:   Lookback bars for rolling channel (default 20).
            extreme_pct:      Fraction of channel range that qualifies as extreme (default 0.20).
            close_threshold:  Bar close position cutoff; close_pos >= threshold = strong close
                              (default 0.65). Bearish threshold mirrors as (1 - close_threshold).

        Returns:
            np.ndarray[int8]:
                 1 = long entry
                -1 = short entry
                 0 = no signal / hold
                 2 = force exit (price returned to channel mid)
        """
        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        n = len(close)

        hs = pd.Series(high)
        ls = pd.Series(low)

        # ── 通道相对位置 (shift(1) 防止前瞻) ────────────────────────────────────
        # 使用前一根K线之前的数据构建通道，当前K线收盘时通道已确定
        ch_high_arr = hs.shift(1).rolling(channel_period).max().values
        ch_low_arr  = ls.shift(1).rolling(channel_period).min().values
        ch_range_arr = ch_high_arr - ch_low_arr

        with np.errstate(divide="ignore", invalid="ignore"):
            ch_pos = np.where(
                ch_range_arr > 0,
                (close - ch_low_arr) / ch_range_arr,
                0.5,
            )

        # 超卖区 (通道底部 extreme_pct 以内) → 做多机会
        near_low  = ch_pos <= extreme_pct
        # 超买区 (通道顶部 extreme_pct 以内) → 做空机会
        near_high = ch_pos >= (1.0 - extreme_pct)

        # ── K线收盘位置 (使用当前K线自身OHLC，无前瞻) ───────────────────────────
        # 当信号被评估时K线已完成，使用当前bar的high/low合规
        bar_range = high - low
        with np.errstate(divide="ignore", invalid="ignore"):
            close_pos = np.where(bar_range > 0, (close - low) / bar_range, 0.5)

        # 强势收盘: 收盘靠近K线高点 → 买方占优
        strong_close = close_pos >= close_threshold
        # 弱势收盘: 收盘靠近K线低点 → 卖方占优
        weak_close   = close_pos <= (1.0 - close_threshold)

        # ── 入场信号 (点信号，非持仓状态) ────────────────────────────────────────
        long_entry_arr  = near_low  & strong_close
        short_entry_arr = near_high & weak_close

        # ── 有状态出场: 价格回归通道中轨时平仓 ───────────────────────────────────
        signals = np.zeros(n, dtype=np.int8)
        active = 0  # 0=空仓, 1=多头, -1=空头

        for i in range(n):
            # NaN 安全: 通道尚未预热时跳过
            if np.isnan(ch_low_arr[i]) or np.isnan(ch_high_arr[i]) or ch_range_arr[i] <= 0:
                active = 0
                continue

            ch_mid = ch_low_arr[i] + 0.5 * ch_range_arr[i]

            # 持仓状态下检查回归出场
            if active == 1 and close[i] >= ch_mid:
                signals[i] = 2
                active = 0
                continue
            elif active == -1 and close[i] <= ch_mid:
                signals[i] = 2
                active = 0
                continue

            # 空仓状态下检查入场
            if active == 0:
                if long_entry_arr[i]:
                    signals[i] = 1
                    active = 1
                elif short_entry_arr[i]:
                    signals[i] = -1
                    active = -1
                # else: 无信号，signals[i] 保持 0
            else:
                # 持仓中，未触发出场条件，保持持仓（signals[i] 保持 0，由引擎维护持仓）
                pass

        # ── NaN 安全: 强制清零预热期 ─────────────────────────────────────────────
        # channel_period + 5: 通道需要 channel_period 根shift(1)后的K线 + 少量缓冲
        warmup = channel_period + 5
        signals[:warmup] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
            unit=1,
        )
