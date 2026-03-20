"""
Strategy #36 — Large Body Reversal at Channel Extreme (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 价格处于滚动通道极端区间 + 当前K线出现大实体阳线/阴线 → 均值回归入场

  设计思路:
  - 继承S11的通道相对位置方法 (Sharpe 1.02验证有效): 价格处于N根K线通道的上/下X%区间
    即 channel_position = (close - channel_low) / (channel_high - channel_low)
  - 新增"大实体"确认: 当前K线实体 > body_mult × 近期平均实体
    大实体阳线 = 强烈买入冲量，表明买方已主动出击，反转已经开始（而非即将开始）
    大实体阴线 = 强烈卖出冲量，表明卖方已主动出击

  信号逻辑 (三个条件同时满足):
  ┌────────────────────────────────────────────────────────────────┐
  │ 做多: 价格在通道底部极端区 + 当前K线为大实体阳线（收 > 开）     │
  │       → 极端低位出现强力买盘 → 买方强势介入 → 反转已启动       │
  │ 做空: 价格在通道顶部极端区 + 当前K线为大实体阴线（收 < 开）     │
  │       → 极端高位出现强力卖盘 → 卖方强势介入 → 反转已启动       │
  └────────────────────────────────────────────────────────────────┘

  与S11对比:
  - S11 使用形态（吞噬/Pin Bar/Inside Bar）确认反转，捕捉的是反转形态结构
  - S36 使用大实体确认反转，捕捉的是实体冲量强度，逻辑更直接：大实体=买方已到场
  - 通道极值检测方式与S11完全一致

  核心优势 (S11精神的延伸):
  - 大实体阳线出现在通道底部极端区 = 可量化的"反转已开始"证据
    （实体大小 > body_mult × 平均实体，说明此根K线的买入力量显著强于平时）
  - 防前瞻: avg_body 使用 shift(1).rolling(body_window).mean()，
    即当前bar的平均实体只使用前一根bar之前的数据

  出场逻辑 (有状态):
  - 价格回归至通道中轨 (ch_low + 0.5 * ch_range) 时平仓
  - 在状态机中持续持有，直至回归或风控触发

  参数设计 (3个，27种组合，WFO可承受):
  - channel_period: 通道周期 [10, 20, 30]       — 约50min到2.5h的通道
  - extreme_pct:    极端区间比例 [0.15, 0.20, 0.25] — 通道两端多少%视为极端
  - body_mult:      大实体倍数 [1.2, 1.5, 2.0]   — 实体须为均值的多少倍
  body_window FIXED = 20 bars

  研究依据:
  - 铁矿石2023+均值回归特性已由3个胜出策略验证
  - S11通道位置方法已证明有效 (Sharpe 1.02)
  - 大实体冲量是成熟技术分析中确认买卖压力的核心指标
  - 5min周期: 信号充足，与VwapZscore/SessionGap同频
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams

_BODY_WINDOW = 20  # fixed lookback for average body size


class LargeBodyReversal(BaseResearchStrategy):
    name = "Large Body Reversal"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "channel_period": [10, 20, 30],         # 滚动通道周期
            "extreme_pct":    [0.15, 0.20, 0.25],   # 极端区间比例
            "body_mult":      [1.2, 1.5, 2.0],      # 大实体倍数阈值
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        channel_period: int = 20,
        extreme_pct: float = 0.20,
        body_mult: float = 1.5,
    ) -> np.ndarray:
        """
        Generate mean-reversion signals using channel position + large body confirmation.

        Args:
            df:             OHLCV DataFrame.
            channel_period: Lookback bars for rolling channel (default 20).
            extreme_pct:    Fraction of channel range that qualifies as extreme (default 0.20).
            body_mult:      Current bar body must exceed this multiple of avg body to qualify
                            as a large body (default 1.5).

        Returns:
            np.ndarray[int8]:
                 1 = long entry
                -1 = short entry
                 0 = no signal / hold
                 2 = force exit (price returned to channel mid)
        """
        close  = df["close"].values.astype(np.float64)
        open_  = df["open"].values.astype(np.float64)
        high   = df["high"].values.astype(np.float64)
        low    = df["low"].values.astype(np.float64)
        n = len(close)

        hs = pd.Series(high)
        ls = pd.Series(low)

        # ── 通道相对位置 (shift(1) 防止前瞻) ────────────────────────────────────
        ch_high_arr  = hs.shift(1).rolling(channel_period).max().values
        ch_low_arr   = ls.shift(1).rolling(channel_period).min().values
        ch_range_arr = ch_high_arr - ch_low_arr

        with np.errstate(divide="ignore", invalid="ignore"):
            ch_pos = np.where(
                ch_range_arr > 0,
                (close - ch_low_arr) / ch_range_arr,
                0.5,
            )

        near_low  = ch_pos <= extreme_pct           # 超卖区 → 做多机会
        near_high = ch_pos >= (1.0 - extreme_pct)   # 超买区 → 做空机会

        # ── 大实体确认 (shift(1) 防止前瞻) ───────────────────────────────────────
        # 实体大小（绝对值），与涨跌方向无关
        body = np.abs(close - open_).astype(np.float64)
        body_s = pd.Series(body)
        # avg_body: 使用 shift(1) 确保当前bar只看前一根bar之前的历史均值
        avg_body = body_s.shift(1).rolling(_BODY_WINDOW).mean().values

        # 大实体阳线 (收 > 开, 且实体 > body_mult × 历史均值)
        large_bull = (close > open_) & (body > body_mult * avg_body)
        # 大实体阴线 (收 < 开, 且实体 > body_mult × 历史均值)
        large_bear = (close < open_) & (body > body_mult * avg_body)

        # NaN safety: avg_body 为 NaN 时 large_bull/large_bear 自动为 False

        # ── 入场信号 ─────────────────────────────────────────────────────────────
        long_entry_arr  = near_low  & large_bull
        short_entry_arr = near_high & large_bear

        # ── 有状态出场: 价格回归通道中轨时平仓 ───────────────────────────────────
        signals = np.zeros(n, dtype=np.int8)
        active = 0  # 0=空仓, 1=多头, -1=空头

        for i in range(n):
            # NaN 安全: 通道或均值实体尚未预热时跳过
            if (
                np.isnan(ch_low_arr[i])
                or np.isnan(ch_high_arr[i])
                or ch_range_arr[i] <= 0
                or np.isnan(avg_body[i])
            ):
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
                # 持仓中，未触发出场条件，保持持仓（signals[i] 保持 0）
                pass

        # ── NaN 安全: 强制清零预热期 ─────────────────────────────────────────────
        # 预热期 = max(channel_period, body_window) + 5，取最大预热需求
        warmup = max(channel_period, _BODY_WINDOW) + 5
        signals[:warmup] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(
            hard_stop_pct=0.5,
            trailing_pct=0.4,
            tp1_pct=0.4,
            tp2_pct=0.8,
            max_lots=1,
            unit=1,
        )
