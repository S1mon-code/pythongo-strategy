"""
Strategy #37 — Intrabar Reversal at Channel Extreme (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 价格处于滚动通道极端区间 + 当前K线开弱收强（盘中吸收）→ 均值回归入场

  设计思路:
  - 继承S11的通道相对位置方法 (Sharpe 1.02验证有效): 价格处于N根K线通道的上/下X%区间
    即 channel_position = (close - channel_low) / (channel_high - channel_low)
  - 新增"盘中反转"确认:
    做多版: 当前K线开盘价 ≤ 前收价（开盘偏弱）但收盘价 > 前收价（盘中买方介入）
    做空版: 当前K线开盘价 ≥ 前收价（开盘偏强）但收盘价 < 前收价（盘中卖方介入）
    这种结构说明：
    1. 开盘时市场延续前一根K线的弱势（或强势）
    2. 但在本根K线存续期间买方（卖方）完全逆转了局面
    = 买卖压力在K线内部被吸收 = 反转已经开始的可量化证据

  防前瞻说明:
  - prev_close = close.shift(1) — 使用前一根K线收盘价，信号在当前K线收盘时才能确定
  - 当前K线的 open / close 在K线完成时均已可知，无前瞻问题

  出场逻辑 (有状态):
  - 价格回归至通道中轨 (ch_low + 0.5 * ch_range) 时平仓
  - 或当 ch_pos >= exit_pct (多头) / ch_pos <= (1 - exit_pct) (空头) 时平仓
  - 在状态机中持续持有，直至回归或风控触发

  参数设计 (3个，27种组合，WFO可承受):
  - channel_period: 通道周期 [10, 20, 30]       — 约50min到2.5h的通道
  - extreme_pct:    极端区间比例 [0.15, 0.20, 0.25] — 通道两端多少%视为极端
  - exit_pct:       出场通道位置 [0.40, 0.50, 0.60]  — 多头ch_pos达到此值时平仓

  与S11/S30对比:
  - S11 捕捉形态结构（吞噬/Pin Bar/Inside Bar），信号精准但频率低
  - S30 捕捉K线收盘位置（当前bar内部多空比），逻辑简单
  - S37 捕捉跨bar的价格行为：开盘延续弱势 → 盘中逆转 → 买方在极值处完成吸收
    信号的含义更明确：开盘时卖压仍在（open ≤ prev_close），但收盘时买方已完全反转

  研究依据:
  - 铁矿石2023+均值回归特性已由3个胜出策略验证
  - S11通道位置方法已证明有效 (Sharpe 1.02)
  - "盘中吸收"（intrabar absorption）是价格行为交易的核心概念
  - 5min周期: 信号充足，与VwapZscore/SessionGap同频
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class IntrabarReversal(BaseResearchStrategy):
    name = "Intrabar Reversal"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "channel_period": [10, 20, 30],         # 滚动通道周期
            "extreme_pct":    [0.15, 0.20, 0.25],   # 极端区间比例
            "exit_pct":       [0.40, 0.50, 0.60],   # 多头出场通道位置阈值
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        channel_period: int = 20,
        extreme_pct: float = 0.20,
        exit_pct: float = 0.50,
    ) -> np.ndarray:
        """
        Generate mean-reversion signals using channel position + intrabar reversal.

        The intrabar reversal pattern confirms that buying (or selling) pressure absorbed
        the prevailing move WITHIN the bar itself — the single strongest bar-level evidence
        that a reversal has already started.

        Args:
            df:             OHLCV DataFrame.
            channel_period: Lookback bars for rolling channel (default 20).
            extreme_pct:    Fraction of channel range that qualifies as extreme (default 0.20).
            exit_pct:       Long exits when ch_pos >= exit_pct; short exits when
                            ch_pos <= (1 - exit_pct). Also exits at channel midpoint
                            (default 0.50).

        Returns:
            np.ndarray[int8]:
                 1 = long entry
                -1 = short entry
                 0 = no signal / hold
                 2 = force exit (price returned to channel mid or exit_pct reached)
        """
        close = df["close"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        n = len(close)

        hs = pd.Series(high)
        ls = pd.Series(low)
        cs = pd.Series(close)

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

        # ── 盘中反转确认 ──────────────────────────────────────────────────────────
        # prev_close: 前一根K线的收盘价（shift(1)，在当前K线收盘时已知，无前瞻）
        prev_close = cs.shift(1).values

        # 盘中反转-多头: 开盘 ≤ 前收（延续弱势）但收盘 > 前收（盘中买方完全逆转）
        intrabar_bull = (open_ <= prev_close) & (close > prev_close)
        # 盘中反转-空头: 开盘 ≥ 前收（延续强势）但收盘 < 前收（盘中卖方完全逆转）
        intrabar_bear = (open_ >= prev_close) & (close < prev_close)

        # NaN safety: prev_close[0] = NaN → 第0根bar的比较结果为 False，自然排除

        # ── 入场信号 ─────────────────────────────────────────────────────────────
        long_entry_arr  = near_low  & intrabar_bull
        short_entry_arr = near_high & intrabar_bear

        # ── 有状态出场 ────────────────────────────────────────────────────────────
        # 出场条件: 价格回归通道中轨 OR ch_pos 到达 exit_pct（多头）/(1-exit_pct)（空头）
        signals = np.zeros(n, dtype=np.int8)
        active = 0  # 0=空仓, 1=多头, -1=空头

        for i in range(n):
            # NaN 安全: 通道尚未预热时跳过
            if (
                np.isnan(ch_low_arr[i])
                or np.isnan(ch_high_arr[i])
                or ch_range_arr[i] <= 0
                or np.isnan(prev_close[i])
            ):
                active = 0
                continue

            ch_mid = ch_low_arr[i] + 0.5 * ch_range_arr[i]
            cpos_i = ch_pos[i]

            # 持仓状态下检查出场
            if active == 1:
                # 回归中轨或通道位置达到出场阈值
                if close[i] >= ch_mid or cpos_i >= exit_pct:
                    signals[i] = 2
                    active = 0
                    continue
            elif active == -1:
                if close[i] <= ch_mid or cpos_i <= (1.0 - exit_pct):
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
        warmup = channel_period + 5
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
