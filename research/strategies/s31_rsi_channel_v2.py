"""
Strategy #31 — RSI Channel Mean-Reversion v2 (5-min bars)
================================================================================

【策略思路】
  本策略是 S27 (RSI-Channel Mean-Reversion) 的改进版本，针对 S27 在实盘/回测中
  因交易频率过低 (约6.7笔/年) 而无法有效统计的缺陷进行参数网格修正。

  S27 回顾与问题诊断:
  - S27 WFO 特性优秀: 4个窗口中3个盈利，参数稳健，OOS/IS = 0.55
  - 致命缺陷: WFO 倾向于选出 rsi_oversold=20，该阈值极少触发，
    导致全年仅约6.7笔交易，统计显著性严重不足
  - 根本原因: 参数网格 [20, 25, 30] 中包含了过于严苛的20阈值

  S31 修复方案 (两项关键调整):
  1. 移除 rsi_oversold=20，替换为 [25, 30, 35]
     - 最低阈值从20提升至25，确保最严苛条件下也有合理信号频率
     - 上限扩展至35，捕获更多轻度超卖/超买场景
  2. 缩短通道周期 [15, 25, 40] → [10, 15, 25]
     - 更短的通道周期意味着更频繁地触及极端区间
     - 10根5分钟K线 ≈ 50分钟通道（短期快速回归目标）
     - 25根5分钟K线 ≈ 2小时通道（与S27最短通道持平）

  核心逻辑 (继承自S27，与S11通道逻辑兼容):
  - RSI超卖/超买确认 + 价格处于滚动通道极端区间 → 均值回归入场

  通道位置计算 (完全沿用S11/S27，已验证):
    ch_high = rolling(channel_period).max  对前N根K线最高价 (shift(1)防前瞻)
    ch_low  = rolling(channel_period).min  对前N根K线最低价 (shift(1)防前瞻)
    ch_range = ch_high - ch_low
    ch_pos  = (close - ch_low) / ch_range  ∈ [0, 1]
    near_low  = ch_pos ≤ 0.20              → 通道底部20%区域
    near_high = ch_pos ≥ 0.80              → 通道顶部20%区域

  RSI计算 (无ta-lib，Wilder EWM平滑):
    ret       = close.diff()
    gain      = ret.clip(lower=0)
    loss      = (-ret).clip(lower=0)
    avg_gain  = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss  = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
    rs        = avg_gain / avg_loss  (avg_loss=0时置NaN → rsi=100)
    rsi       = 100 - 100 / (1 + rs)，NaN填充为中性值50
    rsi_s     = rsi.shift(1)         (防前瞻: 使用前一根K线的RSI)

  信号合成:
    做多入场:  near_low  & (rsi_s < rsi_oversold)         → 通道底部 + 超卖
    做空入场:  near_high & (rsi_s > (100 - rsi_oversold)) → 通道顶部 + 超买

  出场逻辑 (有状态循环):
    持多仓时: rsi_s > 50 OR close ≥ 通道中轴 → 动量恢复中性，平多
    持空仓时: rsi_s < 50 OR close ≤ 通道中轴 → 动量恢复中性，平空
    双重出场条件任一触发即平仓，快速锁定均值回归利润

  参数网格 (3个，27种组合，WFO可承受) — 与S27的关键区别:
    - rsi_period:    RSI计算周期 [9, 14, 21]
    - rsi_oversold:  超卖阈值 [25, 30, 35]   ← 移除20，上限扩至35 (S27为[20,25,30])
    - channel_period: 通道周期 [10, 15, 25]  ← 整体缩短 (S27为[15,25,40])

  NaN/缺失处理:
    - rsi_s 为 NaN (前 rsi_period*3+1 根K线): 信号置 0
    - ch_pos 为 NaN (前 channel_period+1 根K线): 信号置 0
    - 预热期 (前 max(rsi_period*3, channel_period) + 5 根): 信号全置 0

  研究依据:
    - S27 WFO 验证了 RSI + 通道双重过滤逻辑在铁矿石5分钟级别的有效性
    - S31 通过扩大信号触发区间解决交易稀少问题，保留双重过滤核心机制
    - 预期年均交易频率: 由 ~7笔 提升至 ~20-40笔 (基于阈值放宽估算)
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class RsiChannelReversionV2(BaseResearchStrategy):
    name = "RSI-Channel MR v2"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "rsi_period":     [9, 14, 21],      # RSI计算周期
            "rsi_oversold":   [25, 30, 35],     # 超卖阈值 (超买阈值 = 100 - 此值); 相比S27移除了20，增加了35
            "channel_period": [10, 15, 25],     # 滚动通道周期; 相比S27整体缩短以提高信号频率
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        channel_period: int = 15,
    ) -> np.ndarray:
        """
        生成交易信号 (S27 v2 修正版)。

        与 S27 的差异仅在参数默认值和 param_grid 上，核心计算逻辑完全相同，
        以确保两版本的回测结果可对比。

        Args:
            df:             包含 open/high/low/close 列的 OHLCV DataFrame。
            rsi_period:     RSI计算周期 (默认14根K线)。
            rsi_oversold:   RSI超卖阈值; 超买阈值 = 100 - rsi_oversold
                            (默认30; S31网格范围为[25,30,35]，S27为[20,25,30])。
            channel_period: 滚动通道计算周期 (默认15根K线;
                            S31网格范围为[10,15,25]，S27为[15,25,40])。

        Returns:
            np.ndarray (int8):
                 1  = 做多信号 (通道底部 + RSI超卖)
                -1  = 做空信号 (通道顶部 + RSI超买)
                 0  = 无信号 / 持仓中继续持有
                 2  = 强制平仓 (RSI回归中性 或 价格到达通道中轴)
        """
        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        n     = len(close)

        cs = pd.Series(close)
        hs = pd.Series(high)
        ls = pd.Series(low)

        # ── RSI (Wilder EWM, 无ta-lib) ────────────────────────────────────────
        # alpha = 1/rsi_period 对应 Wilder 平滑 (span = 2*rsi_period - 1)
        ret  = cs.diff()
        gain = ret.clip(lower=0)
        loss = (-ret).clip(lower=0)

        avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()

        # avg_loss = 0 → RS = inf → RSI = 100; 用 replace 防止零除
        with np.errstate(divide="ignore", invalid="ignore"):
            rs  = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100.0 - 100.0 / (1.0 + rs)

        # NaN (首bar无diff) 填充为中性值50，不产生虚假信号
        rsi = rsi.fillna(50.0)

        # shift(1): 使用前一根K线的RSI，防前瞻
        rsi_s = rsi.shift(1).fillna(50.0).values

        # ── 通道相对位置 (shift(1) 防前瞻) ──────────────────────────────────────
        ch_high  = hs.shift(1).rolling(channel_period).max().values
        ch_low   = ls.shift(1).rolling(channel_period).min().values
        ch_range = ch_high - ch_low

        with np.errstate(divide="ignore", invalid="ignore"):
            ch_pos = np.where(ch_range > 0, (close - ch_low) / ch_range, 0.5)

        # 通道底部20% → 做多机会; 通道顶部20% → 做空机会
        near_low  = ch_pos <= 0.20
        near_high = ch_pos >= 0.80

        # ── 入场条件 (非有状态，纯触发标志) ──────────────────────────────────────
        rsi_overbought = 100.0 - rsi_oversold

        long_entry  = near_low  & (rsi_s < rsi_oversold)    # 通道底部 + 超卖
        short_entry = near_high & (rsi_s > rsi_overbought)  # 通道顶部 + 超买

        # ── 有状态循环: 持仓跟踪 + 双重出场条件 ──────────────────────────────────
        #   active:  1 = 持多, -1 = 持空, 0 = 空仓
        #   signals: 1=入多, -1=入空, 2=平仓, 0=持续/空仓
        signals = np.zeros(n, dtype=np.int8)

        active: int = 0

        for i in range(n):
            # 通道中轴: ch_low + 50% × ch_range (用于出场)
            if ch_range[i] > 0 and np.isfinite(ch_low[i]):
                ch_mid = ch_low[i] + 0.5 * ch_range[i]
            else:
                ch_mid = np.nan

            # 先检查出场条件 (已持仓时优先处理)
            if active == 1:
                # 出场: RSI恢复至中性区(>50) 或 价格达到通道中轴
                if rsi_s[i] > 50.0 or (np.isfinite(ch_mid) and close[i] >= ch_mid):
                    signals[i] = 2
                    active = 0
                    continue
                else:
                    signals[i] = 0
                    continue

            elif active == -1:
                # 出场: RSI恢复至中性区(<50) 或 价格达到通道中轴
                if rsi_s[i] < 50.0 or (np.isfinite(ch_mid) and close[i] <= ch_mid):
                    signals[i] = 2
                    active = 0
                    continue
                else:
                    signals[i] = 0
                    continue

            # 空仓: 检查入场信号
            if long_entry[i]:
                signals[i] = 1
                active = 1
            elif short_entry[i]:
                signals[i] = -1
                active = -1
            else:
                signals[i] = 0

        # ── NaN safety — 预热期清零 ───────────────────────────────────────────────
        # RSI 需要 rsi_period*3 根K线充分平滑; 通道需要 channel_period 根K线
        # 额外+5作为安全缓冲
        warmup = max(rsi_period * 3, channel_period) + 5
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
