"""
Strategy #27 — RSI + Channel Position Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: RSI超卖/超买确认 + 价格处于滚动通道极端区间 → 均值回归入场

  设计思路 (S27 — S11的RSI增强版):
  - 继承S11 (CandlePatternMeanReversion, Sharpe 1.02) 的通道位置检测逻辑
  - 将K线形态确认替换为RSI动量过滤器
  - 核心假设: 通道极端位置 + RSI极端读数 = 双重超卖/超买确认，
    反转信号置信度更高

  双重确认的市场含义:
  - 通道底部 + RSI < 超卖阈值:
    · 趋势市: 上升趋势中的深度回调，低成本做多机会
    · 震荡市: 超卖区间反弹，均值回归信号
  - 通道顶部 + RSI > 超买阈值:
    · 趋势市: 下降趋势中的深度反弹，做空布局
    · 震荡市: 超买区间回落，均值回归信号
  双重机制兼容两种市场状态，是S11隐含优势的显式化

  通道位置计算 (完全沿用S11，已验证):
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
    双重出场条件任一触发即平仓，更快锁定均值回归利润

  参数设计 (3个，27种组合，WFO可承受):
    - rsi_period:    RSI计算周期 [9, 14, 21]      约45min ~ 1.75h动量窗口
    - rsi_oversold:  超卖阈值 [20, 25, 30]        对应超买阈值 = 100 - rsi_oversold
    - channel_period: 通道周期 [15, 25, 40]       约1.25h ~ 3.3h通道

  NaN/缺失处理:
    - rsi_s 为 NaN (前 rsi_period*3+1 根K线): 信号置 0
    - ch_pos 为 NaN (前 channel_period+1 根K线): 信号置 0
    - 预热期 (前 max(rsi_period*3, channel_period) + 5 根): 信号全置 0

  研究依据:
    - 铁矿石2023+均值回归特性已由3个胜出策略验证
    - S11通道位置逻辑Sharpe 1.02，本策略以RSI替换K线形态作为第二过滤层
    - RSI与通道位置的低相关性确保双重过滤不过度收窄信号
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class RsiChannelReversion(BaseResearchStrategy):
    name = "RSI-Channel Mean-Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "rsi_period":     [9, 14, 21],      # RSI计算周期
            "rsi_oversold":   [20, 25, 30],     # 超卖阈值 (超买阈值 = 100 - 此值)
            "channel_period": [15, 25, 40],     # 滚动通道周期
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        rsi_period: int = 14,
        rsi_oversold: int = 25,
        channel_period: int = 25,
    ) -> np.ndarray:
        """
        生成交易信号。

        Args:
            df:             包含 open/high/low/close 列的 OHLCV DataFrame。
            rsi_period:     RSI计算周期 (默认14根K线)。
            rsi_oversold:   RSI超卖阈值; 超买阈值 = 100 - rsi_oversold
                            (默认25，即RSI<25做多，RSI>75做空)。
            channel_period: 滚动通道计算周期 (默认25根K线)。

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
        rsi_s     = rsi.shift(1).fillna(50.0).values

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
