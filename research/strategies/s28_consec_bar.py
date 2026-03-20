"""
Strategy #28 — Consecutive Bar Exhaustion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: N根连续阳线/阴线后价格反转概率显著提升 → 配合通道位置过滤入场均值回归

  学术依据:
  - "Short-term return reversals" (DeBondt & Thaler 1985 及大量后续研究)
  - 连续同向运动 = 短期方向性过度延伸，随后均值回归概率上升
  - 铁矿石2023+ 均值回归特性已由3个胜出策略验证 (S10/S11/S14)

  设计优势:
  1. 趋势市与震荡市均适用:
     - 趋势市: N根连续上涨 = 超短期超买，可能出现回撤修复
     - 震荡市: N根连续上涨 = 通道顶部过度延伸，完整均值回归
  2. 信号定义极简 — 无需复杂指标，只计数连续涨跌K线数量
  3. 与S11 (CandlePatternMeanReversion, Sharpe 1.02) 互补:
     S11依赖形态识别 (吞噬/Pin Bar/Inside Bar)，S28依赖连续Bar计数，
     两者在通道位置过滤上完全一致，但确认信号来源不同

  通道位置过滤 (完全沿用S11/S26已验证逻辑):
    ch_high  = rolling(channel_period).max  对前N根K线最高价 (shift(1)防前瞻)
    ch_low   = rolling(channel_period).min  对前N根K线最低价 (shift(1)防前瞻)
    ch_range = ch_high - ch_low
    ch_pos   = (close - ch_low) / ch_range  ∈ [0, 1]
    near_low  = ch_pos ≤ extreme_pct        → 超卖极端区
    near_high = ch_pos ≥ 1 - extreme_pct   → 超买极端区

  连续Bar计数:
    up_bar[i]   = 1 if close[i] > close[i-1] else 0
    down_bar[i] = 1 if close[i] < close[i-1] else 0

    consec_up[i]   = consec_up[i-1] + 1   if up_bar[i]   else 0
    consec_down[i] = consec_down[i-1] + 1 if down_bar[i] else 0

  信号触发 (使用前移1根，即上一根完成的连续计数触发当前bar入场):
    prev_consec_up[i]   = consec_up[i-1]
    prev_consec_down[i] = consec_down[i-1]

    做多入场: prev_consec_down >= n_bars  AND  near_low
    做空入场: prev_consec_up   >= n_bars  AND  near_high

  出场逻辑 (有状态，双重出场):
    1. 通道中轴出场 (主出场):
       持多: close >= ch_low + 0.5 * ch_range → 平仓
       持空: close <= ch_low + 0.5 * ch_range → 平仓
    2. 强制超时出场 (安全出场):
       持仓超过 n_bars + 3 根K线后强制平仓，防止信号失效时无限持仓

  参数设计 (3个，27种组合，WFO可承受):
    - n_bars:         连续Bar阈值 [3, 4, 5]          最少几根连续涨/跌才触发信号
    - extreme_pct:    极端区间比例 [0.15, 0.20, 0.25] 通道两端多少%视为极端
    - channel_period: 通道周期 [10, 20, 30]           约50min ~ 2.5h通道

  NaN/缺失处理:
    - ch_range = 0 或 ch 为 NaN: 该bar信号置0
    - 预热期 (前 channel_period + 5 根): 信号全置0
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class ConsecBarExhaustion(BaseResearchStrategy):
    """
    连续Bar耗竭策略 — 均值回归。

    N根连续同向K线 + 价格处于通道极端区间 → 反向入场，等待通道中轴出场。
    """

    name = "Consecutive Bar Exhaustion"
    freq = "5min"

    def param_grid(self) -> dict:
        """
        返回参数优化网格。

        Returns:
            dict: 参数名 → 候选值列表，共 3×3×3 = 27 种组合。
        """
        return {
            "n_bars":         [3, 4, 5],           # 触发信号所需的最少连续Bar数
            "extreme_pct":    [0.15, 0.20, 0.25],  # 通道极端区间比例
            "channel_period": [10, 20, 30],         # 滚动通道计算周期
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        n_bars: int = 4,
        extreme_pct: float = 0.20,
        channel_period: int = 20,
    ) -> np.ndarray:
        """
        生成交易信号。

        算法分三步:
          1. 计算通道位置 (ch_pos)，识别超买/超卖极端区；
          2. 统计连续上涨/下跌Bar计数，前移1根得到触发条件；
          3. 有状态循环: 入场 → 通道中轴出场 / 超时强制出场。

        Args:
            df:             包含 open/high/low/close 列的 OHLCV DataFrame。
            n_bars:         触发信号的最少连续同向Bar数 (默认4)。
            extreme_pct:    通道极端区间阈值，[0,1] 区间 (默认0.20 = 两端各20%)。
            channel_period: 滚动通道计算周期，单位: 根K线 (默认20)。

        Returns:
            np.ndarray (int8), 长度与 df 相同:
                 1  = 做多信号 (连续下跌 >= n_bars + 处于通道底部极端区)
                -1  = 做空信号 (连续上涨 >= n_bars + 处于通道顶部极端区)
                 0  = 无信号 / 持仓中继续持有
                 2  = 强制平仓 (到达通道中轴 或 持仓超时)
        """
        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        n     = len(close)

        hs = pd.Series(high)
        ls = pd.Series(low)

        # ── 通道相对位置 (shift(1) 防前瞻) ──────────────────────────────────────
        ch_high  = hs.shift(1).rolling(channel_period).max().values
        ch_low   = ls.shift(1).rolling(channel_period).min().values
        ch_range = ch_high - ch_low

        with np.errstate(divide="ignore", invalid="ignore"):
            ch_pos = np.where(ch_range > 0, (close - ch_low) / ch_range, 0.5)

        # 超卖区 (通道底部 extreme_pct 以内) → 做多机会
        near_low  = ch_pos <= extreme_pct
        # 超买区 (通道顶部 extreme_pct 以内) → 做空机会
        near_high = ch_pos >= (1.0 - extreme_pct)

        # ── 连续Bar计数 ───────────────────────────────────────────────────────────
        # prev_close[i] = close[i-1]，用于判断当前Bar是否为阳线/阴线
        prev_close    = np.empty(n, dtype=np.float64)
        prev_close[0] = np.nan
        prev_close[1:] = close[:-1]

        # up_bar[i]=1 表示第i根K线收盘高于前一根 (阳线); down_bar同理
        up_bar   = (close > prev_close).astype(np.int32)
        down_bar = (close < prev_close).astype(np.int32)
        # 第0根无前一根，不计入任何连续序列
        up_bar[0]   = 0
        down_bar[0] = 0

        # 连续计数: consec_up[i] = 到第i根为止连续阳线数量
        consec_up   = np.zeros(n, dtype=np.int32)
        consec_down = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            consec_up[i]   = consec_up[i - 1]   + 1 if up_bar[i]   else 0
            consec_down[i] = consec_down[i - 1] + 1 if down_bar[i] else 0

        # 前移1根: prev_consec_up[i] 表示前一根bar完成的连续阳线计数
        # 信号含义: 上一根bar刚结束了N根连续上涨，当前bar是潜在反转bar
        prev_consec_up              = np.zeros(n, dtype=np.int32)
        prev_consec_down            = np.zeros(n, dtype=np.int32)
        prev_consec_up[1:]          = consec_up[:-1]
        prev_consec_down[1:]        = consec_down[:-1]

        # ── 入场条件 (非有状态，纯布尔数组) ────────────────────────────────────────
        # 做多: 前一根完成了 >= n_bars 根连续下跌，且当前价格在通道底部极端区
        long_entry  = (prev_consec_down >= n_bars) & near_low
        # 做空: 前一根完成了 >= n_bars 根连续上涨，且当前价格在通道顶部极端区
        short_entry = (prev_consec_up   >= n_bars) & near_high

        # ── 有状态循环: 持仓跟踪 + 双重出场 ──────────────────────────────────────
        #   active:     1 = 持多, -1 = 持空, 0 = 空仓
        #   hold_count: 当前持仓已持续的Bar数 (用于超时强制出场)
        #   max_hold:   最大持仓Bar数 = n_bars + 3
        #
        #   出场优先级:
        #     1. 通道中轴出场 (ch_mid)     — 目标达到，正常获利出场
        #     2. 超时强制出场 (max_hold)   — 信号失效保护
        #
        #   注: 硬止损 / 移动止损 / TP 由 backtest_engine 的 PositionParams 统一管理，
        #       此处不重复实现。
        signals    = np.zeros(n, dtype=np.int8)
        active     = 0
        hold_count = 0
        max_hold   = n_bars + 3

        for i in range(n):
            # 通道中轴; ch_range 为 NaN 或 0 时 ch_mid = NaN → 出场条件不触发 (安全)
            if ch_range[i] > 0 and np.isfinite(ch_low[i]):
                ch_mid = ch_low[i] + 0.5 * ch_range[i]
            else:
                ch_mid = np.nan

            # ── 已持仓: 先检查出场条件 ──────────────────────────────────────────
            if active == 1:
                hold_count += 1
                # 出场条件1: 价格回归通道中轴
                if np.isfinite(ch_mid) and close[i] >= ch_mid:
                    signals[i] = 2
                    active     = 0
                    hold_count = 0
                    continue
                # 出场条件2: 持仓超时强制平仓
                elif hold_count >= max_hold:
                    signals[i] = 2
                    active     = 0
                    hold_count = 0
                    continue
                else:
                    signals[i] = 0
                    continue

            elif active == -1:
                hold_count += 1
                # 出场条件1: 价格回归通道中轴
                if np.isfinite(ch_mid) and close[i] <= ch_mid:
                    signals[i] = 2
                    active     = 0
                    hold_count = 0
                    continue
                # 出场条件2: 持仓超时强制平仓
                elif hold_count >= max_hold:
                    signals[i] = 2
                    active     = 0
                    hold_count = 0
                    continue
                else:
                    signals[i] = 0
                    continue

            # ── 空仓: 检查入场信号 ────────────────────────────────────────────────
            if long_entry[i]:
                signals[i] = 1
                active      = 1
                hold_count  = 0
            elif short_entry[i]:
                signals[i] = -1
                active      = -1
                hold_count  = 0
            else:
                signals[i] = 0

        # ── NaN safety — 预热期清零 ───────────────────────────────────────────────
        # 通道需要 channel_period 根K线预热，额外留5根缓冲
        warmup = channel_period + 5
        signals[:warmup] = 0

        return signals

    def position_params(self) -> PositionParams:
        """
        返回仓位管理参数。

        均值回归策略目标利润较小，止损控制要严格:
          - hard_stop_pct=0.5:  硬止损0.5%，防止信号方向判断完全错误
          - trailing_pct=0.4:   移动止损0.4%，锁定已实现盈利
          - tp1_pct=0.4:        第一目标0.4% (减仓至半仓)
          - tp2_pct=0.8:        第二目标0.8% (清仓)
          - max_lots=1:         最大持仓1手，控制单策略风险敞口
        """
        return PositionParams(
            hard_stop_pct=0.5,
            trailing_pct=0.4,
            tp1_pct=0.4,
            tp2_pct=0.8,
            max_lots=1,
            unit=1,
        )
