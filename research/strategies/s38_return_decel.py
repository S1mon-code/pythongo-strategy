"""
Strategy #38 — Return Deceleration at Channel Extreme (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 价格处于滚动通道极端区间 + 下跌动能持续衰减（收益率减速）→ 均值回归入场

  设计思路:
  - 继承S11的通道相对位置方法 (Sharpe 1.02验证有效)
  - 新增"收益率减速"确认:
    1. 计算最近 decel_window 根K线的累积收益率（衡量当前趋势的总位移）
    2. 若累积收益率为负（处于下跌趋势）且绝对值超过 ret_threshold（确认真实跌幅）
    3. 但本期累积收益率 > 上期累积收益率（即下跌幅度在收窄 = 动能衰减）
    → "虽然仍在跌，但跌得越来越慢" = 卖方精疲力竭的可量化证据

  防前瞻说明:
  - ret = close.pct_change(1)    — 1根bar的价格变化率，使用当前bar已完成的收盘价
  - cum_ret = ret.rolling(decel_window).sum() — 用过去 decel_window 根完成K线
  - prev_cum_ret = cum_ret.shift(1)             — 与上一时间点比较，无前瞻
  - ch_high/ch_low 均使用 shift(1).rolling()    — 通道构建无前瞻

  "减速"的数学含义:
  - cum_ret[t]     = sum(ret[t-W+1 : t])    当前窗口的累积收益
  - cum_ret[t-1]   = sum(ret[t-W   : t-1])  前一根bar的累积收益
  - cum_ret[t] > cum_ret[t-1]（在下跌中）
    意味着: ret[t] > ret[t-W]（最新一根bar的跌幅小于W根前那根bar的跌幅）
    即：新增的跌幅 < 滚出窗口的旧跌幅 → 跌速放缓 = 动能衰减

  出场逻辑 (有状态):
  - 价格回归至通道中轨 (ch_low + 0.5 * ch_range) 时平仓
  - 在状态机中持续持有，直至回归或风控触发

  参数设计 (3个，27种组合，WFO可承受):
  - decel_window:   减速衡量窗口 [5, 8, 12]         — 用多少根bar衡量趋势的累积收益
  - ret_threshold:  最小趋势幅度 [0.002, 0.004, 0.006] — 小于此幅度不视为有效趋势
  - channel_period: 通道周期 [10, 20, 30]            — 约50min到2.5h的通道

  与S11/S30/S36/S37对比:
  - S11 使用形态（吞噬/Pin Bar/Inside Bar）确认单根K线级别的反转信号
  - S30 使用K线收盘位置确认当前K线内的买卖压力
  - S36 使用大实体阳线确认当前K线买入冲量
  - S37 使用跨bar的盘中吸收（开弱收强）确认
  - S38 使用多bar时间维度上的动能衰减确认，捕捉的是"趋势正在失去动力"这一更早期信号

  研究依据:
  - 铁矿石2023+均值回归特性已由3个胜出策略验证
  - S11通道位置方法已证明有效 (Sharpe 1.02)
  - 动量衰减（momentum deceleration）是量化策略中衡量趋势可持续性的经典方法
  - 5min周期: decel_window=5~12 对应约25~60分钟窗口，捕捉日内趋势的衰竭
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class ReturnDecelReversion(BaseResearchStrategy):
    name = "Return Decel Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "decel_window":   [5, 8, 12],                   # 累积收益率衡量窗口
            "ret_threshold":  [0.002, 0.004, 0.006],        # 最小趋势幅度
            "channel_period": [10, 20, 30],                 # 滚动通道周期
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        decel_window: int = 8,
        ret_threshold: float = 0.004,
        channel_period: int = 20,
    ) -> np.ndarray:
        """
        Generate mean-reversion signals using channel position + return deceleration.

        A declining move that is losing momentum (each successive bar contributes less
        to the cumulative drop than the bar it replaces in the rolling window) signals
        seller exhaustion. When this occurs at a channel extreme, it is high-probability
        evidence that the reversal has already started.

        Args:
            df:             OHLCV DataFrame.
            decel_window:   Bars over which to measure cumulative return trend (default 8).
            ret_threshold:  Minimum absolute cumulative return to qualify as "in a trend"
                            (default 0.004, i.e. 0.4% over the window). Filters noise.
            channel_period: Lookback bars for rolling channel (default 20).

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

        near_low  = ch_pos <= 0.20   # 通道底部20%区间 → 做多机会（固定，不作为参数避免与extreme_pct混淆）
        near_high = ch_pos >= 0.80   # 通道顶部20%区间 → 做空机会

        # ── 收益率减速指标 ────────────────────────────────────────────────────────
        # 1根bar的价格变化率（当前bar收盘/前一根收盘 - 1）
        ret = cs.pct_change(1)  # ret[i] = (close[i] - close[i-1]) / close[i-1]

        # 滚动窗口内累积收益率（衡量当前趋势的总强度）
        cum_ret = ret.rolling(decel_window).sum()

        # 上一时刻的累积收益率（用于比较是否减速）
        prev_cum_ret = cum_ret.shift(1)

        cum_ret_vals      = cum_ret.values
        prev_cum_ret_vals = prev_cum_ret.values

        # 下跌趋势中的减速:
        # - declining: 累积收益为负且绝对值超过阈值（确认处于真实跌势中）
        # - decelerating_down: 本期累积收益 > 上期（下跌在收窄 = 卖方精疲力竭）
        declining         = cum_ret_vals < -ret_threshold
        decelerating_down = (cum_ret_vals > prev_cum_ret_vals) & declining

        # 上涨趋势中的减速（做空信号）
        rising            = cum_ret_vals > ret_threshold
        decelerating_up   = (cum_ret_vals < prev_cum_ret_vals) & rising

        # ── 入场信号 ─────────────────────────────────────────────────────────────
        long_entry_arr  = near_low  & decelerating_down
        short_entry_arr = near_high & decelerating_up

        # ── 有状态出场: 价格回归通道中轨时平仓 ───────────────────────────────────
        signals = np.zeros(n, dtype=np.int8)
        active = 0  # 0=空仓, 1=多头, -1=空头

        for i in range(n):
            # NaN 安全: 通道或累积收益率尚未预热时跳过
            if (
                np.isnan(ch_low_arr[i])
                or np.isnan(ch_high_arr[i])
                or ch_range_arr[i] <= 0
                or np.isnan(cum_ret_vals[i])
                or np.isnan(prev_cum_ret_vals[i])
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
        # 预热期 = max(channel_period, decel_window) + 5（两个滚动窗口各自的预热期）
        warmup = max(channel_period, decel_window) + 5
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
