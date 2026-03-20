"""
Strategy #26 — Channel Position + Volume Surge Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 价格处于滚动通道极端区间 + 成交量异常放大 → 均值回归入场

  设计思路 (S26 — S11扩展版):
  - 继承S11 (CandlePatternMeanReversion, Sharpe 1.02) 的通道位置检测逻辑
  - 将K线形态确认替换为"成交量激增"过滤器
  - 核心假设: 极端区间 + 成交量激增 = 高成本方向性追涨/追跌，往往意味着
    方向性资金耗尽，随后价格向通道中轴回归

  成交量激增的市场含义:
  - 通道底部高量: 恐慌性抛盘 → 空头力量衰竭 → 反弹
  - 通道顶部高量: 狂热性追涨 → 多头力量衰竭 → 回落
  研究支持: "成交量在支撑/阻力位的异常放大是高置信度反转信号"
            (Pacific-Basin Finance Journal 2024，中国商品期货市场研究)

  通道位置计算 (完全沿用S11，已验证):
    ch_high = rolling(channel_period).max  对前N根K线最高价
    ch_low  = rolling(channel_period).min  对前N根K线最低价
    ch_range = ch_high - ch_low
    ch_pos  = (close - ch_low) / ch_range  ∈ [0, 1]
    near_low  = ch_pos ≤ extreme_pct       → 超卖极端区
    near_high = ch_pos ≥ 1 - extreme_pct  → 超买极端区

  成交量激增过滤器:
    vol_ma = rolling(vol_window=20).mean  对前N根K线成交量 (shift(1)防前瞻)
    vol_surge = 当前成交量 > vol_mult × vol_ma

  信号合成:
    做多入场:  near_low  & vol_surge  → 超卖+爆量 → 空头耗尽
    做空入场:  near_high & vol_surge  → 超买+爆量 → 多头耗尽

  出场逻辑 (有状态):
    持多仓时: close ≥ 通道中轴 (ch_low + 0.5 * ch_range) → 平仓
    持空仓时: close ≤ 通道中轴 (ch_low + 0.5 * ch_range) → 平仓

  参数设计 (3个，27种组合，WFO可承受):
    - channel_period: 通道周期 [10, 20, 30]          约50min ~ 2.5h通道
    - extreme_pct:    极端区间比例 [0.15, 0.20, 0.25] 通道两端多少%视为极端
    - vol_mult:       成交量倍数阈值 [1.2, 1.5, 2.0]  当前量高于均值多少倍算激增
    - vol_window:     固定为20根，与通道中间量级匹配

  NaN/缺失处理:
    - volume列不存在或全为零: vol_surge 全置 False，策略退化为纯通道位置信号
    - vol_ma 为零或 NaN: 该bar的 vol_surge 置 False
    - ch_range = 0 或 ch_low = NaN: 该bar的信号置 0
    - 预热期 (前 max(channel_period, vol_window) + 5 根): 信号全置 0
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams

# vol_window is fixed — not part of the optimisation grid
_VOL_WINDOW: int = 20


class ChannelVolumeReversion(BaseResearchStrategy):
    name = "Channel-Volume Mean-Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "channel_period": [10, 20, 30],         # 滚动通道周期
            "extreme_pct":    [0.15, 0.20, 0.25],   # 极端区间比例
            "vol_mult":       [1.2, 1.5, 2.0],      # 成交量激增倍数阈值
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        channel_period: int = 20,
        extreme_pct: float = 0.20,
        vol_mult: float = 1.5,
    ) -> np.ndarray:
        """
        生成交易信号。

        Args:
            df:             包含 open/high/low/close/volume 列的 OHLCV DataFrame。
            channel_period: 滚动通道计算周期 (默认20根K线)。
            extreme_pct:    通道极端区间比例 (默认0.20 = 两端各20%)。
            vol_mult:       成交量激增倍数阈值; 当前量 > vol_mult × vol_ma 才算激增
                            (默认1.5，即超过均量的1.5倍)。

        Returns:
            np.ndarray (int8):
                 1  = 做多信号 (超卖 + 爆量)
                -1  = 做空信号 (超买 + 爆量)
                 0  = 无信号 / 持仓中继续持有
                 2  = 强制平仓 (价格到达通道中轴)
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

        # ── 成交量激增过滤器 ──────────────────────────────────────────────────────
        # 当 volume 列缺失或全零时，降级为 False (不过滤，退化为纯通道信号)
        if "volume" not in df.columns:
            vol_surge = np.zeros(n, dtype=bool)
        else:
            vol_raw = df["volume"].values.astype(np.float64)

            # vol_ma: 用前N根K线均量 (shift(1) 防前瞻)，固定窗口 _VOL_WINDOW
            vol_ma = (
                pd.Series(vol_raw)
                .shift(1)
                .rolling(_VOL_WINDOW)
                .mean()
                .values
            )

            # vol_ma 为 0 或 NaN 时，该 bar 不满足激增条件
            safe_ma = np.where(
                np.isfinite(vol_ma) & (vol_ma > 0),
                vol_ma,
                np.inf,   # 除以 inf → 当前量/inf = 0，永不满足激增
            )

            # 当前 bar 成交量使用原始值 (无 shift)，以当前 bar 成交量作为确认
            vol_surge = (vol_raw > vol_mult * safe_ma)

        # ── 信号合成 (非有状态: 纯入场触发标志) ──────────────────────────────────
        long_entry  = near_low  & vol_surge   # 超卖 + 爆量 = 空头耗尽
        short_entry = near_high & vol_surge   # 超买 + 爆量 = 多头耗尽

        # ── 有状态循环: 持仓跟踪 + 通道中轴出场 ─────────────────────────────────
        #   active:  1 = 持多, -1 = 持空, 0 = 空仓
        #   signals: 1=入多, -1=入空, 2=平仓, 0=持续/空仓
        signals = np.zeros(n, dtype=np.int8)

        long_entry_arr  = long_entry
        short_entry_arr = short_entry
        ch_low_arr      = ch_low
        ch_range_arr    = ch_range

        active: int = 0

        for i in range(n):
            # 通道中轴: ch_low + 50% × ch_range
            # 若 ch_range 为 NaN 或 0，ch_mid 也为 NaN → 出场条件不满足 (安全)
            if ch_range_arr[i] > 0 and np.isfinite(ch_low_arr[i]):
                ch_mid = ch_low_arr[i] + 0.5 * ch_range_arr[i]
            else:
                ch_mid = np.nan

            # 先检查出场条件 (已持仓)
            if active == 1:
                if np.isfinite(ch_mid) and close[i] >= ch_mid:
                    signals[i] = 2
                    active = 0
                    continue
                else:
                    # 持仓中，无新动作
                    signals[i] = 0
                    continue

            elif active == -1:
                if np.isfinite(ch_mid) and close[i] <= ch_mid:
                    signals[i] = 2
                    active = 0
                    continue
                else:
                    signals[i] = 0
                    continue

            # 空仓: 检查入场信号
            if long_entry_arr[i]:
                signals[i] = 1
                active = 1
            elif short_entry_arr[i]:
                signals[i] = -1
                active = -1
            else:
                signals[i] = 0

        # ── NaN safety — 预热期清零 ───────────────────────────────────────────────
        # 前 max(channel_period, vol_window) + 5 根 bar 不产生有效信号
        warmup = max(channel_period, _VOL_WINDOW) + 5
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
