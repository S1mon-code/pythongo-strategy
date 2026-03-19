"""
Strategy #6 — Opening Range Breakout (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 开盘区间突破 + 成交量确认

  开盘后一段时间的高低点形成的区间，代表了多空双方的初始博弈结果。
  当价格带量突破这个区间时，意味着一方取得优势，趋势大概率延续。

  DCE铁矿石日盘从09:00开盘。

  信号生成:
  - 开盘区间: 日盘前 opening_min 分钟的最高价和最低价
  - 对于5分钟K线: opening_min/5 根K线
  - 量能确认: 突破bar成交量 > vol_mult × 开盘区间均量
  - 做多: 收盘价突破区间上沿 且 放量确认
  - 做空: 收盘价跌破区间下沿 且 放量确认
  - 每日每个方向仅一次信号

  参数设计 (2个):
  - opening_min: 开盘区间时长 [15,30,45,60] — 3到12根5分钟K线
  - vol_mult: 量能乘数 [1.2,1.5,2.0,2.5] — 放量确认阈值

  适用环境: 有隔夜消息驱动的方向性行情
  风险提示: 假突破频繁，尤其在无重大消息的交易日

  回测表现:
  - 训练集 (2013-2022, 无止损): Sharpe 0.65 | PF 1.44 | 664笔交易
  - 测试集 (2023-2026): Sharpe -0.86 — 假突破侵蚀严重
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class OpeningRangeBreakout(BaseResearchStrategy):
    name = "Opening Range Breakout"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "opening_min": [15, 30, 45, 60],
            "vol_mult": [1.2, 1.5, 2.0, 2.5],
        }

    def generate_signals(
        self, df: pd.DataFrame, opening_min: int = 30, vol_mult: float = 1.5
    ) -> np.ndarray:
        signals = np.zeros(len(df), dtype=int)
        n_bars = max(1, opening_min // 5)  # number of opening-range bars

        # Derive trading day from index (date portion)
        # DCE day session starts at 09:00; night session belongs to next tday.
        # Use 'tday' column if available, otherwise derive from index.
        if "tday" in df.columns:
            tday = df["tday"].values
        else:
            idx = df.index
            hour = idx.hour
            # Night session (21:00-23:00, 00:00-02:30) belongs to next calendar day
            dates = idx.date
            tday = np.array(dates, dtype="datetime64[D]")
            # Shift: bars before 09:00 belong to the previous tday assignment
            # but for opening range we only care about day session (>= 09:00)
            # so tday grouping by calendar date is fine for day-session logic.

        # Unique trading days
        unique_days = np.unique(tday)

        for day in unique_days:
            day_mask = tday == day
            day_idx = np.where(day_mask)[0]
            if len(day_idx) == 0:
                continue

            # Filter to day-session bars only (hour >= 9 and hour < 15:15)
            day_session_idx = []
            for i in day_idx:
                h = df.index[i].hour
                m = df.index[i].minute
                # DCE day session: 09:00 - 11:30, 13:30 - 15:00
                if 9 <= h < 16:
                    day_session_idx.append(i)

            if len(day_session_idx) < n_bars + 1:
                continue  # not enough bars to form opening range + trade

            day_session_idx = np.array(day_session_idx)

            # Opening range: first n_bars of the day session
            or_idx = day_session_idx[:n_bars]
            or_high = df["high"].iloc[or_idx].max()
            or_low = df["low"].iloc[or_idx].min()
            or_avg_vol = df["volume"].iloc[or_idx].mean()

            # Skip if opening range values are invalid
            if np.isnan(or_high) or np.isnan(or_low) or np.isnan(or_avg_vol):
                continue
            if or_avg_vol <= 0:
                continue

            # Scan bars after the opening range
            post_or_idx = day_session_idx[n_bars:]
            long_triggered = False
            short_triggered = False

            for i in post_or_idx:
                close_val = df["close"].iat[i]
                vol_val = df["volume"].iat[i]

                if np.isnan(close_val) or np.isnan(vol_val):
                    continue

                vol_ok = vol_val > vol_mult * or_avg_vol

                if not long_triggered and close_val > or_high and vol_ok:
                    signals[i] = 1
                    long_triggered = True

                if not short_triggered and close_val < or_low and vol_ok:
                    signals[i] = -1
                    short_triggered = True

                if long_triggered and short_triggered:
                    break

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(hard_stop_pct=1.0, trailing_pct=0.7, tp1_pct=0.7, tp2_pct=1.5, max_lots=3)
