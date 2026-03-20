"""
Strategy #11 — Candle Pattern Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 价格处于滚动通道的极端区间 + 裸K反转形态 → 均值回归入场

  设计改进 (v3):
  - 降至5分钟: 信号数量更充足，与VwapZscore/SessionGap同频
  - 极值检测改为"通道相对位置": 价格处于N根K线通道的上/下X%区间
    即 channel_position = (close - channel_low) / (channel_high - channel_low)
    ≥ 1 - extreme_pct → 超买区 (做空)
    ≤ extreme_pct     → 超卖区 (做多)
    比固定百分比距离更自适应，信号数量合理

  三种反转形态 (同上):
  1. 吞噬形态 (Engulfing)   — 实体完全吞噬前根，反转信号最强
  2. Pin Bar (锤子/射击之星) — 长影线被弹回，拒绝信号明确
  3. Inside Bar 突破         — 压缩后向均值方向突破

  信号逻辑:
  - 做多: 价格在通道底部极端区 + 任意1个看涨形态
  - 做空: 价格在通道顶部极端区 + 任意1个看跌形态

  参数设计 (3个，27种组合，WFO可承受):
  - channel_period: 通道周期 [10,20,30]   — 约50min到2.5h的通道
  - extreme_pct:    极端区间比例 [0.15,0.20,0.25] — 通道两端多少%视为极端
  - pin_ratio:      Pin Bar影线/实体比 [2.0,2.5,3.0]

  研究依据:
  - 铁矿石2023+均值回归特性已由3个胜出策略验证
  - 5min周期: VwapZscore/SessionGap均有效，信号充足
  - 通道位置式极值检测: 自适应波动，不依赖固定%距离
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class CandlePatternMeanReversion(BaseResearchStrategy):
    name = "Candle Pattern Mean-Reversion"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "channel_period": [10, 20, 30],        # 滚动通道周期
            "extreme_pct":    [0.15, 0.20, 0.25],  # 极端区间比例
            "pin_ratio":      [2.0, 2.5, 3.0],     # Pin Bar 影线/实体比
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        channel_period: int = 20,
        extreme_pct: float = 0.20,
        pin_ratio: float = 2.5,
    ) -> np.ndarray:
        close = df["close"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        n = len(close)

        hs = pd.Series(high)
        ls = pd.Series(low)
        cs = pd.Series(close)

        # ── 通道相对位置 (用shift(1)防前瞻) ──────────────────────────────────────
        ch_high = hs.shift(1).rolling(channel_period).max().values
        ch_low  = ls.shift(1).rolling(channel_period).min().values
        ch_range = ch_high - ch_low

        with np.errstate(divide="ignore", invalid="ignore"):
            ch_pos = np.where(ch_range > 0, (close - ch_low) / ch_range, 0.5)

        # 超卖区 (通道底部 extreme_pct 以内) → 做多机会
        near_low  = ch_pos <= extreme_pct
        # 超买区 (通道顶部 extreme_pct 以内) → 做空机会
        near_high = ch_pos >= (1.0 - extreme_pct)

        # ── 形态1: 吞噬形态 ──────────────────────────────────────────────────────
        prev_open  = np.empty(n); prev_open[0]  = np.nan; prev_open[1:]  = open_[:-1]
        prev_close = np.empty(n); prev_close[0] = np.nan; prev_close[1:] = close[:-1]

        bull_engulf = (
            (prev_close < prev_open) & (close > open_)
            & (open_ <= prev_close) & (close >= prev_open)
        )
        bear_engulf = (
            (prev_close > prev_open) & (close < open_)
            & (open_ >= prev_close) & (close <= prev_open)
        )

        engulf_score = np.where(bull_engulf, 1, np.where(bear_engulf, -1, 0))

        # ── 形态2: Pin Bar ────────────────────────────────────────────────────────
        body         = np.abs(close - open_)
        candle_range = high - low
        lower_shadow = np.minimum(open_, close) - low
        upper_shadow = high - np.maximum(open_, close)
        body_pos     = np.where(body > 0, body, np.nan)

        hammer        = (lower_shadow >= pin_ratio * body_pos) & (upper_shadow <= body_pos) & (body <= 0.33 * candle_range)
        shooting_star = (upper_shadow >= pin_ratio * body_pos) & (lower_shadow <= body_pos) & (body <= 0.33 * candle_range)

        pinbar_score = np.nan_to_num(
            np.where(hammer, 1, np.where(shooting_star, -1, 0)).astype(float), nan=0.0
        ).astype(int)

        # ── 形态3: Inside Bar 突破 ────────────────────────────────────────────────
        inside_bar  = (hs < hs.shift(1)) & (ls > ls.shift(1))
        prev_inside = inside_bar.shift(1).fillna(False)
        mother_high = hs.shift(2)
        mother_low  = ls.shift(2)

        inside_bull = prev_inside & (cs > mother_high)
        inside_bear = prev_inside & (cs < mother_low)

        insidebar_score = np.where(inside_bull.values, 1, np.where(inside_bear.values, -1, 0))

        # ── 信号合成 ──────────────────────────────────────────────────────────────
        pattern_score = engulf_score + pinbar_score + insidebar_score

        signals = np.zeros(n, dtype=np.int8)
        signals[near_low  & (pattern_score >= 1)] =  1
        signals[near_high & (pattern_score <= -1)] = -1

        # NaN safety — 通道预热期
        signals[:channel_period + 2] = 0

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(
            hard_stop_pct=0.5,
            trailing_pct=0.4,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
            unit=1,
        )


# 别名，保持 run_all.py 注册兼容
CandlePatternTrend = CandlePatternMeanReversion
