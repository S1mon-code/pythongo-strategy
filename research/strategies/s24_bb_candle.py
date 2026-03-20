"""
Strategy #24 — Bollinger Band + Candle Pattern Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 布林带极端触及 + 裸K反转形态双重确认 → 均值回归入场

  设计思路:
  - 本策略是 S13 (布林带极值检测) 与 S11 (裸K反转形态) 的组合策略。两者已在
    铁矿石2023+数据上分别取得 Sharpe > 1.0。两个机制同时触发意味着:
      1. 价格已偏离统计意义上的正常区间 (BB下轨/上轨之外)
      2. 当前K线形态本身已出现反转迹象
    两者交集显著压缩低质量入场，预期胜率更高、假信号更少。

  布林带设置 (与 S13 保持一致 — 已验证有效):
  ──────────────────────────────────────────────
    cs       = pd.Series(close)
    mid_bb   = cs.shift(1).rolling(bb_period).mean()    — 防止未来函数
    std_bb   = cs.shift(1).rolling(bb_period).std()
    upper_bb = mid_bb + 2.0 × std_bb                    — bb_std 固定为 2.0
    lower_bb = mid_bb - 2.0 × std_bb

  裸K反转形态 (与 S11 完全一致 — 三种形态，各打分):
  ──────────────────────────────────────────────────
  1. 吞噬形态 (Engulfing):
       看涨: 前根阴线 + 当前阳线 + 当前实体完全吞噬前根实体
       看跌: 前根阳线 + 当前阴线 + 当前实体完全吞噬前根实体
       得分: +1 / -1 / 0

  2. Pin Bar (锤子 / 射击之星):
       锤子:     下影线 ≥ pin_ratio × 实体 且 上影线 ≤ 实体 且 实体 ≤ 0.33 × K线总幅
       射击之星: 上影线 ≥ pin_ratio × 实体 且 下影线 ≤ 实体 且 实体 ≤ 0.33 × K线总幅
       得分: +1 / -1 / 0  (NaN安全处理实体为0的十字星)

  3. Inside Bar 突破:
       inside_bar = 当前高低点完全在前根高低点范围内
       看涨突破: 前一根为 inside bar 且当前 close > 母根 (shift(2)) 最高价
       看跌突破: 前一根为 inside bar 且当前 close < 母根 (shift(2)) 最低价
       得分: +1 / -1 / 0

  总形态分: pattern_score = engulf_score + pinbar_score + insidebar_score

  信号逻辑:
  ──────────
    做多 (+1): close < lower_bb (前根) AND pattern_score >= min_score
    做空 (-1): close > upper_bb (前根) AND pattern_score <= -min_score
    出场 ( 2): 持多且 close >= mid_bb；持空且 close <= mid_bb

  有状态循环处理出场与持仓维持，确保出场信号优先于新入场信号。

  热身期: 前 bb_period + 5 根bar强制置零，确保布林带指标有效。

  参数设计 (3个，27种组合):
  - bb_period : 布林带回看周期  [10, 20, 30]  — 约50min到2.5h的波动估计
  - pin_ratio : Pin Bar影线/实体比 [2.0, 2.5, 3.0]
  - min_score : 最低形态确认分  [1, 2, 3]   — 1=任意单一形态，3=三种全部确认

  研究依据:
  - S13 (SemivarianceBB) 与 S11 (CandlePatternMeanReversion) 均为铁矿石CTA
    已验证胜出策略，各自 Sharpe > 1.0；
  - 两种不同维度的信号 (统计偏离 + 价格形态) 同时成立时，假信号概率乘法式下降；
  - min_score 参数允许从"宽松确认 (任意1种形态)" 到"严格确认 (全部3种形态)"
    的连续调优，WFO 可自动选取最优约束强度。
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class BBCandleReversion(BaseResearchStrategy):
    """布林带 + 裸K形态均值回归策略 — S24。

    当价格突破布林带上/下轨 (bb_std 固定为2.0)，同时出现满足最低确认分数的
    裸K反转形态时入场做均值回归；价格回归至中轨时强制出场。

    形态分由三种形态 (吞噬、Pin Bar、Inside Bar突破) 各自计分后求和，
    min_score 控制进入所需的最低总分，范围 [1, 3]。
    """

    name = "BB-Candle Mean-Reversion"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回 3 参数网格，共 3×3×3 = 27 个参数组合。

        bb_period : int
            布林带回看周期。10 → 较短、对近期波动敏感；30 → 较长、带宽更平稳。
            bb_std 固定为 2.0，不参与网格搜索 (与 S13 保持一致)。
        pin_ratio : float
            Pin Bar 判定的影线/实体比阈值。
            2.0 = 较宽松 (影线2倍实体即可)；3.0 = 较严格 (影线需达3倍实体)。
        min_score : int
            触发入场所需的最低形态总分。
            1 = 最宽松 (三种形态中任意一种看涨/看跌即可)；
            2 = 中等   (至少两种形态同向确认)；
            3 = 最严格 (三种形态全部同向确认，信号极少但质量最高)。
        """
        return {
            "bb_period": [10, 20, 30],      # 布林带回看周期
            "pin_ratio": [2.0, 2.5, 3.0],  # Pin Bar 影线/实体比
            "min_score": [1, 2, 3],         # 最低形态确认分
        }

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        bb_period: int = 20,
        pin_ratio: float = 2.5,
        min_score: int = 1,
    ) -> np.ndarray:
        """生成布林带 + 裸K形态均值回归信号。

        实现分两步:
        1. 向量化预计算布林带、三种形态得分、入场候选 (long_ok/short_ok) 及
           中轨数组 (用于出场判断)。
        2. 有状态循环处理出场 (signal == 2) 与持仓维持逻辑。

        Parameters
        ----------
        df : pd.DataFrame
            含 DatetimeIndex 的 OHLCV DataFrame，至少包含 open/high/low/close 列。
        bb_period : int
            布林带回看周期；std 乘数固定为 2.0。
        pin_ratio : float
            Pin Bar 判定阈值：影线长度需达实体长度的 pin_ratio 倍。
        min_score : int
            触发入场所需的最低形态总分 (1/2/3)。

        Returns
        -------
        np.ndarray of int8
            与 df 等长的信号数组:
              +1  做多 (入场或持有)
              -1  做空 (入场或持有)
               0  空仓 / 无信号
               2  强制出场 (价格回归至中轨)
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        # ------------------------------------------------------------------
        # 提取 OHLC 序列
        # ------------------------------------------------------------------
        close  = df["close"].values.astype(np.float64)
        open_  = df["open"].values.astype(np.float64)
        high   = df["high"].values.astype(np.float64)
        low    = df["low"].values.astype(np.float64)

        cs = pd.Series(close)
        hs = pd.Series(high)
        ls = pd.Series(low)

        # ------------------------------------------------------------------
        # 布林带 (shift(1) 防止未来函数 — 与 S13 完全一致)
        # ------------------------------------------------------------------
        # 使用前根已完成的 close 序列计算均值与标准差，避免当前bar信息泄漏
        mid_bb_s = cs.shift(1).rolling(bb_period).mean()
        std_bb_s = cs.shift(1).rolling(bb_period).std()
        upper_bb = mid_bb_s + 2.0 * std_bb_s
        lower_bb = mid_bb_s - 2.0 * std_bb_s

        mid_bb_arr  = mid_bb_s.values   # numpy array，供循环出场判断
        upper_arr   = upper_bb.values
        lower_arr   = lower_bb.values

        # ------------------------------------------------------------------
        # 形态1: 吞噬形态 (Engulfing)
        # ------------------------------------------------------------------
        prev_open  = np.empty(n, dtype=np.float64)
        prev_close = np.empty(n, dtype=np.float64)
        prev_open[0]  = np.nan
        prev_close[0] = np.nan
        prev_open[1:]  = open_[:-1]
        prev_close[1:] = close[:-1]

        bull_engulf = (
            (prev_close < prev_open)    # 前根为阴线
            & (close > open_)           # 当前为阳线
            & (open_ <= prev_close)     # 当前开盘 ≤ 前根收盘
            & (close >= prev_open)      # 当前收盘 ≥ 前根开盘
        )
        bear_engulf = (
            (prev_close > prev_open)    # 前根为阳线
            & (close < open_)           # 当前为阴线
            & (open_ >= prev_close)     # 当前开盘 ≥ 前根收盘
            & (close <= prev_open)      # 当前收盘 ≤ 前根开盘
        )

        engulf_score = np.where(bull_engulf, 1, np.where(bear_engulf, -1, 0))

        # ------------------------------------------------------------------
        # 形态2: Pin Bar (锤子 / 射击之星)  — NaN安全处理零实体十字星
        # ------------------------------------------------------------------
        body         = np.abs(close - open_)
        candle_range = high - low
        lower_shadow = np.minimum(open_, close) - low
        upper_shadow = high - np.maximum(open_, close)

        # body_pos: 实体为0时置 NaN，使乘法比较结果也为 NaN → 最终 nan_to_num 归零
        body_pos = np.where(body > 0.0, body, np.nan)

        hammer = (
            (lower_shadow >= pin_ratio * body_pos)   # 下影线 ≥ 阈值
            & (upper_shadow <= body_pos)             # 上影线较短
            & (body <= 0.33 * candle_range)          # 小实体
        )
        shooting_star = (
            (upper_shadow >= pin_ratio * body_pos)   # 上影线 ≥ 阈值
            & (lower_shadow <= body_pos)             # 下影线较短
            & (body <= 0.33 * candle_range)          # 小实体
        )

        # np.where 在 NaN 比较时返回 0 (False)，但显式 nan_to_num 确保安全
        pinbar_score = np.nan_to_num(
            np.where(hammer, 1, np.where(shooting_star, -1, 0)).astype(np.float64),
            nan=0.0,
        ).astype(np.int8)

        # ------------------------------------------------------------------
        # 形态3: Inside Bar 突破
        # ------------------------------------------------------------------
        inside_bar  = (hs < hs.shift(1)) & (ls > ls.shift(1))
        prev_inside = inside_bar.shift(1).fillna(False)
        mother_high = hs.shift(2)
        mother_low  = ls.shift(2)

        inside_bull = prev_inside & (cs > mother_high)
        inside_bear = prev_inside & (cs < mother_low)

        insidebar_score = np.where(
            inside_bull.values, 1, np.where(inside_bear.values, -1, 0)
        )

        # ------------------------------------------------------------------
        # 总形态分
        # ------------------------------------------------------------------
        pattern_score = (
            engulf_score.astype(np.int16)
            + pinbar_score.astype(np.int16)
            + insidebar_score.astype(np.int16)
        )

        # ------------------------------------------------------------------
        # 入场候选 (向量化)
        # ------------------------------------------------------------------
        # 做多: 当前收盘低于前根下轨 AND 形态总分 ≥ min_score
        long_ok  = (close < lower_arr) & (pattern_score >= min_score)
        # 做空: 当前收盘高于前根上轨 AND 形态总分 ≤ -min_score
        short_ok = (close > upper_arr) & (pattern_score <= -min_score)

        # ------------------------------------------------------------------
        # NaN 安全: 任意关键指标为 NaN 时禁止入场
        # ------------------------------------------------------------------
        nan_mask = (
            np.isnan(mid_bb_arr)
            | np.isnan(upper_arr)
            | np.isnan(lower_arr)
            | np.isnan(close)
            | np.isnan(prev_open)    # 首bar无前根数据
            | np.isnan(prev_close)
        )
        long_ok[nan_mask]  = False
        short_ok[nan_mask] = False

        # ------------------------------------------------------------------
        # 热身期: 前 bb_period + 5 根bar不产生信号
        # ------------------------------------------------------------------
        warmup = bb_period + 5
        long_ok[:warmup]  = False
        short_ok[:warmup] = False

        # ------------------------------------------------------------------
        # 有状态循环: 出场 (signal=2) + 持仓维持
        # ------------------------------------------------------------------
        # 信号优先级: 出场 > 入场 > 空仓
        #   出场 (2): 持多且 close >= mid_bb → 价格回归至中轨，平多
        #             持空且 close <= mid_bb → 价格回归至中轨，平空
        #   入场 (+1/-1): long_ok/short_ok 为 True 且当前空仓
        #   持仓: 每根bar持续写入方向信号 (+1/-1)，直至出场
        active = 0  # 当前仓位方向: +1 做多, -1 做空, 0 空仓

        for i in range(n):
            mid_ref = mid_bb_arr[i]  # 当前bar对应的中轨 (出场参考)

            # ---- 持多: 检查是否回归至中轨 ------------------------------------
            if active == 1:
                if not np.isnan(mid_ref) and close[i] >= mid_ref:
                    signals[i] = 2   # 价格回归中轨，强制平多
                    active = 0
                else:
                    signals[i] = 1   # 维持多头
                continue

            # ---- 持空: 检查是否回归至中轨 ------------------------------------
            if active == -1:
                if not np.isnan(mid_ref) and close[i] <= mid_ref:
                    signals[i] = 2   # 价格回归中轨，强制平空
                    active = 0
                else:
                    signals[i] = -1  # 维持空头
                continue

            # ---- 空仓: 检查入场候选 ------------------------------------------
            if long_ok[i]:
                signals[i] = 1
                active = 1
            elif short_ok[i]:
                signals[i] = -1
                active = -1
            # 否则: 空仓无信号，signals[i] 保持 0

        return signals

    # ------------------------------------------------------------------
    # 持仓参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """返回适合双重确认均值回归策略的风控参数。

        参数设计理由:
        - hard_stop_pct=0.6 : 布林带外侧的假突破有时会延伸，0.6% 硬止损在
                              容忍小幅超调的同时防止趋势行情大幅损失；略高于
                              bb_std=2.0 对应的统计区间外沿。
        - trailing_pct=0.5  : 价格回归中轨过程中，0.5% 移动止损锁定已实现收益，
                              防止回归行情二次反转导致盈利回吐。
        - tp1_pct=0.5       : 半仓止盈 0.5%，在价格回归初期锁定部分收益，
                              降低剩余仓位的风险敞口；匹配铁矿石5分钟K线均幅。
        - tp2_pct=1.0       : 余仓止盈 1.0%，对应完整均值回归 (从下轨/上轨至中轨)
                              的典型收益区间，捕获最大回归幅度。
        - max_lots=1        : 研究阶段保守单手，聚焦信号质量评估。
        """
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
