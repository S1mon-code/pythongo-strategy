"""
Strategy #33 — First Reversal Bar at Channel Extreme (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 价格处于滚动通道极端区间 + 前根K线确认趋势方向 + 当前K线开始反转
           (收盘高于前根收盘) → 均值回归入场。捕捉第一根反转K线，比吞噬形态
           入场更早，代价是信号精度略低。

  "第一反转K线"定义:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 做多: 价格在通道底部极端区                                           │
  │       + 前根K线为阴线 (prev_close < prev_open，下跌趋势确认)         │
  │       + 当前K线收盘 > 前根收盘 (反转开始，首次收复前根跌幅)          │
  │ 做空: 价格在通道顶部极端区                                           │
  │       + 前根K线为阳线 (prev_close > prev_open，上涨趋势确认)         │
  │       + 当前K线收盘 < 前根收盘 (反转开始，首次下破前根涨幅)          │
  └─────────────────────────────────────────────────────────────────────┘

  与 S11 (Candle Pattern) 的对比:
  - S11 要求吞噬/Pin Bar/Inside Bar 等明确形态，信号精确但延迟一根K线
  - S33 仅需"当前收盘反转前根收盘"，信号在当根立即确认，入场时机更早
  - S33 虚假信号略多（未完全反转的早期摸底），故配合通道位置作双重过滤

  通道相对位置计算 (防前瞻):
    ch_high = shift(1).rolling(channel_period).max(high)
    ch_low  = shift(1).rolling(channel_period).min(low)
    ch_range = ch_high - ch_low
    ch_pos   = (close - ch_low) / ch_range  (ch_range=0 时取 0.5)
    near_low  : ch_pos ≤ extreme_pct         (通道底部极端区 → 超卖)
    near_high : ch_pos ≥ 1 - extreme_pct    (通道顶部极端区 → 超买)

  前根K线数据 (防前瞻):
    prev_close = close.shift(1)   ← 信号计算时前根K线已完成，合规
    prev_open  = open.shift(1)    ← 同上

  出场逻辑 (有状态):
    当通道相对位置达到 exit_target 时平多 (价格从底部回归)
    当通道相对位置降至 1 - exit_target 时平空 (价格从顶部下落)
    exit_target 默认 0.50 = 通道中轴

  NaN 安全:
    - ch_range = 0 时 ch_pos 取 0.5，不满足极端区条件，不产生信号
    - 前 channel_period + 5 根 bar (预热期) 信号全置 0
    - 有状态循环中通道 NaN 时重置持仓状态

  参数设计 (3个，27种组合，WFO 可承受):
    - channel_period: 通道周期 [10, 20, 30]         约50min到2.5h的通道
    - extreme_pct:    极端区间比例 [0.15, 0.20, 0.25] 通道两端多少%视为极端
    - exit_target:    出场通道位置阈值 [0.40, 0.50, 0.60] 做多时 ch_pos ≥ 此值出场

  适用环境: 铁矿石5分钟震荡行情中，通道极值处的第一根反转K线具有均值回归拉力
  风险提示: 强趋势延续行情中"第一反转K线"可能成为假反转，硬止损严格保护

  回测参数:
    - 数据频率: 5分钟K线
    - hard_stop_pct : 0.5%  — 假反转延续亏损时快速止损
    - trailing_pct  : 0.4%  — 保护早期入场浮盈
    - tp1_pct       : 0.4%  — 半仓止盈，反转初段锁定收益
    - tp2_pct       : 0.8%  — 余仓止盈，捕获反转延伸段
    - max_lots      : 1     — 研究阶段保守单手
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class FirstReversalBar(BaseResearchStrategy):
    """通道极值处第一反转K线均值回归策略 — S33。

    在价格处于滚动通道底部/顶部极端区间时，检测前根K线是否确认原趋势
    (阴线/阳线)，以及当前K线收盘是否开始反转，两个条件同时满足则入场。
    出场以通道相对位置 exit_target 为目标，价格回归至中轴区域时平仓。
    """

    name = "First Reversal Bar"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回 3 参数网格，共 3×3×3 = 27 个参数组合。

        channel_period : int
            滚动通道的回看周期。10 → 约50分钟短通道 (信号频率高，对近期价格
            极值更敏感)；30 → 约2.5小时长通道 (极值定义更严格，信号更可靠)。
        extreme_pct : float
            通道两端判定为极端区间的比例。0.15 → 通道宽度的15%以内视为极端
            (信号较少，精确度高)；0.25 → 通道宽度的25%以内 (信号更多，容忍
            价格离通道极值稍远时入场)。
        exit_target : float
            出场时要求通道相对位置达到的阈值。0.40 → 价格仅需回到通道40%
            位置即出场 (早出场，止盈较保守)；0.60 → 需回到60%位置 (晚出场，
            争取更大回归幅度，但风险敞口时间更长)。
        """
        return {
            "channel_period": [10, 20, 30],
            "extreme_pct":    [0.15, 0.20, 0.25],
            "exit_target":    [0.40, 0.50, 0.60],
        }

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        channel_period: int = 20,
        extreme_pct: float = 0.20,
        exit_target: float = 0.50,
    ) -> np.ndarray:
        """生成基于通道极值 + 第一反转K线的均值回归信号。

        实现分四步:
        1. 计算滚动通道高低点与通道相对位置，确定超卖/超买区。
        2. 使用 shift(1) 获取前根K线 open/close，检测前根趋势方向与当根反转。
        3. 向量化预计算入场候选。
        4. 有状态循环: 入场后持续持仓，通道位置达到 exit_target 时出场。

        Parameters
        ----------
        df : pd.DataFrame
            含 'open', 'high', 'low', 'close', 'volume' 列的 OHLCV DataFrame，
            index 为时间序列。
        channel_period : int
            滚动通道的回看周期 (默认 20 根 bar，约 100 分钟)。
        extreme_pct : float
            通道端部极端区间比例 (默认 0.20，即通道宽度的 20% 以内)。
        exit_target : float
            出场通道位置阈值 (默认 0.50 = 通道中轴)。做多持仓时 ch_pos >=
            exit_target 触发出场；做空持仓时 ch_pos <= (1 - exit_target) 触发。

        Returns
        -------
        np.ndarray of int8
            与 df 等长的信号数组:
               1  = 做多入场 / 维持多头
              -1  = 做空入场 / 维持空头
               0  = 空仓 / 无信号
               2  = 强制出场 (通道位置达到 exit_target)
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        n = len(df)
        close_s = df["close"].astype(np.float64)
        open_s  = df["open"].astype(np.float64)
        high_s  = df["high"].astype(np.float64)
        low_s   = df["low"].astype(np.float64)

        close = close_s.values
        open_ = open_s.values

        # ------------------------------------------------------------------
        # 步骤1: 滚动通道位置 (shift(1) 防前瞻)
        #   当前 bar 的通道仅依赖前一根 bar 之前的 high/low 极值
        # ------------------------------------------------------------------
        ch_high  = high_s.shift(1).rolling(channel_period).max()
        ch_low   = low_s.shift(1).rolling(channel_period).min()
        ch_range = ch_high - ch_low

        ch_high_arr  = ch_high.values
        ch_low_arr   = ch_low.values
        ch_range_arr = ch_range.values

        with np.errstate(divide="ignore", invalid="ignore"):
            ch_pos = np.where(
                ch_range_arr > 0,
                (close - ch_low_arr) / ch_range_arr,
                0.5,
            )

        # 超卖区 → 做多机会；超买区 → 做空机会
        near_low  = ch_pos <= extreme_pct
        near_high = ch_pos >= (1.0 - extreme_pct)

        # ------------------------------------------------------------------
        # 步骤2: 前根K线数据 (shift(1)，防前瞻)
        #   prev_close/prev_open 对应已完成的前根K线，无前瞻
        # ------------------------------------------------------------------
        prev_close = close_s.shift(1).values
        prev_open  = open_s.shift(1).values

        # 前根阴线: 前根下跌趋势确认 (做多场景)
        prev_bearish = prev_close < prev_open
        # 前根阳线: 前根上涨趋势确认 (做空场景)
        prev_bullish = prev_close > prev_open

        # 当根反转 (使用当前 bar 自身 close，K线完成时有效):
        #   做多反转: 当根收盘 > 前根收盘 (买方首次收复跌幅)
        #   做空反转: 当根收盘 < 前根收盘 (卖方首次收复涨幅)
        curr_reversal_up   = close > prev_close
        curr_reversal_down = close < prev_close

        # ------------------------------------------------------------------
        # 步骤3: 向量化入场候选
        # ------------------------------------------------------------------
        # 做多: 通道底部 + 前根阴线 + 当根向上反转
        long_entry_arr = near_low & prev_bearish & curr_reversal_up
        # 做空: 通道顶部 + 前根阳线 + 当根向下反转
        short_entry_arr = near_high & prev_bullish & curr_reversal_down

        # NaN 安全: prev_close/prev_open 首行为 NaN，抑制首行信号
        valid = np.isfinite(prev_close) & np.isfinite(prev_open)
        long_entry_arr  = long_entry_arr  & valid
        short_entry_arr = short_entry_arr & valid

        # ------------------------------------------------------------------
        # 步骤4: 有状态循环 — 入场 / 持仓维持 / 出场
        # ------------------------------------------------------------------
        signals = np.zeros(n, dtype=np.int8)
        active: int = 0   # +1 = 持多, -1 = 持空, 0 = 空仓

        for i in range(n):
            chl = ch_low_arr[i]
            chr_ = ch_range_arr[i]

            # 通道 NaN 或零宽: 跳过，重置持仓
            if not np.isfinite(chl) or not np.isfinite(chr_) or chr_ <= 0:
                active = 0
                continue

            cp = ch_pos[i]

            # ---- 持多: 通道位置达到 exit_target 时出场 ----------------------
            if active == 1:
                if cp >= exit_target:
                    signals[i] = 2
                    active = 0
                else:
                    signals[i] = 1   # 维持多头
                continue

            # ---- 持空: 通道位置降至 (1 - exit_target) 时出场 ----------------
            if active == -1:
                if cp <= (1.0 - exit_target):
                    signals[i] = 2
                    active = 0
                else:
                    signals[i] = -1  # 维持空头
                continue

            # ---- 空仓: 检查入场候选 ------------------------------------------
            if long_entry_arr[i]:
                signals[i] = 1
                active = 1
            elif short_entry_arr[i]:
                signals[i] = -1
                active = -1
            # 否则: 无信号，signals[i] 保持 0

        # ------------------------------------------------------------------
        # NaN safety — 预热期清零
        #   channel_period 根 shift(1)+rolling 需要额外 1 根偏移 + 5 根缓冲
        # ------------------------------------------------------------------
        warmup = channel_period + 5
        signals[:warmup] = 0

        return signals

    # ------------------------------------------------------------------
    # 持仓参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """返回适合第一反转K线策略的风控参数。

        参数设计理由:
        - hard_stop_pct=0.5 : 假反转延续下跌时快速止损，0.5% 硬止损
                              与铁矿石5分钟日内波动幅度匹配，避免过早止损。
        - trailing_pct=0.4  : 移动止损保护早期入场浮盈，0.4% 略窄于硬止损，
                              确保反转成功后不被回调磨损。
        - tp1_pct=0.4       : 半仓止盈，反转初段即锁定部分收益，降低持仓风险。
        - tp2_pct=0.8       : 余仓止盈，捕获反转延伸至通道中轴的完整行程。
        - max_lots=1        : 研究阶段保守单手，专注评估信号质量。
        """
        return PositionParams(
            hard_stop_pct=0.5,
            trailing_pct=0.4,
            tp1_pct=0.4,
            tp2_pct=0.8,
            max_lots=1,
            unit=1,
        )
