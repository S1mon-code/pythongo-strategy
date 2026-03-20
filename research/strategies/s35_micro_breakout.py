"""
Strategy #35 — Micro Breakout Reversal at Channel Extreme (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 价格处于滚动通道底部/顶部极端区间时，若当前K线收盘突破前根K线
           的最高价 (做多) 或最低价 (做空)，说明买方/卖方已获得短期阻力位
           的控制权，均值回归的启动信号更为可靠。

  "微突破"的逻辑:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 做多: 价格在通道底部极端区 (超卖)                                    │
  │       + 当前收盘 > 前根K线最高价                                     │
  │       → 买方突破了前根K线设定的短期阻力，说明回升动能已经启动        │
  │ 做空: 价格在通道顶部极端区 (超买)                                    │
  │       + 当前收盘 < 前根K线最低价                                     │
  │       → 卖方突破了前根K线设定的短期支撑，说明下跌动能已经启动        │
  └─────────────────────────────────────────────────────────────────────┘

  与 S33 (第一反转K线) 的对比:
  - S33: 当前收盘 > 前根收盘 (简单收盘反转，条件宽松，信号更多)
  - S35: 当前收盘 > 前根最高 (突破前根K线全部价格区间，条件更严格，信号更少但更强)
  - S35 的"微突破"相当于一个2根K线的迷你突破，在极值处发生则反转确认更强

  可选的最小突破幅度 (min_break_pct):
    仅在 min_break_pct > 0 时生效:
    做多要求: close > prev_high × (1 + min_break_pct)
    做空要求: close < prev_low  × (1 - min_break_pct)
    过滤掉"刚好突破1个tick"的弱突破，提高信号质量

  通道相对位置计算 (防前瞻，来自 S11):
    ch_high = shift(1).rolling(channel_period).max(high)
    ch_low  = shift(1).rolling(channel_period).min(low)
    ch_range = ch_high - ch_low
    ch_pos   = (close - ch_low) / ch_range  (ch_range=0 时取 0.5)
    near_low  : ch_pos ≤ extreme_pct
    near_high : ch_pos ≥ 1 - extreme_pct

  前根K线高低 (防前瞻):
    prev_high = high.shift(1)   ← 前根K线已完成，合规
    prev_low  = low.shift(1)    ← 同上

  出场逻辑 (有状态):
    做多出场: 当前收盘 ≥ 通道中轴 (ch_low + 0.5 * ch_range)
    做空出场: 当前收盘 ≤ 通道中轴

  NaN 安全:
    - ch_range=0 时 ch_pos 取 0.5，不满足极端区条件，不产生信号
    - prev_high / prev_low 首行为 NaN，NaN 比较结果为 False，不触发信号
    - 前 channel_period + 5 根 bar (预热期) 信号全置 0
    - 有状态循环中通道 NaN 时重置持仓状态

  参数设计 (3个，27种组合，WFO 可承受):
    - channel_period : 通道周期 [10, 20, 30]           约50min到2.5h的通道
    - extreme_pct    : 极端区间比例 [0.15, 0.20, 0.25] 通道两端多少%视为极端
    - min_break_pct  : 最小突破幅度 [0.0, 0.001, 0.002] 0=无要求; 0.001=需超0.1%

  适用环境: 铁矿石5分钟震荡行情，通道极值处的微突破具有强烈的买卖力量转换信号
  风险提示: 强趋势行情中极值处可能持续出现无效突破 (假突破)；
           min_break_pct > 0 能减少部分假突破，但会降低信号频率

  回测参数:
    - 数据频率: 5分钟K线
    - hard_stop_pct : 0.5%  — 假突破失败时快速止损，幅度与 S33 一致
    - trailing_pct  : 0.4%  — 突破后保护浮动盈利
    - tp1_pct       : 0.4%  — 半仓止盈，微突破后初始收益锁定
    - tp2_pct       : 0.8%  — 余仓止盈，捕获均值回归至通道中轴的完整行程
    - max_lots      : 1     — 研究阶段保守单手
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class MicroBreakoutReversal(BaseResearchStrategy):
    """通道极值处微突破反转均值回归策略 — S35。

    在价格处于滚动通道底部/顶部极端区间时，检测当前K线收盘是否突破前根K线
    的最高价 (做多) 或最低价 (做空)。可选 min_break_pct 过滤弱突破。
    出场以通道中轴为目标，价格回归至 ch_low + 0.5 * ch_range 时平仓。
    """

    name = "Micro Breakout Reversal"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回 3 参数网格，共 3×3×3 = 27 个参数组合。

        channel_period : int
            滚动通道的回看周期。10 → 约50分钟短通道 (极值定义更敏感，信号
            频率高)；30 → 约2.5小时长通道 (极值定义更严格，信号更可靠)。
        extreme_pct : float
            通道两端判定为极端区间的比例。0.15 → 严格极值 (信号少但精确)；
            0.25 → 宽松极值 (允许价格在通道四分之一内即视为极端，信号更多)。
        min_break_pct : float
            最小突破幅度要求 (占价格的百分比小数)。0.0 → 无额外要求，只需
            close > prev_high 即可 (最宽松)；0.002 → 需超越前根高点 0.2% 以上
            才视为有效突破 (最严格，过滤微弱突破噪声)。
        """
        return {
            "channel_period": [10, 20, 30],
            "extreme_pct":    [0.15, 0.20, 0.25],
            "min_break_pct":  [0.0, 0.001, 0.002],
        }

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        channel_period: int = 20,
        extreme_pct: float = 0.20,
        min_break_pct: float = 0.001,
    ) -> np.ndarray:
        """生成基于通道极值处微突破的均值回归信号。

        实现分四步:
        1. 计算滚动通道高低点与通道相对位置，确定超卖/超买区。
        2. 使用 shift(1) 获取前根K线 high/low，构建微突破条件。
        3. 向量化预计算入场候选 (含可选 min_break_pct 过滤)。
        4. 有状态循环: 入场后持续持仓，价格到达通道中轴时出场。

        Parameters
        ----------
        df : pd.DataFrame
            含 'open', 'high', 'low', 'close', 'volume' 列的 OHLCV DataFrame，
            index 为时间序列。
        channel_period : int
            滚动通道的回看周期 (默认 20 根 bar，约 100 分钟)。
        extreme_pct : float
            通道端部极端区间比例 (默认 0.20，即通道宽度的 20% 以内)。
        min_break_pct : float
            最小突破幅度 (默认 0.001，即 0.1%)。设为 0.0 则无额外幅度要求。

        Returns
        -------
        np.ndarray of int8
            与 df 等长的信号数组:
               1  = 做多入场 / 维持多头
              -1  = 做空入场 / 维持空头
               0  = 空仓 / 无信号
               2  = 强制出场 (价格到达通道中轴)
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        n = len(df)
        close_s = df["close"].astype(np.float64)
        high_s  = df["high"].astype(np.float64)
        low_s   = df["low"].astype(np.float64)

        close = close_s.values
        high  = high_s.values
        low   = low_s.values

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
        # 步骤2: 前根K线高低 (shift(1)，防前瞻)
        # ------------------------------------------------------------------
        prev_high = high_s.shift(1).values
        prev_low  = low_s.shift(1).values

        # ------------------------------------------------------------------
        # 步骤3: 向量化入场候选
        #   微突破做多: 收盘突破前根K线最高价 (+ 可选最小幅度)
        #   微突破做空: 收盘跌破前根K线最低价 (- 可选最小幅度)
        # ------------------------------------------------------------------
        if min_break_pct > 0.0:
            # 带最小幅度过滤: close 须超越前根高点 min_break_pct 以上
            long_break  = close > prev_high * (1.0 + min_break_pct)
            short_break = close < prev_low  * (1.0 - min_break_pct)
        else:
            # 无额外幅度要求: 仅需 close > prev_high (或 < prev_low)
            long_break  = close > prev_high
            short_break = close < prev_low

        long_entry_arr  = near_low  & long_break
        short_entry_arr = near_high & short_break

        # NaN 安全: prev_high / prev_low 首行为 NaN，numpy 比较 NaN 结果为 False
        # np.where 已将 ch_range=0 处理为 0.5，near_low/near_high 不会误触发
        # 额外保护: 将含 NaN 的前根数据行显式置 False
        valid = np.isfinite(prev_high) & np.isfinite(prev_low)
        long_entry_arr  = long_entry_arr  & valid
        short_entry_arr = short_entry_arr & valid

        # ------------------------------------------------------------------
        # 步骤4: 有状态循环 — 入场 / 持仓维持 / 出场
        # ------------------------------------------------------------------
        signals = np.zeros(n, dtype=np.int8)
        active: int = 0   # +1 = 持多, -1 = 持空, 0 = 空仓

        for i in range(n):
            chl  = ch_low_arr[i]
            chr_ = ch_range_arr[i]

            # 通道 NaN 或零宽: 跳过，重置持仓
            if not np.isfinite(chl) or not np.isfinite(chr_) or chr_ <= 0:
                active = 0
                continue

            ch_mid = chl + 0.5 * chr_
            cl = close[i]

            # ---- 持多: 价格到达通道中轴时出场 ----------------------------------
            if active == 1:
                if cl >= ch_mid:
                    signals[i] = 2
                    active = 0
                else:
                    signals[i] = 1   # 维持多头
                continue

            # ---- 持空: 价格到达通道中轴时出场 ----------------------------------
            if active == -1:
                if cl <= ch_mid:
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
        """返回适合微突破反转策略的风控参数。

        参数设计理由:
        - hard_stop_pct=0.5 : 假突破失败时快速止损，0.5% 与铁矿石5分钟日内
                              波动幅度匹配，防止假突破导致较大亏损。
        - trailing_pct=0.4  : 突破成功后以 0.4% 移动止损保护浮动盈利，
                              略窄于硬止损确保有效锁定早期收益。
        - tp1_pct=0.4       : 半仓止盈，微突破成功后初始行程锁定部分收益，
                              降低持仓风险敞口。
        - tp2_pct=0.8       : 余仓止盈，捕获价格从极值微突破后回归通道中轴
                              的完整行程，兼顾盈亏比。
        - max_lots=1        : 研究阶段保守单手，专注评估突破信号质量。
        """
        return PositionParams(
            hard_stop_pct=0.5,
            trailing_pct=0.4,
            tp1_pct=0.4,
            tp2_pct=0.8,
            max_lots=1,
            unit=1,
        )
