"""
Strategy #32 — EMA Squeeze Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 当快速EMA与慢速EMA非常接近 ("挤压"状态)，市场处于多空均衡区间。
  均衡被打破后偏离量越大，越容易被修正回零轴。本策略直接用百分比价差
  (EMA Spread) 超过固定阈值作为入场依据，利用价差向零回归的均值回归特性。

  EMA 价差 (Spread) 定义:
    spread = (EMA_fast - EMA_slow) / EMA_slow
    即快速EMA相对于慢速EMA的百分比偏离。spread = 0 表示两条EMA完全重合，
    市场处于均衡"挤压"状态; spread 绝对值越大，短期动量越偏离均衡。

  与 S29 (Price Oscillator) 的本质区别:
    ① S29 将 PO 进行滚动 Z-Score 规范化后以 σ 倍数作为阈值；
       S32 直接用百分比绝对阈值 (squeeze_threshold)，无统计归一化
    ② S32 额外引入通道位置过滤 (来自 S11): 仅在通道底部做多，通道顶部做空，
       双重确认信号质量，减少假突破
    ③ S32 的信号频率更稳定: 阈值固定，不随历史波动分布变化而漂移

  挤压均值回归假设:
    当两条不同速度的EMA高度重合时，说明短期和中期趋势方向暂时一致，
    市场在该价位存在较强的价格锚定。一旦快速EMA向下偏离 (spread < -threshold)，
    说明短期动量突然向下，但中期趋势未变，价差具有向零回归的拉力。
    通道位置过滤进一步确认: 做多信号仅在价格处于通道低端 (已下探) 时有效，
    避免在通道中部的随机波动中误入场。

  EMA 价差计算 (防前瞻):
    spread = (ema_fast - ema_slow) / ema_slow   (当前 bar 计算)
    spread_s = spread.shift(1)                   (shift(1) → 信号使用上一 bar 的值)
    通道高低点同样使用 shift(1) 后的滚动极值，确保无前瞻偏差

  入场逻辑:
    做多: spread_s < -squeeze_threshold  (快EMA低于慢EMA超过阈值)
          AND 价格处于通道低端 (ch_pos ≤ 0.20)
    做空: spread_s > +squeeze_threshold  (快EMA高于慢EMA超过阈值)
          AND 价格处于通道高端 (ch_pos ≥ 0.80)

  出场逻辑 (有状态):
    优先出场条件1 — 价差回零: spread_s 反向穿越零轴 (回归完成)
      持多且 spread_s >= 0: 快EMA已回到慢EMA上方，价差修正完成 → 平多
      持空且 spread_s <= 0: 快EMA已回到慢EMA下方，价差修正完成 → 平空
    备用出场条件2 — 通道中轴: 价格抵达通道中位线 (ch_mid)
      持多且 close >= ch_mid: 价格从底部回升至中轴，均值回归阶段性完成
      持空且 close <= ch_mid: 价格从顶部下跌至中轴，均值回归阶段性完成

  通道位置计算 (来自 S11, channel_period 固定为 20):
    ch_high = shift(1).rolling(20).max(high)     ← 防前瞻
    ch_low  = shift(1).rolling(20).min(low)      ← 防前瞻
    ch_range = ch_high - ch_low
    ch_pos  = (close - ch_low) / ch_range        (ch_range=0 时取 0.5)
    near_low  : ch_pos ≤ 0.20  (价格处于通道底部20%区间)
    near_high : ch_pos ≥ 0.80  (价格处于通道顶部20%区间)

  NaN 安全处理:
    - ema_slow 为 0 或 NaN 时 spread 保持 NaN
    - spread_s 含 NaN 的 bar 不触发入场
    - ch_range = 0 时 ch_pos 取 0.5 (中性，不满足极端条件)
    - 预热期 (前 slow_period + channel_period + 5 根 bar): 信号全置 0

  参数设计 (3个, 27种组合, WFO 可承受):
    - fast_period      : 快速EMA周期 [5, 8, 13]         约25min ~ 65min EMA
    - slow_period      : 慢速EMA周期 [15, 25, 40]       约75min ~ 200min EMA
    - squeeze_threshold: EMA价差入场阈值 [0.001, 0.002, 0.003]  (0.1%~0.3%)
    - channel_period   : 固定为 20，不纳入参数网格 (与 S11 最优参数一致)

  适用环境: 铁矿石5分钟行情中短期动量偏离后均值回归特征显著的震荡行情
  风险提示: 强趋势行情中 spread 可能持续扩大，阈值触发后价差不回归；
            通道位置过滤能减少部分误入，但无法完全过滤趋势行情风险；
            建议配合严格的硬止损参数

  回测参数:
    - 数据频率: 5分钟K线
    - hard_stop_pct : 0.6%  — 价差持续扩张时限制单笔最大亏损
    - trailing_pct  : 0.5%  — 价差收敛过程中保护浮动盈利
    - tp1_pct       : 0.5%  — 半仓止盈，回归早期降低风险敞口
    - tp2_pct       : 1.0%  — 余仓止盈，捕获价差完整回归的最大收益
    - max_lots      : 1     — 研究阶段保守单手，聚焦信号质量评估
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams

# channel_period 固定为 20，与 S11 最优参数一致，不纳入参数网格
_CHANNEL_PERIOD: int = 20


class EmaSqueezeReversion(BaseResearchStrategy):
    """EMA 挤压均值回归策略 — S32。

    计算快慢EMA之间的百分比价差 (EMA Spread)，当价差绝对值超过固定阈值
    且通道位置处于极端区间时入场做均值回归，价差回归零轴或价格抵达通道
    中轴时出场。相比 S29 (Z-Score 归一化)，本策略使用绝对阈值，信号
    频率不依赖历史波动分布，适合铁矿石日内短期动量偏离行情。
    """

    name = "EMA Squeeze Mean-Reversion"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回 3 参数网格，共 3×3×3 = 27 个参数组合。

        fast_period : int
            快速EMA的 span 参数。5 → ~25min EMA (对短期波动高度敏感)；
            13 → ~65min EMA (较平滑，减少高频噪声触发)。
        slow_period : int
            慢速EMA的 span 参数。15 → ~75min EMA (近期短中期趋势基准)；
            40 → ~200min EMA (较长期趋势基准，price差绝对值更稳定)。
            需大于 fast_period，参数组合中已确保此约束。
        squeeze_threshold : float
            EMA 价差入场阈值 (百分比小数)。0.001 → 价差超过 0.1% 即触发
            (信号较频繁)；0.003 → 价差超过 0.3% 才触发 (信号较保守)。
            铁矿石5分钟级别日内波动中，0.1%~0.3% 为合理的"偏离均衡"范围。
        """
        return {
            "fast_period":       [5, 8, 13],
            "slow_period":       [15, 25, 40],
            "squeeze_threshold": [0.001, 0.002, 0.003],
        }

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        fast_period: int = 8,
        slow_period: int = 25,
        squeeze_threshold: float = 0.002,
    ) -> np.ndarray:
        """生成基于 EMA 价差阈值与通道位置双重确认的均值回归信号。

        实现分四步:
        1. 计算快慢 EMA，得到百分比价差 (spread)，shift(1) 防前瞻。
        2. 计算滚动通道高低点与通道相对位置，确定极端区间 (来自 S11)。
        3. 向量化预计算入场候选: 价差超阈值 AND 通道位置极端。
        4. 有状态循环: 价差超阈值时入场，价差回零轴或价格达通道中轴时出场。

        Parameters
        ----------
        df : pd.DataFrame
            含 'open', 'high', 'low', 'close', 'volume' 列的 OHLCV DataFrame，
            index 为时间序列。
        fast_period : int
            快速 EMA 的 span 参数 (默认 8)。
        slow_period : int
            慢速 EMA 的 span 参数 (默认 25)。需大于 fast_period。
        squeeze_threshold : float
            EMA 百分比价差的绝对值入场阈值 (默认 0.002，即 0.2%)。

        Returns
        -------
        np.ndarray of int8
            与 df 等长的信号数组:
               1  = 做多 (含持仓维持)
              -1  = 做空 (含持仓维持)
               0  = 空仓 / 无信号
               2  = 强制出场 (价差回零轴 或 价格达通道中轴)
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        n = len(df)
        close_s = df["close"].astype(np.float64)
        high_s  = df["high"].astype(np.float64)
        low_s   = df["low"].astype(np.float64)

        # ------------------------------------------------------------------
        # 步骤1: 计算快慢 EMA 和百分比价差 (spread)
        # ------------------------------------------------------------------
        ema_fast = close_s.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close_s.ewm(span=slow_period, adjust=False).mean()

        # spread = (EMA_fast - EMA_slow) / EMA_slow (百分比偏离)
        # ema_slow 为 0 时用 NaN 替代，避免除零；pandas 会自然传播 NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            spread = (ema_fast - ema_slow) / ema_slow.replace(0.0, np.nan)

        # shift(1): 当前 bar 的信号使用上一 bar 结束时的价差，防前瞻
        spread_s = spread.shift(1)

        # ------------------------------------------------------------------
        # 步骤2: 滚动通道位置 (channel_period 固定为 20, 与 S11 一致)
        #   shift(1) 后再做 rolling，确保当前 bar 的通道仅依赖历史 high/low
        # ------------------------------------------------------------------
        ch_high  = high_s.shift(1).rolling(_CHANNEL_PERIOD).max()
        ch_low   = low_s.shift(1).rolling(_CHANNEL_PERIOD).min()
        ch_range = ch_high - ch_low

        close_arr    = close_s.values
        ch_high_arr  = ch_high.values
        ch_low_arr   = ch_low.values
        ch_range_arr = ch_range.values

        # ch_range = 0 时取 0.5 (中性，不满足极端区间条件)
        with np.errstate(divide="ignore", invalid="ignore"):
            ch_pos = np.where(
                ch_range_arr > 0,
                (close_arr - ch_low_arr) / ch_range_arr,
                0.5,
            )

        # 通道极端区间: 底部20%为超卖候选，顶部20%为超买候选 (固定，与 S11 最优一致)
        near_low  = ch_pos <= 0.20
        near_high = ch_pos >= 0.80

        # ------------------------------------------------------------------
        # 步骤3: 向量化预计算入场候选
        # ------------------------------------------------------------------
        spread_s_arr = spread_s.values

        # NaN 安全: spread_s 含 NaN 的 bar 不触发入场
        valid = np.isfinite(spread_s_arr)

        # 做多: 价差向下偏离超过阈值 (快EMA低于慢EMA，短期被压制) AND 通道底部
        long_entry_arr  = valid & (spread_s_arr < -squeeze_threshold) & near_low
        # 做空: 价差向上偏离超过阈值 (快EMA高于慢EMA，短期被拉升) AND 通道顶部
        short_entry_arr = valid & (spread_s_arr >  squeeze_threshold) & near_high

        # ------------------------------------------------------------------
        # 步骤4: 有状态循环 — 入场 / 持仓维持 / 出场
        # ------------------------------------------------------------------
        signals = np.zeros(n, dtype=np.int8)
        active: int = 0   # +1 = 持多, -1 = 持空, 0 = 空仓

        for i in range(n):
            sp  = spread_s_arr[i]
            cl  = close_arr[i]
            chl = ch_low_arr[i]
            chr = ch_range_arr[i]

            # 通道中轴 (用于备用出场): ch_low + 0.5 * ch_range
            # ch_range NaN 时 ch_mid 为 NaN，np.isfinite 保护后续比较
            ch_mid = chl + 0.5 * chr if np.isfinite(chr) else np.nan

            # ---- 持多: 检查是否满足出场条件 ------------------------------------
            if active == 1:
                # 出场条件1: 价差回零轴 (spread_s >= 0 → 快EMA已不再低于慢EMA)
                if np.isfinite(sp) and sp >= 0.0:
                    signals[i] = 2
                    active = 0
                    continue
                # 出场条件2: 价格抵达通道中轴 (均值回归阶段性完成)
                elif np.isfinite(ch_mid) and cl >= ch_mid:
                    signals[i] = 2
                    active = 0
                    continue
                else:
                    signals[i] = 1   # 维持多头
                    continue

            # ---- 持空: 检查是否满足出场条件 ------------------------------------
            if active == -1:
                # 出场条件1: 价差回零轴 (spread_s <= 0 → 快EMA已不再高于慢EMA)
                if np.isfinite(sp) and sp <= 0.0:
                    signals[i] = 2
                    active = 0
                    continue
                # 出场条件2: 价格抵达通道中轴 (均值回归阶段性完成)
                elif np.isfinite(ch_mid) and cl <= ch_mid:
                    signals[i] = 2
                    active = 0
                    continue
                else:
                    signals[i] = -1  # 维持空头
                    continue

            # ---- 空仓: 检查入场候选 --------------------------------------------
            if long_entry_arr[i]:
                signals[i] = 1
                active = 1
            elif short_entry_arr[i]:
                signals[i] = -1
                active = -1
            # 否则: 空仓且无信号，signals[i] 保持 0

        # ------------------------------------------------------------------
        # NaN safety — 预热期清零
        # 前 slow_period + channel_period + 5 根 bar 的 EMA/通道计算不稳定
        # ------------------------------------------------------------------
        warmup = slow_period + _CHANNEL_PERIOD + 5
        signals[:warmup] = 0

        return signals

    # ------------------------------------------------------------------
    # 持仓参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """返回适合 EMA 挤压均值回归策略的风控参数。

        参数设计理由:
        - hard_stop_pct=0.6 : 强趋势行情中 spread 可能持续扩大，0.6% 硬止损
                              在价差无法及时收敛时限制单笔最大损失。比 0.5%
                              略宽，给价差短暂继续扩张后再回归预留空间。
        - trailing_pct=0.5  : spread 收敛过程以 0.5% 移动止损保护浮动盈利，
                              防止价格在回归中途出现二次偏离磨损收益。
        - tp1_pct=0.5       : 半仓止盈 0.5%，均值回归早期 (价差部分收敛)
                              锁定一半仓位收益，与铁矿石5分钟回归幅度匹配。
        - tp2_pct=1.0       : 余仓止盈 1.0%，捕获价差从阈值外完整回归
                              至零轴的最大收益，兼顾盈亏比。
        - max_lots=1        : 研究阶段保守单手，专注评估入场信号质量，
                              不引入仓位加减管理变量干扰策略纯净性。
        """
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
