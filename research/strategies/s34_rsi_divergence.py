"""
Strategy #34 — RSI Price Divergence Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 经典RSI背离 — 价格创新低但RSI未创新低 (看涨背离)，说明下跌动能
           已耗尽，价格具有均值回归潜力。结合通道位置过滤，仅在价格处于通道
           底部/顶部时入场，双重确认提高信号质量。

  RSI背离的理论基础:
    价格创新低 → 空头持续发力，价格下探新低
    RSI未创新低 → 下跌每单位价格消耗更少能量，空头动能减弱
    两者背离 → 趋势延续的"燃料"不足，均值回归概率上升
  镜像逻辑适用于看跌背离 (价格新高但RSI未新高 → 做空)。

  RSI 计算 (Wilder EWM 方法):
    ret       = close.diff()
    avg_gain  = ret.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss  = (-ret).clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
    rsi       = 100 - 100 / (1 + avg_gain / avg_loss)
    avg_loss=0 时用 NaN 替代，避免除零；fillna(50) 使预热期中性化

  防前瞻设计:
    rsi_s    = rsi.shift(1)    ← 使用上一根已完成 bar 的 RSI 值
    close_s  = close.shift(1) ← 使用上一根已完成 bar 的收盘价
    通道高低点使用 shift(1).rolling() (与 S11 保持一致)

  看涨背离检测 (divergence_period 根 bar 回看):
    price_low = close_s.rolling(divergence_period).min()
    rsi_low   = rsi_s.rolling(divergence_period).min()
    bull_div  = (close_s ≤ price_low × 1.001)   ← 价格接近/处于区间最低
               AND (rsi_s > rsi_low + rsi_gap)  ← RSI 显著高于区间最低
    额外过滤: 通道位置 ≤ 0.25 (价格处于通道底部四分之一区间)

  看跌背离 (镜像):
    price_high = close_s.rolling(divergence_period).max()
    rsi_high   = rsi_s.rolling(divergence_period).max()
    bear_div   = (close_s ≥ price_high × 0.999)
               AND (rsi_s < rsi_high - rsi_gap)
    额外过滤: 通道位置 ≥ 0.75

  出场逻辑 (有状态):
    价格回归至通道中轴 (ch_low + 0.5 * ch_range) 时平仓
    通道 channel_period 固定为 20，不纳入参数网格 (与 S11 最优参数一致)

  NaN 安全:
    - avg_loss=0 时 rsi 取 NaN，fillna(50) 后不影响入场判断 (rsi_gap > 0 时 50-50=0 不满足)
    - ch_range=0 时 ch_pos 取 0.5，不满足通道过滤条件，不产生信号
    - 前 max(rsi_period, divergence_period) + channel_period + 5 根 bar 预热期清零
    - 有状态循环中通道 NaN 时重置持仓状态

  参数设计 (3个，27种组合，WFO 可承受):
    - divergence_period : 背离检测回看周期 [8, 12, 20]  约40min到1.7h
    - rsi_period        : RSI 计算周期 [9, 14, 21]      约45min到1.75h
    - rsi_gap           : RSI 背离最小点差 [3, 5, 8]    RSI 需高于区间最低多少点

  channel_period 固定为 20，与 S11 最优参数一致，不纳入参数网格

  适用环境: 铁矿石5分钟下跌动能耗尽后的均值回归，RSI背离在成熟商品期货中
           是经典且有效的动量衰竭信号
  风险提示: 背离信号时效性有限，强趋势中可能出现多次假背离；
           rsi_gap 过小会导致虚假背离频繁触发，建议不低于 3 点

  回测参数:
    - 数据频率: 5分钟K线
    - hard_stop_pct : 0.6%  — 背离失败时亏损较深，略宽止损给信号验证空间
    - trailing_pct  : 0.5%  — 均值回归过程中保护浮动盈利
    - tp1_pct       : 0.5%  — 半仓止盈，背离修复初段锁定收益
    - tp2_pct       : 1.0%  — 余仓止盈，捕获价格回归通道中轴的完整行程
    - max_lots      : 1     — 研究阶段保守单手
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams

# channel_period 固定为 20，与 S11 最优参数一致，不纳入参数网格
_CHANNEL_PERIOD: int = 20


class RsiDivergenceReversion(BaseResearchStrategy):
    """RSI 价格背离均值回归策略 — S34。

    检测价格与 RSI 之间的看涨/看跌背离 (经典动量背离信号)，结合滚动通道
    位置过滤，仅在价格处于通道四分之一极端区间时入场做均值回归。出场以
    通道中轴为目标，价格回归至 ch_low + 0.5 * ch_range 时平仓。
    """

    name = "RSI Divergence Mean-Reversion"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回 3 参数网格，共 3×3×3 = 27 个参数组合。

        divergence_period : int
            背离检测的回看 bar 数。8 → 约40分钟内的背离 (捕捉短期动量衰竭，
            信号频率高)；20 → 约100分钟内的背离 (趋势级别更大，信号更可靠
            但频率低)。
        rsi_period : int
            Wilder EWM 方式的 RSI 周期。9 → 短期RSI (波动较大，背离检测更
            灵敏)；21 → 长期RSI (较平滑，背离信号更稳定但滞后)。
        rsi_gap : float
            RSI 背离的最小点差。3 → 宽松条件，背离信号频率高；8 → 严格条件，
            仅接受显著背离，信号少但质量高。铁矿石5分钟 RSI 波动约5-15点/bar，
            3-8 点为合理的"显著背离"范围。
        """
        return {
            "divergence_period": [8, 12, 20],
            "rsi_period":        [9, 14, 21],
            "rsi_gap":           [3, 5, 8],
        }

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        divergence_period: int = 12,
        rsi_period: int = 14,
        rsi_gap: float = 5.0,
    ) -> np.ndarray:
        """生成基于 RSI 价格背离与通道位置双重确认的均值回归信号。

        实现分四步:
        1. 计算 Wilder EWM RSI，shift(1) 防前瞻。
        2. 计算滚动通道高低点与通道相对位置 (channel_period=20，固定)。
        3. 向量化检测看涨/看跌背离，结合通道位置过滤生成入场候选。
        4. 有状态循环: 入场后持续持仓，价格到达通道中轴时出场。

        Parameters
        ----------
        df : pd.DataFrame
            含 'open', 'high', 'low', 'close', 'volume' 列的 OHLCV DataFrame，
            index 为时间序列。
        divergence_period : int
            背离检测的回看 bar 数 (默认 12，约60分钟)。
        rsi_period : int
            Wilder EWM RSI 的周期 (默认 14)。
        rsi_gap : float
            RSI 背离的最小点差 (默认 5.0 点)。看涨背离要求当前 rsi_s 比
            区间内 rsi 最低值高出至少 rsi_gap 点。

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

        # ------------------------------------------------------------------
        # 步骤1: Wilder EWM RSI (防前瞻: shift(1) 后使用)
        # ------------------------------------------------------------------
        ret = close_s.diff()
        avg_gain = ret.clip(lower=0).ewm(alpha=1.0 / rsi_period, adjust=False).mean()
        avg_loss = (-ret).clip(lower=0).ewm(alpha=1.0 / rsi_period, adjust=False).mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0.0, np.nan))
        rsi = rsi.fillna(50.0)  # 预热期 / 零损失时取中性值 50

        # shift(1): 当前 bar 使用上一根已完成 bar 的 RSI
        rsi_s    = rsi.shift(1)
        # shift(1): 当前 bar 使用上一根已完成 bar 的收盘价 (背离检测基准)
        close_s1 = close_s.shift(1)

        # ------------------------------------------------------------------
        # 步骤2: 滚动通道位置 (channel_period=20，shift(1) 防前瞻)
        # ------------------------------------------------------------------
        ch_high  = high_s.shift(1).rolling(_CHANNEL_PERIOD).max()
        ch_low   = low_s.shift(1).rolling(_CHANNEL_PERIOD).min()
        ch_range = ch_high - ch_low

        close_arr    = close_s.values
        ch_high_arr  = ch_high.values
        ch_low_arr   = ch_low.values
        ch_range_arr = ch_range.values

        with np.errstate(divide="ignore", invalid="ignore"):
            ch_pos = np.where(
                ch_range_arr > 0,
                (close_arr - ch_low_arr) / ch_range_arr,
                0.5,
            )

        # 通道底部四分之一区间 → 超卖候选；顶部四分之一区间 → 超买候选
        near_low  = ch_pos <= 0.25
        near_high = ch_pos >= 0.75

        # ------------------------------------------------------------------
        # 步骤3: 向量化背离检测
        #   price_low / rsi_low: 在 divergence_period 根 bar 内的最低值
        #   看涨背离: 价格接近/创区间新低 但 RSI 显著高于区间最低 RSI
        # ------------------------------------------------------------------
        price_low  = close_s1.rolling(divergence_period).min()
        rsi_low    = rsi_s.rolling(divergence_period).min()

        # 看涨背离: 价格低点 (允许 0.1% 容差) + RSI 未创新低 (高出 rsi_gap 点)
        bull_div = (close_s1 <= price_low * 1.001) & (rsi_s > rsi_low + rsi_gap)

        # 看跌背离 (镜像)
        price_high = close_s1.rolling(divergence_period).max()
        rsi_high   = rsi_s.rolling(divergence_period).max()

        # 看跌背离: 价格高点 (允许 0.1% 容差) + RSI 未创新高 (低于 rsi_gap 点)
        bear_div = (close_s1 >= price_high * 0.999) & (rsi_s < rsi_high - rsi_gap)

        # 结合通道位置过滤
        long_entry_arr  = bull_div.values & near_low
        short_entry_arr = bear_div.values & near_high

        # NaN 安全: rsi_s / close_s1 含 NaN 的行置 False
        valid_rsi   = np.isfinite(rsi_s.values)
        valid_price = np.isfinite(close_s1.values)
        valid = valid_rsi & valid_price
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
            cl = close_arr[i]

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
        #   RSI 需要 rsi_period 根 bar 预热
        #   背离检测需要 divergence_period 根 shift(1) 数据
        #   通道需要 channel_period 根 shift(1)+rolling 数据
        #   + 5 根额外缓冲
        # ------------------------------------------------------------------
        warmup = max(rsi_period, divergence_period) + _CHANNEL_PERIOD + 5
        signals[:warmup] = 0

        return signals

    # ------------------------------------------------------------------
    # 持仓参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """返回适合 RSI 背离均值回归策略的风控参数。

        参数设计理由:
        - hard_stop_pct=0.6 : 背离失败 (趋势延续) 时亏损可能较深，0.6% 硬止损
                              比 S33 略宽，给背离信号验证的时间窗口更充裕。
        - trailing_pct=0.5  : 均值回归过程中以 0.5% 移动止损保护浮动盈利，
                              防止价格在回归途中出现二次下探磨损收益。
        - tp1_pct=0.5       : 半仓止盈 0.5%，背离修复初段锁定部分收益，
                              与铁矿石5分钟均值回归幅度匹配。
        - tp2_pct=1.0       : 余仓止盈 1.0%，捕获价格从极值回归通道中轴
                              的完整行程，兼顾盈亏比。
        - max_lots=1        : 研究阶段保守单手，专注评估背离信号质量。
        """
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
            unit=1,
        )
