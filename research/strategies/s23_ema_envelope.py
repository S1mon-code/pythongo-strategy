"""
Strategy #23 — EMA Envelope Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: EMA包络线均值回归 — 当价格偏离EMA一定百分比时进行均值回归操作，
  当价格回归至EMA时出场。

  EMA包络线 vs 布林带的关键区别:
  - 布林带带宽 = 均值 ± k × 滚动标准差，在波动率上升时自动扩宽、在低波动
    率时收窄，导致信号频率随市场状态剧烈变化；
  - EMA包络线带宽 = EMA × (1 ± env_pct)，以固定百分比为基础，带宽随价格
    水平等比例变化，不依赖历史波动率估计，在不同价格区制下更稳定。
  - 在铁矿石等价格区间波动的大宗商品中，固定百分比带比波动率带更适合
    刻画市场常态振荡区间，信号质量更一致。

  学术与实践依据:
  - EMA包络线 (Envelope Channel) 是商品期货技术分析的经典工具，
    早于布林带出现，被大量CTA策略采用于均值回归滤波；
  - Kaufman (2013) "Trading Systems and Methods" 第5版系统梳理了
    包络线在商品市场中的应用，指出百分比带在价格绝对水平变化较大的
    合约上优于标准差带；
  - 大宗商品市场实证研究 (如铁矿石DCE) 表明价格在大多数时间内在
    EMA附近的一个固定百分比区间内振荡，趋势行情占比相对较低。

  动量确认过滤 (降低低质量信号):
  - 仅在短期动量出现反转迹象时才允许入场，避免追随已经有强动能的方向：
    * 做多: 价格跌破下轨 AND 入场前一根bar的短期动量为负
      (即价格在下跌中触及下轨，具备反转基础)
    * 做空: 价格突破上轨 AND 入场前一根bar的短期动量为正
      (即价格在上涨中触及上轨，具备反转基础)
  - 动量 = (close - close.shift(mom_period)) / close.shift(mom_period)

  EMA包络线计算 (使用shift(1)防止未来函数):
  ──────────────────────────────────────────
    ema     = close.ewm(span=ema_period, adjust=False).mean()
    ema_s   = ema.shift(1)            — 前一根bar的EMA，防止当前bar信息泄漏
    upper   = ema_s × (1 + env_pct)  — 上轨
    lower   = ema_s × (1 - env_pct)  — 下轨

  信号生成 (有状态循环):
  ──────────────────────
    入场:
      做多 (+1): close < lower_band  AND  momentum.shift(1) < 0
      做空 (-1): close > upper_band  AND  momentum.shift(1) > 0
    出场:
      价格回归至EMA (close >= ema_s 平多 / close <= ema_s 平空)
      → signal = 2 (强制出场)
    持仓期间每根bar持续写入方向信号 (+1/-1)，告知回测引擎仓位状态

  热身期: 前 ema_period + mom_period + 5 根bar强制置零，确保EMA和动量
         均已充分稳定后才产生信号。

  参数设计 (3个，27组合):
  - ema_period : EMA周期        [20, 50, 100]
  - env_pct    : 包络线宽度比例 [0.003, 0.005, 0.008]  (0.3%/0.5%/0.8%)
  - mom_period : 动量回看周期   [3, 5, 8]

  适用环境: 铁矿石日内及短周期振荡行情；EMA附近存在反复拉锯的市场
  风险提示: 单边趋势行情中价格持续偏离EMA，均值回归失效；
            env_pct设置过小 (如0.003) 时在高波动期信号噪音显著增加；
            ema_period=100时热身期较长，需数据量充足
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class EmaEnvelopeReversion(BaseResearchStrategy):
    """EMA包络线均值回归策略 — S23。

    以固定百分比EMA包络线为通道，价格跌破下轨且短期动量出现反转迹象时
    做多，价格突破上轨且短期动量出现反转迹象时做空，价格回归至EMA时出场。
    相比布林带，百分比包络线带宽在不同价格区制下更稳定，适合铁矿石等
    长期在一定价格范围内振荡的商品期货合约。
    """

    name = "EMA Envelope Mean-Reversion"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回 3 参数网格，共 3×3×3 = 27 个参数组合。

        ema_period : int
            EMA平滑周期。20 → 短期快速响应；100 → 长期趋势中枢。
            较大的周期使包络带更稳定，但热身期更长。
        env_pct : float
            包络线宽度比例 (相对于EMA的百分比)。
            0.003 = 0.3%（较窄，信号频繁）；0.008 = 0.8%（较宽，信号保守）。
            建议根据目标合约的历史平均振幅校准此参数。
        mom_period : int
            动量确认的回看周期。
            mom = (close - close[i-mom_period]) / close[i-mom_period]
            较小的周期 (3) 对近期短线反转更敏感；较大的周期 (8) 要求更明确
            的方向确认后才允许入场。
        """
        return {
            "ema_period": [20, 50, 100],          # EMA周期
            "env_pct":    [0.003, 0.005, 0.008],  # 包络线宽度 (0.3%/0.5%/0.8%)
            "mom_period": [3, 5, 8],              # 动量确认回看周期
        }

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        ema_period: int = 50,
        env_pct: float = 0.005,
        mom_period: int = 5,
    ) -> np.ndarray:
        """生成EMA包络线均值回归信号。

        实现分两步:
        1. 向量化预计算EMA包络带、动量确认条件和入场候选 (long_ok/short_ok)。
        2. 有状态循环处理出场信号 (signal == 2) 及持仓维持逻辑。

        Parameters
        ----------
        df : pd.DataFrame
            含 DatetimeIndex 的 OHLCV DataFrame，至少包含 'close' 列。
        ema_period : int
            EMA平滑周期，span 参数传入 ewm()。
        env_pct : float
            包络线宽度比例。上轨 = EMA × (1 + env_pct)，
            下轨 = EMA × (1 - env_pct)。
        mom_period : int
            动量回看周期，用于确认价格运动方向是否具备反转基础。

        Returns
        -------
        np.ndarray of int8
            与 df 等长的信号数组:
              +1  做多 (持有)
              -1  做空 (持有)
               0  空仓 / 无信号
               2  强制出场 (价格回归至EMA)
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        # ------------------------------------------------------------------
        # 提取收盘价序列
        # ------------------------------------------------------------------
        close = pd.Series(df["close"].values, dtype=np.float64)
        close_arr = close.values  # numpy array, 供循环使用

        # ------------------------------------------------------------------
        # 向量化计算EMA包络带
        # ------------------------------------------------------------------
        # EMA: adjust=False 使用递推式，与实盘one-pass计算一致
        ema = close.ewm(span=ema_period, adjust=False).mean()

        # shift(1) 防止未来函数: 使用前一根bar已完成的EMA作为带宽基准
        ema_s = ema.shift(1)
        ema_s_arr = ema_s.values  # numpy array

        upper_band = ema_s * (1.0 + env_pct)
        lower_band = ema_s * (1.0 - env_pct)

        upper_arr = upper_band.values
        lower_arr = lower_band.values

        # ------------------------------------------------------------------
        # 向量化计算短期动量
        # ------------------------------------------------------------------
        # momentum[i] = (close[i] - close[i - mom_period]) / close[i - mom_period]
        # shift(mom_period) 产生 NaN 在前 mom_period 根bar
        close_shifted = close.shift(mom_period)
        with np.errstate(invalid="ignore", divide="ignore"):
            momentum = (close - close_shifted) / close_shifted
        # 以shift(1)取前一根bar的动量用于入场确认，避免使用当前bar自身
        momentum_prev = momentum.shift(1)
        mom_arr = momentum_prev.values

        # ------------------------------------------------------------------
        # 热身期: 前 ema_period + mom_period + 5 根bar强制不产生信号
        # ------------------------------------------------------------------
        warmup = ema_period + mom_period + 5

        # ------------------------------------------------------------------
        # 向量化预计算入场候选
        # ------------------------------------------------------------------
        # 做多: 当前close跌破下轨 AND 前一根动量为负 (价格处于下跌势中反转)
        long_ok = (close_arr < lower_arr) & (mom_arr < 0.0)
        # 做空: 当前close突破上轨 AND 前一根动量为正 (价格处于上涨势中反转)
        short_ok = (close_arr > upper_arr) & (mom_arr > 0.0)

        # ------------------------------------------------------------------
        # NaN安全: 将任意关键值为NaN的bar对应的入场候选置False
        # ------------------------------------------------------------------
        nan_mask = (
            np.isnan(ema_s_arr)
            | np.isnan(upper_arr)
            | np.isnan(lower_arr)
            | np.isnan(mom_arr)
            | np.isnan(close_arr)
        )
        long_ok[nan_mask]  = False
        short_ok[nan_mask] = False

        # 热身期内也不产生信号
        long_ok[:warmup]  = False
        short_ok[:warmup] = False

        # ------------------------------------------------------------------
        # 有状态循环: 处理出场 (signal=2) 与持仓维持逻辑
        # ------------------------------------------------------------------
        # 信号优先级: 出场 > 入场 > 空仓
        #   出场 (2): 持多且 close >= ema_s → 价格回归至EMA，平多
        #             持空且 close <= ema_s → 价格回归至EMA，平空
        #   入场 (+1/-1): long_ok 或 short_ok 为 True（且当前空仓）
        #   持仓: 已入场后每根bar持续写入方向信号，直到出场
        active = 0  # 当前持仓方向: +1 做多, -1 做空, 0 空仓

        for i in range(n):
            ema_ref = ema_s_arr[i]  # 当前bar对应的shifted EMA (出场参考)

            # ---- 持多: 检查是否回归至EMA出场 --------------------------------
            if active == 1:
                if not np.isnan(ema_ref) and close_arr[i] >= ema_ref:
                    signals[i] = 2  # 价格回归EMA，强制出场
                    active = 0
                else:
                    signals[i] = 1  # 维持多头
                continue

            # ---- 持空: 检查是否回归至EMA出场 --------------------------------
            if active == -1:
                if not np.isnan(ema_ref) and close_arr[i] <= ema_ref:
                    signals[i] = 2  # 价格回归EMA，强制出场
                    active = 0
                else:
                    signals[i] = -1  # 维持空头
                continue

            # ---- 空仓: 检查入场候选 -----------------------------------------
            if long_ok[i]:
                # 价格跌破下轨且动量确认反转 → 做多
                signals[i] = 1
                active = 1
            elif short_ok[i]:
                # 价格突破上轨且动量确认反转 → 做空
                signals[i] = -1
                active = -1
            # 否则: 空仓且无信号，signals[i] 保持 0

        return signals

    # ------------------------------------------------------------------
    # 持仓参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """返回适合EMA包络线均值回归策略的风控参数。

        参数设计理由:
        - hard_stop_pct=0.6 : 均值回归失败 (市场单边趋势突破包络线后持续延伸)
                              时，0.6% 硬止损及时截断亏损；高于 env_pct 上限
                              (0.8%) 的一半，留有合理容错空间。
        - trailing_pct=0.5  : 价格回归EMA过程中以 0.5% 移动止损保护已实现收益，
                              防止回归行情二次反转导致盈利回吐。
        - tp1_pct=0.4       : 半仓止盈 0.4%，在回归行情早期锁定部分收益，降低
                              风险敞口；与铁矿石5分钟K线平均波动幅度匹配。
        - tp2_pct=0.8       : 余仓止盈 0.8%，对应包络线上限 env_pct=0.008 的
                              完整回归幅度，捕获最大均值回归收益。
        - max_lots=1        : 研究阶段保守单手，聚焦评估信号质量本身。
        """
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.5,
            tp1_pct=0.4,
            tp2_pct=0.8,
            max_lots=1,
        )
