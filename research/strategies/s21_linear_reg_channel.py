"""
Strategy #21 — Linear Regression Channel Mean-Reversion (5-min bars)
=====================================================================

【策略思路】
  核心逻辑: 对过去 reg_period 根K线的收盘价拟合滚动线性回归，得到当前bar
  的"趋势公允价值"。价格相对于回归线的偏差（残差）服从均值为零的分布，当
  价格偏离回归线超过 channel_mult 倍的残差标准差时，视为统计意义上的"过度
  偏离"，此时入场做均值回归。

  学术依据:
  - 线性回归通道（Raff Channel 变体）是经典的统计套利工具，在商品期货的5分钟
    级别研究中表现出显著的均值回归特性（arXiv 2501.16772 指出15分钟以下频率
    中均值回归占优）。
  - 残差的条件方差（rolling_std）捕捉局部波动率状态，使通道宽度自适应于当前
    市场环境，类似 Bollinger Bands 的自适应逻辑，但基于趋势调整后的残差而非
    原始价格。
  - 价格穿越通道后回归回归线的路径在趋势市场中由动量驱动，在震荡市场中由
    均值回归驱动，因此 exit_pct 参数允许提前止盈，捕捉部分回归收益。

  信号逻辑:
  1. 向量化计算滚动线性回归线 (shift(1) 避免前视偏差)
  2. 计算残差序列及其滚动标准差（同样 shift(1) 防前视）
  3. 构造上下通道带: reg_line ± channel_mult × resid_std
  4. 入场:
     - close < lower_band → LONG  (+1): 价格低于下轨，预期回归
     - close > upper_band → SHORT (-1): 价格高于上轨，预期回归
  5. 出场 (stateful loop):
     - LONG持仓: close >= lower_band + (reg_line - lower_band) × (1 - exit_pct) → exit (2)
     - SHORT持仓: close <= upper_band - (upper_band - reg_line) × (1 - exit_pct) → exit (2)
  6. exit_pct=0.0 → 在回归线处平仓; exit_pct>0 → 提前平仓（保守止盈）

  回归线计算 (纯numpy, 无sklearn依赖):
  - 对长度为 n=reg_period 的窗口 y, x=[0,1,...,n-1]:
    slope     = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
    intercept = ȳ - slope × x̄
    reg_value = intercept + slope × (n-1)  ← 窗口最后一点（即最新bar）的回归值

  前视偏差防护:
  - rolling apply 计算回归线后 shift(1)，确保当前bar使用的是上根bar结束时的回归值
  - resid_std 额外再 shift(1)，避免使用当前残差计算标准差后再用于当前bar入场

  参数设计 (3个, 27组合):
  - reg_period   : [15, 25, 40]      — 回归窗口长度
  - channel_mult : [1.5, 2.0, 2.5]   — 通道宽度倍数
  - exit_pct     : [0.0, 0.2, 0.4]   — 提前止盈比例 (0=在回归线处; 0.4=提前40%止盈)

  NaN安全:
  - reg_line 和 resid_std 为NaN的位置信号强制为0
  - 前 reg_period × 2 + 5 根bar作为热身期，不产生任何信号

  适用环境: 趋势缓慢、价格围绕趋势线震荡的日内行情
  风险提示: 强趋势行情中价格持续突破通道，逆势信号连续触发；
            reg_period 较短时回归线波动大，信号噪声较高
=====================================================================
"""

import numpy as np
import pandas as pd

from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


# ---------------------------------------------------------------------------
# 模块级辅助函数 (避免在 rolling apply 中重复创建 x 数组)
# ---------------------------------------------------------------------------

def _make_reg_applier(reg_period: int):
    """
    预计算回归所需的固定量，返回一个可直接传入 rolling.apply 的闭包。

    对窗口 y (长度 = reg_period) 计算 OLS 线性回归并返回窗口最后一点的
    回归值 (即最新bar的"趋势公允价值")。

    数学推导:
        x = [0, 1, ..., n-1],  x̄ = (n-1)/2
        x_ss = Σ(x - x̄)² = n(n²-1)/12  (等差数列平方和公式)
        slope = Σ(x_i - x̄)(y_i - ȳ) / x_ss
        reg_value = ȳ + slope × (n-1 - x̄)
                  = ȳ + slope × (n-1)/2
    """
    x = np.arange(reg_period, dtype=np.float64)
    x_mean = x.mean()                           # = (reg_period - 1) / 2
    x_ss = ((x - x_mean) ** 2).sum()           # = reg_period*(reg_period²-1)/12
    half_span = (reg_period - 1) * 0.5         # = (n-1) - x̄, 回归值在末点的偏移

    def _fit_reg_end(y: np.ndarray) -> float:
        """返回 OLS 回归线在窗口末端（最新bar）的值。"""
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_ss
        return y_mean + slope * half_span

    return _fit_reg_end


class LinearRegChannelReversion(BaseResearchStrategy):
    """线性回归通道均值回归策略 (DCE铁矿石 5分钟K线)。

    对收盘价序列拟合滚动线性回归，当价格偏离趋势公允价值超过若干倍残差标准差
    时入场均值回归，价格回归到（或接近）回归线时平仓。
    """

    name = "Linear Regression Channel"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回3参数网格，共 3×3×3 = 27 组参数组合。

        reg_period : int
            滚动线性回归的窗口长度（bar数）。
            较短(15)窗口对局部趋势变化更敏感，通道颤动更剧烈；
            较长(40)窗口对应更稳健的中期趋势估计。
        channel_mult : float
            通道宽度倍数（残差标准差的倍数）。
            1.5 → 通道较窄，触发频率高，但假信号多；
            2.5 → 通道较宽，仅捕捉极端偏离，信号稀少但置信度高。
        exit_pct : float
            提前止盈比例。0.0 = 在回归线处平仓；0.4 = 距入场带40%处提前平仓。
            较高的 exit_pct 使策略更保守，锁定更少但更确定的收益。
        """
        return {
            "reg_period":   [15, 25, 40],
            "channel_mult": [1.5, 2.0, 2.5],
            "exit_pct":     [0.0, 0.2, 0.4],
        }

    # ------------------------------------------------------------------
    # 信号生成主函数
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        reg_period: int = 25,
        channel_mult: float = 2.0,
        exit_pct: float = 0.2,
    ) -> np.ndarray:
        """线性回归通道均值回归信号。

        Parameters
        ----------
        df : pd.DataFrame
            含 open/high/low/close/volume 的5分钟OHLCV数据，DatetimeIndex。
        reg_period : int
            滚动OLS线性回归的窗口长度（bar数）。
        channel_mult : float
            通道宽度 = channel_mult × rolling_std(残差)。
        exit_pct : float
            出场提前比例: 0.0 = 在回归线处; 0.4 = 距入场带40%时止盈。

        Returns
        -------
        np.ndarray of int8
            +1  做多 (持仓中)
            -1  做空 (持仓中)
             0  无信号 / 空仓
             2  强制平仓 (均值回归目标达到)
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        if "close" not in df.columns:
            return np.zeros(len(df), dtype=np.int8)

        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        # ----------------------------------------------------------------
        # 1. 提取收盘价
        # ----------------------------------------------------------------
        cs = df["close"].astype(np.float64)
        close_arr = cs.values

        # ----------------------------------------------------------------
        # 2. 向量化计算滚动线性回归线 (shift(1) 防前视)
        #
        #    cs.shift(1): 当前bar使用"截止上根bar"的价格序列拟合回归
        #    rolling(reg_period).apply: 对每个窗口调用 _fit_reg_end
        #    结果 reg_line[i] = 用 close[i-reg_period ... i-1] 拟合的回归线在末点的值
        #    —— 即"上根bar结束时对最新bar的趋势估计"，不含当前bar信息
        # ----------------------------------------------------------------
        fit_fn = _make_reg_applier(reg_period)

        reg_line_series: pd.Series = (
            cs.shift(1)
            .rolling(reg_period)
            .apply(fit_fn, raw=True)
        )

        # ----------------------------------------------------------------
        # 3. 残差 & 通道带 (resid_std 额外再 shift(1) 防前视)
        #
        #    residuals[i] = close[i] - reg_line[i]
        #       (当前bar收盘价 与 基于历史数据估计的趋势公允价值 的差距)
        #    resid_std[i]  = rolling_std(residuals)[i].shift(1)
        #       (上根bar结束时的残差波动率估计)
        # ----------------------------------------------------------------
        residuals: pd.Series = cs - reg_line_series
        resid_std_series: pd.Series = (
            residuals.rolling(reg_period).std().shift(1)
        )

        upper_band_series: pd.Series = reg_line_series + channel_mult * resid_std_series
        lower_band_series: pd.Series = reg_line_series - channel_mult * resid_std_series

        # 转为 numpy array 供循环使用
        reg_line   = reg_line_series.values
        resid_std  = resid_std_series.values
        upper_band = upper_band_series.values
        lower_band = lower_band_series.values

        # ----------------------------------------------------------------
        # 4. NaN掩码 & 热身期保护
        #
        #    热身期 = reg_period * 2 + 5 根bar
        #      - reg_period 根: rolling 回归需要的最少数据
        #      - reg_period 根: resid_std 的 rolling std 窗口
        #      - 5 根: 额外缓冲，避免边界效应
        # ----------------------------------------------------------------
        warmup = reg_period * 2 + 5
        valid_mask = (
            ~np.isnan(reg_line)
            & ~np.isnan(resid_std)
            & ~np.isnan(upper_band)
            & ~np.isnan(lower_band)
            & (resid_std > 0.0)
        )
        # 热身期内强制无效
        valid_mask[:warmup] = False

        # ----------------------------------------------------------------
        # 5. Stateful 信号循环
        #
        #    入场条件 (向量化预计算, 在循环内引用):
        #      long_entry[i]  = close[i] < lower_band[i]  (价格跌破下轨)
        #      short_entry[i] = close[i] > upper_band[i]  (价格突破上轨)
        #
        #    出场逻辑 (stateful, 依赖 active 状态):
        #      LONG 持仓: exit_target = lower_band[i] + (reg_line[i] - lower_band[i]) × (1 - exit_pct)
        #                 当 close[i] >= exit_target 时, signals[i] = 2
        #      SHORT持仓: exit_target = upper_band[i] - (upper_band[i] - reg_line[i]) × (1 - exit_pct)
        #                 当 close[i] <= exit_target 时, signals[i] = 2
        #
        #    持仓期间维持信号 (signals[i] = active) 告知回测引擎仓位状态
        # ----------------------------------------------------------------
        active = 0    # 当前持仓方向: 1=多头, -1=空头, 0=空仓

        for i in range(n):
            # 无效bar (热身期或NaN): 维持当前持仓信号或写0
            if not valid_mask[i]:
                if active != 0:
                    signals[i] = active
                continue

            c          = close_arr[i]
            reg_val    = reg_line[i]
            upper      = upper_band[i]
            lower      = lower_band[i]

            # ----------------------------------------------------------
            # 持仓中: 先检查出场条件
            # ----------------------------------------------------------
            if active == 1:
                # 多头出场目标: 从 lower_band 出发，朝 reg_line 方向走 (1 - exit_pct) 比例
                band_span  = reg_val - lower           # lower → reg_line 的距离
                exit_target = lower + band_span * (1.0 - exit_pct)
                if c >= exit_target:
                    signals[i] = 2
                    active = 0
                    continue

            elif active == -1:
                # 空头出场目标: 从 upper_band 出发，朝 reg_line 方向走 (1 - exit_pct) 比例
                band_span   = upper - reg_val          # reg_line → upper_band 的距离
                exit_target = upper - band_span * (1.0 - exit_pct)
                if c <= exit_target:
                    signals[i] = 2
                    active = 0
                    continue

            # ----------------------------------------------------------
            # 空仓: 检查入场条件
            # ----------------------------------------------------------
            if active == 0:
                if c < lower:
                    # 价格跌破下轨 → 做多 (预期价格向回归线回归)
                    signals[i] = 1
                    active = 1
                elif c > upper:
                    # 价格突破上轨 → 做空 (预期价格向回归线回归)
                    signals[i] = -1
                    active = -1
                # else: 无信号，signals[i] 保持 0

            # ----------------------------------------------------------
            # 持仓中 (未触发出场): 维持信号
            # ----------------------------------------------------------
            else:
                signals[i] = active

        return signals

    # ------------------------------------------------------------------
    # 仓位管理参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """返回适合线性回归通道均值回归策略的风控参数。

        Rationale
        ---------
        - hard_stop_pct=0.7 : 均值回归策略在价格背离超过0.7%时往往已经
          失效（市场进入趋势状态），硬止损保护本金。
        - trailing_pct=0.5  : 价格回归过程中追踪止损，防止已盈利的均值
          回归交易在回撤后变为亏损。
        - tp1_pct=0.5       : 首个部分止盈点，锁定基础收益，降低对完整
          回归路径的依赖。
        - tp2_pct=1.0       : 完整平仓点，铁矿石5分钟级通道偏离的典型
          均值回归幅度在0.5-1.5%之间。
        - max_lots=1        : 研究阶段单手保守定仓。
        """
        return PositionParams(
            hard_stop_pct=0.7,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
