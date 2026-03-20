"""
Strategy #29 — Price Oscillator Extreme Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 价格振荡器 (PO) 的 Z-Score 处于极端区间时，快慢 EMA 之间的价差
  相对于近期历史严重偏离，动量过度延伸，价差趋于收敛 → 均值回归入场

  价格振荡器 (Price Oscillator, PO) 定义:
    PO = (EMA_fast - EMA_slow) / EMA_slow
    即快速EMA相对慢速EMA的百分比偏差。例如 PO = 0.005 表示快速EMA
    高于慢速EMA约0.5%。相比原始MACD (绝对差值)，PO为百分比归一化，
    跨合约/跨价位的可比性更强，Z-Score阈值参数无需随价格水平调整。

  Z-Score 规范化:
    po_mean    = rolling(zscore_window).mean(po.shift(1))  ← 防前瞻
    po_std     = rolling(zscore_window).std(po.shift(1))   ← 防前瞻
    po_zscore  = (po - po_mean) / po_std

    当 po_zscore > +po_threshold: 快速EMA相对慢速EMA的正向价差极大，
      说明短期动量过度向上延伸 → 回落至均值的概率高 → 做空信号
    当 po_zscore < -po_threshold: 快速EMA相对慢速EMA的负向价差极大，
      说明短期动量过度向下延伸 → 反弹至均值的概率高 → 做多信号

  与传统MACD策略的三点本质区别:
    ① 百分比归一化 (PO) 而非绝对差值 (MACD)，使阈值参数具备价格无关性
    ② 入场基于 Z-Score (极端偏离)，而非 MACD 零轴上下穿越 (方向判断)
    ③ 适应性强: 在趋势市 (价差持续扩张后极度延伸) 和震荡市 (价差来回偏摆)
       中均能捕捉到"快慢EMA价差回均"的机会

  核心市场假设:
    无论趋势方向如何，当短期动量指标 (快速EMA) 与长期趋势 (慢速EMA) 的
    价差相对自身历史分布严重偏离时，这种偏离往往具有均值回归特性。
    市场参与者的止盈行为、程序化追涨/杀跌的反向力量均会促使价差收敛。

  入场逻辑:
    做多: po_zscore < -po_threshold  → 价差极度负向延伸，反弹概率高
    做空: po_zscore > +po_threshold  → 价差极度正向延伸，回落概率高

  出场逻辑 (有状态):
    持多且 po_zscore > -0.5:   价差已向中性区回归，做多逻辑消解 → 平仓
    持空且 po_zscore < +0.5:   价差已向中性区回归，做空逻辑消解 → 平仓
    中性区定义: |po_zscore| < 0.5 视为价差已充分回归均值

  EMA 计算 (全部防前瞻):
    - ema_fast/ema_slow 使用 pandas ewm(span=period, adjust=False).mean()
    - PO 在 ema_slow 上做除法，使用 shift(1) 前已逐 bar 计算，无前瞻
    - po_mean / po_std 使用 po.shift(1).rolling(zscore_window) 以保证
      当前 bar 的统计量仅依赖历史 PO 序列

  NaN 安全处理:
    - ema_slow 为 0 或 NaN 时 PO 置 NaN
    - po_std 为 0 或 NaN 时 po_zscore 置 NaN
    - po_zscore 含 NaN 的 bar: 入场候选强制清除 (False)
    - 预热期 (前 slow_period + zscore_window + 5 根 bar): 信号全置 0

  参数设计 (3个, 27种组合, WFO 可承受):
    - fast_period  : 快速EMA周期  [5, 8, 12]      约25min ~ 1h
    - slow_period  : 慢速EMA周期  [20, 30, 50]    约1.7h ~ 4.2h
    - po_threshold : Z-Score入场阈值 [1.5, 2.0, 2.5]
    - zscore_window: 固定为30根bar，不纳入参数网格

  适用环境: 铁矿石5分钟行情中动量阶段性过冲、价差均值回归特征显著的行情
  风险提示: 强单边趋势中 po_zscore 可能持续处于极端区，连续触发多次同向
            入场亏损；建议配合较严格的硬止损和移动止损参数

  回测参数:
    - 数据频率: 5分钟K线
    - hard_stop_pct : 0.6%  — 防止极端趋势行情中价差持续扩张的深度亏损
    - trailing_pct  : 0.5%  — 价差收敛过程中锁定浮动盈利
    - tp1_pct       : 0.5%  — 半仓止盈，均值回归早期降低风险敞口
    - tp2_pct       : 1.0%  — 余仓止盈，捕获价差完整收敛的最大收益
    - max_lots      : 1     — 研究阶段保守单手配置
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams

# zscore_window is fixed — not part of the optimisation grid
_ZSCORE_WINDOW: int = 30


class PriceOscillatorReversion(BaseResearchStrategy):
    """价格振荡器极端区间均值回归策略 — S29。

    计算快慢EMA之间百分比价差 (PO) 的滚动 Z-Score，在 Z-Score 处于
    极端区间时入场做均值回归，价差回归中性区 (|Z| < 0.5) 时出场。
    相比 MACD 穿越策略，本策略信号基于价差的统计极端程度而非方向，
    适合铁矿石日内动量过冲行情。
    """

    name = "Price Oscillator Extreme"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回 3 参数网格，共 3×3×3 = 27 个参数组合。

        fast_period : int
            快速EMA周期。5 → ~25min EMA (对短期价格波动高度敏感)；
            12 → ~1h EMA (较平滑，减少噪声触发)。
        slow_period : int
            慢速EMA周期。20 → ~1.7h EMA (近期中期趋势基准)；
            50 → ~4.2h EMA (较长期趋势基准，PO波动幅度更大)。
        po_threshold : float
            PO Z-Score入场阈值。1.5 → 信号频率较高 (1.5σ极端)；
            2.5 → 信号保守 (2.5σ极端，质量更高但信号稀少)。
        """
        return {
            "fast_period":  [5, 8, 12],         # 快速EMA周期
            "slow_period":  [20, 30, 50],        # 慢速EMA周期
            "po_threshold": [1.5, 2.0, 2.5],    # Z-Score入场阈值
        }

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        fast_period: int = 8,
        slow_period: int = 30,
        po_threshold: float = 2.0,
    ) -> np.ndarray:
        """生成基于价格振荡器 Z-Score 极端值的均值回归信号。

        实现分三步:
        1. 计算快慢 EMA，得到百分比价格振荡器 (PO)。
        2. 对 PO 序列做 shift(1) 后计算滚动均值和标准差，求 Z-Score。
        3. 有状态循环: Z-Score 超过阈值时入场，回归中性区时出场。

        Parameters
        ----------
        df : pd.DataFrame
            含 'close' 列的 OHLCV DataFrame，index 为时间序列。
        fast_period : int
            快速 EMA 的 span 参数 (默认 8)。
        slow_period : int
            慢速 EMA 的 span 参数 (默认 30)。需大于 fast_period。
        po_threshold : float
            触发入场的 PO Z-Score 绝对值阈值 (默认 2.0)。

        Returns
        -------
        np.ndarray of int8
            与 df 等长的信号数组:
               1  = 做多 (含持仓维持)
              -1  = 做空 (含持仓维持)
               0  = 空仓 / 无信号
               2  = 强制出场 (PO Z-Score 回归中性区)
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        n = len(df)
        close_s = df["close"].astype(np.float64)

        # ------------------------------------------------------------------
        # 步骤1: 计算快慢 EMA 和 Price Oscillator (PO)
        # ------------------------------------------------------------------
        ema_fast = close_s.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close_s.ewm(span=slow_period, adjust=False).mean()

        # PO = (EMA_fast - EMA_slow) / EMA_slow (百分比归一化)
        # ema_slow 为 0 或 NaN 时 PO 保持 NaN (由 pandas 自然处理)
        with np.errstate(divide="ignore", invalid="ignore"):
            po = (ema_fast - ema_slow) / ema_slow.replace(0.0, np.nan)

        # ------------------------------------------------------------------
        # 步骤2: PO 的滚动 Z-Score (shift(1) 防前瞻)
        #   po_mean / po_std 仅依赖历史 PO 值，当前 bar 的 PO 不参与统计量
        # ------------------------------------------------------------------
        po_shifted = po.shift(1)
        po_mean = po_shifted.rolling(_ZSCORE_WINDOW).mean()
        po_std  = po_shifted.rolling(_ZSCORE_WINDOW).std()

        # po_std 为 0 或 NaN 时 po_zscore 置 NaN (安全: 无信号)
        with np.errstate(divide="ignore", invalid="ignore"):
            po_zscore = (po - po_mean) / po_std.replace(0.0, np.nan)

        # 转为 numpy 数组加速循环
        po_zscore_arr   = po_zscore.values
        close_arr       = close_s.values

        # ------------------------------------------------------------------
        # 步骤3: 向量化预计算入场候选
        # ------------------------------------------------------------------
        # NaN 安全: po_zscore 含 NaN 的 bar 不触发入场
        valid = np.isfinite(po_zscore_arr)

        # 做多: Z-Score 极度负向 → 快EMA远低于慢EMA → 价差将向上收敛
        long_entry_arr  = valid & (po_zscore_arr < -po_threshold)
        # 做空: Z-Score 极度正向 → 快EMA远高于慢EMA → 价差将向下收敛
        short_entry_arr = valid & (po_zscore_arr >  po_threshold)

        # ------------------------------------------------------------------
        # 步骤4: 有状态循环 — 入场 / 持仓维持 / 中性区出场
        # ------------------------------------------------------------------
        signals = np.zeros(n, dtype=np.int8)
        active: int = 0   # +1 = 持多, -1 = 持空, 0 = 空仓

        for i in range(n):
            z = po_zscore_arr[i]

            # ---- 持多: 检查 PO Z-Score 是否回归中性区 ----------------------
            if active == 1:
                # po_zscore > -0.5: 价差负向极端已消解，动量已充分正常化
                if np.isfinite(z) and z > -0.5:
                    signals[i] = 2   # 强制平多
                    active = 0
                else:
                    signals[i] = 1   # 维持多头
                continue

            # ---- 持空: 检查 PO Z-Score 是否回归中性区 ----------------------
            if active == -1:
                # po_zscore < +0.5: 价差正向极端已消解，动量已充分正常化
                if np.isfinite(z) and z < 0.5:
                    signals[i] = 2   # 强制平空
                    active = 0
                else:
                    signals[i] = -1  # 维持空头
                continue

            # ---- 空仓: 检查入场候选 -----------------------------------------
            if long_entry_arr[i]:
                signals[i] = 1
                active = 1
            elif short_entry_arr[i]:
                signals[i] = -1
                active = -1
            # 否则: 空仓且无信号，signals[i] 保持 0

        # ------------------------------------------------------------------
        # NaN safety — 预热期清零
        # 前 slow_period + zscore_window + 5 根 bar 的 EMA/Z-Score 不稳定
        # ------------------------------------------------------------------
        warmup = slow_period + _ZSCORE_WINDOW + 5
        signals[:warmup] = 0

        return signals

    # ------------------------------------------------------------------
    # 持仓参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """返回适合价格振荡器均值回归策略的风控参数。

        参数设计理由:
        - hard_stop_pct=0.6 : 强单边趋势中 PO 可能持续扩张，0.6% 硬止损
                              在价差未能及时收敛时限制单笔最大损失。
        - trailing_pct=0.5  : PO 价差收敛过程中以 0.5% 移动止损保护
                              已积累的浮动盈利，防止二次反转磨损。
        - tp1_pct=0.5       : 半仓止盈 0.5%，均值回归早期锁定部分收益，
                              与铁矿石5分钟价差收敛幅度匹配。
        - tp2_pct=1.0       : 余仓止盈 1.0%，捕获 PO 从极端值完整回归
                              至中性区的最大收益。
        - max_lots=1        : 研究阶段保守单手，聚焦评估 Z-Score 信号
                              质量本身，不引入仓位管理变量干扰。
        """
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
