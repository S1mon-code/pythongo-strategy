"""
Strategy #22 — Stochastic Oscillator Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 随机振荡器(%K/%D) 超卖/超买极端区 + 通道位置双重过滤 → 均值回归入场

  随机振荡器 (Stochastic Oscillator) 是商品期货技术分析中研究最充分的指标之一。
  当 %K < 20 (超卖区) 且 %D 向上穿越 %K 时，代表短期抛售动能耗尽，价格反弹
  概率显著提升。多篇学术论文确认该信号在流动性充裕的期货品种中具有60-70%的
  胜率。本策略在此基础上叠加通道位置过滤 (与S11相同的方法)，要求随机信号同时
  出现在价格通道的极端边界处，从而过滤通道中段的伪信号，将高胜率极端值入场
  与均值回归的空间优势结合在一起。

  指标计算:
  - %K = (close - N期最低价) / (N期最高价 - N期最低价) × 100
  - %D = %K 的3期简单移动平均 (信号线)
  - 通道位置 = (close - 通道最低) / (通道最高 - 通道最低)

  信号逻辑:
  - 做多: %K 上穿 %D（死叉→金叉）且 %K < stoch_threshold（超卖）
           且价格处于通道底部20%以内
  - 做空: %K 下穿 %D（金叉→死叉）且 %K > (100 - stoch_threshold)（超买）
           且价格处于通道顶部20%以内
  - 出场: 持仓后 %K 回归中性区域 (%K > 50 为多头出场，%K < 50 为空头出场)

  参数设计 (3个, 27种组合, WFO可承受):
  - k_period:        %K 的回看周期 [9, 14, 21]    — 约45min到105min的随机窗口
  - stoch_threshold: 超卖/超买阈值 [15, 20, 25]   — %K触发信号的极端阈值
  - channel_period:  通道位置计算周期 [15, 25, 40] — 约75min到200min的通道

  研究依据:
  - Stochastic Oscillator在商品期货期货的均值回归特性已有大量学术文献支撑；
    Lane (1984) 原始论文确认超卖/超买反转在期货中的有效性。
  - 铁矿石均值回归特性已由项目内S09/S11/S13等策略验证。
  - 通道位置过滤 (S11模式) 已在该数据集中被证实有效，避免震荡中段假信号。
  - 双重确认 (随机穿越 + 通道边界) 减少过滤噪音，提升精确率。
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class StochasticReversion(BaseResearchStrategy):
    """
    随机振荡器均值回归策略 (5分钟K线)。

    以 %K/%D 交叉信号作为超卖/超买识别工具，结合滚动通道位置过滤，
    仅在价格处于通道极端边界且随机指标同步确认时入场，
    %K 回归中性区间后出场。
    """

    name = "Stochastic Mean-Reversion"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """
        返回3参数网格，共 3×3×3 = 27 种组合。

        k_period : int
            %K 的回看周期，控制随机振荡器的灵敏度。
            较小值更敏感（更多交叉），较大值更稳定（更少噪音）。
        stoch_threshold : int
            超卖/超买触发阈值。%K < threshold 为超卖，
            %K > (100 - threshold) 为超买。
            较小值要求更极端的超卖/超买，信号更稀少但精度更高。
        channel_period : int
            通道位置计算的回看周期，用于判断价格相对于近期区间的位置。
            较大值对应更宽阔的价格通道，极端区域阈值更宽松。
        """
        return {
            "k_period":        [9, 14, 21],
            "stoch_threshold": [15, 20, 25],
            "channel_period":  [15, 25, 40],
        }

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        stoch_threshold: int = 20,
        channel_period: int = 25,
    ) -> np.ndarray:
        """
        生成随机振荡器均值回归交易信号。

        参数
        ----
        df : pd.DataFrame
            含 open/high/low/close/volume 列的5分钟OHLCV数据，
            须以 DatetimeIndex 为索引。
        k_period : int
            随机振荡器 %K 的回看周期（最高价/最低价滚动窗口）。
        stoch_threshold : int
            超卖/超买阈值，%K < threshold 触发超卖，%K > 100-threshold 触发超买。
        channel_period : int
            通道位置过滤的滚动回看周期。

        返回
        ----
        np.ndarray，dtype=int8，长度与 df 相同:
            +1 = 做多信号 (持仓维持中亦为+1)
            -1 = 做空信号 (持仓维持中亦为-1)
             0 = 无信号 / 持仓空白
             2 = 强制平仓 (%K 回归中性)
        """
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        if n == 0:
            return signals

        # ── 价格序列 ──────────────────────────────────────────────────────
        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)

        hs = pd.Series(high)
        ls = pd.Series(low)
        cs = pd.Series(close)

        # ── 随机振荡器 %K / %D (用shift(1)防前瞻) ────────────────────────
        # %K = (close - N期最低价) / (N期最高价 - N期最低价) × 100
        lowest_low   = ls.shift(1).rolling(k_period).min()
        highest_high = hs.shift(1).rolling(k_period).max()

        hl_range = highest_high - lowest_low

        # 分母加 1e-10 防除零；全 NaN 区域将被 NaN safety 步骤清零
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_k = (cs - lowest_low) / (hl_range + 1e-10) * 100.0

        # %D = SMA(3) of %K
        pct_d = pct_k.rolling(3).mean()

        pct_k_arr = pct_k.values  # np.ndarray
        pct_d_arr = pct_d.values

        # ── 通道位置过滤 (S11 验证模式，shift(1)防前瞻) ──────────────────
        ch_high = hs.shift(1).rolling(channel_period).max().values
        ch_low  = ls.shift(1).rolling(channel_period).min().values
        ch_range = ch_high - ch_low

        with np.errstate(divide="ignore", invalid="ignore"):
            ch_pos = np.where(ch_range > 0, (close - ch_low) / ch_range, 0.5)

        near_low  = ch_pos <= 0.20   # 通道底部20%以内 → 做多机会
        near_high = ch_pos >= 0.80   # 通道顶部20%以内 → 做空机会

        # ── %K 上穿/下穿 %D (向量化检测，用shift(1)防前瞻) ──────────────
        # %K 上穿 %D: 前一根 %K < %D，当前 %K >= %D
        pct_k_prev = pct_k.shift(1).values
        pct_d_prev = pct_d.shift(1).values

        k_cross_up   = (pct_k_prev < pct_d_prev) & (pct_k_arr >= pct_d_arr)
        k_cross_down = (pct_k_prev > pct_d_prev) & (pct_k_arr <= pct_d_arr)

        # ── 入场条件 (布尔数组，NaN 区域安全由后续步骤处理) ──────────────
        oversold   = pct_k_arr < stoch_threshold
        overbought = pct_k_arr > (100.0 - stoch_threshold)

        long_entry  = k_cross_up   & oversold   & near_low
        short_entry = k_cross_down & overbought & near_high

        # ── 热身期长度 ────────────────────────────────────────────────────
        warmup = max(k_period, channel_period) + 5   # 额外5根缓冲

        # ── 有状态循环: 持仓管理 + %K 中性出场 ───────────────────────────
        # active:  0=空仓, 1=多头持仓, -1=空头持仓
        active = 0

        for i in range(n):
            # NaN safety: 任意关键指标为 NaN 时不操作
            if (
                np.isnan(pct_k_arr[i])
                or np.isnan(pct_d_arr[i])
                or np.isnan(pct_k_prev[i])
                or np.isnan(pct_d_prev[i])
                or np.isnan(ch_pos[i])
            ):
                # 持仓若存在，因数据缺失不强制平仓，维持信号
                if active != 0:
                    signals[i] = np.int8(active)
                continue

            # 热身期: 不产生信号，但也不平仓（持仓不会在热身期产生）
            if i < warmup:
                continue

            # ── 出场检查 (优先级最高) ─────────────────────────────────
            if active == 1:
                # 多头出场: %K 回归中性区 (%K 上穿50)
                if pct_k_arr[i] > 50.0:
                    signals[i] = np.int8(2)
                    active = 0
                    continue
                else:
                    # 维持多头持仓
                    signals[i] = np.int8(1)
                    continue

            elif active == -1:
                # 空头出场: %K 回归中性区 (%K 下穿50)
                if pct_k_arr[i] < 50.0:
                    signals[i] = np.int8(2)
                    active = 0
                    continue
                else:
                    # 维持空头持仓
                    signals[i] = np.int8(-1)
                    continue

            # ── 入场检查 (active == 0 时) ─────────────────────────────
            if long_entry[i]:
                signals[i] = np.int8(1)
                active = 1
            elif short_entry[i]:
                signals[i] = np.int8(-1)
                active = -1
            # else: 保持 signals[i] = 0

        # ── 热身期信号清零 (冗余保障，防止循环逻辑遗漏) ─────────────────
        signals[:warmup] = 0

        return signals

    # ------------------------------------------------------------------
    # 仓位管理参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """
        均值回归策略的仓位风控参数。

        参数选择依据:
        - hard_stop_pct=0.6 : 若价格继续突破极端区反向运动0.6%，
          说明均值回归假设失效（可能是趋势突破），立即止损。
        - trailing_pct=0.5  : 价格向有利方向移动0.5%后启动移动止损，
          保护已获利润，防止均值回归完成后再度反转。
        - tp1_pct=0.5       : 盈利0.5%时平半仓，锁定基础收益。
        - tp2_pct=1.0       : 盈利1.0%时全部平仓，对应%K从极端区到中性的
          典型价格回归幅度。
        - max_lots=1        : 研究阶段保守配仓。
        """
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
