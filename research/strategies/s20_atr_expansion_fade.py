"""
Strategy #20 — ATR Expansion Fade (Session Overextension Reversal, 5-min bars)
================================================================================

【策略思路】
  核心逻辑: 当日内价格波动幅度超过滚动ATR的若干倍时，说明该session出现了
  "过度延伸"行情，此类单session极端波动在商品期货中具有较高的均值回归概率。

  学术依据:
  - 大量商品期货研究表明单session异常大幅波动后存在显著的部分回归效应，
    尤其是DCE铁矿石等流动性充裕的合约。
  - 市场微观结构视角：日内单边极端行情往往由短期流动性冲击驱动，而非基本面
    信息，因此价格随后的反转被视为"过度反应修正"。
  - 信号有效区间为session后半段，此时session_range已经充分展开，
    且session内剩余时间足够让均值回归发生。

  Session窗口 (5分钟K线):
  - Day1: 09:00 - 11:30  (150分钟 = 30根5分钟K线)
  - Day2: 13:30 - 15:00  (90分钟  = 18根5分钟K线)
  夜盘不参与，仅交易日盘。

  信号逻辑:
  1. 计算Wilder平滑ATR(atr_period周期)
  2. 每个session实时追踪 session_high / session_low / session_open
  3. session后半段 (bar_in_session >= min(10, total_session_bars // 2)):
     - 若 session_range > atr_mult × ATR:
       * 收盘价处于session区间顶部20%以内 → SHORT (做空过度上行)
       * 收盘价处于session区间底部20%以内 → LONG  (做多过度下行)
  4. 出场条件:
     - 价格回撤 retracement_pct × session_range → signal = 2
     - session结束强制平仓 → signal = 2 写在session最后一根bar

  ATR计算 (Wilder平滑法，不依赖ta-lib):
  - TR[i] = max(high-low, |high-prev_close|, |low-prev_close|)
  - ATR[0] = mean(TR[:atr_period])  (SMA种子)
  - ATR[i] = ATR[i-1] × (n-1)/n + TR[i] / n

  参数设计 (3个, 18组合):
  - atr_mult    : [1.2, 1.5, 2.0]  — session_range超过ATR的倍数阈值
  - retracement : [0.3, 0.5]       — 目标回撤比例 (相对于session_range)
  - atr_period  : [10, 14, 20]     — Wilder ATR平滑周期

  关键设计决策:
  - ATR在进入循环前向量化计算，warmup期 (前atr_period根) 内不产生信号
  - session_high / session_low 在每根bar处实时更新，target在入场时锁定
  - signal = +1/-1 在持仓期间每根bar持续写入，告知回测引擎仓位状态
  - signal = 2 (强制平仓) 写在 i-1 (session内最后一根bar)，
    i 为第一根不属于当前session的bar
  - 同一session内已有仓位时不重复开仓 (signal_direction != 0 保护)
  - NaN安全: ATR热身期及prev_close为NaN时 overextended 判断均为False

  适用环境: 日内震荡收敛行情，session内出现情绪化单边急涨急跌后的修复
  风险提示: 若市场出现连续单边趋势 (如重大政策冲击)，逆势信号持续亏损；
            atr_mult=1.2 阈值较低时信号频率较高，噪音交易占比上升
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class AtrExpansionFade(BaseResearchStrategy):
    name = "ATR Expansion Fade"
    freq = "5min"

    def param_grid(self) -> dict:
        return {
            "atr_mult":    [1.2, 1.5, 2.0],   # session_range / ATR 触发阈值
            "retracement": [0.3, 0.5],         # 均值回归目标 (session_range 的比例)
            "atr_period":  [10, 14, 20],       # Wilder ATR 周期
        }

    # ------------------------------------------------------------------
    # ATR 计算 (Wilder平滑法，无ta-lib依赖)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_wilder_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """
        Wilder平滑真实波幅 (ATR)。

        TR[i] = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR种子 = mean(TR[:period])
        ATR[i]  = ATR[i-1] * (period-1)/period + TR[i] / period

        warmup期 (i < period) 返回 NaN，确保信号安全。
        """
        n = len(close)
        prev_close = np.empty(n, dtype=np.float64)
        prev_close[0] = np.nan
        prev_close[1:] = close[:-1]

        # True Range — NaN传播安全：第0根prev_close为NaN → tr[0]=NaN
        tr = np.where(
            np.isnan(prev_close),
            np.nan,
            np.maximum(
                high - low,
                np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
            ),
        )

        atr = np.full(n, np.nan, dtype=np.float64)

        # 种子: SMA of first `period` valid TR values (index 1..period)
        # tr[0] is NaN so we start from index 1
        if n > period:
            seed_slice = tr[1 : period + 1]                # period values
            if not np.any(np.isnan(seed_slice)):
                atr[period] = np.mean(seed_slice)           # 种子落在 index=period
                alpha = 1.0 / period
                for i in range(period + 1, n):
                    if np.isnan(tr[i]):
                        atr[i] = np.nan
                    else:
                        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha

        return atr

    # ------------------------------------------------------------------
    # 信号生成主函数
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        atr_mult: float = 1.5,
        retracement: float = 0.3,
        atr_period: int = 14,
    ) -> np.ndarray:
        """
        ATR扩张反转信号。

        参数:
            df           : 含 open/high/low/close/volume 的5分钟OHLCV数据
            atr_mult     : session_range 超过 atr_mult × ATR 时视为过度延伸
            retracement  : 目标为 session_range × retracement 的价格回归幅度
            atr_period   : Wilder ATR 的平滑周期

        返回:
            signals: np.ndarray(dtype=int8)
                +1 = 做多信号 (fade下行过度延伸)
                -1 = 做空信号 (fade上行过度延伸)
                 0 = 无信号
                 2 = 强制平仓
        """
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        # ---- 价格序列 ----
        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)

        # ---- 向量化计算ATR (Wilder平滑) ----
        atr = self._compute_wilder_atr(high, low, close, atr_period)

        # ---- Session时间判断 ----
        # time_float = hour + minute / 60.0
        hour_arr   = df.index.hour
        minute_arr = df.index.minute
        time_float = hour_arr + minute_arr / 60.0

        is_day1 = (time_float >= 9.0)  & (time_float < 11.5)   # 09:00–11:30
        is_day2 = (time_float >= 13.5) & (time_float < 15.0)   # 13:30–15:00
        is_day  = is_day1 | is_day2

        # ---- 日内session状态 ----
        session_start_idx: int   = -1
        session_open:      float = np.nan
        session_high:      float = np.nan
        session_low:       float = np.nan

        signal_direction: int   = 0      # +1 多仓 / -1 空仓 / 0 空仓
        target_price:     float = np.nan  # 均值回归目标价
        in_day:           bool  = False

        # 后半段判断阈值 (固定10根，与规格一致)
        LATTER_HALF_MIN = 10

        for i in range(n):
            cur_is_day = bool(is_day[i])

            # ----------------------------------------------------------
            # Session转换检测
            # ----------------------------------------------------------
            if cur_is_day and not in_day:
                # 新session开始: 重置所有session状态
                session_start_idx = i
                session_open      = close[i]   # 用首根bar收盘价作为session_open
                session_high      = high[i]
                session_low       = low[i]
                signal_direction  = 0
                target_price      = np.nan
                in_day            = True

            elif not cur_is_day:
                # Session刚结束 — 在最后一根session内bar (i-1) 强制平仓
                if in_day and signal_direction != 0 and i > 0:
                    signals[i - 1] = 2    # i-1 是session内最后一根bar
                in_day           = False
                signal_direction = 0
                continue

            # ----------------------------------------------------------
            # NaN安全检查: ATR热身期或价格异常时跳过
            # ----------------------------------------------------------
            if np.isnan(atr[i]) or np.isnan(session_open) or session_open == 0.0:
                # 仍要更新session_high/low，避免漏掉极值
                if not np.isnan(high[i]):
                    session_high = max(session_high, high[i]) if not np.isnan(session_high) else high[i]
                if not np.isnan(low[i]):
                    session_low  = min(session_low,  low[i])  if not np.isnan(session_low)  else low[i]
                continue

            # ----------------------------------------------------------
            # 更新session极值
            # ----------------------------------------------------------
            session_high = max(session_high, high[i])
            session_low  = min(session_low,  low[i])

            bar_in_session = i - session_start_idx
            session_range  = session_high - session_low

            # ----------------------------------------------------------
            # 持仓中: 检查均值回归目标是否命中
            # ----------------------------------------------------------
            if signal_direction != 0 and not np.isnan(target_price):
                if signal_direction == 1 and close[i] >= target_price:
                    # 多头目标价触达 → 平仓
                    signals[i]       = 2
                    signal_direction = 0
                    continue
                elif signal_direction == -1 and close[i] <= target_price:
                    # 空头目标价触达 → 平仓
                    signals[i]       = 2
                    signal_direction = 0
                    continue

            # ----------------------------------------------------------
            # 持仓中: 维持信号 (告知回测引擎仓位仍然存续)
            # ----------------------------------------------------------
            if signal_direction != 0:
                signals[i] = signal_direction
                continue

            # ----------------------------------------------------------
            # 入场条件判断 (仅在signal_direction == 0 时评估)
            # ----------------------------------------------------------
            # 条件1: session后半段
            in_latter_half = bar_in_session >= LATTER_HALF_MIN

            # 条件2: session_range 超过 atr_mult × ATR
            overextended = (session_range > atr_mult * atr[i]) and (session_range > 0.0)

            if not (in_latter_half and overextended):
                continue

            # 条件3: 价格位置判断 — 区间顶部/底部20%以内
            zone = 0.20 * session_range
            near_high = close[i] > session_high - zone    # 处于顶部20%区间
            near_low  = close[i] < session_low  + zone    # 处于底部20%区间

            if near_high:
                # 上行过度延伸 → 做空，目标为从session_high向下回撤
                signals[i]       = -1
                signal_direction = -1
                target_price     = session_high - retracement * session_range
            elif near_low:
                # 下行过度延伸 → 做多，目标为从session_low向上回升
                signals[i]       = 1
                signal_direction = 1
                target_price     = session_low + retracement * session_range

        return signals

    def position_params(self) -> PositionParams:
        return PositionParams(
            hard_stop_pct=0.8,
            trailing_pct=0.6,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
