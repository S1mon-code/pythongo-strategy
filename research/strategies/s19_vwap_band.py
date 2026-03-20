"""
Strategy #19 — Intraday VWAP Dynamic Band Mean-Reversion (5-min bars)
================================================================================

【策略思路】
  核心逻辑: VWAP ± 动态布林带 → 均值回归入场，VWAP 出场

  VWAP (成交量加权平均价) 是日内的"公允价格"。本策略计算价格偏离
  VWAP 的偏差 (dev = close - VWAP)，并用滚动标准差构建动态带状通道
  (类似布林带)。当价格从带内 *穿越* 下轨时做多，穿越上轨时做空，
  价格回归至 VWAP 中轴时强制出场。

  与 S10 (VwapZscoreReversion) 的区别:
  - S10 使用 Z-Score 阈值 (偏差 / 滚动标准差 > z_threshold)，
    任意时刻 Z 超过阈值即发出信号；
  - S19 使用布林带式通道 (VWAP ± band_mult × std(dev))，只在价格
    *穿越* 边界 (前一根在带内，当前根在带外) 时触发，入场更干净；
  - S19 在每个交易 session 开始时额外重置 band 计算基准（不只重置
    VWAP），减少跨 session 噪声对带宽估计的污染；
  - S19 新增 min_dev_pct 过滤：偏差绝对值 / VWAP 必须超过最低比例，
    剔除 VWAP 附近的微小穿越信号，避免高频虚假信号。

  VWAP 计算 (session 级重置):
  ────────────────────────────
  Session 划分:
    - 夜盘  : hour == 21  (21:00 开始)
    - 日盘1 : hour == 9,  minute == 0  (09:00 开始)
    - 日盘2 : hour == 13, minute == 30 (13:30 开始)

  每个 session 开始时重置累计量:
    cum_tp_vol = 0,  cum_vol = 0
  典型价格: typical_price = (high + low + close) / 3
  VWAP = cumsum(typical_price × volume) / cumsum(volume)

  动态带宽:
  ──────────
  dev = close - VWAP
  rolling_std = std(dev, band_window)   — 使用 shift(1) 避免未来函数
  upper_band  = VWAP + band_mult × rolling_std
  lower_band  = VWAP − band_mult × rolling_std

  信号生成 (穿越逻辑):
  ─────────────────────
  做多 (1) : 前一根 dev >= lower_band，当前根 dev < lower_band
             且 abs(dev/vwap) > min_dev_pct
  做空 (-1): 前一根 dev <= upper_band，当前根 dev > upper_band
             且 abs(dev/vwap) > min_dev_pct
  强制出场 (2):
    - 做多且 close >= vwap → 价格回归至 VWAP 中轴，平多
    - 做空且 close <= vwap → 价格回归至 VWAP 中轴，平空

  出场逻辑依赖持仓状态，采用有状态循环实现；入场信号可向量化预计算。

  参数设计 (3个，27组合):
  - band_mult   : 带宽倍数 [1.5, 2.0, 2.5] — 控制入场灵敏度
  - band_window : 滚动窗口 [10, 20, 30]     — 控制 std 估计的平滑度
  - min_dev_pct : 最小偏差比例 [0.001, 0.002, 0.003] — 过滤微小穿越

  适用环境: 日内振荡、VWAP 有明显支撑/压力的市场
  风险提示: 单边趋势行情中 VWAP 持续单侧运行，均值回归失效；
            夜盘流动性较低时 volume 权重可能失真
================================================================================
"""

import numpy as np
import pandas as pd

from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class VwapBandReversion(BaseResearchStrategy):
    """VWAP 动态带均值回归策略 — S19。

    计算日内 session 级重置的 VWAP，以偏差的滚动标准差构建动态通道，
    当价格穿越通道边界时入场做均值回归，价格回归至 VWAP 时出场。
    """

    name = "VWAP Band Mean-Reversion"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回 3 参数网格，共 3×3×3 = 27 个参数组合。

        band_mult : float
            带宽倍数，即 VWAP ± band_mult × rolling_std(dev)。
            1.5 → 较窄通道，信号更频繁；2.5 → 较宽通道，信号更保守。
        band_window : int
            计算偏差滚动标准差的窗口大小。
            窗口越小对近期波动越敏感；窗口越大带宽越平滑。
        min_dev_pct : float
            入场时 abs(dev/vwap) 的最低阈值，过滤 VWAP 附近的微小穿越。
            0.001 = 0.1%；0.003 = 0.3%。
        """
        return {
            "band_mult":   [1.5, 2.0, 2.5],
            "band_window": [10, 20, 30],
            "min_dev_pct": [0.001, 0.002, 0.003],
        }

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_session_starts(df: pd.DataFrame) -> np.ndarray:
        """检测每个 session 的起始位置，返回布尔数组。

        Session 起始规则:
          - 夜盘  : hour == 21 且 minute == 0  (21:00 bar)
          - 日盘1 : hour == 9  且 minute == 0  (09:00 bar)
          - 日盘2 : hour == 13 且 minute == 30 (13:30 bar)

        注意: 仅在精确起始 bar 置 True，用于在该 bar 重置累计量。
        """
        hour = df.index.hour
        minute = df.index.minute

        night_start = (hour == 21) & (minute == 0)
        day1_start  = (hour == 9)  & (minute == 0)
        day2_start  = (hour == 13) & (minute == 30)

        return (night_start | day1_start | day2_start).values

    @staticmethod
    def _compute_session_vwap(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        session_starts: np.ndarray,
    ) -> np.ndarray:
        """计算 session 级重置的 VWAP。

        在每个 session 起始 bar 重置累计典型价格×成交量和累计成交量，
        避免跨 session 的 VWAP 污染。

        Parameters
        ----------
        high, low, close : np.ndarray
            OHLC 价格数组。
        volume : np.ndarray
            成交量数组；若全为零则使用 1.0 作为等权替代。
        session_starts : np.ndarray of bool
            每个 session 起始 bar 为 True 的布尔数组。

        Returns
        -------
        np.ndarray of float64
            与输入等长的 VWAP 数组；session 第一根 bar 的 VWAP 等于典型价格。
        """
        n = len(close)
        vwap = np.empty(n, dtype=np.float64)
        vwap[:] = np.nan

        typical_price = (high + low + close) / 3.0

        cum_tp_vol = 0.0
        cum_vol    = 0.0
        initialized = False  # 首个 session 开始前不计算 VWAP

        for i in range(n):
            # session 起始 bar：重置累计量
            if session_starts[i]:
                cum_tp_vol  = 0.0
                cum_vol     = 0.0
                initialized = True

            if not initialized:
                # 尚未遇到第一个 session 起始，跳过
                continue

            v = volume[i]
            cum_tp_vol += typical_price[i] * v
            cum_vol    += v

            if cum_vol > 0.0:
                vwap[i] = cum_tp_vol / cum_vol
            # 若 cum_vol == 0（成交量全零）则 vwap[i] 保持 nan，
            # 后续 nan_mask 会将对应信号归零

        return vwap

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        band_mult: float = 2.0,
        band_window: int = 20,
        min_dev_pct: float = 0.002,
    ) -> np.ndarray:
        """生成 VWAP 动态带均值回归信号。

        实现分两步:
        1. 向量化预计算 VWAP、偏差、动态带和入场候选 (entry_long/short)。
        2. 有状态循环处理出场信号 (signal == 2) 及持仓维持逻辑。

        Parameters
        ----------
        df : pd.DataFrame
            含 DatetimeIndex 的 OHLCV DataFrame。必须包含 close/high/low
            列；volume 列可选（缺失时以 1.0 等权替代）。
        band_mult : float
            带宽乘数，控制 VWAP ± band_mult×std(dev) 的宽窄。
        band_window : int
            计算偏差滚动标准差的滚动窗口大小。
        min_dev_pct : float
            入场时要求的最小偏差比例 (abs(dev)/vwap)；低于此值的穿越
            被视为噪声，不触发信号。

        Returns
        -------
        np.ndarray of int8
            与 df 等长的信号数组:
              +1  做多 (持有)
              -1  做空 (持有)
               0  空仓 / 无信号
              +2  强制出场 (价格回归至 VWAP)
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        # ------------------------------------------------------------------
        # 提取价格与成交量数组
        # ------------------------------------------------------------------
        close  = df["close"].values.astype(np.float64)
        high   = df["high"].values.astype(np.float64) if "high"   in df.columns else close.copy()
        low    = df["low"].values.astype(np.float64)  if "low"    in df.columns else close.copy()

        # 成交量：缺失或全零时退化为等权 (1.0)
        if "volume" in df.columns:
            volume = df["volume"].values.astype(np.float64)
            if not np.any(volume > 0):
                volume = np.ones(n, dtype=np.float64)
        else:
            volume = np.ones(n, dtype=np.float64)

        # ------------------------------------------------------------------
        # Session 起始检测
        # ------------------------------------------------------------------
        session_starts = self._detect_session_starts(df)

        # 若数据中没有检测到任何 session 起始（如数据不含标准时间戳），
        # 则将第一行视为 session 起始，确保 VWAP 可以被计算
        if not np.any(session_starts):
            session_starts = session_starts.copy()
            session_starts[0] = True

        # ------------------------------------------------------------------
        # 计算 session 级重置的 VWAP
        # ------------------------------------------------------------------
        vwap = self._compute_session_vwap(high, low, close, volume, session_starts)

        # ------------------------------------------------------------------
        # 计算偏差与动态带 (向量化)
        # ------------------------------------------------------------------
        # dev = close - vwap (价格对 VWAP 的偏差)
        dev = close - vwap

        # 使用 pandas 滚动标准差，并用 shift(1) 确保无未来函数泄漏
        # min_periods=band_window 保证窗口满足后才产生有效值
        dev_series = pd.Series(dev)
        rolling_std = (
            dev_series
            .shift(1)                              # shift(1) 避免当前 bar 信息泄漏
            .rolling(window=band_window, min_periods=band_window)
            .std()
            .values
        )

        # 动态带宽: VWAP ± band_mult × rolling_std(dev)
        upper_band = vwap + band_mult * rolling_std
        lower_band = vwap - band_mult * rolling_std

        # ------------------------------------------------------------------
        # 向量化预计算穿越信号 (入场候选)
        # ------------------------------------------------------------------
        # 前一根偏差与带 (shift(1) 方向)
        prev_dev   = dev_series.shift(1).values
        prev_lower = pd.Series(lower_band).shift(1).values
        prev_upper = pd.Series(upper_band).shift(1).values

        # 做多穿越: 前一根在下轨之上 (或等于下轨), 当前根突破至下轨之下
        # 同时需满足最小偏差比例过滤
        with np.errstate(invalid="ignore", divide="ignore"):
            abs_dev_pct = np.where(vwap != 0.0, np.abs(dev) / np.abs(vwap), 0.0)

        long_cross  = (
            (prev_dev >= prev_lower)      # 前一根: 偏差在下轨之上
            & (dev < lower_band)          # 当前根: 偏差穿越至下轨之下
            & (abs_dev_pct > min_dev_pct) # 最小偏差比例过滤
        )
        short_cross = (
            (prev_dev <= prev_upper)      # 前一根: 偏差在上轨之下
            & (dev > upper_band)          # 当前根: 偏差穿越至上轨之上
            & (abs_dev_pct > min_dev_pct) # 最小偏差比例过滤
        )

        # NaN 安全: 任意关键值为 nan 的 bar 不允许入场
        nan_mask = (
            np.isnan(vwap)
            | np.isnan(rolling_std)
            | np.isnan(lower_band)
            | np.isnan(upper_band)
            | np.isnan(prev_dev)
            | np.isnan(prev_lower)
            | np.isnan(prev_upper)
        )
        long_cross[nan_mask]  = False
        short_cross[nan_mask] = False

        # ------------------------------------------------------------------
        # 有状态循环: 处理出场 (signal=2) 与持仓维持
        # ------------------------------------------------------------------
        # 循环逻辑优先级: 出场 > 入场 > 持仓
        #   - 出场 (2): 当前持多且 close >= vwap，或持空且 close <= vwap
        #   - 入场 (+1/-1): long_cross 或 short_cross 为 True（且当前空仓）
        #   - 持仓: 已入场后每根 bar 维持信号方向，直到出场
        sig = 0  # 当前持仓方向: +1 做多, -1 做空, 0 空仓

        for i in range(n):
            # ---- 出场检查 (优先于入场) ------------------------------------
            if sig == 1:
                # 做多持仓: close 回归至 VWAP，强制出场
                if not np.isnan(vwap[i]) and close[i] >= vwap[i]:
                    signals[i] = 2   # 强制出场信号
                    sig = 0
                    continue
                # 未到出场条件: 维持多头信号
                signals[i] = 1
                continue

            if sig == -1:
                # 做空持仓: close 回归至 VWAP，强制出场
                if not np.isnan(vwap[i]) and close[i] <= vwap[i]:
                    signals[i] = 2   # 强制出场信号
                    sig = 0
                    continue
                # 未到出场条件: 维持空头信号
                signals[i] = -1
                continue

            # ---- 当前空仓: 检查入场候选 -----------------------------------
            if long_cross[i]:
                # 价格从带内向下穿越下轨 → 做多
                signals[i] = 1
                sig = 1
            elif short_cross[i]:
                # 价格从带内向上穿越上轨 → 做空
                signals[i] = -1
                sig = -1
            # 否则: 空仓且无信号，signals[i] 保持 0

        return signals

    # ------------------------------------------------------------------
    # 持仓参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """返回适合 VWAP 均值回归策略的风控参数。

        参数设计理由:
        - hard_stop_pct=0.6 : 均值回归失败（市场单边趋势）时，0.6% 硬止损
                              快速截断亏损，避免偏离持续扩大。
        - trailing_pct=0.5  : 价格回归过程中设置 0.5% 移动止损，防止已
                              盈利的回归行情二次反转。
        - tp1_pct=0.5        : 半仓止盈 0.5%，锁定部分收益，降低风险敞口。
        - tp2_pct=1.0        : 余仓止盈 1.0%，捕获完整回归幅度（铁矿石
                              5 分钟级偏离典型回归幅度约 0.5–1.0%）。
        - max_lots=1         : 研究阶段保守单手定额，便于比较策略本身的
                              信号质量，不受仓位叠加影响。
        """
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
