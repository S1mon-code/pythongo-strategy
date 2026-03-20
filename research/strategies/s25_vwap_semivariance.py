"""
Strategy #25 — VWAP Z-Score + Semivariance Confluence (5-min bars)
================================================================================

【策略思路】
  核心逻辑: 两个独立有效信号的交叉验证 — VWAP Z-Score偏离 × 半方差不对称性
  组合确认，双重过滤大幅提升信号质量

  信号来源:
  ① VWAP Z-Score偏离 (来自S10, 已通过验证):
       价格相对于日内VWAP的标准化偏离度。VWAP是日内"公允价格"，当价格
       显著偏离VWAP时，均值回归概率大。Z-Score < -阈值 表示价格严重低估，
       Z-Score > +阈值 表示价格严重高估。

  ② 半方差不对称性 (来自S13, Sharpe 1.51):
       将收益率分解为上行半方差 (RS+) 和下行半方差 (RS-)，计算不对称比率:
         asym = RS+ / (RS+ + RS-)
       asym < 0.45: 下行波动主导 → 价格处于被迫抛售状态，反弹概率高
       asym > 0.55: 上行波动主导 → 价格处于被迫追涨状态，回落概率高

  入场逻辑 (两个信号同时成立才入场):
  ──────────────────────────────────
    做多: Z-Score < -z_threshold  AND  asym < (0.5 - asym_margin)
          → 价格严重低于VWAP + 下行波动主导 (强制卖盘) → 超卖反弹
    做空: Z-Score > +z_threshold  AND  asym > (0.5 + asym_margin)
          → 价格严重高于VWAP + 上行波动主导 (强制买盘) → 超买回落

  出场逻辑 (有状态循环):
    持多且 close >= VWAP → 均值回归完成，强制平仓 (signal = 2)
    持空且 close <= VWAP → 均值回归完成，强制平仓 (signal = 2)
    持仓期间每根bar持续写入方向信号 (+1/-1)

  VWAP计算 (按交易时段重置, 防止跨时段污染):
  ─────────────────────────────────────────
    时段识别:
      夜盘时段: hour == 21, minute == 0 → 新时段开始
      日盘上午: hour == 9,  minute == 0 → 新时段开始
      日盘下午: hour == 13, minute == 30 → 新时段开始

    typical_price = (high + low + close) / 3
    VWAP = cumsum(typical_price × volume) / cumsum(volume)  [时段内累积]

    VWAP使用shift(1)延迟一根bar，防止当前bar自身信息泄漏到信号中。

  半方差不对称性计算 (固定窗口20根bar):
  ──────────────────────────────────────
    ret = close.pct_change()
    RS+ = sum(max(ret, 0)^2, sv_window)   — 上行半方差
    RS- = sum(max(-ret, 0)^2, sv_window)  — 下行半方差
    asym = RS+ / (RS+ + RS-)              — 落到[0,1]; 总方差为0时填0.5

    asym同样使用shift(1)防止未来函数。

  NaN安全处理:
    VWAP为0或NaN、Z-Score为NaN、asym为NaN时信号全部置零。

  参数设计 (3个, 27组合):
  - z_threshold : VWAP Z-Score入场阈值       [1.0, 1.5, 2.0]
  - z_window    : VWAP偏离滚动标准差窗口     [10, 20, 30]
  - asym_margin : asym偏离0.5的最小幅度      [0.03, 0.05, 0.08]
  注: sv_window固定为20根bar，不纳入参数网格

  适用环境: 铁矿石日内有明确VWAP价值中枢的行情 + 波动方向出现阶段性倾斜
  风险提示: 单边趋势市中VWAP持续偏离，Z-Score长期处于极端值，单靠Z-Score
            容易被套；半方差单独使用在弱趋势市效果下降；两者组合能互相
            抵消各自的弱市场环境。

  回测参数:
  - 数据频率: 5分钟K线
  - hard_stop_pct : 0.6%  — 防止极端行情下的深度亏损
  - trailing_pct  : 0.5%  — 锁定均值回归过程中的浮动盈利
  - tp1_pct       : 0.5%  — 半仓止盈，回归行情早期降低风险敞口
  - tp2_pct       : 1.0%  — 余仓止盈，捕获完整均值回归收益
  - max_lots      : 1     — 单手保守研究配置
================================================================================
"""

import numpy as np
import pandas as pd
from .base_strategy import BaseResearchStrategy
from ..backtest_engine import PositionParams


class VwapSemivarianceReversion(BaseResearchStrategy):
    """VWAP Z-Score与半方差不对称性双重确认均值回归策略 — S25。

    将S10的VWAP Z-Score偏离信号与S13的半方差不对称性信号合并，
    仅在两个信号同时指向同一方向时才入场，以交叉验证减少虚假信号。
    价格回归至VWAP时强制出场，适合铁矿石日内振荡行情。
    """

    name = "VWAP-Semivariance Confluence"
    freq = "5min"

    # ------------------------------------------------------------------
    # 参数网格
    # ------------------------------------------------------------------

    def param_grid(self) -> dict:
        """返回 3 参数网格，共 3×3×3 = 27 个参数组合。

        z_threshold : float
            VWAP Z-Score入场阈值。1.0 → 偏离1个标准差即入场 (信号频繁)；
            2.0 → 偏离2个标准差才入场 (信号保守，质量更高)。
        z_window : int
            计算VWAP偏离滚动标准差的窗口大小 (单位: 根bar)。
            窗口越大，标准差估计越稳定但响应越慢；10根bar约50分钟。
        asym_margin : float
            半方差不对称比率偏离0.5的最小幅度，用于过滤中性状态。
            0.03 → asym < 0.47 或 > 0.53 时才激活过滤 (宽松)；
            0.08 → asym < 0.42 或 > 0.58 时才激活过滤 (严格)。
        """
        return {
            "z_threshold": [1.0, 1.5, 2.0],    # VWAP Z-Score入场阈值
            "z_window":    [10, 20, 30],         # 滚动标准差窗口
            "asym_margin": [0.03, 0.05, 0.08],  # asym偏离0.5的最小幅度
        }

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_session_vwap(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        hours: np.ndarray,
        minutes: np.ndarray,
    ) -> np.ndarray:
        """按交易时段重置的VWAP计算。

        三个时段的起始点触发重置:
          - 夜盘: hour == 21 AND minute == 0
          - 日盘上午: hour == 9 AND minute == 0
          - 日盘下午: hour == 13 AND minute == 30

        Parameters
        ----------
        high, low, close, volume : np.ndarray
            逐根bar的高低收量数据。
        hours, minutes : np.ndarray
            从 DatetimeIndex 提取的小时和分钟数组。

        Returns
        -------
        np.ndarray of float64
            与输入等长的VWAP数组。每个时段第一根bar的VWAP即为该bar的
            typical_price (无法用前序数据平滑)。不满足计算条件时填 np.nan。
        """
        typical_price = (high + low + close) / 3.0

        n = len(close)
        vwap = np.empty(n, dtype=np.float64)

        cum_tp_vol = 0.0
        cum_vol = 0.0

        for i in range(n):
            h = hours[i]
            m = minutes[i]

            # 检测新时段开始: 重置累积量
            is_new_session = (
                (h == 21 and m == 0)
                or (h == 9 and m == 0)
                or (h == 13 and m == 30)
            )
            if is_new_session:
                cum_tp_vol = 0.0
                cum_vol = 0.0

            cum_tp_vol += typical_price[i] * volume[i]
            cum_vol += volume[i]

            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else np.nan

        return vwap

    @staticmethod
    def _compute_semivariance_asym(
        close: np.ndarray,
        sv_window: int,
    ) -> np.ndarray:
        """计算滚动半方差不对称比率。

        RS_pos = sum(max(ret, 0)^2, sv_window)
        RS_neg = sum(max(-ret, 0)^2, sv_window)
        asym   = RS_pos / (RS_pos + RS_neg)

        当 RS_pos + RS_neg == 0 (无波动) 时，asym 填 0.5 (中性)。
        结果已 shift(1) 防止未来函数。

        Parameters
        ----------
        close : np.ndarray
            收盘价数组。
        sv_window : int
            滚动计算窗口 (根bar数)。

        Returns
        -------
        np.ndarray of float64
            与输入等长的 asym 数组 (已 shift(1))。
        """
        s = pd.Series(close, dtype=np.float64)
        ret = s.pct_change()

        rs_pos = ret.clip(lower=0.0).pow(2).rolling(sv_window).sum()
        rs_neg = (-ret).clip(lower=0.0).pow(2).rolling(sv_window).sum()
        rs_total = rs_pos + rs_neg

        # 总方差为0时回退到中性值0.5，防止除零
        asym = (rs_pos / rs_total).where(rs_total > 0.0, 0.5)

        # shift(1): 使用前一根bar的半方差，防止当前bar信息泄漏
        asym_shifted = asym.shift(1)

        return asym_shifted.values

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        z_threshold: float = 1.5,
        z_window: int = 20,
        asym_margin: float = 0.05,
    ) -> np.ndarray:
        """生成VWAP Z-Score与半方差不对称性双重确认的均值回归信号。

        实现分三步:
        1. 按时段重置VWAP，对偏离序列做 shift(1) 后计算滚动 Z-Score。
        2. 计算半方差不对称比率 (已 shift(1))。
        3. 有状态循环: 双重信号同时满足时入场，价格回归至VWAP时出场。

        Parameters
        ----------
        df : pd.DataFrame
            含 DatetimeIndex 的 OHLCV DataFrame，至少包含
            'high', 'low', 'close', 'volume' 列。
        z_threshold : float
            VWAP Z-Score入场阈值，绝对值超过此阈值时触发VWAP偏离信号。
        z_window : int
            计算 VWAP 偏离滚动标准差的窗口 (根bar数)。
        asym_margin : float
            半方差不对称比率偏离 0.5 的最小幅度，要求
            asym < (0.5 - asym_margin) 做多，asym > (0.5 + asym_margin) 做空。

        Returns
        -------
        np.ndarray of int8
            与 df 等长的信号数组:
              +1  做多 (含持仓维持)
              -1  做空 (含持仓维持)
               0  空仓 / 无信号
               2  强制出场 (价格回归至VWAP)
        """
        if df.empty:
            return np.zeros(0, dtype=np.int8)

        n = len(df)
        signals = np.zeros(n, dtype=np.int8)

        # ------------------------------------------------------------------
        # 提取价格和时间序列
        # ------------------------------------------------------------------
        high_arr   = df["high"].values.astype(np.float64)
        low_arr    = df["low"].values.astype(np.float64)
        close_arr  = df["close"].values.astype(np.float64)
        volume_arr = df["volume"].values.astype(np.float64)

        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
        hours_arr   = idx.hour
        minutes_arr = idx.minute

        # ------------------------------------------------------------------
        # 步骤1: 按时段重置VWAP，计算 Z-Score
        # ------------------------------------------------------------------
        # VWAP: 按夜盘/日盘上午/日盘下午三个时段独立累积，防止跨时段污染
        vwap_raw = self._compute_session_vwap(
            high_arr, low_arr, close_arr, volume_arr, hours_arr, minutes_arr
        )

        # shift(1): 使用前一根bar已完成的VWAP，防止当前bar自身信息参与信号
        vwap_s = pd.Series(vwap_raw).shift(1).values

        # 偏离量: close - VWAP (已shift)
        deviation = close_arr - vwap_s

        # 滚动标准差: 在偏离序列上计算，反映VWAP偏离的历史波动幅度
        rolling_std = pd.Series(deviation).rolling(z_window).std().values

        # Z-Score: 标准化偏离量; std=0时置0 (无波动无信号)
        with np.errstate(invalid="ignore", divide="ignore"):
            z_score = np.where(rolling_std > 0.0, deviation / rolling_std, 0.0)

        # ------------------------------------------------------------------
        # 步骤2: 半方差不对称比率 (固定 sv_window=20，已内含 shift(1))
        # ------------------------------------------------------------------
        sv_window_fixed = 20
        asym_arr = self._compute_semivariance_asym(close_arr, sv_window_fixed)

        # ------------------------------------------------------------------
        # 步骤3: 向量化预计算入场候选 (双重信号同时满足)
        # ------------------------------------------------------------------
        # VWAP Z-Score方向
        z_neg = z_score < -z_threshold   # 价格严重低于VWAP → 做多候选
        z_pos = z_score >  z_threshold   # 价格严重高于VWAP → 做空候选

        # 半方差方向
        asym_long  = asym_arr < (0.5 - asym_margin)  # 下行波动主导 → 做多候选
        asym_short = asym_arr > (0.5 + asym_margin)  # 上行波动主导 → 做空候选

        # 双重确认入场候选
        long_entry_ok  = z_neg & asym_long
        short_entry_ok = z_pos & asym_short

        # ------------------------------------------------------------------
        # NaN安全: 关键值为NaN时清除对应入场候选
        # ------------------------------------------------------------------
        nan_mask = (
            np.isnan(vwap_s)
            | np.isnan(z_score)
            | np.isnan(asym_arr)
            | np.isnan(rolling_std)
            | (vwap_s == 0.0)   # VWAP为0时VWAP无意义
        )
        long_entry_ok[nan_mask]  = False
        short_entry_ok[nan_mask] = False

        # ------------------------------------------------------------------
        # 步骤4: 有状态循环 — 入场 / 持仓维持 / 出场
        # ------------------------------------------------------------------
        # 出场条件 (价格回归至VWAP):
        #   持多且 close >= vwap_s → 均值回归至公允价值，强制平多
        #   持空且 close <= vwap_s → 均值回归至公允价值，强制平空
        #
        # 信号优先级: 出场(2) > 入场(+1/-1) > 空仓(0)
        # 持仓期间每根bar写入方向信号，告知回测引擎当前仓位状态。

        active = 0  # 当前持仓方向: +1 做多, -1 做空, 0 空仓

        for i in range(n):
            vwap_ref = vwap_s[i]  # 当前bar使用的shifted VWAP (出场参考价)

            # ---- 持多: 检查是否回归至VWAP出场 --------------------------------
            if active == 1:
                if not np.isnan(vwap_ref) and vwap_ref > 0.0 and close_arr[i] >= vwap_ref:
                    signals[i] = 2   # 价格回归至VWAP，强制平多
                    active = 0
                else:
                    signals[i] = 1   # 维持多头
                continue

            # ---- 持空: 检查是否回归至VWAP出场 --------------------------------
            if active == -1:
                if not np.isnan(vwap_ref) and vwap_ref > 0.0 and close_arr[i] <= vwap_ref:
                    signals[i] = 2   # 价格回归至VWAP，强制平空
                    active = 0
                else:
                    signals[i] = -1  # 维持空头
                continue

            # ---- 空仓: 检查双重确认入场候选 ------------------------------------
            if long_entry_ok[i]:
                # VWAP Z-Score严重偏低 + 下行半方差主导 → 做多
                signals[i] = 1
                active = 1
            elif short_entry_ok[i]:
                # VWAP Z-Score严重偏高 + 上行半方差主导 → 做空
                signals[i] = -1
                active = -1
            # 否则: 空仓且无信号，signals[i] 保持 0

        return signals

    # ------------------------------------------------------------------
    # 持仓参数
    # ------------------------------------------------------------------

    def position_params(self) -> PositionParams:
        """返回适合VWAP均值回归策略的风控参数。

        参数设计理由:
        - hard_stop_pct=0.6 : 双重确认信号质量较高，但极端趋势行情中VWAP
                              持续偏离，0.6% 硬止损及时截断深度亏损。
        - trailing_pct=0.5  : 价格向VWAP回归过程中以 0.5% 移动止损保护
                              浮动盈利，防止回归行情二次反转。
        - tp1_pct=0.5       : 半仓止盈 0.5%，在回归早期锁定部分收益，
                              与铁矿石5分钟均值回归幅度匹配。
        - tp2_pct=1.0       : 余仓止盈 1.0%，捕获VWAP偏离从极端值完整
                              回归的最大收益。
        - max_lots=1        : 研究阶段保守单手，聚焦评估双重信号质量本身。
        """
        return PositionParams(
            hard_stop_pct=0.6,
            trailing_pct=0.5,
            tp1_pct=0.5,
            tp2_pct=1.0,
            max_lots=1,
        )
