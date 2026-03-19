"""
================================================================================
  RGL 铁矿石 CTA 策略 — 双均线 + 趋势过滤 + Z-Score 评级
================================================================================

  适用平台   : 无限易 PythonGo (BaseStrategy)
  合约       : 铁矿石主力 (I9999 / I2605 等)
  K线周期    : 5 分钟 (可在界面调整)
  方向       : 仅做多 (Long Only)
  版本       : 2.0 (迁移至 PythonGO 框架)
  日期       : 2026-03-19

================================================================================
  运行逻辑: on_start -> [ on_tick -> (real_time_callback | callback) ] 循环
  - on_tick:             接收 Tick，推送给 KLineGenerator 合成 K 线
  - real_time_callback:  每个 Tick 调用，仅更新图表（不交易）
  - callback:            K 线完成时调用，主策略逻辑在此执行
================================================================================
"""

import numpy as np

from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import KLineData, OrderData, TickData, TradeData
from pythongo.core import KLineStyleType
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGenerator


# ══════════════════════════════════════════════════════════════════════════════
#  参数 / 状态 模型
# ══════════════════════════════════════════════════════════════════════════════

class Params(BaseParams):
    """参数映射模型 — 显示在无限易界面"""
    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="i2605", title="合约代码")
    volume: int = Field(default=1, title="每手下单量", ge=1)
    kline_style: KLineStyleType = Field(default="M5", title="K线周期")

    # 均线参数
    fast_period: int = Field(default=20, title="快线周期", ge=2)
    slow_period: int = Field(default=60, title="慢线周期", ge=2)
    regime_period: int = Field(default=100, title="趋势MA周期", ge=2)

    # 风控参数
    trailing_pct: float = Field(default=2.0, title="移动止损(%)")
    hard_stop_pct: float = Field(default=5.0, title="硬止损(%)")
    flatten_eod: int = Field(default=1, title="盘前清仓(1是0否)")

    # 信号评级
    zscore_lb: int = Field(default=100, title="Z-Score回看")
    vol_lb: int = Field(default=50, title="量能回看")
    mom_lb: int = Field(default=10, title="动量回看")
    min_rating: float = Field(default=5.0, title="最低评级")

    # 仓位
    max_lots: int = Field(default=5, title="最大手数", ge=1)


class State(BaseState):
    """状态映射模型 — 显示在无限易状态栏"""
    fast_ma: float = Field(default=0.0, title="快均线")
    slow_ma: float = Field(default=0.0, title="慢均线")
    net_pos: int = Field(default=0, title="当前持仓")
    signal_rating: float = Field(default=0.0, title="信号评级")
    regime_bull: str = Field(default="等待数据", title="趋势方向")
    entry_price: float = Field(default=0.0, title="入场价格")
    last_action: str = Field(default="FLAT", title="最近动作")


# ══════════════════════════════════════════════════════════════════════════════
#  策略类
# ══════════════════════════════════════════════════════════════════════════════

class RGL_DualMA_PythonGo(BaseStrategy):
    """双均线趋势跟踪策略 — 带趋势过滤、Z-Score 评级、盘前清仓"""

    def __init__(self) -> None:
        super().__init__()

        self.params_map = Params()
        self.state_map = State()

        self.kline_generator: KLineGenerator = None

        # ── 持仓状态 ────────────────────────────────────────────────────
        self.in_position: bool = False
        self.entry_price: float = 0.0
        self.peak_price: float = 0.0

        # ── 待执行信号 (next-bar 规则: 信号在当前 bar 产生，下一根 bar 执行) ──
        self._pending_entry: tuple[int, float] | None = None  # (lots, rating)
        self._pending_exit: str | None = None                 # reason str

        # ── 委托 ID 集合 ─────────────────────────────────────────────────
        self.order_id: set[int] = set()

    # ── 主图指标 (显示在 K 线图上) ─────────────────────────────────────────
    @property
    def main_indicator_data(self) -> dict[str, float]:
        return {
            f"MA{self.params_map.fast_period}": self.state_map.fast_ma,
            f"MA{self.params_map.slow_period}": self.state_map.slow_ma,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  生命周期
    # ══════════════════════════════════════════════════════════════════════

    def on_start(self) -> None:
        """策略启动：初始化 K 线合成器，推送历史数据预热指标"""
        self.kline_generator = KLineGenerator(
            callback=self.callback,
            real_time_callback=self.real_time_callback,
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            style=self.params_map.kline_style
        )

        # 必须在 super().on_start() 之前推送历史数据
        # 因为 super() 会把 self.trading 置为 True，历史回放阶段不应下单
        self.kline_generator.push_history_data()

        super().on_start()

        self.output(
            f"RGL_DualMA 启动 | "
            f"合约: {self.params_map.instrument_id}@{self.params_map.exchange} | "
            f"K线: {self.params_map.kline_style} | "
            f"fast={self.params_map.fast_period} slow={self.params_map.slow_period} "
            f"regime={self.params_map.regime_period} | "
            f"trail={self.params_map.trailing_pct}% hard={self.params_map.hard_stop_pct}% | "
            f"min_rating={self.params_map.min_rating} max_lots={self.params_map.max_lots}"
        )

    def on_stop(self) -> None:
        super().on_stop()

    # ══════════════════════════════════════════════════════════════════════
    #  行情 / 委托 / 成交 回调
    # ══════════════════════════════════════════════════════════════════════

    def on_tick(self, tick: TickData) -> None:
        super().on_tick(tick)
        self.kline_generator.tick_to_kline(tick)

    def on_order_cancel(self, order: OrderData) -> None:
        super().on_order_cancel(order)
        self.order_id.discard(order.order_id)

    def on_trade(self, trade: TradeData, log: bool = False) -> None:
        super().on_trade(trade, log=True)
        self.order_id.discard(trade.order_id)
        # 更新持仓状态显示
        self.state_map.net_pos = self.get_position(
            self.params_map.instrument_id
        ).net_position
        self.update_status_bar()

    # ══════════════════════════════════════════════════════════════════════
    #  K 线回调
    # ══════════════════════════════════════════════════════════════════════

    def callback(self, kline: KLineData) -> None:
        """K 线完成回调 — 主策略逻辑

        执行顺序:
          1. 撤销所有未成交挂单
          2. 执行上一根 bar 产生的待执行信号 (next-bar 规则)
          3. 计算指标 (数据不足则跳过交易逻辑)
          4. 盘前清仓检查
          5. 持仓中: 检查出场条件 → 排队到下一根 bar 执行
          6. 空仓: 检查入场条件 → 排队到下一根 bar 执行
        """
        signal_price = 0.0

        # ── 1. 撤销所有未成交挂单 ─────────────────────────────────────
        for oid in list(self.order_id):
            self.cancel_order(oid)

        # ── 2. 执行 pending 信号 (next-bar 规则) ─────────────────────
        if self._pending_entry is not None:
            target_lots, rating = self._pending_entry
            self._pending_entry = None
            signal_price = self._exec_entry(kline, target_lots, rating)

        if self._pending_exit is not None:
            reason = self._pending_exit
            self._pending_exit = None
            signal_price = self._exec_exit(kline, reason)

        # ── 3. 计算指标 ───────────────────────────────────────────────
        if not self._calc_indicator():
            self._push_widget(kline, signal_price)
            return

        producer = self.kline_generator.producer
        close_arr = producer.close
        vol_arr = producer.volume
        current_close = float(close_arr[-1])

        fast_ma_arr = producer.sma(self.params_map.fast_period, array=True)
        slow_ma_arr = producer.sma(self.params_map.slow_period, array=True)
        regime_ma_arr = producer.sma(self.params_map.regime_period, array=True)

        fast_ma = float(fast_ma_arr[-1])
        slow_ma = float(slow_ma_arr[-1])
        regime_ma = float(regime_ma_arr[-1])
        # regime MA 在 slow_period 根 bar 之前的值，用于判断斜率
        regime_ma_prev = float(regime_ma_arr[-1 - self.params_map.slow_period])

        # 趋势过滤
        regime_slope = regime_ma - regime_ma_prev
        regime_bull = (regime_slope > 0) and (current_close > regime_ma)

        # 信号评级
        signal_rating = self._calc_rating(close_arr, vol_arr, fast_ma, slow_ma)

        # 更新状态显示
        self.state_map.fast_ma = round(fast_ma, 2)
        self.state_map.slow_ma = round(slow_ma, 2)
        self.state_map.regime_bull = "多头" if regime_bull else "空头"
        self.state_map.signal_rating = round(signal_rating, 2)

        # ── 4. 盘前清仓 (立即执行，不等下一根 bar) ───────────────────
        if self.params_map.flatten_eod and self._is_session_end(kline):
            if self.in_position:
                self._exec_exit(kline, "EXIT_EOD")
                self._pending_entry = None
            self._push_widget(kline, signal_price)
            self.update_status_bar()
            return

        # ── 5. 持仓中: 检查出场 ───────────────────────────────────────
        if self.in_position:
            if current_close > self.peak_price:
                self.peak_price = current_close

            if current_close <= self.entry_price * (1 - self.params_map.hard_stop_pct / 100):
                self._pending_exit = "EXIT_HARD"
            elif current_close <= self.peak_price * (1 - self.params_map.trailing_pct / 100):
                self._pending_exit = "EXIT_TRAIL"
            elif fast_ma < slow_ma:
                self._pending_exit = "EXIT_CROSS"
            elif not regime_bull:
                self._pending_exit = "EXIT_REGIME"
            else:
                self.state_map.last_action = "HOLD"

        # ── 6. 空仓: 检查入场 ─────────────────────────────────────────
        else:
            if (
                fast_ma > slow_ma
                and regime_bull
                and signal_rating >= self.params_map.min_rating
            ):
                strength = max(signal_rating / 10.0, 0.1)
                scale = (strength - 0.1) / 0.9
                target_lots = max(
                    1, min(round(self.params_map.max_lots * scale), self.params_map.max_lots)
                )
                self._pending_entry = (target_lots, signal_rating)

        self._push_widget(kline, signal_price)
        self.update_status_bar()

    def real_time_callback(self, kline: KLineData) -> None:
        """每个 Tick 都调用 — 仅更新图表，不执行交易逻辑"""
        self._calc_indicator()
        self._push_widget(kline)

    # ══════════════════════════════════════════════════════════════════════
    #  交易执行
    # ══════════════════════════════════════════════════════════════════════

    def _exec_entry(self, kline: KLineData, target_lots: int, rating: float) -> float:
        """执行开多仓，返回 signal_price (供图表显示)"""
        price = kline.close  # 仅用于显示，实际以市价成交

        order_id = self.send_order(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            volume=target_lots,
            price=price,
            order_direction="buy",
            market=True,
        )

        if order_id is not None:
            self.order_id.add(order_id)

        self.in_position = True
        self.entry_price = price
        self.peak_price = price

        self.state_map.entry_price = price
        self.state_map.last_action = f"ENTER R={rating:.1f} L={target_lots}"

        self.output(
            f"[开多] {target_lots}手 @ {price:.1f} | "
            f"评级={rating:.1f} | next-bar执行"
        )
        return price  # 正值 → 图表显示绿色买入标记

    def _exec_exit(self, kline: KLineData, reason: str) -> float:
        """执行平多仓，返回 signal_price (供图表显示，负值=卖出标记)"""
        position = self.get_position(self.params_map.instrument_id)
        net_pos = position.net_position

        if net_pos <= 0:
            self.in_position = False
            return 0.0

        price = kline.close  # 仅用于显示，实际以市价成交

        order_id = self.auto_close_position(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            volume=net_pos,
            price=price,
            order_direction="sell",
            market=True,
        )

        if order_id is not None:
            self.order_id.add(order_id)

        pnl_pct = (
            (price - self.entry_price) / self.entry_price * 100
            if self.entry_price > 0 else 0.0
        )
        self.output(
            f"[平仓] {reason} | {net_pos}手 @ {price:.1f} | "
            f"入场={self.entry_price:.1f} | 盈亏={pnl_pct:+.2f}%"
        )

        self.in_position = False
        self.entry_price = 0.0
        self.peak_price = 0.0

        self.state_map.entry_price = 0.0
        self.state_map.last_action = reason

        return -price  # 负值 → 图表显示红色卖出标记

    # ══════════════════════════════════════════════════════════════════════
    #  指标计算
    # ══════════════════════════════════════════════════════════════════════

    def _calc_indicator(self) -> bool:
        """
        计算并缓存快慢均线到 state_map，返回 True 表示数据充足可执行交易逻辑。

        判断数据是否充足：需要 regime_period + zscore_lb 根 K 线。
        """
        if self.kline_generator is None:
            return False

        producer = self.kline_generator.producer
        min_bars = self.params_map.regime_period + self.params_map.zscore_lb

        if len(producer.close) < min_bars:
            return False

        fast_ma_arr = producer.sma(self.params_map.fast_period, array=True)
        slow_ma_arr = producer.sma(self.params_map.slow_period, array=True)

        self.state_map.fast_ma = round(float(fast_ma_arr[-1]), 2)
        self.state_map.slow_ma = round(float(slow_ma_arr[-1]), 2)

        return True

    def _calc_rating(
        self,
        close_arr: np.ndarray,
        vol_arr: np.ndarray,
        fast_ma: float,
        slow_ma: float
    ) -> float:
        """
        计算信号评级 (0-10 分，基于 rolling z-score)

        维度              满分    计算方式
        ──────────────    ────    ──────────────────────────
        MA 展幅 z-score   2.5    快慢线差距的 z-score
        趋势强度 z-score  2.5    慢线斜率的 z-score
        量能 z-score      3.0    成交量 / 均量的 z-score
        动量 z-score      2.0    近期涨幅的 z-score
        ──────────────    ────
        合计              10.0
        """
        lb = self.params_map.zscore_lb

        spread_z = self._rolling_zscore(
            self._calc_spread_series(close_arr, self.params_map.fast_period, self.params_map.slow_period),
            lb
        )
        trend_z = self._rolling_zscore(
            self._calc_trend_series(close_arr, self.params_map.slow_period, self.params_map.mom_lb),
            lb
        )
        vol_z = self._rolling_zscore(
            self._calc_vol_ratio_series(vol_arr, self.params_map.vol_lb),
            lb
        )
        mom_z = self._rolling_zscore(
            self._calc_momentum_series(close_arr, self.params_map.mom_lb),
            lb
        )

        return (
            max(0.0, min(spread_z, 2.5)) +
            max(0.0, min(trend_z, 2.5)) +
            max(0.0, min(vol_z, 3.0)) +
            max(0.0, min(mom_z, 2.0))
        )

    @staticmethod
    def _rolling_zscore(series: np.ndarray, lookback: int) -> float:
        """计算序列末尾的 rolling z-score"""
        if len(series) < lookback:
            return 0.0
        window = series[-lookback:]
        std = window.std()
        if std == 0:
            return 0.0
        return float((series[-1] - window.mean()) / std)

    @staticmethod
    def _calc_spread_series(close_arr: np.ndarray, fast_p: int, slow_p: int) -> np.ndarray:
        """计算快慢线展幅时间序列"""
        if len(close_arr) < slow_p:
            return np.zeros(1)
        fast = np.convolve(close_arr, np.ones(fast_p) / fast_p, mode="valid")
        slow = np.convolve(close_arr, np.ones(slow_p) / slow_p, mode="valid")
        min_len = min(len(fast), len(slow))
        fast, slow = fast[-min_len:], slow[-min_len:]
        return np.where(slow != 0, (fast - slow) / slow, 0)

    @staticmethod
    def _calc_trend_series(close_arr: np.ndarray, slow_p: int, mom_lb: int) -> np.ndarray:
        """计算慢线斜率时间序列"""
        if len(close_arr) < slow_p + mom_lb:
            return np.zeros(1)
        slow = np.convolve(close_arr, np.ones(slow_p) / slow_p, mode="valid")
        if len(slow) <= mom_lb:
            return np.zeros(1)
        return (slow[mom_lb:] - slow[:-mom_lb]) / np.where(slow[:-mom_lb] != 0, slow[:-mom_lb], 1)

    @staticmethod
    def _calc_vol_ratio_series(vol_arr: np.ndarray, vol_lb: int) -> np.ndarray:
        """计算量比时间序列"""
        if len(vol_arr) < vol_lb:
            return np.ones(1)
        vol_ma = np.convolve(vol_arr, np.ones(vol_lb) / vol_lb, mode="valid")
        tail = vol_arr[-len(vol_ma):]
        return np.where(vol_ma != 0, tail / vol_ma, 1.0)

    @staticmethod
    def _calc_momentum_series(close_arr: np.ndarray, mom_lb: int) -> np.ndarray:
        """计算动量时间序列"""
        if len(close_arr) <= mom_lb:
            return np.zeros(1)
        prev = close_arr[:-mom_lb]
        curr = close_arr[mom_lb:]
        return np.where(prev != 0, (curr - prev) / prev, 0)

    # ══════════════════════════════════════════════════════════════════════
    #  辅助方法
    # ══════════════════════════════════════════════════════════════════════

    def _push_widget(self, kline: KLineData, signal_price: float = 0.0) -> None:
        """更新 K 线图表"""
        try:
            self.widget.recv_kline({
                "kline": kline,
                "signal_price": signal_price,
                **self.main_indicator_data
            })
        except Exception:
            pass

    @staticmethod
    def _is_session_end(kline: KLineData) -> bool:
        """
        判断当前 K 线是否为盘前最后一根 (铁矿石 DCE)

        大商所铁矿石交易时段:
          日盘:  09:00 - 11:30,  13:30 - 15:00
          夜盘:  21:00 - 23:00

        在以下时间触发平仓:
          11:25  (午休前 5 分钟)
          14:55  (日盘收盘前 5 分钟)
          22:55  (夜盘收盘前 5 分钟)
        """
        if not hasattr(kline, "datetime") or kline.datetime is None:
            return False
        h, m = kline.datetime.hour, kline.datetime.minute
        return (h == 11 and m >= 25) or (h == 14 and m >= 55) or (h == 22 and m >= 55)
