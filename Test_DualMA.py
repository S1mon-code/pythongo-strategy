"""
================================================================================
  Test_DualMA v2 — 布林带 + 双均线趋势策略 (秒级 K 线)
================================================================================

  策略逻辑:
    入场  — fast_ma > slow_ma (趋势向上) + 价格突破布林中轨 (上区间确认)
    加仓  — 持仓中价格回踩布林中轨后反弹，趋势仍向上，最多加至 max_lots 手
    止盈1 — 盈利达到 tp1_pct% 时平半仓锁利，剩余继续持有
    止盈2 — 价格触达布林上轨时全平，带走最大利润
    止损  — 价格跌破布林下轨 或 硬止损% 触发，全平
    出场  — fast_ma < slow_ma 趋势逆转，全平

  信号优先级: 止损 > 止盈2 > 止盈1 > 趋势出场 > 加仓 > 建仓

  规则:
    - Next-bar 规则: 当前 bar 产生信号，下一根 bar 以市价执行
    - 秒级 K 线无历史数据，需等待 bb_period + 2 根 bar 预热

================================================================================
"""

from collections import deque

import numpy as np

from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import KLineData, OrderData, TickData, TradeData
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGeneratorSec


# ══════════════════════════════════════════════════════════════════════════════
#  参数 / 状态 模型
# ══════════════════════════════════════════════════════════════════════════════

class Params(BaseParams):
    """参数映射"""
    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="i2605", title="合约代码")
    seconds: int = Field(default=10, title="K线秒数", ge=1)

    fast_period: int = Field(default=5, title="快线周期", ge=2)
    slow_period: int = Field(default=20, title="慢线周期", ge=2)
    bb_period: int = Field(default=20, title="布林带周期", ge=5)
    bb_std: float = Field(default=2.0, title="布林带标准差倍数")

    unit_volume: int = Field(default=1, title="每次手数", ge=1)
    max_lots: int = Field(default=3, title="最大持仓手数", ge=1)

    tp1_pct: float = Field(default=0.15, title="止盈1触发(%),平半仓")
    stop_loss_pct: float = Field(default=0.20, title="硬止损(%)")


class State(BaseState):
    """状态映射"""
    fast_ma: float = Field(default=0.0, title="快均线")
    slow_ma: float = Field(default=0.0, title="慢均线")
    bb_upper: float = Field(default=0.0, title="布林上轨")
    bb_mid: float = Field(default=0.0, title="布林中轨")
    bb_lower: float = Field(default=0.0, title="布林下轨")
    net_pos: int = Field(default=0, title="净持仓")
    avg_price: float = Field(default=0.0, title="持仓均价")
    bar_count: int = Field(default=0, title="已收K线数")
    pending: str = Field(default="—", title="待执行信号")
    last_action: str = Field(default="—", title="上次操作")


# ══════════════════════════════════════════════════════════════════════════════
#  策略类
# ══════════════════════════════════════════════════════════════════════════════

class Test_DualMA(BaseStrategy):
    """布林带 + 双均线趋势策略 (秒级 K 线)"""

    def __init__(self):
        super().__init__()

        self.params_map = Params()
        self.state_map = State()

        self.kline_generator: KLineGeneratorSec = None

        # 收盘价序列 (手动维护，计算均线和布林带)
        self._closes: deque = deque(maxlen=600)

        # 持仓状态
        self.avg_price: float = 0.0
        self.peak_price: float = 0.0   # 持仓期间最高价 (供未来移动止损扩展)

        # 止盈1是否已执行 (避免重复触发半仓止盈)
        self._tp1_done: bool = False

        # 上一根 bar 指标缓存
        self.pre_close: float = 0.0
        self.pre_bb_mid: float = 0.0

        # Next-bar 待执行信号
        self._pending: str | None = None

        # 委托 ID 集合
        self.order_id: set[int] = set()

    @property
    def main_indicator_data(self) -> dict[str, float]:
        return {
            f"MA{self.params_map.fast_period}": self.state_map.fast_ma,
            f"MA{self.params_map.slow_period}": self.state_map.slow_ma,
            "BB_UP": self.state_map.bb_upper,
            "BB_MID": self.state_map.bb_mid,
            "BB_LOW": self.state_map.bb_lower,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  生命周期
    # ══════════════════════════════════════════════════════════════════════

    def on_start(self) -> None:
        self.widget.set_xrange_event_signal.emit()
        self.kline_generator = KLineGeneratorSec(
            callback=self.callback,
            seconds=self.params_map.seconds,
        )
        super().on_start()
        warmup = self.params_map.bb_period + 2
        self.output(
            f"Test_DualMA v2 启动 | {self.params_map.instrument_id} "
            f"{self.params_map.seconds}s K线 | "
            f"fast={self.params_map.fast_period} slow={self.params_map.slow_period} "
            f"BB({self.params_map.bb_period},{self.params_map.bb_std}) | "
            f"TP1={self.params_map.tp1_pct}% SL={self.params_map.stop_loss_pct}% | "
            f"预热需 {warmup} 根K线 (约 {warmup * self.params_map.seconds} 秒)"
        )

    def on_stop(self) -> None:
        super().on_stop()
        self.widget.kline_widget.cancel_xrange_event()

    # ══════════════════════════════════════════════════════════════════════
    #  行情 / 委托 / 成交 回调
    # ══════════════════════════════════════════════════════════════════════

    def on_tick(self, tick: TickData) -> None:
        super().on_tick(tick)
        self.kline_generator.tick_to_kline(tick)

    def on_order_cancel(self, order: OrderData) -> None:
        super().on_order_cancel(order)
        self.order_id.discard(order.order_id)

    def on_trade(self, trade: TradeData, log: bool = True) -> None:
        super().on_trade(trade, log=True)
        self.order_id.discard(trade.order_id)
        self.update_status_bar()

    def on_error(self, error: dict) -> None:
        self.output(f"[错误] {error}")

    # ══════════════════════════════════════════════════════════════════════
    #  K 线主回调
    # ══════════════════════════════════════════════════════════════════════

    def callback(self, kline: KLineData) -> None:
        """K 线完成回调 — 主策略逻辑"""
        signal_price = 0.0

        # ── 1. 撤销未成交挂单 ─────────────────────────────────────────
        for oid in list(self.order_id):
            self.cancel_order(oid)

        # ── 2. 执行 pending 信号 (next-bar 规则) ─────────────────────
        if self._pending:
            signal_price = self._execute(kline, self._pending)
            self._pending = None

        # ── 3. 追加收盘价 ─────────────────────────────────────────────
        self._closes.append(kline.close)
        self.state_map.bar_count = len(self._closes)

        # ── 4. 计算指标，数据不足则等待预热 ──────────────────────────
        if not self._calc_indicator():
            self._push_widget(kline, signal_price)
            return

        fast_ma = self.state_map.fast_ma
        slow_ma = self.state_map.slow_ma
        bb_upper = self.state_map.bb_upper
        bb_mid = self.state_map.bb_mid
        bb_lower = self.state_map.bb_lower
        close = kline.close

        # ── 5. 获取持仓和更新峰值 ─────────────────────────────────────
        net_pos = self.get_position(self.params_map.instrument_id).net_position

        if net_pos == 0:
            self.avg_price = 0.0
            self.peak_price = 0.0
            self._tp1_done = False
        elif close > self.peak_price:
            self.peak_price = close

        # ── 6. 生成信号 ───────────────────────────────────────────────
        tp1_pct = self.params_map.tp1_pct
        sl_pct = self.params_map.stop_loss_pct

        # 止损: 跌破布林下轨 或 硬止损
        if (net_pos > 0
                and self.avg_price > 0
                and (close <= bb_lower
                     or close <= self.avg_price * (1 - sl_pct / 100))):
            self._pending = "STOP"
            reason = "跌破下轨" if close <= bb_lower else f"硬止损{sl_pct}%"
            self.output(
                f"[信号] 止损({reason}) | close={close:.1f} "
                f"均价={self.avg_price:.1f} 下轨={bb_lower:.1f}"
            )

        # 止盈2: 价格触达布林上轨，全平
        elif (net_pos > 0
              and close >= bb_upper):
            self._pending = "TP2"
            self.output(
                f"[信号] 止盈2(上轨) | close={close:.1f} "
                f"上轨={bb_upper:.1f} 盈利≈{((close - self.avg_price) / self.avg_price * 100):.2f}%"
            )

        # 止盈1: 盈利达到 tp1_pct%，平半仓
        elif (net_pos >= 2
              and not self._tp1_done
              and self.avg_price > 0
              and close >= self.avg_price * (1 + tp1_pct / 100)):
            self._pending = "TP1"
            self.output(
                f"[信号] 止盈1({tp1_pct}%) | close={close:.1f} "
                f"均价={self.avg_price:.1f} 平半仓"
            )

        # 趋势出场: fast_ma 跌破 slow_ma，全平
        elif net_pos > 0 and fast_ma < slow_ma:
            self._pending = "CLOSE"
            self.output(f"[信号] 趋势出场 | fast={fast_ma:.2f} < slow={slow_ma:.2f}")

        # 加仓: 持仓中价格回踩中轨后反弹 (close > bb_mid)，趋势向上，仓位未满
        elif (0 < net_pos < self.params_map.max_lots
              and fast_ma > slow_ma
              and self.pre_close <= self.pre_bb_mid  # 上根 bar 在中轨下方
              and close > bb_mid                     # 本根 bar 反弹回中轨上
              and self.avg_price > 0
              and close > self.avg_price):           # 不亏损加仓
            self._pending = "ADD"
            self.output(
                f"[信号] 加仓(中轨反弹) | close={close:.1f} "
                f"中轨={bb_mid:.1f} 当前={net_pos}手"
            )

        # 建仓: 趋势向上 + 价格突破中轨进入上区间
        elif (net_pos == 0
              and fast_ma > slow_ma
              and self.pre_close <= self.pre_bb_mid  # 上根 bar 在中轨下方或等于
              and close > bb_mid):                   # 本根 bar 突破中轨
            self._pending = "OPEN"
            self.output(
                f"[信号] 建仓(突破中轨) | close={close:.1f} "
                f"中轨={bb_mid:.1f} fast={fast_ma:.2f}"
            )

        # 缓存本 bar 数据，供下 bar 判断
        self.pre_close = close
        self.pre_bb_mid = bb_mid

        # ── 7. 更新状态栏 ─────────────────────────────────────────────
        self.state_map.net_pos = net_pos
        self.state_map.avg_price = round(self.avg_price, 1)
        self.state_map.pending = self._pending or "—"

        self._push_widget(kline, signal_price)
        self.update_status_bar()

    # ══════════════════════════════════════════════════════════════════════
    #  交易执行
    # ══════════════════════════════════════════════════════════════════════

    def _execute(self, kline: KLineData, action: str) -> float:
        """执行 pending 信号，返回 signal_price 供图表标记"""
        price = kline.close  # 仅用于显示，实际以市价成交

        if action == "OPEN":
            vol = self.params_map.unit_volume
            oid = self.send_order(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=vol,
                price=price,
                order_direction="buy",
                market=True,
            )
            if oid is not None:
                self.order_id.add(oid)
            self.avg_price = price
            self.peak_price = price
            self._tp1_done = False
            self.state_map.last_action = f"建仓 {vol}手@{price:.1f}"
            self.output(f"[执行] 建仓 {vol}手 market order_id={oid}")
            return price

        elif action == "ADD":
            vol = self.params_map.unit_volume
            actual_pos = self.get_position(self.params_map.instrument_id).net_position
            oid = self.send_order(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=vol,
                price=price,
                order_direction="buy",
                market=True,
            )
            if oid is not None:
                self.order_id.add(oid)
            if actual_pos > 0:
                self.avg_price = (
                    (self.avg_price * actual_pos + price * vol) / (actual_pos + vol)
                )
            else:
                self.avg_price = price
            self.state_map.last_action = f"加仓 {vol}手@{price:.1f} 均={self.avg_price:.1f}"
            self.output(
                f"[执行] 加仓 {vol}手 market order_id={oid} | "
                f"均价≈{self.avg_price:.1f} 总仓≈{actual_pos + vol}手"
            )
            return price

        elif action == "TP1":
            actual_pos = self.get_position(self.params_map.instrument_id).net_position
            tp_vol = max(1, actual_pos // 2)  # 平半仓
            oid = self.auto_close_position(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=tp_vol,
                price=price,
                order_direction="sell",
                market=True,
            )
            if oid is not None:
                self.order_id.add(oid)
            self._tp1_done = True
            pnl = (price - self.avg_price) / self.avg_price * 100 if self.avg_price > 0 else 0
            self.state_map.last_action = f"止盈1 {tp_vol}手@{price:.1f} +{pnl:.2f}%"
            self.output(
                f"[执行] 止盈1 平{tp_vol}手 market order_id={oid} | "
                f"盈利≈{pnl:.2f}% 剩余≈{actual_pos - tp_vol}手"
            )
            return -price

        elif action in ("TP2", "CLOSE", "STOP"):
            label = {"TP2": "止盈2", "CLOSE": "趋势出场", "STOP": "止损"}[action]
            actual_pos = self.get_position(self.params_map.instrument_id).net_position
            oid = None
            if actual_pos > 0:
                oid = self.auto_close_position(
                    exchange=self.params_map.exchange,
                    instrument_id=self.params_map.instrument_id,
                    volume=actual_pos,
                    price=price,
                    order_direction="sell",
                    market=True,
                )
                if oid is not None:
                    self.order_id.add(oid)
            pnl = (price - self.avg_price) / self.avg_price * 100 if self.avg_price > 0 else 0
            self.avg_price = 0.0
            self.peak_price = 0.0
            self._tp1_done = False
            self.state_map.last_action = f"{label} {actual_pos}手@{price:.1f} {pnl:+.2f}%"
            self.output(
                f"[执行] {label} {actual_pos}手 market order_id={oid} | 盈亏≈{pnl:+.2f}%"
            )
            return -price

        return 0.0

    # ══════════════════════════════════════════════════════════════════════
    #  指标计算
    # ══════════════════════════════════════════════════════════════════════

    def _calc_indicator(self) -> bool:
        """计算双均线和布林带，数据不足返回 False"""
        need = self.params_map.bb_period + 2
        if len(self._closes) < need:
            return False

        arr = np.array(self._closes, dtype=float)
        fp = self.params_map.fast_period
        sp = self.params_map.slow_period
        bp = self.params_map.bb_period

        self.state_map.fast_ma = round(float(arr[-fp:].mean()), 2)
        self.state_map.slow_ma = round(float(arr[-sp:].mean()), 2)

        bb_window = arr[-bp:]
        bb_mid = float(bb_window.mean())
        bb_std = float(bb_window.std())
        mult = self.params_map.bb_std

        self.state_map.bb_mid = round(bb_mid, 2)
        self.state_map.bb_upper = round(bb_mid + mult * bb_std, 2)
        self.state_map.bb_lower = round(bb_mid - mult * bb_std, 2)

        return True

    # ══════════════════════════════════════════════════════════════════════
    #  辅助
    # ══════════════════════════════════════════════════════════════════════

    def _push_widget(self, kline: KLineData, signal_price: float = 0.0) -> None:
        self.widget.recv_kline({
            "kline": kline,
            "signal_price": signal_price,
            **self.main_indicator_data,
        })
