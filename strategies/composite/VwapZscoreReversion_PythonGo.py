"""
================================================================================
  VWAP Z-Score 均值回归策略 — 铁矿石 CTA
================================================================================

  适用平台   : 无限易 PythonGo (BaseStrategy)
  合约       : 铁矿石主力 (I9999 / I2605 等)
  K线周期    : 5 分钟
  方向       : 多空双向 (Long + Short)
  版本       : 1.0
  日期       : 2026-03-19

================================================================================
  回测结果 (2023-01 ~ 2026-02)
================================================================================

  Sharpe: 0.65 | Calmar: 0.49 | Max DD: 2.12%
  Annual Return: 1.03% | Win Rate: 62.6% | PF: 1.20
  Trades/Year: ~103 | Yearly: -0.41%, +1.98%, +1.49%, +0.10%
  Best Params: z_threshold=3.0, min_bars=30

================================================================================
  策略逻辑
================================================================================

  - 计算日内 VWAP (每个交易日重置)
  - Z-Score = (close - vwap) / rolling_std(deviation, 30)
  - Z < -3.0 → 开多 (价格远低于 VWAP，预期均值回归)
  - Z > +3.0 → 开空 (价格远高于 VWAP，预期均值回归)
  - 出场: 信号反转 或 硬止损 5%

================================================================================
  运行逻辑: on_start -> [ on_tick -> (real_time_callback | callback) ] 循环
  - on_tick:             接收 Tick，推送给 KLineGenerator 合成 K 线
  - real_time_callback:  每个 Tick 调用，仅更新图表（不交易）
  - callback:            K 线完成时调用，主策略逻辑在此执行
================================================================================
"""

from collections import deque
from datetime import datetime
from math import sqrt

from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import KLineData, OrderData, TickData, TradeData
from pythongo.core import KLineStyleType
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGenerator


# ══════════════════════════════════════════════════════════════════════════════
#  参数 / 状态 模型
# ══════════════════════════════════════════════════════════════════════════════

class Params(BaseParams):
    """参数映射模型 -- 显示在无限易界面"""
    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="i2605", title="合约代码")
    volume: int = Field(default=1, title="每手下单量", ge=1)
    kline_style: KLineStyleType = Field(default="M5", title="K线周期")
    z_threshold: float = Field(default=3.0, title="Z-Score阈值")
    min_bars: int = Field(default=30, title="最少K线数")
    hard_stop_pct: float = Field(default=5.0, title="硬止损(%)")


class State(BaseState):
    """状态映射模型 -- 显示在无限易状态栏"""
    vwap: float = Field(default=0.0, title="VWAP")
    z_score: float = Field(default=0.0, title="Z-Score")
    net_pos: int = Field(default=0, title="当前持仓")
    entry_price: float = Field(default=0.0, title="入场价格")
    bars_today: int = Field(default=0, title="今日K线数")
    last_action: str = Field(default="FLAT", title="最近动作")


# ══════════════════════════════════════════════════════════════════════════════
#  策略类
# ══════════════════════════════════════════════════════════════════════════════

class VwapZscoreReversion_PythonGo(BaseStrategy):
    """VWAP Z-Score 均值回归策略 -- 多空双向"""

    def __init__(self) -> None:
        super().__init__()

        self.params_map = Params()
        self.state_map = State()

        self.kline_generator: KLineGenerator = None

        # ── 持仓状态 ────────────────────────────────────────────────────
        self.in_position: bool = False
        self.position_side: str = ""      # "long" or "short"
        self.entry_price: float = 0.0

        # ── VWAP 累积量 (每个交易日重置) ──────────────────────────────
        self._cum_tp_vol: float = 0.0     # cumulative(typical_price * volume)
        self._cum_vol: float = 0.0        # cumulative(volume)
        self._current_day: int = 0        # 当前交易日 (用于检测日切换)
        self._bars_today: int = 0         # 当日已生成的 K 线数

        # ── Z-Score 偏差序列 (滚动窗口) ──────────────────────────────
        self._deviations: deque = deque(maxlen=200)

        # ── 待执行信号 (next-bar 规则) ──────────────────────────────────
        self._pending: str | None = None  # "LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT", "STOP"

        # ── 委托 ID 集合 ─────────────────────────────────────────────────
        self.order_id: set[int] = set()

    # ── 主图指标 ─────────────────────────────────────────────────────────
    @property
    def main_indicator_data(self) -> dict[str, float]:
        return {
            "VWAP": self.state_map.vwap,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  生命周期
    # ══════════════════════════════════════════════════════════════════════

    def on_start(self) -> None:
        """策略启动: 初始化 K 线合成器，推送历史数据预热"""
        self.kline_generator = KLineGenerator(
            callback=self.callback,
            real_time_callback=self.real_time_callback,
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            style=self.params_map.kline_style,
        )

        # 必须在 super().on_start() 之前推送历史数据
        self.kline_generator.push_history_data()

        super().on_start()

        self.output(
            f"VwapZscoreReversion 启动 | "
            f"合约: {self.params_map.instrument_id}@{self.params_map.exchange} | "
            f"K线: {self.params_map.kline_style} | "
            f"z_threshold={self.params_map.z_threshold} "
            f"min_bars={self.params_map.min_bars} | "
            f"hard_stop={self.params_map.hard_stop_pct}%"
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
        self.state_map.net_pos = self.get_position(
            self.params_map.instrument_id
        ).net_position
        self.update_status_bar()

    # ══════════════════════════════════════════════════════════════════════
    #  K 线回调
    # ══════════════════════════════════════════════════════════════════════

    def callback(self, kline: KLineData) -> None:
        """K 线完成回调 -- 主策略逻辑

        执行顺序:
          1. 撤销所有未成交挂单
          2. 执行上一根 bar 产生的待执行信号 (next-bar 规则)
          3. 检测交易日切换，重置 VWAP 累积量
          4. 更新 VWAP 和 Z-Score
          5. 持仓中: 检查出场条件
          6. 空仓: 检查入场条件
        """
        signal_price = 0.0

        # ── 1. 撤销所有未成交挂单 ─────────────────────────────────────
        for oid in list(self.order_id):
            self.cancel_order(oid)

        # ── 2. 执行 pending 信号 (next-bar 规则) ─────────────────────
        if self._pending is not None:
            action = self._pending
            self._pending = None
            signal_price = self._execute_signal(kline, action)

        # ── 3. 交易日切换检测 & VWAP 重置 ────────────────────────────
        trading_day = self._get_trading_day(kline)
        if trading_day != self._current_day:
            self._current_day = trading_day
            self._cum_tp_vol = 0.0
            self._cum_vol = 0.0
            self._bars_today = 0
            self._deviations.clear()

        # ── 4. 更新 VWAP 和 Z-Score ──────────────────────────────────
        typical_price = (kline.high + kline.low + kline.close) / 3.0
        bar_volume = max(kline.volume, 1)  # 避免除以零

        self._cum_tp_vol += typical_price * bar_volume
        self._cum_vol += bar_volume
        self._bars_today += 1

        vwap = self._cum_tp_vol / self._cum_vol
        deviation = kline.close - vwap
        self._deviations.append(deviation)

        z_score = 0.0
        if len(self._deviations) >= self.params_map.min_bars:
            rolling_std = self._calc_std(self._deviations, self.params_map.min_bars)
            if rolling_std > 0:
                z_score = deviation / rolling_std

        # 更新状态显示
        self.state_map.vwap = round(vwap, 2)
        self.state_map.z_score = round(z_score, 2)
        self.state_map.bars_today = self._bars_today

        # ── 5. 数据不足，跳过交易逻辑 ────────────────────────────────
        if self._bars_today < self.params_map.min_bars:
            self._push_widget(kline, signal_price)
            self.update_status_bar()
            return

        # ── 6. 持仓中: 检查出场条件 ──────────────────────────────────
        if self.in_position:
            # 硬止损检查
            if self.position_side == "long":
                if kline.close <= self.entry_price * (1 - self.params_map.hard_stop_pct / 100):
                    self._pending = "STOP"
                elif z_score > 0:
                    # 价格回归到 VWAP 上方，信号反转平仓
                    self._pending = "EXIT_LONG"
                else:
                    self.state_map.last_action = "HOLD_LONG"

            elif self.position_side == "short":
                if kline.close >= self.entry_price * (1 + self.params_map.hard_stop_pct / 100):
                    self._pending = "STOP"
                elif z_score < 0:
                    # 价格回归到 VWAP 下方，信号反转平仓
                    self._pending = "EXIT_SHORT"
                else:
                    self.state_map.last_action = "HOLD_SHORT"

        # ── 7. 空仓: 检查入场条件 ────────────────────────────────────
        else:
            if z_score < -self.params_map.z_threshold:
                self._pending = "LONG"
            elif z_score > self.params_map.z_threshold:
                self._pending = "SHORT"

        self._push_widget(kline, signal_price)
        self.update_status_bar()

    def real_time_callback(self, kline: KLineData) -> None:
        """每个 Tick 都调用 -- 仅更新图表，不执行交易逻辑"""
        self._push_widget(kline)

    # ══════════════════════════════════════════════════════════════════════
    #  交易执行
    # ══════════════════════════════════════════════════════════════════════

    def _execute_signal(self, kline: KLineData, action: str) -> float:
        """根据 action 执行交易，返回 signal_price (供图表显示)"""
        if action == "LONG":
            return self._exec_open_long(kline)
        elif action == "SHORT":
            return self._exec_open_short(kline)
        elif action == "EXIT_LONG":
            return self._exec_close_long(kline, "EXIT_REVERT")
        elif action == "EXIT_SHORT":
            return self._exec_close_short(kline, "EXIT_REVERT")
        elif action == "STOP":
            if self.position_side == "long":
                return self._exec_close_long(kline, "STOP_LOSS")
            elif self.position_side == "short":
                return self._exec_close_short(kline, "STOP_LOSS")
        return 0.0

    def _exec_open_long(self, kline: KLineData) -> float:
        """执行开多仓"""
        price = kline.close

        order_id = self.send_order(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            volume=self.params_map.volume,
            price=price,
            order_direction="buy",
            market=True,
        )

        if order_id is not None:
            self.order_id.add(order_id)

        self.in_position = True
        self.position_side = "long"
        self.entry_price = price

        self.state_map.entry_price = price
        self.state_map.last_action = f"OPEN_LONG Z={self.state_map.z_score}"

        self.output(
            f"[开多] {self.params_map.volume}手 @ {price:.1f} | "
            f"Z={self.state_map.z_score:.2f} | VWAP={self.state_map.vwap:.1f}"
        )
        return price  # 正值 -> 图表绿色买入标记

    def _exec_open_short(self, kline: KLineData) -> float:
        """执行开空仓"""
        price = kline.close

        order_id = self.send_order(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            volume=self.params_map.volume,
            price=price,
            order_direction="sell",
            market=True,
        )

        if order_id is not None:
            self.order_id.add(order_id)

        self.in_position = True
        self.position_side = "short"
        self.entry_price = price

        self.state_map.entry_price = price
        self.state_map.last_action = f"OPEN_SHORT Z={self.state_map.z_score}"

        self.output(
            f"[开空] {self.params_map.volume}手 @ {price:.1f} | "
            f"Z={self.state_map.z_score:.2f} | VWAP={self.state_map.vwap:.1f}"
        )
        return -price  # 负值 -> 图表红色卖出标记

    def _exec_close_long(self, kline: KLineData, reason: str) -> float:
        """执行平多仓"""
        position = self.get_position(self.params_map.instrument_id)
        net_pos = position.net_position

        if net_pos <= 0:
            self.in_position = False
            self.position_side = ""
            return 0.0

        price = kline.close

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
            f"[平多] {reason} | {net_pos}手 @ {price:.1f} | "
            f"入场={self.entry_price:.1f} | 盈亏={pnl_pct:+.2f}%"
        )

        self.in_position = False
        self.position_side = ""
        self.entry_price = 0.0

        self.state_map.entry_price = 0.0
        self.state_map.last_action = reason

        return -price  # 负值 -> 图表红色卖出标记

    def _exec_close_short(self, kline: KLineData, reason: str) -> float:
        """执行平空仓"""
        position = self.get_position(self.params_map.instrument_id)
        net_pos = position.net_position

        if net_pos >= 0:
            self.in_position = False
            self.position_side = ""
            return 0.0

        close_vol = abs(net_pos)
        price = kline.close

        order_id = self.auto_close_position(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            volume=close_vol,
            price=price,
            order_direction="buy",
            market=True,
        )

        if order_id is not None:
            self.order_id.add(order_id)

        pnl_pct = (
            (self.entry_price - price) / self.entry_price * 100
            if self.entry_price > 0 else 0.0
        )
        self.output(
            f"[平空] {reason} | {close_vol}手 @ {price:.1f} | "
            f"入场={self.entry_price:.1f} | 盈亏={pnl_pct:+.2f}%"
        )

        self.in_position = False
        self.position_side = ""
        self.entry_price = 0.0

        self.state_map.entry_price = 0.0
        self.state_map.last_action = reason

        return price  # 正值 -> 图表绿色买入标记 (买入平空)

    # ══════════════════════════════════════════════════════════════════════
    #  辅助方法
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_trading_day(kline: KLineData) -> int:
        """
        获取交易日编号 (用于检测日切换、重置 VWAP)

        夜盘 (hour >= 21) 属于下一个交易日:
          - 周一 21:00 的 K 线 → 属于周二交易日
          - 所以 hour >= 21 时，day += 1
        返回一个整数编号用于比较即可 (不需要精确日期)
        """
        if not hasattr(kline, "datetime") or kline.datetime is None:
            return 0
        dt = kline.datetime
        day = dt.toordinal()
        if dt.hour >= 21:
            day += 1
        return day

    @staticmethod
    def _calc_std(data: deque, window: int) -> float:
        """计算 deque 末尾 window 个元素的标准差"""
        if len(data) < window:
            return 0.0
        # 取最近 window 个值
        values = list(data)[-window:]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return sqrt(variance) if variance > 0 else 0.0

    def _push_widget(self, kline: KLineData, signal_price: float = 0.0) -> None:
        """更新 K 线图表"""
        try:
            self.widget.recv_kline({
                "kline": kline,
                "signal_price": signal_price,
                **self.main_indicator_data,
            })
        except Exception:
            pass
