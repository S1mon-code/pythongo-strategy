"""
================================================================================
  Micro Mean-Reversion 铁矿石 CTA 策略 — 布林带均值回归
================================================================================

  适用平台   : 无限易 PythonGo (BaseStrategy)
  合约       : 铁矿石主力 (I9999 / I2605 等)
  K线周期    : 1 分钟 (M1)，布林带周期=30 等效于 2 分钟 K 线 15 根
  方向       : 多空双向 (Long + Short)
  版本       : 1.0
  日期       : 2026-03-19

================================================================================
  回测结果 (2023-01 ~ 2026-02)
================================================================================

  Sharpe: 1.17 | Calmar: 0.90 | Max DD: 2.09%
  Annual Return: 1.89% | Win Rate: 63.8% | PF: 1.16
  Trades/Year: ~1034 | Yearly: +1.08%, +2.66%, +1.48%, +0.61%
  Best Params: bb_period=15, bb_std=2.0
  (M1 等效参数: bb_period=30, bb_std=2.0)

================================================================================
  运行逻辑: on_start -> [ on_tick -> (real_time_callback | callback) ] 循环
  - on_tick:             接收 Tick，推送给 KLineGenerator 合成 K 线
  - real_time_callback:  每个 Tick 调用，仅更新图表（不交易）
  - callback:            K 线完成时调用，主策略逻辑在此执行

  信号逻辑:
  - 做多: 前一根 close > lower，当前 close <= lower (跌破下轨，均值回归买入)
  - 做空: 前一根 close < upper，当前 close >= upper (突破上轨，均值回归卖出)
  - 出场: 反向信号出现 (信号反转) 或硬止损 5%
================================================================================
"""

import numpy as np

from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import KLineData, OrderData, TickData, TradeData
from pythongo.core import KLineStyleType
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGenerator


# ==============================================================================
#  参数 / 状态 模型
# ==============================================================================

class Params(BaseParams):
    """参数映射模型 -- 显示在无限易界面"""
    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="i2605", title="合约代码")
    volume: int = Field(default=1, title="每手下单量", ge=1)
    kline_style: KLineStyleType = Field(default="M1", title="K线周期")
    bb_period: int = Field(default=30, title="布林带周期", ge=5)
    bb_std: float = Field(default=2.0, title="布林带标准差倍数")
    hard_stop_pct: float = Field(default=5.0, title="硬止损(%)")


class State(BaseState):
    """状态映射模型 -- 显示在无限易状态栏"""
    bb_upper: float = Field(default=0.0, title="布林上轨")
    bb_mid: float = Field(default=0.0, title="布林中轨")
    bb_lower: float = Field(default=0.0, title="布林下轨")
    net_pos: int = Field(default=0, title="当前持仓")
    entry_price: float = Field(default=0.0, title="入场价格")
    last_action: str = Field(default="FLAT", title="最近动作")


# ==============================================================================
#  策略类
# ==============================================================================

class MicroMeanReversion_PythonGo(BaseStrategy):
    """布林带微观均值回归策略 -- 多空双向，信号反转出场，硬止损 5%"""

    def __init__(self) -> None:
        super().__init__()

        self.params_map = Params()
        self.state_map = State()

        self.kline_generator: KLineGenerator = None

        # -- 持仓状态 ----------------------------------------------------------
        self.in_position: bool = False
        self.position_side: str = ""       # "long", "short", "" = flat
        self.entry_price: float = 0.0

        # -- 前一根 bar 的收盘价与布林带值 (用于判断穿越) -----------------------
        self.prev_close: float = 0.0
        self.prev_upper: float = 0.0
        self.prev_lower: float = 0.0

        # -- 待执行信号 (next-bar 规则) ----------------------------------------
        self._pending: str | None = None  # "LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT", "STOP"

        # -- 委托 ID 集合 ------------------------------------------------------
        self.order_id: set[int] = set()

    # -- 主图指标 (显示在 K 线图上) --------------------------------------------
    @property
    def main_indicator_data(self) -> dict[str, float]:
        return {
            "BB_Upper": self.state_map.bb_upper,
            "BB_Mid": self.state_map.bb_mid,
            "BB_Lower": self.state_map.bb_lower,
        }

    # ==========================================================================
    #  生命周期
    # ==========================================================================

    def on_start(self) -> None:
        """策略启动: 初始化 K 线合成器，推送历史数据预热指标"""
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
            f"MicroMeanReversion 启动 | "
            f"合约: {self.params_map.instrument_id}@{self.params_map.exchange} | "
            f"K线: {self.params_map.kline_style} | "
            f"bb_period={self.params_map.bb_period} bb_std={self.params_map.bb_std} | "
            f"hard_stop={self.params_map.hard_stop_pct}%"
        )

    def on_stop(self) -> None:
        super().on_stop()

    # ==========================================================================
    #  行情 / 委托 / 成交 回调
    # ==========================================================================

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

    # ==========================================================================
    #  K 线回调
    # ==========================================================================

    def callback(self, kline: KLineData) -> None:
        """K 线完成回调 -- 主策略逻辑

        执行顺序:
          1. 撤销所有未成交挂单
          2. 执行上一根 bar 产生的待执行信号 (next-bar 规则)
          3. 计算布林带指标 (数据不足则跳过)
          4. 持仓中: 检查硬止损 / 反向信号
          5. 空仓: 检查入场条件
          6. 保存当前 bar 数据供下一根 bar 判断穿越
        """
        signal_price = 0.0

        # -- 1. 撤销所有未成交挂单 ---------------------------------------------
        for oid in list(self.order_id):
            self.cancel_order(oid)

        # -- 2. 执行 pending 信号 (next-bar 规则) ------------------------------
        if self._pending is not None:
            action = self._pending
            self._pending = None
            signal_price = self._execute_signal(kline, action)

        # -- 3. 计算布林带 -----------------------------------------------------
        bb = self._calc_bollinger()
        if bb is None:
            self._push_widget(kline, signal_price)
            return

        upper, mid, lower, current_close = bb

        # 更新状态显示
        self.state_map.bb_upper = round(upper, 2)
        self.state_map.bb_mid = round(mid, 2)
        self.state_map.bb_lower = round(lower, 2)

        # -- 4/5. 信号生成 (优先级: 止损 > 反向出场 > 入场) --------------------
        if self.in_position and self.position_side == "long":
            # 持多仓
            if self.entry_price > 0 and current_close <= self.entry_price * (1 - self.params_map.hard_stop_pct / 100):
                self._pending = "STOP_LONG"
            elif self.prev_close < upper and current_close >= upper:
                # 反向信号: 触及上轨 -> 平多开空
                self._pending = "REVERSE_TO_SHORT"
            else:
                self.state_map.last_action = "HOLD_LONG"

        elif self.in_position and self.position_side == "short":
            # 持空仓
            if self.entry_price > 0 and current_close >= self.entry_price * (1 + self.params_map.hard_stop_pct / 100):
                self._pending = "STOP_SHORT"
            elif self.prev_close > lower and current_close <= lower:
                # 反向信号: 触及下轨 -> 平空开多
                self._pending = "REVERSE_TO_LONG"
            else:
                self.state_map.last_action = "HOLD_SHORT"

        else:
            # 空仓: 检查入场
            if self.prev_close > lower and current_close <= lower:
                self._pending = "LONG"
            elif self.prev_close < upper and current_close >= upper:
                self._pending = "SHORT"

        # -- 6. 保存当前 bar 数据 ----------------------------------------------
        self.prev_close = current_close
        self.prev_upper = upper
        self.prev_lower = lower

        self._push_widget(kline, signal_price)
        self.update_status_bar()

    def real_time_callback(self, kline: KLineData) -> None:
        """每个 Tick 都调用 -- 仅更新图表，不执行交易逻辑"""
        self._calc_bollinger()
        self._push_widget(kline)

    # ==========================================================================
    #  交易执行
    # ==========================================================================

    def _execute_signal(self, kline: KLineData, action: str) -> float:
        """根据 action 执行对应交易，返回 signal_price 用于图表标记"""
        price = kline.close

        if action == "LONG":
            return self._exec_open_long(kline, price)

        elif action == "SHORT":
            return self._exec_open_short(kline, price)

        elif action == "STOP_LONG":
            return self._exec_close_long(kline, price, "STOP_LONG")

        elif action == "STOP_SHORT":
            return self._exec_close_short(kline, price, "STOP_SHORT")

        elif action == "REVERSE_TO_SHORT":
            # 先平多，再开空
            self._exec_close_long(kline, price, "EXIT_LONG->SHORT")
            return self._exec_open_short(kline, price)

        elif action == "REVERSE_TO_LONG":
            # 先平空，再开多
            self._exec_close_short(kline, price, "EXIT_SHORT->LONG")
            return self._exec_open_long(kline, price)

        return 0.0

    def _exec_open_long(self, kline: KLineData, price: float) -> float:
        """开多仓"""
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
        self.state_map.last_action = "OPEN_LONG"

        self.output(f"[开多] {self.params_map.volume}手 @ {price:.1f} | next-bar执行")
        return price  # 正值 -> 绿色买入标记

    def _exec_open_short(self, kline: KLineData, price: float) -> float:
        """开空仓"""
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
        self.state_map.last_action = "OPEN_SHORT"

        self.output(f"[开空] {self.params_map.volume}手 @ {price:.1f} | next-bar执行")
        return -price  # 负值 -> 红色卖出标记

    def _exec_close_long(self, kline: KLineData, price: float, reason: str) -> float:
        """平多仓"""
        position = self.get_position(self.params_map.instrument_id)
        net_pos = position.net_position

        if net_pos <= 0:
            self.in_position = False
            self.position_side = ""
            return 0.0

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

        pnl_pct = (price - self.entry_price) / self.entry_price * 100 if self.entry_price > 0 else 0.0
        self.output(
            f"[平多] {reason} | {net_pos}手 @ {price:.1f} | "
            f"入场={self.entry_price:.1f} | 盈亏={pnl_pct:+.2f}%"
        )

        self.in_position = False
        self.position_side = 0
        self.entry_price = 0.0
        self.state_map.entry_price = 0.0
        self.state_map.last_action = reason

        return -price  # 负值 -> 红色卖出标记

    def _exec_close_short(self, kline: KLineData, price: float, reason: str) -> float:
        """平空仓"""
        position = self.get_position(self.params_map.instrument_id)
        net_pos = position.net_position

        if net_pos >= 0:
            self.in_position = False
            self.position_side = ""
            return 0.0

        close_vol = abs(net_pos)

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

        pnl_pct = (self.entry_price - price) / self.entry_price * 100 if self.entry_price > 0 else 0.0
        self.output(
            f"[平空] {reason} | {close_vol}手 @ {price:.1f} | "
            f"入场={self.entry_price:.1f} | 盈亏={pnl_pct:+.2f}%"
        )

        self.in_position = False
        self.position_side = 0
        self.entry_price = 0.0
        self.state_map.entry_price = 0.0
        self.state_map.last_action = reason

        return price  # 正值 -> 绿色买入标记 (买入平空)

    # ==========================================================================
    #  指标计算
    # ==========================================================================

    def _calc_bollinger(self) -> tuple[float, float, float, float] | None:
        """
        计算布林带指标，返回 (upper, mid, lower, current_close)。
        数据不足时返回 None。
        """
        if self.kline_generator is None:
            return None

        producer = self.kline_generator.producer
        close_arr = producer.close

        if len(close_arr) < self.params_map.bb_period:
            return None

        window = close_arr[-self.params_map.bb_period:]
        mid = float(np.mean(window))
        std = float(np.std(window, ddof=0))

        upper = mid + self.params_map.bb_std * std
        lower = mid - self.params_map.bb_std * std
        current_close = float(close_arr[-1])

        # 更新状态显示
        self.state_map.bb_upper = round(upper, 2)
        self.state_map.bb_mid = round(mid, 2)
        self.state_map.bb_lower = round(lower, 2)

        return upper, mid, lower, current_close

    # ==========================================================================
    #  辅助方法
    # ==========================================================================

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
