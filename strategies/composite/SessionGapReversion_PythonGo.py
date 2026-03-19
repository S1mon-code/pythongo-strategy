"""
================================================================================
  Session-Gap Reversion 铁矿石 CTA 策略 — 跳空回归
================================================================================

  适用平台   : 无限易 PythonGo (BaseStrategy)
  合约       : 铁矿石主力 (I9999 / I2605 等)
  K线周期    : 5 分钟 (M5)
  方向       : 多空双向 (Long + Short)
  版本       : 1.0
  日期       : 2026-03-19

================================================================================
  回测结果 (2023-01 ~ 2026-02)
  ─────────────────────────────────────────────
  Sharpe: 0.56 | Calmar: 0.31 | Max DD: 2.46%
  Annual Return: 0.75% | Win Rate: 40.9% | PF: 1.55
  Trades/Year: ~7 | Yearly: +2.28%, +0.37%, +0.14%, -0.38%
  Best Params: gap_threshold=0.005, atr_mult=1.0
================================================================================

  策略逻辑:
  - 检测日盘开盘 (09:00) 与前一交易时段收盘价之间的跳空
  - 跳空幅度 > 0.5%: 做空 (预期跳空回补)
  - 跳空幅度 < -0.5%: 做多 (预期跳空回补)
  - 交易窗口: 跳空后前 30 分钟 (前 6 根 5 分钟 K 线)
  - 出场: 信号反转 或 硬止损 5%
  - ATR(20) 用于仓位参考

================================================================================
  运行逻辑: on_start -> [ on_tick -> (real_time_callback | callback) ] 循环
  - on_tick:             接收 Tick，推送给 KLineGenerator 合成 K 线
  - real_time_callback:  每个 Tick 调用，仅更新图表（不交易）
  - callback:            K 线完成时调用，主策略逻辑在此执行
================================================================================
"""

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
    kline_style: KLineStyleType = Field(default="M5", title="K线周期")
    gap_threshold: float = Field(default=0.5, title="跳空阈值(%)")
    hard_stop_pct: float = Field(default=5.0, title="硬止损(%)")
    trade_window: int = Field(default=6, title="交易窗口(bars)")


class State(BaseState):
    """状态映射模型 -- 显示在无限易状态栏"""
    net_pos: int = Field(default=0, title="当前持仓")
    gap_pct: float = Field(default=0.0, title="跳空幅度(%)")
    atr_20: float = Field(default=0.0, title="ATR(20)")
    entry_price: float = Field(default=0.0, title="入场价格")
    prev_close: float = Field(default=0.0, title="前收盘价")
    day_open: float = Field(default=0.0, title="日盘开盘价")
    bars_since_gap: int = Field(default=0, title="跳空后bars")
    last_action: str = Field(default="FLAT", title="最近动作")


# ==============================================================================
#  策略类
# ==============================================================================

class SessionGapReversion_PythonGo(BaseStrategy):
    """跳空回归策略 -- 日盘跳空 > 阈值时逆向交易，期望价格回补缺口"""

    def __init__(self) -> None:
        super().__init__()

        self.params_map = Params()
        self.state_map = State()

        self.kline_generator: KLineGenerator = None

        # -- 持仓状态 --
        self.in_position: bool = False
        self.entry_price: float = 0.0
        self.position_side: str = ""  # "long" or "short"

        # -- 跳空检测状态 --
        self.prev_session_close: float = 0.0   # 前一交易时段最后收盘价
        self.day_open_price: float = 0.0        # 日盘开盘价
        self.gap_pct: float = 0.0               # 当前跳空幅度
        self.gap_detected: bool = False         # 当日是否检测到跳空
        self.bars_since_gap: int = 0            # 跳空后经过的 bar 数
        self.last_bar_date: int = -1            # 上一根 bar 的日期 (用于重置)

        # -- 待执行信号 (next-bar 规则) --
        self._pending_entry: str | None = None    # "long" or "short"
        self._pending_exit: str | None = None     # reason str

        # -- 委托 ID 集合 --
        self.order_id: set[int] = set()

    # -- 主图指标 (显示在 K 线图上) --
    @property
    def main_indicator_data(self) -> dict[str, float]:
        return {
            "ATR20": self.state_map.atr_20,
        }

    # ======================================================================
    #  生命周期
    # ======================================================================

    def on_start(self) -> None:
        """策略启动：初始化 K 线合成器，推送历史数据预热指标"""
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
            f"SessionGapReversion 启动 | "
            f"合约: {self.params_map.instrument_id}@{self.params_map.exchange} | "
            f"K线: {self.params_map.kline_style} | "
            f"gap_threshold={self.params_map.gap_threshold}% | "
            f"hard_stop={self.params_map.hard_stop_pct}% | "
            f"trade_window={self.params_map.trade_window} bars"
        )

    def on_stop(self) -> None:
        super().on_stop()

    # ======================================================================
    #  行情 / 委托 / 成交 回调
    # ======================================================================

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

    # ======================================================================
    #  K 线回调
    # ======================================================================

    def callback(self, kline: KLineData) -> None:
        """K 线完成回调 -- 主策略逻辑

        执行顺序:
          1. 撤销所有未成交挂单
          2. 执行上一根 bar 产生的待执行信号 (next-bar 规则)
          3. 计算指标 (数据不足则跳过交易逻辑)
          4. 跳空检测与日期重置
          5. 持仓中: 检查出场条件
          6. 空仓: 检查入场条件 (窗口期内)
        """
        signal_price = 0.0

        # -- 1. 撤销所有未成交挂单 --
        for oid in list(self.order_id):
            self.cancel_order(oid)

        # -- 2. 执行 pending 信号 (next-bar 规则) --
        if self._pending_exit is not None:
            reason = self._pending_exit
            self._pending_exit = None
            signal_price = self._exec_exit(kline, reason)

        if self._pending_entry is not None:
            side = self._pending_entry
            self._pending_entry = None
            signal_price = self._exec_entry(kline, side)

        # -- 3. 计算指标 --
        if not self._calc_indicator():
            self._push_widget(kline, signal_price)
            return

        producer = self.kline_generator.producer
        current_close = float(producer.close[-1])

        # -- 4. 跳空检测与日期重置 --
        bar_dt = kline.datetime
        bar_date = bar_dt.year * 10000 + bar_dt.month * 100 + bar_dt.day
        h, m = bar_dt.hour, bar_dt.minute

        # 新的一天: 重置跳空检测状态
        if bar_date != self.last_bar_date:
            self.last_bar_date = bar_date
            self.gap_detected = False
            self.bars_since_gap = 0
            self.day_open_price = 0.0
            self.gap_pct = 0.0

        # 检测日盘开盘 (09:00 的第一根 M5 bar)
        if h == 9 and m == 0 and self.day_open_price == 0.0:
            self.day_open_price = float(kline.open)
            self.state_map.day_open = self.day_open_price

            if self.prev_session_close > 0:
                self.gap_pct = (
                    (self.day_open_price - self.prev_session_close)
                    / self.prev_session_close * 100
                )
                self.state_map.gap_pct = round(self.gap_pct, 4)

                threshold = self.params_map.gap_threshold

                if abs(self.gap_pct) > threshold:
                    self.gap_detected = True
                    self.bars_since_gap = 0
                    self.output(
                        f"[跳空检测] gap={self.gap_pct:+.3f}% | "
                        f"prev_close={self.prev_session_close:.1f} | "
                        f"day_open={self.day_open_price:.1f}"
                    )

        # 跳空后计数 bar
        if self.gap_detected:
            self.bars_since_gap += 1
            self.state_map.bars_since_gap = self.bars_since_gap

        # 每根 bar 都更新 prev_session_close (最终保留当天最后一根 bar 的收盘价)
        # 但只在非 09:00 开盘 bar 时更新，避免用开盘 bar 覆盖
        if not (h == 9 and m == 0):
            self.prev_session_close = current_close
            self.state_map.prev_close = round(self.prev_session_close, 2)

        # -- 5. 持仓中: 检查出场条件 --
        if self.in_position:
            # 硬止损
            if self.position_side == "long":
                if current_close <= self.entry_price * (1 - self.params_map.hard_stop_pct / 100):
                    self._pending_exit = "EXIT_HARD_STOP"
                # 信号反转: 跳空回补完成 (价格回到 prev_close 以下)
                elif self.prev_session_close > 0 and current_close <= self.prev_session_close:
                    self._pending_exit = "EXIT_GAP_FILLED"
            elif self.position_side == "short":
                if current_close >= self.entry_price * (1 + self.params_map.hard_stop_pct / 100):
                    self._pending_exit = "EXIT_HARD_STOP"
                # 信号反转: 跳空回补完成 (价格回到 prev_close 以上)
                elif self.prev_session_close > 0 and current_close >= self.prev_session_close:
                    self._pending_exit = "EXIT_GAP_FILLED"

            if self._pending_exit is None:
                self.state_map.last_action = "HOLD"

        # -- 6. 空仓: 检查入场条件 (窗口期内) --
        else:
            if (
                self.gap_detected
                and self.bars_since_gap <= self.params_map.trade_window
            ):
                threshold = self.params_map.gap_threshold
                if self.gap_pct > threshold:
                    # 跳空高开 -> 做空 (fade the gap)
                    self._pending_entry = "short"
                elif self.gap_pct < -threshold:
                    # 跳空低开 -> 做多 (fade the gap)
                    self._pending_entry = "long"

        self._push_widget(kline, signal_price)
        self.update_status_bar()

    def real_time_callback(self, kline: KLineData) -> None:
        """每个 Tick 都调用 -- 仅更新图表，不执行交易逻辑"""
        self._calc_indicator()
        self._push_widget(kline)

    # ======================================================================
    #  交易执行
    # ======================================================================

    def _exec_entry(self, kline: KLineData, side: str) -> float:
        """执行开仓，返回 signal_price (供图表显示)"""
        price = kline.close
        volume = self.params_map.volume

        if side == "long":
            order_id = self.send_order(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=volume,
                price=price,
                order_direction="buy",
                market=True,
            )
        else:  # short
            order_id = self.send_order(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=volume,
                price=price,
                order_direction="sell",
                market=True,
            )

        if order_id is not None:
            self.order_id.add(order_id)

        self.in_position = True
        self.entry_price = price
        self.position_side = side

        self.state_map.entry_price = price
        self.state_map.last_action = f"ENTER_{side.upper()} gap={self.gap_pct:+.2f}%"

        direction_cn = "开多" if side == "long" else "开空"
        self.output(
            f"[{direction_cn}] {volume}手 @ {price:.1f} | "
            f"gap={self.gap_pct:+.3f}% | next-bar执行"
        )
        # 正值 = 买入标记 (绿), 负值 = 卖出标记 (红)
        return price if side == "long" else -price

    def _exec_exit(self, kline: KLineData, reason: str) -> float:
        """执行平仓，返回 signal_price (供图表显示)"""
        position = self.get_position(self.params_map.instrument_id)
        net_pos = position.net_position
        price = kline.close

        if net_pos == 0:
            self.in_position = False
            self.position_side = ""
            return 0.0

        if net_pos > 0:
            # 平多仓
            order_id = self.auto_close_position(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=net_pos,
                price=price,
                order_direction="sell",
                market=True,
            )
        else:
            # 平空仓
            order_id = self.auto_close_position(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=abs(net_pos),
                price=price,
                order_direction="buy",
                market=True,
            )

        if order_id is not None:
            self.order_id.add(order_id)

        # 盈亏计算
        if self.position_side == "long":
            pnl_pct = (price - self.entry_price) / self.entry_price * 100 if self.entry_price > 0 else 0.0
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price * 100 if self.entry_price > 0 else 0.0

        direction_cn = "平多" if self.position_side == "long" else "平空"
        self.output(
            f"[{direction_cn}] {reason} | {abs(net_pos)}手 @ {price:.1f} | "
            f"入场={self.entry_price:.1f} | 盈亏={pnl_pct:+.2f}%"
        )

        # 平多返回负值(红), 平空返回正值(绿)
        signal_ret = -price if self.position_side == "long" else price

        self.in_position = False
        self.entry_price = 0.0
        self.position_side = ""

        self.state_map.entry_price = 0.0
        self.state_map.last_action = reason

        return signal_ret

    # ======================================================================
    #  指标计算
    # ======================================================================

    def _calc_indicator(self) -> bool:
        """计算 ATR(20)，返回 True 表示数据充足可执行交易逻辑"""
        if self.kline_generator is None:
            return False

        producer = self.kline_generator.producer
        min_bars = 20  # ATR lookback

        if len(producer.close) < min_bars:
            return False

        atr_arr = producer.atr(20, array=True)
        self.state_map.atr_20 = round(float(atr_arr[-1]), 2)

        return True

    # ======================================================================
    #  辅助方法
    # ======================================================================

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
