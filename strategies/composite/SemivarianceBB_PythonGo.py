"""
================================================================================
  Semivariance BB 均值回归策略 — 铁矿石 CTA
================================================================================
  适用平台   : 无限易 PythonGo (BaseStrategy)
  合约       : 铁矿石主力 (I9999 / I2605 等)
  K线周期    : 5 分钟 (M5)
  方向       : 多空双向 (Long + Short)
  版本       : 1.0
  日期       : 2026-03-20

  回测结果 (2023-01 ~ 2025-12)
  Sharpe: 1.51 | Max DD: 1.44% | Annual Return: 2.36%
  Win Rate: 55.9% | PF: 1.76 | Trades/Year: ~62
  Yearly: +2.03% (2023), +1.52% (2024), +3.18% (2025)
  Best Params: sv_window=30, bb_period=20, asym_threshold=0.55

  策略逻辑:
  - 正半方差 RS+ = sum(max(ret,0)^2) over sv_window bars
  - 负半方差 RS- = sum(max(-ret,0)^2) over sv_window bars
  - 不对称比 asym = RS+ / (RS+ + RS-)
  - 做多: price 跌破 BB 下轨 AND asym > 0.55 (上行波动占优，均值回归更有把握)
  - 做空: price 突破 BB 上轨 AND asym < 0.45
  - 出场: 反向穿越BB对面轨道 或 硬止损 5%

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
    """参数映射模型 -- 显示在无限易界面"""
    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="i2605", title="合约代码")
    volume: int = Field(default=1, title="每手下单量", ge=1)
    kline_style: KLineStyleType = Field(default="M5", title="K线周期")
    sv_window: int = Field(default=30, title="半方差窗口(bars)")
    bb_period: int = Field(default=20, title="布林带周期(bars)")
    asym_threshold: float = Field(default=0.55, title="不对称比阈值")
    hard_stop_pct: float = Field(default=5.0, title="硬止损(%)")


class State(BaseState):
    """状态映射模型 -- 显示在无限易状态栏"""
    bb_upper: float = Field(default=0.0, title="BB上轨")
    bb_lower: float = Field(default=0.0, title="BB下轨")
    asym: float = Field(default=0.5, title="不对称比(asym)")
    net_pos: int = Field(default=0, title="当前持仓")
    entry_price: float = Field(default=0.0, title="入场价格")
    last_action: str = Field(default="FLAT", title="最近动作")


# ══════════════════════════════════════════════════════════════════════════════
#  策略类
# ══════════════════════════════════════════════════════════════════════════════

class SemivarianceBB_PythonGo(BaseStrategy):
    """Semivariance BB 均值回归策略 -- 多空双向"""

    def __init__(self) -> None:
        super().__init__()

        self.params_map = Params()
        self.state_map = State()

        self.kline_generator: KLineGenerator = None

        # ── 持仓状态 ────────────────────────────────────────────────────
        self.in_position: bool = False
        self.position_side: str = ""      # "long" or "short"
        self.entry_price: float = 0.0

        # ── 跨 bar 跟踪 (用于穿越检测) ──────────────────────────────────
        self.prev_close: float = 0.0
        self.prev_upper: float = 0.0
        self.prev_lower: float = 0.0

        # ── 待执行信号 (next-bar 规则) ──────────────────────────────────
        self._pending: str | None = None  # "LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT", "STOP"

        # ── 委托 ID 集合 ─────────────────────────────────────────────────
        self.order_id: set[int] = set()

    # ── 主图指标 ─────────────────────────────────────────────────────────
    @property
    def main_indicator_data(self) -> dict[str, float]:
        return {
            "BB_Upper": self.state_map.bb_upper,
            "BB_Lower": self.state_map.bb_lower,
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
            f"SemivarianceBB 启动 | "
            f"合约: {self.params_map.instrument_id}@{self.params_map.exchange} | "
            f"K线: {self.params_map.kline_style} | "
            f"sv_window={self.params_map.sv_window} "
            f"bb_period={self.params_map.bb_period} "
            f"asym_threshold={self.params_map.asym_threshold} | "
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
          3. 计算半方差不对称比 (asym) 和布林带 (BB)
          4. 数据不足则跳过交易逻辑
          5. 持仓中: 检查出场条件 (硬止损 / 反向穿越对面轨道)
          6. 空仓: 检查入场条件 (穿越BB轨道 + asym过滤)
          7. 更新 prev_close / prev_upper / prev_lower
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

        # ── 3. 计算指标 ───────────────────────────────────────────────
        producer = self.kline_generator.producer
        close_arr = producer.close  # numpy array of close prices

        n_sv = self.params_map.sv_window
        n_bb = self.params_map.bb_period

        # 需要至少 max(n_sv+1, n_bb) 根 K 线
        min_required = max(n_sv + 1, n_bb)
        if len(close_arr) < min_required:
            self._push_widget(kline, signal_price)
            self.update_status_bar()
            return

        # ── 半方差不对称比 ────────────────────────────────────────────
        sv_window_prices = close_arr[-(n_sv + 1):]   # sv_window+1 个价格
        rets = np.diff(sv_window_prices) / sv_window_prices[:-1]  # sv_window 个收益率

        rs_pos = float(np.sum(np.maximum(rets, 0) ** 2))
        rs_neg = float(np.sum(np.maximum(-rets, 0) ** 2))
        rs_total = rs_pos + rs_neg
        asym = rs_pos / rs_total if rs_total > 0 else 0.5

        # ── 布林带 ────────────────────────────────────────────────────
        bb_window = close_arr[-n_bb:]
        mid = float(np.mean(bb_window))
        std = float(np.std(bb_window, ddof=0))
        upper = mid + 2.0 * std
        lower = mid - 2.0 * std
        current_close = float(close_arr[-1])

        # 更新状态显示
        self.state_map.bb_upper = round(upper, 2)
        self.state_map.bb_lower = round(lower, 2)
        self.state_map.asym = round(asym, 4)

        # ── 4. 穿越检测 (需要 prev 值初始化完毕) ─────────────────────
        # prev_close == 0.0 表示首根有效 bar，跳过信号检测
        if self.prev_close == 0.0:
            self.prev_close = current_close
            self.prev_upper = upper
            self.prev_lower = lower
            self._push_widget(kline, signal_price)
            self.update_status_bar()
            return

        cross_below = (
            self.prev_close >= self.prev_lower
            and current_close < lower
        )
        cross_above = (
            self.prev_close <= self.prev_upper
            and current_close > upper
        )

        # ── 5. 持仓中: 检查出场条件 ──────────────────────────────────
        if self.in_position:
            if self.position_side == "long":
                # 硬止损
                if current_close <= self.entry_price * (
                    1 - self.params_map.hard_stop_pct / 100
                ):
                    self._pending = "STOP"
                # 反向穿越 BB 上轨 → 平多
                elif cross_above:
                    self._pending = "EXIT_LONG"
                else:
                    self.state_map.last_action = "HOLD_LONG"

            elif self.position_side == "short":
                # 硬止损
                if current_close >= self.entry_price * (
                    1 + self.params_map.hard_stop_pct / 100
                ):
                    self._pending = "STOP"
                # 反向穿越 BB 下轨 → 平空
                elif cross_below:
                    self._pending = "EXIT_SHORT"
                else:
                    self.state_map.last_action = "HOLD_SHORT"

        # ── 6. 空仓: 检查入场条件 ────────────────────────────────────
        else:
            if cross_below and asym > self.params_map.asym_threshold:
                # 价格跌破下轨 且 上行波动占优 → 预期均值回归做多
                self._pending = "LONG"
            elif cross_above and asym < (1.0 - self.params_map.asym_threshold):
                # 价格突破上轨 且 下行波动占优 → 预期均值回归做空
                self._pending = "SHORT"

        # ── 7. 更新 prev 值 ──────────────────────────────────────────
        self.prev_close = current_close
        self.prev_upper = upper
        self.prev_lower = lower

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
        self.state_map.last_action = (
            f"OPEN_LONG asym={self.state_map.asym:.3f}"
        )

        self.output(
            f"[开多] {self.params_map.volume}手 @ {price:.1f} | "
            f"asym={self.state_map.asym:.3f} | "
            f"BB_Lower={self.state_map.bb_lower:.1f}"
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
        self.state_map.last_action = (
            f"OPEN_SHORT asym={self.state_map.asym:.3f}"
        )

        self.output(
            f"[开空] {self.params_map.volume}手 @ {price:.1f} | "
            f"asym={self.state_map.asym:.3f} | "
            f"BB_Upper={self.state_map.bb_upper:.1f}"
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
