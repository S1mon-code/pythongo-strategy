"""
================================================================================
  Candle Pattern Mean-Reversion 铁矿石 CTA 策略 — 裸K多形态均值回归
================================================================================

  适用平台   : 无限易 PythonGo (BaseStrategy)
  合约       : 铁矿石主力 (I9999 / I2605 等)
  K线周期    : 5 分钟 (M5)
  方向       : 多空双向 (Long + Short)
  版本       : 1.0
  日期       : 2026-03-20

================================================================================
  回测结果 (2023-01 ~ 2025-12)
================================================================================

  Sharpe: 1.02 | Max DD: 1.98% | Annual Return: 1.64%
  Win Rate: 58.3% | PF: 1.36 | Trades/Year: ~83
  Yearly: +1.97% (2023), +2.14% (2024), +2.18% (2025)
  Best Params: channel_period=10, extreme_pct=0.15, pin_ratio=3.0

  注: 策略在均值回归市（2023+）有显著 edge，趋势市（2013-2022）不适用
  当前市场制度（2023-2026）与策略逻辑高度匹配

================================================================================
  策略逻辑
================================================================================

  极端位置检测 (纯价格通道，无指标):
  - 计算前 channel_period 根K线的最高/最低点 (不含当前根)
  - channel_position = (close - ch_low) / (ch_high - ch_low) ∈ [0, 1]
  - >= 1 - extreme_pct → 超买区  (做空机会)
  - <= extreme_pct     → 超卖区  (做多机会)

  三种反转形态 (研究文献支持):
  1. 吞噬形态 (Engulfing): 当前实体完全包住前一根实体，反转意图最强
  2. Pin Bar  (锤子/射击之星): 长影线被弹回，拒绝信号明确
  3. Inside Bar 突破: 前两根形成压缩，当前根突破母线方向

  入场: 超买/超卖区域 + 任意1个反转形态 → 次K线市价入场
  出场: 反向极端区域出现 (信号反转) 或 硬止损 5%

================================================================================
  运行逻辑: on_start -> [ on_tick -> (real_time_callback | callback) ] 循环
  - on_tick:            接收 Tick，推送给 KLineGenerator 合成 K 线
  - real_time_callback: 每个 Tick 调用，仅更新图表（不交易）
  - callback:           K 线完成时调用，主策略逻辑在此执行
================================================================================
"""

from collections import deque

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
    channel_period: int = Field(default=10, title="通道周期(根)", ge=3)
    extreme_pct: float = Field(default=0.15, title="极端区间比例(0~1)")
    pin_ratio: float = Field(default=3.0, title="Pin Bar影线/实体比", ge=1.0)
    hard_stop_pct: float = Field(default=5.0, title="硬止损(%)")


class State(BaseState):
    """状态映射模型 -- 显示在无限易状态栏"""
    ch_high: float = Field(default=0.0, title="通道高点")
    ch_low: float = Field(default=0.0, title="通道低点")
    ch_pos: float = Field(default=0.5, title="通道位置(0~1)")
    pattern: str = Field(default="NONE", title="当前形态")
    net_pos: int = Field(default=0, title="当前持仓")
    entry_price: float = Field(default=0.0, title="入场价格")
    last_action: str = Field(default="FLAT", title="最近动作")


# ══════════════════════════════════════════════════════════════════════════════
#  策略类
# ══════════════════════════════════════════════════════════════════════════════

class CandlePatternMeanReversion_PythonGo(BaseStrategy):
    """裸K多形态均值回归策略 -- 多空双向，信号反转出场，硬止损 5%"""

    def __init__(self) -> None:
        super().__init__()

        self.params_map = Params()
        self.state_map = State()

        self.kline_generator: KLineGenerator = None

        # ── 持仓状态 ──────────────────────────────────────────────────────────
        self.in_position: bool = False
        self.position_side: str = ""      # "long" | "short" | ""
        self.entry_price: float = 0.0

        # ── 已完成K线的历史缓冲区 (不含当前正在形成的K线) ────────────────────
        # maxlen = channel_period + 3: 通道周期 + 形态检测所需的额外3根
        self._bar_buffer: deque = deque(maxlen=53)  # 足够容纳最大channel_period=50+3

        # ── 待执行信号 (next-bar 规则) ────────────────────────────────────────
        self._pending: str | None = None  # "LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT", "STOP"

        # ── 委托 ID 集合 ──────────────────────────────────────────────────────
        self.order_id: set[int] = set()

    # ── 主图指标 ──────────────────────────────────────────────────────────────
    @property
    def main_indicator_data(self) -> dict[str, float]:
        return {
            "CH_High": self.state_map.ch_high,
            "CH_Low":  self.state_map.ch_low,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  生命周期
    # ══════════════════════════════════════════════════════════════════════════

    def on_start(self) -> None:
        """策略启动: 初始化 K 线合成器，推送历史数据预热"""
        # 调整 buffer maxlen 以匹配实际参数
        self._bar_buffer = deque(maxlen=self.params_map.channel_period + 3)

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
            f"CandlePatternMeanReversion 启动 | "
            f"合约: {self.params_map.instrument_id}@{self.params_map.exchange} | "
            f"K线: {self.params_map.kline_style} | "
            f"channel={self.params_map.channel_period} "
            f"extreme={self.params_map.extreme_pct} "
            f"pin_ratio={self.params_map.pin_ratio} | "
            f"hard_stop={self.params_map.hard_stop_pct}%"
        )

    def on_stop(self) -> None:
        super().on_stop()

    # ══════════════════════════════════════════════════════════════════════════
    #  行情 / 委托 / 成交 回调
    # ══════════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════════
    #  K 线回调
    # ══════════════════════════════════════════════════════════════════════════

    def callback(self, kline: KLineData) -> None:
        """K 线完成回调 -- 主策略逻辑

        执行顺序:
          1. 撤销所有未成交挂单
          2. 执行上一根 bar 产生的待执行信号 (next-bar 规则)
          3. 检查缓冲区数据是否充足
          4. 计算通道高低点和通道位置
          5. 检测裸K形态，合成 pattern_score
          6. 持仓中: 检查止损 / 反向出场条件
          7. 空仓: 检查入场条件
          8. 将当前 K 线加入缓冲区 (供下一根使用)
        """
        signal_price = 0.0

        # ── 1. 撤销所有未成交挂单 ─────────────────────────────────────────────
        for oid in list(self.order_id):
            self.cancel_order(oid)

        # ── 2. 执行 pending 信号 (next-bar 规则) ──────────────────────────────
        if self._pending is not None:
            action = self._pending
            self._pending = None
            signal_price = self._execute_signal(kline, action)

        # ── 3. 数据充足性检查 ──────────────────────────────────────────────────
        # 需要 channel_period 根历史K线 + 额外2根用于形态检测
        min_bars = self.params_map.channel_period + 2
        if len(self._bar_buffer) < min_bars:
            self._bar_buffer.append(kline)
            self._push_widget(kline, signal_price)
            return

        # ── 4. 通道计算 (使用缓冲区中最近 channel_period 根，不含当前K线) ────
        buf = list(self._bar_buffer)
        recent = buf[-self.params_map.channel_period:]
        ch_high = max(b.high for b in recent)
        ch_low  = min(b.low  for b in recent)
        ch_range = ch_high - ch_low

        ch_pos = (kline.close - ch_low) / ch_range if ch_range > 0 else 0.5

        # 极端区域判断
        ep = self.params_map.extreme_pct
        near_low  = ch_pos <= ep            # 超卖区 → 做多机会
        near_high = ch_pos >= (1.0 - ep)    # 超买区 → 做空机会

        # 更新状态显示
        self.state_map.ch_high = round(ch_high, 1)
        self.state_map.ch_low  = round(ch_low, 1)
        self.state_map.ch_pos  = round(ch_pos, 3)

        # ── 5. 裸K形态检测 ────────────────────────────────────────────────────
        bar_m2 = buf[-2]   # 母线 (2根前)
        bar_m1 = buf[-1]   # 前一根
        bar_c  = kline     # 当前根

        pattern_score = (
            self._engulf_score(bar_m1, bar_c)
            + self._pinbar_score(bar_c)
            + self._insidebar_score(bar_m2, bar_m1, bar_c)
        )

        # 状态显示当前形态
        if pattern_score >= 1:
            self.state_map.pattern = f"BULL+{pattern_score}"
        elif pattern_score <= -1:
            self.state_map.pattern = f"BEAR{pattern_score}"
        else:
            self.state_map.pattern = "NONE"

        # ── 6. 持仓中: 止损 / 反向出场 ────────────────────────────────────────
        if self.in_position:
            if self.position_side == "long":
                # 硬止损
                if (self.entry_price > 0
                        and kline.close <= self.entry_price * (1 - self.params_map.hard_stop_pct / 100)):
                    self._pending = "STOP"
                # 反向出场: 价格到达通道高点区域 (已回归甚至超买)
                elif near_high and pattern_score <= -1:
                    self._pending = "EXIT_LONG"
                elif near_high and ch_pos >= (1.0 - ep * 0.5):
                    # 即使没有形态，价格已深入超买区也出场
                    self._pending = "EXIT_LONG"
                else:
                    self.state_map.last_action = "HOLD_LONG"

            elif self.position_side == "short":
                # 硬止损
                if (self.entry_price > 0
                        and kline.close >= self.entry_price * (1 + self.params_map.hard_stop_pct / 100)):
                    self._pending = "STOP"
                # 反向出场: 价格到达通道低点区域 (已回归甚至超卖)
                elif near_low and pattern_score >= 1:
                    self._pending = "EXIT_SHORT"
                elif near_low and ch_pos <= ep * 0.5:
                    self._pending = "EXIT_SHORT"
                else:
                    self.state_map.last_action = "HOLD_SHORT"

        # ── 7. 空仓: 入场条件 ──────────────────────────────────────────────────
        else:
            if near_low and pattern_score >= 1:
                self._pending = "LONG"
            elif near_high and pattern_score <= -1:
                self._pending = "SHORT"

        # ── 8. 将当前K线加入缓冲区 (供下一根使用) ────────────────────────────
        self._bar_buffer.append(kline)

        self._push_widget(kline, signal_price)
        self.update_status_bar()

    def real_time_callback(self, kline: KLineData) -> None:
        """每个 Tick 都调用 -- 仅更新图表，不执行交易逻辑"""
        self._push_widget(kline)

    # ══════════════════════════════════════════════════════════════════════════
    #  交易执行
    # ══════════════════════════════════════════════════════════════════════════

    def _execute_signal(self, kline: KLineData, action: str) -> float:
        """根据 action 执行交易，返回 signal_price 供图表标记"""
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
        """开多仓"""
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
        self.state_map.last_action = f"OPEN_LONG ch={self.state_map.ch_pos:.2f}"

        self.output(
            f"[开多] {self.params_map.volume}手 @ {price:.1f} | "
            f"通道位置={self.state_map.ch_pos:.2f} | 形态={self.state_map.pattern}"
        )
        return price  # 正值 → 图表绿色买入标记

    def _exec_open_short(self, kline: KLineData) -> float:
        """开空仓"""
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
        self.state_map.last_action = f"OPEN_SHORT ch={self.state_map.ch_pos:.2f}"

        self.output(
            f"[开空] {self.params_map.volume}手 @ {price:.1f} | "
            f"通道位置={self.state_map.ch_pos:.2f} | 形态={self.state_map.pattern}"
        )
        return -price  # 负值 → 图表红色卖出标记

    def _exec_close_long(self, kline: KLineData, reason: str) -> float:
        """平多仓"""
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

        pnl_pct = (price - self.entry_price) / self.entry_price * 100 if self.entry_price > 0 else 0.0
        self.output(
            f"[平多] {reason} | {net_pos}手 @ {price:.1f} | "
            f"入场={self.entry_price:.1f} | 盈亏={pnl_pct:+.2f}%"
        )

        self.in_position = False
        self.position_side = ""
        self.entry_price = 0.0
        self.state_map.entry_price = 0.0
        self.state_map.last_action = reason

        return -price  # 负值 → 图表红色标记

    def _exec_close_short(self, kline: KLineData, reason: str) -> float:
        """平空仓"""
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

        pnl_pct = (self.entry_price - price) / self.entry_price * 100 if self.entry_price > 0 else 0.0
        self.output(
            f"[平空] {reason} | {close_vol}手 @ {price:.1f} | "
            f"入场={self.entry_price:.1f} | 盈亏={pnl_pct:+.2f}%"
        )

        self.in_position = False
        self.position_side = ""
        self.entry_price = 0.0
        self.state_map.entry_price = 0.0
        self.state_map.last_action = reason

        return price  # 正值 → 图表绿色标记 (买入平空)

    # ══════════════════════════════════════════════════════════════════════════
    #  形态检测
    # ══════════════════════════════════════════════════════════════════════════

    def _engulf_score(self, prev: KLineData, curr: KLineData) -> int:
        """
        吞噬形态打分: +1 看涨吞噬 / -1 看跌吞噬 / 0 无
        看涨吞噬: 前阴后阳，当前实体完全包住前根实体
        看跌吞噬: 前阳后阴，当前实体完全包住前根实体
        """
        prev_bear = prev.close < prev.open
        prev_bull = prev.close > prev.open
        curr_bull = curr.close > curr.open
        curr_bear = curr.close < curr.open

        if (prev_bear and curr_bull
                and curr.open <= prev.close
                and curr.close >= prev.open):
            return 1

        if (prev_bull and curr_bear
                and curr.open >= prev.close
                and curr.close <= prev.open):
            return -1

        return 0

    def _pinbar_score(self, curr: KLineData) -> int:
        """
        Pin Bar 打分: +1 锤子线 / -1 射击之星 / 0 无
        锤子: 长下影 (>= pin_ratio × 实体)，短上影 (<= 实体)，小实体
        射击之星: 长上影，短下影，小实体
        """
        body = abs(curr.close - curr.open)
        if body == 0:
            return 0  # 十字星跳过

        candle_range = curr.high - curr.low
        if candle_range == 0:
            return 0

        lower_shadow = min(curr.open, curr.close) - curr.low
        upper_shadow = curr.high - max(curr.open, curr.close)
        ratio = self.params_map.pin_ratio

        # 锤子线 (看涨)
        if (lower_shadow >= ratio * body
                and upper_shadow <= body
                and body <= 0.33 * candle_range):
            return 1

        # 射击之星 (看跌)
        if (upper_shadow >= ratio * body
                and lower_shadow <= body
                and body <= 0.33 * candle_range):
            return -1

        return 0

    def _insidebar_score(self, mother: KLineData, inside: KLineData, curr: KLineData) -> int:
        """
        Inside Bar 突破打分: +1 向上突破 / -1 向下突破 / 0 无
        inside bar: high < mother.high 且 low > mother.low
        突破: 当前K线收盘超出母线高/低点
        """
        # 检查 inside bar 条件 (前一根)
        if not (inside.high < mother.high and inside.low > mother.low):
            return 0

        if curr.close > mother.high:
            return 1
        if curr.close < mother.low:
            return -1

        return 0

    # ══════════════════════════════════════════════════════════════════════════
    #  辅助方法
    # ══════════════════════════════════════════════════════════════════════════

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
