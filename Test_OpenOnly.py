"""
Test_OpenOnly — 最简开仓测试
参考官方文档 DemoTest 示例，直接在 on_start 里下单，不依赖 tick。
"""

from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import OrderData, TradeData
from pythongo.ui import BaseStrategy


class Params(BaseParams):
    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="i2605", title="合约代码")
    order_price: float = Field(default=0.0, title="下单价格(0=市价)")
    order_volume: int = Field(default=1, title="下单手数", ge=1)
    order_direction: str = Field(default="buy", title="方向(buy/sell)")


class State(BaseState):
    order_id: str = Field(default="未下单", title="委托ID")


class Test_OpenOnly(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.params_map = Params()
        self.state_map = State()

    @property
    def main_indicator_data(self) -> dict[str, float]:
        return {}

    def on_start(self):
        super().on_start()

        price = self.params_map.order_price
        oid = self.send_order(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            volume=self.params_map.order_volume,
            price=price,
            order_direction=self.params_map.order_direction,
            market=(price == 0.0),  # 价格为0则市价单
        )

        self.state_map.order_id = str(oid) if oid is not None else "失败(None)"
        self.update_status_bar()
        self.output(f"[下单] {self.params_map.order_direction} {self.params_map.instrument_id} "
                    f"price={price} order_id={oid}")

    def on_stop(self):
        super().on_stop()

    def on_order(self, order: OrderData):
        self.output(f"[委托] status={order.status} price={order.price} "
                    f"vol={order.total_volume} traded={order.traded_volume}")

    def on_trade(self, trade: TradeData, log: bool = False):
        self.output(f"[成交] price={trade.price} vol={trade.volume}")
        self.state_map.order_id = f"成交@{trade.price}"
        self.update_status_bar()
