# PythonGo Strategy Development

This project is a **strategy library** for the 无限易 PythonGo platform. We do NOT run code here — only develop, organize, and version strategies.

- Simon provides ideas → I convert them to strategy code following the template below
- Strategies categorized into subfolders (expandable as needed)
- Reference files: `Test_DualMA.py` and `Test_OpenOnly.py` are proven working templates

## Platform

- **Framework**: 无限易 PythonGo (`BaseStrategy`)
- **Asset**: 铁矿石 (DCE) — but strategies should be contract-agnostic via `Params`
- **Direction**: Long Only (做多)

## Project Structure

```
pythongo-strategy/
├── CLAUDE.md
├── RGL_DualMA_PythonGo.py    # Reference: full strategy (minute K-line)
├── Test_DualMA.py             # Reference: working strategy (second K-line)
├── Test_OpenOnly.py           # Reference: minimal order test
└── strategies/
    ├── ma/                    # 均线类策略
    ├── bollinger/             # 布林带类策略
    ├── volume/                # 成交量类策略
    ├── momentum/              # 动量类策略
    └── composite/             # 复合策略 (多指标组合)
```

New category folders can be added as needed.

## Strategy Template Rules

Every strategy MUST follow these patterns (extracted from the two working Test files).

### Class Structure

```python
from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import KLineData, OrderData, TickData, TradeData
from pythongo.ui import BaseStrategy

class Params(BaseParams):
    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="i2605", title="合约代码")
    # ... strategy-specific params ...

class State(BaseState):
    # ... display fields for 无限易 status bar ...

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.params_map = Params()
        self.state_map = State()
```

### K-Line Generators

- **秒级 K 线** (no history, must warm up in real-time):
  ```python
  from pythongo.utils import KLineGeneratorSec
  self.kline_generator = KLineGeneratorSec(callback=self.callback, seconds=10)
  ```
- **分钟级 K 线** (has history, call `push_history_data()` before `super().on_start()`):
  ```python
  from pythongo.utils import KLineGenerator
  self.kline_generator = KLineGenerator(
      callback=self.callback,
      real_time_callback=self.real_time_callback,
      exchange=..., instrument_id=..., style=...
  )
  self.kline_generator.push_history_data()
  super().on_start()
  ```

### Order Execution (PROVEN WORKING)

**Open position (开仓/建仓):**
```python
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
```

**Add position (加仓):**
Same as open — use `send_order` with `order_direction="buy"`. Update `avg_price` manually:
```python
self.avg_price = (self.avg_price * actual_pos + price * vol) / (actual_pos + vol)
```

**Reduce position (减仓/止盈1平半仓):**
```python
tp_vol = max(1, actual_pos // 2)
oid = self.auto_close_position(
    exchange=self.params_map.exchange,
    instrument_id=self.params_map.instrument_id,
    volume=tp_vol,
    price=price,
    order_direction="sell",
    market=True,
)
```

**Close all (平仓/止损/止盈2/趋势出场):**
```python
actual_pos = self.get_position(self.params_map.instrument_id).net_position
if actual_pos > 0:
    oid = self.auto_close_position(
        exchange=self.params_map.exchange,
        instrument_id=self.params_map.instrument_id,
        volume=actual_pos,
        price=price,
        order_direction="sell",
        market=True,
    )
```

### Key Rules

- **开仓** → `send_order(order_direction="buy", market=True)`
- **平仓/减仓** → `auto_close_position(order_direction="sell", market=True)`
- **获取持仓** → `self.get_position(instrument_id).net_position`
- **price 参数**: 传 `kline.close` 用于显示，实际以市价成交

### Next-Bar Rule (必须遵守)

信号在当前 bar 产生，存入 `self._pending`，下一根 bar 开头执行：
```python
# 当前 bar: 产生信号
self._pending = "OPEN"  # or "CLOSE", "STOP", "TP1", "TP2", "ADD"

# 下一根 bar callback 开头: 执行信号
if self._pending:
    signal_price = self._execute(kline, self._pending)
    self._pending = None
```

### Mandatory Risk Management

Every strategy MUST include:

1. **硬止损 (Hard Stop)** — default `0.3%`, exit when:
   ```python
   close <= self.avg_price * (1 - stop_loss_pct / 100)
   ```

2. **移动止损 (Trailing Stop)** — default `0.5%`, track peak price:
   ```python
   if close > self.peak_price:
       self.peak_price = close
   if close <= self.peak_price * (1 - trailing_pct / 100):
       self._pending = "TRAIL_STOP"
   ```

3. **Signal priority**: 止损 > 止盈 > 出场 > 加仓 > 建仓

### Chart Widget

```python
def _push_widget(self, kline: KLineData, signal_price: float = 0.0) -> None:
    self.widget.recv_kline({
        "kline": kline,
        "signal_price": signal_price,
        **self.main_indicator_data,
    })
```
- `signal_price > 0` → green buy marker
- `signal_price < 0` → red sell marker

### Cancel Pending Orders

Always cancel unfilled orders at the start of each bar callback:
```python
for oid in list(self.order_id):
    self.cancel_order(oid)
```
