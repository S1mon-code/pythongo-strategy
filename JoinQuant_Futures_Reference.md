# 聚宽 JoinQuant — 期货策略基础语法速查

---

## 一、程序结构（固定函数名，平台自动调用）

```python
def initialize(context):       # 启动时运行一次，做所有初始化
def before_market_open(context): # 每日盘前运行
def market_open(context):        # 每日盘中运行
def after_market_close(context): # 每日收盘后运行
```

---

## 二、initialize — 初始化配置

```python
def initialize(context):

    # ── 基准 ──────────────────────────────────────────────────────────
    set_benchmark('000300.XSHG')           # 设置业绩基准（沪深300）

    # ── 价格模式 ───────────────────────────────────────────────────────
    set_option('use_real_price', True)     # 使用真实价格（动态复权）

    # ── 账户类型（期货必须设置子账户）────────────────────────────────────
    set_subportfolios([
        SubPortfolioConfig(
            cash=context.portfolio.starting_cash,
            type='index_futures'           # 股指期货账户类型
        )
    ])

    # ── 手续费（期货）──────────────────────────────────────────────────
    set_order_cost(
        OrderCost(
            open_commission=0.000023,      # 开仓手续费（万分之0.23）
            close_commission=0.000023,     # 平仓手续费
            close_today_commission=0.0023  # 平今仓手续费（万分之23，更高）
        ),
        type='index_futures'
    )

    # ── 保证金比例 ─────────────────────────────────────────────────────
    set_option('futures_margin_rate', 0.15)   # 15% 保证金

    # ── 滑点 ───────────────────────────────────────────────────────────
    set_slippage(StepRelatedSlippage(2))      # 2档滑点

    # ── 调度（有夜盘品种必须用绝对时间，不能用 open/close）─────────────────
    run_daily(before_market_open, time='09:00', reference_security='IF8888.CCFX')
    run_daily(market_open,        time='09:30', reference_security='IF8888.CCFX')
    run_daily(after_market_close, time='15:30', reference_security='IF8888.CCFX')
```

---

## 三、全局变量 g（跨函数传递数据）

```python
# 在 before_market_open 中赋值
g.contract_A = get_future_contracts('IF')[0]   # 当月合约
g.contract_B = get_future_contracts('IF')[2]   # 下季合约

# 在 market_open 中读取
contract = g.contract_A
```

---

## 四、获取期货合约

```python
contracts = get_future_contracts('IF')   # 返回合约列表，[0]=当月，[1]=下月，[2]=下季
dominant   = get_dominant_future('IF')   # 获取主力合约代码（字符串）

# 获取合约到期日
end_date = get_security_info('IF2505.CCFX').end_date
```

---

## 五、获取历史行情数据

```python
# 获取最近 N 根 K 线
bars = get_bars(
    security='IF2505.CCFX',
    count=20,
    unit='1d',           # '1d' 日线，'1m' 分钟线，'5m' 5分钟线
    fields=['open', 'high', 'low', 'close', 'volume']
)
close_price = bars['close'][-1]   # 最新收盘价
```

---

## 六、获取实时行情

```python
cur = get_current_data()
last_price   = cur['IF2505.CCFX'].last_price    # 最新价
high_limit   = cur['IF2505.CCFX'].high_limit    # 涨停价
low_limit    = cur['IF2505.CCFX'].low_limit     # 跌停价
```

---

## 七、下单函数

```python
# 买入 / 做多 N 手
order('IF2505.CCFX', 1, side='long')

# 卖出 / 做空 N 手
order('IF2505.CCFX', 1, side='short')

# 调仓到目标手数（0 = 平仓）
order_target('IF2505.CCFX', 0, side='long')    # 平多仓
order_target('IF2505.CCFX', 0, side='short')   # 平空仓
order_target('IF2505.CCFX', 2, side='long')    # 持多 2 手
```

---

## 八、持仓查询

```python
subportfolio = context.subportfolios[0]   # 第 0 个子账户

# 多头持仓（dict: symbol -> position）
long_positions  = subportfolio.long_positions
short_positions = subportfolio.short_positions

# 检查是否空仓
is_flat = (len(long_positions) == 0) and (len(short_positions) == 0)

# 获取某合约持仓
p = long_positions.get('IF2505.CCFX')
if p:
    qty  = p.total_amount    # 持仓手数
    side = p.side            # 'long' 或 'short'
```

---

## 九、context 对象常用属性

```python
context.current_dt          # 当前时间 (datetime)
context.current_dt.date()   # 当前日期
context.current_dt.time()   # 当前时间

context.portfolio.starting_cash    # 初始资金
context.portfolio.total_value      # 当前总资产
context.portfolio.cash             # 可用现金

context.subportfolios[0]           # 第 0 个子账户
```

---

## 十、日志输出

```python
log.info('普通信息')
log.warning('警告信息')
log.error('错误信息')
```

---

## 十一、完整策略骨架（期货跨期套利示例）

```python
from jqdata import *

def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_subportfolios([SubPortfolioConfig(
        cash=context.portfolio.starting_cash, type='index_futures'
    )])
    set_order_cost(OrderCost(
        open_commission=0.000023,
        close_commission=0.000023,
        close_today_commission=0.0023
    ), type='index_futures')
    set_option('futures_margin_rate', 0.15)
    set_slippage(StepRelatedSlippage(2))

    run_daily(before_market_open, time='09:00', reference_security='IF8888.CCFX')
    run_daily(market_open,        time='09:30', reference_security='IF8888.CCFX')
    run_daily(after_market_close, time='15:30', reference_security='IF8888.CCFX')


def before_market_open(context):
    g.near  = get_future_contracts('IF')[0]   # 当月
    g.far   = get_future_contracts('IF')[2]   # 下季


def market_open(context):
    near_close = get_bars(g.near, count=1, unit='1d', fields=['close'])['close'][-1]
    far_close  = get_bars(g.far,  count=1, unit='1d', fields=['close'])['close'][-1]
    spread = near_close - far_close

    sub = context.subportfolios[0]
    no_pos = (len(sub.long_positions) == 0) and (len(sub.short_positions) == 0)
    has_pos = not no_pos

    end_date = get_security_info(g.near).end_date

    # 开仓：价差 > 80，且不是交割日，且当前空仓
    if spread > 80 and context.current_dt.date() != end_date and no_pos:
        order(g.near, 1, side='short')   # 做空近月
        order(g.far,  1, side='long')    # 做多远月

    # 平仓：价差收敛至 70 以内
    if spread < 70 and has_pos:
        order_target(g.near, 0, side='short')
        order_target(g.far,  0, side='long')


def after_market_close(context):
    trades = get_trades()
    for t in trades.values():
        log.info('成交：' + str(t))
```

---

## 十二、注意事项

| 要点 | 说明 |
|------|------|
| 有夜盘品种 | `run_daily` 必须用绝对时间（如 `'09:00'`），不能用 `'open'`/`'close'` |
| 平今仓手续费 | `close_today_commission` 比普通平仓高 100 倍，当天开当天平成本极高 |
| 涨跌停检查 | 开仓前必须检查 `high_limit`/`low_limit`，否则订单可能失败 |
| `order_target` 平仓 | 传 `0` 即为完全平仓，比手动计算持仓数量更安全 |
| 全局变量 `g` | 跨函数共享数据的标准方式，不要用模块级全局变量 |
| `context.subportfolios[0]` | 期货账户持仓在子账户里，不在 `context.portfolio` |
