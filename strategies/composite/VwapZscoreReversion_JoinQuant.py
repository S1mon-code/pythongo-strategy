"""
================================================================================
  VWAP Z-Score 均值回归策略 — 铁矿石 CTA
================================================================================

  适用平台   : 聚宽 JoinQuant
  合约       : 铁矿石主力 (I8888.XDCE)，自动滚动
  K线周期    : 5 分钟
  方向       : 多空双向 (Long + Short)
  版本       : 1.0 (JoinQuant 移植版)

================================================================================
  回测结果 (来自 PythonGo 版，参数相同)
================================================================================

  Sharpe: 0.65 | Calmar: 0.49 | Max DD: 2.12%
  Annual Return: 1.03% | Win Rate: 62.6% | PF: 1.20
  Trades/Year: ~103
  Best Params: z_threshold=3.0, min_bars=30

================================================================================
  策略逻辑
================================================================================

  - 计算日内 VWAP (日盘 09:00 / 夜盘 21:00 重置)
  - Z-Score = (close - vwap) / rolling_std(deviation, min_bars)
  - Z < -z_threshold → 开多 (价格远低于 VWAP，预期均值回归)
  - Z > +z_threshold → 开空 (价格远高于 VWAP，预期均值回归)
  - 出场: Z 穿越 0 (回归至 VWAP) 或 硬止损 5%
  - 收盘: 14:55 / 22:55 强制平仓，不持仓过夜

================================================================================
"""

from jqdata import *
import numpy as np


# ==============================================================================
#  策略参数
# ==============================================================================

Z_THRESHOLD   = 3.0    # Z-Score 入场阈值
MIN_BARS      = 30     # 计算 Z-Score 所需最少 bar 数
HARD_STOP_PCT = 0.05   # 硬止损比例 5%
VOLUME        = 1      # 每次下单手数
UNDERLYING    = 'I'    # 标的代码 (铁矿石)


# ==============================================================================
#  初始化
# ==============================================================================

def initialize(context):
    """平台启动时执行一次 — 设置账户、费率、滑点、调度"""

    set_benchmark('I8888.XDCE')
    set_option('use_real_price', True)

    set_subportfolios([
        SubPortfolioConfig(
            cash=context.portfolio.starting_cash,
            type='futures'
        )
    ])

    set_order_cost(
        OrderCost(
            open_commission=0.000023,
            close_commission=0.000023,
            close_today_commission=0.0023
        ),
        type='futures'
    )

    set_option('futures_margin_rate', 0.10)
    set_slippage(StepRelatedSlippage(2))

    # ── VWAP 累积量 (每个交易时段重置) ────────────────────────────────────
    g.cum_tp_vol  = 0.0     # cumulative(typical_price × volume)
    g.cum_vol     = 0.0     # cumulative(volume)
    g.deviations  = []      # 偏差序列，用于 rolling std
    g.bars_today  = 0       # 当前时段 bar 数

    # ── 持仓状态 ──────────────────────────────────────────────────────────
    g.in_position    = False
    g.position_side  = ''    # 'long' 或 'short'
    g.entry_price    = 0.0

    # ── 调度 ──────────────────────────────────────────────────────────────
    run_daily(strategy_logic, time='every_bar', reference_security='I8888.XDCE')
    run_daily(close_all, time='14:55', reference_security='I8888.XDCE')
    run_daily(close_all, time='22:55', reference_security='I8888.XDCE')


# ==============================================================================
#  主策略逻辑 (每 5 分钟 K 线调用)
# ==============================================================================

def strategy_logic(context):
    """
    每 5 分钟执行：
      1. 日盘/夜盘开盘时重置 VWAP 累积量
      2. 用已完成的最新 1 根 5-min bar 更新 VWAP 和 Z-Score
      3. 检查硬止损
      4. 持仓中: Z 穿越 0 时平仓
      5. 空仓: Z < -threshold 开多 / Z > +threshold 开空
    """

    contract = get_dominant_future(UNDERLYING)
    h = context.current_dt.hour
    m = context.current_dt.minute

    # ── 1. 交易时段开盘重置 ──────────────────────────────────────────────
    # 日盘第一根 5min bar 收盘于 09:05；夜盘第一根收盘于 21:05
    if (h == 9 and m == 5) or (h == 21 and m == 5):
        g.cum_tp_vol = 0.0
        g.cum_vol    = 0.0
        g.deviations = []
        g.bars_today = 0

    # ── 2. 获取最新已完成 bar 并更新 VWAP ───────────────────────────────
    bars = get_bars(
        security=contract,
        count=1,
        unit='5m',
        fields=['high', 'low', 'close', 'volume'],
        include_now=False
    )
    if bars is None or len(bars) == 0:
        return

    high    = float(bars['high'][-1])
    low     = float(bars['low'][-1])
    close   = float(bars['close'][-1])
    volume  = max(float(bars['volume'][-1]), 1)

    typical_price = (high + low + close) / 3.0
    g.cum_tp_vol += typical_price * volume
    g.cum_vol    += volume
    g.bars_today += 1

    vwap      = g.cum_tp_vol / g.cum_vol
    deviation = close - vwap
    g.deviations.append(deviation)

    # 滚动窗口裁剪（保留最近 MIN_BARS × 2 个，防止列表无限增长）
    if len(g.deviations) > MIN_BARS * 2:
        g.deviations = g.deviations[-MIN_BARS * 2:]

    # ── 3. 数据不足，跳过交易逻辑 ────────────────────────────────────────
    if g.bars_today < MIN_BARS or len(g.deviations) < MIN_BARS:
        return

    rolling_std = float(np.std(g.deviations[-MIN_BARS:], ddof=0))
    if rolling_std <= 0:
        return

    z_score = deviation / rolling_std

    # ── 4. 查询持仓 ──────────────────────────────────────────────────────
    sub       = context.subportfolios[0]
    has_long  = _has_position(sub, contract, 'long')
    has_short = _has_position(sub, contract, 'short')

    # ── 5. 硬止损检查 ────────────────────────────────────────────────────
    if has_long and g.entry_price > 0:
        if close <= g.entry_price * (1 - HARD_STOP_PCT):
            order_target(contract, 0, side='long')
            log.info(f'[硬止损-多] 现价={close:.1f} 入场={g.entry_price:.1f}')
            g.in_position = False
            g.position_side = ''
            g.entry_price = 0.0
            return

    if has_short and g.entry_price > 0:
        if close >= g.entry_price * (1 + HARD_STOP_PCT):
            order_target(contract, 0, side='short')
            log.info(f'[硬止损-空] 现价={close:.1f} 入场={g.entry_price:.1f}')
            g.in_position = False
            g.position_side = ''
            g.entry_price = 0.0
            return

    # ── 6. 出场: Z 穿越 0 (价格回归 VWAP) ───────────────────────────────
    if has_long and z_score > 0:
        order_target(contract, 0, side='long')
        log.info(f'[平多-回归] Z={z_score:.2f} VWAP={vwap:.1f}')
        g.in_position = False
        g.position_side = ''
        g.entry_price = 0.0
        return

    if has_short and z_score < 0:
        order_target(contract, 0, side='short')
        log.info(f'[平空-回归] Z={z_score:.2f} VWAP={vwap:.1f}')
        g.in_position = False
        g.position_side = ''
        g.entry_price = 0.0
        return

    # ── 7. 空仓入场 ──────────────────────────────────────────────────────
    if not has_long and not has_short:
        if z_score < -Z_THRESHOLD:
            order(contract, VOLUME, side='long')
            g.entry_price = close
            g.in_position = True
            g.position_side = 'long'
            log.info(f'[开多] @ {close:.1f} | Z={z_score:.2f} | VWAP={vwap:.1f}')

        elif z_score > Z_THRESHOLD:
            order(contract, VOLUME, side='short')
            g.entry_price = close
            g.in_position = True
            g.position_side = 'short'
            log.info(f'[开空] @ {close:.1f} | Z={z_score:.2f} | VWAP={vwap:.1f}')


# ==============================================================================
#  收盘强制平仓
# ==============================================================================

def close_all(context):
    """14:55 / 22:55 强制平所有仓位，不持仓过夜"""
    contract = get_dominant_future(UNDERLYING)
    sub      = context.subportfolios[0]

    if _has_position(sub, contract, 'long'):
        order_target(contract, 0, side='long')
        log.info(f'[收盘平多] {context.current_dt.strftime("%H:%M")}')

    if _has_position(sub, contract, 'short'):
        order_target(contract, 0, side='short')
        log.info(f'[收盘平空] {context.current_dt.strftime("%H:%M")}')

    g.in_position   = False
    g.position_side = ''
    g.entry_price   = 0.0
    # 重置 VWAP 状态，次日重新预热
    g.cum_tp_vol = 0.0
    g.cum_vol    = 0.0
    g.deviations = []
    g.bars_today = 0


# ==============================================================================
#  辅助函数
# ==============================================================================

def _has_position(sub, contract, side):
    """判断指定合约是否有 side 方向的持仓"""
    positions = sub.long_positions if side == 'long' else sub.short_positions
    return contract in positions and positions[contract].total_amount > 0
