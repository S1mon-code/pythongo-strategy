"""
================================================================================
  Micro Mean-Reversion 铁矿石 CTA 策略 — 布林带均值回归
================================================================================

  适用平台   : 聚宽 JoinQuant
  合约       : 铁矿石主力 (I8888.XDCE)，自动滚动
  K线周期    : 1 分钟
  方向       : 多空双向 (Long + Short)
  版本       : 1.0 (JoinQuant 移植版)

================================================================================
  回测结果 (来自 PythonGo 版，参数相同)
================================================================================

  Sharpe: 1.17 | Calmar: 0.90 | Max DD: 2.09%
  Annual Return: 1.89% | Win Rate: 63.8% | PF: 1.16
  Trades/Year: ~1034
  Best Params: bb_period=30, bb_std=2.0

================================================================================
  策略逻辑
================================================================================

  指标 : 布林带 BB(30, 2.0) — 基于 30 根 1-min K 线收盘价
  做多 : 前收盘 > 下轨，当前收盘 <= 下轨 (跌破下轨，均值回归买入)
  做空 : 前收盘 < 上轨，当前收盘 >= 上轨 (突破上轨，均值回归卖出)
  出场 : 反向信号出现 (直接反转) 或硬止损 5%
  收盘 : 日盘 14:55 / 夜盘 22:55 前强制平仓，不持仓过夜

  每分钟调度 strategy_logic() → 读取已完成的 30 根 1-min K 线 →
  计算 BB → 判断穿越信号 → 下单

================================================================================
"""

from jqdata import *
import numpy as np


# ==============================================================================
#  策略参数 (模块级常量，对应 PythonGo 版的 Params 类)
# ==============================================================================

BB_PERIOD        = 30       # 布林带周期 (1分钟K线数量)
BB_STD           = 2.0      # 布林带标准差倍数
HARD_STOP_PCT    = 0.05     # 硬止损比例 5%
VOLUME           = 1        # 每次下单手数
UNDERLYING       = 'I'      # 标的代码 (铁矿石)


# ==============================================================================
#  初始化
# ==============================================================================

def initialize(context):
    """平台启动时执行一次 — 设置账户、费率、滑点、调度"""

    set_benchmark('I8888.XDCE')         # 以铁矿石主力为业绩基准
    set_option('use_real_price', True)  # 使用真实价格

    # ── 账户类型：期货 ──────────────────────────────────────────────────────
    set_subportfolios([
        SubPortfolioConfig(
            cash=context.portfolio.starting_cash,
            type='futures'
        )
    ])

    # ── 手续费 (铁矿石期货) ─────────────────────────────────────────────────
    set_order_cost(
        OrderCost(
            open_commission=0.000023,       # 开仓万分之 0.23
            close_commission=0.000023,      # 平仓万分之 0.23
            close_today_commission=0.0023   # 平今仓万分之 23
        ),
        type='futures'
    )

    # ── 保证金 & 滑点 ───────────────────────────────────────────────────────
    set_option('futures_margin_rate', 0.10)  # 铁矿石保证金约 10%
    set_slippage(StepRelatedSlippage(2))     # 2 档滑点

    # ── 策略状态 (跨 bar 共享) ──────────────────────────────────────────────
    g.prev_close  = None    # 上一根 bar 的收盘价
    g.prev_upper  = None    # 上一根 bar 的布林上轨
    g.prev_lower  = None    # 上一根 bar 的布林下轨
    g.entry_price = 0.0     # 入场价格 (用于硬止损计算)

    # ── 调度 ────────────────────────────────────────────────────────────────
    # 每根 1 分钟 K 线调用一次主策略逻辑
    run_daily(strategy_logic, time='every_bar', reference_security='I8888.XDCE')

    # 日盘 & 夜盘收盘前强制平仓（不持仓过夜）
    run_daily(close_all, time='14:55', reference_security='I8888.XDCE')
    run_daily(close_all, time='22:55', reference_security='I8888.XDCE')


# ==============================================================================
#  主策略逻辑 (每 1 分钟 K 线调用)
# ==============================================================================

def strategy_logic(context):
    """
    每分钟执行：
      1. 获取主力合约
      2. 读取已完成的 30 根 1-min K 线，计算布林带
      3. 检查硬止损
      4. 判断反转 / 入场信号并下单
      5. 保存当前 bar 数据供下根 bar 判断穿越
    """

    contract = get_dominant_future(UNDERLYING)   # 主力合约

    # ── 1. 计算布林带 ─────────────────────────────────────────────────────
    bb = _calc_bb(contract)
    if bb is None:
        return                                   # 数据不足，跳过
    upper, mid, lower, current_close = bb

    # ── 2. 初始化：第一根 bar 仅记录状态 ──────────────────────────────────
    if g.prev_close is None:
        _save_prev(current_close, upper, lower)
        return

    # ── 3. 查询实际持仓 ───────────────────────────────────────────────────
    sub       = context.subportfolios[0]
    has_long  = _has_position(sub, contract, 'long')
    has_short = _has_position(sub, contract, 'short')

    # ── 4. 硬止损检查 ─────────────────────────────────────────────────────
    if has_long and g.entry_price > 0:
        if current_close <= g.entry_price * (1 - HARD_STOP_PCT):
            entry = g.entry_price
            order_target(contract, 0, side='long')
            g.entry_price = 0.0
            log.info(f'[硬止损-多] 现价={current_close:.1f} 入场={entry:.1f}')
            _save_prev(current_close, upper, lower)
            return

    if has_short and g.entry_price > 0:
        if current_close >= g.entry_price * (1 + HARD_STOP_PCT):
            entry = g.entry_price
            order_target(contract, 0, side='short')
            g.entry_price = 0.0
            log.info(f'[硬止损-空] 现价={current_close:.1f} 入场={entry:.1f}')
            _save_prev(current_close, upper, lower)
            return

    # ── 5. 信号生成 (穿越布林带) ─────────────────────────────────────────
    crossed_up   = g.prev_close < g.prev_upper and current_close >= upper  # 向上穿越上轨
    crossed_down = g.prev_close > g.prev_lower and current_close <= lower  # 向下穿越下轨

    if has_long:
        # 持多仓：上轨触发 → 平多 + 开空
        if crossed_up:
            order_target(contract, 0, side='long')
            order(contract, VOLUME, side='short')
            g.entry_price = current_close
            log.info(f'[平多→开空] @ {current_close:.1f} | BB_upper={upper:.1f}')

    elif has_short:
        # 持空仓：下轨触发 → 平空 + 开多
        if crossed_down:
            order_target(contract, 0, side='short')
            order(contract, VOLUME, side='long')
            g.entry_price = current_close
            log.info(f'[平空→开多] @ {current_close:.1f} | BB_lower={lower:.1f}')

    else:
        # 空仓：检查入场
        if crossed_down:
            order(contract, VOLUME, side='long')
            g.entry_price = current_close
            log.info(f'[开多] @ {current_close:.1f} | BB_lower={lower:.1f}')

        elif crossed_up:
            order(contract, VOLUME, side='short')
            g.entry_price = current_close
            log.info(f'[开空] @ {current_close:.1f} | BB_upper={upper:.1f}')

    # ── 6. 保存本 bar 数据 ────────────────────────────────────────────────
    _save_prev(current_close, upper, lower)


# ==============================================================================
#  收盘强制平仓
# ==============================================================================

def close_all(context):
    """日盘 14:55 / 夜盘 22:55 强制平所有仓位，不持仓过夜"""
    contract = get_dominant_future(UNDERLYING)
    sub      = context.subportfolios[0]

    if _has_position(sub, contract, 'long'):
        order_target(contract, 0, side='long')
        log.info(f'[收盘平多] {context.current_dt.strftime("%H:%M")}')

    if _has_position(sub, contract, 'short'):
        order_target(contract, 0, side='short')
        log.info(f'[收盘平空] {context.current_dt.strftime("%H:%M")}')

    # 重置状态，次日重新预热
    g.entry_price = 0.0
    g.prev_close  = None
    g.prev_upper  = None
    g.prev_lower  = None


# ==============================================================================
#  辅助函数
# ==============================================================================

def _calc_bb(contract):
    """
    读取已完成的最近 BB_PERIOD 根 1-min K 线，计算布林带。
    返回 (upper, mid, lower, current_close)，数据不足返回 None。
    """
    bars = get_bars(
        security=contract,
        count=BB_PERIOD + 1,    # 多取 1 根，保证滚动窗口刚好 BB_PERIOD 根
        unit='1m',
        fields=['close'],
        include_now=False       # 只取已完成的 K 线，防止前瞻
    )

    if bars is None or len(bars) < BB_PERIOD:
        return None

    closes        = bars['close']
    current_close = float(closes[-1])
    window        = closes[-BB_PERIOD:]

    mid   = float(np.mean(window))
    std   = float(np.std(window, ddof=0))    # 总体标准差，与 PythonGo 版一致
    upper = mid + BB_STD * std
    lower = mid - BB_STD * std

    return upper, mid, lower, current_close


def _has_position(sub, contract, side):
    """判断指定合约是否有 side 方向的持仓"""
    positions = sub.long_positions if side == 'long' else sub.short_positions
    return contract in positions and positions[contract].total_amount > 0


def _save_prev(close, upper, lower):
    """保存当前 bar 数据到全局状态，供下根 bar 判断穿越"""
    g.prev_close = close
    g.prev_upper = upper
    g.prev_lower = lower
