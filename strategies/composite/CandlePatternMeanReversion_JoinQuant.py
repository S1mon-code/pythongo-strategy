"""
================================================================================
  Candle Pattern Mean-Reversion 铁矿石 CTA 策略 — 裸K多形态均值回归
================================================================================

  适用平台   : 聚宽 JoinQuant
  合约       : 铁矿石主力 (I8888.XDCE)，自动滚动
  K线周期    : 5 分钟
  方向       : 多空双向 (Long + Short)
  版本       : 1.0 (JoinQuant 移植版)

================================================================================
  回测结果 (来自 PythonGo 版，参数相同)
================================================================================

  Sharpe: 1.02 | Max DD: 1.98% | Annual Return: 1.64%
  Win Rate: 58.3% | PF: 1.36 | Trades/Year: ~83
  Best Params: channel_period=10, extreme_pct=0.15, pin_ratio=3.0

================================================================================
  策略逻辑
================================================================================

  极端位置检测 (纯价格通道):
  - 前 channel_period 根K线的最高/最低点构成通道
  - channel_position = (close - ch_low) / (ch_high - ch_low)
  - >= 1 - extreme_pct → 超买区 (做空机会)
  - <= extreme_pct     → 超卖区 (做多机会)

  三种反转形态:
  1. 吞噬 (Engulfing): 当前实体完全包住前一根实体
  2. Pin Bar: 长影线/小实体 (锤子线 / 射击之星)
  3. Inside Bar 突破: 前两根压缩，当前根突破母线方向

  入场: 超买/超卖区域 + 任意 1 个反转形态触发
  出场: 反向极端区域出现 或 硬止损 5%
  收盘: 14:55 / 22:55 强制平仓，不持仓过夜

  每次调用读取已完成的 channel_period+3 根 5-min K 线:
  - bars[-channel_period-1:-1] → 通道计算 (不含当前根)
  - bars[-3], bars[-2] → 形态检测的母线和前根
  - bars[-1] → 当前根 (形态检测 + 极端位判断)

================================================================================
"""

from jqdata import *
import numpy as np


# ==============================================================================
#  策略参数
# ==============================================================================

CHANNEL_PERIOD = 10     # 价格通道周期 (根)
EXTREME_PCT    = 0.15   # 极端区间比例
PIN_RATIO      = 3.0    # Pin Bar 影线/实体比
HARD_STOP_PCT  = 0.05   # 硬止损比例 5%
VOLUME         = 1      # 每次下单手数
UNDERLYING     = 'I'    # 标的代码 (铁矿石)


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

    # ── 持仓状态 ──────────────────────────────────────────────────────────
    g.in_position   = False
    g.position_side = ''    # 'long' 或 'short'
    g.entry_price   = 0.0

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
      1. 读取已完成的 channel_period+3 根 5-min K 线
      2. 计算通道高低点和通道位置
      3. 检测三种裸K反转形态，合成 pattern_score
      4. 检查硬止损
      5. 持仓中: 反向极端区域出场
      6. 空仓: 极端区域 + 形态触发入场
    """

    contract = get_dominant_future(UNDERLYING)

    # ── 1. 获取历史 K 线 ────────────────────────────────────────────────
    count = CHANNEL_PERIOD + 3   # 通道+母线+前根+当前根
    bars = get_bars(
        security=contract,
        count=count,
        unit='5m',
        fields=['open', 'high', 'low', 'close'],
        include_now=False
    )
    if bars is None or len(bars) < count:
        return

    # 当前根 (bar_c)
    c_o = float(bars['open'][-1]);  c_h = float(bars['high'][-1])
    c_l = float(bars['low'][-1]);   c_c = float(bars['close'][-1])

    # 前一根 (bar_m1)
    m1_o = float(bars['open'][-2]); m1_h = float(bars['high'][-2])
    m1_l = float(bars['low'][-2]);  m1_c = float(bars['close'][-2])

    # 母线 (bar_m2，用于 inside bar)
    m2_h = float(bars['high'][-3]);  m2_l = float(bars['low'][-3])

    # ── 2. 通道计算 (最近 channel_period 根，不含当前根) ────────────────
    ch_bars_h = bars['high'][-(CHANNEL_PERIOD + 1):-1]
    ch_bars_l = bars['low'][-(CHANNEL_PERIOD + 1):-1]
    ch_high   = float(np.max(ch_bars_h))
    ch_low    = float(np.min(ch_bars_l))
    ch_range  = ch_high - ch_low

    if ch_range <= 0:
        return

    ch_pos  = (c_c - ch_low) / ch_range
    near_low  = ch_pos <= EXTREME_PCT
    near_high = ch_pos >= (1.0 - EXTREME_PCT)

    # ── 3. 裸K形态检测 ─────────────────────────────────────────────────
    pattern_score = (
        _engulf_score(m1_o, m1_c, c_o, c_c)
        + _pinbar_score(c_o, c_h, c_l, c_c)
        + _insidebar_score(m2_h, m2_l, m1_h, m1_l, c_c)
    )

    # ── 4. 查询持仓 ────────────────────────────────────────────────────
    sub       = context.subportfolios[0]
    has_long  = _has_position(sub, contract, 'long')
    has_short = _has_position(sub, contract, 'short')

    # ── 5. 硬止损检查 ──────────────────────────────────────────────────
    if has_long and g.entry_price > 0:
        if c_c <= g.entry_price * (1 - HARD_STOP_PCT):
            order_target(contract, 0, side='long')
            log.info(f'[硬止损-多] 现价={c_c:.1f} 入场={g.entry_price:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0
            return

    if has_short and g.entry_price > 0:
        if c_c >= g.entry_price * (1 + HARD_STOP_PCT):
            order_target(contract, 0, side='short')
            log.info(f'[硬止损-空] 现价={c_c:.1f} 入场={g.entry_price:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0
            return

    # ── 6. 持仓中: 反向极端区域出场 ────────────────────────────────────
    if has_long:
        # 价格进入超买区 (已回归甚至超涨) → 平多
        if near_high and (pattern_score <= -1 or ch_pos >= (1.0 - EXTREME_PCT * 0.5)):
            order_target(contract, 0, side='long')
            log.info(f'[平多-反转] ch_pos={ch_pos:.2f} score={pattern_score}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0
        return

    if has_short:
        # 价格进入超卖区 (已回归甚至超跌) → 平空
        if near_low and (pattern_score >= 1 or ch_pos <= EXTREME_PCT * 0.5):
            order_target(contract, 0, side='short')
            log.info(f'[平空-反转] ch_pos={ch_pos:.2f} score={pattern_score}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0
        return

    # ── 7. 空仓入场 ────────────────────────────────────────────────────
    if near_low and pattern_score >= 1:
        order(contract, VOLUME, side='long')
        g.entry_price = c_c
        g.in_position = True
        g.position_side = 'long'
        log.info(f'[开多] @ {c_c:.1f} | ch_pos={ch_pos:.2f} | score={pattern_score}')

    elif near_high and pattern_score <= -1:
        order(contract, VOLUME, side='short')
        g.entry_price = c_c
        g.in_position = True
        g.position_side = 'short'
        log.info(f'[开空] @ {c_c:.1f} | ch_pos={ch_pos:.2f} | score={pattern_score}')


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

    g.in_position = False; g.position_side = ''; g.entry_price = 0.0


# ==============================================================================
#  形态检测函数
# ==============================================================================

def _engulf_score(prev_o, prev_c, curr_o, curr_c):
    """吞噬形态: +1 看涨吞噬 / -1 看跌吞噬 / 0 无"""
    prev_bear = prev_c < prev_o
    prev_bull = prev_c > prev_o
    curr_bull = curr_c > curr_o
    curr_bear = curr_c < curr_o

    if prev_bear and curr_bull and curr_o <= prev_c and curr_c >= prev_o:
        return 1
    if prev_bull and curr_bear and curr_o >= prev_c and curr_c <= prev_o:
        return -1
    return 0


def _pinbar_score(o, h, l, c):
    """Pin Bar: +1 锤子线 / -1 射击之星 / 0 无"""
    body = abs(c - o)
    if body == 0:
        return 0
    candle_range = h - l
    if candle_range == 0:
        return 0

    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)

    # 锤子线 (看涨): 长下影，短上影，小实体
    if (lower_shadow >= PIN_RATIO * body
            and upper_shadow <= body
            and body <= 0.33 * candle_range):
        return 1

    # 射击之星 (看跌): 长上影，短下影，小实体
    if (upper_shadow >= PIN_RATIO * body
            and lower_shadow <= body
            and body <= 0.33 * candle_range):
        return -1

    return 0


def _insidebar_score(mother_h, mother_l, inside_h, inside_l, curr_c):
    """Inside Bar 突破: +1 向上突破 / -1 向下突破 / 0 无"""
    if not (inside_h < mother_h and inside_l > mother_l):
        return 0
    if curr_c > mother_h:
        return 1
    if curr_c < mother_l:
        return -1
    return 0


# ==============================================================================
#  辅助函数
# ==============================================================================

def _has_position(sub, contract, side):
    """判断指定合约是否有 side 方向的持仓"""
    positions = sub.long_positions if side == 'long' else sub.short_positions
    return contract in positions and positions[contract].total_amount > 0
