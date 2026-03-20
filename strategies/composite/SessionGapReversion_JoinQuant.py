"""
================================================================================
  Session-Gap Reversion 铁矿石 CTA 策略 — 日盘跳空回归
================================================================================

  适用平台   : 聚宽 JoinQuant
  合约       : 铁矿石主力 (I8888.XDCE)，自动滚动
  K线周期    : 5 分钟
  方向       : 多空双向 (Long + Short)
  版本       : 1.0 (JoinQuant 移植版)

================================================================================
  回测结果 (来自 PythonGo 版，参数相同)
================================================================================

  Sharpe: 0.56 | Calmar: 0.31 | Max DD: 2.46%
  Annual Return: 0.75% | Win Rate: 40.9% | PF: 1.55
  Trades/Year: ~7
  Best Params: gap_threshold=0.5%, trade_window=6 bars

================================================================================
  策略逻辑
================================================================================

  - 检测日盘开盘 (09:00-09:05 bar) 与前一交易时段收盘价的跳空幅度
  - 跳空 > +0.5%: 做空 (高开预期回补)
  - 跳空 < -0.5%: 做多 (低开预期回补)
  - 交易窗口: 跳空后前 trade_window 根 5-min K 线 (默认 6 根 = 30 分钟)
  - 出场: 价格回到前收盘价附近 (缺口回补) 或 硬止损 5%
  - 收盘: 14:55 / 22:55 强制平仓，不持仓过夜

  时间检测说明:
  - JoinQuant every_bar 在 bar 完成时触发，context.current_dt 为 bar 收盘时间
  - 日盘第一根 5-min bar: 09:00-09:05，触发时 current_dt.hour==9, minute==5
  - 日盘最后 5-min bar: 14:55-15:00，触发时 current_dt.hour==15, minute==0

================================================================================
"""

from jqdata import *


# ==============================================================================
#  策略参数
# ==============================================================================

GAP_THRESHOLD  = 0.005  # 跳空阈值 (0.5%)
TRADE_WINDOW   = 6      # 交易窗口 (bars，即 30 分钟)
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

    # ── 跳空检测状态 ────────────────────────────────────────────────────
    g.prev_session_close = 0.0   # 前一交易时段最后收盘价
    g.gap_pct            = 0.0   # 当日跳空幅度
    g.gap_detected       = False  # 当日是否检测到有效跳空
    g.bars_since_gap     = 0      # 跳空后已过的 bar 数
    g.current_date       = None   # 当前日期，用于日切换检测

    # ── 持仓状态 ────────────────────────────────────────────────────────
    g.in_position   = False
    g.position_side = ''    # 'long' 或 'short'
    g.entry_price   = 0.0

    # ── 调度 ────────────────────────────────────────────────────────────
    run_daily(strategy_logic, time='every_bar', reference_security='I8888.XDCE')
    run_daily(close_all, time='14:55', reference_security='I8888.XDCE')
    run_daily(close_all, time='22:55', reference_security='I8888.XDCE')


# ==============================================================================
#  主策略逻辑 (每 5 分钟 K 线调用)
# ==============================================================================

def strategy_logic(context):
    """
    每 5 分钟执行：
      1. 日切换检测，重置跳空状态
      2. 检测日盘开盘 bar (09:05)，计算跳空幅度
      3. 跳空后计数 bar
      4. 检查硬止损
      5. 持仓中: 检查缺口回补出场条件
      6. 空仓: 在交易窗口内根据跳空方向入场
      7. 更新 prev_session_close (排除开盘 bar 以防覆盖)
    """

    contract = get_dominant_future(UNDERLYING)

    # ── 1. 获取当前已完成 bar ────────────────────────────────────────────
    bars = get_bars(
        security=contract,
        count=1,
        unit='5m',
        fields=['open', 'close'],
        include_now=False
    )
    if bars is None or len(bars) == 0:
        return

    current_open  = float(bars['open'][-1])
    current_close = float(bars['close'][-1])

    h = context.current_dt.hour
    m = context.current_dt.minute

    # ── 2. 日切换检测 (新日历日 → 重置跳空检测) ─────────────────────────
    current_date = context.current_dt.date()
    if current_date != g.current_date:
        g.current_date   = current_date
        g.gap_detected   = False
        g.bars_since_gap = 0
        g.gap_pct        = 0.0

    # ── 3. 日盘开盘 bar 检测 (09:05 收盘 = 09:00-09:05 bar) ───────────
    is_day_open_bar = (h == 9 and m == 5)

    if is_day_open_bar and g.prev_session_close > 0 and not g.gap_detected:
        g.gap_pct = (current_open - g.prev_session_close) / g.prev_session_close

        if abs(g.gap_pct) > GAP_THRESHOLD:
            g.gap_detected   = True
            g.bars_since_gap = 0
            log.info(
                f'[跳空检测] gap={g.gap_pct * 100:+.3f}% | '
                f'prev_close={g.prev_session_close:.1f} | '
                f'day_open={current_open:.1f}'
            )

    # ── 4. 跳空后计数 ───────────────────────────────────────────────────
    if g.gap_detected:
        g.bars_since_gap += 1

    # ── 5. 查询持仓 ─────────────────────────────────────────────────────
    sub       = context.subportfolios[0]
    has_long  = _has_position(sub, contract, 'long')
    has_short = _has_position(sub, contract, 'short')

    # ── 6. 硬止损检查 ───────────────────────────────────────────────────
    if has_long and g.entry_price > 0:
        if current_close <= g.entry_price * (1 - HARD_STOP_PCT):
            order_target(contract, 0, side='long')
            log.info(f'[硬止损-多] 现价={current_close:.1f} 入场={g.entry_price:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0
            _update_prev(h, m, current_close, is_day_open_bar)
            return

    if has_short and g.entry_price > 0:
        if current_close >= g.entry_price * (1 + HARD_STOP_PCT):
            order_target(contract, 0, side='short')
            log.info(f'[硬止损-空] 现价={current_close:.1f} 入场={g.entry_price:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0
            _update_prev(h, m, current_close, is_day_open_bar)
            return

    # ── 7. 持仓中: 缺口回补出场 ─────────────────────────────────────────
    if has_long:
        # 价格回升至前收盘价 → 缺口回补完成
        if g.prev_session_close > 0 and current_close >= g.prev_session_close:
            order_target(contract, 0, side='long')
            log.info(f'[平多-回补] 现价={current_close:.1f} 前收={g.prev_session_close:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0

    elif has_short:
        # 价格回落至前收盘价 → 缺口回补完成
        if g.prev_session_close > 0 and current_close <= g.prev_session_close:
            order_target(contract, 0, side='short')
            log.info(f'[平空-回补] 现价={current_close:.1f} 前收={g.prev_session_close:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0

    # ── 8. 空仓: 交易窗口内入场 ─────────────────────────────────────────
    elif g.gap_detected and g.bars_since_gap <= TRADE_WINDOW:
        if g.gap_pct > GAP_THRESHOLD:
            # 高开跳空 → 做空 (fade the gap)
            order(contract, VOLUME, side='short')
            g.entry_price = current_close
            g.in_position = True
            g.position_side = 'short'
            log.info(f'[开空] @ {current_close:.1f} | gap={g.gap_pct * 100:+.3f}%')

        elif g.gap_pct < -GAP_THRESHOLD:
            # 低开跳空 → 做多 (fade the gap)
            order(contract, VOLUME, side='long')
            g.entry_price = current_close
            g.in_position = True
            g.position_side = 'long'
            log.info(f'[开多] @ {current_close:.1f} | gap={g.gap_pct * 100:+.3f}%')

    # ── 9. 更新 prev_session_close ──────────────────────────────────────
    _update_prev(h, m, current_close, is_day_open_bar)


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
    g.gap_detected = False; g.bars_since_gap = 0


# ==============================================================================
#  辅助函数
# ==============================================================================

def _update_prev(h, m, current_close, is_day_open_bar):
    """
    更新 prev_session_close。
    开盘 bar (09:05) 不更新，避免用开盘价覆盖前收盘价。
    """
    if not is_day_open_bar:
        g.prev_session_close = current_close


def _has_position(sub, contract, side):
    """判断指定合约是否有 side 方向的持仓"""
    positions = sub.long_positions if side == 'long' else sub.short_positions
    return contract in positions and positions[contract].total_amount > 0
