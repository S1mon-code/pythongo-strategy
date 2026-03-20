"""
================================================================================
  Semivariance BB 均值回归策略 — 铁矿石 CTA
================================================================================

  适用平台   : 聚宽 JoinQuant
  合约       : 铁矿石主力 (I8888.XDCE)，自动滚动
  K线周期    : 5 分钟
  方向       : 多空双向 (Long + Short)
  版本       : 1.0 (JoinQuant 移植版)

================================================================================
  回测结果 (来自 PythonGo 版，参数相同)
================================================================================

  Sharpe: 1.51 | Max DD: 1.44% | Annual Return: 2.36%
  Win Rate: 55.9% | PF: 1.76 | Trades/Year: ~62
  Best Params: sv_window=30, bb_period=20, asym_threshold=0.55

================================================================================
  策略逻辑
================================================================================

  指标:
  - 正半方差 RS+ = sum(max(ret, 0)^2)，窗口 sv_window 根
  - 负半方差 RS- = sum(max(-ret, 0)^2)，窗口 sv_window 根
  - 不对称比 asym = RS+ / (RS+ + RS-)
  - 布林带 BB(bb_period, 2.0) — 基于收盘价

  入场:
  - 价格跌破 BB 下轨 (穿越) AND asym > 0.55 → 开多 (上行波动占优)
  - 价格突破 BB 上轨 (穿越) AND asym < 0.45 → 开空 (下行波动占优)

  出场:
  - 持多: 价格穿越 BB 上轨 (反向极端) 或 硬止损 5%
  - 持空: 价格穿越 BB 下轨 (反向极端) 或 硬止损 5%

  收盘: 14:55 / 22:55 强制平仓，不持仓过夜

  每次调用读取 max(sv_window+2, bb_period+1) 根 5-min K 线:
  - 上一根 bar 数值 (prev_close/upper/lower) 存入 g，用于穿越检测

================================================================================
"""

from jqdata import *
import numpy as np


# ==============================================================================
#  策略参数
# ==============================================================================

SV_WINDOW      = 30     # 半方差计算窗口 (bars)
BB_PERIOD      = 20     # 布林带周期 (bars)
ASYM_THRESHOLD = 0.55   # 不对称比阈值
HARD_STOP_PCT  = 0.05   # 硬止损比例 5%
VOLUME         = 1      # 每次下单手数
UNDERLYING     = 'I'    # 标的代码 (铁矿石)

# 需要读取的历史 K 线数量
_FETCH_COUNT = max(SV_WINDOW + 2, BB_PERIOD + 2)


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

    # ── 穿越检测状态 (上一根 bar 数值) ────────────────────────────────────
    g.prev_close = 0.0
    g.prev_upper = 0.0
    g.prev_lower = 0.0

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
      1. 读取已完成的历史 K 线
      2. 计算半方差不对称比 (asym) 和布林带 (BB)
      3. 检查硬止损
      4. 持仓中: 反向穿越对面 BB 轨道时出场
      5. 空仓: 穿越 BB 轨道 + asym 过滤时入场
      6. 保存当前 bar 的 close/upper/lower 供下根 bar 穿越检测
    """

    contract = get_dominant_future(UNDERLYING)

    # ── 1. 获取历史 K 线 ────────────────────────────────────────────────
    bars = get_bars(
        security=contract,
        count=_FETCH_COUNT,
        unit='5m',
        fields=['close'],
        include_now=False
    )
    if bars is None or len(bars) < _FETCH_COUNT:
        return

    closes = bars['close'].astype(np.float64)
    current_close = float(closes[-1])

    # ── 2. 计算半方差不对称比 ────────────────────────────────────────────
    sv_prices = closes[-(SV_WINDOW + 1):]               # sv_window+1 个价格
    rets      = np.diff(sv_prices) / sv_prices[:-1]     # sv_window 个收益率

    rs_pos   = float(np.sum(np.maximum(rets, 0) ** 2))
    rs_neg   = float(np.sum(np.maximum(-rets, 0) ** 2))
    rs_total = rs_pos + rs_neg
    asym     = rs_pos / rs_total if rs_total > 0 else 0.5

    # ── 3. 计算布林带 ────────────────────────────────────────────────────
    bb_window = closes[-BB_PERIOD:]
    mid   = float(np.mean(bb_window))
    std   = float(np.std(bb_window, ddof=0))
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std

    # ── 4. 初始化: 第一根有效 bar 仅记录状态 ────────────────────────────
    if g.prev_close == 0.0:
        g.prev_close = current_close
        g.prev_upper = upper
        g.prev_lower = lower
        return

    # ── 5. 穿越检测 ──────────────────────────────────────────────────────
    cross_below = g.prev_close >= g.prev_lower and current_close < lower   # 跌破下轨
    cross_above = g.prev_close <= g.prev_upper and current_close > upper   # 突破上轨

    # ── 6. 查询持仓 ──────────────────────────────────────────────────────
    sub       = context.subportfolios[0]
    has_long  = _has_position(sub, contract, 'long')
    has_short = _has_position(sub, contract, 'short')

    # ── 7. 硬止损检查 ────────────────────────────────────────────────────
    if has_long and g.entry_price > 0:
        if current_close <= g.entry_price * (1 - HARD_STOP_PCT):
            order_target(contract, 0, side='long')
            log.info(f'[硬止损-多] 现价={current_close:.1f} 入场={g.entry_price:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0
            _save_prev(current_close, upper, lower)
            return

    if has_short and g.entry_price > 0:
        if current_close >= g.entry_price * (1 + HARD_STOP_PCT):
            order_target(contract, 0, side='short')
            log.info(f'[硬止损-空] 现价={current_close:.1f} 入场={g.entry_price:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0
            _save_prev(current_close, upper, lower)
            return

    # ── 8. 持仓中: 反向穿越出场 ─────────────────────────────────────────
    if has_long:
        if cross_above:
            order_target(contract, 0, side='long')
            log.info(f'[平多-穿上轨] 现价={current_close:.1f} BB_upper={upper:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0

    elif has_short:
        if cross_below:
            order_target(contract, 0, side='short')
            log.info(f'[平空-穿下轨] 现价={current_close:.1f} BB_lower={lower:.1f}')
            g.in_position = False; g.position_side = ''; g.entry_price = 0.0

    # ── 9. 空仓入场 ──────────────────────────────────────────────────────
    else:
        if cross_below and asym > ASYM_THRESHOLD:
            # 价格跌破下轨 且 上行波动占优 → 均值回归做多
            order(contract, VOLUME, side='long')
            g.entry_price = current_close
            g.in_position = True
            g.position_side = 'long'
            log.info(f'[开多] @ {current_close:.1f} | asym={asym:.3f} | BB_lower={lower:.1f}')

        elif cross_above and asym < (1.0 - ASYM_THRESHOLD):
            # 价格突破上轨 且 下行波动占优 → 均值回归做空
            order(contract, VOLUME, side='short')
            g.entry_price = current_close
            g.in_position = True
            g.position_side = 'short'
            log.info(f'[开空] @ {current_close:.1f} | asym={asym:.3f} | BB_upper={upper:.1f}')

    # ── 10. Debug 日志 (确认条件命中情况，调试完成后可删除) ───────────────
    if cross_below or cross_above:
        log.info(
            f'[CROSS] cross_below={cross_below} cross_above={cross_above} | '
            f'asym={asym:.3f}(需>{ASYM_THRESHOLD}) | '
            f'close={current_close:.1f} lower={lower:.1f} upper={upper:.1f} | '
            f'has_long={has_long} has_short={has_short}'
        )

    # ── 11. 保存本 bar 数据 ──────────────────────────────────────────────
    _save_prev(current_close, upper, lower)


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
    # 重置穿越检测状态，次日重新预热
    g.prev_close = 0.0; g.prev_upper = 0.0; g.prev_lower = 0.0


# ==============================================================================
#  辅助函数
# ==============================================================================

def _save_prev(close, upper, lower):
    """保存当前 bar 数据供下根 bar 穿越检测"""
    g.prev_close = close
    g.prev_upper = upper
    g.prev_lower = lower


def _has_position(sub, contract, side):
    """判断指定合约是否有 side 方向的持仓"""
    positions = sub.long_positions if side == 'long' else sub.short_positions
    return contract in positions and positions[contract].total_amount > 0
