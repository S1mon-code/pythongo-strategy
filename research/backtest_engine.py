"""
Iron Ore CTA Research — Backtest Engine

Bar-by-bar simulation with full position management:
  - Long + Short direction
  - Next-bar execution
  - Scaling in/out (add, TP1 half-close)
  - Hard stop, trailing stop, TP1, TP2
  - Commission + slippage

Priority: hard_stop > trailing_stop > TP2 > TP1 > signal_exit > add > entry
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from .config import (
    COMMISSION_RATE, SLIPPAGE, MULTIPLIER, TICK_SIZE,
    DEFAULT_UNIT, DEFAULT_MAX_LOTS, DEFAULT_ADD_THRESHOLD,
    DEFAULT_ADD_COOLDOWN, DEFAULT_TP1_PCT, DEFAULT_TP2_PCT,
    DEFAULT_HARD_STOP_PCT, DEFAULT_TRAILING_PCT,
    TRADING_DAYS_PER_YEAR,
)


@dataclass
class PositionParams:
    """Position management parameters."""
    unit: int = DEFAULT_UNIT
    max_lots: int = DEFAULT_MAX_LOTS
    add_threshold: float = DEFAULT_ADD_THRESHOLD    # % floating profit to add
    add_cooldown: int = DEFAULT_ADD_COOLDOWN         # bars between adds
    tp1_pct: float = DEFAULT_TP1_PCT                 # % → close half
    tp2_pct: float = DEFAULT_TP2_PCT                 # % → close all
    hard_stop_pct: float = DEFAULT_HARD_STOP_PCT     # %
    trailing_pct: float = DEFAULT_TRAILING_PCT       # %


@dataclass
class Trade:
    """Completed trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int           # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    lots: int
    pnl: float               # net PnL after costs
    exit_reason: str


@dataclass
class BacktestResult:
    """Full backtest output."""
    equity: np.ndarray
    trades: list
    daily_returns: pd.Series = None

    # Summary stats (computed after backtest)
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    trades_per_year: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


def run_backtest(
    df: pd.DataFrame,
    signals: np.ndarray,
    pos_params: PositionParams = None,
    initial_capital: float = 1_000_000.0,
) -> BacktestResult:
    """
    Run backtest on a DataFrame with pre-computed signals.

    Args:
        df: OHLCV DataFrame with datetime index.
        signals: Array of same length as df.
                 1 = long entry signal, -1 = short entry signal, 0 = neutral.
                 2 = force exit (close position).
        pos_params: Position management parameters.
        initial_capital: Starting capital.

    Returns:
        BacktestResult with equity curve, trades, and stats.
    """
    if pos_params is None:
        pos_params = PositionParams()

    n = len(df)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    equity = np.full(n, initial_capital, dtype=np.float64)
    trades = []

    # Position state
    direction = 0          # 0=flat, 1=long, -1=short
    lots = 0
    avg_price = 0.0
    peak_price = 0.0       # for trailing stop (best price since entry)
    tp1_done = False
    bars_since_add = 999
    entry_time = None

    def _calc_cost(price, num_lots):
        """Commission + slippage for a trade."""
        value = price * MULTIPLIER * num_lots
        return value * COMMISSION_RATE + SLIPPAGE * num_lots

    def _close_position(bar_idx, price, reason, close_lots=None):
        nonlocal direction, lots, avg_price, peak_price, tp1_done, entry_time
        if close_lots is None:
            close_lots = lots

        pnl_per_lot = (price - avg_price) * direction * MULTIPLIER
        gross_pnl = pnl_per_lot * close_lots
        cost = _calc_cost(price, close_lots)
        net_pnl = gross_pnl - cost

        trades.append(Trade(
            entry_time=entry_time,
            exit_time=df.index[bar_idx],
            direction=direction,
            entry_price=avg_price,
            exit_price=price,
            lots=close_lots,
            pnl=net_pnl,
            exit_reason=reason,
        ))

        lots -= close_lots
        if lots <= 0:
            direction = 0
            lots = 0
            avg_price = 0.0
            peak_price = 0.0
            tp1_done = False
            entry_time = None
        return net_pnl

    def _open_position(bar_idx, price, dir_, num_lots):
        nonlocal direction, lots, avg_price, peak_price, tp1_done, bars_since_add, entry_time
        cost = _calc_cost(price, num_lots)

        if lots == 0:
            # Fresh entry
            direction = dir_
            avg_price = price
            peak_price = price
            tp1_done = False
            entry_time = df.index[bar_idx]
        else:
            # Adding to position
            avg_price = (avg_price * lots + price * num_lots) / (lots + num_lots)

        lots += num_lots
        bars_since_add = 0
        return -cost  # only cost on entry (PnL realized on exit)

    # Main loop: iterate bar by bar
    cash_pnl = 0.0  # cumulative realized PnL

    for i in range(1, n):
        bars_since_add += 1
        exec_price = opens[i]  # next-bar execution at open
        signal = signals[i - 1] if i > 0 else 0  # signal from previous bar

        bar_pnl = 0.0

        # ── Priority 1: Check stops on current position ──────────────────
        if direction != 0:
            # Update peak price for trailing stop
            if direction == 1:
                peak_price = max(peak_price, highs[i])
                unrealized_pct = (closes[i] - avg_price) / avg_price * 100
                # Hard stop: check if low breached
                hard_stop_price = avg_price * (1 - pos_params.hard_stop_pct / 100)
                if lows[i] <= hard_stop_price:
                    exit_p = max(hard_stop_price, lows[i])
                    bar_pnl += _close_position(i, exit_p, "HARD_STOP")
                    cash_pnl += bar_pnl
                    equity[i] = initial_capital + cash_pnl
                    continue
                # Trailing stop
                trail_price = peak_price * (1 - pos_params.trailing_pct / 100)
                if lows[i] <= trail_price and peak_price > avg_price:
                    exit_p = max(trail_price, lows[i])
                    bar_pnl += _close_position(i, exit_p, "TRAIL_STOP")
                    cash_pnl += bar_pnl
                    equity[i] = initial_capital + cash_pnl
                    continue
            else:  # short
                peak_price = min(peak_price, lows[i])
                unrealized_pct = (avg_price - closes[i]) / avg_price * 100
                # Hard stop
                hard_stop_price = avg_price * (1 + pos_params.hard_stop_pct / 100)
                if highs[i] >= hard_stop_price:
                    exit_p = min(hard_stop_price, highs[i])
                    bar_pnl += _close_position(i, exit_p, "HARD_STOP")
                    cash_pnl += bar_pnl
                    equity[i] = initial_capital + cash_pnl
                    continue
                # Trailing stop
                trail_price = peak_price * (1 + pos_params.trailing_pct / 100)
                if highs[i] >= trail_price and peak_price < avg_price:
                    exit_p = min(trail_price, highs[i])
                    bar_pnl += _close_position(i, exit_p, "TRAIL_STOP")
                    cash_pnl += bar_pnl
                    equity[i] = initial_capital + cash_pnl
                    continue

            # TP2: close all
            if direction == 1:
                tp2_price = avg_price * (1 + pos_params.tp2_pct / 100)
                if highs[i] >= tp2_price:
                    exit_p = tp2_price
                    bar_pnl += _close_position(i, exit_p, "TP2")
                    cash_pnl += bar_pnl
                    equity[i] = initial_capital + cash_pnl
                    continue
            else:
                tp2_price = avg_price * (1 - pos_params.tp2_pct / 100)
                if lows[i] <= tp2_price:
                    exit_p = tp2_price
                    bar_pnl += _close_position(i, exit_p, "TP2")
                    cash_pnl += bar_pnl
                    equity[i] = initial_capital + cash_pnl
                    continue

            # TP1: close half
            if not tp1_done and lots >= 2:
                if direction == 1:
                    tp1_price = avg_price * (1 + pos_params.tp1_pct / 100)
                    if highs[i] >= tp1_price:
                        half = max(1, lots // 2)
                        bar_pnl += _close_position(i, tp1_price, "TP1", close_lots=half)
                        tp1_done = True
                else:
                    tp1_price = avg_price * (1 - pos_params.tp1_pct / 100)
                    if lows[i] <= tp1_price:
                        half = max(1, lots // 2)
                        bar_pnl += _close_position(i, tp1_price, "TP1", close_lots=half)
                        tp1_done = True

        # ── Priority 2: Execute signals (next-bar at open) ───────────────
        if signal == 2 and direction != 0:
            # Force exit
            bar_pnl += _close_position(i, exec_price, "SIGNAL_EXIT")

        elif signal == -direction and direction != 0:
            # Reverse: close then open opposite
            bar_pnl += _close_position(i, exec_price, "REVERSE")
            add_lots = min(pos_params.unit, pos_params.max_lots)
            bar_pnl += _open_position(i, exec_price, signal, add_lots)

        elif signal != 0 and direction == 0:
            # Fresh entry
            add_lots = min(pos_params.unit, pos_params.max_lots)
            bar_pnl += _open_position(i, exec_price, signal, add_lots)

        elif signal == direction and direction != 0:
            # Same direction signal while in position → try to add
            if (lots < pos_params.max_lots
                    and bars_since_add >= pos_params.add_cooldown):
                float_pct = ((closes[i] - avg_price) / avg_price * 100 * direction)
                if float_pct >= pos_params.add_threshold:
                    add_lots = min(pos_params.unit, pos_params.max_lots - lots)
                    if add_lots > 0:
                        bar_pnl += _open_position(i, exec_price, direction, add_lots)

        cash_pnl += bar_pnl
        equity[i] = initial_capital + cash_pnl

        # Mark-to-market unrealized PnL for equity curve
        if direction != 0:
            unrealized = (closes[i] - avg_price) * direction * MULTIPLIER * lots
            equity[i] += unrealized

    # Close any open position at end
    if direction != 0:
        final_pnl = _close_position(n - 1, closes[-1], "END_OF_DATA")
        cash_pnl += final_pnl
        equity[-1] = initial_capital + cash_pnl

    result = BacktestResult(equity=equity, trades=trades)
    _compute_stats(result, df, initial_capital)
    return result


def _compute_stats(result: BacktestResult, df: pd.DataFrame, initial_capital: float):
    """Compute summary statistics from equity curve and trades."""
    equity = result.equity
    trades = result.trades
    n = len(equity)

    # Daily returns from equity curve
    equity_series = pd.Series(equity, index=df.index)
    daily_equity = equity_series.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    result.daily_returns = daily_returns

    # Total / annual return
    total_return = (equity[-1] / initial_capital) - 1
    n_years = max(len(daily_equity) / TRADING_DAYS_PER_YEAR, 0.01)
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 else -1.0

    result.total_return = total_return
    result.annual_return = annual_return

    # Sharpe ratio (annualized, assume rf=0)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        result.sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        result.sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    result.max_drawdown = abs(dd.min()) if len(dd) > 0 else 0.0

    # Calmar ratio
    result.calmar = annual_return / result.max_drawdown if result.max_drawdown > 0 else 0.0

    # Trade statistics
    result.num_trades = len(trades)
    result.trades_per_year = len(trades) / max(n_years, 0.01)

    if trades:
        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        result.win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
        result.avg_trade_pnl = pnls.mean()
        result.avg_win = wins.mean() if len(wins) > 0 else 0.0
        result.avg_loss = losses.mean() if len(losses) > 0 else 0.0

        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")


def print_stats(result: BacktestResult, name: str = "Strategy"):
    """Pretty-print backtest statistics."""
    print(f"\n{'='*60}")
    print(f"  {name} — Backtest Results")
    print(f"{'='*60}")
    print(f"  Total Return:     {result.total_return:>10.2%}")
    print(f"  Annual Return:    {result.annual_return:>10.2%}")
    print(f"  Sharpe Ratio:     {result.sharpe:>10.2f}")
    print(f"  Calmar Ratio:     {result.calmar:>10.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown:>10.2%}")
    print(f"  Profit Factor:    {result.profit_factor:>10.2f}")
    print(f"  Win Rate:         {result.win_rate:>10.2%}")
    print(f"  Num Trades:       {result.num_trades:>10d}")
    print(f"  Trades/Year:      {result.trades_per_year:>10.1f}")
    print(f"  Avg Trade PnL:    {result.avg_trade_pnl:>10.0f}")
    print(f"  Avg Win:          {result.avg_win:>10.0f}")
    print(f"  Avg Loss:         {result.avg_loss:>10.0f}")
    print(f"{'='*60}\n")
