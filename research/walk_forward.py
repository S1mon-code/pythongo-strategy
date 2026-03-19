"""
Iron Ore CTA Research — Walk-Forward Optimizer

4-window anchored expanding walk-forward:
  Window 1: IS = [0..25%],  OOS = [25%..50%]
  Window 2: IS = [0..50%],  OOS = [50%..62.5%]  (approx)
  Window 3: IS = [0..62.5%], OOS = [62.5%..75%]
  Window 4: IS = [0..75%],  OOS = [75%..100%]

Key anti-overfitting rules:
  - Max 2-3 params, coarse grid
  - Must be profitable in 3/4 OOS windows
  - OOS/IS Sharpe > 0.5
  - Parameter sensitivity: +-1 step must not break
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Callable
from .backtest_engine import run_backtest, BacktestResult, PositionParams
from .config import (
    WFO_WINDOWS, MIN_SHARPE, MAX_DRAWDOWN, MIN_TRADES_PER_YEAR,
    MIN_PROFITABLE_WINDOWS, OOS_IS_SHARPE_RATIO,
    W_SHARPE, W_CALMAR, W_PF, W_WR, W_ROBUST,
)


def generate_wfo_splits(df: pd.DataFrame, n_windows: int = WFO_WINDOWS):
    """
    Generate anchored expanding walk-forward splits.

    Yields (is_df, oos_df) pairs where IS always starts from the beginning.
    """
    n = len(df)
    # Divide into (n_windows + 1) segments for expanding IS + OOS
    segment_size = n // (n_windows + 1)

    for w in range(n_windows):
        is_end = segment_size * (w + 1)
        oos_end = segment_size * (w + 2)

        is_df = df.iloc[:is_end]
        oos_df = df.iloc[is_end:oos_end]

        yield is_df, oos_df


def optimize_strategy(
    signal_func: Callable,
    param_grid: dict,
    df: pd.DataFrame,
    pos_params: PositionParams = None,
) -> tuple:
    """
    Grid search over parameter combinations, return best params and result.

    Args:
        signal_func: Function(df, **params) -> np.ndarray of signals.
        param_grid: Dict of param_name -> list of values to search.
        df: Training data DataFrame.
        pos_params: Position management parameters.

    Returns:
        (best_params, best_result, all_results)
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_sharpe = -np.inf
    best_params = None
    best_result = None
    all_results = []

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        try:
            signals = signal_func(df, **params)
            result = run_backtest(df, signals, pos_params=pos_params)
            all_results.append((params, result))

            if result.sharpe > best_sharpe and result.num_trades >= 10:
                best_sharpe = result.sharpe
                best_params = params
                best_result = result
        except Exception:
            continue

    return best_params, best_result, all_results


def walk_forward_optimize(
    signal_func: Callable,
    param_grid: dict,
    train_df: pd.DataFrame,
    pos_params: PositionParams = None,
    n_windows: int = WFO_WINDOWS,
) -> dict:
    """
    Run full walk-forward optimization.

    Args:
        signal_func: Function(df, **params) -> np.ndarray of signals.
        param_grid: Dict of param_name -> list of values.
        train_df: Full training data.
        pos_params: Position management parameters.
        n_windows: Number of WFO windows.

    Returns:
        Dict with keys: windows, oos_equity, best_params, oos_sharpes,
        is_sharpes, passed, composite_score.
    """
    windows = []
    oos_sharpes = []
    is_sharpes = []
    oos_profitable = 0

    for i, (is_df, oos_df) in enumerate(generate_wfo_splits(train_df, n_windows)):
        # Optimize on IS
        best_params, is_result, _ = optimize_strategy(
            signal_func, param_grid, is_df, pos_params
        )

        if best_params is None:
            windows.append({
                "window": i + 1,
                "best_params": None,
                "is_sharpe": 0.0,
                "oos_sharpe": 0.0,
                "oos_return": 0.0,
                "oos_trades": 0,
            })
            oos_sharpes.append(0.0)
            is_sharpes.append(0.0)
            continue

        # Evaluate on OOS with best IS params
        oos_signals = signal_func(oos_df, **best_params)
        oos_result = run_backtest(oos_df, oos_signals, pos_params=pos_params)

        is_sharpe = is_result.sharpe if is_result else 0.0
        oos_sharpe = oos_result.sharpe

        if oos_result.total_return > 0:
            oos_profitable += 1

        windows.append({
            "window": i + 1,
            "best_params": best_params,
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "oos_return": oos_result.total_return,
            "oos_dd": oos_result.max_drawdown,
            "oos_trades": oos_result.num_trades,
            "oos_pf": oos_result.profit_factor,
            "oos_wr": oos_result.win_rate,
        })

        oos_sharpes.append(oos_sharpe)
        is_sharpes.append(is_sharpe)

    # Aggregate OOS metrics
    avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0
    avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
    oos_is_ratio = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0.0

    # Check pass criteria
    passed = (
        avg_oos_sharpe >= MIN_SHARPE
        and oos_profitable >= MIN_PROFITABLE_WINDOWS
        and oos_is_ratio >= OOS_IS_SHARPE_RATIO
    )

    return {
        "windows": windows,
        "avg_oos_sharpe": avg_oos_sharpe,
        "avg_is_sharpe": avg_is_sharpe,
        "oos_is_ratio": oos_is_ratio,
        "oos_profitable_windows": oos_profitable,
        "passed": passed,
    }


def check_param_sensitivity(
    signal_func: Callable,
    best_params: dict,
    param_grid: dict,
    df: pd.DataFrame,
    pos_params: PositionParams = None,
) -> bool:
    """
    Check that +-1 grid step from best params doesn't break the strategy.
    Returns True if robust (all neighbors have Sharpe > 0).
    """
    param_names = list(param_grid.keys())
    neighbors_ok = 0
    neighbors_total = 0

    for name in param_names:
        grid_vals = param_grid[name]
        best_val = best_params[name]

        if best_val not in grid_vals:
            continue

        idx = grid_vals.index(best_val)

        for offset in [-1, 1]:
            ni = idx + offset
            if 0 <= ni < len(grid_vals):
                neighbors_total += 1
                neighbor_params = dict(best_params)
                neighbor_params[name] = grid_vals[ni]
                try:
                    signals = signal_func(df, **neighbor_params)
                    result = run_backtest(df, signals, pos_params=pos_params)
                    if result.sharpe > 0:
                        neighbors_ok += 1
                except Exception:
                    pass

    return neighbors_ok == neighbors_total if neighbors_total > 0 else False


def compute_composite_score(result: BacktestResult, robustness: float = 1.0) -> float:
    """
    Compute composite selection score.
    Score = 0.30*Sharpe + 0.25*Calmar + 0.20*PF + 0.15*WR + 0.10*Robustness
    """
    # Normalize components to roughly 0-1 range
    sharpe_norm = min(result.sharpe / 3.0, 1.0)
    calmar_norm = min(result.calmar / 3.0, 1.0)
    pf_norm = min(result.profit_factor / 3.0, 1.0)
    wr_norm = result.win_rate

    score = (
        W_SHARPE * sharpe_norm
        + W_CALMAR * calmar_norm
        + W_PF * pf_norm
        + W_WR * wr_norm
        + W_ROBUST * robustness
    )
    return score


def evaluate_on_test(
    signal_func: Callable,
    best_params: dict,
    test_df: pd.DataFrame,
    pos_params: PositionParams = None,
) -> BacktestResult:
    """Run strategy with best params on test data."""
    signals = signal_func(test_df, **best_params)
    return run_backtest(test_df, signals, pos_params=pos_params)


def print_wfo_report(wfo_result: dict, name: str = "Strategy"):
    """Pretty-print walk-forward optimization report."""
    print(f"\n{'='*70}")
    print(f"  {name} — Walk-Forward Optimization Report")
    print(f"{'='*70}")

    for w in wfo_result["windows"]:
        params_str = str(w["best_params"]) if w["best_params"] else "FAILED"
        print(f"\n  Window {w['window']}:")
        print(f"    Params:     {params_str}")
        print(f"    IS Sharpe:  {w['is_sharpe']:.2f}")
        print(f"    OOS Sharpe: {w['oos_sharpe']:.2f}")
        print(f"    OOS Return: {w.get('oos_return', 0):.2%}")
        print(f"    OOS Trades: {w.get('oos_trades', 0)}")

    print(f"\n  {'─'*50}")
    print(f"  Avg OOS Sharpe:      {wfo_result['avg_oos_sharpe']:.2f}")
    print(f"  Avg IS Sharpe:       {wfo_result['avg_is_sharpe']:.2f}")
    print(f"  OOS/IS Ratio:        {wfo_result['oos_is_ratio']:.2f}")
    print(f"  Profitable Windows:  {wfo_result['oos_profitable_windows']}/{WFO_WINDOWS}")
    print(f"  PASSED:              {'✓ YES' if wfo_result['passed'] else '✗ NO'}")
    print(f"{'='*70}\n")
