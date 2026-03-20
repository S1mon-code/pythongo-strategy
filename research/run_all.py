"""
Iron Ore CTA Research — Main Runner

Runs walk-forward optimization on all 10 strategies, scores and selects winners.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.config import *
from research.data_loader import load_and_prepare
from research.backtest_engine import run_backtest, print_stats, PositionParams
from research.walk_forward import (
    walk_forward_optimize, check_param_sensitivity,
    compute_composite_score, evaluate_on_test, print_wfo_report,
)

# Import all strategies
from research.strategies.s01_tick_momentum import TickMomentumScalper
from research.strategies.s02_micro_mean_reversion import MicroMeanReversion
from research.strategies.s03_order_flow_imbalance import OrderFlowImbalance
from research.strategies.s04_dual_thrust import DualThrustBreakout
from research.strategies.s05_kama_atr import KamaAtrTrend
from research.strategies.s06_opening_range import OpeningRangeBreakout
from research.strategies.s07_vol_regime_donchian import VolRegimeDonchian
from research.strategies.s08_session_gap import SessionGapReversion
from research.strategies.s09_rsi_exhaustion import RsiExhaustionReversal
from research.strategies.s10_vwap_zscore import VwapZscoreReversion
from research.strategies.s11_candle_pattern_trend import CandlePatternTrend
from research.strategies.s12_night_day_gap import NightDayGapZScore
from research.strategies.s13_semivariance_bb import SemivarianceBB
from research.strategies.s14_open_reversion import OpenReversionStrategy
from research.strategies.s15_variance_ratio import VarianceRatioReversion
from research.strategies.s16_kalman_reversion import KalmanMeanReversion
from research.strategies.s17_session_half import SessionHalfReversion
from research.strategies.s18_hurst_bb import HurstBBReversion
from research.strategies.s19_vwap_band import VwapBandReversion
from research.strategies.s20_atr_expansion_fade import AtrExpansionFade
from research.strategies.s21_linear_reg_channel import LinearRegChannelReversion
from research.strategies.s22_stochastic import StochasticReversion
from research.strategies.s23_ema_envelope import EmaEnvelopeReversion
from research.strategies.s24_bb_candle import BBCandleReversion
from research.strategies.s25_vwap_semivariance import VwapSemivarianceReversion
from research.strategies.s26_channel_volume import ChannelVolumeReversion
from research.strategies.s27_rsi_channel import RsiChannelReversion
from research.strategies.s28_consec_bar import ConsecBarExhaustion
from research.strategies.s29_price_oscillator import PriceOscillatorReversion
from research.strategies.s30_channel_barclose import ChannelBarCloseReversion
from research.strategies.s31_rsi_channel_v2 import RsiChannelReversionV2
from research.strategies.s32_ema_squeeze import EmaSqueezeReversion
from research.strategies.s33_first_reversal_bar import FirstReversalBar
from research.strategies.s34_rsi_divergence import RsiDivergenceReversion
from research.strategies.s35_micro_breakout import MicroBreakoutReversal
from research.strategies.s36_large_body_reversal import LargeBodyReversal
from research.strategies.s37_intrabar_reversal import IntrabarReversal
from research.strategies.s38_return_decel import ReturnDecelReversion

ALL_STRATEGIES = [
    TickMomentumScalper(),
    MicroMeanReversion(),
    OrderFlowImbalance(),
    DualThrustBreakout(),
    KamaAtrTrend(),
    OpeningRangeBreakout(),
    VolRegimeDonchian(),
    SessionGapReversion(),
    RsiExhaustionReversal(),
    VwapZscoreReversion(),
    CandlePatternTrend(),
    NightDayGapZScore(),
    SemivarianceBB(),
    OpenReversionStrategy(),
    VarianceRatioReversion(),
    KalmanMeanReversion(),
    SessionHalfReversion(),
    HurstBBReversion(),
    VwapBandReversion(),
    AtrExpansionFade(),
    LinearRegChannelReversion(),
    StochasticReversion(),
    EmaEnvelopeReversion(),
    BBCandleReversion(),
    VwapSemivarianceReversion(),
    ChannelVolumeReversion(),
    RsiChannelReversion(),
    ConsecBarExhaustion(),
    PriceOscillatorReversion(),
    ChannelBarCloseReversion(),
    RsiChannelReversionV2(),
    EmaSqueezeReversion(),
    FirstReversalBar(),
    RsiDivergenceReversion(),
    MicroBreakoutReversal(),
    LargeBodyReversal(),
    IntrabarReversal(),
    ReturnDecelReversion(),
]


def run_single_strategy(strat, verbose=True):
    """Run WFO + test evaluation for a single strategy."""
    name = strat.name
    freq = strat.freq

    if verbose:
        print(f"\n{'#'*70}")
        print(f"  Processing: {name} ({freq})")
        print(f"{'#'*70}")

    # Load data at the strategy's preferred timeframe
    train_df, test_df = load_and_prepare(freq, with_session=True)
    # Safety-net stops only: let signal reversals handle normal exits
    pos_params = PositionParams(hard_stop_pct=5.0, trailing_pct=99.0, tp1_pct=99.0, tp2_pct=99.0, max_lots=1)
    param_grid = strat.param_grid()

    if verbose:
        print(f"  Train: {len(train_df)} bars | Test: {len(test_df)} bars")
        print(f"  Params: {list(param_grid.keys())}")
        grid_size = 1
        for v in param_grid.values():
            grid_size *= len(v)
        print(f"  Grid size: {grid_size} combinations")

    # Phase 1: Walk-forward optimization on training data
    t0 = time.time()
    wfo_result = walk_forward_optimize(
        signal_func=strat.generate_signals,
        param_grid=param_grid,
        train_df=train_df,
        pos_params=pos_params,
    )
    wfo_time = time.time() - t0

    if verbose:
        print_wfo_report(wfo_result, name)
        print(f"  WFO time: {wfo_time:.1f}s")

    # Get best params from last window (most data)
    best_params = None
    for w in reversed(wfo_result["windows"]):
        if w["best_params"] is not None:
            best_params = w["best_params"]
            break

    if best_params is None:
        if verbose:
            print(f"  ✗ SKIP: No valid params found")
        return None

    # Phase 2: Parameter sensitivity check
    robust = check_param_sensitivity(
        strat.generate_signals, best_params, param_grid, train_df, pos_params
    )

    if verbose:
        print(f"  Parameter sensitivity: {'✓ Robust' if robust else '✗ Fragile'}")

    # Phase 3: Full train backtest with best params
    train_signals = strat.generate_signals(train_df, **best_params)
    train_result = run_backtest(train_df, train_signals, pos_params=pos_params)

    if verbose:
        print_stats(train_result, f"{name} [TRAIN]")

    # Phase 4: Out-of-sample test
    test_result = evaluate_on_test(
        strat.generate_signals, best_params, test_df, pos_params
    )

    if verbose:
        print_stats(test_result, f"{name} [TEST]")

    # Phase 5: Composite score
    robustness_score = 1.0 if robust else 0.0
    composite = compute_composite_score(test_result, robustness_score)

    # Check pass criteria
    passed = (
        wfo_result["passed"]
        and test_result.sharpe >= MIN_SHARPE
        and test_result.max_drawdown <= MAX_DRAWDOWN
        and test_result.trades_per_year >= MIN_TRADES_PER_YEAR
    )

    summary = {
        "name": name,
        "freq": freq,
        "best_params": best_params,
        "wfo_passed": wfo_result["passed"],
        "avg_oos_sharpe": wfo_result["avg_oos_sharpe"],
        "oos_is_ratio": wfo_result["oos_is_ratio"],
        "oos_profitable_windows": wfo_result["oos_profitable_windows"],
        "train_sharpe": train_result.sharpe,
        "train_return": train_result.total_return,
        "train_dd": train_result.max_drawdown,
        "test_sharpe": test_result.sharpe,
        "test_return": test_result.total_return,
        "test_dd": test_result.max_drawdown,
        "test_pf": test_result.profit_factor,
        "test_wr": test_result.win_rate,
        "test_calmar": test_result.calmar,
        "test_trades": test_result.num_trades,
        "test_trades_yr": test_result.trades_per_year,
        "robust": robust,
        "composite_score": composite,
        "passed": passed,
    }

    if verbose:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  {status} | Score: {composite:.3f} | "
              f"Test Sharpe: {test_result.sharpe:.2f} | "
              f"Test DD: {test_result.max_drawdown:.1%}")

    return summary


def run_all(verbose=True):
    """Run all strategies and produce final ranking."""
    results = []

    for strat in ALL_STRATEGIES:
        try:
            summary = run_single_strategy(strat, verbose=verbose)
            if summary:
                results.append(summary)
        except Exception as e:
            print(f"  ✗ ERROR in {strat.name}: {e}")
            import traceback
            traceback.print_exc()

    # Sort by composite score
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # Final report
    print(f"\n{'='*80}")
    print(f"  FINAL RANKING — Iron Ore CTA Strategies")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Strategy':<28} {'Pass':<6} {'Score':<7} "
          f"{'Sharpe':<8} {'Return':<9} {'DD':<8} {'PF':<6} {'WR':<6} {'Trades':<7}")
    print(f"{'─'*80}")

    winners = []
    for i, r in enumerate(results):
        status = "✓" if r["passed"] else "✗"
        print(f"{i+1:<5} {r['name']:<28} {status:<6} {r['composite_score']:<7.3f} "
              f"{r['test_sharpe']:<8.2f} {r['test_return']:<9.1%} "
              f"{r['test_dd']:<8.1%} {r['test_pf']:<6.2f} "
              f"{r['test_wr']:<6.1%} {r['test_trades']:<7d}")
        if r["passed"]:
            winners.append(r)

    print(f"\n  Winners: {len(winners)} / {len(results)} strategies passed")

    # Save results
    output_dir = os.path.join(PROJECT_ROOT, "research", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {json_path}")

    return results, winners


if __name__ == "__main__":
    results, winners = run_all(verbose=True)
