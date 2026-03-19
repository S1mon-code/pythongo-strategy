# Iron Ore CTA Strategy Development — Task Tracker

## Phase 1: Research Infrastructure ✅
- [x] config.py — train/test split, DCE fees, slippage, multiplier
- [x] data_loader.py — load parquet, resample 1min~1H, tag sessions
- [x] backtest_engine.py — bar-by-bar L+S, next-bar exec, stops, commission
- [x] walk_forward.py — 4-window anchored expanding WFO
- [x] base_strategy.py — abstract base class
- [x] run_all.py — main runner with ranking

## Phase 2: 10 Strategies ✅
- [x] S01: Tick Momentum Scalper (1min) — Test Sharpe -0.67
- [x] S02: Micro Mean-Reversion (2min) — **Test Sharpe 1.17** ⭐
- [x] S03: Order Flow Imbalance (3min) — Test Sharpe -1.13
- [x] S04: Dual Thrust Breakout (15min) — Test Sharpe -0.23
- [x] S05: KAMA + ATR Trend (15min) — Test Sharpe -0.12
- [x] S06: Opening Range Breakout (5min) — Test Sharpe -0.86
- [x] S07: Vol Regime + Donchian (15min) — Test Sharpe -0.40
- [x] S08: Session-Gap Reversion (5min) — **Test Sharpe 0.56** ⭐
- [x] S09: RSI Exhaustion Reversal (15min) — Test Sharpe -0.13
- [x] S10: VWAP Z-Score Reversion (5min) — **Test Sharpe 0.65** ⭐

## Phase 3: Anti-Overfitting & Evaluation ✅
- [x] Raw signal quality analysis (all 10 strategies positive without stops)
- [x] ATR calibration: stops were too tight, destroying edge
- [x] Safety-net stops approach: 5% hard stop, signal reversals handle exits
- [x] Walk-forward optimization
- [x] Yearly breakdown analysis
- [x] Final report: research/results/FINAL_REPORT.md

## Phase 4: PythonGo Conversion (in progress)
- [ ] Micro Mean-Reversion → strategies/bollinger/
- [ ] VWAP Z-Score Reversion → strategies/composite/
- [ ] Session-Gap Reversion → strategies/composite/

## Phase 5: Final
- [ ] Review all code
- [ ] Commit + push
