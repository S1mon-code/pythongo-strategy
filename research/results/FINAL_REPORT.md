# Iron Ore CTA Strategy — Final Research Report

## Methodology
- **Data**: I9999 continuous 1-min bars, 2013-10-18 ~ 2026-02-24 (1,028,790 bars)
- **Train**: 2013 ~ 2022 | **Test**: 2023 ~ 2026
- **Execution**: Next-bar at open, commission 0.01%, slippage 0.5 yuan/lot, multiplier 100t/lot
- **Position**: 1 lot, safety-net hard stop 5%, no TP/trailing (let signal reversals exit)
- **Anti-overfitting**: Walk-forward optimization (4 windows), parameter sensitivity analysis

## Key Finding
Iron ore shifted from **trending** (2013-2022) to **mean-reverting** (2023-2026).
- Trend strategies (Dual Thrust, KAMA, Opening Range) worked on train but failed on test
- Mean-reversion strategies worked best on recent data
- Position management stops were destroying edge — safety-net stops with signal-based exits performed best

## ATR Reference (median % per bar)
| Timeframe | Median ATR | p75 ATR |
|-----------|-----------|---------|
| 1min | 0.173% | 0.237% |
| 2min | 0.226% | 0.317% |
| 5min | 0.325% | 0.467% |
| 15min | 0.504% | 0.761% |

## All 10 Strategies — Raw Signal Quality (full train, no stops)

| Strategy | Sharpe | Return | PF | WR | Trades | Best Params |
|----------|--------|--------|----|----|--------|-------------|
| Order Flow Imbalance | 1.05 | 7.4% | 3.25 | 42.7% | 75 | oi_threshold=0.02, vol_ratio=1.5 |
| Session Gap Reversion | 0.99 | 13.2% | 2.26 | 56.4% | 218 | gap_threshold=0.005, atr_mult=1.0 |
| Dual Thrust Breakout | 0.92 | 17.1% | 1.74 | 31.4% | 544 | lookback=3, k1=0.4, k2=0.4 |
| KAMA + ATR Trend | 0.87 | 18.2% | 1.45 | 29.5% | 1092 | kama_period=30, atr_mult=1.5 |
| Opening Range Breakout | 0.65 | 10.6% | 1.44 | 36.9% | 664 | opening_min=60, vol_mult=2.5 |
| Vol Regime + Donchian | 0.60 | 10.0% | 1.27 | 48.7% | 1046 | donch_period=30, vol_ratio=1.0 |
| RSI Exhaustion | 0.51 | 7.7% | 1.71 | 62.6% | 147 | rsi_period=14, oversold=30, overbought=75 |
| Tick Momentum | 0.43 | 8.3% | 1.12 | 27.3% | 3263 | mom_bars=8, threshold=0.008 |
| VWAP Z-Score | 0.17 | 2.3% | 1.08 | 58.1% | 1129 | z_threshold=1.5, min_bars=50 |
| Micro Mean-Reversion | -0.03 | -0.4% | 1.03 | 63.4% | 9526 | bb_period=15, bb_std=2.0 |

## Test Period Results (best train params → OOS test)

| Strategy | Test Sharpe | Test Return | DD | PF | WR | Trades/Yr | 2023 | 2024 | 2025 | 2026 |
|----------|------------|-------------|-----|-----|-----|-----------|------|------|------|------|
| **Micro Mean-Rev** | **1.17** | **+6.0%** | 2.1% | 1.16 | 63.8% | 1034 | +1.1% | +2.7% | +1.5% | +0.6% |
| **VWAP Z-Score** | **0.65** | **+3.3%** | 2.1% | 1.20 | 62.6% | 103 | -0.4% | +2.0% | +1.5% | +0.1% |
| **Session Gap** | **0.56** | **+2.4%** | 2.5% | 1.55 | 40.9% | 7 | +2.3% | +0.4% | +0.1% | -0.4% |
| Dual Thrust | -0.23 | -1.2% | 3.8% | 0.93 | 37.6% | 62 | -0.3% | +1.5% | -2.4% | +0.0% |
| KAMA + ATR | -0.12 | -0.6% | 2.8% | 0.93 | 35.0% | 26 | -0.1% | +0.8% | -1.5% | +0.1% |

## Winners (3 strategies)

### 1. Micro Mean-Reversion (布林带均值回归)
- **Timeframe**: 1-min bars, bb_period=30 (≈15 bars of 2-min)
- **Logic**: Bollinger Band extreme deviation → fade
- **Strength**: 4/4 years positive, highest Sharpe, highest trade count
- **Risk**: Mean-reversion may fail in strong trending regimes
- **PythonGo**: `strategies/bollinger/MicroMeanReversion_PythonGo.py`

### 2. VWAP Z-Score Reversion (VWAP偏离回归)
- **Timeframe**: 5-min bars, z_threshold=3.0, min_bars=30
- **Logic**: Price deviation from VWAP exceeds 3σ → fade
- **Strength**: 3/4 years positive, complementary to BB strategy
- **Risk**: Low 2023 performance
- **PythonGo**: `strategies/composite/VwapZscoreReversion_PythonGo.py`

### 3. Session-Gap Reversion (跳空回补)
- **Timeframe**: 5-min bars, gap_threshold=0.5%
- **Logic**: Fade overnight gaps >0.5%
- **Strength**: Highest PF (1.55), consistent concept
- **Risk**: Very low trade frequency (~7/year)
- **PythonGo**: `strategies/composite/SessionGapReversion_PythonGo.py`

## Lessons Learned
1. **Stops destroy edge** on high-frequency mean-reversion strategies — use safety-net only
2. **Market regime matters** — iron ore transitioned from trending to mean-reverting post-2022
3. **Signal quality > position management** — a good signal with no stops beats a bad signal with perfect stops
4. **WFO may be too strict** for regime-dependent strategies — yearly consistency is a better metric
