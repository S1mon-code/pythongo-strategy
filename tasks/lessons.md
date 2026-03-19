# Lessons Learned

## Iron Ore CTA Project

### Data
- I9999.parquet has 1,028,790 1-min bars from 2013-10-18 to 2026-02-24
- Columns: datetime, OHLCV (adjusted + raw), oi, trading_day, factor, is_rollover
- Adjusted prices: raw * factor (factor goes from 0.127 to 1.0)
- Night session bars (21:00-23:00) belong to next trading day

### Infrastructure
- Backtest uses bar-by-bar loop for complex position management (scaling, multi-stop)
- WFO uses anchored expanding windows (IS always starts from beginning)
- Commission: 0.01% of trade value + 0.5 yuan slippage per lot
