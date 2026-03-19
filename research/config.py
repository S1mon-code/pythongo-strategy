"""
Iron Ore CTA Research — Global Configuration
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PARQUET_PATH = os.path.join(DATA_DIR, "I9999.parquet")
MAPPING_PATH = os.path.join(DATA_DIR, "I9999_mapping.csv")

# ── Train / Test Split ────────────────────────────────────────────────────
TRAIN_START = "2013-10-18"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2026-12-31"

# ── DCE Iron Ore Contract Specs ───────────────────────────────────────────
COMMISSION_RATE = 0.0001      # 0.01% of trade value
SLIPPAGE = 0.5                # yuan per lot per side
MULTIPLIER = 100              # tons per lot
TICK_SIZE = 0.5               # minimum price increment

# ── Position Management Defaults ──────────────────────────────────────────
DEFAULT_UNIT = 1              # base lots per entry
DEFAULT_MAX_LOTS = 3          # max position size
DEFAULT_ADD_THRESHOLD = 0.3   # % floating profit to add
DEFAULT_ADD_COOLDOWN = 10     # bars between adds
DEFAULT_TP1_PCT = 0.5         # % profit → close half
DEFAULT_TP2_PCT = 1.0         # % profit → close all
DEFAULT_HARD_STOP_PCT = 0.3   # % hard stop loss
DEFAULT_TRAILING_PCT = 0.5    # % trailing stop

# ── Walk-Forward Optimization ─────────────────────────────────────────────
WFO_WINDOWS = 4               # number of walk-forward windows
WFO_EXPANDING = True          # anchored expanding window

# ── Selection Criteria ────────────────────────────────────────────────────
MIN_SHARPE = 0.8
MAX_DRAWDOWN = 0.15           # 15%
MIN_TRADES_PER_YEAR = 30
MIN_PROFITABLE_WINDOWS = 3    # out of 4
OOS_IS_SHARPE_RATIO = 0.5

# Score weights
W_SHARPE = 0.30
W_CALMAR = 0.25
W_PF = 0.20
W_WR = 0.15
W_ROBUST = 0.10

# ── DCE Iron Ore Trading Sessions ────────────────────────────────────────
# Day session 1: 09:00 - 11:30
# Day session 2: 13:30 - 15:00
# Night session:  21:00 - 23:00  (varies by year, simplified)
SESSION_DAY1 = ("09:00", "11:30")
SESSION_DAY2 = ("13:30", "15:00")
SESSION_NIGHT = ("21:00", "23:00")

# ── Annualization ─────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 242
