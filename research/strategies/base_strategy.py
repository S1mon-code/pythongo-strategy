"""
Base class for all research strategies.

Each strategy must implement:
  - name: str
  - freq: str (target timeframe)
  - param_grid: dict of param -> list of values
  - generate_signals(df, **params) -> np.ndarray
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from ..backtest_engine import PositionParams


class BaseResearchStrategy(ABC):
    """Abstract base for research strategies."""

    name: str = "BaseStrategy"
    freq: str = "5min"   # default timeframe

    @abstractmethod
    def param_grid(self) -> dict:
        """Return dict of param_name -> [values] for optimization."""
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, **params) -> np.ndarray:
        """
        Generate trading signals from OHLCV data.

        Args:
            df: DataFrame with open, high, low, close, volume, oi columns.
            **params: Strategy-specific parameters.

        Returns:
            np.ndarray of same length as df:
                1  = long entry signal
                -1 = short entry signal
                0  = no signal / neutral
                2  = force exit
        """
        ...

    def position_params(self) -> PositionParams:
        """Override to customize position management."""
        return PositionParams()
