"""AMATIS Backtesting Engine.

Event replay, simulation, and strategy evaluation.
"""

from amatix.backtesting.engine import BacktestEngine
from amatix.backtesting.simulator import MarketSimulator

__all__ = [
    "BacktestEngine",
    "MarketSimulator",
]
