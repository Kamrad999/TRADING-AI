"""
Strategy system for TRADING-AI.
Following Freqtrade/Jesse patterns for clean strategy abstraction.
"""

from .base_strategy import BaseStrategy
from .strategy_interface import IStrategy
from .market_data_pipeline import MarketDataPipeline
from .strategy_manager import StrategyManager
from .news_strategy import NewsStrategy
from .technical_strategy import TechnicalStrategy
from .hybrid_strategy import HybridStrategy

__all__ = [
    "BaseStrategy",
    "IStrategy", 
    "MarketDataPipeline",
    "StrategyManager",
    "NewsStrategy",
    "TechnicalStrategy",
    "HybridStrategy"
]
