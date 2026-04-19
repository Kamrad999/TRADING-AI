"""
Agent modules for the Trading AI system.

Agents are responsible for data collection, signal generation, and optimization.
They form the intelligence layer of the trading pipeline.
"""

from .news_collector import NewsCollector
from .institutional_signal_generator import InstitutionalSignalGenerator
from .regime_detector import RegimeDetector
from .optimizer import SignalOptimizer

__all__ = [
    "NewsCollector",
    "InstitutionalSignalGenerator",
    "RegimeDetector",
    "SignalOptimizer"
]
