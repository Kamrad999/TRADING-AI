"""
Market data module for TRADING-AI system.
Contains data provider and technical indicators following reference patterns.
"""

from .data_provider import DataProvider
from .technical_indicators import TechnicalIndicators

__all__ = [
    "DataProvider",
    "TechnicalIndicators"
]
