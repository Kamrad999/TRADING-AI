"""
Market data models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict


@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None


@dataclass
class PriceData:
    """Price data with indicators."""
    symbol: str
    price: float
    volume: float
    change_24h: float
    change_pct_24h: float
    high_24h: float
    low_24h: float
    indicators: Dict[str, float]
    timestamp: datetime
