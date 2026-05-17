"""Market data models for AMATIS.

Immutable dataclasses representing market data events.
All timestamps are UTC-aware. Prices use Decimal for precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any

import whenever


class TradeSide(Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class DataSource(Enum):
    """Origin of market data."""
    ALPACA = "alpaca"
    YAHOO = "yahoo"
    POLYGON = "polygon"
    IBKR = "ibkr"
    BINANCE = "binance"
    COINBASE = "coinbase"
    MOCK = "mock"


@dataclass(frozen=True)
class Symbol:
    """Canonical symbol representation.
    
    Normalizes symbols across different exchanges and data sources.
    
    Examples:
        >>> Symbol("AAPL", "NASDAQ", "equity")
        >>> Symbol("BTC", "BINANCE", "crypto")
        >>> Symbol("EUR", "FOREX", "forex", quote_currency="USD")
    """
    base: str
    exchange: str
    asset_class: str = "equity"
    quote_currency: Optional[str] = None
    
    def __post_init__(self):
        # Normalize base to uppercase
        object.__setattr__(self, 'base', self.base.upper())
        object.__setattr__(self, 'exchange', self.exchange.upper())
        if self.quote_currency:
            object.__setattr__(self, 'quote_currency', self.quote_currency.upper())
    
    @property
    def canonical(self) -> str:
        """Canonical string representation."""
        if self.quote_currency:
            return f"{self.base}/{self.quote_currency}"
        return self.base
    
    def __str__(self) -> str:
        if self.quote_currency:
            return f"{self.base}/{self.quote_currency}"
        return self.base


@dataclass(frozen=True)
class Tick:
    """Individual price tick (quote change).
    
    Represents a single quote update from the market.
    """
    symbol: Symbol
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    source: DataSource
    exchange_timestamp: Optional[datetime] = None
    
    @property
    def mid(self) -> Decimal:
        """Midpoint price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> Decimal:
        """Bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> Decimal:
        """Spread in basis points."""
        if self.mid == 0:
            return Decimal("0")
        return (self.spread / self.mid) * Decimal("10000")


@dataclass(frozen=True)
class Quote:
    """Market quote (simplified tick representation).
    
    Lighter weight than Tick for high-frequency updates.
    """
    symbol: Symbol
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    timestamp: datetime
    source: DataSource
    
    @property
    def mid(self) -> Decimal:
        """Midpoint price."""
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class Trade:
    """Individual trade print.
    
    Represents an executed trade on the exchange.
    """
    symbol: Symbol
    price: Decimal
    size: Decimal
    timestamp: datetime
    side: TradeSide
    trade_id: Optional[str] = None
    source: DataSource = DataSource.MOCK
    
    @property
    def value(self) -> Decimal:
        """Notional value of trade."""
        return self.price * self.size


@dataclass(frozen=True)
class OrderBookLevel:
    """Single level in order book (price + size)."""
    price: Decimal
    size: Decimal
    order_count: Optional[int] = None
    
    @property
    def value(self) -> Decimal:
        """Value at this level."""
        return self.price * self.size


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Order book snapshot (L2 data).
    
    Contains bid and ask levels at a point in time.
    """
    symbol: Symbol
    timestamp: datetime
    bids: List[OrderBookLevel]  # Sorted best to worst
    asks: List[OrderBookLevel]  # Sorted best to worst
    source: DataSource
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Best (highest) bid."""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Best (lowest) ask."""
        return self.asks[0] if self.asks else None
    
    @property
    def mid(self) -> Optional[Decimal]:
        """Midpoint from best bid/ask."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Spread from best bid/ask."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    def get_bid_depth(self, levels: int = 5) -> Decimal:
        """Total bid size for top N levels."""
        return sum(level.size for level in self.bids[:levels])
    
    def get_ask_depth(self, levels: int = 5) -> Decimal:
        """Total ask size for top N levels."""
        return sum(level.size for level in self.asks[:levels])


@dataclass(frozen=True)
class OHLCV:
    """OHLCV bar (candlestick data).
    
    Standard price bar with open, high, low, close, volume.
    """
    symbol: Symbol
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    source: DataSource = DataSource.MOCK
    
    # Optional fields
    trades_count: Optional[int] = None
    vwap: Optional[Decimal] = None
    
    @property
    def range(self) -> Decimal:
        """Price range (high - low)."""
        return self.high - self.low
    
    @property
    def change(self) -> Decimal:
        """Price change (close - open)."""
        return self.close - self.open
    
    @property
    def change_pct(self) -> Decimal:
        """Percentage change."""
        if self.open == 0:
            return Decimal("0")
        return (self.change / self.open) * Decimal("100")
    
    @property
    def is_green(self) -> bool:
        """True if close >= open."""
        return self.close >= self.open
    
    @property
    def is_red(self) -> bool:
        """True if close < open."""
        return self.close < self.open
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": str(self.symbol),
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
            "change_pct": str(self.change_pct),
        }


@dataclass(frozen=True)
class MarketEvent:
    """Generic market event wrapper.
    
    Used for streaming data normalization across providers.
    """
    event_type: str  # "quote", "trade", "ohlcv", "book"
    data: Any  # Quote, Trade, OHLCV, or OrderBookSnapshot
    symbol: Symbol
    timestamp: datetime
    source: DataSource
    latency_ms: Optional[float] = None  # Provider latency
