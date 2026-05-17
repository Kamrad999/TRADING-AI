"""Base classes for market data providers.

Extends amatix.interfaces.DataProvider with provider-specific
utilities and common functionality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from amatix.core.circuit_breaker import CircuitBreaker
from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import EventPriority, EventType
from amatix.core.observability import get_logger, get_metrics, timed
from amatix.data.market.cache import MarketDataCache
from amatix.data.market.models import (
    OHLCV,
    Quote,
    Symbol,
    Tick,
    Trade,
    DataSource,
)
from amatix.interfaces import DataProvider as IDataProvider

logger = get_logger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for data providers."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    websocket_url: Optional[str] = None
    
    # Rate limiting
    requests_per_second: float = 10.0
    burst_size: int = 5
    
    # Caching
    cache_ttl_seconds: float = 5.0
    quote_cache_ttl: float = 1.0
    
    # Streaming
    enable_websocket: bool = True
    reconnect_attempts: int = 5


class BaseDataProvider(IDataProvider, ABC):
    """Base class for market data providers.
    
    Provides common functionality:
        - Circuit breaker protection
        - Caching layer
        - Rate limiting
        - Event emission
        - Metrics collection
    
    Subclasses implement provider-specific logic.
    """
    
    def __init__(
        self,
        config: ProviderConfig,
        event_bus: HardenedEventBusV2,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        """Initialize base provider.
        
        Args:
            config: Provider configuration
            event_bus: Event bus for data fanout
            circuit_breaker: Optional circuit breaker
        """
        self._config = config
        self._event_bus = event_bus
        self._circuit_breaker = circuit_breaker
        
        # Initialize caches
        self._quote_cache = MarketDataCache[Quote](
            ttl_seconds=config.quote_cache_ttl,
            max_size=10000,
        )
        self._ohlcv_cache = MarketDataCache[List[OHLCV]](
            ttl_seconds=config.cache_ttl_seconds,
            max_size=1000,
        )
        
        self._connected = False
        self._stream_callbacks: Dict[str, Callable[[Any], None]] = {}
    
    @property
    @abstractmethod
    def source(self) -> DataSource:
        """Data source identifier."""
        pass
    
    @abstractmethod
    async def _connect_internal(self) -> None:
        """Provider-specific connection logic."""
        pass
    
    @abstractmethod
    async def _disconnect_internal(self) -> None:
        """Provider-specific disconnection logic."""
        pass
    
    @abstractmethod
    async def _get_price_internal(self, symbol: Symbol) -> Decimal:
        """Provider-specific price fetch."""
        pass
    
    @abstractmethod
    async def _get_quote_internal(self, symbol: Symbol) -> Quote:
        """Provider-specific quote fetch."""
        pass
    
    @abstractmethod
    async def _get_ohlcv_internal(
        self,
        symbol: Symbol,
        timeframe: str,
        limit: int,
    ) -> List[OHLCV]:
        """Provider-specific OHLCV fetch."""
        pass
    
    async def connect(self) -> None:
        """Connect to provider with circuit breaker."""
        if self._circuit_breaker:
            await self._circuit_breaker.call(self._connect_internal)
        else:
            await self._connect_internal()
        
        self._connected = True
        logger.info(f"{self.name} connected")
    
    async def disconnect(self) -> None:
        """Disconnect from provider."""
        await self._disconnect_internal()
        self._connected = False
        logger.info(f"{self.name} disconnected")
    
    async def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected
    
    @timed("provider_get_price_latency")
    async def get_price(self, symbol: Symbol) -> Decimal:
        """Get current price with caching."""
        cache_key = f"price:{symbol.canonical}"
        
        # Try cache
        cached = await self._quote_cache.get(cache_key)
        if cached:
            return cached.mid  # type: ignore
        
        # Fetch fresh
        if self._circuit_breaker:
            price = await self._circuit_breaker.call(
                lambda: self._get_price_internal(symbol)
            )
        else:
            price = await self._get_price_internal(symbol)
        
        get_metrics().counter(
            "provider_price_requests",
            labels={"provider": self.name, "symbol": symbol.canonical},
        )
        
        return price
    
    @timed("provider_get_quote_latency")
    async def get_quote(self, symbol: Symbol) -> Quote:
        """Get current quote with caching."""
        cache_key = f"quote:{symbol.canonical}"
        
        # Try cache
        cached = await self._quote_cache.get(cache_key)
        if cached:
            get_metrics().counter("provider_quote_cache_hits")
            return cached
        
        # Fetch fresh
        if self._circuit_breaker:
            quote = await self._circuit_breaker.call(
                lambda: self._get_quote_internal(symbol)
            )
        else:
            quote = await self._get_quote_internal(symbol)
        
        # Update cache
        await self._quote_cache.set(cache_key, quote)
        
        # Emit event
        await self._emit_quote_event(quote)
        
        get_metrics().counter(
            "provider_quote_requests",
            labels={"provider": self.name, "symbol": symbol.canonical},
        )
        
        return quote
    
    @timed("provider_get_ohlcv_latency")
    async def get_ohlcv(
        self,
        symbol: Symbol,
        timeframe: str,
        limit: int = 100,
    ) -> List[OHLCV]:
        """Get OHLCV bars with caching."""
        cache_key = f"ohlcv:{symbol.canonical}:{timeframe}:{limit}"
        
        # Try cache
        cached = await self._ohlcv_cache.get(cache_key)
        if cached:
            get_metrics().counter("provider_ohlcv_cache_hits")
            return cached
        
        # Fetch fresh
        if self._circuit_breaker:
            bars = await self._circuit_breaker.call(
                lambda: self._get_ohlcv_internal(symbol, timeframe, limit)
            )
        else:
            bars = await self._get_ohlcv_internal(symbol, timeframe, limit)
        
        # Update cache
        await self._ohlcv_cache.set(cache_key, bars)
        
        get_metrics().counter(
            "provider_ohlcv_requests",
            labels={
                "provider": self.name,
                "symbol": symbol.canonical,
                "timeframe": timeframe,
            },
        )
        
        return bars
    
    async def subscribe_quotes(
        self,
        symbols: List[Symbol],
        callback: Callable[[Quote], None],
    ) -> None:
        """Subscribe to streaming quotes."""
        for symbol in symbols:
            key = f"quote:{symbol.canonical}"
            self._stream_callbacks[key] = callback
        
        await self._subscribe_quotes_internal(symbols)
    
    async def subscribe_trades(
        self,
        symbols: List[Symbol],
        callback: Callable[[Trade], None],
    ) -> None:
        """Subscribe to streaming trades."""
        for symbol in symbols:
            key = f"trade:{symbol.canonical}"
            self._stream_callbacks[key] = callback
        
        await self._subscribe_trades_internal(symbols)
    
    @abstractmethod
    async def _subscribe_quotes_internal(self, symbols: List[Symbol]) -> None:
        """Provider-specific quote subscription."""
        pass
    
    @abstractmethod
    async def _subscribe_trades_internal(self, symbols: List[Symbol]) -> None:
        """Provider-specific trade subscription."""
        pass
    
    async def _emit_quote_event(self, quote: Quote) -> None:
        """Emit quote to event bus."""
        await self._event_bus.emit_new(
            EventType.MARKET_DATA_RECEIVED,
            {
                "type": "quote",
                "symbol": str(quote.symbol),
                "bid": str(quote.bid),
                "ask": str(quote.ask),
                "mid": str(quote.mid),
            },
            priority=EventPriority.HIGH,
            source=self.name,
        )
    
    async def _emit_trade_event(self, trade: Trade) -> None:
        """Emit trade to event bus."""
        await self._event_bus.emit_new(
            EventType.MARKET_DATA_RECEIVED,
            {
                "type": "trade",
                "symbol": str(trade.symbol),
                "price": str(trade.price),
                "size": str(trade.size),
                "side": trade.side.value,
            },
            priority=EventPriority.HIGH,
            source=self.name,
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "quote_cache": self._quote_cache.get_stats(),
            "ohlcv_cache": self._ohlcv_cache.get_stats(),
        }
