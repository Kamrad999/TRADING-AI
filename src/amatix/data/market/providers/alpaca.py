"""Alpaca Markets data provider for AMATIS.

Implements DataProvider for:
    - Historical bars
    - Latest quotes
    - Streaming quotes (WebSocket)
    - Streaming trades

Docs: https://alpaca.markets/docs/
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urljoin

import aiohttp
import websockets

from amatix.core.config import get_settings
from amatix.core.observability import get_logger, get_metrics
from amatix.data.market.models import (
    OHLCV,
    Quote,
    Symbol,
    Tick,
    Trade,
    TradeSide,
    DataSource,
)
from amatix.data.market.normalizer import normalize_symbol
from amatix.data.market.providers.base import BaseDataProvider, ProviderConfig
from amatix.data.market.stream_manager import StreamManager

logger = get_logger(__name__)


class AlpacaDataProvider(BaseDataProvider):
    """Alpaca Markets data provider.
    
    Supports both live and paper trading environments.
    
    Features:
        - Historical data via REST API
        - Real-time quotes via WebSocket
        - Rate limiting compliance
        - Automatic reconnection
        - Event-driven streaming
    
    Example:
        >>> config = ProviderConfig(
        ...     api_key="YOUR_KEY",
        ...     api_secret="YOUR_SECRET",
        ... )
        >>> provider = AlpacaDataProvider(config, event_bus)
        >>> await provider.connect()
        >>> 
        >>> quote = await provider.get_quote(Symbol("AAPL", "NASDAQ"))
        >>> bars = await provider.get_ohlcv(Symbol("AAPL", "NASDAQ"), "1D", 100)
    """
    
    # API endpoints
    BASE_URL_PROD = "https://data.alpaca.markets"
    BASE_URL_PAPER = "https://data.sandbox.alpaca.markets"
    WS_URL_PROD = "wss://stream.data.alpaca.markets"
    WS_URL_PAPER = "wss://stream.data.sandbox.alpaca.markets"
    
    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        event_bus: Optional[Any] = None,
    ) -> None:
        """Initialize Alpaca provider.
        
        Args:
            config: Provider configuration (loads from env if None)
            event_bus: Event bus for data fanout
        """
        # Load from settings if not provided
        if config is None:
            settings = get_settings()
            config = ProviderConfig(
                api_key=settings.execution.alpaca_api_key if hasattr(settings, 'execution') else None,
                api_secret=settings.execution.alpaca_secret_key if hasattr(settings, 'execution') else None,
            )
        
        super().__init__(config, event_bus)
        
        # Determine environment
        self._paper_mode = self._config.base_url and "sandbox" in self._config.base_url
        
        # Set endpoints
        self._base_url = self._config.base_url or (
            self.BASE_URL_PAPER if self._paper_mode else self.BASE_URL_PROD
        )
        self._ws_url = self._config.websocket_url or (
            self.WS_URL_PAPER if self._paper_mode else self.WS_URL_PROD
        )
        
        # Session
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_manager: Optional[StreamManager] = None
        
        # Rate limiting
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[float] = None
    
    @property
    def name(self) -> str:
        """Provider name."""
        return "alpaca"
    
    @property
    def source(self) -> DataSource:
        """Data source identifier."""
        return DataSource.ALPACA
    
    @property
    def is_paper(self) -> bool:
        """Check if using paper trading."""
        return self._paper_mode
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "APCA-API-KEY-ID": self._config.api_key or "",
            "APCA-API-SECRET-KEY": self._config.api_secret or "",
        }
    
    async def _connect_internal(self) -> None:
        """Establish HTTP session."""
        self._session = aiohttp.ClientSession(
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(total=30),
        )
        
        # Initialize WebSocket manager
        self._ws_manager = StreamManager(self._event_bus)
        await self._ws_manager.connect(self._ws_url)
        
        # Authenticate WebSocket
        await self._authenticate_websocket()
        
        logger.info(
            "Alpaca connected",
            environment="paper" if self._paper_mode else "live",
            base_url=self._base_url,
        )
    
    async def _authenticate_websocket(self) -> None:
        """Send authentication message to WebSocket."""
        if not self._ws_manager:
            return
        
        auth_msg = {
            "action": "auth",
            "key": self._config.api_key,
            "secret": self._config.api_secret,
        }
        
        await self._ws_manager.send(json.dumps(auth_msg))
        logger.debug("WebSocket authentication sent")
    
    async def _disconnect_internal(self) -> None:
        """Close connections."""
        if self._ws_manager:
            await self._ws_manager.disconnect()
            self._ws_manager = None
        
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _get_price_internal(self, symbol: Symbol) -> Decimal:
        """Get latest price via quote."""
        quote = await self._get_quote_internal(symbol)
        return quote.mid
    
    async def _get_quote_internal(self, symbol: Symbol) -> Quote:
        """Get latest quote from Alpaca."""
        url = urljoin(self._base_url, f"/v2/stocks/{symbol.base}/quotes/latest")
        
        async with self._session.get(url) as response:
            await self._handle_rate_limit(response)
            response.raise_for_status()
            data = await response.json()
        
        quote_data = data.get("quote", {})
        
        return Quote(
            symbol=symbol,
            bid=Decimal(str(quote_data.get("bp", 0))),
            ask=Decimal(str(quote_data.get("ap", 0))),
            bid_size=Decimal(str(quote_data.get("bs", 0))),
            ask_size=Decimal(str(quote_data.get("as", 0))),
            timestamp=self._parse_timestamp(quote_data.get("t")),
            source=self.source,
        )
    
    async def _get_ohlcv_internal(
        self,
        symbol: Symbol,
        timeframe: str,
        limit: int,
    ) -> List[OHLCV]:
        """Get historical bars from Alpaca."""
        # Map AMATIS timeframe to Alpaca format
        alpaca_timeframe = self._map_timeframe(timeframe)
        
        url = urljoin(
            self._base_url,
            f"/v2/stocks/{symbol.base}/bars"
        )
        
        params = {
            "timeframe": alpaca_timeframe,
            "limit": min(limit, 10000),  # Alpaca max
            "adjustment": "raw",
            "feed": "iex",  # or "sip" for paid plans
        }
        
        async with self._session.get(url, params=params) as response:
            await self._handle_rate_limit(response)
            response.raise_for_status()
            data = await response.json()
        
        bars = data.get("bars", [])
        
        return [
            OHLCV(
                symbol=symbol,
                timestamp=self._parse_timestamp(bar.get("t")),
                open=Decimal(str(bar.get("o", 0))),
                high=Decimal(str(bar.get("h", 0))),
                low=Decimal(str(bar.get("l", 0))),
                close=Decimal(str(bar.get("c", 0))),
                volume=Decimal(str(bar.get("v", 0))),
                source=self.source,
                vwap=Decimal(str(bar.get("vw", 0))) if "vw" in bar else None,
            )
            for bar in bars
        ]
    
    async def _subscribe_quotes_internal(self, symbols: List[Symbol]) -> None:
        """Subscribe to quote updates."""
        if not self._ws_manager:
            return
        
        subscribe_msg = {
            "action": "subscribe",
            "quotes": [s.base for s in symbols],
        }
        
        await self._ws_manager.send(json.dumps(subscribe_msg))
        
        # Override message processor
        self._ws_manager._process_message = self._process_ws_message
        
        logger.debug(
            "Subscribed to quotes",
            symbols=[s.base for s in symbols],
        )
    
    async def _subscribe_trades_internal(self, symbols: List[Symbol]) -> None:
        """Subscribe to trade updates."""
        if not self._ws_manager:
            return
        
        subscribe_msg = {
            "action": "subscribe",
            "trades": [s.base for s in symbols],
        }
        
        await self._ws_manager.send(json.dumps(subscribe_msg))
        
        # Override message processor
        self._ws_manager._process_message = self._process_ws_message
        
        logger.debug(
            "Subscribed to trades",
            symbols=[s.base for s in symbols],
        )
    
    async def _process_ws_message(self, raw_message: str) -> None:
        """Process WebSocket message."""
        try:
            messages = json.loads(raw_message)
            
            # Alpaca sends array of messages
            if not isinstance(messages, list):
                messages = [messages]
            
            for msg in messages:
                msg_type = msg.get("T")  # Message type
                
                if msg_type == "q":  # Quote
                    await self._handle_quote_message(msg)
                elif msg_type == "t":  # Trade
                    await self._handle_trade_message(msg)
                elif msg_type == "b":  # Bar
                    await self._handle_bar_message(msg)
                elif msg_type == "success":
                    logger.debug("WebSocket success", message=msg.get("msg"))
                elif msg_type == "error":
                    logger.error("WebSocket error", message=msg.get("msg"))
                    
        except json.JSONDecodeError:
            logger.error("Invalid JSON from WebSocket", message=raw_message[:200])
        except Exception as e:
            logger.error("Error processing WebSocket message", error=str(e))
    
    async def _handle_quote_message(self, msg: Dict[str, Any]) -> None:
        """Handle quote update."""
        symbol = normalize_symbol(msg.get("S", ""), "NASDAQ", "equity")
        
        quote = Quote(
            symbol=symbol,
            bid=Decimal(str(msg.get("bp", 0))),
            ask=Decimal(str(msg.get("ap", 0))),
            bid_size=Decimal(str(msg.get("bs", 0))),
            ask_size=Decimal(str(msg.get("as", 0))),
            timestamp=self._parse_timestamp(msg.get("t")),
            source=self.source,
        )
        
        # Update cache
        cache_key = f"quote:{symbol.canonical}"
        await self._quote_cache.set(cache_key, quote)
        
        # Emit event
        await self._emit_quote_event(quote)
        
        # Call registered callback
        callback_key = f"quote:{symbol.canonical}"
        if callback_key in self._stream_callbacks:
            self._stream_callbacks[callback_key](quote)
        
        get_metrics().counter("alpaca_quote_received")
    
    async def _handle_trade_message(self, msg: Dict[str, Any]) -> None:
        """Handle trade update."""
        symbol = normalize_symbol(msg.get("S", ""), "NASDAQ", "equity")
        
        # Determine side
        side = TradeSide.UNKNOWN
        if msg.get("t") == "@":  # @ indicates buyer-initiated
            side = TradeSide.BUY
        elif msg.get("t") == "Z":  # Z indicates seller-initiated
            side = TradeSide.SELL
        
        trade = Trade(
            symbol=symbol,
            price=Decimal(str(msg.get("p", 0))),
            size=Decimal(str(msg.get("s", 0))),
            timestamp=self._parse_timestamp(msg.get("t")),
            side=side,
            trade_id=str(msg.get("i", "")),
            source=self.source,
        )
        
        # Emit event
        await self._emit_trade_event(trade)
        
        # Call registered callback
        callback_key = f"trade:{symbol.canonical}"
        if callback_key in self._stream_callbacks:
            self._stream_callbacks[callback_key](trade)
        
        get_metrics().counter("alpaca_trade_received")
    
    async def _handle_bar_message(self, msg: Dict[str, Any]) -> None:
        """Handle real-time bar update."""
        symbol = normalize_symbol(msg.get("S", ""), "NASDAQ", "equity")
        
        bar = OHLCV(
            symbol=symbol,
            timestamp=self._parse_timestamp(msg.get("t")),
            open=Decimal(str(msg.get("o", 0))),
            high=Decimal(str(msg.get("h", 0))),
            low=Decimal(str(msg.get("l", 0))),
            close=Decimal(str(msg.get("c", 0))),
            volume=Decimal(str(msg.get("v", 0))),
            source=self.source,
        )
        
        # Emit event
        await self._event_bus.emit_new(
            EventType.MARKET_DATA_RECEIVED,
            {
                "type": "bar",
                "symbol": str(symbol),
                "open": str(bar.open),
                "high": str(bar.high),
                "low": str(bar.low),
                "close": str(bar.close),
                "volume": str(bar.volume),
            },
            priority=EventPriority.HIGH,
            source=self.name,
        )
        
        get_metrics().counter("alpaca_bar_received")
    
    async def _handle_rate_limit(self, response: aiohttp.ClientResponse) -> None:
        """Track rate limit headers."""
        self._rate_limit_remaining = int(
            response.headers.get("X-Ratelimit-Remaining", 0)
        )
        
        get_metrics().gauge(
            "alpaca_rate_limit_remaining",
            self._rate_limit_remaining,
        )
        
        if self._rate_limit_remaining < 10:
            logger.warning(
                "Alpaca rate limit low",
                remaining=self._rate_limit_remaining,
            )
    
    def _map_timeframe(self, timeframe: str) -> str:
        """Map AMATIS timeframe to Alpaca format.
        
        AMATIS: 1m, 5m, 15m, 1h, 4h, 1D, 1W, 1M
        Alpaca: 1Min, 5Min, 15Min, 1Hour, 4Hour, 1Day, 1Week, 1Month
        """
        mapping = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "4h": "4Hour",
            "1D": "1Day",
            "1W": "1Week",
            "1M": "1Month",
        }
        return mapping.get(timeframe, timeframe)
    
    def _parse_timestamp(self, ts: Optional[str]) -> datetime:
        """Parse Alpaca timestamp string."""
        from datetime import datetime
        
        if not ts:
            return datetime.utcnow()
        
        # Alpaca uses RFC3339
        try:
            # Remove Z and parse
            ts = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(ts)
        except ValueError:
            return datetime.utcnow()
    
    async def get_trades(
        self,
        symbol: Symbol,
        start: datetime,
        end: datetime,
        limit: int = 10000,
    ) -> List[Trade]:
        """Get historical trades (Alpaca-specific).
        
        Args:
            symbol: Symbol to query
            start: Start timestamp
            end: End timestamp
            limit: Max trades to return
        
        Returns:
            List of Trade objects
        """
        url = urljoin(self._base_url, f"/v2/stocks/{symbol.base}/trades")
        
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": min(limit, 10000),
        }
        
        async with self._session.get(url, params=params) as response:
            await self._handle_rate_limit(response)
            response.raise_for_status()
            data = await response.json()
        
        trades = data.get("trades", [])
        
        return [
            Trade(
                symbol=symbol,
                price=Decimal(str(trade.get("p", 0))),
                size=Decimal(str(trade.get("s", 0))),
                timestamp=self._parse_timestamp(trade.get("t")),
                side=TradeSide.UNKNOWN,  # Alpaca doesn't provide side in history
                trade_id=str(trade.get("i", "")),
                source=self.source,
            )
            for trade in trades
        ]
