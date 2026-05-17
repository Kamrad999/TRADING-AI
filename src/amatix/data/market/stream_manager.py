"""Stream Manager for real-time market data.

Coordinates:
    - Multiple symbol subscriptions
    - Connection management
    - Reconnection with exponential backoff
    - Heartbeat monitoring
    - Event fanout to EventBus
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
from uuid import uuid4

import websockets

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import EventContext, EventPriority, EventType
from amatix.core.observability import get_logger, get_metrics
from amatix.data.market.models import Symbol

logger = get_logger(__name__)


class StreamState(Enum):
    """Connection state for a stream."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()


@dataclass
class Subscription:
    """Active subscription metadata."""
    symbol: Symbol
    channels: List[str]  # "quotes", "trades", "bars"
    callback: Optional[Callable[[Any], Coroutine[Any, Any, None]]] = None
    

class StreamManager:
    """Central coordinator for real-time data streams.
    
    Manages WebSocket connections to market data providers:
        - Automatic reconnection
        - Heartbeat monitoring
        - Subscription management
        - Event fanout
    
    Example:
        >>> manager = StreamManager(event_bus)
        >>> await manager.connect("wss://stream.data.provider")
        >>> await manager.subscribe(Symbol("AAPL", "NASDAQ"), ["quotes"])
        >>> 
        >>> # Data flows:
        >>> # Provider -> StreamManager -> EventBus -> Components
    """
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        heartbeat_interval: float = 30.0,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 60.0,
    ) -> None:
        """Initialize stream manager.
        
        Args:
            event_bus: Event bus for fanout
            heartbeat_interval: Seconds between heartbeats
            reconnect_base_delay: Base delay for exponential backoff
            reconnect_max_delay: Max delay for reconnection
        """
        self._event_bus = event_bus
        
        # Connection settings
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay
        
        # State
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._state = StreamState.DISCONNECTED
        self._url: Optional[str] = None
        
        # Subscriptions
        self._subscriptions: Dict[str, Subscription] = {}
        self._global_subscriptions: List[str] = []
        
        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._messages_received = 0
        self._reconnect_count = 0
        self._last_message_time: Optional[float] = None
    
    @property
    def state(self) -> StreamState:
        """Current connection state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._state == StreamState.CONNECTED
    
    async def connect(self, url: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Connect to WebSocket endpoint.
        
        Args:
            url: WebSocket URL
            headers: Optional connection headers
        """
        if self._state == StreamState.CONNECTED:
            logger.warning("Already connected, disconnecting first")
            await self.disconnect()
        
        self._url = url
        self._state = StreamState.CONNECTING
        
        try:
            logger.info("Connecting to stream", url=url)
            
            self._websocket = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=None,  # We handle heartbeats
            )
            
            self._state = StreamState.CONNECTED
            self._reconnect_count = 0
            
            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info("Stream connected successfully")
            
            # Emit event
            await self._event_bus.emit_new(
                EventType.MARKET_DATA_RECEIVED,  # Reuse or create new type
                {"event": "connected", "url": url},
                source="stream_manager",
            )
            
        except Exception as e:
            self._state = StreamState.FAILED
            logger.error("Stream connection failed", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Disconnect and clean up."""
        logger.info("Disconnecting stream")
        
        self._state = StreamState.DISCONNECTED
        
        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close websocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        logger.info("Stream disconnected")
    
    async def subscribe(
        self,
        symbol: Symbol,
        channels: List[str],
        callback: Optional[Callable[[Any], Coroutine[Any, Any, None]]] = None,
    ) -> str:
        """Subscribe to symbol data.
        
        Args:
            symbol: Symbol to subscribe to
            channels: Data channels (quotes, trades, bars)
            callback: Optional callback for this symbol
        
        Returns:
            Subscription ID
        """
        sub_id = f"{symbol.canonical}:{uuid4().hex[:8]}"
        
        self._subscriptions[sub_id] = Subscription(
            symbol=symbol,
            channels=channels,
            callback=callback,
        )
        
        logger.debug(
            "Subscribed to symbol",
            subscription_id=sub_id,
            symbol=str(symbol),
            channels=channels,
        )
        
        get_metrics().counter(
            "stream_subscriptions",
            labels={"symbol": str(symbol), "channels": ",".join(channels)},
        )
        
        return sub_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe by ID."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.debug("Unsubscribed", subscription_id=subscription_id)
            return True
        return False
    
    async def _receive_loop(self) -> None:
        """Main receive loop."""
        try:
            while self._state == StreamState.CONNECTED and self._websocket:
                try:
                    message = await self._websocket.recv()
                    self._messages_received += 1
                    self._last_message_time = asyncio.get_event_loop().time()
                    
                    # Process message
                    await self._process_message(message)
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error("Error processing message", error=str(e))
                    get_metrics().counter("stream_message_errors")
        
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        
        finally:
            # Trigger reconnection if appropriate
            if self._state != StreamState.DISCONNECTED:
                asyncio.create_task(self._handle_reconnect())
    
    async def _process_message(self, raw_message: str) -> None:
        """Process incoming message.
        
        Override this in provider-specific subclasses.
        """
        # Base implementation just logs
        logger.debug("Received message", size=len(raw_message))
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        try:
            while self._state == StreamState.CONNECTED:
                await asyncio.sleep(self._heartbeat_interval)
                
                # Check if we should be connected
                if not self._websocket:
                    break
                
                # Check for stale connection
                if self._last_message_time:
                    elapsed = asyncio.get_event_loop().time() - self._last_message_time
                    if elapsed > self._heartbeat_interval * 3:
                        logger.warning("Stale connection detected")
                        break
                
                # Send heartbeat (if provider requires)
                # await self._send_heartbeat()
        
        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
    
    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        if self._state == StreamState.DISCONNECTED:
            return
        
        self._state = StreamState.RECONNECTING
        self._reconnect_count += 1
        
        # Calculate backoff
        delay = min(
            self._reconnect_base_delay * (2 ** (self._reconnect_count - 1)),
            self._reconnect_max_delay,
        )
        
        logger.warning(
            "Reconnecting",
            attempt=self._reconnect_count,
            delay=delay,
        )
        
        await asyncio.sleep(delay)
        
        try:
            if self._url:
                await self.connect(self._url)
                
                # Re-subscribe
                for sub_id, sub in self._subscriptions.items():
                    logger.debug("Re-subscribing", subscription_id=sub_id)
                    # Provider-specific re-subscription logic
        
        except Exception as e:
            logger.error("Reconnection failed", error=str(e))
            # Will trigger another reconnect via _receive_loop
    
    async def send(self, message: str) -> None:
        """Send message to websocket."""
        if self._websocket and self._state == StreamState.CONNECTED:
            await self._websocket.send(message)
        else:
            raise RuntimeError("Not connected")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            "state": self._state.name,
            "connected": self.is_connected,
            "subscriptions": len(self._subscriptions),
            "messages_received": self._messages_received,
            "reconnect_count": self._reconnect_count,
            "last_message_time": self._last_message_time,
        }


class MultiStreamManager:
    """Manages multiple stream connections (e.g., quotes + trades)."""
    
    def __init__(self, event_bus: HardenedEventBusV2) -> None:
        """Initialize multi-stream manager."""
        self._event_bus = event_bus
        self._streams: Dict[str, StreamManager] = {}
    
    def add_stream(self, name: str, stream: StreamManager) -> None:
        """Add named stream."""
        self._streams[name] = stream
    
    async def connect_all(self) -> None:
        """Connect all streams."""
        tasks = [
            asyncio.create_task(
                stream.connect(stream._url)  # type: ignore
            )
            for stream in self._streams.values()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def disconnect_all(self) -> None:
        """Disconnect all streams."""
        tasks = [
            asyncio.create_task(stream.disconnect())
            for stream in self._streams.values()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all streams."""
        return {
            name: stream.get_stats()
            for name, stream in self._streams.items()
        }
