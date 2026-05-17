"""Canonical Event Definitions for AMATIS.

All system events are defined here with:
    - Strict typing (no Any)
    - Validation rules
    - Version compatibility
    - Deterministic serialization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import whenever

from amatix.data.market.models import OHLCV, Quote, Symbol, Tick, Trade
from amatix.signals.models import Signal, SignalDirection, SignalStrength


class EventVersion(Enum):
    """Schema versions for event compatibility."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


@dataclass(frozen=True)
class EventMetadata:
    """Immutable event metadata."""
    event_id: UUID
    timestamp: datetime
    source: str
    version: EventVersion
    trace_id: Optional[UUID] = None
    
    @classmethod
    def create(
        cls,
        source: str,
        version: EventVersion = EventVersion.V1_0,
        trace_id: Optional[UUID] = None,
    ) -> EventMetadata:
        return cls(
            event_id=uuid4(),
            timestamp=whenever.now().py_datetime(),
            source=source,
            version=version,
            trace_id=trace_id or uuid4(),
        )


# =============================================================================
# MARKET DATA EVENTS
# =============================================================================

@dataclass(frozen=True)
class MarketDataEvent:
    """Base market data event."""
    symbol: str
    timestamp: datetime
    source: str
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("market"))


@dataclass(frozen=True)
class OHLCVEvent:
    """OHLCV bar event."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    timeframe: str  # "1m", "5m", "1h", "1d"
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("market"))
    
    @classmethod
    def from_ohlcv(cls, ohlcv: OHLCV, timeframe: str = "1m") -> OHLCVEvent:
        return cls(
            symbol=ohlcv.symbol,
            timestamp=ohlcv.timestamp,
            open=ohlcv.open,
            high=ohlcv.high,
            low=ohlcv.low,
            close=ohlcv.close,
            volume=ohlcv.volume,
            timeframe=timeframe,
            metadata=EventMetadata.create("market"),
        )
    
    def validate(self) -> None:
        """Validate invariants."""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) < Low ({self.low})")
        if self.open <= 0 or self.close <= 0:
            raise ValueError("Prices must be positive")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")


@dataclass(frozen=True)
class QuoteEvent:
    """Quote/tick event."""
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    bid_size: int
    ask_size: int
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("market"))
    
    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid
    
    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2
    
    def validate(self) -> None:
        if self.ask < self.bid:
            raise ValueError(f"Ask ({self.ask}) < Bid ({self.bid})")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("Size cannot be negative")


@dataclass(frozen=True)
class TradeEvent:
    """Trade execution event."""
    symbol: str
    timestamp: datetime
    price: Decimal
    size: int
    side: str  # "buy" or "sell"
    exchange: Optional[str] = None
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("market"))
    
    def validate(self) -> None:
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.size <= 0:
            raise ValueError("Size must be positive")
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {self.side}")


# =============================================================================
# ORDER EVENTS
# =============================================================================

@dataclass(frozen=True)
class OrderSubmittedEvent:
    """Order submitted to OMS."""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    order_type: str  # "market", "limit", "stop"
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("oms"))
    
    def validate(self) -> None:
        if not self.order_id:
            raise ValueError("order_id required")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == "limit" and (not self.limit_price or self.limit_price <= 0):
            raise ValueError("limit_price required for limit orders")


@dataclass(frozen=True)
class OrderAcceptedEvent:
    """Order accepted by broker."""
    order_id: str
    broker_order_id: str
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("oms"))


@dataclass(frozen=True)
class OrderRejectedEvent:
    """Order rejected."""
    order_id: str
    reason: str
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("oms"))


@dataclass(frozen=True)
class OrderFilledEvent:
    """Order fill event."""
    order_id: str
    symbol: str
    filled_quantity: Decimal
    filled_price: Decimal
    commission: Decimal
    remaining_quantity: Decimal
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("oms"))
    
    @property
    def total_value(self) -> Decimal:
        return self.filled_quantity * self.filled_price
    
    def validate(self) -> None:
        if self.filled_quantity <= 0:
            raise ValueError("filled_quantity must be positive")
        if self.filled_price <= 0:
            raise ValueError("filled_price must be positive")
        if self.commission < 0:
            raise ValueError("commission cannot be negative")


@dataclass(frozen=True)
class OrderCancelledEvent:
    """Order cancelled."""
    order_id: str
    reason: str
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("oms"))


@dataclass(frozen=True)
class OrderExpiredEvent:
    """Order expired."""
    order_id: str
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("oms"))


# =============================================================================
# SIGNAL EVENTS
# =============================================================================

@dataclass(frozen=True)
class SignalGeneratedEvent:
    """Trading signal generated."""
    signal_id: str
    symbol: str
    direction: str  # "long", "short", "neutral"
    strength: str  # "weak", "moderate", "strong", "extreme"
    confidence: float  # 0.0 to 1.0
    source: str  # "momentum", "news", "ml"
    metadata: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    event_metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("signals"))
    
    def validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.direction not in ("long", "short", "neutral"):
            raise ValueError(f"Invalid direction: {self.direction}")
        if self.strength not in ("weak", "moderate", "strong", "extreme"):
            raise ValueError(f"Invalid strength: {self.strength}")


@dataclass(frozen=True)
class SignalExpiredEvent:
    """Signal expired."""
    signal_id: str
    reason: str
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("signals"))


@dataclass(frozen=True)
class SignalValidatedEvent:
    """Signal passed validation."""
    signal_id: str
    validated_by: str
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("signals"))


# =============================================================================
# RISK EVENTS
# =============================================================================

@dataclass(frozen=True)
class RiskAssessmentEvent:
    """Risk assessment completed."""
    order_id: str
    passed: bool
    risk_score: float  # 0.0 to 1.0
    rules_checked: List[str]
    violations: List[str]
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("risk"))
    
    def validate(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(f"risk_score must be in [0, 1], got {self.risk_score}")


@dataclass(frozen=True)
class RiskLimitBreachedEvent:
    """Risk limit breached."""
    limit_type: str  # "drawdown", "exposure", "concentration"
    current_value: Decimal
    limit_value: Decimal
    severity: str  # "warning", "critical"
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("risk"))


@dataclass(frozen=True)
class KillSwitchTriggeredEvent:
    """Kill switch activated."""
    reason: str
    triggered_by: str
    level: str  # "soft", "hard", "emergency"
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("risk"))


# =============================================================================
# PORTFOLIO EVENTS
# =============================================================================

@dataclass(frozen=True)
class PositionOpenedEvent:
    """New position opened."""
    symbol: str
    side: str  # "long" or "short"
    quantity: Decimal
    entry_price: Decimal
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("portfolio"))


@dataclass(frozen=True)
class PositionUpdatedEvent:
    """Position updated (partial fill, etc.)."""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("portfolio"))


@dataclass(frozen=True)
class PositionClosedEvent:
    """Position closed."""
    symbol: str
    exit_price: Decimal
    realized_pnl: Decimal
    exit_reason: str  # "signal", "stop_loss", "take_profit", "manual"
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("portfolio"))


@dataclass(frozen=True)
class PortfolioUpdatedEvent:
    """Portfolio state updated."""
    total_value: Decimal
    cash: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    open_positions: int
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("portfolio"))


# =============================================================================
# SYSTEM EVENTS
# =============================================================================

@dataclass(frozen=True)
class SystemStartedEvent:
    """System started."""
    version: str
    environment: str
    config_hash: str
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("system"))


@dataclass(frozen=True)
class SystemStoppedEvent:
    """System stopped."""
    reason: str
    uptime_seconds: float
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("system"))


@dataclass(frozen=True)
class ConfigurationChangedEvent:
    """Configuration changed."""
    changed_keys: List[str]
    old_hash: str
    new_hash: str
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("system"))


@dataclass(frozen=True)
class ErrorEvent:
    """System error."""
    error_type: str
    message: str
    component: str
    recoverable: bool
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    metadata: EventMetadata = field(default_factory=lambda: EventMetadata.create("system"))
