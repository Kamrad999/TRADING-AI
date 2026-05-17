"""AMATIS Event Contracts — Canonical Event Schemas.

This module defines immutable, versioned event contracts for all
AMATIS system events. These schemas ensure:
    - Backward compatibility
    - Replay determinism
    - Schema validation
    - Documentation

Usage:
    from amatix.contracts import MarketDataEvent, OrderSubmittedEvent
    
    # Events are validated on construction
    event = MarketDataEvent(
        symbol="AAPL",
        price=150.25,
        timestamp=datetime.utcnow(),
    )
"""

from __future__ import annotations

from amatix.contracts.events import (
    # Market data events
    MarketDataEvent,
    OHLCVEvent,
    QuoteEvent,
    TradeEvent,
    
    # Order events
    OrderSubmittedEvent,
    OrderAcceptedEvent,
    OrderRejectedEvent,
    OrderFilledEvent,
    OrderCancelledEvent,
    OrderExpiredEvent,
    
    # Signal events
    SignalGeneratedEvent,
    SignalExpiredEvent,
    SignalValidatedEvent,
    
    # Risk events
    RiskAssessmentEvent,
    RiskLimitBreachedEvent,
    KillSwitchTriggeredEvent,
    
    # Portfolio events
    PositionOpenedEvent,
    PositionUpdatedEvent,
    PositionClosedEvent,
    PortfolioUpdatedEvent,
    
    # System events
    SystemStartedEvent,
    SystemStoppedEvent,
    ConfigurationChangedEvent,
    ErrorEvent,
)

from amatix.contracts.schemas import (
    EventSchema,
    EventField,
    FieldType,
    ValidationRule,
    SchemaVersion,
    SchemaRegistry,
)

__all__ = [
    # Events
    "MarketDataEvent",
    "OHLCVEvent",
    "QuoteEvent",
    "TradeEvent",
    "OrderSubmittedEvent",
    "OrderAcceptedEvent",
    "OrderRejectedEvent",
    "OrderFilledEvent",
    "OrderCancelledEvent",
    "OrderExpiredEvent",
    "SignalGeneratedEvent",
    "SignalExpiredEvent",
    "SignalValidatedEvent",
    "RiskAssessmentEvent",
    "RiskLimitBreachedEvent",
    "KillSwitchTriggeredEvent",
    "PositionOpenedEvent",
    "PositionUpdatedEvent",
    "PositionClosedEvent",
    "PortfolioUpdatedEvent",
    "SystemStartedEvent",
    "SystemStoppedEvent",
    "ConfigurationChangedEvent",
    "ErrorEvent",
    # Schemas
    "EventSchema",
    "EventField",
    "FieldType",
    "ValidationRule",
    "SchemaVersion",
    "SchemaRegistry",
]
