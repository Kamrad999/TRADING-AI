"""Event models for AMATIS event-driven architecture.

All significant system events are modeled as dataclasses with strict typing,
enabling type-safe event handling, replay, and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import whenever


class EventType(Enum):
    """Domain events that flow through the AMATIS system.
    
    These events represent significant state changes and decisions
    that must be auditable and replayable.
    """
    
    # System lifecycle
    SYSTEM_STARTED = auto()
    SYSTEM_SHUTDOWN = auto()
    COMPONENT_INITIALIZED = auto()
    COMPONENT_FAILED = auto()
    
    # Data events
    MARKET_DATA_RECEIVED = auto()
    MARKET_DATA_STALE = auto()
    NEWS_ARRIVED = auto()
    NEWS_FILTERED = auto()
    
    # Signal events
    SIGNAL_GENERATED = auto()
    SIGNAL_VALIDATED = auto()
    SIGNAL_EXPIRED = auto()
    SIGNAL_REJECTED = auto()
    
    # Risk events
    RISK_CHECK_REQUESTED = auto()
    RISK_CHECK_PASSED = auto()
    RISK_CHECK_FAILED = auto()
    KILL_SWITCH_TRIGGERED = auto()
    DRAWDOWN_LIMIT_HIT = auto()
    
    # Execution events
    ORDER_SUBMITTED = auto()
    ORDER_ACCEPTED = auto()
    ORDER_REJECTED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    
    # Portfolio events
    PORTFOLIO_UPDATED = auto()
    ALLOCATION_CHANGED = auto()
    
    # Regime events
    REGIME_DETECTED = auto()
    REGIME_CHANGED = auto()
    
    # Meta/learning events
    DECISION_RECORDED = auto()
    PERFORMANCE_UPDATED = auto()
    ALPHA_DECAY_DETECTED = auto()
    
    # Custom extension point
    CUSTOM = auto()


class EventPriority(Enum):
    """Priority levels for event processing.
    
    Critical events (risk, execution) are processed before
    analytical events to ensure system safety.
    """
    CRITICAL = 0    # Risk events, kill switches
    HIGH = 1        # Execution events
    NORMAL = 2      # Signals, portfolio updates
    LOW = 3         # Analytics, logging
    BACKGROUND = 4  # Maintenance, cleanup


@dataclass(frozen=True)
class EventContext:
    """Immutable context carried with every event.
    
    Provides traceability across the distributed system.
    """
    trace_id: UUID
    parent_id: Optional[UUID] = None
    source_component: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def with_child_context(self, component: str) -> EventContext:
        """Create a child context for nested operations."""
        return EventContext(
            trace_id=self.trace_id,
            parent_id=self.trace_id,
            source_component=component,
            correlation_id=self.correlation_id,
        )


@dataclass
class Event:
    """Base event class for all AMATIS domain events.
    
    Events are the primary mechanism for communication between
    decoupled components. They are:
        - Immutable (via frozen context)
        - Serializable (for replay and audit)
        - Typed (for safe handling)
        - Observable (for monitoring)
    
    Attributes:
        event_type: The type of event (from EventType enum)
        payload: Event-specific data (type varies by event_type)
        context: Immutable tracing and metadata context
        event_id: Unique identifier for this event instance
        priority: Processing priority
    
    Example:
        >>> event = Event(
        ...     event_type=EventType.SIGNAL_GENERATED,
        ...     payload={"symbol": "AAPL", "direction": "LONG"},
        ...     context=EventContext(trace_id=uuid4(), source_component="signal_engine"),
        ... )
    """
    
    event_type: EventType
    payload: Dict[str, Any]
    context: EventContext
    event_id: UUID = field(default_factory=uuid4)
    priority: EventPriority = EventPriority.NORMAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary for storage/transmission."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.name,
            "priority": self.priority.name,
            "timestamp": self.context.timestamp.isoformat(),
            "trace_id": str(self.context.trace_id),
            "parent_id": str(self.context.parent_id) if self.context.parent_id else None,
            "source": self.context.source_component,
            "correlation_id": self.context.correlation_id,
            "payload": self.payload,
            "metadata": self.context.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """Deserialize event from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            event_type=EventType[data["event_type"]],
            priority=EventPriority[data["priority"]],
            payload=data["payload"],
            context=EventContext(
                trace_id=UUID(data["trace_id"]),
                parent_id=UUID(data["parent_id"]) if data["parent_id"] else None,
                source_component=data["source"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                correlation_id=data.get("correlation_id"),
                metadata=data.get("metadata", {}),
            ),
        )
    
    def with_payload(self, **updates: Any) -> Event:
        """Create a new event with updated payload (for transformations)."""
        new_payload = {**self.payload, **updates}
        return Event(
            event_type=self.event_type,
            payload=new_payload,
            context=self.context,
            priority=self.priority,
        )


# Convenience factory functions for common events

def create_market_data_event(
    symbol: str,
    price: float,
    volume: float,
    timestamp: datetime,
    source: str = "market_data",
) -> Event:
    """Factory for market data events."""
    return Event(
        event_type=EventType.MARKET_DATA_RECEIVED,
        payload={
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": timestamp.isoformat(),
        },
        context=EventContext(
            trace_id=uuid4(),
            source_component=source,
        ),
        priority=EventPriority.HIGH,
    )


def create_signal_event(
    symbol: str,
    direction: str,
    confidence: float,
    strategy: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Event:
    """Factory for signal generation events."""
    return Event(
        event_type=EventType.SIGNAL_GENERATED,
        payload={
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "strategy": strategy,
            "metadata": metadata or {},
        },
        context=EventContext(
            trace_id=uuid4(),
            source_component="signal_engine",
        ),
        priority=EventPriority.NORMAL,
    )


def create_risk_event(
    check_type: str,
    passed: bool,
    details: Dict[str, Any],
    source: str = "risk_engine",
) -> Event:
    """Factory for risk check events."""
    event_type = EventType.RISK_CHECK_PASSED if passed else EventType.RISK_CHECK_FAILED
    return Event(
        event_type=event_type,
        payload={
            "check_type": check_type,
            "passed": passed,
            "details": details,
        },
        context=EventContext(
            trace_id=uuid4(),
            source_component=source,
        ),
        priority=EventPriority.CRITICAL,
    )


def create_kill_switch_event(
    reason: str,
    triggered_by: str,
    severity: str = "CRITICAL",
) -> Event:
    """Factory for kill switch events - highest priority."""
    return Event(
        event_type=EventType.KILL_SWITCH_TRIGGERED,
        payload={
            "reason": reason,
            "triggered_by": triggered_by,
            "severity": severity,
        },
        context=EventContext(
            trace_id=uuid4(),
            source_component="risk_engine",
        ),
        priority=EventPriority.CRITICAL,
    )
