"""AMATIS Core - Operating System Foundation.

This module contains the foundational infrastructure for AMATIS:
    - Event system for decoupled communication
    - Configuration management
    - Structured logging
    - Circuit breakers for resilience
"""

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import Event, EventType, EventPriority
from amatix.core.config import AmatixConfig, Settings
from amatix.core.circuit_breaker import CircuitBreaker
from amatix.core.observability import get_logger, MetricsCollector

__all__ = [
    "HardenedEventBusV2",
    "Event",
    "EventType",
    "EventPriority",
    "AmatixConfig",
    "Settings",
    "CircuitBreaker",
    "get_logger",
    "MetricsCollector",
]
