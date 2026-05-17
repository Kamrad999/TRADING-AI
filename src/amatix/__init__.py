"""AMATIS: Adaptive Multi-Agent Trading Intelligence System.

AMATIS is an institutional-grade, event-driven, modular trading intelligence
platform designed for evolution into autonomous multi-agent systems.

Architecture Principles:
    - Event-driven: All significant actions emit events for audit/replay
    - Modular: Clean interfaces enable component swapping
    - Observable: Everything is logged, measured, and traceable
    - Risk-first: Risk system has veto authority over all operations
    - Explainable: All decisions are auditable and traceable

Example:
    Basic usage of the core event bus:

    >>> from amatix.core import EventBus
    >>> bus = EventBus()
    >>> @bus.on("signal.generated")
    ... def handle_signal(event):
    ...     print(f"Signal: {event.payload}")
"""

__version__ = "0.1.0"
__author__ = "AMATIS Engineering Team"

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.config import AmatixConfig

__all__ = ["HardenedEventBusV2", "AmatixConfig"]
