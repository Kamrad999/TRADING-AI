"""AMATIS Signal Infrastructure.

Signal generation, processing, and management for trading decisions.

Engines:
    - news_engine: News-based signal generation
    - momentum_engine: Technical/momentum signals
"""

from amatix.signals.models import Signal, SignalDirection, SignalStrength

__all__ = [
    "Signal",
    "SignalDirection",
    "SignalStrength",
]
