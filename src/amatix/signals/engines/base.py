"""Base class for AMATIS signal engines.

Provides common functionality for signal generation engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.observability import get_logger
from amatix.interfaces import SignalEngine as ISignalEngine
from amatix.signals.models import Signal

logger = get_logger(__name__)


class BaseSignalEngine(ISignalEngine, ABC):
    """Base class for signal engines.
    
    Provides:
        - Event bus access
        - Configuration management
        - Metrics collection
    """
    
    def __init__(
        self,
        name: str,
        version: str,
        event_bus: HardenedEventBusV2,
    ) -> None:
        """Initialize base engine.
        
        Args:
            name: Engine identifier
            version: Engine version
            event_bus: Event bus for communication
        """
        self._name = name
        self._version = version
        self._event_bus = event_bus
        self._config: Dict[str, Any] = {}
        self._initialized = False
        self._signals_generated = 0
    
    @property
    def name(self) -> str:
        """Engine name."""
        return self._name
    
    @property
    def version(self) -> str:
        """Engine version."""
        return self._version
    
    @property
    def supported_asset_classes(self) -> List[str]:
        """Asset classes supported by this engine."""
        return ["equity", "crypto", "forex"]
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize engine with configuration."""
        self._config = config
        self._initialized = True
        logger.info(
            "Signal engine initialized",
            engine=self._name,
            version=self._version,
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Default health check."""
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "name": self._name,
            "version": self._version,
            "signals_generated": self._signals_generated,
        }
    
    async def shutdown(self) -> None:
        """Default shutdown."""
        self._initialized = False
        logger.info("Signal engine shutdown", engine=self._name)
    
    @abstractmethod
    async def generate(self, context: Dict[str, Any]) -> List[Signal]:
        """Generate signals - must be implemented by subclass."""
        pass
    
    def _track_signal(self) -> None:
        """Track signal generation."""
        self._signals_generated += 1
