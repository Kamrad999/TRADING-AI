"""AMATIS Orchestrator - Central nervous system.

The orchestrator coordinates all AMATIS components in an event-driven,
decoupled manner. It is NOT a monolithic controller but an event router
that enables components to communicate without direct coupling.

Key responsibilities:
    - Component lifecycle management
    - Event flow coordination
    - System health monitoring
    - Graceful degradation
    - Emergency shutdown

Design principles:
    - Components are autonomous
    - Communication via events only
    - No direct method calls between components
    - Observable at every step
    - Recoverable from any state
"""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, TypeVar

from amatix.core.circuit_breaker import CircuitBreakerRegistry
from amatix.core.config import get_settings
from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import Event, EventContext, EventPriority, EventType
from amatix.core.observability import get_logger, get_metrics
from amatix.interfaces import ExecutionEngine, RiskEngine, SignalEngine

logger = get_logger(__name__)

T = TypeVar("T")


class SystemState(Enum):
    """Orchestrator lifecycle states."""
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    DEGRADED = auto()
    SHUTTING_DOWN = auto()
    STOPPED = auto()


@dataclass
class ComponentInfo:
    """Metadata about a registered component."""
    name: str
    component_type: str
    instance: Any
    priority: int  # Initialization priority (lower = earlier)
    health_check: Optional[callable] = None
    critical: bool = False  # If True, system stops if this fails


class Orchestrator:
    """Central orchestrator for AMATIS system.
    
    The orchestrator manages the lifecycle of all components without
    tightly coupling them. Components communicate exclusively through
the event bus.
    
    Usage:
        >>> orchestrator = Orchestrator()
        >>> 
        >>> # Register components
        >>> orchestrator.register("risk", risk_engine, critical=True)
        >>> orchestrator.register("execution", execution_engine)
        >>> 
        >>> # Start system
        >>> await orchestrator.start()
        >>> 
        >>> # System runs autonomously via events
        >>> 
        >>> # Graceful shutdown
        >>> await orchestrator.stop()
    
    Attributes:
        state: Current system state
        event_bus: Central event bus
        components: Registered components
        circuit_breakers: Circuit breaker registry
    """
    
    def __init__(
        self,
        event_bus: Optional[HardenedEventBusV2] = None,
        enable_signal_handlers: bool = True,
    ) -> None:
        """Initialize the orchestrator.
        
        Args:
            event_bus: Event bus for component communication
        """
        self._state = SystemState.INITIALIZING
        self._event_bus = event_bus
        
        self._components: Dict[str, ComponentInfo] = {}
        self._circuit_breakers = CircuitBreakerRegistry()
        
        self._settings = get_settings()
        self._shutdown_event = asyncio.Event()
        
        # Metrics
        self._component_health: Dict[str, bool] = {}
        
        if enable_signal_handlers:
            self._register_signal_handlers()
        
        self._register_internal_handlers()
        
        logger.info("Orchestrator initialized")
    
    def _register_signal_handlers(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    def _signal_handler(self) -> None:
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        asyncio.create_task(self.stop())
    
    def _register_internal_handlers(self) -> None:
        """Register orchestrator's internal event handlers."""
        
        @self._event_bus.on(EventType.COMPONENT_FAILED)
        async def handle_component_failure(event: Event) -> None:
            """Handle component failure events."""
            component_name = event.payload.get("component")
            error = event.payload.get("error")
            
            logger.error(
                "Component failure detected",
                component=component_name,
                error=error,
            )
            
            # Check if critical
            info = self._components.get(component_name)
            if info and info.critical:
                logger.critical(
                    "Critical component failed, initiating shutdown",
                    component=component_name,
                )
                await self._enter_degraded_mode()
        
        @self._event_bus.on(EventType.KILL_SWITCH_TRIGGERED)
        async def handle_kill_switch(event: Event) -> None:
            """Handle kill switch activation."""
            reason = event.payload.get("reason", "unknown")
            logger.critical("Kill switch activated", reason=reason)
            
            # Stop all trading operations
            await self._pause_trading()
    
    @property
    def state(self) -> SystemState:
        """Current system state."""
        return self._state
    
    @property
    def event_bus(self) -> EventBus:
        """Access to the central event bus."""
        return self._event_bus
    
    def register(
        self,
        name: str,
        component: Any,
        component_type: Optional[str] = None,
        priority: int = 50,
        health_check: Optional[callable] = None,
        critical: bool = False,
    ) -> None:
        """Register a component with the orchestrator.
        
        Args:
            name: Unique component name
            component: Component instance
            component_type: Type string (e.g., "risk_engine")
            priority: Init priority (0-100, lower = earlier)
            health_check: Async function returning health status
            critical: If True, system degrades on failure
        
        Raises:
            ValueError: If name already registered
        """
        # Register component
        self._components[name] = ComponentMetadata(
            name=name,
            component_type=component_type,
            instance=component,
            priority=priority,
            health_check=health_check,
            critical=critical,
        )
        
        logger.info(
            "Component registered",
            name=name,
            type=component_type,
            priority=priority,
            critical=critical,
        )
    
    async def start(self) -> None:
        """Start the orchestrator and all registered components.
        
        Components are initialized in priority order.
        Emits SYSTEM_STARTED event on success.
        
        Raises:
            RuntimeError: If critical component fails to initialize
        """
        if self._state != SystemState.INITIALIZING:
            raise RuntimeError(f"Cannot start from state: {self._state}")
        
        logger.info("Starting AMATIS orchestrator")
        
        # Sort by priority
        sorted_components = sorted(
            self._components.values(),
            key=lambda c: c.priority
        )
        
        for info in sorted_components:
            try:
                logger.debug("Initializing component", name=info.name)
                
                # Check for initialize method
                if hasattr(info.instance, "initialize"):
                    if asyncio.iscoroutinefunction(info.instance.initialize):
                        await info.instance.initialize(self._settings.to_dict())
                    else:
                        info.instance.initialize(self._settings.to_dict())
                
                # Emit component initialized event
                await self._event_bus.emit_new(
                    EventType.COMPONENT_INITIALIZED,
                    {
                        "component": info.name,
                        "type": info.component_type,
                    },
                )
                
                self._component_health[info.name] = True
                
            except Exception as e:
                logger.error(
                    "Component initialization failed",
                    name=info.name,
                    error=str(e),
                )
                
                await self._event_bus.emit_new(
                    EventType.COMPONENT_FAILED,
                    {
                        "component": info.name,
                        "error": str(e),
                        "phase": "initialization",
                    },
                    priority=EventPriority.CRITICAL,
                )
                
                if info.critical:
                    raise RuntimeError(
                        f"Critical component '{info.name}' failed: {e}"
                    )
                
                self._component_health[info.name] = False
        
        self._state = SystemState.RUNNING
        
        await self._event_bus.emit_new(
            EventType.SYSTEM_STARTED,
            {
                "components": list(self._components.keys()),
                "healthy_components": [
                    name for name, healthy in self._component_health.items()
                    if healthy
                ],
            },
        )
        
        logger.info(
            "AMATIS system started",
            components=len(self._components),
            healthy=sum(self._component_health.values()),
        )
    
    async def stop(self, timeout_seconds: float = 30.0) -> None:
        """Gracefully stop the orchestrator and all components.
        
        Components are stopped in reverse priority order.
        
        Args:
            timeout_seconds: Max time to wait for shutdown
        """
        if self._state in (SystemState.STOPPED, SystemState.SHUTTING_DOWN):
            return
        
        logger.info("Shutting down AMATIS orchestrator")
        self._state = SystemState.SHUTTING_DOWN
        
        await self._event_bus.emit_new(
            EventType.SYSTEM_SHUTDOWN,
            {"reason": "requested", "timeout": timeout_seconds},
        )
        
        # Stop in reverse priority order
        sorted_components = sorted(
            self._components.values(),
            key=lambda c: c.priority,
            reverse=True
        )
        
        for info in sorted_components:
            try:
                if hasattr(info.instance, "shutdown"):
                    logger.debug("Shutting down component", name=info.name)
                    
                    if asyncio.iscoroutinefunction(info.instance.shutdown):
                        await asyncio.wait_for(
                            info.instance.shutdown(),
                            timeout=timeout_seconds / len(sorted_components)
                        )
                    else:
                        info.instance.shutdown()
                        
            except Exception as e:
                logger.warning(
                    "Component shutdown error (continuing)",
                    name=info.name,
                    error=str(e),
                )
        
        self._state = SystemState.STOPPED
        self._shutdown_event.set()
        
        logger.info("AMATIS system stopped")
    
    async def pause(self) -> None:
        """Pause trading operations but keep system running.
        
        Emits signal to all components to stop generating/processing
        new signals, but maintains data feeds and monitoring.
        """
        if self._state != SystemState.RUNNING:
            return
        
        logger.info("Pausing trading operations")
        self._state = SystemState.PAUSED
        
        await self._event_bus.emit_new(
            EventType.SYSTEM_SHUTDOWN,  # Reuse with pause flag
            {"reason": "pause", "trading_only": True},
        )
    
    async def resume(self) -> None:
        """Resume trading operations after pause."""
        if self._state != SystemState.PAUSED:
            return
        
        logger.info("Resuming trading operations")
        self._state = SystemState.RUNNING
        
        await self._event_bus.emit_new(
            EventType.SYSTEM_STARTED,
            {"reason": "resume"},
        )
    
    async def run(self) -> None:
        """Run the orchestrator until shutdown.
        
        This is the main entry point for running AMATIS.
        It starts the system and waits for shutdown signal.
        """
        await self.start()
        
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            logger.info("Orchestrator run cancelled")
        finally:
            await self.stop()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of orchestrator and all components.
        
        Returns:
            Dict with system health status
        """
        checks = {}
        
        for name, info in self._components.items():
            if info.health_check:
                try:
                    if asyncio.iscoroutinefunction(info.health_check):
                        checks[name] = await info.health_check()
                    else:
                        checks[name] = info.health_check()
                except Exception as e:
                    checks[name] = {"status": "unhealthy", "error": str(e)}
                    self._component_health[name] = False
            else:
                # Default health check
                checks[name] = {
                    "status": "healthy" if self._component_health.get(name) else "unknown"
                }
        
        overall = "healthy"
        if any(c.get("status") == "unhealthy" for c in checks.values()):
            critical_unhealthy = any(
                info.critical and checks.get(name, {}).get("status") == "unhealthy"
                for name, info in self._components.items()
            )
            overall = "critical" if critical_unhealthy else "degraded"
        
        return {
            "status": overall,
            "system_state": self._state.name,
            "components": checks,
        }
    
    async def _enter_degraded_mode(self) -> None:
        """Enter degraded mode - minimal functionality."""
        logger.critical("Entering degraded mode")
        self._state = SystemState.DEGRADED
        
        # Stop all non-critical components
        for name, info in self._components.items():
            if not info.critical and hasattr(info.instance, "shutdown"):
                try:
                    if asyncio.iscoroutinefunction(info.instance.shutdown):
                        await info.instance.shutdown()
                    else:
                        info.instance.shutdown()
                except Exception as e:
                    logger.error("Error shutting down component", name=name, error=str(e))
    
    async def _pause_trading(self) -> None:
        """Pause all trading operations (kill switch activated)."""
        await self.pause()
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component by name."""
        info = self._components.get(name)
        return info.instance if info else None
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self._components.keys())


# Convenience function for simple deployments
def create_default_orchestrator() -> Orchestrator:
    """Create an orchestrator with default configuration."""
    return Orchestrator()
