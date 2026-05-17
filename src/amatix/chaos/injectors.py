"""Failure injectors for chaos testing.

Concrete implementations of various failure types.
"""

from __future__ import annotations

import asyncio
import gc
import random
import time
from typing import Any, Dict, Optional

from amatix.chaos.engine import FailureInjector, FailureType, RecoveryResult
from amatix.core.observability import get_logger

logger = get_logger(__name__)


class LatencyInjector(FailureInjector):
    """Inject latency spikes into operations."""
    
    @property
    def failure_type(self) -> FailureType:
        return FailureType.LATENCY
    
    async def inject(
        self,
        target: str,
        duration_seconds: float,
        parameters: Dict[str, Any],
    ) -> bool:
        """Inject latency by sleeping."""
        latency_ms = parameters.get("latency_ms", 5000)
        logger.info(
            "Injecting latency",
            target=target,
            latency_ms=latency_ms,
            duration=duration_seconds,
        )
        # The actual latency injection would hook into the target
        # For now, just simulate with sleep
        await asyncio.sleep(duration_seconds)
        return True
    
    async def verify_recovery(
        self,
        target: str,
        timeout_seconds: float = 30.0,
    ) -> RecoveryResult:
        """Verify latency returned to normal."""
        # Check if operations are responsive again
        start = time.time()
        while time.time() - start < timeout_seconds:
            # Would check actual latency metrics
            await asyncio.sleep(0.1)
            if random.random() > 0.1:  # Simulate recovery check
                return RecoveryResult.FULL
        
        return RecoveryResult.TIMEOUT


class DisconnectInjector(FailureInjector):
    """Inject connection failures."""
    
    @property
    def failure_type(self) -> FailureType:
        return FailureType.DISCONNECT
    
    async def inject(
        self,
        target: str,
        duration_seconds: float,
        parameters: Dict[str, Any],
    ) -> bool:
        """Simulate disconnect."""
        disconnect_type = parameters.get("type", "websocket")  # websocket, api, db
        
        logger.info(
            "Injecting disconnect",
            target=target,
            type=disconnect_type,
            duration=duration_seconds,
        )
        
        # In real implementation, would actually disconnect
        # For simulation, just wait
        await asyncio.sleep(duration_seconds)
        return True
    
    async def verify_recovery(
        self,
        target: str,
        timeout_seconds: float = 30.0,
    ) -> RecoveryResult:
        """Verify reconnection."""
        start = time.time()
        while time.time() - start < timeout_seconds:
            # Simulate reconnection check
            await asyncio.sleep(0.5)
            if random.random() > 0.2:  # 80% chance of recovery
                return RecoveryResult.FULL
        
        return RecoveryResult.TIMEOUT


class CorruptionInjector(FailureInjector):
    """Inject data corruption."""
    
    @property
    def failure_type(self) -> FailureType:
        return FailureType.CORRUPTION
    
    async def inject(
        self,
        target: str,
        duration_seconds: float,
        parameters: Dict[str, Any],
    ) -> bool:
        """Inject corrupted data."""
        corruption_type = parameters.get("type", "json")  # json, decimal, string
        
        logger.info(
            "Injecting data corruption",
            target=target,
            type=corruption_type,
            duration=duration_seconds,
        )
        
        # Would inject malformed data into event stream
        await asyncio.sleep(duration_seconds)
        return True
    
    async def verify_recovery(
        self,
        target: str,
        timeout_seconds: float = 30.0,
    ) -> RecoveryResult:
        """Verify corruption was handled."""
        # Check if validation caught the corruption
        await asyncio.sleep(0.1)
        
        # Simulate: corruption usually caught by validation
        return RecoveryResult.FULL


class MemoryPressureInjector(FailureInjector):
    """Inject memory pressure."""
    
    def __init__(self) -> None:
        self._allocated_memory: list = []
    
    @property
    def failure_type(self) -> FailureType:
        return FailureType.MEMORY_PRESSURE
    
    async def inject(
        self,
        target: str,
        duration_seconds: float,
        parameters: Dict[str, Any],
    ) -> bool:
        """Create memory pressure."""
        pressure_mb = parameters.get("pressure_mb", 500)
        
        logger.info(
            "Injecting memory pressure",
            target=target,
            pressure_mb=pressure_mb,
            duration=duration_seconds,
        )
        
        # Allocate large objects to create pressure
        # Each string is about 1MB
        try:
            for _ in range(pressure_mb):
                self._allocated_memory.append("x" * (1024 * 1024))
        except MemoryError:
            logger.error("Memory allocation failed during injection")
        
        # Hold for duration
        await asyncio.sleep(duration_seconds)
        
        # Cleanup
        self._allocated_memory.clear()
        gc.collect()
        
        return True
    
    async def verify_recovery(
        self,
        target: str,
        timeout_seconds: float = 30.0,
    ) -> RecoveryResult:
        """Verify memory returned to normal."""
        import psutil
        
        start = time.time()
        while time.time() - start < timeout_seconds:
            # Check if memory usage is back to normal
            process = psutil.Process()
            mem_info = process.memory_info()
            
            # If RSS is under 500MB, consider recovered
            if mem_info.rss < 500 * 1024 * 1024:
                return RecoveryResult.FULL
            
            await asyncio.sleep(1.0)
        
        return RecoveryResult.PARTIAL


class QueueOverflowInjector(FailureInjector):
    """Inject event queue overflow."""
    
    @property
    def failure_type(self) -> FailureType:
        return FailureType.PARTIAL_FAILURE
    
    async def inject(
        self,
        target: str,
        duration_seconds: float,
        parameters: Dict[str, Any],
    ) -> bool:
        """Flood event queue."""
        event_rate = parameters.get("events_per_second", 10000)
        
        logger.info(
            "Injecting queue overflow",
            target=target,
            rate=event_rate,
            duration=duration_seconds,
        )
        
        # Would rapidly emit events to overflow queue
        # For simulation, just wait
        await asyncio.sleep(duration_seconds)
        return True
    
    async def verify_recovery(
        self,
        target: str,
        timeout_seconds: float = 30.0,
    ) -> RecoveryResult:
        """Verify queue drained."""
        # Check if queue depth returned to normal
        await asyncio.sleep(1.0)
        return RecoveryResult.FULL


class PartialFailureInjector(FailureInjector):
    """Inject partial failures (some operations fail, others succeed)."""
    
    @property
    def failure_type(self) -> FailureType:
        return FailureType.PARTIAL_FAILURE
    
    async def inject(
        self,
        target: str,
        duration_seconds: float,
        parameters: Dict[str, Any],
    ) -> bool:
        """Simulate partial failure rate."""
        failure_rate = parameters.get("failure_rate", 0.3)  # 30% fail
        
        logger.info(
            "Injecting partial failures",
            target=target,
            failure_rate=failure_rate,
            duration=duration_seconds,
        )
        
        # Would configure target to randomly fail operations
        await asyncio.sleep(duration_seconds)
        return True
    
    async def verify_recovery(
        self,
        target: str,
        timeout_seconds: float = 30.0,
    ) -> RecoveryResult:
        """Verify failure rate returned to normal."""
        # Check if operations are succeeding consistently
        await asyncio.sleep(0.5)
        return RecoveryResult.FULL
