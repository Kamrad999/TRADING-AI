"""Circuit breaker pattern implementation for resilience.

Circuit breakers prevent cascading failures by stopping requests to
failing services. They have three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail fast
    - HALF_OPEN: Testing if service recovered

Design based on:
    - Microsoft Azure patterns
    - Hystrix (Netflix)
    - Resilience4j
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union

from amatix.core.observability import get_logger, get_metrics

logger = get_logger(__name__)
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject fast
    HALF_OPEN = auto()   # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    failure_threshold: int = 5
    """Number of failures before opening circuit."""
    
    success_threshold: int = 3
    """Number of successes in HALF_OPEN to close circuit."""
    
    timeout_seconds: float = 60.0
    """Seconds before attempting recovery (HALF_OPEN)."""
    
    half_open_max_calls: int = 3
    """Max calls allowed in HALF_OPEN state."""
    
    excluded_exceptions: tuple = (Exception,)
    """Exceptions that count toward failure (default: all)."""


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    
    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    total_calls: int = 0
    rejected_calls: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is OPEN."""
    
    def __init__(self, name: str, last_error: Optional[str] = None):
        self.name = name
        self.last_error = last_error
        super().__init__(f"Circuit breaker '{name}' is OPEN" + 
                        (f": {last_error}" if last_error else ""))


class CircuitBreaker:
    """Circuit breaker for resilient service calls.
    
    Wraps external service calls (brokers, data feeds) to prevent
    cascading failures during outages.
    
    Example:
        >>> breaker = CircuitBreaker("alpaca", CircuitBreakerConfig())
        >>> 
        >>> @breaker
        ... async def call_api():
        ...     return await broker.get_positions()
        >>> 
        >>> try:
        ...     result = await call_api()
        ... except CircuitBreakerOpenError:
        ...     # Fail fast - use cached data
        ...     result = cached_positions
    
    Attributes:
        name: Identifier for this circuit breaker
        config: Configuration parameters
        _state: Current circuit state
        _lock: Async lock for state transitions
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats(state=CircuitState.CLOSED)
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
        logger.debug(
            "Circuit breaker created",
            name=name,
            failure_threshold=self.config.failure_threshold,
        )
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state
    
    @property
    def stats(self) -> CircuitBreakerStats:
        """Current statistics (read-only copy)."""
        return CircuitBreakerStats(**self._stats.__dict__)
    
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN
    
    async def call(
        self,
        fn: Callable[[], Coroutine[Any, Any, T]],
        fallback: Optional[Callable[[], T]] = None,
    ) -> T:
        """Execute function with circuit breaker protection.
        
        Args:
            fn: Async function to execute
            fallback: Optional fallback function if circuit is open
        
        Returns:
            Result from fn or fallback
        
        Raises:
            CircuitBreakerOpenError: If circuit is open and no fallback
            Exception: Any exception from fn (if circuit closed)
        """
        async with self._lock:
            await self._transition_state()
            
            if self._state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                get_metrics().counter("circuit_breaker_rejected", labels={"name": self.name})
                
                if fallback:
                    logger.debug("Circuit open, using fallback", name=self.name)
                    return fallback()
                
                raise CircuitBreakerOpenError(
                    self.name,
                    str(self._stats.last_failure_time),
                )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._stats.rejected_calls += 1
                    raise CircuitBreakerOpenError(self.name, "Half-open call limit reached")
                self._half_open_calls += 1
        
        # Execute the call (outside lock)
        return await self._execute(fn)
    
    async def _execute(self, fn: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Execute the wrapped function and track result."""
        self._stats.total_calls += 1
        start_time = time.time()
        
        try:
            result = await fn()
            await self._on_success()
            
            get_metrics().histogram(
                "circuit_breaker_call_duration",
                time.time() - start_time,
                labels={"name": self.name, "result": "success"},
            )
            
            return result
            
        except Exception as e:
            await self._on_failure(e)
            
            get_metrics().histogram(
                "circuit_breaker_call_duration",
                time.time() - start_time,
                labels={"name": self.name, "result": "failure"},
            )
            
            raise
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self._stats.success_count += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Check if we can close the circuit
                if self._stats.success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    logger.info("Circuit breaker closed", name=self.name)
    
    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self._stats.failure_count += 1
            self._stats.last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed during test - open again
                await self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker opened (half-open failure)",
                    name=self.name,
                    error=str(error),
                )
            
            elif self._state == CircuitState.CLOSED:
                # Check if we should open
                if self._stats.failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        "Circuit breaker opened",
                        name=self.name,
                        failure_count=self._stats.failure_count,
                        error=str(error),
                    )
    
    async def _transition_state(self) -> None:
        """Check and transition state if needed."""
        if self._state == CircuitState.OPEN:
            # Check if timeout elapsed for recovery attempt
            if self._stats.last_failure_time:
                elapsed = time.time() - self._stats.last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    await self._transition_to(CircuitState.HALF_OPEN)
                    logger.info(
                        "Circuit breaker half-open (testing recovery)",
                        name=self.name,
                    )
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state = new_state
        
        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._stats.failure_count = 0
            self._stats.success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._stats.success_count = 0
            self._half_open_calls = 0
        
        get_metrics().counter(
            "circuit_breaker_state_change",
            labels={
                "name": self.name,
                "from": old_state.name,
                "to": new_state.name,
            },
        )
    
    def manual_reset(self) -> None:
        """Manually reset circuit to CLOSED state.
        
        Use with caution - only when service recovery is confirmed.
        """
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats(state=CircuitState.CLOSED)
        self._half_open_calls = 0
        logger.info("Circuit breaker manually reset", name=self.name)
    
    def __call__(
        self,
        fn: Callable[[], Coroutine[Any, Any, T]],
    ) -> Callable[[], Coroutine[Any, Any, T]]:
        """Decorator syntax for circuit breaker."""
        async def wrapper() -> T:
            return await self.call(fn)
        return wrapper


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def register(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Register a new circuit breaker."""
        if name in self._breakers:
            raise ValueError(f"Circuit breaker '{name}' already registered")
        
        breaker = CircuitBreaker(name, config)
        self._breakers[name] = breaker
        return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            return self.register(name, config)
        return self._breakers[name]
    
    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get stats for all circuit breakers."""
        return {name: breaker.stats for name, breaker in self._breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED."""
        for breaker in self._breakers.values():
            breaker.manual_reset()


# Global registry
_global_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry
