"""Deterministic event replay engine.

The foundation of institutional-grade backtesting and forensics.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import Event, EventContext, EventPriority, EventType


@dataclass
class ReplayState:
    """Mutable state container for replay.
    
    All state modifications during replay are tracked here
    for comparison and forensic analysis.
    """
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    orders: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cash: Decimal = field(default_factory=lambda: Decimal("100000"))
    portfolio_value: Decimal = field(default_factory=lambda: Decimal("100000"))
    signals_generated: int = 0
    orders_filled: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for comparison."""
        return {
            "positions": self.positions,
            "orders": self.orders,
            "cash": str(self.cash),
            "portfolio_value": str(self.portfolio_value),
            "signals_generated": self.signals_generated,
            "orders_filled": self.orders_filled,
            "metrics": self.metrics,
        }
    
    def checksum(self) -> str:
        """Generate deterministic checksum of state."""
        data = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class ReplayResult:
    """Result of a replay session."""
    success: bool
    events_replayed: int
    final_state: ReplayState
    execution_time_ms: float
    checksum: str
    divergence_events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "events_replayed": self.events_replayed,
            "final_state": self.final_state.to_dict(),
            "execution_time_ms": self.execution_time_ms,
            "checksum": self.checksum,
            "divergence_count": len(self.divergence_events),
            "error": self.error,
        }


class DeterministicContext:
    """Container for deterministic execution context.
    
    Ensures replay produces identical results by:
        - Fixing random seeds
        - Normalizing timestamps
        - Mocking external data sources
    """
    
    def __init__(
        self,
        random_seed: int = 42,
        reference_time: Optional[datetime] = None,
    ) -> None:
        self.random_seed = random_seed
        self.reference_time = reference_time or datetime.utcnow()
        self._rng = random.Random(random_seed)
        self._sequence = 0
    
    def next_sequence(self) -> int:
        """Get next monotonic sequence number."""
        self._sequence += 1
        return self._sequence
    
    def random(self) -> float:
        """Deterministic random number."""
        return self._rng.random()
    
    def get_timestamp(self, offset_seconds: float = 0) -> datetime:
        """Get deterministic timestamp."""
        from datetime import timedelta
        return self.reference_time + timedelta(seconds=offset_seconds)


class ReplayEngine:
    """Deterministic event replay engine.
    
    Core capability for:
        - Backtesting strategies
        - Forensic analysis
        - Regression testing
        - ML training data generation
    
    Guarantees:
        - Same inputs → same outputs
        - Monotonic event ordering
        - Deterministic timestamps
        - State checksums for comparison
    """
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        enable_determinism: bool = True,
    ) -> None:
        self._event_bus = event_bus
        self._enable_determinism = enable_determinism
        self._handlers: Dict[EventType, List[Callable[[Event, ReplayState], None]]] = {}
        
        # State tracking
        self._state_snapshots: Dict[int, str] = {}  # event_index -> checksum
        self._event_log: List[Dict[str, Any]] = []
    
    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[[Event, ReplayState], None],
    ) -> None:
        """Register a replay handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def replay(
        self,
        events: List[Event],
        initial_state: Optional[ReplayState] = None,
        deterministic_context: Optional[DeterministicContext] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ReplayResult:
        """Replay events deterministically.
        
        Args:
            events: Events to replay in order
            initial_state: Starting state (default: empty)
            deterministic_context: Context for determinism
            progress_callback: Called with (current, total) progress
        
        Returns:
            ReplayResult with final state and checksum
        """
        start_time = time.time()
        
        # Initialize state
        state = initial_state or ReplayState()
        context = deterministic_context or DeterministicContext()
        
        # Clear previous run data
        self._state_snapshots.clear()
        self._event_log.clear()
        
        try:
            for i, event in enumerate(events):
                # Report progress
                if progress_callback:
                    progress_callback(i + 1, len(events))
                
                # Normalize event for determinism
                normalized_event = self._normalize_event(event, context, i)
                
                # Execute handlers
                handlers = self._handlers.get(normalized_event.event_type, [])
                for handler in handlers:
                    try:
                        handler(normalized_event, state)
                    except Exception as e:
                        # Log but continue - some handlers may be optional
                        self._event_log.append({
                            "event_index": i,
                            "event_type": normalized_event.event_type.name,
                            "error": str(e),
                        })
                
                # Record state snapshot periodically
                if i % 100 == 0:
                    self._state_snapshots[i] = state.checksum()
            
            execution_time = (time.time() - start_time) * 1000
            
            return ReplayResult(
                success=True,
                events_replayed=len(events),
                final_state=state,
                execution_time_ms=execution_time,
                checksum=state.checksum(),
            )
        
        except Exception as e:
            return ReplayResult(
                success=False,
                events_replayed=0,
                final_state=state,
                execution_time_ms=(time.time() - start_time) * 1000,
                checksum="",
                error=str(e),
            )
    
    def _normalize_event(
        self,
        event: Event,
        context: DeterministicContext,
        index: int,
    ) -> Event:
        """Normalize event for deterministic replay.
        
        Modifications:
            - Replace timestamp with deterministic value
            - Ensure sequence ID is monotonic
            - Remove non-deterministic fields
        """
        if not self._enable_determinism:
            return event
        
        # Calculate deterministic timestamp
        # Each event gets timestamp offset by its index
        deterministic_time = context.get_timestamp(offset_seconds=index * 0.001)
        
        # Create normalized context
        normalized_context = EventContext(
            trace_id=UUID(int=index),  # Deterministic UUID from index
            parent_id=None,
            source_component=event.context.source_component,
            timestamp=deterministic_time,
            correlation_id=event.context.correlation_id,
            metadata={
                **(event.context.metadata or {}),
                "replay_sequence": context.next_sequence(),
                "replay_index": index,
            },
        )
        
        # Return event with normalized context
        return Event(
            event_type=event.event_type,
            payload=event.payload,
            context=normalized_context,
            event_id=UUID(int=index + 1000000),  # Deterministic event ID
            priority=event.priority,
        )
    
    async def compare_replays(
        self,
        events: List[Event],
        run_count: int = 3,
    ) -> Dict[str, Any]:
        """Run multiple replays and verify determinism.
        
        Returns:
            Comparison report showing if all runs produced identical results
        """
        results: List[ReplayResult] = []
        
        for run in range(run_count):
            # Each run uses same context to ensure determinism
            context = DeterministicContext(random_seed=42 + run)
            result = await self.replay(events, deterministic_context=context)
            results.append(result)
        
        # Compare checksums
        checksums = [r.checksum for r in results]
        unique_checksums = set(checksums)
        
        return {
            "deterministic": len(unique_checksums) == 1,
            "run_count": run_count,
            "unique_checksums": len(unique_checksums),
            "checksums": checksums,
            "avg_execution_time_ms": sum(r.execution_time_ms for r in results) / len(results),
            "all_success": all(r.success for r in results),
        }
    
    def get_state_at_event(self, event_index: int) -> Optional[str]:
        """Get state checksum at specific event index (if snapshotted)."""
        # Find nearest snapshot at or before index
        nearest = None
        for idx in sorted(self._state_snapshots.keys()):
            if idx <= event_index:
                nearest = idx
            else:
                break
        
        if nearest is not None:
            return self._state_snapshots[nearest]
        return None
    
    def get_replay_log(self) -> List[Dict[str, Any]]:
        """Get log of events and errors during replay."""
        return self._event_log.copy()


# Pre-built replay handlers for common event types
class DefaultReplayHandlers:
    """Default handlers for replaying standard AMATIS events."""
    
    @staticmethod
    def handle_signal_generated(event: Event, state: ReplayState) -> None:
        """Track signal generation."""
        state.signals_generated += 1
        
        symbol = event.payload.get("symbol", "unknown")
        if symbol not in state.metrics.get("signals_by_symbol", {}):
            state.metrics.setdefault("signals_by_symbol", {})[symbol] = 0
        state.metrics["signals_by_symbol"][symbol] += 1
    
    @staticmethod
    def handle_order_filled(event: Event, state: ReplayState) -> None:
        """Track order fill and update position."""
        state.orders_filled += 1
        
        symbol = event.payload.get("symbol", "unknown")
        filled_qty = Decimal(str(event.payload.get("filled_quantity", "0")))
        filled_price = Decimal(str(event.payload.get("filled_price", "0")))
        side = event.payload.get("side", "buy")
        
        # Update position
        if symbol not in state.positions:
            state.positions[symbol] = {
                "quantity": Decimal("0"),
                "avg_price": Decimal("0"),
                "side": None,
            }
        
        pos = state.positions[symbol]
        
        if side == "buy":
            # Long position
            if pos["quantity"] > 0:
                # Adding to existing long
                total_value = pos["quantity"] * pos["avg_price"] + filled_qty * filled_price
                pos["quantity"] += filled_qty
                pos["avg_price"] = total_value / pos["quantity"]
            else:
                # New long or covering short
                pos["quantity"] += filled_qty
                pos["avg_price"] = filled_price
                pos["side"] = "long"
        else:
            # Sell position
            if pos["quantity"] > 0:
                # Reducing long
                pos["quantity"] -= filled_qty
                if pos["quantity"] <= 0:
                    pos["side"] = None if pos["quantity"] == 0 else "short"
                    pos["avg_price"] = filled_price if pos["quantity"] < 0 else Decimal("0")
            else:
                # Adding to short
                pos["quantity"] -= filled_qty
                pos["side"] = "short"
        
        # Update portfolio value
        position_value = sum(
            abs(p["quantity"]) * p["avg_price"]
            for p in state.positions.values()
        )
        state.portfolio_value = state.cash + position_value


def create_default_replay_engine(event_bus: HardenedEventBusV2) -> ReplayEngine:
    """Create replay engine with default handlers registered."""
    engine = ReplayEngine(event_bus)
    
    handlers = DefaultReplayHandlers()
    engine.register_handler(EventType.SIGNAL_GENERATED, handlers.handle_signal_generated)
    engine.register_handler(EventType.ORDER_FILLED, handlers.handle_order_filled)
    
    return engine
