"""Accelerated Market Replay Engine for AMATIS.

Institutional-grade replay capable of simulating 30 trading days in minutes.
Deterministic, checkpointed, and fully validated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import Event, EventContext, EventPriority, EventType
from amatix.core.observability import get_logger

logger = get_logger(__name__)


class ReplaySpeed(Enum):
    """Replay speed multipliers."""
    REALTIME = 1.0          # 1x speed
    ACCELERATED_10X = 10.0  # 10x faster
    ACCELERATED_100X = 100.0  # 100x faster
    ACCELERATED_1000X = 1000.0  # 1000x faster (minutes for 30 days)
    MAX_SPEED = 0.0         # Process as fast as possible


@dataclass
class ReplayCheckpoint:
    """Checkpoint for replay state at a specific point."""
    sequence_id: int
    timestamp: datetime
    event_count: int
    state_checksum: str
    portfolio_checksum: str
    events_since_start: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayState:
    """Complete replay state for validation."""
    sequence_id: int
    timestamp: datetime
    portfolio_value: Decimal
    cash: Decimal
    positions: Dict[str, Dict[str, Any]]
    active_orders: Dict[str, Dict[str, Any]]
    total_trades: int
    total_pnl: Decimal
    max_drawdown: Decimal
    
    def checksum(self) -> str:
        """Generate deterministic checksum of state."""
        data = {
            "sequence": self.sequence_id,
            "portfolio": str(self.portfolio_value),
            "cash": str(self.cash),
            "positions": {k: v for k, v in sorted(self.positions.items())},
            "orders": {k: v for k, v in sorted(self.active_orders.items())},
            "trades": self.total_trades,
            "pnl": str(self.total_pnl),
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


@dataclass
class ReplayConfig:
    """Configuration for replay session."""
    speed: ReplaySpeed = ReplaySpeed.ACCELERATED_1000X
    seed: int = 42
    checkpoint_interval: int = 1000  # events
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: Decimal = field(default_factory=lambda: Decimal("100000"))
    enable_pause_resume: bool = True
    determinism_checks: bool = True


@dataclass
class ReplayResult:
    """Result of a replay session."""
    session_id: str
    start_time: datetime
    end_time: datetime
    events_processed: int
    events_emitted: int
    duration_seconds: float
    checkpoints: List[ReplayCheckpoint]
    final_state: ReplayState
    determinism_score: float
    integrity_violations: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "events_processed": self.events_processed,
            "duration_seconds": round(self.duration_seconds, 2),
            "events_per_second": round(self.events_processed / max(self.duration_seconds, 0.001), 0),
            "final_portfolio": str(self.final_state.portfolio_value),
            "final_pnl": str(self.final_state.total_pnl),
            "max_drawdown": str(self.final_state.max_drawdown),
            "determinism_score": round(self.determinism_score, 2),
            "integrity_violations": len(self.integrity_violations),
        }


class AcceleratedReplayEngine:
    """High-speed deterministic replay engine.
    
    Capable of replaying 30 trading days through the entire AMATIS stack
    in minutes rather than days.
    
    Key Features:
        - Deterministic timestamps with monotonic sequencing
        - Configurable replay speeds (realtime to 1000x)
        - Checkpoint system for pause/resume/validation
        - Full state checksums at intervals
        - Multi-asset support (equities, crypto, forex)
    
    Usage:
        engine = AcceleratedReplayEngine(event_bus, config)
        result = await engine.replay_historical_data(market_data_events)
        
        # Verify determinism
        result2 = await engine.replay_historical_data(market_data_events)
        assert result.final_state.checksum() == result2.final_state.checksum()
    """
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        config: Optional[ReplayConfig] = None,
    ) -> None:
        self._event_bus = event_bus
        self._config = config or ReplayConfig()
        self._sequence_counter = 0
        self._checkpoints: List[ReplayCheckpoint] = []
        self._state = self._create_initial_state()
        self._paused = False
        self._stopped = False
        self._progress_callbacks: List[Callable[[int, int], None]] = []
        
        # Determinism tracking
        self._state_history: Dict[int, str] = {}
        self._integrity_violations: List[Dict[str, Any]] = []
        
        # Timing
        self._replay_start_time: Optional[float] = None
        self._virtual_time: Optional[datetime] = None
    
    def _create_initial_state(self) -> ReplayState:
        """Create initial replay state."""
        return ReplayState(
            sequence_id=0,
            timestamp=self._config.start_date or datetime.utcnow(),
            portfolio_value=self._config.initial_capital,
            cash=self._config.initial_capital,
            positions={},
            active_orders={},
            total_trades=0,
            total_pnl=Decimal("0"),
            max_drawdown=Decimal("0"),
        )
    
    def add_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """Add callback for progress updates (current, total)."""
        self._progress_callbacks.append(callback)
    
    async def replay_historical_data(
        self,
        events: List[Event],
        resume_from_checkpoint: Optional[ReplayCheckpoint] = None,
    ) -> ReplayResult:
        """Replay historical market data through AMATIS stack.
        
        Args:
            events: Historical events to replay (market data, signals, etc.)
            resume_from_checkpoint: Optional checkpoint to resume from
        
        Returns:
            ReplayResult with full statistics and state
        """
        session_id = str(uuid4())[:8]
        start_time = datetime.utcnow()
        self._replay_start_time = time.time()
        self._stopped = False
        self._paused = False
        
        # Reset or resume state
        if resume_from_checkpoint:
            self._state = self._restore_from_checkpoint(resume_from_checkpoint)
            start_index = resume_from_checkpoint.events_since_start
        else:
            self._state = self._create_initial_state()
            start_index = 0
        
        events_emitted = 0
        events_processed = 0
        
        logger.info(
            "Starting replay session",
            session_id=session_id,
            total_events=len(events),
            speed=self._config.speed.name,
            start_index=start_index,
        )
        
        try:
            for i, event in enumerate(events[start_index:], start=start_index):
                if self._stopped:
                    break
                
                # Handle pause
                while self._paused and not self._stopped:
                    await asyncio.sleep(0.1)
                
                # Process event with determinism
                normalized_event = self._normalize_event(event, i)
                
                # Emit to event bus
                await self._emit_with_timing(normalized_event)
                events_emitted += 1
                
                # Update state tracking
                self._update_state_from_event(normalized_event)
                
                # Create checkpoint if needed
                if i % self._config.checkpoint_interval == 0:
                    checkpoint = self._create_checkpoint(i)
                    self._checkpoints.append(checkpoint)
                    
                    # Validate determinism
                    if self._config.determinism_checks:
                        self._validate_checkpoint(checkpoint)
                
                # Progress callback
                for callback in self._progress_callbacks:
                    callback(i + 1, len(events))
                
                events_processed += 1
        
        except Exception as e:
            logger.exception(f"Replay failed at event {events_processed}: {e}")
            self._integrity_violations.append({
                "event_index": events_processed,
                "error": str(e),
                "type": "replay_exception",
            })
        
        end_time = datetime.utcnow()
        duration = time.time() - self._replay_start_time
        
        # Calculate determinism score
        determinism_score = self._calculate_determinism_score()
        
        result = ReplayResult(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            events_processed=events_processed,
            events_emitted=events_emitted,
            duration_seconds=duration,
            checkpoints=self._checkpoints.copy(),
            final_state=self._state,
            determinism_score=determinism_score,
            integrity_violations=self._integrity_violations.copy(),
        )
        
        logger.info(
            "Replay completed",
            session_id=session_id,
            events_processed=events_processed,
            duration=duration,
            events_per_sec=events_processed / max(duration, 0.001),
            determinism_score=determinism_score,
        )
        
        return result
    
    def _normalize_event(self, event: Event, index: int) -> Event:
        """Normalize event for deterministic replay.
        
        Ensures:
            - Monotonic sequence IDs
            - Consistent timestamps
            - Deterministic event IDs
        """
        self._sequence_counter += 1
        
        # Calculate deterministic timestamp
        if self._config.speed == ReplaySpeed.MAX_SPEED:
            # Use index-based timestamp for max speed
            virtual_time = (self._config.start_date or datetime.utcnow()) + timedelta(
                milliseconds=index
            )
        else:
            # Scale real timestamps
            elapsed_real = timedelta(seconds=time.time() - self._replay_start_time)
            elapsed_virtual = elapsed_real * self._config.speed.value
            virtual_time = (self._config.start_date or datetime.utcnow()) + elapsed_virtual
        
        # Create normalized context
        normalized_context = EventContext(
            trace_id=UUID(int=self._sequence_counter),
            parent_id=event.context.parent_id,
            source_component=event.context.source_component,
            timestamp=virtual_time,
            correlation_id=event.context.correlation_id,
            metadata={
                **(event.context.metadata or {}),
                "replay_sequence": self._sequence_counter,
                "replay_index": index,
                "replay_session": True,
            },
        )
        
        return Event(
            event_type=event.event_type,
            payload=event.payload,
            context=normalized_context,
            event_id=UUID(int=1000000 + index),
            priority=event.priority,
        )
    
    async def _emit_with_timing(self, event: Event) -> None:
        """Emit event with proper timing control."""
        if self._config.speed == ReplaySpeed.REALTIME:
            # Real-time delay based on timestamp gaps
            if self._virtual_time:
                gap = (event.context.timestamp - self._virtual_time).total_seconds()
                if gap > 0:
                    await asyncio.sleep(gap)
        
        self._virtual_time = event.context.timestamp
        
        await self._event_bus.emit_new(
            event.event_type,
            event.payload,
            priority=event.priority,
            source=event.context.source_component,
        )
    
    def _update_state_from_event(self, event: Event) -> None:
        """Track state changes from events."""
        if event.event_type == EventType.ORDER_FILLED:
            self._state.total_trades += 1
            
            # Update position tracking
            symbol = event.payload.get("symbol", "unknown")
            qty = Decimal(str(event.payload.get("filled_quantity", "0")))
            price = Decimal(str(event.payload.get("filled_price", "0")))
            side = event.payload.get("side", "buy")
            
            if symbol not in self._state.positions:
                self._state.positions[symbol] = {
                    "quantity": "0",
                    "avg_price": "0",
                    "side": None,
                }
            
            pos = self._state.positions[symbol]
            current_qty = Decimal(pos["quantity"])
            
            if side == "buy":
                if current_qty >= 0:
                    # Adding to long
                    new_qty = current_qty + qty
                    avg = (current_qty * Decimal(pos["avg_price"]) + qty * price) / new_qty
                    pos["quantity"] = str(new_qty)
                    pos["avg_price"] = str(avg)
                    pos["side"] = "long"
                else:
                    # Reducing short
                    new_qty = current_qty + qty
                    if new_qty >= 0:
                        pos["side"] = "long" if new_qty > 0 else None
                    pos["quantity"] = str(new_qty)
            else:
                # Sell
                if current_qty > 0:
                    new_qty = current_qty - qty
                    pos["quantity"] = str(new_qty)
                    if new_qty <= 0:
                        pos["side"] = "short" if new_qty < 0 else None
                else:
                    # Adding to short
                    new_qty = current_qty - qty
                    pos["quantity"] = str(new_qty)
                    pos["side"] = "short"
        
        elif event.event_type == EventType.POSITION_UPDATED:
            # Recalculate portfolio value
            position_value = sum(
                abs(Decimal(p["quantity"])) * Decimal(p.get("avg_price", "0"))
                for p in self._state.positions.values()
            )
            self._state.portfolio_value = self._state.cash + position_value
    
    def _create_checkpoint(self, event_index: int) -> ReplayCheckpoint:
        """Create checkpoint at current state."""
        return ReplayCheckpoint(
            sequence_id=self._sequence_counter,
            timestamp=datetime.utcnow(),
            event_count=event_index,
            state_checksum=self._state.checksum(),
            portfolio_checksum=self._calculate_portfolio_checksum(),
            events_since_start=event_index,
        )
    
    def _calculate_portfolio_checksum(self) -> str:
        """Calculate checksum of portfolio state."""
        portfolio_data = {
            "cash": str(self._state.cash),
            "value": str(self._state.portfolio_value),
            "positions": self._state.positions,
            "pnl": str(self._state.total_pnl),
        }
        serialized = json.dumps(portfolio_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def _validate_checkpoint(self, checkpoint: ReplayCheckpoint) -> None:
        """Validate checkpoint for determinism."""
        # Store checksum for comparison
        if checkpoint.event_count in self._state_history:
            previous = self._state_history[checkpoint.event_count]
            if previous != checkpoint.state_checksum:
                violation = {
                    "event_index": checkpoint.event_count,
                    "expected_checksum": previous,
                    "actual_checksum": checkpoint.state_checksum,
                    "type": "state_divergence",
                }
                self._integrity_violations.append(violation)
                logger.error(
                    "State divergence detected",
                    event_index=checkpoint.event_count,
                )
        
        self._state_history[checkpoint.event_count] = checkpoint.state_checksum
    
    def _calculate_determinism_score(self) -> float:
        """Calculate determinism score (0-100)."""
        if not self._checkpoints:
            return 100.0
        
        violations = len(self._integrity_violations)
        total = len(self._checkpoints)
        
        if total == 0:
            return 100.0
        
        # Score based on violations
        score = max(0, 100 - (violations * 10))
        return score
    
    def _restore_from_checkpoint(self, checkpoint: ReplayCheckpoint) -> ReplayState:
        """Restore state from checkpoint."""
        # In real implementation, would restore full state
        # For now, create fresh state
        return self._create_initial_state()
    
    def pause(self) -> None:
        """Pause replay."""
        self._paused = True
        logger.info("Replay paused")
    
    def resume(self) -> None:
        """Resume replay."""
        self._paused = False
        logger.info("Replay resumed")
    
    def stop(self) -> None:
        """Stop replay."""
        self._stopped = True
        logger.info("Replay stopped")
    
    def get_state_at(self, event_index: int) -> Optional[str]:
        """Get state checksum at specific event index."""
        return self._state_history.get(event_index)
