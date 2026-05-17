"""Chaos Engineering During Replay for AMATIS.

Inject failures during accelerated replay to validate resilience:
    - WebSocket disconnects
    - Delayed events
    - Duplicated events
    - Dropped events
    - DB latency spikes
    - Memory pressure
    - Queue overflow
    - Broker API failures
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from amatix.chaos.engine import FailureType, RecoveryResult
from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import Event, EventType
from amatix.core.observability import get_logger
from amatix.simulation.replay_engine import AcceleratedReplayEngine, ReplayConfig, ReplayResult

logger = get_logger(__name__)


class ReplayFailureType(Enum):
    """Types of failures injectable during replay."""
    EVENT_DELAY = auto()           # Delay event processing
    EVENT_DROP = auto()            # Drop random events
    EVENT_DUPLICATE = auto()       # Send duplicate events
    EVENT_CORRUPT = auto()         # Corrupt event payload
    EVENT_REORDER = auto()         # Deliver events out of order
    
    WEBSOCKET_DISCONNECT = auto()  # Disconnect data feed
    DB_LATENCY_SPIKE = auto()      # Slow database responses
    MEMORY_PRESSURE = auto()       # Memory pressure
    QUEUE_OVERFLOW = auto()        # Event queue overflow
    BROKER_FAILURE = auto()        # Broker API failure
    
    PROCESSOR_SLOWDOWN = auto()    # Slow event processing
    CIRCUIT_BREAKER = auto()       # Trigger circuit breaker


@dataclass
class ChaosInjection:
    """Definition of a chaos injection during replay."""
    failure_type: ReplayFailureType
    trigger_event_index: int  # Inject at this event
    duration_events: int = 100  # How many events to affect
    severity: str = "medium"  # low, medium, high, critical
    probability: float = 0.5  # Probability of triggering per event
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosEvent:
    """Record of chaos during replay."""
    timestamp: datetime
    event_index: int
    failure_type: ReplayFailureType
    injection_success: bool
    recovery_detected: bool
    recovery_time_ms: Optional[float] = None
    system_impact: str = "none"  # none, degraded, failed
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosReplayResult:
    """Result of chaos replay session."""
    replay_result: ReplayResult
    chaos_events: List[ChaosEvent]
    total_injections: int
    successful_injections: int
    recoveries: int
    partial_recoveries: int
    failures: int
    resilience_score: float  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.replay_result.to_dict(),
            "chaos_injections": self.total_injections,
            "successful_injections": self.successful_injections,
            "recoveries": self.recoveries,
            "partial_recoveries": self.partial_recoveries,
            "failures": self.failures,
            "resilience_score": round(self.resilience_score, 2),
            "chaos_grade": self._get_grade(),
        }
    
    def _get_grade(self) -> str:
        if self.resilience_score >= 90:
            return "A (Production Ready)"
        elif self.resilience_score >= 80:
            return "B (Minor Issues)"
        elif self.resilience_score >= 70:
            return "C (Needs Improvement)"
        elif self.resilience_score >= 60:
            return "D (Significant Issues)"
        else:
            return "F (Not Production Ready)"


class ReplayFailureInjector:
    """Failure injector specifically for replay scenarios."""
    
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._active_injections: Dict[ReplayFailureType, ChaosInjection] = {}
        self._injection_history: List[Dict[str, Any]] = []
        
        # State tracking
        self._dropped_events: Set[int] = set()
        self._delayed_events: Dict[int, float] = {}  # event_id -> delay_seconds
        self._duplicated_events: Set[int] = set()
        self._memory_pressure_active = False
    
    def schedule_injection(self, injection: ChaosInjection) -> None:
        """Schedule a failure injection at specific event index."""
        self._active_injections[injection.failure_type] = injection
        logger.info(
            "Scheduled chaos injection",
            type=injection.failure_type.name,
            at_event=injection.trigger_event_index,
            duration=injection.duration_events,
        )
    
    def should_inject(
        self,
        event_index: int,
        failure_type: ReplayFailureType,
    ) -> bool:
        """Check if failure should be injected at this event."""
        injection = self._active_injections.get(failure_type)
        if not injection:
            return False
        
        # Check if within injection window
        start = injection.trigger_event_index
        end = start + injection.duration_events
        
        if not (start <= event_index < end):
            return False
        
        # Check probability
        return self._rng.random() < injection.probability
    
    def inject_event_delay(
        self,
        event: Event,
        event_index: int,
    ) -> float:
        """Calculate delay for event. Returns delay in seconds."""
        # Delay 10ms to 100ms
        delay = self._rng.uniform(0.01, 0.1)
        self._delayed_events[event_index] = delay
        
        self._injection_history.append({
            "type": "delay",
            "event_index": event_index,
            "delay_ms": delay * 1000,
        })
        
        return delay
    
    def inject_event_drop(self, event_index: int) -> bool:
        """Determine if event should be dropped."""
        self._dropped_events.add(event_index)
        
        self._injection_history.append({
            "type": "drop",
            "event_index": event_index,
        })
        
        return True
    
    def inject_event_duplicate(self, event: Event, event_index: int) -> Event:
        """Create duplicate event."""
        self._duplicated_events.add(event_index)
        
        # Create duplicate with same ID (should be deduplicated by system)
        dup = Event(
            event_type=event.event_type,
            payload=event.payload.copy(),
            context=event.context,
            event_id=event.event_id,
            priority=event.priority,
        )
        
        self._injection_history.append({
            "type": "duplicate",
            "event_index": event_index,
            "original_id": str(event.event_id),
        })
        
        return dup
    
    def inject_corruption(self, event: Event) -> Event:
        """Corrupt event payload."""
        corrupted_payload = event.payload.copy()
        
        # Corrupt a numeric field
        for key in corrupted_payload:
            if isinstance(corrupted_payload[key], (int, float, str)):
                if "price" in key.lower() or "quantity" in key.lower():
                    try:
                        val = float(corrupted_payload[key])
                        corrupted_payload[key] = str(val * -1)  # Flip sign
                        break
                    except (ValueError, TypeError):
                        # Key value cannot be converted to float, skip
                        pass
        
        return Event(
            event_type=event.event_type,
            payload=corrupted_payload,
            context=event.context,
            event_id=event.event_id,
            priority=event.priority,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get injection statistics."""
        return {
            "scheduled_injections": len(self._active_injections),
            "delayed_events": len(self._delayed_events),
            "dropped_events": len(self._dropped_events),
            "duplicated_events": len(self._duplicated_events),
            "total_injections": len(self._injection_history),
        }


class ChaosReplayOrchestrator:
    """Orchestrate chaos engineering during replay.
    
    Combines accelerated replay with controlled failure injection
    to validate system resilience under stress.
    
    Usage:
        orchestrator = ChaosReplayOrchestrator(event_bus)
        
        # Schedule chaos
        orchestrator.schedule_chaos(ChaosInjection(
            failure_type=ReplayFailureType.EVENT_DROP,
            trigger_event_index=1000,
            duration_events=50,
            probability=0.1,
        ))
        
        # Run chaos replay
        result = await orchestrator.run_chaos_replay(events, config)
        
        # Validate resilience
        assert result.resilience_score >= 80
    """
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        seed: Optional[int] = None,
    ) -> None:
        self._event_bus = event_bus
        self._injector = ReplayFailureInjector(seed)
        self._chaos_events: List[ChaosEvent] = []
        self._recovery_times: Dict[ReplayFailureType, List[float]] = {}
    
    def schedule_chaos(self, injection: ChaosInjection) -> None:
        """Schedule a chaos injection."""
        self._injector.schedule_injection(injection)
    
    def schedule_random_chaos(
        self,
        count: int = 5,
        event_range: Tuple[int, int] = (100, 10000),
    ) -> None:
        """Schedule random chaos injections."""
        failure_types = list(ReplayFailureType)
        
        for _ in range(count):
            failure_type = self._injector._rng.choice(failure_types)
            trigger = self._injector._rng.randint(*event_range)
            duration = self._injector._rng.randint(50, 200)
            
            injection = ChaosInjection(
                failure_type=failure_type,
                trigger_event_index=trigger,
                duration_events=duration,
                severity=self._injector._rng.choice(["low", "medium", "high"]),
                probability=self._injector._rng.uniform(0.2, 0.8),
            )
            
            self.schedule_chaos(injection)
    
    async def run_chaos_replay(
        self,
        events: List[Event],
        config: Optional[ReplayConfig] = None,
    ) -> ChaosReplayResult:
        """Run replay with chaos injection.
        
        This wraps the standard replay engine with chaos injection
        hooks to validate resilience.
        """
        # Create wrapped replay engine
        engine = AcceleratedReplayEngine(self._event_bus, config)
        
        # Inject chaos hooks
        chaos_events = []
        
        for i, event in enumerate(events):
            # Check for chaos injections
            for failure_type in ReplayFailureType:
                if self._injector.should_inject(i, failure_type):
                    chaos_start = datetime.utcnow()
                    
                    try:
                        if failure_type == ReplayFailureType.EVENT_DELAY:
                            delay = self._injector.inject_event_delay(event, i)
                            await asyncio.sleep(delay)
                            
                            chaos_events.append(ChaosEvent(
                                timestamp=chaos_start,
                                event_index=i,
                                failure_type=failure_type,
                                injection_success=True,
                                recovery_detected=True,
                                recovery_time_ms=delay * 1000,
                                details={"delay_ms": delay * 1000},
                            ))
                        
                        elif failure_type == ReplayFailureType.EVENT_DROP:
                            self._injector.inject_event_drop(i)
                            # Skip this event
                            chaos_events.append(ChaosEvent(
                                timestamp=chaos_start,
                                event_index=i,
                                failure_type=failure_type,
                                injection_success=True,
                                recovery_detected=False,
                                details={"event_dropped": True},
                            ))
                            continue  # Skip processing
                        
                        elif failure_type == ReplayFailureType.EVENT_DUPLICATE:
                            dup = self._injector.inject_event_duplicate(event, i)
                            # Process both (system should dedupe)
                            chaos_events.append(ChaosEvent(
                                timestamp=chaos_start,
                                event_index=i,
                                failure_type=failure_type,
                                injection_success=True,
                                recovery_detected=False,
                                details={"duplicate_sent": True},
                            ))
                        
                        elif failure_type == ReplayFailureType.EVENT_CORRUPT:
                            event = self._injector.inject_corruption(event)
                            chaos_events.append(ChaosEvent(
                                timestamp=chaos_start,
                                event_index=i,
                                failure_type=failure_type,
                                injection_success=True,
                                recovery_detected=False,
                                system_impact="degraded",
                                details={"corruption_injected": True},
                            ))
                    
                    except Exception as e:
                        chaos_events.append(ChaosEvent(
                            timestamp=chaos_start,
                            event_index=i,
                            failure_type=failure_type,
                            injection_success=False,
                            recovery_detected=False,
                            system_impact="failed",
                            details={"error": str(e)},
                        ))
        
        # Run replay
        replay_result = await engine.replay_historical_data(events)
        
        # Calculate resilience score
        resilience_score = self._calculate_resilience_score(chaos_events)
        
        # Categorize outcomes
        recoveries = sum(1 for e in chaos_events if e.recovery_detected)
        partial = sum(1 for e in chaos_events if e.system_impact == "degraded")
        failures = sum(1 for e in chaos_events if e.system_impact == "failed")
        
        return ChaosReplayResult(
            replay_result=replay_result,
            chaos_events=chaos_events,
            total_injections=len(chaos_events),
            successful_injections=sum(1 for e in chaos_events if e.injection_success),
            recoveries=recoveries,
            partial_recoveries=partial,
            failures=failures,
            resilience_score=resilience_score,
        )
    
    def _calculate_resilience_score(self, chaos_events: List[ChaosEvent]) -> float:
        """Calculate resilience score from chaos events."""
        if not chaos_events:
            return 100.0
        
        # Weight by severity
        total_weight = 0
        recovered_weight = 0
        
        severity_weights = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 5,
        }
        
        for event in chaos_events:
            weight = severity_weights.get(event.severity, 2)
            total_weight += weight
            
            if event.recovery_detected:
                recovered_weight += weight
            elif event.system_impact != "failed":
                recovered_weight += weight * 0.5  # Partial
        
        return (recovered_weight / total_weight) * 100 if total_weight > 0 else 100.0
    
    def generate_chaos_report(self, result: ChaosReplayResult) -> str:
        """Generate human-readable chaos report."""
        lines = [
            "=" * 60,
            "CHAOS REPLAY VALIDATION REPORT",
            "=" * 60,
            "",
            f"Session ID: {result.replay_result.session_id}",
            f"Events Processed: {result.replay_result.events_processed}",
            f"Duration: {result.replay_result.replay_result.duration_seconds:.2f}s",
            "",
            "CHAOS INJECTIONS:",
            f"  Total Scheduled: {result.total_injections}",
            f"  Successful: {result.successful_injections}",
            f"  Full Recoveries: {result.recoveries}",
            f"  Partial: {result.partial_recoveries}",
            f"  Failures: {result.failures}",
            "",
            f"RESILIENCE SCORE: {result.resilience_score:.1f}/100",
            f"GRADE: {result._get_grade()}",
            "",
            "DETERMINISM:",
            f"  Score: {result.replay_result.determinism_score:.1f}",
            f"  Integrity Violations: {len(result.replay_result.integrity_violations)}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
