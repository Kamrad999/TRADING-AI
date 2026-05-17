"""Chaos engineering engine for resilience testing.

Inspired by Netflix Chaos Monkey and institutional stress testing.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from amatix.core.observability import get_logger

logger = get_logger(__name__)


class FailureType(Enum):
    """Types of failures that can be injected."""
    LATENCY = auto()
    DISCONNECT = auto()
    CORRUPTION = auto()
    MEMORY_PRESSURE = auto()
    CPU_SPIKE = auto()
    DEADLOCK = auto()
    PARTIAL_FAILURE = auto()


class RecoveryResult(Enum):
    """Result of failure recovery attempt."""
    FULL = auto()      # Fully recovered
    PARTIAL = auto()   # Partially recovered, degraded
    NONE = auto()      # No recovery
    TIMEOUT = auto()   # Recovery timed out


@dataclass
class FailureScenario:
    """Definition of a failure scenario for testing.
    
    Examples:
        - WebSocket disconnect during order submission
        - 5-second latency spike during risk check
        - Corrupted market data payload
        - Memory exhaustion during event storm
    """
    name: str
    failure_type: FailureType
    target_component: str
    trigger_condition: Callable[[], bool] = field(default=lambda: True)
    duration_seconds: float = 5.0
    severity: str = "medium"  # low, medium, high, critical
    probability: float = 1.0  # 0.0 to 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def should_trigger(self) -> bool:
        """Check if scenario should trigger based on probability."""
        return random.random() < self.probability and self.trigger_condition()


@dataclass
class ChaosEvent:
    """Record of a chaos event during testing."""
    timestamp: datetime
    scenario_name: str
    failure_type: FailureType
    target: str
    duration_seconds: float
    recovery_result: RecoveryResult
    system_impact: str  # "none", "degraded", "failed"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResilienceReport:
    """Report from chaos testing session."""
    total_scenarios: int
    executed_scenarios: int
    successful_recoveries: int
    partial_recoveries: int
    failed_recoveries: int
    events: List[ChaosEvent]
    score: float  # 0.0 to 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "executed": self.executed_scenarios,
            "full_recovery": self.successful_recoveries,
            "partial_recovery": self.partial_recoveries,
            "failed_recovery": self.failed_recoveries,
            "resilience_score": round(self.score, 2),
            "grade": self._get_grade(),
        }
    
    def _get_grade(self) -> str:
        if self.score >= 90:
            return "A (Production Ready)"
        elif self.score >= 80:
            return "B (Minor Issues)"
        elif self.score >= 70:
            return "C (Needs Improvement)"
        elif self.score >= 60:
            return "D (Significant Issues)"
        else:
            return "F (Not Production Ready)"


class FailureInjector(ABC):
    """Base class for failure injectors.
    
    Each injector implements a specific type of failure:
        - How to trigger the failure
        - How to monitor the failure
        - How to verify recovery
    """
    
    @property
    @abstractmethod
    def failure_type(self) -> FailureType:
        """Return the type of failure this injector creates."""
        pass
    
    @abstractmethod
    async def inject(
        self,
        target: str,
        duration_seconds: float,
        parameters: Dict[str, Any],
    ) -> bool:
        """Inject the failure.
        
        Returns:
            True if injection was successful
        """
        pass
    
    @abstractmethod
    async def verify_recovery(
        self,
        target: str,
        timeout_seconds: float = 30.0,
    ) -> RecoveryResult:
        """Verify system recovered from failure.
        
        Returns:
            Recovery result status
        """
        pass


class ChaosEngine:
    """Chaos engineering engine for resilience testing.
    
    Inspired by institutional stress testing practices:
        - Controlled failure injection
        - Recovery validation
        - Resilience scoring
        - Forensic analysis
    
    Usage:
        engine = ChaosEngine()
        
        # Register scenarios
        engine.register_scenario(FailureScenario(
            name="websocket_disconnect",
            failure_type=FailureType.DISCONNECT,
            target_component="market_data_provider",
        ))
        
        # Run chaos test
        report = await engine.run_chaos_session(duration_seconds=300)
    """
    
    def __init__(self) -> None:
        self._scenarios: List[FailureScenario] = []
        self._injectors: Dict[FailureType, FailureInjector] = {}
        self._events: List[ChaosEvent] = []
        self._active = False
        self._executed_scenarios: Set[str] = set()
    
    def register_scenario(self, scenario: FailureScenario) -> None:
        """Register a failure scenario."""
        self._scenarios.append(scenario)
        logger.info(
            "Registered chaos scenario",
            name=scenario.name,
            type=scenario.failure_type.name,
            target=scenario.target_component,
        )
    
    def register_injector(
        self,
        failure_type: FailureType,
        injector: FailureInjector,
    ) -> None:
        """Register a failure injector."""
        self._injectors[failure_type] = injector
        logger.info(
            "Registered failure injector",
            type=failure_type.name,
            injector=injector.__class__.__name__,
        )
    
    async def run_chaos_session(
        self,
        duration_seconds: float = 300.0,
        max_concurrent: int = 3,
    ) -> ResilienceReport:
        """Run a chaos testing session.
        
        Args:
            duration_seconds: How long to run chaos
            max_concurrent: Max concurrent failures
        
        Returns:
            ResilienceReport with results
        """
        self._active = True
        self._events.clear()
        self._executed_scenarios.clear()
        
        start_time = datetime.utcnow()
        end_time = start_time.timestamp() + duration_seconds
        
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks: List[asyncio.Task] = []
        
        logger.info(
            "Starting chaos session",
            duration=duration_seconds,
            scenarios=len(self._scenarios),
        )
        
        try:
            while datetime.utcnow().timestamp() < end_time and self._active:
                # Check for scenarios to trigger
                for scenario in self._scenarios:
                    if scenario.name in self._executed_scenarios:
                        continue
                    
                    if scenario.should_trigger():
                        # Execute scenario with semaphore
                        task = asyncio.create_task(
                            self._execute_scenario_with_semaphore(scenario, semaphore)
                        )
                        tasks.append(task)
                        self._executed_scenarios.add(scenario.name)
                
                await asyncio.sleep(1.0)  # Check every second
        
        except asyncio.CancelledError:
            logger.info("Chaos session cancelled")
        
        finally:
            self._active = False
            # Wait for all tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # Generate report
        return self._generate_report()
    
    async def _execute_scenario_with_semaphore(
        self,
        scenario: FailureScenario,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute scenario with concurrency limit."""
        async with semaphore:
            await self._execute_scenario(scenario)
    
    async def _execute_scenario(self, scenario: FailureScenario) -> None:
        """Execute a single failure scenario."""
        injector = self._injectors.get(scenario.failure_type)
        if not injector:
            logger.warning(
                "No injector for failure type",
                type=scenario.failure_type.name,
            )
            return
        
        start_time = datetime.utcnow()
        
        logger.info(
            "Injecting failure",
            scenario=scenario.name,
            target=scenario.target_component,
            duration=scenario.duration_seconds,
        )
        
        # Inject failure
        try:
            injection_success = await injector.inject(
                target=scenario.target_component,
                duration_seconds=scenario.duration_seconds,
                parameters=scenario.parameters,
            )
            
            if not injection_success:
                logger.error("Failed to inject failure", scenario=scenario.name)
                return
            
            # Wait for failure duration
            await asyncio.sleep(scenario.duration_seconds)
            
            # Verify recovery
            recovery_result = await injector.verify_recovery(
                target=scenario.target_component,
                timeout_seconds=30.0,
            )
            
            # Determine system impact
            impact = self._determine_impact(recovery_result)
            
            # Record event
            event = ChaosEvent(
                timestamp=start_time,
                scenario_name=scenario.name,
                failure_type=scenario.failure_type,
                target=scenario.target_component,
                duration_seconds=scenario.duration_seconds,
                recovery_result=recovery_result,
                system_impact=impact,
                details={"injection_success": injection_success},
            )
            self._events.append(event)
            
            logger.info(
                "Scenario completed",
                scenario=scenario.name,
                recovery=recovery_result.name,
                impact=impact,
            )
        
        except Exception as e:
            logger.error(
                "Scenario execution failed",
                scenario=scenario.name,
                error=str(e),
            )
            
            event = ChaosEvent(
                timestamp=start_time,
                scenario_name=scenario.name,
                failure_type=scenario.failure_type,
                target=scenario.target_component,
                duration_seconds=0,
                recovery_result=RecoveryResult.NONE,
                system_impact="failed",
                details={"error": str(e)},
            )
            self._events.append(event)
    
    def _determine_impact(self, recovery: RecoveryResult) -> str:
        """Determine system impact based on recovery result."""
        if recovery == RecoveryResult.FULL:
            return "none"
        elif recovery == RecoveryResult.PARTIAL:
            return "degraded"
        else:
            return "failed"
    
    def _generate_report(self) -> ResilienceReport:
        """Generate resilience report from events."""
        if not self._events:
            return ResilienceReport(
                total_scenarios=len(self._scenarios),
                executed_scenarios=0,
                successful_recoveries=0,
                partial_recoveries=0,
                failed_recoveries=0,
                events=[],
                score=0.0,
            )
        
        successful = sum(1 for e in self._events if e.recovery_result == RecoveryResult.FULL)
        partial = sum(1 for e in self._events if e.recovery_result == RecoveryResult.PARTIAL)
        failed = sum(1 for e in self._events if e.recovery_result in [RecoveryResult.NONE, RecoveryResult.TIMEOUT])
        
        # Calculate score
        total_weight = len(self._events)
        if total_weight == 0:
            score = 0.0
        else:
            full_weight = 1.0
            partial_weight = 0.5
            failed_weight = 0.0
            
            score = (
                (successful * full_weight + partial * partial_weight + failed * failed_weight)
                / total_weight
            ) * 100
        
        return ResilienceReport(
            total_scenarios=len(self._scenarios),
            executed_scenarios=len(self._executed_scenarios),
            successful_recoveries=successful,
            partial_recoveries=partial,
            failed_recoveries=failed,
            events=self._events,
            score=score,
        )
    
    def stop(self) -> None:
        """Stop chaos session."""
        self._active = False
        logger.info("Chaos session stopped")
    
    def get_events(self) -> List[ChaosEvent]:
        """Get all chaos events from last session."""
        return self._events.copy()
