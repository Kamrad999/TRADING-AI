"""Guardian Risk Engine - Core risk assessment orchestrator.

The RiskEngine has FINAL VETO AUTHORITY over all trading decisions.
No order can execute without risk approval.

Architecture:
    - Pluggable rule system
    - Async evaluation
    - Multi-layer checks
    - Event emission
    - Kill switch integration
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Type

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import EventPriority, EventType
from amatix.core.observability import get_logger, get_metrics
from amatix.interfaces import Order, Position, Symbol
from amatix.risk.models import (
    RiskAdjustment,
    RiskAssessment,
    RiskConfig,
    RiskRule,
    RiskSeverity,
    RiskSnapshot,
    RiskVerdict,
    RiskViolation,
)
from amatix.risk.rules import (
    BaseRiskRule,
    ConcentrationRule,
    DrawdownRule,
    ExposureRule,
    LiquidityRule,
    PositionSizeRule,
    VolatilityRule,
)

logger = get_logger(__name__)


class RiskEngine:
    """Guardian Risk Engine with veto authority.
    
    Multi-layer risk assessment:
        1. Pre-trade checks (symbol, liquidity)
        2. Position-level checks (size, concentration)
        3. Portfolio-level checks (exposure, correlation)
        4. Emergency checks (drawdown, kill switch)
    
    All orders MUST pass risk assessment before execution.
    
    Example:
        >>> engine = RiskEngine(event_bus, config)
        >>> await engine.initialize()
        >>> 
        >>> assessment = await engine.assess_order(
        ...     order=order,
        ...     portfolio=portfolio_state,
        ...     market=market_state,
        ... )
        >>> 
        >>> if assessment.is_approved:
        ...     await execute_order(order)
    """
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        config: Optional[RiskConfig] = None,
    ) -> None:
        """Initialize risk engine.
        
        Args:
            event_bus: Event bus for risk event emission
            config: Risk configuration (uses defaults if None)
        """
        self._event_bus = event_bus
        self._config = config or RiskConfig()
        
        # Rule registry
        self._rules: List[BaseRiskRule] = []
        self._rule_map: Dict[str, BaseRiskRule] = {}
        
        # State
        self._initialized = False
        self._kill_switch_active = False
        self._circuit_breaker_active = False
        
        # Portfolio tracking
        self._current_snapshot: Optional[RiskSnapshot] = None
        self._assessments: List[RiskAssessment] = []  # Recent assessments
        
        # Metrics
        self._total_assessments = 0
        self._rejected_count = 0
    
    async def initialize(self) -> None:
        """Initialize risk engine and register default rules."""
        logger.info("Initializing Guardian Risk Engine")
        
        # Register default rules
        self._register_default_rules()
        
        self._initialized = True
        
        # Emit initialization event
        await self._event_bus.emit_new(
            EventType.RISK_CHECK_PASSED,  # Reuse or create new
            {
                "event": "risk_engine_initialized",
                "rules": len(self._rules),
            },
            priority=EventPriority.NORMAL,
            source="risk_engine",
        )
        
        logger.info(
            "Risk engine initialized",
            rules=len(self._rules),
        )
    
    def _register_default_rules(self) -> None:
        """Register the default set of risk rules."""
        default_rules: List[Type[BaseRiskRule]] = [
            PositionSizeRule,
            LiquidityRule,
            ConcentrationRule,
            ExposureRule,
            DrawdownRule,
            VolatilityRule,
        ]
        
        for rule_class in default_rules:
            rule = rule_class(self._config)
            self.register_rule(rule)
    
    def register_rule(self, rule: BaseRiskRule) -> None:
        """Register a risk rule.
        
        Args:
            rule: Risk rule implementation
        """
        self._rules.append(rule)
        self._rule_map[rule.name] = rule
        
        # Sort by priority
        self._rules.sort(key=lambda r: r.priority)
        
        logger.debug("Risk rule registered", rule=rule.name)
    
    def unregister_rule(self, name: str) -> bool:
        """Unregister a rule by name."""
        if name in self._rule_map:
            rule = self._rule_map.pop(name)
            self._rules.remove(rule)
            return True
        return False
    
    async def assess_order(
        self,
        order: Order,
        portfolio: Dict[str, Any],
        market: Dict[str, Any],
    ) -> RiskAssessment:
        """Assess order risk - MAIN ENTRY POINT.
        
        This method has FINAL VETO AUTHORITY.
        No order can execute without passing this assessment.
        
        Args:
            order: Order to assess
            portfolio: Current portfolio state
            market: Current market conditions
        
        Returns:
            RiskAssessment with verdict and adjustments
        """
        start_time = time.time()
        self._total_assessments += 1
        
        # Check kill switch first
        if self._kill_switch_active:
            return self._create_kill_switch_assessment(order)
        
        # Check circuit breaker
        if self._circuit_breaker_active:
            return self._create_circuit_breaker_assessment(order)
        
        # Initialize assessment
        assessment = RiskAssessment.create(
            verdict=RiskVerdict.APPROVED,
            final_size=order.quantity,
        )
        
        # Evaluate all rules
        for rule in self._rules:
            if not rule.enabled:
                continue
            
            try:
                violation = await rule.evaluate(order, portfolio, market)
                
                if violation:
                    assessment.violations.append(violation)
                    assessment.rules_evaluated.append(rule.name)
                    
                    # Update risk score
                    severity_scores = {
                        RiskSeverity.INFO: 0.1,
                        RiskSeverity.WARNING: 0.3,
                        RiskSeverity.CRITICAL: 0.6,
                        RiskSeverity.FATAL: 1.0,
                    }
                    assessment.risk_score += severity_scores.get(
                        violation.severity, 0.0
                    )
                    
                    # Handle block
                    if rule.block_on_violation:
                        assessment.verdict = RiskVerdict.REJECTED
                        break
                    
                    # Handle reduction
                    if rule.reduce_on_violation:
                        new_size = order.quantity * Decimal(
                            str(rule.reduction_factor)
                        )
                        assessment.adjustments.append(
                            RiskAdjustment(
                                parameter="size",
                                original_value=order.quantity,
                                suggested_value=new_size,
                                reason=f"{rule.name} violation",
                            )
                        )
                        assessment.final_size = new_size
                        assessment.verdict = RiskVerdict.REDUCED
            
            except Exception as e:
                logger.exception(
                    "Risk rule evaluation failed - treating as critical violation",
                    rule=rule.name,
                    error=str(e),
                )
                # Conservative: treat failure as critical (FAIL CLOSED)
                assessment.violations.append(
                    RiskViolation(
                        rule_name=rule.name,
                        severity=RiskSeverity.CRITICAL,
                        message=f"Rule evaluation failed: {e}",
                        current_value="error",
                        limit_value="unknown",
                    )
                )
        
        # Cap risk score
        assessment.risk_score = min(1.0, assessment.risk_score)
        
        # Determine final verdict
        if assessment.has_critical_violations:
            if assessment.verdict not in [RiskVerdict.REJECTED, RiskVerdict.EMERGENCY_HALT]:
                assessment.verdict = RiskVerdict.REJECTED
        
        # Calculate evaluation time
        assessment.evaluation_time_ms = (time.time() - start_time) * 1000
        
        # Update stats
        if assessment.is_rejected:
            self._rejected_count += 1
        
        # Emit event
        await self._emit_assessment(assessment, order)
        
        # Store for history
        self._assessments.append(assessment)
        if len(self._assessments) > 1000:
            self._assessments.pop(0)
        
        return assessment
    
    async def assess_signal(
        self,
        signal: Any,  # Signal
        portfolio: Dict[str, Any],
    ) -> RiskAssessment:
        """Quick pre-trade assessment for signals.
        
        Lighter-weight than full order assessment.
        Used to filter signals before they become orders.
        
        Args:
            signal: Trading signal
            portfolio: Portfolio state
        
        Returns:
            RiskAssessment
        """
        # Simplified assessment for signals
        assessment = RiskAssessment.create(
            verdict=RiskVerdict.APPROVED,
            final_size=Decimal("0"),  # Signals don't have size yet
        )
        
        # Check exposure limits
        symbol = signal.symbol
        positions = portfolio.get("positions", {})
        
        if symbol.canonical in positions:
            position = positions[symbol.canonical]
            
            # Check for duplicate signal direction
            if (
                (signal.direction.value == "long" and position["side"] == "long") or
                (signal.direction.value == "short" and position["side"] == "short")
            ):
                # Already in position, signal is redundant
                assessment.verdict = RiskVerdict.REJECTED
                assessment.violations.append(
                    RiskViolation(
                        rule_name="duplicate_signal",
                        severity=RiskSeverity.INFO,
                        message=f"Already in {position['side']} position",
                        current_value=position["side"],
                        limit_value=signal.direction.value,
                        symbol=symbol,
                    )
                )
        
        return assessment
    
    def _create_kill_switch_assessment(self, order: Order) -> RiskAssessment:
        """Create assessment for kill switch rejection."""
        return RiskAssessment.create(
            verdict=RiskVerdict.EMERGENCY_HALT,
            final_size=Decimal("0"),
            kill_switch_triggered=True,
            violations=[
                RiskViolation(
                    rule_name="kill_switch",
                    severity=RiskSeverity.FATAL,
                    message="Kill switch is active - all trading halted",
                    current_value="active",
                    limit_value="inactive",
                )
            ],
        )
    
    def _create_circuit_breaker_assessment(self, order: Order) -> RiskAssessment:
        """Create assessment for circuit breaker."""
        return RiskAssessment.create(
            verdict=RiskVerdict.REJECTED,
            final_size=Decimal("0"),
            violations=[
                RiskViolation(
                    rule_name="circuit_breaker",
                    severity=RiskSeverity.CRITICAL,
                    message="Circuit breaker active - trading temporarily disabled",
                    current_value="active",
                    limit_value="inactive",
                )
            ],
        )
    
    async def _emit_assessment(
        self,
        assessment: RiskAssessment,
        order: Order,
    ) -> None:
        """Emit risk assessment event."""
        event_type = (
            EventType.RISK_CHECK_FAILED
            if assessment.is_rejected
            else EventType.RISK_CHECK_PASSED
        )
        
        await self._event_bus.emit_new(
            event_type,
            {
                "assessment_id": str(assessment.assessment_id),
                "order_id": order.order_id,
                "symbol": str(order.symbol),
                "verdict": assessment.verdict.value,
                "risk_score": assessment.risk_score,
                "violation_count": len(assessment.violations),
                "evaluation_time_ms": assessment.evaluation_time_ms,
            },
            priority=EventPriority.CRITICAL,
            source="risk_engine",
        )
        
        get_metrics().counter(
            "risk_assessments",
            labels={
                "verdict": assessment.verdict.value,
                "symbol": order.symbol.canonical,
            },
        )
    
    def update_snapshot(self, snapshot: RiskSnapshot) -> None:
        """Update current risk snapshot."""
        self._current_snapshot = snapshot
        
        # Check for emergency conditions
        if snapshot.current_drawdown > self._config.kill_switch_drawdown:
            self._activate_kill_switch(
                f"Drawdown {snapshot.current_drawdown:.2%} exceeded limit"
            )
    
    def _activate_kill_switch(self, reason: str) -> None:
        """Activate emergency kill switch."""
        if self._kill_switch_active:
            return
        
        self._kill_switch_active = True
        logger.critical("KILL SWITCH ACTIVATED", reason=reason)
        
        # Emit immediately
        asyncio.create_task(
            self._event_bus.emit_new(
                EventType.KILL_SWITCH_TRIGGERED,
                {"reason": reason, "drawdown": self._current_snapshot.current_drawdown if self._current_snapshot else None},
                priority=EventPriority.CRITICAL,
                source="risk_engine",
            )
        )
    
    def deactivate_kill_switch(self, auth_token: str) -> bool:
        """Deactivate kill switch (requires auth)."""
        # TODO: Implement proper auth
        if auth_token == "emergency_override":
            self._kill_switch_active = False
            self._circuit_breaker_active = False
            logger.info("Kill switch deactivated")
            return True
        return False
    
    def activate_circuit_breaker(self, reason: str) -> None:
        """Activate circuit breaker (temporary halt)."""
        self._circuit_breaker_active = True
        logger.warning("Circuit breaker activated", reason=reason)
    
    def deactivate_circuit_breaker(self) -> None:
        """Deactivate circuit breaker."""
        self._circuit_breaker_active = False
        logger.info("Circuit breaker deactivated")
    
    @property
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch_active
    
    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return not (self._kill_switch_active or self._circuit_breaker_active)
    
    def get_rules(self) -> List[BaseRiskRule]:
        """Get all registered rules."""
        return self._rules.copy()
    
    def get_rule(self, name: str) -> Optional[BaseRiskRule]:
        """Get rule by name."""
        return self._rule_map.get(name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get risk engine statistics."""
        return {
            "initialized": self._initialized,
            "kill_switch_active": self._kill_switch_active,
            "circuit_breaker_active": self._circuit_breaker_active,
            "trading_allowed": self.is_trading_allowed,
            "total_assessments": self._total_assessments,
            "rejected_count": self._rejected_count,
            "rejection_rate": (
                self._rejected_count / self._total_assessments
                if self._total_assessments > 0 else 0
            ),
            "rules": len(self._rules),
            "current_snapshot": self._current_snapshot is not None,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for risk engine."""
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "kill_switch": self._kill_switch_active,
            "circuit_breaker": self._circuit_breaker_active,
            "rules_healthy": len(self._rules),
        }
