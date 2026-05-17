"""Drawdown protection risk rule.

Monitors and enforces drawdown limits.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from amatix.interfaces import Order
from amatix.risk.models import RiskConfig, RiskSeverity, RiskViolation
from amatix.risk.rules.base import BaseRiskRule


class DrawdownRule(BaseRiskRule):
    """Risk rule for drawdown protection.
    
    Checks:
        - Daily drawdown limit
        - Total drawdown limit
        - Proximity to kill switch threshold
    """
    
    def __init__(self, config: RiskConfig) -> None:
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "drawdown"
    
    @property
    def priority(self) -> int:
        return 5  # Very high priority - check early
    
    @property
    def severity(self) -> RiskSeverity:
        return RiskSeverity.CRITICAL
    
    @property
    def block_on_violation(self) -> bool:
        return True
    
    @property
    def reduce_on_violation(self) -> bool:
        return False
    
    async def evaluate(
        self,
        order: Order,
        portfolio: Dict[str, Any],
        market: Dict[str, Any],
    ) -> Optional[RiskViolation]:
        """Evaluate drawdown constraints."""
        # Get drawdown metrics
        current_drawdown = portfolio.get("current_drawdown", 0.0)
        daily_drawdown = portfolio.get("daily_drawdown", 0.0)
        
        # Check daily drawdown limit
        if daily_drawdown > self._config.max_daily_drawdown:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.CRITICAL,
                message=f"Daily drawdown {daily_drawdown:.2%} exceeds limit {self._config.max_daily_drawdown:.2%}",
                current_value=daily_drawdown,
                limit_value=self._config.max_daily_drawdown,
                symbol=order.symbol,
            )
        
        # Check total drawdown limit
        if current_drawdown > self._config.max_total_drawdown:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.FATAL,
                message=f"Total drawdown {current_drawdown:.2%} exceeds limit {self._config.max_total_drawdown:.2%}",
                current_value=current_drawdown,
                limit_value=self._config.max_total_drawdown,
                symbol=order.symbol,
            )
        
        # Warn if approaching kill switch (within 50% of threshold)
        kill_switch_threshold = self._config.kill_switch_drawdown
        warning_threshold = kill_switch_threshold * 0.5
        
        if current_drawdown > warning_threshold:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.WARNING,
                message=f"Approaching kill switch: drawdown {current_drawdown:.2%} (threshold: {kill_switch_threshold:.2%})",
                current_value=current_drawdown,
                limit_value=kill_switch_threshold,
                symbol=order.symbol,
            )
        
        return None
