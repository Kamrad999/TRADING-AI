"""Volatility risk rule.

Adjusts risk based on market volatility conditions.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from amatix.interfaces import Order
from amatix.risk.models import RiskConfig, RiskSeverity, RiskViolation
from amatix.risk.rules.base import BaseRiskRule


class VolatilityRule(BaseRiskRule):
    """Risk rule for volatility-based adjustments.
    
    Checks:
        - Current market volatility vs maximum
        - Volatility-adjusted position sizing
        - Volatility regime changes
    """
    
    def __init__(self, config: RiskConfig) -> None:
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "volatility"
    
    @property
    def priority(self) -> int:
        return 50
    
    @property
    def severity(self) -> RiskSeverity:
        return RiskSeverity.WARNING
    
    @property
    def block_on_violation(self) -> bool:
        return False
    
    @property
    def reduce_on_violation(self) -> bool:
        return True
    
    @property
    def reduction_factor(self) -> float:
        return 0.5
    
    async def evaluate(
        self,
        order: Order,
        portfolio: Dict[str, Any],
        market: Dict[str, Any],
    ) -> Optional[RiskViolation]:
        """Evaluate volatility constraints."""
        if not self._config.volatility_scaling:
            return None
        
        # Get volatility metrics
        current_vol = market.get("volatility", 0.0)
        vix_value = market.get("vix", None)  # VIX if available
        
        # Check if volatility exceeds max
        if current_vol > self._config.max_volatility:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.WARNING,
                message=f"Volatility {current_vol:.1%} exceeds max {self._config.max_volatility:.1%}",
                current_value=current_vol,
                limit_value=self._config.max_volatility,
                symbol=order.symbol,
                metadata={"vix": vix_value},
            )
        
        # High volatility warning (>80% of max)
        high_vol_threshold = self._config.max_volatility * 0.8
        if current_vol > high_vol_threshold:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.INFO,
                message=f"Elevated volatility {current_vol:.1%} - reducing position size",
                current_value=current_vol,
                limit_value=high_vol_threshold,
                symbol=order.symbol,
            )
        
        return None
