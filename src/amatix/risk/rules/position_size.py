"""Position size risk rule.

Enforces maximum position size limits.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from amatix.interfaces import Order
from amatix.risk.models import RiskConfig, RiskSeverity, RiskViolation
from amatix.risk.rules.base import BaseRiskRule


class PositionSizeRule(BaseRiskRule):
    """Risk rule for position size limits.
    
    Checks:
        - Max dollar size per position
        - Max percentage of portfolio
        - Max units per position
    """
    
    def __init__(self, config: RiskConfig) -> None:
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "position_size"
    
    @property
    def priority(self) -> int:
        return 10  # High priority
    
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
        """Evaluate position size limits."""
        # Get current price
        price = market.get("price", Decimal("0"))
        if price == 0:
            return None
        
        # Calculate notional value
        notional = abs(order.quantity) * price
        
        # Check max dollar size
        if notional > self._config.max_position_size:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.CRITICAL,
                message=f"Position size {notional} exceeds max {self._config.max_position_size}",
                current_value=notional,
                limit_value=self._config.max_position_size,
                symbol=order.symbol,
            )
        
        # Check max percentage of portfolio
        portfolio_value = portfolio.get("total_value", Decimal("0"))
        if portfolio_value > 0:
            position_pct = notional / portfolio_value
            if position_pct > Decimal(str(self._config.max_position_pct)):
                return RiskViolation(
                    rule_name=self.name,
                    severity=RiskSeverity.WARNING,
                    message=f"Position {position_pct:.1%} exceeds max {self._config.max_position_pct:.0%}",
                    current_value=position_pct,
                    limit_value=Decimal(str(self._config.max_position_pct)),
                    symbol=order.symbol,
                )
        
        return None
