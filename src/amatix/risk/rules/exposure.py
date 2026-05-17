"""Exposure risk rule.

Enforces portfolio-level exposure limits.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from amatix.interfaces import Order
from amatix.risk.models import RiskConfig, RiskSeverity, RiskViolation
from amatix.risk.rules.base import BaseRiskRule


class ExposureRule(BaseRiskRule):
    """Risk rule for portfolio exposure limits.
    
    Checks:
        - Gross exposure (long + short)
        - Net exposure (long - short)
        - Leverage ratio
    """
    
    def __init__(self, config: RiskConfig) -> None:
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "exposure"
    
    @property
    def priority(self) -> int:
        return 40
    
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
        """Evaluate exposure constraints."""
        # Get current exposures
        long_exposure = portfolio.get("long_exposure", Decimal("0"))
        short_exposure = portfolio.get("short_exposure", Decimal("0"))
        portfolio_value = portfolio.get("total_value", Decimal("1"))
        
        # Calculate new exposure from this order
        price = market.get("price", Decimal("0"))
        if price == 0:
            return None
        
        order_value = abs(order.quantity) * price
        
        # Determine if adding to long or short
        # OrderSide.BUY increases long, OrderSide.SELL increases short (simplified)
        is_buy = order.side.value == "buy" if hasattr(order.side, 'value') else str(order.side).lower() == "buy"
        
        if is_buy:
            new_long = long_exposure + order_value
            new_short = short_exposure
        else:
            new_long = long_exposure
            new_short = short_exposure + order_value
        
        # Calculate metrics
        new_gross = new_long + new_short
        new_net = new_long - new_short
        new_leverage = new_gross / portfolio_value if portfolio_value > 0 else Decimal("0")
        
        gross_pct = new_gross / portfolio_value if portfolio_value > 0 else Decimal("0")
        net_pct = abs(new_net) / portfolio_value if portfolio_value > 0 else Decimal("0")
        
        # Check gross exposure
        max_gross = Decimal(str(self._config.max_gross_exposure))
        if gross_pct > max_gross:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.CRITICAL,
                message=f"Gross exposure {gross_pct:.1%} exceeds max {max_gross:.0%}",
                current_value=gross_pct,
                limit_value=max_gross,
                symbol=order.symbol,
            )
        
        # Check net exposure
        max_net = Decimal(str(self._config.max_net_exposure))
        if net_pct > max_net:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.WARNING,
                message=f"Net exposure {net_pct:.1%} exceeds max {max_net:.0%}",
                current_value=net_pct,
                limit_value=max_net,
                symbol=order.symbol,
            )
        
        # Check leverage
        max_leverage = Decimal(str(self._config.max_leverage))
        if new_leverage > max_leverage:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.CRITICAL,
                message=f"Leverage {new_leverage:.2f}x exceeds max {max_leverage:.2f}x",
                current_value=new_leverage,
                limit_value=max_leverage,
                symbol=order.symbol,
            )
        
        return None
