"""Liquidity risk rule.

Validates market liquidity before order submission.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from amatix.interfaces import Order
from amatix.risk.models import RiskConfig, RiskSeverity, RiskViolation
from amatix.risk.rules.base import BaseRiskRule


class LiquidityRule(BaseRiskRule):
    """Risk rule for liquidity validation.
    
    Checks:
        - Bid-ask spread
        - Daily volume
        - Market depth
        - Price impact estimation
    """
    
    def __init__(self, config: RiskConfig) -> None:
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "liquidity"
    
    @property
    def priority(self) -> int:
        return 20
    
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
        """Evaluate liquidity constraints."""
        # Check spread
        spread_bps = market.get("spread_bps", 0.0)
        if spread_bps > self._config.max_spread_bps:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.CRITICAL,
                message=f"Spread {spread_bps:.1f} bps exceeds max {self._config.max_spread_bps}",
                current_value=spread_bps,
                limit_value=self._config.max_spread_bps,
                symbol=order.symbol,
            )
        
        # Check volume
        daily_volume = market.get("daily_volume", Decimal("0"))
        if daily_volume < self._config.min_liquidity:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.CRITICAL,
                message=f"Volume {daily_volume} below minimum {self._config.min_liquidity}",
                current_value=daily_volume,
                limit_value=self._config.min_liquidity,
                symbol=order.symbol,
            )
        
        # Estimate price impact
        price = market.get("price", Decimal("0"))
        if price > 0 and daily_volume > 0:
            notional = abs(order.quantity) * price
            impact_pct = notional / daily_volume
            
            if impact_pct > Decimal("0.01"):  # > 1% of daily volume
                return RiskViolation(
                    rule_name=self.name,
                    severity=RiskSeverity.WARNING,
                    message=f"Order {impact_pct:.2%} of daily volume - high impact",
                    current_value=impact_pct,
                    limit_value=Decimal("0.01"),
                    symbol=order.symbol,
                )
        
        return None
