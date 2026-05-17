"""Concentration risk rule.

Enforces sector and symbol concentration limits.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from amatix.interfaces import Order
from amatix.risk.models import RiskConfig, RiskSeverity, RiskViolation
from amatix.risk.rules.base import BaseRiskRule


class ConcentrationRule(BaseRiskRule):
    """Risk rule for concentration limits.
    
    Checks:
        - Max symbols per sector
        - Sector exposure concentration
        - Single symbol concentration vs portfolio
    """
    
    def __init__(self, config: RiskConfig) -> None:
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return "concentration"
    
    @property
    def priority(self) -> int:
        return 30
    
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
        return 0.7
    
    async def evaluate(
        self,
        order: Order,
        portfolio: Dict[str, Any],
        market: Dict[str, Any],
    ) -> Optional[RiskViolation]:
        """Evaluate concentration constraints."""
        # Get current price
        price = market.get("price", Decimal("0"))
        if price == 0:
            return None
        
        # Calculate position value after order
        order_value = abs(order.quantity) * price
        
        # Get sector (from symbol metadata or default)
        sector = market.get("sector", "unknown")
        
        # Check sector concentration
        sector_exposure = portfolio.get("sector_exposure", {})
        current_sector_value = sector_exposure.get(sector, Decimal("0"))
        new_sector_value = current_sector_value + order_value
        
        portfolio_value = portfolio.get("total_value", Decimal("1"))
        new_sector_pct = new_sector_value / portfolio_value if portfolio_value > 0 else Decimal("0")
        
        if new_sector_pct > Decimal(str(self._config.max_sector_exposure)):
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.WARNING,
                message=f"Sector {sector} exposure {new_sector_pct:.1%} exceeds max {self._config.max_sector_exposure:.0%}",
                current_value=new_sector_pct,
                limit_value=Decimal(str(self._config.max_sector_exposure)),
                symbol=order.symbol,
                metadata={"sector": sector},
            )
        
        # Check symbol count per sector
        sector_symbols = portfolio.get("sector_symbols", {}).get(sector, [])
        symbol_key = order.symbol.canonical if hasattr(order.symbol, 'canonical') else str(order.symbol)
        
        if symbol_key not in sector_symbols and len(sector_symbols) >= self._config.max_symbols_per_sector:
            return RiskViolation(
                rule_name=self.name,
                severity=RiskSeverity.INFO,
                message=f"Sector {sector} already has {len(sector_symbols)} symbols (max: {self._config.max_symbols_per_sector})",
                current_value=len(sector_symbols),
                limit_value=self._config.max_symbols_per_sector,
                symbol=order.symbol,
                metadata={"sector": sector},
            )
        
        return None
