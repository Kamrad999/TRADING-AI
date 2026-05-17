"""Base class for risk rules.

Defines the interface for all risk rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from amatix.interfaces import Order
from amatix.risk.models import RiskConfig, RiskRule, RiskSeverity, RiskViolation


class BaseRiskRule(ABC):
    """Abstract base class for risk rules.
    
    All risk rules must implement:
        - evaluate(): Check if order violates rule
        - name: Rule identifier
        - priority: Evaluation order
        - enabled: Whether rule is active
    
    Example:
        >>> class MyRule(BaseRiskRule):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_rule"
        ...     
        ...     async def evaluate(self, order, portfolio, market):
        ...         if violation_detected:
        ...             return RiskViolation(...)
        ...         return None
    """
    
    def __init__(self, config: RiskConfig) -> None:
        """Initialize rule with config.
        
        Args:
            config: Global risk configuration
        """
        self._config = config
        self._enabled = True
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name/identifier."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Evaluation priority (lower = first)."""
        pass
    
    @property
    @abstractmethod
    def severity(self) -> RiskSeverity:
        """Default severity for violations."""
        pass
    
    @property
    @abstractmethod
    def block_on_violation(self) -> bool:
        """Whether to block order on violation."""
        pass
    
    @property
    @abstractmethod
    def reduce_on_violation(self) -> bool:
        """Whether to suggest size reduction on violation."""
        pass
    
    @property
    def enabled(self) -> bool:
        """Whether rule is enabled."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable rule."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable rule."""
        self._enabled = False
    
    @abstractmethod
    async def evaluate(
        self,
        order: Order,
        portfolio: Dict[str, Any],
        market: Dict[str, Any],
    ) -> Optional[RiskViolation]:
        """Evaluate order against this rule.
        
        Args:
            order: Order to evaluate
            portfolio: Current portfolio state
            market: Current market conditions
        
        Returns:
            RiskViolation if rule violated, None otherwise
        """
        pass
    
    def to_rule_config(self) -> RiskRule:
        """Export rule configuration."""
        return RiskRule(
            name=self.name,
            enabled=self.enabled,
            priority=self.priority,
            severity=self.severity,
            block_on_violation=self.block_on_violation,
            reduce_on_violation=self.reduce_on_violation,
        )
