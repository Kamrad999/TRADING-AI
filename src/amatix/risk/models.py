"""Risk models for AMATIS Guardian Risk Engine.

Dataclasses for risk assessment, rules, and violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import whenever

from amatix.interfaces import Order, Position, Symbol


class RiskVerdict(Enum):
    """Risk assessment verdict."""
    APPROVED = "approved"
    REJECTED = "rejected"
    REDUCED = "reduced"  # Approved with reduced size
    PENDING = "pending"  # Requires manual review
    EMERGENCY_HALT = "emergency_halt"  # Kill switch activated


class RiskSeverity(Enum):
    """Severity of risk violation."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class RiskViolation:
    """Individual risk rule violation."""
    rule_name: str
    severity: RiskSeverity
    message: str
    current_value: Any
    limit_value: Any
    symbol: Optional[Symbol] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAdjustment:
    """Suggested adjustment to reduce risk."""
    parameter: str  # "size", "leverage", "exposure"
    original_value: Decimal
    suggested_value: Decimal
    reason: str


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    assessment_id: UUID
    timestamp: datetime
    
    # Verdict
    verdict: RiskVerdict
    final_size: Decimal  # May be reduced from original
    
    # Violations
    violations: List[RiskViolation] = field(default_factory=list)
    
    # Adjustments
    adjustments: List[RiskAdjustment] = field(default_factory=list)
    
    # Scores
    risk_score: float = 0.0  # 0.0 (safe) to 1.0 (dangerous)
    confidence: float = 1.0  # Assessment confidence
    
    # Attribution
    rules_evaluated: List[str] = field(default_factory=list)
    evaluation_time_ms: float = 0.0
    
    # Emergency
    kill_switch_triggered: bool = False
    emergency_liquidation: bool = False
    
    @property
    def is_approved(self) -> bool:
        """Check if order is approved."""
        return self.verdict == RiskVerdict.APPROVED
    
    @property
    def is_rejected(self) -> bool:
        """Check if order is rejected."""
        return self.verdict in [
            RiskVerdict.REJECTED,
            RiskVerdict.EMERGENCY_HALT,
        ]
    
    @property
    def has_critical_violations(self) -> bool:
        """Check for critical or fatal violations."""
        return any(
            v.severity in [RiskSeverity.CRITICAL, RiskSeverity.FATAL]
            for v in self.violations
        )
    
    @classmethod
    def create(
        cls,
        verdict: RiskVerdict,
        final_size: Decimal,
    ) -> RiskAssessment:
        """Factory method to create assessment."""
        return cls(
            assessment_id=uuid4(),
            timestamp=whenever.now().py_datetime(),
            verdict=verdict,
            final_size=final_size,
        )


@dataclass
class RiskRule:
    """Configuration for a risk rule."""
    name: str
    enabled: bool = True
    priority: int = 50  # Lower = evaluated first
    severity: RiskSeverity = RiskSeverity.WARNING
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Rule behavior
    block_on_violation: bool = False
    reduce_on_violation: bool = False
    reduction_factor: float = 0.5


@dataclass
class RiskSnapshot:
    """Point-in-time risk snapshot."""
    snapshot_id: UUID
    timestamp: datetime
    
    # Portfolio state
    portfolio_value: Decimal
    cash: Decimal
    margin_used: Decimal
    buying_power: Decimal
    
    # Exposure metrics
    gross_exposure: Decimal
    net_exposure: Decimal
    long_exposure: Decimal
    short_exposure: Decimal
    
    # Concentration
    top_positions: List[tuple]  # (symbol, exposure_pct)
    sector_exposure: Dict[str, Decimal]
    
    # Risk metrics
    var_95: Decimal  # Value at Risk
    expected_shortfall: Decimal
    beta: float
    volatility: float
    
    # Drawdown
    peak_value: Decimal
    current_drawdown: float  # Percentage
    max_drawdown: float
    
    # Limits
    limits_used: Dict[str, float]  # Percentage of limit used
    
    @property
    def is_healthy(self) -> bool:
        """Check if portfolio is within healthy risk bounds."""
        return (
            self.current_drawdown < 0.05 and  # < 5% drawdown
            self.gross_exposure < self.buying_power * Decimal("1.5") and
            len(self.top_positions) > 0 and self.top_positions[0][1] < 0.20  # No >20% positions
        )


@dataclass
class RiskConfig:
    """Global risk configuration."""
    # Position limits
    max_position_size: Decimal = Decimal("100000")  # Max $ per position
    max_position_pct: float = 0.20  # Max 20% of portfolio
    max_open_positions: int = 20
    
    # Portfolio limits
    max_gross_exposure: float = 2.0  # 200% gross
    max_net_exposure: float = 1.0  # 100% net
    max_leverage: float = 2.0
    
    # Sector limits
    max_sector_exposure: float = 0.40  # 40% per sector
    
    # Drawdown
    max_daily_drawdown: float = 0.03  # 3% daily
    max_total_drawdown: float = 0.10  # 10% total
    
    # Symbol limits
    max_symbols_per_sector: int = 5
    min_liquidity: Decimal = Decimal("1000000")  # Min daily volume
    max_spread_bps: float = 50.0  # Max 50bps spread
    
    # Volatility
    max_volatility: float = 0.50  # Max 50% annualized vol
    volatility_scaling: bool = True
    
    # Emergency
    kill_switch_drawdown: float = 0.15  # 15% triggers kill
    circuit_breaker_enabled: bool = True
    
    # Time-based
    market_hours_only: bool = True
    macro_event_freeze: bool = True
