"""Signal models for AMATIS.

Enhanced signal representation with:
    - Multi-dimensional scoring
    - Feature attribution
    - Rationale tracking
    - Timeframe specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import whenever

from amatix.data.market.models import Symbol


class SignalDirection(Enum):
    """Direction of trading signal."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    EXIT = "exit"  # Signal to close position
    HOLD = "hold"  # Do nothing


class SignalStrength(Enum):
    """Strength categorization."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4


class SignalTimeframe(Enum):
    """Expected signal duration."""
    SCALP = "scalp"        # Minutes
    INTRADAY = "intraday"  # Hours
    SWING = "swing"        # Days
    POSITION = "position"  # Weeks
    LONG_TERM = "long_term"  # Months


@dataclass(frozen=True)
class SignalFeature:
    """Feature that contributed to signal generation."""
    name: str
    value: Any
    weight: float  # 0.0 to 1.0
    category: str  # "technical", "fundamental", "sentiment", "risk"
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": str(self.value) if isinstance(self.value, Decimal) else self.value,
            "weight": self.weight,
            "category": self.category,
        }


@dataclass
class Signal:
    """Trading signal with full attribution.
    
    Represents a decision to trade (or not) with complete
    explanation of why the decision was made.
    """
    # Identity
    signal_id: UUID
    symbol: Symbol
    created_at: datetime
    trace_id: UUID
    
    # Core signal
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    strength: SignalStrength
    timeframe: SignalTimeframe
    
    # Source attribution
    source: str  # Engine/strategy that generated
    source_version: str  # Version for tracking changes
    
    # Features (explainability)
    features: List[SignalFeature] = field(default_factory=list)
    
    # Price levels
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size_pct: Optional[Decimal] = None  # Suggested position size
    
    # Metadata
    rationale: str = ""  # Human-readable explanation
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle
    expires_at: Optional[datetime] = None
    executed: bool = False
    execution_id: Optional[UUID] = None
    
    def __post_init__(self):
        """Validate signal."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
    
    @classmethod
    def create(
        cls,
        symbol: Symbol,
        direction: SignalDirection,
        confidence: float,
        source: str,
        trace_id: UUID,
        strength: SignalStrength = SignalStrength.MODERATE,
        timeframe: SignalTimeframe = SignalTimeframe.INTRADAY,
        source_version: str = "1.0",
    ) -> Signal:
        """Factory method to create a new signal."""
        return cls(
            signal_id=uuid4(),
            symbol=symbol,
            created_at=whenever.now().py_datetime(),
            trace_id=trace_id,
            direction=direction,
            confidence=confidence,
            strength=strength,
            timeframe=timeframe,
            source=source,
            source_version=source_version,
            expires_at=cls._calculate_expiry(timeframe),
        )
    
    @staticmethod
    def _calculate_expiry(timeframe: SignalTimeframe) -> datetime:
        """Calculate default expiry based on timeframe."""
        now = whenever.now().py_datetime()
        
        durations = {
            SignalTimeframe.SCALP: timedelta(minutes=30),
            SignalTimeframe.INTRADAY: timedelta(hours=4),
            SignalTimeframe.SWING: timedelta(days=2),
            SignalTimeframe.POSITION: timedelta(weeks=1),
            SignalTimeframe.LONG_TERM: timedelta(weeks=4),
        }
        
        return now + durations.get(timeframe, timedelta(hours=4))
    
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.expires_at is None:
            return False
        return whenever.now().py_datetime() > self.expires_at
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (not expired, not executed)."""
        return not self.is_expired and not self.executed
    
    @property
    def top_features(self) -> List[SignalFeature]:
        """Get top contributing features sorted by weight."""
        return sorted(
            self.features,
            key=lambda f: f.weight,
            reverse=True,
        )[:5]
    
    def with_features(self, features: List[SignalFeature]) -> Signal:
        """Return signal with features added."""
        new_signal = Signal(
            signal_id=self.signal_id,
            symbol=self.symbol,
            created_at=self.created_at,
            trace_id=self.trace_id,
            direction=self.direction,
            confidence=self.confidence,
            strength=self.strength,
            timeframe=self.timeframe,
            source=self.source,
            source_version=self.source_version,
            features=features,
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            position_size_pct=self.position_size_pct,
            rationale=self.rationale,
            tags=self.tags,
            metadata=self.metadata,
            expires_at=self.expires_at,
            executed=self.executed,
            execution_id=self.execution_id,
        )
        return new_signal
    
    def with_price_levels(
        self,
        entry: Decimal,
        stop: Optional[Decimal] = None,
        target: Optional[Decimal] = None,
    ) -> Signal:
        """Return signal with price levels set."""
        new_signal = Signal(
            signal_id=self.signal_id,
            symbol=self.symbol,
            created_at=self.created_at,
            trace_id=self.trace_id,
            direction=self.direction,
            confidence=self.confidence,
            strength=self.strength,
            timeframe=self.timeframe,
            source=self.source,
            source_version=self.source_version,
            features=self.features,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            position_size_pct=self.position_size_pct,
            rationale=self.rationale,
            tags=self.tags,
            metadata=self.metadata,
            expires_at=self.expires_at,
            executed=self.executed,
            execution_id=self.execution_id,
        )
        return new_signal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_id": str(self.signal_id),
            "symbol": str(self.symbol),
            "direction": self.direction.value,
            "confidence": self.confidence,
            "strength": self.strength.name,
            "timeframe": self.timeframe.value,
            "source": self.source,
            "is_actionable": self.is_actionable,
            "is_expired": self.is_expired,
            "features": [f.to_dict() for f in self.top_features],
            "entry_price": str(self.entry_price) if self.entry_price else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "rationale": self.rationale[:200] if self.rationale else "",
        }


@dataclass
class SignalBatch:
    """Batch of signals for processing."""
    signals: List[Signal]
    source: str
    created_at: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    
    @property
    def count(self) -> int:
        return len(self.signals)
    
    @property
    def actionable_signals(self) -> List[Signal]:
        """Get only actionable (not expired) signals."""
        return [s for s in self.signals if s.is_actionable]
    
    def by_direction(self, direction: SignalDirection) -> List[Signal]:
        """Filter by direction."""
        return [s for s in self.signals if s.direction == direction]


@dataclass
class SignalFilterConfig:
    """Configuration for signal filtering."""
    min_confidence: float = 0.70
    max_signals_per_symbol: int = 3
    max_age_minutes: int = 60
    allowed_directions: List[SignalDirection] = None
    
    def __post_init__(self):
        if self.allowed_directions is None:
            self.allowed_directions = [
                SignalDirection.LONG,
                SignalDirection.SHORT,
            ]
