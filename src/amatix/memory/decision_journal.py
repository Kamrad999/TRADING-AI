"""Decision journal for explainability and auditability.

Every trading decision is recorded with:
    - Full context (what was known at decision time)
    - Rationale (why the decision was made)
    - Features (what data points influenced the decision)
    - Confidence (how certain was the system)
    - Outcome (what happened - filled in later)

This enables:
    - Post-trade analysis
    - Strategy debugging
    - Alpha decay detection
    - RL training data generation
    - Compliance reporting
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import whenever

from amatix.core.event_models import Event, EventType
from amatix.interfaces import Order, Signal, Symbol


class DecisionType(Enum):
    """Types of trading decisions."""
    SIGNAL_GENERATED = auto()
    SIGNAL_REJECTED = auto()
    ORDER_SUBMITTED = auto()
    ORDER_REJECTED = auto()
    POSITION_SIZED = auto()
    KILL_SWITCH_TRIGGERED = auto()


class DecisionStatus(Enum):
    """Status of a decision in its lifecycle."""
    PENDING = auto()      # Decision made, outcome unknown
    CONFIRMED = auto()    # Outcome confirmed (e.g., order filled)
    REJECTED = auto()     # Decision blocked by risk/system
    EXPIRED = auto()      # Signal/decision timed out
    CANCELLED = auto()    # Manually cancelled


@dataclass
class FeatureSnapshot:
    """Feature values at decision time.
    
    Captures the exact state of features that influenced a decision,
    enabling:
        - Reproducibility (know what was known)
        - Attribution (which features mattered)
        - Debugging (why did this happen?)
    """
    name: str
    value: Any
    importance: Optional[float] = None  # Feature importance if available
    category: str = "general"  # e.g., "technical", "fundamental", "sentiment"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": str(self.value) if isinstance(self.value, Decimal) else self.value,
            "importance": self.importance,
            "category": self.category,
        }


@dataclass
class ContextSnapshot:
    """Complete context at decision time.
    
    Everything the system knew when making this decision.
    This is a snapshot - it doesn't change even if the world changes.
    """
    timestamp: datetime
    symbol: Optional[Symbol] = None
    
    # Market context
    current_price: Optional[Decimal] = None
    bid_ask_spread: Optional[Decimal] = None
    market_regime: Optional[str] = None
    volatility_percentile: Optional[float] = None
    
    # Portfolio context
    portfolio_value: Optional[Decimal] = None
    available_cash: Optional[Decimal] = None
    current_position: Optional[Decimal] = None
    daily_pnl: Optional[Decimal] = None
    
    # Risk context
    current_drawdown: Optional[float] = None
    portfolio_heat: Optional[float] = None
    
    # System context
    signal_confidence: Optional[float] = None
    risk_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "symbol": str(self.symbol) if self.symbol else None,
            "market": {
                "current_price": str(self.current_price) if self.current_price else None,
                "bid_ask_spread": str(self.bid_ask_spread) if self.bid_ask_spread else None,
                "regime": self.market_regime,
                "volatility_percentile": self.volatility_percentile,
            },
            "portfolio": {
                "value": str(self.portfolio_value) if self.portfolio_value else None,
                "cash": str(self.available_cash) if self.available_cash else None,
                "position": str(self.current_position) if self.current_position else None,
                "daily_pnl": str(self.daily_pnl) if self.daily_pnl else None,
            },
            "risk": {
                "drawdown": self.current_drawdown,
                "heat": self.portfolio_heat,
            },
            "system": {
                "signal_confidence": self.signal_confidence,
                "risk_score": self.risk_score,
            },
        }
        return result


@dataclass
class DecisionRationale:
    """Human-readable and machine-readable rationale.
    
    Explains WHY a decision was made in multiple formats:
        - Summary: One-line explanation
        - Detailed: Full reasoning
        - Factors: List of contributing factors
        - Rules: Which rules triggered
    """
    summary: str  # e.g., "Buy AAPL: momentum + earnings beat"
    detailed: str  # Full explanation
    contributing_factors: List[str] = field(default_factory=list)
    rules_triggered: List[str] = field(default_factory=list)
    counter_indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "detailed": self.detailed,
            "contributing_factors": self.contributing_factors,
            "rules_triggered": self.rules_triggered,
            "counter_indicators": self.counter_indicators,
        }


@dataclass
class DecisionOutcome:
    """Outcome of a decision (filled in later).
    
    Links the decision to what actually happened,
    enabling performance attribution and learning.
    """
    status: DecisionStatus
    timestamp: Optional[datetime] = None
    
    # For executed orders
    fill_price: Optional[Decimal] = None
    fill_quantity: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    slippage_bps: Optional[float] = None
    
    # Performance tracking
    exit_price: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    holding_period_minutes: Optional[int] = None
    
    # Meta
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "execution": {
                "fill_price": str(self.fill_price) if self.fill_price else None,
                "fill_quantity": str(self.fill_quantity) if self.fill_quantity else None,
                "commission": str(self.commission) if self.commission else None,
                "slippage_bps": self.slippage_bps,
            },
            "performance": {
                "exit_price": str(self.exit_price) if self.exit_price else None,
                "realized_pnl": str(self.realized_pnl) if self.realized_pnl else None,
                "holding_period_minutes": self.holding_period_minutes,
            },
            "notes": self.notes,
            "tags": self.tags,
        }


@dataclass
class DecisionRecord:
    """Complete record of a single trading decision.
    
    This is the core data structure for the decision journal.
    Every significant system action creates one of these.
    """
    decision_id: UUID
    decision_type: DecisionType
    timestamp: datetime
    trace_id: UUID
    
    # What was decided
    signal: Optional[Signal] = None
    order: Optional[Order] = None
    
    # Context (what was known)
    context: ContextSnapshot = field(default_factory=lambda: ContextSnapshot(
        timestamp=whenever.now().py_datetime()
    ))
    
    # Features (what data influenced this)
    features: List[FeatureSnapshot] = field(default_factory=list)
    
    # Rationale (why was this decided)
    rationale: DecisionRationale = field(default_factory=lambda: DecisionRationale(
        summary="", detailed=""
    ))
    
    # Confidence (how certain was the system)
    confidence: float = 0.0
    confidence_calibration: Optional[str] = None  # e.g., "well_calibrated", "over_confident"
    
    # Outcome (what happened - filled in later)
    outcome: DecisionOutcome = field(default_factory=lambda: DecisionOutcome(
        status=DecisionStatus.PENDING
    ))
    
    # Meta
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure timestamps are set."""
        if self.context.timestamp is None:
            self.context.timestamp = whenever.now().py_datetime()
    
    @classmethod
    def create(
        cls,
        decision_type: DecisionType,
        trace_id: UUID,
        signal: Optional[Signal] = None,
        order: Optional[Order] = None,
    ) -> DecisionRecord:
        """Factory method to create a new decision record."""
        return cls(
            decision_id=uuid4(),
            decision_type=decision_type,
            timestamp=whenever.now().py_datetime(),
            trace_id=trace_id,
            signal=signal,
            order=order,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision_id": str(self.decision_id),
            "decision_type": self.decision_type.name,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": str(self.trace_id),
            "signal": self._signal_to_dict() if self.signal else None,
            "order": self._order_to_dict() if self.order else None,
            "context": self.context.to_dict(),
            "features": [f.to_dict() for f in self.features],
            "rationale": self.rationale.to_dict(),
            "confidence": self.confidence,
            "confidence_calibration": self.confidence_calibration,
            "outcome": self.outcome.to_dict(),
            "version": self.version,
            "metadata": self.metadata,
        }
    
    def _signal_to_dict(self) -> Dict[str, Any]:
        """Convert signal to dict."""
        if not self.signal:
            return {}
        return {
            "symbol": str(self.signal.symbol),
            "direction": self.signal.direction.name,
            "confidence": self.signal.confidence,
            "strength": self.signal.strength,
            "source": self.signal.source,
        }
    
    def _order_to_dict(self) -> Dict[str, Any]:
        """Convert order to dict."""
        if not self.order:
            return {}
        return {
            "symbol": str(self.order.symbol),
            "side": self.order.side.name,
            "quantity": str(self.order.quantity),
            "order_type": self.order.order_type.name,
            "limit_price": str(self.order.limit_price) if self.order.limit_price else None,
        }
    
    def update_outcome(self, outcome: DecisionOutcome) -> None:
        """Update the outcome of this decision."""
        self.outcome = outcome
    
    def add_feature(self, feature: FeatureSnapshot) -> None:
        """Add a feature that influenced this decision."""
        self.features.append(feature)
    
    def set_rationale(self, rationale: DecisionRationale) -> None:
        """Set the decision rationale."""
        self.rationale = rationale


class DecisionJournal:
    """Journal for recording and querying trading decisions.
    
    The decision journal is the foundation for:
        - Explainability (why did we do that?)
        - Debugging (what went wrong?)
        - Compliance (audit trail)
        - Learning (what worked?)
        - Alpha decay detection (is this strategy still working?)
    
    Storage backends:
        - JSON files (default, for development)
        - PostgreSQL (for production)
        - TimescaleDB (for time-series queries)
    
    Example:
        >>> journal = DecisionJournal("./decision_journal")
        >>> 
        >>> record = DecisionRecord.create(
        ...     decision_type=DecisionType.ORDER_SUBMITTED,
        ...     trace_id=uuid4(),
        ...     signal=signal,
        ...     order=order,
        ... )
        >>> record.set_rationale(DecisionRationale(
        ...     summary="Buy AAPL on momentum",
        ...     detailed="RSI broke above 70 with volume surge",
        ... ))
        >>> 
        >>> await journal.record(decision)
    """
    
    def __init__(self, storage_path: str = "./decision_journal") -> None:
        """Initialize the decision journal.
        
        Args:
            storage_path: Directory for storing decision records
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for recent decisions
        self._recent_decisions: List[DecisionRecord] = []
        self._max_cache_size = 1000
    
    async def record(self, decision: DecisionRecord) -> None:
        """Record a decision to the journal.
        
        Args:
            decision: The decision to record
        """
        # Add to cache
        self._recent_decisions.append(decision)
        if len(self._recent_decisions) > self._max_cache_size:
            self._recent_decisions.pop(0)
        
        # Write to storage (async file I/O)
        await self._write_to_storage(decision)
    
    async def _write_to_storage(self, decision: DecisionRecord) -> None:
        """Write decision to persistent storage."""
        # Organize by date for easier querying
        date_str = decision.timestamp.strftime("%Y-%m-%d")
        date_dir = self._storage_path / date_str
        date_dir.mkdir(exist_ok=True)
        
        # Filename includes decision ID for uniqueness
        filename = f"{decision.decision_id}.json"
        filepath = date_dir / filename
        
        # Write atomically
        temp_path = filepath.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(decision.to_dict(), f, indent=2, default=str)
        temp_path.rename(filepath)
    
    async def update_outcome(
        self,
        decision_id: UUID,
        outcome: DecisionOutcome,
    ) -> bool:
        """Update the outcome of a decision.
        
        Args:
            decision_id: ID of decision to update
            outcome: Outcome to set
        
        Returns:
            True if decision found and updated
        """
        # Try to find in cache first
        for decision in reversed(self._recent_decisions):
            if decision.decision_id == decision_id:
                decision.update_outcome(outcome)
                await self._write_to_storage(decision)
                return True
        
        # TODO: Query from persistent storage if not in cache
        return False
    
    async def get_decision(self, decision_id: UUID) -> Optional[DecisionRecord]:
        """Retrieve a specific decision by ID.
        
        Args:
            decision_id: Decision to retrieve
        
        Returns:
            DecisionRecord or None if not found
        """
        # Check cache first
        for decision in self._recent_decisions:
            if decision.decision_id == decision_id:
                return decision
        
        # TODO: Query from storage
        return None
    
    async def query(
        self,
        symbol: Optional[Symbol] = None,
        decision_type: Optional[DecisionType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        outcome_status: Optional[DecisionStatus] = None,
        limit: int = 100,
    ) -> List[DecisionRecord]:
        """Query decisions by criteria.
        
        This is a simplified query - production would use SQL.
        
        Args:
            symbol: Filter by symbol
            decision_type: Filter by type
            start_time: Filter by timestamp >=
            end_time: Filter by timestamp <=
            outcome_status: Filter by outcome status
            limit: Maximum results
        
        Returns:
            List of matching decisions
        """
        results = []
        
        # Search in cache first (most recent)
        for decision in reversed(self._recent_decisions):
            if len(results) >= limit:
                break
            
            if symbol and decision.context.symbol != symbol:
                continue
            if decision_type and decision.decision_type != decision_type:
                continue
            if start_time and decision.timestamp < start_time:
                continue
            if end_time and decision.timestamp > end_time:
                continue
            if outcome_status and decision.outcome.status != outcome_status:
                continue
            
            results.append(decision)
        
        return results
    
    async def get_performance_summary(
        self,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get performance summary for recent decisions.
        
        Args:
            days: Lookback period in days
        
        Returns:
            Summary statistics dict
        """
        # Query recent decisions
        from datetime import timedelta
        start = whenever.now().py_datetime() - timedelta(days=days)
        decisions = await self.query(start_time=start, limit=10000)
        
        # Calculate metrics
        total = len(decisions)
        confirmed = sum(1 for d in decisions if d.outcome.status == DecisionStatus.CONFIRMED)
        rejected = sum(1 for d in decisions if d.outcome.status == DecisionStatus.REJECTED)
        
        pnl_list = [
            d.outcome.realized_pnl for d in decisions
            if d.outcome.realized_pnl is not None
        ]
        
        total_pnl = sum(pnl_list) if pnl_list else Decimal("0")
        avg_confidence = sum(d.confidence for d in decisions) / total if total > 0 else 0
        
        return {
            "period_days": days,
            "total_decisions": total,
            "confirmed": confirmed,
            "rejected": rejected,
            "fill_rate": confirmed / total if total > 0 else 0,
            "total_pnl": str(total_pnl),
            "avg_confidence": avg_confidence,
        }
    
    async def export_for_training(self, filepath: str) -> None:
        """Export decisions as RL training dataset.
        
        Creates a JSONL file suitable for training RL agents,
        with (state, action, reward) tuples.
        
        Args:
            filepath: Output file path
        """
        decisions = await self.query(limit=100000)
        
        with open(filepath, "w") as f:
            for decision in decisions:
                # Only export confirmed decisions with outcomes
                if decision.outcome.status != DecisionStatus.CONFIRMED:
                    continue
                if decision.outcome.realized_pnl is None:
                    continue
                
                training_record = {
                    "context": decision.context.to_dict(),
                    "features": [f.to_dict() for f in decision.features],
                    "action": {
                        "symbol": str(decision.signal.symbol) if decision.signal else None,
                        "direction": decision.signal.direction.name if decision.signal else None,
                    },
                    "reward": float(decision.outcome.realized_pnl),
                    "timestamp": decision.timestamp.isoformat(),
                }
                f.write(json.dumps(training_record) + "\n")


class FeatureAttribution:
    """Track feature importance for explainability.
    
    Records which features were most influential in decisions,
    enabling:
        - Feature debugging
        - Model interpretability
        - Alpha attribution
    """
    
    def __init__(self) -> None:
        self._feature_counts: Dict[str, int] = {}
        self._feature_importance: Dict[str, List[float]] = {}
    
    def record_features(self, features: List[FeatureSnapshot]) -> None:
        """Record features used in a decision."""
        for feature in features:
            self._feature_counts[feature.name] = self._feature_counts.get(feature.name, 0) + 1
            
            if feature.importance is not None:
                if feature.name not in self._feature_importance:
                    self._feature_importance[feature.name] = []
                self._feature_importance[feature.name].append(feature.importance)
    
    def get_top_features(self, n: int = 10) -> List[tuple]:
        """Get most frequently used features."""
        sorted_features = sorted(
            self._feature_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]
    
    def get_average_importance(self) -> Dict[str, float]:
        """Get average importance score per feature."""
        return {
            name: sum(values) / len(values)
            for name, values in self._feature_importance.items()
            if values
        }
