"""SQLAlchemy ORM models for PostgreSQL persistence.

Models for signals, orders, fills, positions, and journal entries.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class SignalRecord(Base):
    """Database record for trading signals."""
    
    __tablename__ = "signals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    signal_id = Column(String(64), unique=True, nullable=False, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    direction = Column(String(16), nullable=False)  # long, short, neutral
    confidence = Column(Numeric(5, 4), nullable=False)  # 0.0000 to 1.0000
    strength = Column(String(16), nullable=True)  # weak, moderate, strong
    
    # Source attribution
    source = Column(String(64), nullable=False)  # engine name
    source_version = Column(String(32), nullable=True)
    trace_id = Column(String(64), nullable=True)
    
    # Signal features stored as JSON
    features = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    # Lifecycle
    generated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(16), default="active")  # active, expired, executed
    
    # Relationships
    orders = relationship("OrderRecord", back_populates="signal")
    
    # Indexes
    __table_args__ = (
        Index("ix_signals_symbol_time", "symbol", "generated_at"),
        Index("ix_signals_confidence", "confidence"),
    )


class OrderRecord(Base):
    """Database record for orders."""
    
    __tablename__ = "orders"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    order_id = Column(String(64), unique=True, nullable=False, index=True)
    broker_order_id = Column(String(64), nullable=True, index=True)
    
    # Symbol and side
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False)  # buy, sell
    order_type = Column(String(16), nullable=False)  # market, limit, stop
    
    # Quantity
    requested_quantity = Column(Numeric(24, 8), nullable=False)
    filled_quantity = Column(Numeric(24, 8), default=Decimal("0"))
    remaining_quantity = Column(Numeric(24, 8), nullable=True)
    
    # Pricing
    limit_price = Column(Numeric(24, 8), nullable=True)
    stop_price = Column(Numeric(24, 8), nullable=True)
    avg_fill_price = Column(Numeric(24, 8), nullable=True)
    
    # Risk
    risk_assessment_id = Column(String(64), nullable=True)
    risk_score = Column(Numeric(5, 4), nullable=True)
    
    # State machine
    status = Column(String(16), nullable=False, default="created")  # matches OrderState
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    signal_id = Column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=True)
    signal = relationship("SignalRecord", back_populates="orders")
    fills = relationship("FillRecord", back_populates="order")
    
    # Metadata
    metadata = Column(JSON, default=dict)
    cancellation_reason = Column(String(255), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_orders_symbol_status", "symbol", "status"),
        Index("ix_orders_created_at", "created_at"),
    )


class FillRecord(Base):
    """Database record for order fills/executions."""
    
    __tablename__ = "fills"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    fill_id = Column(String(64), unique=True, nullable=False)
    
    # Link to order
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False)
    order = relationship("OrderRecord", back_populates="fills")
    
    # Execution details
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    filled_quantity = Column(Numeric(24, 8), nullable=False)
    filled_price = Column(Numeric(24, 8), nullable=False)
    commission = Column(Numeric(24, 8), default=Decimal("0"))
    
    # Remaining after this fill
    remaining_quantity = Column(Numeric(24, 8), nullable=True)
    
    # Timestamp
    filled_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Broker data
    broker_execution_id = Column(String(64), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_fills_symbol_time", "symbol", "filled_at"),
    )


class PositionRecord(Base):
    """Database record for portfolio positions."""
    
    __tablename__ = "positions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Position key
    symbol = Column(String(32), nullable=False)
    account_id = Column(String(64), nullable=False, default="default")
    
    # Current state
    side = Column(String(8), nullable=False)  # long, short, flat
    quantity = Column(Numeric(24, 8), nullable=False, default=Decimal("0"))
    avg_entry_price = Column(Numeric(24, 8), nullable=True)
    
    # Current market
    current_price = Column(Numeric(24, 8), nullable=True)
    market_value = Column(Numeric(24, 8), nullable=True)
    
    # P&L
    unrealized_pnl = Column(Numeric(24, 8), default=Decimal("0"))
    realized_pnl = Column(Numeric(24, 8), default=Decimal("0"))
    total_pnl = Column(Numeric(24, 8), default=Decimal("0"))
    
    # Exposure
    exposure_pct = Column(Numeric(5, 4), nullable=True)  # % of portfolio
    
    # Lifecycle
    opened_at = Column(DateTime(timezone=True), nullable=True)
    last_updated = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    entry_orders = Column(JSON, default=list)  # List of order IDs
    exit_orders = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("symbol", "account_id", name="uix_position_symbol_account"),
        Index("ix_positions_symbol", "symbol"),
        Index("ix_positions_side", "side"),
    )


class PortfolioSnapshot(Base):
    """Database record for portfolio snapshots."""
    
    __tablename__ = "portfolio_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    snapshot_id = Column(String(64), unique=True, nullable=False)
    account_id = Column(String(64), nullable=False, default="default")
    
    # Values
    total_value = Column(Numeric(24, 8), nullable=False)
    cash = Column(Numeric(24, 8), nullable=False)
    margin_used = Column(Numeric(24, 8), default=Decimal("0"))
    buying_power = Column(Numeric(24, 8), nullable=False)
    
    # Exposure
    gross_exposure = Column(Numeric(24, 8), nullable=True)
    net_exposure = Column(Numeric(24, 8), nullable=True)
    long_exposure = Column(Numeric(24, 8), nullable=True)
    short_exposure = Column(Numeric(24, 8), nullable=True)
    
    # Risk metrics
    var_95 = Column(Numeric(24, 8), nullable=True)
    beta = Column(Numeric(5, 4), nullable=True)
    volatility = Column(Numeric(5, 4), nullable=True)
    
    # Drawdown
    peak_value = Column(Numeric(24, 8), nullable=True)
    current_drawdown = Column(Numeric(5, 4), nullable=True)
    max_drawdown = Column(Numeric(5, 4), nullable=True)
    
    # Metadata
    position_count = Column(Integer, default=0)
    sector_exposure = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_snapshots_time", "timestamp"),
    )


class RiskEvent(Base):
    """Database record for risk engine events."""
    
    __tablename__ = "risk_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    assessment_id = Column(String(64), nullable=False, index=True)
    
    # Event type
    event_type = Column(String(32), nullable=False)  # assessment, violation, kill_switch
    
    # Order/Symbol info
    order_id = Column(String(64), nullable=True, index=True)
    symbol = Column(String(32), nullable=True, index=True)
    
    # Assessment
    verdict = Column(String(16), nullable=True)  # approved, rejected, reduced
    risk_score = Column(Numeric(5, 4), nullable=True)
    
    # Violations (stored as JSON array)
    violations = Column(JSON, default=list)
    
    # Emergency
    kill_switch_triggered = Column(Boolean, default=False)
    emergency_liquidation = Column(Boolean, default=False)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Indexes
    __table_args__ = (
        Index("ix_risk_events_type_time", "event_type", "timestamp"),
    )


class JournalEntry(Base):
    """Database record for decision journal entries."""
    
    __tablename__ = "journal_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    record_id = Column(String(64), unique=True, nullable=False, index=True)
    
    # Decision info
    decision_type = Column(String(32), nullable=False)  # trade, rebalance, etc.
    status = Column(String(16), nullable=False)  # pending, executed, failed
    
    # Symbol and direction
    symbol = Column(String(32), nullable=True, index=True)
    direction = Column(String(16), nullable=True)  # long, short, neutral
    
    # Sizing
    intended_size = Column(Numeric(24, 8), nullable=True)
    executed_size = Column(Numeric(24, 8), nullable=True)
    
    # Attribution
    features = Column(JSON, default=list)
    context = Column(JSON, default=dict)
    
    # Outcome
    outcome = Column(JSON, nullable=True)  # pnl, exit_price, etc.
    
    # Timestamps
    decided_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    executed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Trace
    trace_id = Column(String(64), nullable=True, index=True)
    correlation_id = Column(String(64), nullable=True)
    
    # Full rationale
    rationale = Column(Text, nullable=True)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    
    # Indexes
    __table_args__ = (
        Index("ix_journal_decided_at", "decided_at"),
        Index("ix_journal_type_status", "decision_type", "status"),
    )


class EventLog(Base):
    """Database record for event bus events (for replay)."""
    
    __tablename__ = "event_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    event_id = Column(String(64), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    
    # Context
    trace_id = Column(String(64), nullable=True, index=True)
    correlation_id = Column(String(64), nullable=True)
    source_component = Column(String(64), nullable=True)
    
    # Content
    priority = Column(String(16), nullable=True)
    payload = Column(JSON, nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_event_log_type_time", "event_type", "timestamp"),
        Index("ix_event_log_trace", "trace_id"),
    )


class MetricRecord(Base):
    """Database record for time-series metrics."""
    
    __tablename__ = "metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Metric identification
    metric_name = Column(String(128), nullable=False, index=True)
    metric_type = Column(String(16), nullable=False)  # counter, gauge, histogram
    
    # Labels (dimensions)
    labels = Column(JSON, default=dict)
    
    # Value
    value = Column(Numeric(24, 8), nullable=False)
    
    # Timestamp (primary index for time-series)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Indexes - for time-series queries
    __table_args__ = (
        Index("ix_metrics_name_time", "metric_name", "timestamp"),
    )
