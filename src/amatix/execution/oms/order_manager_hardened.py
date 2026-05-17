"""Hardened Order Management System for AMATIS.

CRITICAL FIXES from VERIFICATION_AUDIT:
    1. Fill deduplication by execution_id
    2. Broker reconciliation
    3. Orphan order detection
    4. Fill validation (qty, price bounds)
    5. Partial fill torture test support
    6. Atomic state transitions
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import whenever

if TYPE_CHECKING:
    from collections.abc import Callable

    from datetime import datetime

    from amatix.core.event_bus_v2 import HardenedEventBusV2

from amatix.core.event_models import EventPriority, EventType
from amatix.core.observability import get_logger, get_metrics
from amatix.execution.oms.state_machine import (
    InvalidStateTransitionError,
    OrderState,
)

logger = get_logger(__name__)


class FillValidationError(Exception):
    """Raised when fill validation fails."""

    pass


class DuplicateFillError(Exception):
    """Raised when duplicate fill detected."""

    pass


@dataclass
class ReconciliationReport:
    """Report from broker reconciliation."""

    discrepancies: list[dict[str, Any]]
    orphaned_orders: list[UUID]
    timestamp: datetime = field(default_factory=lambda: whenever.now().py_datetime())

    @property
    def is_clean(self) -> bool:
        """Check if reconciliation found no issues."""
        return len(self.discrepancies) == 0 and len(self.orphaned_orders) == 0


@dataclass
class FillRecord:
    """Record of a single fill/execution."""

    execution_id: str
    filled_quantity: Decimal
    filled_price: Decimal
    commission: Decimal
    timestamp: datetime

    def validate(self, order_quantity: Decimal, remaining: Decimal) -> None:
        """Validate fill against order constraints."""
        # Check for positive values
        if self.filled_quantity <= 0:
            raise FillValidationError(f"Fill quantity must be positive: {self.filled_quantity}")

        if self.filled_price <= 0:
            raise FillValidationError(f"Fill price must be positive: {self.filled_price}")

        # Check against remaining quantity
        if self.filled_quantity > remaining:
            raise FillValidationError(
                f"Fill quantity {self.filled_quantity} exceeds remaining {remaining}"
            )

        # Sanity check on price (within 50% of reasonable range)
        # In production, this would be based on market data
        if self.filled_price > Decimal("1000000"):
            raise FillValidationError(f"Fill price {self.filled_price} exceeds sanity limit")


class HardenedOrderEntry:
    """Order entry with hardened tracking.

    Additional features:
        - Fill deduplication
        - Execution tracking
        - Validation
    """

    def __init__(
        self,
        order_id: UUID,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> None:
        """Initialize order entry."""
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price

        # State machine
        self.state = OrderState.CREATED
        self.state_history: list[tuple] = [(self.state, {}, whenever.now().py_datetime())]

        # Fill tracking with deduplication
        self.filled_quantity = Decimal("0")
        self.avg_fill_price = Decimal("0")
        self.total_commission = Decimal("0")
        self.fills: list[FillRecord] = []
        self.processed_execution_ids: set[str] = set()  # DEDUPLICATION

        # Metadata
        self.created_at = whenever.now().py_datetime()
        self.updated_at = self.created_at
        self.submitted_at: datetime | None = None
        self.acknowledged_at: datetime | None = None
        self.broker_order_id: str | None = None

        # Reconciliation
        self.last_reconcile_at: datetime | None = None
        self.reconcile_discrepancies: list[str] = []

    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity."""
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        """Check if order is in terminal state."""
        return self.state in {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }

    @property
    def is_filled(self) -> bool:
        """Check if fully filled."""
        return self.filled_quantity >= self.quantity

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate as percentage."""
        if self.quantity == 0:
            return 0.0
        try:
            return float(self.filled_quantity / self.quantity)
        except InvalidOperation:
            return 0.0

    def can_transition(self, new_state: OrderState) -> bool:
        """Check if state transition is valid."""
        from amatix.execution.oms.state_machine import OrderStateMachine

        if self.is_complete:
            return False

        valid_transitions = OrderStateMachine.VALID_TRANSITIONS.get(self.state, set())
        return new_state in valid_transitions

    def transition(self, new_state: OrderState, metadata: dict[str, Any] | None = None) -> None:
        """Transition to new state with validation."""
        if not self.can_transition(new_state):
            raise InvalidStateTransitionError(
                f"Cannot transition from {self.state.name} to {new_state.name}"
            )

        old_state = self.state
        self.state = new_state
        self.updated_at = whenever.now().py_datetime()

        self.state_history.append((new_state, metadata or {}, self.updated_at))

        logger.debug(
            "Order state transition",
            order_id=str(self.order_id),
            from_state=old_state.name,
            to_state=new_state.name,
        )

    def add_fill(self, fill: FillRecord) -> None:
        """Add fill with deduplication check."""
        # DEDUPLICATION CHECK
        if fill.execution_id in self.processed_execution_ids:
            raise DuplicateFillError(
                f"Execution {fill.execution_id} already processed for order {self.order_id}"
            )

        # Validate fill
        fill.validate(self.quantity, self.remaining_quantity)

        # Add to processed set (DEDUPLICATION)
        self.processed_execution_ids.add(fill.execution_id)

        # Calculate new average price (weighted average)
        if self.filled_quantity > 0:
            total_value = (
                self.avg_fill_price * self.filled_quantity
                + fill.filled_price * fill.filled_quantity
            )
            self.avg_fill_price = total_value / (self.filled_quantity + fill.filled_quantity)
        else:
            self.avg_fill_price = fill.filled_price

        self.filled_quantity += fill.filled_quantity
        self.total_commission += fill.commission
        self.fills.append(fill)
        self.updated_at = whenever.now().py_datetime()

        # Auto-transition to PARTIALLY_FILLED or FILLED
        if self.is_filled:
            if self.state != OrderState.FILLED:
                self.transition(OrderState.FILLED, {"fill_count": len(self.fills)})
        else:
            if self.state in {OrderState.CREATED, OrderState.SUBMITTED, OrderState.ACKNOWLEDGED}:
                # Note: Would need PARTIALLY_FILLED state in OrderState enum
                pass

    def mark_submitted(self, broker_order_id: str) -> None:
        """Mark order as submitted."""
        self.broker_order_id = broker_order_id
        self.submitted_at = whenever.now().py_datetime()
        self.transition(OrderState.SUBMITTED, {"broker_order_id": broker_order_id})

    def mark_acknowledged(self) -> None:
        """Mark order as acknowledged by broker."""
        self.acknowledged_at = whenever.now().py_datetime()
        self.transition(OrderState.ACKNOWLEDGED)

    def is_orphaned(self, threshold_seconds: float = 60.0) -> bool:
        """Check if order is orphaned (submitted but no ACK for too long)."""
        if self.state != OrderState.SUBMITTED:
            return False

        if not self.submitted_at:
            return False

        elapsed = (whenever.now().py_datetime() - self.submitted_at).total_seconds()
        return elapsed > threshold_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize order entry."""
        return {
            "order_id": str(self.order_id),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": str(self.quantity),
            "order_type": self.order_type,
            "state": self.state.name,
            "filled_quantity": str(self.filled_quantity),
            "remaining_quantity": str(self.remaining_quantity),
            "avg_fill_price": str(self.avg_fill_price),
            "total_commission": str(self.total_commission),
            "fill_count": len(self.fills),
            "fill_rate": self.fill_rate,
            "is_complete": self.is_complete,
            "is_orphaned": self.is_orphaned(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class HardenedOrderManager:
    """Hardened order manager with institutional-grade reliability.

    CRITICAL FIXES:
        - Fill deduplication
        - Broker reconciliation
        - Orphan order detection
        - Fill validation
    """

    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        max_active_orders: int = 1000,
        orphan_threshold_seconds: float = 60.0,
        enable_reconciliation: bool = True,
        reconcile_interval_seconds: float = 60.0,
    ) -> None:
        """Initialize hardened order manager.

        Args:
            event_bus: Event bus for order events
            max_active_orders: Maximum concurrent orders
            orphan_threshold_seconds: Seconds before order considered orphaned
            enable_reconciliation: Enable periodic broker reconciliation
            reconcile_interval_seconds: Reconciliation interval
        """
        self._event_bus = event_bus
        self._max_active_orders = max_active_orders
        self._orphan_threshold = orphan_threshold_seconds
        self._enable_reconciliation = enable_reconciliation
        self._reconcile_interval = reconcile_interval_seconds

        # Order storage
        self._orders: dict[UUID, HardenedOrderEntry] = {}
        self._broker_id_map: dict[str, UUID] = {}

        # Per-order locks for fine-grained concurrency
        self._order_locks: dict[UUID, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        # Reconciliation
        self._reconcile_task: asyncio.Task | None = None
        self._last_reconcile: datetime | None = None

        # Metrics
        self._total_orders = 0
        self._total_fills = 0
        self._duplicate_fills_rejected = 0
        self._orphaned_detected = 0
        self._validation_failures = 0

    async def initialize(self) -> None:
        """Initialize order manager."""
        if self._enable_reconciliation:
            self._reconcile_task = asyncio.create_task(
                self._reconcile_loop(), name="oms_reconciliation"
            )

        logger.info(
            "HardenedOrderManager initialized",
            max_orders=self._max_active_orders,
            orphan_threshold=self._orphan_threshold,
            reconciliation=self._enable_reconciliation,
        )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._reconcile_task:
            self._reconcile_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reconcile_task

        logger.info("HardenedOrderManager shutdown")

    def _get_order_lock(self, order_id: UUID) -> asyncio.Lock:
        """Get or create lock for specific order."""
        if order_id not in self._order_locks:
            self._order_locks[order_id] = asyncio.Lock()
        return self._order_locks[order_id]

    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HardenedOrderEntry:
        """Create new order entry."""
        async with self._global_lock:
            # Check capacity
            if len(self._orders) >= self._max_active_orders:
                raise RuntimeError(f"Maximum orders ({self._max_active_orders}) reached")

            # Create entry
            order_id = uuid4()
            entry = HardenedOrderEntry(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
            )

            # Store
            self._orders[order_id] = entry
            self._total_orders += 1

        # Emit event (outside lock)
        await self._event_bus.emit_new(
            EventType.ORDER_SUBMITTED,
            {
                "order_id": str(entry.order_id),
                "symbol": symbol,
                "side": side,
                "quantity": str(quantity),
                "order_type": order_type,
            },
            priority=EventPriority.HIGH,
            source="order_manager",
        )

        logger.info(
            "Order created",
            order_id=str(entry.order_id),
            symbol=symbol,
            side=side,
            quantity=str(quantity),
        )

        return entry

    async def mark_submitted(
        self,
        order_id: UUID,
        broker_order_id: str,
    ) -> None:
        """Mark order as submitted to broker."""
        async with self._get_order_lock(order_id):
            entry = self._orders.get(order_id)
            if not entry:
                raise ValueError(f"Order {order_id} not found")

            entry.mark_submitted(broker_order_id)
            self._broker_id_map[broker_order_id] = order_id

    async def update_fill(
        self,
        order_id: UUID,
        execution_id: str,
        filled_quantity: Decimal,
        filled_price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> bool:
        """Update order with fill - WITH DEDUPLICATION.

        Returns:
            True if fill was accepted, False if duplicate or error
        """
        async with self._get_order_lock(order_id):
            entry = self._orders.get(order_id)
            if not entry:
                logger.error("Fill for unknown order", order_id=str(order_id))
                return False

            # Check for duplicates
            if execution_id in entry.processed_execution_ids:
                logger.warning(
                    "Duplicate fill rejected",
                    order_id=str(order_id),
                    execution_id=execution_id,
                )
                self._duplicate_fills_rejected += 1
                get_metrics().counter("oms_duplicate_fills_rejected")
                return False

            # Create fill record
            fill = FillRecord(
                execution_id=execution_id,
                filled_quantity=filled_quantity,
                filled_price=filled_price,
                commission=commission,
                timestamp=whenever.now().py_datetime(),
            )

            # Validate and add
            try:
                entry.add_fill(fill)
                self._total_fills += 1
            except (FillValidationError, DuplicateFillError) as e:
                logger.error(
                    "Fill validation failed",
                    order_id=str(order_id),
                    execution_id=execution_id,
                    error=str(e),
                )
                self._validation_failures += 1
                return False

        # Emit event (outside lock)
        await self._event_bus.emit_new(
            EventType.ORDER_FILLED,
            {
                "order_id": str(order_id),
                "symbol": entry.symbol,
                "execution_id": execution_id,
                "filled_quantity": str(filled_quantity),
                "filled_price": str(filled_price),
                "total_filled": str(entry.filled_quantity),
                "remaining": str(entry.remaining_quantity),
                "is_complete": entry.is_complete,
            },
            priority=EventPriority.HIGH,
            source="order_manager",
        )

        if entry.is_complete:
            logger.info(
                "Order fully filled",
                order_id=str(order_id),
                fills=len(entry.fills),
                avg_price=str(entry.avg_fill_price),
            )

        return True

    async def cancel_order(
        self,
        order_id: UUID,
        reason: str = "",
    ) -> bool:
        """Cancel order."""
        async with self._get_order_lock(order_id):
            entry = self._orders.get(order_id)
            if not entry:
                raise ValueError(f"Order {order_id} not found")

            if entry.is_complete:
                logger.warning(
                    "Cannot cancel completed order",
                    order_id=str(order_id),
                    state=entry.state.name,
                )
                return False

            entry.transition(OrderState.CANCELLED, {"reason": reason})

        # Emit event
        await self._event_bus.emit_new(
            EventType.ORDER_CANCELLED,
            {
                "order_id": str(order_id),
                "reason": reason,
            },
            priority=EventPriority.HIGH,
            source="order_manager",
        )

        logger.info("Order cancelled", order_id=str(order_id), reason=reason)
        return True

    async def get_order(self, order_id: UUID) -> HardenedOrderEntry | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def get_order_by_broker_id(
        self,
        broker_order_id: str,
    ) -> HardenedOrderEntry | None:
        """Get order by broker order ID."""
        order_id = self._broker_id_map.get(broker_order_id)
        if order_id:
            return self._orders.get(order_id)
        return None

    async def get_active_orders(self) -> list[HardenedOrderEntry]:
        """Get all non-terminal orders."""
        return [entry for entry in self._orders.values() if not entry.is_complete]

    async def get_orphaned_orders(self) -> list[HardenedOrderEntry]:
        """Get orders that may be orphaned."""
        return [
            entry for entry in self._orders.values() if entry.is_orphaned(self._orphan_threshold)
        ]

    async def reconcile_with_broker(
        self,
        broker_query_fn: Callable[[str], Any],
    ) -> ReconciliationReport:
        """Reconcile OMS state with broker.

        Args:
            broker_query_fn: Async function that takes broker_order_id and returns status
        """
        discrepancies = []
        orphaned = []

        async with self._global_lock:
            for entry in self._orders.values():
                if (
                    entry.state == OrderState.SUBMITTED
                    and entry.broker_order_id
                    and entry.is_orphaned(self._orphan_threshold)
                ):
                    orphaned.append(entry.order_id)

                    # Query broker
                    try:
                        broker_status = await broker_query_fn(entry.broker_order_id)

                        if broker_status != entry.state.name:
                            discrepancies.append(
                                {
                                    "order_id": str(entry.order_id),
                                    "broker_order_id": entry.broker_order_id,
                                    "oms_state": entry.state.name,
                                    "broker_state": broker_status,
                                    "orphaned_for_seconds": (
                                        whenever.now().py_datetime() - entry.submitted_at
                                    ).total_seconds()
                                    if entry.submitted_at
                                    else None,
                                }
                            )
                            entry.reconcile_discrepancies.append(
                                f"State mismatch: OMS={entry.state.name}, Broker={broker_status}"
                            )
                    except Exception as e:
                        logger.error(
                            "Broker query failed during reconciliation",
                            broker_order_id=entry.broker_order_id,
                            error=str(e),
                        )

        report = ReconciliationReport(
            discrepancies=discrepancies,
            orphaned_orders=orphaned,
        )

        if not report.is_clean:
            logger.warning(
                "Reconciliation found issues",
                discrepancies=len(discrepancies),
                orphaned=len(orphaned),
            )

        self._last_reconcile = whenever.now().py_datetime()
        self._orphaned_detected += len(orphaned)

        return report

    async def _reconcile_loop(self) -> None:
        """Background reconciliation loop."""
        while True:
            try:
                await asyncio.sleep(self._reconcile_interval)

                # This would need broker interface passed in
                # For now, just check for orphans
                orphaned = await self.get_orphaned_orders()
                if orphaned:
                    logger.warning(
                        "Detected orphaned orders",
                        count=len(orphaned),
                        orders=[str(o.order_id) for o in orphaned],
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Reconciliation loop error", error=str(e))

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        active = len([o for o in self._orders.values() if not o.is_complete])
        orphaned = len([o for o in self._orders.values() if o.is_orphaned()])

        return {
            "total_orders": self._total_orders,
            "active_orders": active,
            "total_fills": self._total_fills,
            "duplicate_fills_rejected": self._duplicate_fills_rejected,
            "validation_failures": self._validation_failures,
            "orphaned_detected": self._orphaned_detected,
            "orphaned_current": orphaned,
            "capacity_used_pct": (active / self._max_active_orders) * 100,
            "last_reconcile": self._last_reconcile.isoformat() if self._last_reconcile else None,
        }


# Backward compatibility
OrderManager = HardenedOrderManager
OrderEntry = HardenedOrderEntry
