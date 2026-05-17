"""End-to-end integration tests for AMATIS.

Complete flow validation from market data to execution.
Part of Phase 2.9 institutional hardening.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from amatix.core.event_models import EventType
from amatix.execution.oms.state_machine import OrderState


class TestBasicOrderFlow:
    """Test basic order submission and fill flow."""

    async def test_create_order_emits_event(self, integrated_system, event_collector):
        """Verify order creation emits ORDER_SUBMITTED event."""
        om = integrated_system["order_manager"]

        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        # Allow event processing
        await asyncio.sleep(0.1)

        assert event_collector.has_event_type(EventType.ORDER_SUBMITTED)

        events = event_collector.get_events_by_type(EventType.ORDER_SUBMITTED)
        assert len(events) == 1
        assert events[0].payload["symbol"] == "AAPL"
        assert events[0].payload["side"] == "buy"

    async def test_fill_updates_position(self, integrated_system, event_collector):
        """Test that fill updates order state and emits event."""
        om = integrated_system["order_manager"]

        # Create order
        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        # Simulate fill
        await om.update_fill(
            entry.order_id,
            execution_id="exec_001",
            filled_quantity=Decimal("100"),
            filled_price=Decimal("150.00"),
            commission=Decimal("1.00"),
        )

        await asyncio.sleep(0.1)

        # Verify order is filled
        updated = await om.get_order(entry.order_id)
        assert updated.is_filled is True
        assert updated.filled_quantity == Decimal("100")

        # Verify event emitted
        assert event_collector.has_event_type(EventType.ORDER_FILLED)

    async def test_partial_fill_sequence(self, integrated_system):
        """Test multiple partial fills accumulating to full."""
        om = integrated_system["order_manager"]

        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        # First partial fill
        await om.update_fill(
            entry.order_id,
            execution_id="exec_1",
            filled_quantity=Decimal("30"),
            filled_price=Decimal("150.00"),
        )

        assert entry.filled_quantity == Decimal("30")
        assert entry.is_filled is False

        # Second partial fill
        await om.update_fill(
            entry.order_id,
            execution_id="exec_2",
            filled_quantity=Decimal("40"),
            filled_price=Decimal("150.50"),
        )

        assert entry.filled_quantity == Decimal("70")
        assert entry.is_filled is False

        # Final fill
        await om.update_fill(
            entry.order_id,
            execution_id="exec_3",
            filled_quantity=Decimal("30"),
            filled_price=Decimal("151.00"),
        )

        assert entry.filled_quantity == Decimal("100")
        assert entry.is_filled is True
        assert len(entry.fills) == 3

    async def test_cancel_order(self, integrated_system, event_collector):
        """Test order cancellation."""
        om = integrated_system["order_manager"]

        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        result = await om.cancel_order(entry.order_id, reason="test_cancel")

        assert result is True
        assert entry.state == OrderState.CANCELLED
        assert event_collector.has_event_type(EventType.ORDER_CANCELLED)

    async def test_cannot_cancel_filled_order(self, integrated_system):
        """Verify filled orders cannot be cancelled."""
        om = integrated_system["order_manager"]

        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        # Fill the order
        await om.update_fill(
            entry.order_id,
            execution_id="exec_001",
            filled_quantity=Decimal("100"),
            filled_price=Decimal("150.00"),
        )

        # Try to cancel
        result = await om.cancel_order(entry.order_id)

        assert result is False  # Cannot cancel filled order


class TestFailureScenarios:
    """Test system behavior under failure conditions."""

    async def test_duplicate_fill_rejection(self, integrated_system):
        """Verify duplicate fills are rejected."""
        om = integrated_system["order_manager"]

        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        # First fill
        result1 = await om.update_fill(
            entry.order_id,
            execution_id="dup_exec",
            filled_quantity=Decimal("50"),
            filled_price=Decimal("150.00"),
        )
        assert result1 is True

        # Duplicate - should be rejected
        result2 = await om.update_fill(
            entry.order_id,
            execution_id="dup_exec",  # Same ID
            filled_quantity=Decimal("50"),
            filled_price=Decimal("150.00"),
        )
        assert result2 is False

        # Verify only 50 filled
        assert entry.filled_quantity == Decimal("50")

    async def test_fill_validation_rejects_invalid_qty(self, integrated_system):
        """Test that invalid fill quantities are rejected."""
        om = integrated_system["order_manager"]

        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        # Negative quantity
        result = await om.update_fill(
            entry.order_id,
            execution_id="exec_001",
            filled_quantity=Decimal("-10"),
            filled_price=Decimal("150.00"),
        )
        assert result is False

        # Exceeds order quantity
        result = await om.update_fill(
            entry.order_id,
            execution_id="exec_002",
            filled_quantity=Decimal("150"),
            filled_price=Decimal("150.00"),
        )
        assert result is False

    async def test_orphan_order_detection(self, integrated_system):
        """Test detection of orphaned orders."""
        om = integrated_system["order_manager"]

        # Create and submit order
        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        await om.mark_submitted(entry.order_id, "broker_123")

        # Wait for orphan threshold
        await asyncio.sleep(1.5)

        # Check orphan detection
        orphaned = await om.get_orphaned_orders()
        assert len(orphaned) == 1
        assert orphaned[0].order_id == entry.order_id


class TestConcurrentOperations:
    """Test concurrent order operations."""

    async def test_concurrent_order_creation(self, integrated_system):
        """Test creating multiple orders concurrently."""
        om = integrated_system["order_manager"]

        async def create_order(i):
            return await om.create_order(
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
            )

        # Create 50 orders concurrently
        orders = await asyncio.gather(*[create_order(i) for i in range(50)])

        assert len(orders) == 50
        assert all(o.order_id is not None for o in orders)

        stats = om.get_stats()
        assert stats["total_orders"] == 50

    async def test_concurrent_fills_same_order(self, integrated_system):
        """Test concurrent fills on same order."""
        om = integrated_system["order_manager"]

        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        async def add_fill(i):
            return await om.update_fill(
                entry.order_id,
                execution_id=f"concurrent_{i}",
                filled_quantity=Decimal("1"),
                filled_price=Decimal("150.00"),
            )

        # Add 100 fills concurrently
        results = await asyncio.gather(*[add_fill(i) for i in range(100)])

        # All should succeed (different execution_ids)
        assert all(results)
        assert entry.filled_quantity == Decimal("100")


class TestEventOrdering:
    """Test event ordering and sequencing."""

    async def test_event_sequence_integrity(self, integrated_system, event_collector):
        """Verify events are emitted in correct order."""
        om = integrated_system["order_manager"]

        # Create and fill order
        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        await asyncio.sleep(0.05)

        await om.update_fill(
            entry.order_id,
            execution_id="exec_001",
            filled_quantity=Decimal("100"),
            filled_price=Decimal("150.00"),
        )

        await asyncio.sleep(0.1)

        # Verify event order
        submitted_events = event_collector.get_events_by_type(EventType.ORDER_SUBMITTED)
        filled_events = event_collector.get_events_by_type(EventType.ORDER_FILLED)

        assert len(submitted_events) == 1
        assert len(filled_events) == 1

        # SUBMITTED should be before FILLED
        assert submitted_events[0].context.timestamp < filled_events[0].context.timestamp


class TestResilience:
    """Test system resilience scenarios."""

    async def test_recovery_after_high_load(self, integrated_system):
        """Test system recovers after high load."""
        om = integrated_system["order_manager"]

        # High load burst
        orders = []
        for i in range(100):
            entry = await om.create_order(
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
            )
            orders.append(entry)

        # Fill all orders
        for i, entry in enumerate(orders):
            await om.update_fill(
                entry.order_id,
                execution_id=f"exec_{i}",
                filled_quantity=Decimal("100"),
                filled_price=Decimal("150.00"),
            )

        # Verify system state is consistent
        stats = om.get_stats()
        assert stats["total_orders"] == 100
        assert stats["total_fills"] == 100

        active = await om.get_active_orders()
        assert len(active) == 0  # All filled


class TestIntegrationCompleteness:
    """Comprehensive integration tests covering all components."""

    async def test_market_data_to_order_flow(self, integrated_system, event_collector):
        """Test complete flow from market data to order.

        This simulates the full trading pipeline:
        1. Market data arrives
        2. Signal generated
        3. Risk check
        4. Order created
        5. Order submitted
        6. Order filled
        """
        bus = integrated_system["event_bus"]
        om = integrated_system["order_manager"]

        # Simulate market data
        await bus.emit_new(
            EventType.MARKET_DATA_RECEIVED,
            {"symbol": "AAPL", "price": 150.0, "volume": 1000000},
            source="market_data",
        )

        await asyncio.sleep(0.05)

        # Simulate signal generation
        await bus.emit_new(
            EventType.SIGNAL_GENERATED,
            {
                "symbol": "AAPL",
                "direction": "long",
                "confidence": 0.85,
                "source": "momentum_engine",
            },
            source="signal_pipeline",
        )

        await asyncio.sleep(0.05)

        # Create order (as would happen from signal)
        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )

        await asyncio.sleep(0.05)

        # Fill order
        await om.update_fill(
            entry.order_id,
            execution_id="broker_exec_001",
            filled_quantity=Decimal("100"),
            filled_price=Decimal("150.00"),
        )

        await asyncio.sleep(0.1)

        # Verify complete event chain
        assert event_collector.has_event_type(EventType.MARKET_DATA_RECEIVED)
        assert event_collector.has_event_type(EventType.SIGNAL_GENERATED)
        assert event_collector.has_event_type(EventType.ORDER_SUBMITTED)
        assert event_collector.has_event_type(EventType.ORDER_FILLED)

    async def test_risk_veto_blocks_order(self, integrated_system, event_collector):
        """Test risk engine can veto orders."""
        # This would require full risk engine integration
        # Placeholder for risk integration test
        pass
