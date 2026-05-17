"""Torture tests for Hardened OMS.

Validates edge cases, error conditions, and stress scenarios.
Part of PHASE 2.75 hardening.
"""

from __future__ import annotations

import asyncio
import random
from decimal import Decimal
from uuid import uuid4

import pytest
from amatix.core.event_bus import EventBus
from amatix.core.event_models import EventType
from amatix.execution.oms.order_manager_hardened import (
    DuplicateFillError,
    FillValidationError,
    HardenedOrderEntry,
    HardenedOrderManager,
)
from amatix.execution.oms.state_machine import OrderState


@pytest.fixture
async def event_bus():
    """Create event bus for testing."""
    bus = EventBus(enable_journaling=False)
    yield bus


@pytest.fixture
async def order_manager(event_bus):
    """Create order manager for testing."""
    om = HardenedOrderManager(
        event_bus=event_bus,
        max_active_orders=100,
        orphan_threshold_seconds=1.0,  # Short for testing
        enable_reconciliation=False,  # Disable background task
    )
    await om.initialize()
    yield om
    await om.shutdown()


class TestPartialFillScenarios:
    """Torture tests for partial fill handling."""
    
    async def test_partial_fill_sequence_1_5_10_remainder(self, order_manager):
        """Test 1%, 5%, 10%, remainder fill sequence."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        # Fill 1
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_1",
            filled_quantity=Decimal("1"),
            filled_price=Decimal("150.00"),
        )
        assert result is True
        assert entry.filled_quantity == Decimal("1")
        assert entry.remaining_quantity == Decimal("99")
        assert entry.fill_rate == 0.01
        
        # Fill 5
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_2",
            filled_quantity=Decimal("5"),
            filled_price=Decimal("150.10"),
        )
        assert result is True
        assert entry.filled_quantity == Decimal("6")
        assert entry.remaining_quantity == Decimal("94")
        
        # Fill 10
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_3",
            filled_quantity=Decimal("10"),
            filled_price=Decimal("150.05"),
        )
        assert result is True
        assert entry.filled_quantity == Decimal("16")
        assert entry.remaining_quantity == Decimal("84")
        
        # Remainder (84)
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_4",
            filled_quantity=Decimal("84"),
            filled_price=Decimal("149.95"),
        )
        assert result is True
        assert entry.filled_quantity == Decimal("100")
        assert entry.remaining_quantity == Decimal("0")
        assert entry.is_filled is True
        assert entry.is_complete is True
        assert entry.state == OrderState.FILLED
    
    async def test_many_small_partial_fills(self, order_manager):
        """Test 100 partial fills of 1 share each."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        for i in range(100):
            result = await order_manager.update_fill(
                entry.order_id,
                execution_id=f"exec_{i}",
                filled_quantity=Decimal("1"),
                filled_price=Decimal("150.00") + Decimal("0.01") * i,
            )
            assert result is True
        
        assert entry.filled_quantity == Decimal("100")
        assert entry.is_filled is True
        assert len(entry.fills) == 100
    
    async def test_partial_fill_with_price_volatility(self, order_manager):
        """Test fills with high price volatility."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        prices = [Decimal("150.00"), Decimal("160.00"), Decimal("140.00"), Decimal("155.00")]
        
        for i, price in enumerate(prices):
            await order_manager.update_fill(
                entry.order_id,
                execution_id=f"exec_{i}",
                filled_quantity=Decimal("25"),
                filled_price=price,
            )
        
        # Average should be weighted
        expected_avg = (
            Decimal("150.00") * 25 +
            Decimal("160.00") * 25 +
            Decimal("140.00") * 25 +
            Decimal("155.00") * 25
        ) / 100
        
        assert entry.avg_fill_price == expected_avg


class TestDuplicateFillRejection:
    """Torture tests for fill deduplication."""
    
    async def test_exact_duplicate_rejected(self, order_manager):
        """Test that exact duplicate execution_id is rejected."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        # First fill accepted
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_123",
            filled_quantity=Decimal("10"),
            filled_price=Decimal("150.00"),
        )
        assert result is True
        
        # Duplicate rejected
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_123",  # Same ID
            filled_quantity=Decimal("10"),
            filled_price=Decimal("150.00"),
        )
        assert result is False
        
        # Verify only one fill recorded
        assert entry.filled_quantity == Decimal("10")
        assert len(entry.fills) == 1
    
    async def test_duplicate_different_qty_rejected(self, order_manager):
        """Test duplicate with different quantity is still rejected."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_123",
            filled_quantity=Decimal("10"),
            filled_price=Decimal("150.00"),
        )
        
        # Same execution_id, different quantity - still rejected
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_123",
            filled_quantity=Decimal("20"),  # Different!
            filled_price=Decimal("150.00"),
        )
        assert result is False
    
    async def test_many_concurrent_duplicates(self, order_manager):
        """Test concurrent duplicate fill attempts."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        # Fire 10 concurrent fills with same execution_id
        async def try_fill():
            return await order_manager.update_fill(
                entry.order_id,
                execution_id="concurrent_exec",
                filled_quantity=Decimal("10"),
                filled_price=Decimal("150.00"),
            )
        
        results = await asyncio.gather(*[try_fill() for _ in range(10)])
        
        # Exactly one should succeed
        assert sum(results) == 1
        assert entry.filled_quantity == Decimal("10")


class TestFillValidation:
    """Torture tests for fill validation."""
    
    async def test_negative_quantity_rejected(self, order_manager):
        """Test that negative fill quantity is rejected."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_1",
            filled_quantity=Decimal("-10"),
            filled_price=Decimal("150.00"),
        )
        assert result is False
    
    async def test_zero_quantity_rejected(self, order_manager):
        """Test that zero fill quantity is rejected."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_1",
            filled_quantity=Decimal("0"),
            filled_price=Decimal("150.00"),
        )
        assert result is False
    
    async def test_negative_price_rejected(self, order_manager):
        """Test that negative fill price is rejected."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_1",
            filled_quantity=Decimal("10"),
            filled_price=Decimal("-150.00"),
        )
        assert result is False
    
    async def test_fill_exceeding_order_quantity_rejected(self, order_manager):
        """Test fill exceeding order quantity is rejected."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_1",
            filled_quantity=Decimal("110"),  # More than order!
            filled_price=Decimal("150.00"),
        )
        assert result is False
    
    async def test_cumulative_fill_exceeding_order_rejected(self, order_manager):
        """Test that cumulative fills exceeding order qty is rejected."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        # Fill 60
        await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_1",
            filled_quantity=Decimal("60"),
            filled_price=Decimal("150.00"),
        )
        
        # Try to fill another 60 (would exceed 100)
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_2",
            filled_quantity=Decimal("60"),
            filled_price=Decimal("150.00"),
        )
        assert result is False


class TestOutOfOrderDelivery:
    """Torture tests for out-of-order event delivery."""
    
    async def test_fills_arriving_out_of_chronological_order(self, order_manager):
        """Test fills arriving out of order."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        # Later fill arrives first
        await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_later",
            filled_quantity=Decimal("50"),
            filled_price=Decimal("150.00"),
        )
        
        # Earlier fill arrives second
        await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_earlier",
            filled_quantity=Decimal("50"),
            filled_price=Decimal("149.00"),
        )
        
        # Both should be accepted
        assert entry.filled_quantity == Decimal("100")
        assert len(entry.processed_execution_ids) == 2


class TestOrphanDetection:
    """Torture tests for orphan order detection."""
    
    async def test_orphan_detection_after_threshold(self, order_manager):
        """Test that orders become orphaned after threshold."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        # Mark as submitted
        await order_manager.mark_submitted(entry.order_id, "broker_123")
        
        # Initially not orphaned
        assert entry.is_orphaned(threshold_seconds=1.0) is False
        
        # Wait for threshold
        await asyncio.sleep(1.5)
        
        # Now orphaned
        assert entry.is_orphaned(threshold_seconds=1.0) is True
    
    async def test_orphan_detection_only_for_submitted(self, order_manager):
        """Test that only SUBMITTED orders can be orphaned."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        # CREATED state - not orphaned even after delay
        await asyncio.sleep(2.0)
        assert entry.is_orphaned(threshold_seconds=1.0) is False


class TestStressScenarios:
    """Stress tests for OMS."""
    
    async def test_rapid_order_creation(self, order_manager):
        """Test creating 1000 orders rapidly."""
        orders = []
        
        for i in range(1000):
            entry = await order_manager.create_order(
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
            )
            orders.append(entry)
        
        assert len(orders) == 1000
        stats = order_manager.get_stats()
        assert stats["total_orders"] == 1000
    
    async def test_concurrent_fill_storm(self, order_manager):
        """Test 100 concurrent fills on same order."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        async def add_fill(i):
            return await order_manager.update_fill(
                entry.order_id,
                execution_id=f"storm_{i}",
                filled_quantity=Decimal("1"),
                filled_price=Decimal("150.00"),
            )
        
        results = await asyncio.gather(*[add_fill(i) for i in range(100)])
        
        # All should succeed (different execution_ids)
        assert all(results)
        assert entry.filled_quantity == Decimal("100")
    
    async def test_capacity_limit(self, order_manager):
        """Test that capacity limit is enforced."""
        # Create orders up to limit
        for i in range(100):
            await order_manager.create_order(
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
            )
        
        # Next should fail
        with pytest.raises(RuntimeError, match="Maximum orders"):
            await order_manager.create_order(
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
            )


class TestEdgeCases:
    """Edge case tests."""
    
    async def test_order_with_zero_quantity(self, order_manager):
        """Test order creation with zero quantity."""
        # This should probably be rejected at creation time
        # For now, test behavior
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("0"),
            order_type="market",
        )
        
        # Fill rate calculation should not crash
        _ = entry.fill_rate
        assert entry.is_filled is True  # 0 >= 0
    
    async def test_fill_with_extreme_price_precision(self, order_manager):
        """Test fill with many decimal places."""
        entry = await order_manager.create_order(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1"),
            order_type="market",
        )
        
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="exec_1",
            filled_quantity=Decimal("1"),
            filled_price=Decimal("50000.123456789"),
        )
        assert result is True
    
    async def test_multiple_orders_same_symbol(self, order_manager):
        """Test multiple orders for same symbol."""
        entries = []
        for i in range(10):
            entry = await order_manager.create_order(
                symbol="AAPL",
                side="buy" if i % 2 == 0 else "sell",
                quantity=Decimal("100"),
                order_type="market",
            )
            entries.append(entry)
        
        # Fill each
        for i, entry in enumerate(entries):
            await order_manager.update_fill(
                entry.order_id,
                execution_id=f"exec_{i}",
                filled_quantity=Decimal("100"),
                filled_price=Decimal("150.00"),
            )
        
        # All should be filled
        assert all(e.is_filled for e in entries)


class TestBrokerScenarios:
    """Tests simulating real broker behaviors."""
    
    async def test_broker_duplicate_fill_notification(self, order_manager):
        """Simulate broker sending duplicate fill notification."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        # Broker sends fill
        await order_manager.update_fill(
            entry.order_id,
            execution_id="broker_exec_123",
            filled_quantity=Decimal("50"),
            filled_price=Decimal("150.00"),
        )
        
        # Broker sends same fill again (bug or retry)
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="broker_exec_123",
            filled_quantity=Decimal("50"),
            filled_price=Decimal("150.00"),
        )
        
        # Should be rejected
        assert result is False
        assert entry.filled_quantity == Decimal("50")  # Not 100
    
    async def test_broker_late_fill_after_cancel(self, order_manager):
        """Simulate fill arriving after cancel request."""
        entry = await order_manager.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        
        # Cancel
        await order_manager.cancel_order(entry.order_id)
        assert entry.state == OrderState.CANCELLED
        
        # Late fill arrives (from before cancel)
        result = await order_manager.update_fill(
            entry.order_id,
            execution_id="late_exec",
            filled_quantity=Decimal("10"),
            filled_price=Decimal("150.00"),
        )
        
        # Should still be accepted (fill happened before cancel)
        assert result is True
        assert entry.filled_quantity == Decimal("10")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
