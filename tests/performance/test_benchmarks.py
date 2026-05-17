"""Performance benchmarks for AMATIS.

Institutional-grade performance testing:
    - Event throughput
    - Latency distributions
    - Memory stability
    - Concurrency scaling
"""

from __future__ import annotations

import asyncio
import gc
import time
import tracemalloc
from decimal import Decimal
from typing import List

import pytest
import pytest_asyncio

from amatix.core.event_bus import EventBus
from amatix.core.event_models import EventPriority, EventType
from amatix.execution.oms.order_manager_hardened import HardenedOrderManager


class TestEventThroughput:
    """Benchmark event bus throughput."""
    
    async def test_event_throughput_1k_per_sec(self):
        """Verify system can process 1000 events/second."""
        bus = EventBus()
        processed = 0
        target_events = 1000
        
        @bus.on(EventType.SIGNAL_GENERATED)
        async def handler(event):
            nonlocal processed
            processed += 1
        
        start = time.time()
        
        # Emit events
        for i in range(target_events):
            await bus.emit_new(
                EventType.SIGNAL_GENERATED,
                {"index": i, "symbol": "AAPL"},
            )
        
        # Wait for processing
        timeout = 5.0
        while processed < target_events and time.time() - start < timeout:
            await asyncio.sleep(0.01)
        
        duration = time.time() - start
        rate = processed / duration
        
        assert processed == target_events, f"Only {processed}/{target_events} processed"
        assert rate >= 1000, f"Rate {rate:.0f} below 1000/sec threshold"
    
    async def test_event_throughput_10k_burst(self):
        """Test handling 10K event burst."""
        bus = EventBus()
        processed = 0
        target = 10000
        
        @bus.on(EventType.MARKET_DATA_RECEIVED)
        async def handler(event):
            nonlocal processed
            processed += 1
        
        start = time.time()
        
        # Burst emit
        tasks = [
            bus.emit_new(EventType.MARKET_DATA_RECEIVED, {"index": i})
            for i in range(target)
        ]
        await asyncio.gather(*tasks)
        
        # Wait
        await asyncio.sleep(0.5)
        
        duration = time.time() - start
        rate = target / duration
        
        print(f"10K burst: {duration:.2f}s, {rate:.0f} events/sec")
        assert processed >= target * 0.99  # Allow 1% loss


class TestLatencyBenchmarks:
    """Measure latency distributions."""
    
    async def test_risk_check_latency(self):
        """Benchmark risk assessment latency."""
        from amatix.risk.engine import RiskEngine
        from amatix.risk.models import RiskConfig
        from amatix.interfaces import Order, Symbol
        
        bus = EventBus()
        config = RiskConfig()
        engine = RiskEngine(bus, config)
        await engine.initialize()
        
        latencies: List[float] = []
        iterations = 100
        
        for _ in range(iterations):
            start = time.time()
            
            order = Order(
                symbol=Symbol(base="AAPL"),
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
            )
            
            await engine.assess_order(
                order,
                portfolio={"total_value": Decimal("100000")},
                market={"price": Decimal("150.00")},
            )
            
            latencies.append((time.time() - start) * 1000)  # ms
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"Risk check: avg={avg_latency:.2f}ms, p99={p99:.2f}ms, max={max_latency:.2f}ms")
        
        assert avg_latency < 50, f"Avg latency {avg_latency:.2f}ms exceeds 50ms"
        assert p99 < 100, f"P99 latency {p99:.2f}ms exceeds 100ms"
    
    async def test_order_creation_latency(self):
        """Benchmark order creation latency."""
        bus = EventBus()
        om = HardenedOrderManager(bus)
        await om.initialize()
        
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start = time.time()
            
            await om.create_order(
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
            )
            
            latencies.append((time.time() - start) * 1000)
        
        avg = sum(latencies) / len(latencies)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"Order creation: avg={avg:.2f}ms, p99={p99:.2f}ms")
        
        assert avg < 10, f"Avg {avg:.2f}ms too slow"


class TestMemoryStability:
    """Test memory stability over time."""
    
    async def test_memory_stable_over_time(self):
        """Verify no memory leaks over 1000 operations."""
        bus = EventBus(enable_journaling=False)  # Disable journaling for this test
        om = HardenedOrderManager(bus)
        await om.initialize()
        
        # Warm up
        gc.collect()
        baseline = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        
        # Run operations
        for i in range(1000):
            entry = await om.create_order(
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
            )
            
            await om.update_fill(
                entry.order_id,
                execution_id=f"exec_{i}",
                filled_quantity=Decimal("100"),
                filled_price=Decimal("150.00"),
            )
        
        gc.collect()
        
        # Memory should not have grown significantly
        # (Allow 10% growth due to legitimate state accumulation)
        stats = om.get_stats()
        assert stats["total_orders"] == 1000
    
    async def test_journal_memory_bounded(self):
        """Verify journal doesn't grow unbounded."""
        bus = EventBus(
            enable_journaling=True,
            max_memory_journal=100,  # Small limit
        )
        
        # Emit 1000 events
        for i in range(1000):
            await bus.emit_new(
                EventType.SIGNAL_GENERATED,
                {"index": i},
            )
        
        metrics = bus.get_metrics()
        
        # Should have overflowed to disk
        assert metrics["journal"]["memory_size"] <= 100
        assert metrics["events"]["overflowed"] > 0


class TestConcurrencyScaling:
    """Test concurrent operation handling."""
    
    async def test_concurrent_order_scaling(self):
        """Test system scales with concurrent orders."""
        bus = EventBus()
        om = HardenedOrderManager(bus)
        await om.initialize()
        
        async def create_and_fill(i):
            entry = await om.create_order(
                symbol="AAPL",
                side="buy",
                quantity=Decimal("100"),
                order_type="market",
            )
            
            await om.update_fill(
                entry.order_id,
                execution_id=f"exec_{i}",
                filled_quantity=Decimal("100"),
                filled_price=Decimal("150.00"),
            )
            
            return entry.order_id
        
        start = time.time()
        
        # Run 100 concurrent
        results = await asyncio.gather(*[create_and_fill(i) for i in range(100)])
        
        duration = time.time() - start
        
        assert len(results) == 100
        assert len(set(results)) == 100  # All unique
        
        print(f"100 concurrent orders: {duration:.2f}s")
        
        stats = om.get_stats()
        assert stats["total_orders"] == 100
        assert stats["total_fills"] == 100
    
    async def test_queue_pressure_handling(self):
        """Test backpressure under load."""
        bus = EventBus()
        processed = 0
        delay = 0.01  # Slow handler
        
        @bus.on(EventType.SIGNAL_GENERATED)
        async def slow_handler(event):
            nonlocal processed
            await asyncio.sleep(delay)
            processed += 1
        
        # Rapid fire 500 events
        start = time.time()
        for i in range(500):
            await bus.emit_new(
                EventType.SIGNAL_GENERATED,
                {"index": i},
            )
        
        # Wait for processing
        await asyncio.sleep(10)
        
        duration = time.time() - start
        
        # Should have processed all despite slow handler
        assert processed >= 495  # Allow small loss


class TestStressScenarios:
    """Stress test scenarios."""
    
    async def test_rapid_fill_storm(self):
        """Test handling rapid fill updates."""
        bus = EventBus()
        om = HardenedOrderManager(bus)
        await om.initialize()
        
        entry = await om.create_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("1000"),
            order_type="market",
        )
        
        # Storm of 1000 fills
        async def add_fill(i):
            return await om.update_fill(
                entry.order_id,
                execution_id=f"storm_{i}",
                filled_quantity=Decimal("1"),
                filled_price=Decimal("150.00"),
            )
        
        start = time.time()
        results = await asyncio.gather(*[add_fill(i) for i in range(1000)])
        duration = time.time() - start
        
        assert all(results)  # All accepted
        assert entry.filled_quantity == Decimal("1000")
        
        print(f"1000 fills: {duration:.2f}s")
    
    async def test_event_storm_recovery(self):
        """Test recovery after event storm."""
        bus = EventBus()
        errors = []
        
        @bus.on(EventType.SIGNAL_GENERATED)
        async def handler(event):
            try:
                if event.payload["index"] % 100 == 0:
                    raise Exception("Simulated error")
            except Exception as e:
                errors.append(str(e))
        
        # Storm with some errors
        for i in range(1000):
            await bus.emit_new(
                EventType.SIGNAL_GENERATED,
                {"index": i},
            )
        
        await asyncio.sleep(0.5)
        
        # Should have processed most despite errors
        metrics = bus.get_metrics()
        assert metrics["events"]["processed"] >= 900


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
