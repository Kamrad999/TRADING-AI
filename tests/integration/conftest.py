"""Integration test fixtures for AMATIS.

Provides isolated test environments with:
    - In-memory event bus
    - Fake broker implementations
    - Deterministic random seeds
    - Test containers
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

import pytest
import pytest_asyncio

from amatix.core.config import Settings
from amatix.core.event_bus import EventBus
from amatix.core.event_models import Event, EventContext, EventPriority, EventType
from amatix.execution.oms.order_manager_hardened import HardenedOrderManager
from amatix.risk.engine import RiskEngine
from amatix.risk.models import RiskConfig


class FakeBroker:
    """Simulated broker for integration testing.
    
    Configurable behavior:
        - Latency
        - Failure rate
        - Fill patterns
        - Rejection reasons
    """
    
    def __init__(
        self,
        latency_ms: float = 0.0,
        failure_rate: float = 0.0,
        fill_delay_ms: float = 100.0,
        partial_fill_rate: float = 0.0,
    ) -> None:
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.fill_delay_ms = fill_delay_ms
        self.partial_fill_rate = partial_fill_rate
        
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.fills: List[Dict[str, Any]] = []
        self.order_counter = 0
    
    async def submit_order(self, order: Dict[str, Any]) -> str:
        """Submit order to fake broker."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        if random.random() < self.failure_rate:
            raise Exception("Broker order submission failed")
        
        self.order_counter += 1
        broker_order_id = f"FAKE_{self.order_counter}"
        
        self.orders[broker_order_id] = {
            **order,
            "broker_order_id": broker_order_id,
            "status": "submitted",
            "submitted_at": datetime.utcnow().isoformat(),
        }
        
        # Schedule fill
        asyncio.create_task(self._schedule_fill(broker_order_id))
        
        return broker_order_id
    
    async def _schedule_fill(self, broker_order_id: str) -> None:
        """Schedule delayed fill."""
        await asyncio.sleep(self.fill_delay_ms / 1000)
        
        order = self.orders.get(broker_order_id)
        if not order:
            return
        
        quantity = Decimal(order["quantity"])
        
        # Partial fill scenario
        if random.random() < self.partial_fill_rate:
            # Fill half now
            partial_qty = quantity / 2
            self.fills.append({
                "broker_order_id": broker_order_id,
                "execution_id": f"EXEC_{len(self.fills)}",
                "filled_quantity": str(partial_qty),
                "filled_price": order.get("price", "150.00"),
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            # Schedule remainder
            await asyncio.sleep(self.fill_delay_ms / 1000)
            self.fills.append({
                "broker_order_id": broker_order_id,
                "execution_id": f"EXEC_{len(self.fills)}",
                "filled_quantity": str(partial_qty),
                "filled_price": order.get("price", "150.00"),
                "timestamp": datetime.utcnow().isoformat(),
            })
        else:
            # Full fill
            self.fills.append({
                "broker_order_id": broker_order_id,
                "execution_id": f"EXEC_{len(self.fills)}",
                "filled_quantity": str(quantity),
                "filled_price": order.get("price", "150.00"),
                "timestamp": datetime.utcnow().isoformat(),
            })
    
    async def get_order_status(self, broker_order_id: str) -> str:
        """Get order status from broker."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        order = self.orders.get(broker_order_id)
        if not order:
            return "unknown"
        
        # Check if filled
        fills = [f for f in self.fills if f["broker_order_id"] == broker_order_id]
        if fills:
            total_filled = sum(Decimal(f["filled_quantity"]) for f in fills)
            if total_filled >= Decimal(order["quantity"]):
                return "filled"
            return "partially_filled"
        
        return order.get("status", "submitted")
    
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order at broker."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        order = self.orders.get(broker_order_id)
        if order:
            order["status"] = "cancelled"
            return True
        return False


@pytest.fixture
def deterministic_seed():
    """Set deterministic random seed for reproducible tests."""
    random.seed(42)
    return 42


@pytest_asyncio.fixture
async def event_bus() -> AsyncGenerator[EventBus, None]:
    """Create isolated event bus for testing."""
    bus = EventBus(enable_journaling=True)
    yield bus
    await bus.close()


@pytest_asyncio.fixture
async def fake_broker() -> AsyncGenerator[FakeBroker, None]:
    """Create fake broker for testing."""
    broker = FakeBroker(
        latency_ms=10,
        failure_rate=0.0,
        fill_delay_ms=50,
    )
    yield broker


@pytest_asyncio.fixture
async def order_manager(event_bus) -> AsyncGenerator[HardenedOrderManager, None]:
    """Create order manager for testing."""
    om = HardenedOrderManager(
        event_bus=event_bus,
        max_active_orders=1000,
        orphan_threshold_seconds=1.0,
        enable_reconciliation=False,
    )
    await om.initialize()
    yield om
    await om.shutdown()


@pytest_asyncio.fixture
async def risk_engine(event_bus) -> AsyncGenerator[RiskEngine, None]:
    """Create risk engine for testing."""
    config = RiskConfig(
        max_position_size=Decimal("100000"),
        max_position_pct=0.2,
        max_daily_drawdown=0.03,
        kill_switch_drawdown=0.15,
    )
    engine = RiskEngine(event_bus=event_bus, config=config)
    await engine.initialize()
    yield engine


@pytest_asyncio.fixture
async def integrated_system(event_bus, order_manager, fake_broker):
    """Create fully integrated test system.
    
    Wires together:
        - Event bus
        - Order manager
        - Fake broker
    """
    system = {
        "event_bus": event_bus,
        "order_manager": order_manager,
        "broker": fake_broker,
    }
    yield system


class EventCollector:
    """Collect and inspect events during testing."""
    
    def __init__(self, event_bus: EventBus) -> None:
        self.events: List[Event] = []
        self._handlers = []
        
        # Register handlers for all event types
        @event_bus.on(EventType.ORDER_SUBMITTED)
        async def on_order_submitted(event):
            self.events.append(event)
        
        @event_bus.on(EventType.ORDER_FILLED)
        async def on_order_filled(event):
            self.events.append(event)
        
        @event_bus.on(EventType.ORDER_CANCELLED)
        async def on_order_cancelled(event):
            self.events.append(event)
        
        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def on_signal(event):
            self.events.append(event)
        
        @event_bus.on(EventType.RISK_CHECK_PASSED)
        async def on_risk_passed(event):
            self.events.append(event)
        
        @event_bus.on(EventType.RISK_CHECK_FAILED)
        async def on_risk_failed(event):
            self.events.append(event)
        
        self._handlers = [
            on_order_submitted,
            on_order_filled,
            on_order_cancelled,
            on_signal,
            on_risk_passed,
            on_risk_failed,
        ]
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get events of specific type."""
        return [e for e in self.events if e.event_type == event_type]
    
    def has_event_type(self, event_type: EventType) -> bool:
        """Check if any event of type was received."""
        return any(e.event_type == event_type for e in self.events)
    
    def clear(self) -> None:
        """Clear collected events."""
        self.events.clear()


@pytest_asyncio.fixture
async def event_collector(event_bus) -> AsyncGenerator[EventCollector, None]:
    """Create event collector for testing."""
    collector = EventCollector(event_bus)
    yield collector
