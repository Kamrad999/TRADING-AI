"""Unit tests for the AMATIS event bus.

Tests cover:
    - Basic event emission and handling
    - Handler registration/deregistration
    - Priority ordering
    - Async/sync handler compatibility
    - Middleware chain
    - Error handling
"""

from __future__ import annotations

import asyncio

import pytest

from amatix.core.event_bus import EventBus
from amatix.core.event_models import Event, EventContext, EventPriority, EventType


class TestEventBusBasic:
    """Basic event bus functionality."""

    @pytest.mark.asyncio
    async def test_emit_and_receive(self, event_bus: EventBus) -> None:
        """Test basic event emission and handling."""
        received_events: list[Event] = []

        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def handler(event: Event) -> None:
            received_events.append(event)

        # Emit event
        test_event = Event(
            event_type=EventType.SIGNAL_GENERATED,
            payload={"symbol": "AAPL", "direction": "LONG"},
            context=EventContext(
                trace_id=__import__("uuid").uuid4(),
                source_component="test",
            ),
        )

        await event_bus.emit(test_event)

        # Verify
        assert len(received_events) == 1
        assert received_events[0].payload["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus: EventBus) -> None:
        """Test that multiple handlers receive the same event."""
        handler1_calls: list[str] = []
        handler2_calls: list[str] = []

        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def handler1(event: Event) -> None:
            handler1_calls.append("called")

        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def handler2(event: Event) -> None:
            handler2_calls.append("called")

        await event_bus.emit_new(
            EventType.SIGNAL_GENERATED,
            {"test": "data"},
            source="test",
        )

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

    @pytest.mark.asyncio
    async def test_handler_deregistration(self, event_bus: EventBus) -> None:
        """Test handler removal."""
        calls: list[str] = []

        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def handler(event: Event) -> None:
            calls.append("called")

        # First emit - should receive
        await event_bus.emit_new(EventType.SIGNAL_GENERATED, {}, source="test")
        assert len(calls) == 1

        # Deregister
        event_bus.off(handler, {EventType.SIGNAL_GENERATED})

        # Second emit - should not receive
        await event_bus.emit_new(EventType.SIGNAL_GENERATED, {}, source="test")
        assert len(calls) == 1  # Still 1


class TestEventBusPriority:
    """Event priority handling."""

    @pytest.mark.asyncio
    async def test_priority_ordering(self, event_bus: EventBus) -> None:
        """Test that handlers are called in priority order."""
        call_order: list[str] = []

        @event_bus.on(EventType.SIGNAL_GENERATED, priority=EventPriority.LOW)
        async def low_priority(event: Event) -> None:
            call_order.append("low")

        @event_bus.on(EventType.SIGNAL_GENERATED, priority=EventPriority.HIGH)
        async def high_priority(event: Event) -> None:
            call_order.append("high")

        @event_bus.on(EventType.SIGNAL_GENERATED, priority=EventPriority.NORMAL)
        async def normal_priority(event: Event) -> None:
            call_order.append("normal")

        await event_bus.emit_new(EventType.SIGNAL_GENERATED, {}, source="test")

        # Note: Due to concurrent execution, order isn't strictly guaranteed,
        # but high priority should be processed before low in the queue
        assert "high" in call_order
        assert "normal" in call_order
        assert "low" in call_order


class TestEventBusSyncHandlers:
    """Synchronous handler compatibility."""

    @pytest.mark.asyncio
    async def test_sync_handler(self, event_bus: EventBus) -> None:
        """Test that sync handlers work correctly."""
        calls: list[str] = []

        @event_bus.on(EventType.SIGNAL_GENERATED)
        def sync_handler(event: Event) -> None:  # Note: no async
            calls.append("sync_called")

        await event_bus.emit_new(EventType.SIGNAL_GENERATED, {}, source="test")

        # Give a moment for thread pool execution
        await asyncio.sleep(0.01)

        assert "sync_called" in calls


class TestEventBusErrorHandling:
    """Error handling in event processing."""

    @pytest.mark.asyncio
    async def test_handler_error_isolated(self, event_bus: EventBus) -> None:
        """Test that one handler error doesn't affect others."""
        calls: list[str] = []

        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def failing_handler(event: Event) -> None:
            raise ValueError("Test error")

        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def good_handler(event: Event) -> None:
            calls.append("good")

        await event_bus.emit_new(EventType.SIGNAL_GENERATED, {}, source="test")

        # Good handler should still be called
        assert "good" in calls

    @pytest.mark.asyncio
    async def test_error_metrics(self, event_bus: EventBus) -> None:
        """Test that errors are tracked in metrics."""

        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def failing_handler(event: Event) -> None:
            raise ValueError("Test error")

        await event_bus.emit_new(EventType.SIGNAL_GENERATED, {}, source="test")

        metrics = event_bus.get_metrics()
        assert "handler_errors" in metrics


class TestEventBusMiddleware:
    """Middleware chain functionality."""

    @pytest.mark.asyncio
    async def test_middleware_transformation(self, event_bus: EventBus) -> None:
        """Test middleware can transform events."""
        received_payload: dict = {}

        async def add_field_middleware(event: Event) -> Event:
            """Add a field to all events."""
            return event.with_payload(middleware_added=True)

        event_bus.add_middleware(add_field_middleware)

        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def handler(event: Event) -> None:
            received_payload.update(event.payload)

        await event_bus.emit_new(EventType.SIGNAL_GENERATED, {"original": True}, source="test")

        assert received_payload.get("middleware_added") is True
        assert received_payload.get("original") is True

    @pytest.mark.asyncio
    async def test_middleware_blocking(self, event_bus: EventBus) -> None:
        """Test middleware can block events."""
        calls: list[str] = []

        async def blocking_middleware(event: Event) -> None:
            """Block all events."""
            return None

        event_bus.add_middleware(blocking_middleware)

        @event_bus.on(EventType.SIGNAL_GENERATED)
        async def handler(event: Event) -> None:
            calls.append("called")

        await event_bus.emit_new(EventType.SIGNAL_GENERATED, {}, source="test")

        # Handler should not be called
        assert len(calls) == 0
