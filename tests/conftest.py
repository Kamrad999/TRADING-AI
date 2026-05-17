"""Pytest configuration and shared fixtures.

This module provides:
    - Test configuration
    - Shared fixtures for common test scenarios
    - Mock factories
    - Event bus fixtures
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio

from amatix.core.config import Settings
from amatix.core.event_bus import EventBus
from amatix.interfaces import (
    OHLCV,
    Signal,
    SignalDirection,
    Symbol,
    create_symbol,
)

# =============================================================================
# Event Loop Policy
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Core Fixtures
# =============================================================================


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh event bus for testing."""
    return EventBus(enable_journaling=False)


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with safe defaults."""
    return Settings(
        environment="testing",
        risk__paper_mode=True,  # type: ignore
        risk__kill_switch_enabled=True,  # type: ignore
        enable_auto_trading=False,
    )


# =============================================================================
# Domain Object Fixtures
# =============================================================================


@pytest.fixture
def sample_symbol() -> Symbol:
    """Sample symbol for testing."""
    return create_symbol("AAPL", "NASDAQ")


@pytest.fixture
def sample_ohlcv(sample_symbol: Symbol) -> OHLCV:
    """Sample OHLCV bar for testing."""
    return OHLCV(
        symbol=sample_symbol,
        timestamp=datetime.utcnow(),
        open=Decimal("150.00"),
        high=Decimal("155.00"),
        low=Decimal("149.00"),
        close=Decimal("153.00"),
        volume=Decimal("1000000"),
    )


@pytest.fixture
def sample_signal(sample_symbol: Symbol) -> Signal:
    """Sample signal for testing."""
    return Signal(
        symbol=sample_symbol,
        direction=SignalDirection.LONG,
        confidence=0.85,
        strength=0.75,
        source="test_strategy",
        timestamp=datetime.utcnow(),
        entry_price=Decimal("153.00"),
        stop_loss=Decimal("148.00"),
        take_profit=Decimal("165.00"),
    )


# =============================================================================
# Async Helpers
# =============================================================================


@pytest_asyncio.fixture
async def async_event_bus() -> AsyncGenerator[EventBus, None]:
    """Async fixture for event bus testing."""
    bus = EventBus(enable_journaling=False)
    yield bus
    # Cleanup
    bus.clear_journal()


# =============================================================================
# Mock Factories
# =============================================================================


def create_mock_signal(
    symbol: str = "AAPL",
    direction: SignalDirection = SignalDirection.LONG,
    confidence: float = 0.8,
) -> Signal:
    """Factory for creating mock signals."""
    return Signal(
        symbol=create_symbol(symbol),
        direction=direction,
        confidence=confidence,
        strength=confidence * 0.9,
        source="mock",
        timestamp=datetime.utcnow(),
    )


# =============================================================================
# Configuration
# =============================================================================


def pytest_configure(config: Any) -> None:
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
