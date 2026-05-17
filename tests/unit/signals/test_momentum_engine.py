"""Tests for momentum signal engine.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from amatix.core.event_bus import EventBus
from amatix.data.market.models import OHLCV, Symbol, DataSource
from amatix.signals.engines.momentum_engine import (
    IndicatorConfig,
    MomentumEngine,
)
from amatix.signals.models import SignalDirection


class TestMomentumEngine:
    """Momentum engine tests."""
    
    @pytest.fixture
    def engine(self):
        """Create momentum engine fixture."""
        bus = EventBus(enable_journaling=False)
        engine = MomentumEngine(bus)
        return engine
    
    @pytest.fixture
    def sample_bars(self):
        """Create sample OHLCV bars."""
        symbol = Symbol("AAPL", "NASDAQ")
        base_time = datetime.utcnow()
        
        bars = []
        # Create uptrend then crossover
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                  111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                  122, 123, 124, 125, 126, 127, 128, 129, 130]
        
        for i, price in enumerate(prices):
            bars.append(OHLCV(
                symbol=symbol,
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=Decimal("1000000"),
                source=DataSource.ALPACA,
            ))
        
        return bars
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization."""
        await engine.initialize({})
        assert engine._initialized is True
        assert engine.name == "momentum"
    
    @pytest.mark.asyncio
    async def test_insufficient_bars(self, engine, sample_bars):
        """Test with insufficient bars."""
        await engine.initialize({})
        
        # Only 5 bars (need at least 31 for EMA26 + RSI14)
        context = {
            "bars": sample_bars[:5],
            "symbol": Symbol("AAPL", "NASDAQ"),
        }
        
        signals = await engine.generate(context)
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_rsi_oversold(self, engine):
        """Test RSI oversold signal."""
        await engine.initialize({})
        
        symbol = Symbol("AAPL", "NASDAQ")
        base_time = datetime.utcnow()
        
        # Create sharp decline (RSI should drop)
        prices = [100 - i * 2 for i in range(50)]  # Falling prices
        
        bars = []
        for i, price in enumerate(prices):
            bars.append(OHLCV(
                symbol=symbol,
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=Decimal("1000000"),
                source=DataSource.ALPACA,
            ))
        
        context = {
            "bars": bars,
            "symbol": symbol,
        }
        
        signals = await engine.generate(context)
        
        # Should generate oversold signal
        oversold_signals = [s for s in signals if s.direction == SignalDirection.LONG]
        assert len(oversold_signals) > 0
    
    @pytest.mark.asyncio
    async def test_volume_spike(self, engine):
        """Test volume spike detection."""
        await engine.initialize({})
        
        symbol = Symbol("AAPL", "NASDAQ")
        base_time = datetime.utcnow()
        
        bars = []
        # Normal volume for 20 bars
        for i in range(20):
            bars.append(OHLCV(
                symbol=symbol,
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=Decimal("1000000"),
                source=DataSource.ALPACA,
            ))
        
        # Volume spike with price increase
        bars.append(OHLCV(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=20),
            open=Decimal("100"),
            high=Decimal("103"),
            low=Decimal("100"),
            close=Decimal("102"),  # 2% up
            volume=Decimal("5000000"),  # 5x volume
            source=DataSource.ALPACA,
        ))
        
        context = {
            "bars": bars,
            "symbol": symbol,
        }
        
        signals = await engine.generate(context)
        
        # Should detect volume spike
        volume_signals = [s for s in signals 
                         if any(f.name == "volume_spike" for f in s.features)]
        assert len(volume_signals) > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, engine):
        """Test health check."""
        await engine.initialize({})
        
        health = await engine.health_check()
        assert health["status"] == "healthy"
        assert health["name"] == "momentum"
