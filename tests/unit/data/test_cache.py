"""Tests for market data cache.
"""

import asyncio
from datetime import datetime
from decimal import Decimal

import pytest

from amatix.data.market.cache import MarketDataCache
from amatix.data.market.models import Quote, Symbol, DataSource


class TestMarketDataCache:
    """Cache tests."""
    
    @pytest.mark.asyncio
    async def test_cache_get_set(self):
        """Test basic get/set operations."""
        cache = MarketDataCache(ttl_seconds=60.0)
        
        quote = Quote(
            symbol=Symbol("AAPL", "NASDAQ"),
            bid=Decimal("150.00"),
            ask=Decimal("150.10"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            timestamp=datetime.utcnow(),
            source=DataSource.ALPACA,
        )
        
        # Set
        await cache.set("AAPL", quote)
        
        # Get
        cached = await cache.get("AAPL")
        assert cached is not None
        assert cached.mid == Decimal("150.05")
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test TTL expiration."""
        cache = MarketDataCache(ttl_seconds=0.01)  # 10ms TTL
        
        quote = Quote(
            symbol=Symbol("AAPL", "NASDAQ"),
            bid=Decimal("150.00"),
            ask=Decimal("150.10"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            timestamp=datetime.utcnow(),
            source=DataSource.ALPACA,
        )
        
        await cache.set("AAPL", quote)
        
        # Wait for expiration
        await asyncio.sleep(0.02)
        
        # Should be expired
        cached = await cache.get("AAPL")
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss."""
        cache = MarketDataCache(ttl_seconds=60.0)
        
        cached = await cache.get("NONEXISTENT")
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        cache = MarketDataCache(ttl_seconds=60.0)
        
        quote = Quote(
            symbol=Symbol("AAPL", "NASDAQ"),
            bid=Decimal("150.00"),
            ask=Decimal("150.10"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            timestamp=datetime.utcnow(),
            source=DataSource.ALPACA,
        )
        
        await cache.set("AAPL", quote)
        await cache.get("AAPL")  # Hit
        await cache.get("MISS")  # Miss
        
        stats = await cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
