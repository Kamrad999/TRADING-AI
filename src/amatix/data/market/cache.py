"""Async-safe caching layer for market data.

Provides:
    - TTL-based caching
    - Async-safe operations
    - Cache warming
    - Statistics
    - Multiple backends (memory, Redis-ready)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar, Callable

from amatix.core.observability import get_logger, get_metrics

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry with metadata."""
    value: T
    timestamp: float
    ttl_seconds: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.timestamp > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Age of entry in seconds."""
        return time.time() - self.timestamp


class MarketDataCache(Generic[T]):
    """Async-safe TTL cache for market data.
    
    Designed for high-frequency market data caching with:
        - Automatic expiration
        - Size limits
        - Statistics
        - Async-safe operations
    
    Example:
        >>> cache = MarketDataCache[Quote](ttl_seconds=5.0, max_size=1000)
        >>> await cache.set("AAPL", quote)
        >>> cached = await cache.get("AAPL")
    """
    
    def __init__(
        self,
        ttl_seconds: float = 5.0,
        max_size: int = 10000,
        cleanup_interval: int = 100,
    ) -> None:
        """Initialize cache.
        
        Args:
            ttl_seconds: Default TTL for entries
            max_size: Maximum number of entries
            cleanup_interval: Cleanup every N operations
        """
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._lock = asyncio.Lock()
        
        self._ops_count = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            self._ops_count += 1
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                return None
            
            # Update access stats
            entry.access_count += 1
            entry.last_access = time.time()
            self._hits += 1
            
            return entry.value
    
    async def set(
        self,
        key: str,
        value: T,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Override default TTL
        """
        ttl = ttl_seconds or self._ttl_seconds
        
        async with self._lock:
            self._ops_count += 1
            
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                await self._evict_oldest()
            
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl,
            )
            
            # Periodic cleanup
            if self._ops_count % self._cleanup_interval == 0:
                await self._cleanup_expired()
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl_seconds: Optional[float] = None,
    ) -> T:
        """Get from cache or compute and store.
        
        Args:
            key: Cache key
            factory: Function to create value if not cached
            ttl_seconds: Override default TTL
        
        Returns:
            Cached or newly created value
        """
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        value = factory()
        await self.set(key, value, ttl_seconds)
        return value
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Returns:
            True if entry existed and was deleted
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cache cleared", entries_removed=count)
    
    async def keys(self) -> list[str]:
        """Get all cache keys."""
        async with self._lock:
            return list(self._cache.keys())
    
    async def size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            return len(self._cache)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "ops_count": self._ops_count,
            }
    
    async def _evict_oldest(self) -> None:
        """Evict oldest entry by access time."""
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_access,
        )
        del self._cache[oldest_key]
        self._evictions += 1
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(
                "Cleaned up expired cache entries",
                count=len(expired_keys),
            )
    
    async def warm_cache(
        self,
        keys: list[str],
        fetcher: Callable[[str], T],
    ) -> Dict[str, T]:
        """Pre-populate cache with values.
        
        Args:
            keys: Keys to warm
            fetcher: Function to fetch each value
        
        Returns:
            Dictionary of fetched values
        """
        results = {}
        
        async with self._lock:
            for key in keys:
                if key not in self._cache:
                    try:
                        value = fetcher(key)
                        self._cache[key] = CacheEntry(
                            value=value,
                            timestamp=time.time(),
                            ttl_seconds=self._ttl_seconds,
                        )
                        results[key] = value
                    except Exception as e:
                        logger.warning(
                            "Cache warm failed",
                            key=key,
                            error=str(e),
                        )
        
        logger.info(
            "Cache warming complete",
            requested=len(keys),
            fetched=len(results),
        )
        
        return results


class MultiLevelCache:
    """Two-level cache: L1 (memory) + L2 (Redis-ready).
    
    Future: Add Redis as L2 backend.
    For now: L1 only.
    """
    
    def __init__(
        self,
        l1_ttl: float = 5.0,
        l1_max_size: int = 10000,
    ) -> None:
        """Initialize multi-level cache."""
        self._l1 = MarketDataCache(ttl_seconds=l1_ttl, max_size=l1_max_size)
        self._l2: Optional[Any] = None  # Future: Redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from L1 or L2."""
        # Try L1 first
        value = await self._l1.get(key)
        if value is not None:
            get_metrics().counter("cache_l1_hit")
            return value
        
        # Future: Try L2 (Redis)
        # if self._l2:
        #     value = await self._l2.get(key)
        #     if value:
        #         # Promote to L1
        #         await self._l1.set(key, value)
        #         return value
        
        get_metrics().counter("cache_miss")
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        l1_ttl: Optional[float] = None,
    ) -> None:
        """Set in L1 (and L2 if available)."""
        await self._l1.set(key, value, l1_ttl)
        
        # Future: Also set in L2
        # if self._l2:
        #     await self._l2.set(key, value)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        l1_stats = await self._l1.get_stats()
        
        return {
            "l1": l1_stats,
            "l2": {"enabled": self._l2 is not None},
        }
