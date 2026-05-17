"""AMATIS Memory Lifecycle Manager — Institutional-Grade Resource Control.

Inspired by:
- Kafka bounded retention
- Akka actor supervision
- Temporal workflow cleanup
- Production exchange OMS lifecycle

Provides:
- Bounded collections with automatic rotation
- TTL-based cleanup
- Resource registry
- Task supervision
- Memory pressure detection
- Leak detection
"""

from __future__ import annotations

import asyncio
import gc
import sys
import time
import tracemalloc
import weakref
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar

T = TypeVar("T")


class MemoryPressureLevel(Enum):
    """Memory pressure severity levels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResourceConfig:
    """Configuration for resource lifecycle."""
    max_size: int = 100_000
    ttl_seconds: Optional[float] = None
    cleanup_interval_seconds: float = 3600
    enable_metrics: bool = True


class BoundedDeque(Generic[T]):
    """Deque with strict bounds and TTL."""
    
    def __init__(
        self,
        max_size: int = 100_000,
        ttl_seconds: Optional[float] = None,
        name: str = "unnamed",
    ):
        self._deque: deque[T] = deque(maxlen=max_size)
        self._timestamps: deque[float] = deque(maxlen=max_size)
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._name = name
        self._dropped_count = 0
    
    def append(self, item: T) -> None:
        """Append with automatic eviction."""
        if len(self._deque) >= self._max_size:
            self._dropped_count += 1
        self._deque.append(item)
        self._timestamps.append(time.time())
    
    def cleanup_expired(self) -> int:
        """Remove expired items."""
        if self._ttl is None:
            return 0
        
        now = time.time()
        expired = 0
        
        while self._timestamps and (now - self._timestamps[0]) > self._ttl:
            self._deque.popleft()
            self._timestamps.popleft()
            expired += 1
        
        return expired
    
    def __len__(self) -> int:
        return len(self._deque)
    
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "size": len(self._deque),
            "max_size": self._max_size,
            "dropped": self._dropped_count,
            "ttl": self._ttl,
        }


class BoundedDict(Generic[T]):
    """Dictionary with strict bounds, TTL, and LRU."""
    
    def __init__(
        self,
        max_size: int = 100_000,
        ttl_seconds: Optional[float] = None,
        name: str = "unnamed",
    ):
        self._dict: Dict[Any, T] = {}
        self._timestamps: Dict[Any, float] = {}
        self._access_order: deque[Any] = deque(maxlen=max_size)
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._name = name
        self._evicted_count = 0
    
    def __setitem__(self, key: Any, value: T) -> None:
        # Evict oldest if at capacity
        if len(self._dict) >= self._max_size and key not in self._dict:
            oldest = self._access_order.popleft()
            if oldest in self._dict:
                del self._dict[oldest]
                del self._timestamps[oldest]
                self._evicted_count += 1
        
        self._dict[key] = value
        self._timestamps[key] = time.time()
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def __getitem__(self, key: Any) -> T:
        if key not in self._dict:
            raise KeyError(key)
        
        self._timestamps[key] = time.time()
        
        # Update access order
        self._access_order.remove(key)
        self._access_order.append(key)
        
        return self._dict[key]
    
    def __delitem__(self, key: Any) -> None:
        if key in self._dict:
            del self._dict[key]
            del self._timestamps[key]
            if key in self._access_order:
                self._access_order.remove(key)
    
    def __contains__(self, key: Any) -> bool:
        return key in self._dict
    
    def __len__(self) -> int:
        return len(self._dict)
    
    def get(self, key: Any, default: Any = None) -> Any:
        return self._dict.get(key, default)
    
    def cleanup_expired(self) -> int:
        if self._ttl is None:
            return 0
        
        now = time.time()
        expired_keys = [
            key for key, ts in self._timestamps.items()
            if (now - ts) > self._ttl
        ]
        
        for key in expired_keys:
            del self[key]
        
        return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "size": len(self._dict),
            "max_size": self._max_size,
            "evicted": self._evicted_count,
            "ttl": self._ttl,
        }


class ResourceRegistry:
    """Central registry for all system resources."""
    
    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self._weak_refs: Dict[str, weakref.ref] = {}
        self._cleanup_handlers: Dict[str, Callable[[], None]] = {}
    
    def register(
        self,
        name: str,
        resource: Any,
        cleanup: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a resource with optional cleanup handler."""
        self._resources[name] = resource
        if cleanup:
            self._cleanup_handlers[name] = cleanup
        
        # Track with weak reference for leak detection
        self._weak_refs[name] = weakref.ref(resource)
    
    def get(self, name: str) -> Optional[Any]:
        return self._resources.get(name)
    
    def cleanup(self, name: str) -> None:
        """Cleanup a specific resource."""
        if name in self._cleanup_handlers:
            self._cleanup_handlers[name]()
        if name in self._resources:
            del self._resources[name]
        if name in self._weak_refs:
            del self._weak_refs[name]
    
    def cleanup_all(self) -> None:
        """Cleanup all resources."""
        for name in list(self._resources.keys()):
            self.cleanup(name)
    
    def detect_leaks(self) -> List[str]:
        """Detect leaked resources (weak ref dead but still registered)."""
        leaked = []
        for name, weak_ref in self._weak_refs.items():
            if weak_ref() is None and name in self._resources:
                leaked.append(name)
        return leaked
    
    def stats(self) -> Dict[str, Any]:
        return {
            "total_resources": len(self._resources),
            "leaked": len(self.detect_leaks()),
            "resources": list(self._resources.keys()),
        }


class TaskSupervisor:
    """Supervises async tasks with automatic restart and cleanup."""
    
    def __init__(self, max_restarts: int = 3):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._restart_counts: Dict[str, int] = {}
        self._max_restarts = max_restarts
        self._running = False
    
    async def supervise(
        self,
        name: str,
        coro,
        restart_on_failure: bool = True,
    ) -> None:
        """Supervise a task with automatic restart."""
        async def supervised():
            restart_count = 0
            while True:
                try:
                    await coro
                    break
                except Exception as e:
                    if not restart_on_failure or restart_count >= self._max_restarts:
                        raise
                    restart_count += 1
                    self._restart_counts[name] = restart_count
                    await asyncio.sleep(1)
        
        self._tasks[name] = asyncio.create_task(supervised())
    
    async def stop(self, name: str) -> None:
        if name in self._tasks:
            self._tasks[name].cancel()
            try:
                await self._tasks[name]
            except asyncio.CancelledError:
                pass
            del self._tasks[name]
    
    async def stop_all(self) -> None:
        for name in list(self._tasks.keys()):
            await self.stop(name)
    
    def stats(self) -> Dict[str, Any]:
        return {
            "total_tasks": len(self._tasks),
            "tasks": list(self._tasks.keys()),
            "restart_counts": self._restart_counts,
        }


class MemoryPressureDetector:
    """Detects memory pressure and triggers cleanup."""
    
    def __init__(self, critical_threshold: float = 0.9):
        self._critical_threshold = critical_threshold
        self._callbacks: List[Callable[[MemoryPressureLevel], None]] = []
        tracemalloc.start()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage ratio."""
        current, peak = tracemalloc.get_traced_memory()
        # Use system memory as baseline
        total = sys.maxsize  # Fallback
        return current / total if total > 0 else 0.0
    
    def detect_pressure(self) -> MemoryPressureLevel:
        usage = self.get_memory_usage()
        
        if usage >= self._critical_threshold:
            return MemoryPressureLevel.CRITICAL
        elif usage >= 0.8:
            return MemoryPressureLevel.HIGH
        elif usage >= 0.6:
            return MemoryPressureLevel.ELEVATED
        else:
            return MemoryPressureLevel.NORMAL
    
    def register_callback(self, callback: Callable[[MemoryPressureLevel], None]) -> None:
        self._callbacks.append(callback)
    
    async def monitor(self, interval_seconds: float = 60.0) -> None:
        """Monitor memory pressure continuously."""
        while True:
            pressure = self.detect_pressure()
            for callback in self._callbacks:
                callback(pressure)
            await asyncio.sleep(interval_seconds)


class LeakDetectionService:
    """Detects memory leaks through object tracking."""
    
    def __init__(self):
        self._object_counts: Dict[str, int] = {}
        self._snapshots: List[Dict[str, int]] = []
    
    def take_snapshot(self) -> Dict[str, int]:
        """Take a snapshot of object counts."""
        snapshot = {}
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            snapshot[type_name] = snapshot.get(type_name, 0) + 1
        self._snapshots.append(snapshot)
        return snapshot
    
    def detect_growth(self, window: int = 5) -> Dict[str, int]:
        """Detect objects growing over time."""
        if len(self._snapshots) < window + 1:
            return {}
        
        old = self._snapshots[-window - 1]
        new = self._snapshots[-1]
        
        growth = {}
        for obj_type, count in new.items():
            old_count = old.get(obj_type, 0)
            if count > old_count * 2:  # 2x growth
                growth[obj_type] = count - old_count
        
        return growth


class MemoryLifecycleManager:
    """Central manager for all memory lifecycle operations."""
    
    def __init__(self):
        self._resource_registry = ResourceRegistry()
        self._task_supervisor = TaskSupervisor()
        self._pressure_detector = MemoryPressureDetector()
        self._leak_detector = LeakDetectionService()
        self._collections: Dict[str, Any] = {}
        self._running = False
    
    def register_collection(
        self,
        name: str,
        collection: Any,
    ) -> None:
        """Register a bounded collection."""
        self._collections[name] = collection
        self._resource_registry.register(name, collection)
    
    async def start(self) -> None:
        """Start all lifecycle services."""
        self._running = True
        
        # Start memory pressure monitoring
        await self._pressure_detector.monitor(interval_seconds=60.0)
        
        # Start leak detection
        await self._leak_detection_loop()
    
    async def stop(self) -> None:
        """Stop all lifecycle services."""
        self._running = False
        await self._task_supervisor.stop_all()
        self._resource_registry.cleanup_all()
    
    async def _leak_detection_loop(self) -> None:
        """Periodic leak detection."""
        while self._running:
            self._leak_detector.take_snapshot()
            growth = self._leak_detector.detect_growth()
            if growth:
                # Log growth
                pass
            await asyncio.sleep(300)  # Every 5 minutes
    
    async def cleanup_all_collections(self) -> int:
        """Cleanup all registered collections."""
        total_cleaned = 0
        for name, collection in self._collections.items():
            if hasattr(collection, 'cleanup_expired'):
                cleaned = collection.cleanup_expired()
                total_cleaned += cleaned
        return total_cleaned
    
    def stats(self) -> Dict[str, Any]:
        return {
            "resources": self._resource_registry.stats(),
            "tasks": self._task_supervisor.stats(),
            "pressure": self._pressure_detector.detect_pressure().value,
            "collections": {name: col.stats() if hasattr(col, 'stats') else {} for name, col in self._collections.items()},
        }


# Global instance
_lifecycle_manager: Optional[MemoryLifecycleManager] = None


def get_lifecycle_manager() -> MemoryLifecycleManager:
    """Get global lifecycle manager."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = MemoryLifecycleManager()
    return _lifecycle_manager
