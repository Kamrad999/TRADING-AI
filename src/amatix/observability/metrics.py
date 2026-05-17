"""Prometheus-compatible metrics for AMATIS.

Institutional-grade observability with:
    - Counter, Gauge, Histogram metrics
    - Label support for dimensions
    - Aggregation and export
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class MetricValue:
    """Single metric value with labels."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Monotonically increasing counter metric."""
    
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._values: Dict[frozenset, float] = defaultdict(float)
        self._created = time.time()
    
    def inc(self, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
        """Increment counter."""
        label_key = frozenset((labels or {}).items())
        self._values[label_key] += value
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        label_key = frozenset((labels or {}).items())
        return self._values[label_key]
    
    def get_all(self) -> Dict[frozenset, float]:
        """Get all label combinations."""
        return dict(self._values)
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}"]
        lines.append(f"# TYPE {self.name} counter")
        
        for label_set, value in self._values.items():
            if label_set:
                labels_str = ",".join(f'{k}="{v}"' for k, v in label_set)
                lines.append(f'{self.name}{{{labels_str}}} {value}')
            else:
                lines.append(f'{self.name} {value}')
        
        return "\n".join(lines)


class Gauge:
    """Gauge metric that can go up and down."""
    
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._values: Dict[frozenset, float] = {}
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        label_key = frozenset((labels or {}).items())
        self._values[label_key] = value
    
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment gauge."""
        label_key = frozenset((labels or {}).items())
        self._values[label_key] = self._values.get(label_key, 0) + value
    
    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement gauge."""
        self.inc(-value, labels)
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        label_key = frozenset((labels or {}).items())
        return self._values.get(label_key, 0)
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}"]
        lines.append(f"# TYPE {self.name} gauge")
        
        for label_set, value in self._values.items():
            if label_set:
                labels_str = ",".join(f'{k}="{v}"' for k, v in label_set)
                lines.append(f'{self.name}{{{labels_str}}} {value}')
            else:
                lines.append(f'{self.name} {value}')
        
        return "\n".join(lines)


class Histogram:
    """Histogram metric for latency distributions."""
    
    DEFAULT_BUCKETS = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ]
    
    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: Dict[frozenset, List[int]] = defaultdict(lambda: [0] * len(self.buckets))
        self._sums: Dict[frozenset, float] = defaultdict(float)
        self._totals: Dict[frozenset, int] = defaultdict(int)
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value."""
        label_key = frozenset((labels or {}).items())
        
        # Update bucket counts
        for i, bucket in enumerate(self.buckets):
            if value <= bucket:
                self._counts[label_key][i] += 1
        
        self._sums[label_key] += value
        self._totals[label_key] += 1
    
    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """Get total observations."""
        label_key = frozenset((labels or {}).items())
        return self._totals[label_key]
    
    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get sum of all observations."""
        label_key = frozenset((labels or {}).items())
        return self._sums[label_key]
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [f"# HELP {self.name} {self.description}"]
        lines.append(f"# TYPE {self.name} histogram")
        
        for label_set in self._totals.keys():
            label_str = ""
            if label_set:
                label_str = ",".join(f'{k}="{v}"' for k, v in label_set)
                label_str = "," + label_str
            
            # Bucket counts
            for i, bucket in enumerate(self.buckets):
                lines.append(
                    f'{self.name}_bucket{{le="{bucket}"{label_str}}} {self._counts[label_set][i]}'
                )
            
            # +Inf bucket (total count)
            lines.append(f'{self.name}_bucket{{le="+Inf"{label_str}}} {self._totals[label_set]}')
            
            # Sum
            lines.append(f'{self.name}_sum{{{label_str[1:] if label_str else ""}}} {self._sums[label_set]}')
            
            # Count
            lines.append(f'{self.name}_count{{{label_str[1:] if label_str else ""}}} {self._totals[label_set]}')
        
        return "\n".join(lines)


class MetricsRegistry:
    """Central registry for all metrics.
    
    Usage:
        registry = MetricsRegistry()
        
        # Create metrics
        orders_counter = registry.counter("orders_total", "Total orders")
        latency_hist = registry.histogram("order_latency_seconds", "Order latency")
        
        # Record values
        orders_counter.inc({"side": "buy"})
        latency_hist.observe(0.150)
        
        # Export for Prometheus
        print(registry.to_prometheus())
    """
    
    def __init__(self) -> None:
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
    
    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create counter."""
        if name not in self._counters:
            self._counters[name] = Counter(name, description)
        return self._counters[name]
    
    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create gauge."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description)
        return self._gauges[name]
    
    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Get or create histogram."""
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, description, buckets)
        return self._histograms[name]
    
    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        sections = []
        
        for counter in self._counters.values():
            sections.append(counter.to_prometheus())
        
        for gauge in self._gauges.values():
            sections.append(gauge.to_prometheus())
        
        for histogram in self._histograms.values():
            sections.append(histogram.to_prometheus())
        
        return "\n\n".join(sections)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metric values as dictionary."""
        result = {}
        
        for name, counter in self._counters.items():
            result[name] = {
                "type": "counter",
                "values": {str(k): v for k, v in counter.get_all().items()},
            }
        
        for name, gauge in self._gauges.items():
            result[name] = {
                "type": "gauge",
                "value": gauge.get(),
            }
        
        for name, hist in self._histograms.items():
            result[name] = {
                "type": "histogram",
                "count": sum(hist._totals.values()),
            }
        
        return result


# Global registry for convenience
_global_registry: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    """Get global metrics registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MetricsRegistry()
    return _global_registry


def counter(name: str, description: str = "") -> Counter:
    """Get or create counter in global registry."""
    return get_metrics_registry().counter(name, description)


def gauge(name: str, description: str = "") -> Gauge:
    """Get or create gauge in global registry."""
    return get_metrics_registry().gauge(name, description)


def histogram(name: str, description: str = "") -> Histogram:
    """Get or create histogram in global registry."""
    return get_metrics_registry().histogram(name, description)


# Timing decorator
@dataclass
class timed:
    """Decorator for timing function execution.
    
    Usage:
        @timed("risk_check_latency")
        async def assess_order(order):
            # ... code ...
            pass
    """
    metric_name: str
    
    def __call__(self, func: Callable) -> Callable:
        hist = histogram(self.metric_name, f"Latency for {func.__name__}")
        
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    hist.observe(time.time() - start)
            return wrapper
        else:
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    hist.observe(time.time() - start)
            return wrapper


import asyncio
