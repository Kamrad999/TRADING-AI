"""Observability infrastructure for AMATIS.

Provides structured logging, metrics collection, and distributed tracing
preparation. All logs are machine-readable (JSON) for aggregation.

Design principles:
    - Structured logging: JSON format for log aggregation
    - Context propagation: Trace IDs flow through all components
    - Metrics: Prometheus-compatible counters, gauges, histograms
    - Sampling: Configurable log sampling for high-throughput paths
    - Correlation: All logs include trace_id for request tracing
"""

from __future__ import annotations

import functools
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

import structlog

# Configure structlog for production
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with bound context.
    
    All logs will be JSON-formatted with:
        - timestamp (ISO 8601)
        - logger name
        - log level
        - event message
        - custom bound fields
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured structured logger
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("signal_generated", symbol="AAPL", confidence=0.85)
    """
    return structlog.get_logger(name)


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class MetricValue:
    """Single metric measurement."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """In-process metrics collector (Prometheus-compatible).
    
    Collects counters, gauges, and histograms. Can be scraped by
    Prometheus or exported to other systems.
    
    This is a lightweight implementation. For production, consider
    using prometheus_client directly.
    
    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.counter("signals_generated", labels={"strategy": "momentum"})
        >>> metrics.gauge("portfolio_value", 125000.0)
        >>> metrics.histogram("execution_latency", 0.045, buckets=[0.01, 0.05, 0.1])
    """
    
    def __init__(self) -> None:
        self._counters: Dict[str, Dict[frozenset, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._gauges: Dict[str, Dict[frozenset, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._histograms: Dict[str, Dict[frozenset, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._histogram_buckets: Dict[str, List[float]] = {}
    
    def counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.
        
        Args:
            name: Metric name
            value: Amount to increment (default 1)
            labels: Metric labels/dimensions
        """
        label_key = frozenset((labels or {}).items())
        self._counters[name][label_key] += value
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric.
        
        Args:
            name: Metric name
            value: Current value
            labels: Metric labels/dimensions
        """
        label_key = frozenset((labels or {}).items())
        self._gauges[name][label_key] = value
    
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[list] = None,
    ) -> None:
        """Record a histogram observation.
        
        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels/dimensions
            buckets: Optional bucket boundaries
        """
        label_key = frozenset((labels or {}).items())
        self._histograms[name][label_key].append(value)
        
        if buckets:
            self._histogram_buckets[name] = buckets
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics as a dictionary."""
        return {
            "counters": {
                name: {dict(k): v for k, v in values.items()}
                for name, values in self._counters.items()
            },
            "gauges": {
                name: {dict(k): v for k, v in values.items()}
                for name, values in self._gauges.items()
            },
            "histograms": {
                name: {dict(k): v for k, v in values.items()}
                for name, values in self._histograms.items()
            },
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        # Counters
        for name, label_values in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            for labels, value in label_values.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{name}{{{label_str}}} {value}")
        
        # Gauges
        for name, label_values in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            for labels, value in label_values.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{name}{{{label_str}}} {value}")
        
        # Histograms (simplified - just sum and count)
        for name, label_values in self._histograms.items():
            lines.append(f"# TYPE {name} histogram")
            for labels, values in label_values.items():
                if values:
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels)
                    lines.append(f"{name}_sum{{{label_str}}} {sum(values)}")
                    lines.append(f"{name}_count{{{label_str}}} {len(values)}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


# =============================================================================
# Tracing / Context Propagation
# =============================================================================

@dataclass
class TraceContext:
    """Distributed tracing context."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    sampled: bool = True
    baggage: Dict[str, str] = field(default_factory=dict)


class Tracer:
    """Simple tracer for request flow tracking.
    
    Prepares for OpenTelemetry integration while providing
    basic tracing capabilities now.
    """
    
    def __init__(self) -> None:
        self._active_spans: Dict[str, Dict[str, Any]] = {}
    
    def start_span(
        self,
        name: str,
        context: Optional[TraceContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new trace span."""
        return Span(
            tracer=self,
            name=name,
            context=context or self._generate_context(),
            attributes=attributes or {},
        )
    
    def _generate_context(self) -> TraceContext:
        """Generate a new trace context."""
        import uuid
        return TraceContext(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:16],
        )


@dataclass
class Span:
    """A trace span representing an operation."""
    tracer: Tracer
    name: str
    context: TraceContext
    attributes: Dict[str, Any]
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def __enter__(self) -> Span:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end()
    
    def end(self) -> None:
        """End the span."""
        self.end_time = time.time()
        
        # Log the completed span
        logger = get_logger("amatix.tracing")
        duration = self.end_time - self.start_time
        
        log_data = {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "span_name": self.name,
            "duration_ms": duration * 1000,
            **self.attributes,
        }
        
        if exc_val:
            log_data["error"] = str(exc_val)
            logger.error("span_completed_with_error", **log_data)
        else:
            logger.debug("span_completed", **log_data)
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        self.attributes["error"] = str(exception)
        self.attributes["error_type"] = type(exception).__name__


# =============================================================================
# Decorators for observability
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def timed(metric_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution.
    
    Args:
        metric_name: Name for the timing metric (default: function name)
        labels: Additional labels for the metric
    
    Example:
        >>> @timed("signal_generation_latency")
        ... def generate_signals():
        ...     pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start
                metric_labels = {**(labels or {}), "status": status}
                get_metrics().histogram(name, duration, metric_labels)
        
        return wrapper  # type: ignore
    return decorator


def traced(span_name: Optional[str] = None):
    """Decorator to create a trace span for a function.
    
    Args:
        span_name: Name for the span (default: function name)
    
    Example:
        >>> @traced("risk_assessment")
        ... def assess_risk():
        ...     pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = span_name or func.__name__
            tracer = Tracer()
            
            with tracer.start_span(name) as span:
                span.set_attribute("function", func.__name__)
                span.set_attribute("module", func.__module__)
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        return wrapper  # type: ignore
    return decorator


# =============================================================================
# Health Checks
# =============================================================================

class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class HealthRegistry:
    """Registry for system health checks.
    
    Components can register health checks that are aggregated
    for monitoring and load balancer health endpoints.
    """
    
    def __init__(self) -> None:
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
    
    def register(
        self,
        name: str,
        check_fn: Callable[[], HealthCheck],
    ) -> None:
        """Register a health check."""
        self._checks[name] = check_fn
    
    def check_all(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        results = {}
        for name, check_fn in self._checks.items():
            try:
                results[name] = check_fn()
            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                )
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.check_all().values()
        
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            return HealthStatus.UNHEALTHY
        if any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


# Global health registry
_global_health: Optional[HealthRegistry] = None


def get_health_registry() -> HealthRegistry:
    """Get the global health registry."""
    global _global_health
    if _global_health is None:
        _global_health = HealthRegistry()
    return _global_health
