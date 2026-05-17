"""Signal processing pipeline.

Orchestrates signal flow:
    - Collection from multiple engines
    - Filtering and validation
    - Risk pre-check
    - Emission to event bus
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import EventPriority, EventType
from amatix.core.observability import get_logger, get_metrics, timed
from amatix.interfaces import SignalEngine as ISignalEngine
from amatix.signals.models import Signal, SignalBatch, SignalFilterConfig

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for signal pipeline."""
    min_confidence: float = 0.70
    enable_filtering: bool = True
    enable_risk_pre_check: bool = False  # Future
    max_concurrent_engines: int = 5
    batch_timeout_seconds: float = 5.0


class SignalPipeline:
    """Central pipeline for signal processing.
    
    Coordinates signal engines and manages signal flow:
        1. Collects signals from all engines
        2. Filters low-confidence signals
        3. Removes duplicates
        4. Emits SignalGenerated events
    
    Example:
        >>> pipeline = SignalPipeline(event_bus)
        >>> 
        >>> # Register engines
        >>> pipeline.register_engine(news_engine)
        >>> pipeline.register_engine(momentum_engine)
        >>> 
        >>> # Process signals
        >>> await pipeline.process(context)
    """
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """Initialize signal pipeline.
        
        Args:
            event_bus: Event bus for signal emission
            config: Pipeline configuration
        """
        self._event_bus = event_bus
        self._config = config or PipelineConfig()
        
        # Engines
        self._engines: List[ISignalEngine] = []
        
        # Filtering
        self._filter_config = SignalFilterConfig(
            min_confidence=self._config.min_confidence,
        )
        
        # Tracking
        self._recent_signals: Dict[str, Signal] = {}  # symbol -> last signal
        self._max_recent = 1000
        
        # Metrics
        self._processed_count = 0
        self._filtered_count = 0
    
    def register_engine(self, engine: ISignalEngine) -> None:
        """Register a signal engine.
        
        Args:
            engine: SignalEngine implementation
        """
        self._engines.append(engine)
        logger.info(
            "Signal engine registered",
            engine=engine.name,
            version=getattr(engine, 'version', 'unknown'),
        )
    
    def unregister_engine(self, engine_name: str) -> bool:
        """Unregister engine by name."""
        for i, engine in enumerate(self._engines):
            if engine.name == engine_name:
                self._engines.pop(i)
                return True
        return False
    
    @timed("signal_pipeline_process")
    async def process(self, context: Dict[str, Any]) -> SignalBatch:
        """Process signals from all engines.
        
        Args:
            context: Market context for signal generation
        
        Returns:
            Batch of processed signals
        """
        # Collect from all engines concurrently
        tasks = [
            self._collect_from_engine(engine, context)
            for engine in self._engines
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine signals
        all_signals: List[Signal] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Engine failed",
                    engine=self._engines[i].name,
                    error=str(result),
                )
                continue
            all_signals.extend(result)
        
        # Filter
        if self._config.enable_filtering:
            filtered = self._filter_signals(all_signals)
        else:
            filtered = all_signals
        
        # Create batch
        batch = SignalBatch(
            signals=filtered,
            source="pipeline",
        )
        
        # Emit events
        for signal in filtered:
            await self._emit_signal(signal)
        
        # Update stats
        self._processed_count += len(all_signals)
        self._filtered_count += len(all_signals) - len(filtered)
        
        logger.debug(
            "Pipeline processed signals",
            total=len(all_signals),
            filtered=len(filtered),
            engines=len(self._engines),
        )
        
        return batch
    
    async def _collect_from_engine(
        self,
        engine: ISignalEngine,
        context: Dict[str, Any],
    ) -> List[Signal]:
        """Collect signals from a single engine."""
        try:
            signals = await engine.generate(context)
            return signals if signals else []
        except Exception as e:
            logger.error(
                "Signal generation failed",
                engine=engine.name,
                error=str(e),
            )
            get_metrics().counter(
                "signal_generation_errors",
                labels={"engine": engine.name},
            )
            return []
    
    def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on configuration."""
        filtered: List[Signal] = []
        
        for signal in signals:
            # Confidence check
            if signal.confidence < self._filter_config.min_confidence:
                get_metrics().counter(
                    "signals_filtered_confidence",
                    labels={"engine": signal.source},
                )
                continue
            
            # Expiration check
            if signal.is_expired:
                get_metrics().counter("signals_filtered_expired")
                continue
            
            # Direction check
            if signal.direction not in self._filter_config.allowed_directions:
                continue
            
            # Recent duplicate check (same symbol + direction)
            key = f"{signal.symbol.canonical}:{signal.direction.value}"
            if key in self._recent_signals:
                last = self._recent_signals[key]
                # Skip if very similar to recent signal
                if abs(signal.confidence - last.confidence) < 0.1:
                    continue
            
            filtered.append(signal)
            self._recent_signals[key] = signal
            
            # Maintain cache size
            if len(self._recent_signals) > self._max_recent:
                # Remove oldest
                oldest = list(self._recent_signals.keys())[0]
                del self._recent_signals[oldest]
        
        return filtered
    
    async def _emit_signal(self, signal: Signal) -> None:
        """Emit signal to event bus."""
        await self._event_bus.emit_new(
            EventType.SIGNAL_GENERATED,
            {
                "signal_id": str(signal.signal_id),
                "symbol": str(signal.symbol),
                "direction": signal.direction.value,
                "confidence": signal.confidence,
                "strength": signal.strength.name,
                "source": signal.source,
                "timeframe": signal.timeframe.value,
            },
            priority=EventPriority.NORMAL,
            source="signal_pipeline",
        )
        
        get_metrics().counter(
            "signals_generated",
            labels={
                "source": signal.source,
                "direction": signal.direction.value,
            },
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "engines_registered": len(self._engines),
            "processed_count": self._processed_count,
            "filtered_count": self._filtered_count,
            "recent_signals": len(self._recent_signals),
            "min_confidence": self._config.min_confidence,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all engines."""
        statuses = {}
        
        for engine in self._engines:
            try:
                health = await engine.health_check()
                statuses[engine.name] = health
            except Exception as e:
                statuses[engine.name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
        
        overall = "healthy"
        if any(s.get("status") == "unhealthy" for s in statuses.values()):
            overall = "degraded"
        
        return {
            "status": overall,
            "engines": statuses,
        }
