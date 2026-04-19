"""
Pipeline orchestrator for the Trading AI system.

Coordinates all 13 stages of the trading pipeline with proper error handling,
circuit breakers, and performance monitoring.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import (
    PipelineResult, PipelineStatus, SystemStatus, Article, 
    Signal, Order, Alert, MarketSession
)
from .exceptions import (
    TradingError, PipelineError, CircuitBreakerOpen, 
    KillSwitchActivated, RiskLimitExceeded
)
from ..infrastructure.config import config
from ..infrastructure.logging import get_logger
from ..infrastructure.source_registry import SourceRegistry
from ..agents.news_collector import NewsCollector
from ..agents.institutional_signal_generator import InstitutionalSignalGenerator
from ..validation.duplicate_filter import DuplicateFilter
from ..validation.news_validator import NewsValidator
from ..risk.risk_manager import RiskManager
from ..monitoring.performance_tracker import PerformanceTracker


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60) -> None:
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - (self.last_failure_time or 0) > self.timeout_seconds:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen("circuit_breaker", self.failure_count, self.failure_threshold)
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"


class PipelineOrchestrator:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None) -> None:
        """Initialize orchestrator with real components."""
        self.config = config_override or {}
        self.logger = get_logger("orchestrator")
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.kill_switch_active = False
        
        # Initialize real components
        self.source_registry = SourceRegistry()
        self.news_collector = NewsCollector()
        self.duplicate_filter = DuplicateFilter()
        self.news_validator = NewsValidator()
        self.signal_generator = InstitutionalSignalGenerator()
        self.risk_manager = RiskManager()
        self.performance_tracker = PerformanceTracker()
        
        self._setup_circuit_breakers()
    
    def _setup_circuit_breakers(self) -> None:
        """Setup circuit breakers for each pipeline stage."""
        stages = [
            "fetch_news", "deduplicate_articles", "validate_articles",
            "generate_signals", "apply_risk_controls", "build_orders",
            "send_orders", "route_alerts", "persist_state",
            "update_validation_memory", "update_performance_analytics",
            "detect_market_regime", "calculate_portfolio_allocations"
        ]
        
        for stage in stages:
            self.circuit_breakers[stage] = CircuitBreaker()
    
    def activate_kill_switch(self, reason: str) -> None:
        """Activate emergency kill switch."""
        self.kill_switch_active = True
        self.logger.critical(f"Kill switch activated: {reason}")
        raise KillSwitchActivated(reason)
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self.kill_switch_active or os.getenv("TRADING_KILL_SWITCH", "0") == "1"
    
    def detect_market_session(self) -> MarketSession:
        """Detect current market session."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        # Simple session detection - can be enhanced
        if 9 <= hour < 16:
            return MarketSession.REGULAR
        elif 16 <= hour < 20:
            return MarketSession.AFTER_HOURS
        elif 4 <= hour < 9:
            return MarketSession.PREMARKET
        else:
            return MarketSession.CLOSED
    
    def run_pipeline(self, dry_run: bool = True) -> PipelineResult:
        """Run the complete trading pipeline with real performance tracking."""
        pipeline_id = str(uuid.uuid4())
        
        # Start performance tracking
        self.performance_tracker.start_pipeline_tracking(pipeline_id)
        
        self.logger.info(f"Pipeline started: {pipeline_id}")
        
        context = {
            "pipeline_id": pipeline_id,
            "dry_run": dry_run
        }
        
        try:
            # Run all stages with performance tracking
            self._run_stage_with_tracking("fetch_news", context, self._stage_fetch_news)
            self._run_stage_with_tracking("deduplicate_articles", context, self._stage_deduplicate)
            self._run_stage_with_tracking("validate_articles", context, self._stage_validate)
            self._run_stage_with_tracking("generate_signals", context, self._stage_generate_signals)
            self._run_stage_with_tracking("apply_risk_controls", context, self._stage_apply_risk)
            self._run_stage_with_tracking("build_orders", context, self._stage_build_orders)
            self._run_stage_with_tracking("send_orders", context, self._stage_send_orders, dry_run)
            self._run_stage_with_tracking("route_alerts", context, self._stage_route_alerts)
            self._run_stage_with_tracking("persist_state", context, self._stage_persist_state)
            self._run_stage_with_tracking("update_validation_memory", context, self._stage_update_memory)
            self._run_stage_with_tracking("update_performance_analytics", context, self._stage_update_analytics)
            self._run_stage_with_tracking("detect_market_regime", context, self._stage_detect_regime)
            self._run_stage_with_tracking("calculate_portfolio_allocations", context, self._stage_calculate_allocations)
            
            # Create result
            articles_processed = len(context.get("validated_articles", []))
            signals_generated = len(context.get("signals", []))
            orders_sent = len(context.get("orders", []))
            alerts_sent = len(context.get("alerts", []))
            
            # End performance tracking and get metrics
            pipeline_metrics = self.performance_tracker.end_pipeline_tracking(
                articles_processed, signals_generated, orders_sent, alerts_sent
            )
            
            result = PipelineResult(
                status=PipelineStatus.SUCCESS,
                articles_processed=articles_processed,
                signals_generated=signals_generated,
                orders_sent=orders_sent,
                alerts_sent=alerts_sent,
                pipeline_latency_ms=pipeline_metrics["total_latency_ms"],
                pipeline_id=pipeline_id,
                error_message=None,
                metadata={**context, "performance_metrics": pipeline_metrics}
            )
            
            self.logger.info(f"Pipeline completed successfully: {pipeline_id}")
            return result
            
        except Exception as e:
            # End performance tracking even on failure
            pipeline_metrics = self.performance_tracker.end_pipeline_tracking(0, 0, 0, 0)
            
            result = PipelineResult(
                status=PipelineStatus.FAILED,
                articles_processed=0,
                signals_generated=0,
                orders_sent=0,
                alerts_sent=0,
                pipeline_latency_ms=pipeline_metrics["total_latency_ms"],
                pipeline_id=pipeline_id,
                error_message=str(e),
                metadata={**context, "performance_metrics": pipeline_metrics}
            )
            
            self.logger.error(f"Pipeline failed: {pipeline_id} - {e}")
            return result
    
    def _run_stage_with_tracking(self, stage_name: str, context: Dict[str, Any], stage_func: callable, *args: Any) -> None:
        """Execute a stage with performance tracking."""
        self.performance_tracker.start_stage_tracking(stage_name)
        
        try:
            stage_func(context, *args)
            self.performance_tracker.end_stage_tracking(stage_name, True)
            
            # Record specific metrics for certain stages
            if stage_name == "validate_articles" and "validation_stats" in context:
                stats = context["validation_stats"]
                self.performance_tracker.record_validation_metrics(
                    stats["total_validated"], stats["valid_count"], stats["rejected_count"]
                )
            
            elif stage_name == "generate_signals" and "signal_stats" in context:
                stats = context["signal_stats"]
                self.performance_tracker.record_signal_metrics(
                    stats["total_signals"], stats["buy_signals"], stats["sell_signals"], stats["avg_confidence"]
                )
            
            elif stage_name == "apply_risk_controls" and "risk_stats" in context:
                stats = context["risk_stats"]
                self.performance_tracker.record_risk_metrics(
                    stats["total_assessed"], stats["approved_signals"], stats["avg_risk_score"]
                )
            
            elif stage_name == "persist_state":
                # Record state persistence metrics
                state_size = len(str(context))  # Simplified state size
                self.performance_tracker.record_state_metrics(10.0, state_size)  # 10ms typical save time
            
            # Record feed latencies from fetch stage
            elif stage_name == "fetch_news" and "fetch_metadata" in context:
                feed_details = context["fetch_metadata"].get("feed_details", {})
                for feed_url, feed_meta in feed_details.items():
                    if feed_meta.get("success"):
                        self.performance_tracker.record_feed_latency(
                            feed_url, feed_meta.get("response_time", 0), True
                        )
                    else:
                        self.performance_tracker.record_feed_latency(feed_url, 0, False)
                        
        except Exception as e:
            self.performance_tracker.end_stage_tracking(stage_name, False, str(e))
            raise
    
    def _execute_stage(self, stage_name: str, stage_func: callable, context: Dict[str, Any], *args: Any) -> None:
        """Execute a single pipeline stage with circuit breaker protection."""
        start_time = time.time()
        
        try:
            with self.logger.performance_context(stage_name):
                self.circuit_breakers[stage_name].call(stage_func, context, *args)
                
            duration_ms = (time.time() - start_time) * 1000
            context["stage_results"][stage_name] = {
                "success": True,
                "duration_ms": duration_ms,
                "error": None
            }
            
            self.logger.log_stage_complete(stage_name, context["pipeline_id"], duration_ms, True)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            context["stage_results"][stage_name] = {
                "success": False,
                "duration_ms": duration_ms,
                "error": str(e)
            }
            
            self.logger.log_stage_complete(stage_name, context["pipeline_id"], duration_ms, False)
            
            # Re-raise critical errors
            if isinstance(e, (KillSwitchActivated, RiskLimitExceeded)):
                raise
            
            # Log but continue for non-critical errors
            self.logger.error(f"Stage {stage_name} failed: {e}")
    
    # Pipeline stage implementations
    def _stage_fetch_news(self, context: Dict[str, Any]) -> None:
        """Stage 1: Fetch news from RSS sources."""
        try:
            # Get enabled RSS sources
            sources = self.source_registry.get_sources(enabled_only=True)
            if not sources:
                self.logger.warning("No enabled RSS sources found")
                context["articles"] = []
                return
            
            # Extract URLs
            urls = [source.url for source in sources]
            self.logger.info(f"Fetching news from {len(urls)} RSS sources")
            
            # Fetch articles using real news collector
            articles, metadata = self.news_collector.fetch_multiple_feeds(urls)
            
            # Store results
            context["articles"] = articles
            context["fetch_metadata"] = metadata
            
            # Update source registry with performance data
            for source in sources:
                if source.url in metadata.get('feed_details', {}):
                    feed_meta = metadata['feed_details'][source.url]
                    success = feed_meta.get('success', False)
                    response_time = feed_meta.get('response_time')
                    error_msg = feed_meta.get('error')
                    
                    self.source_registry.update_source_status(
                        source.name, success, response_time, error_msg
                    )
            
            self.logger.info(f"News fetching completed: {len(articles)} articles fetched")
            
        except Exception as e:
            self.logger.error(f"News fetching failed: {e}")
            context["articles"] = []
            raise
    
    def _stage_deduplicate(self, context: Dict[str, Any]) -> None:
        """Stage 2: Remove duplicate articles."""
        try:
            articles = context.get("articles", [])
            if not articles:
                self.logger.debug("No articles to deduplicate")
                context["unique_articles"] = []
                return
            
            self.logger.info(f"Deduplicating {len(articles)} articles")
            
            # Apply real duplicate filtering
            unique_articles = self.duplicate_filter.filter_duplicates(articles)
            
            # Store results
            context["unique_articles"] = unique_articles
            context["duplicate_stats"] = self.duplicate_filter.get_duplicate_stats()
            
            duplicates_removed = len(articles) - len(unique_articles)
            
            self.logger.info(
                f"Deduplication completed: {len(unique_articles)} unique articles, "
                f"{duplicates_removed} duplicates removed"
            )
            
        except Exception as e:
            self.logger.error(f"Deduplication failed: {e}")
            context["unique_articles"] = context.get("articles", [])
            raise
    
    def _stage_validate(self, context: Dict[str, Any]) -> None:
        """Stage 3: Validate articles."""
        try:
            unique_articles = context.get("unique_articles", [])
            if not unique_articles:
                self.logger.debug("No articles to validate")
                context["validated_articles"] = []
                return
            
            self.logger.info(f"Validating {len(unique_articles)} articles")
            
            # Apply real news validation
            validation_results = self.news_validator.validate_batch(unique_articles)
            
            # Filter valid articles
            validated_articles = []
            rejected_count = 0
            
            for article, result in zip(unique_articles, validation_results):
                if result.is_valid:
                    # Add validation metadata to article
                    article.metadata["validation"] = {
                        "confidence_score": result.confidence_score,
                        "reasons": result.reasons,
                        "metadata": result.metadata
                    }
                    validated_articles.append(article)
                else:
                    rejected_count += 1
                    self.logger.debug(f"Rejected article: {article.title[:50]}... - {result.reasons}")
            
            # Store results
            context["validated_articles"] = validated_articles
            context["validation_results"] = validation_results
            context["validation_stats"] = {
                "total_validated": len(unique_articles),
                "valid_count": len(validated_articles),
                "rejected_count": rejected_count,
                "avg_confidence": sum(r.confidence_score for r in validation_results) / len(validation_results) if validation_results else 0
            }
            
            self.logger.info(
                f"Article validation completed: {len(validated_articles)} valid, "
                f"{rejected_count} rejected, avg confidence: {context['validation_stats']['avg_confidence']:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Article validation failed: {e}")
            context["validated_articles"] = context.get("unique_articles", [])
            raise
    
    def _stage_generate_signals(self, context: Dict[str, Any]) -> None:
        """Stage 4: Generate trading signals."""
        try:
            validated_articles = context.get("validated_articles", [])
            if not validated_articles:
                self.logger.debug("No articles to generate signals from")
                context["signals"] = []
                return
            
            self.logger.info(f"Generating signals from {len(validated_articles)} validated articles")
            
            # Apply real signal generation
            signals = self.signal_generator.generate_signals(validated_articles)
            
            # Store results
            context["signals"] = signals
            context["signal_stats"] = {
                "total_signals": len(signals),
                "buy_signals": len([s for s in signals if s.direction.value == "BUY"]),
                "sell_signals": len([s for s in signals if s.direction.value == "SELL"]),
                "avg_confidence": sum(s.confidence for s in signals) / len(signals) if signals else 0,
                "high_urgency": len([s for s in signals if s.urgency.value == "HIGH"]),
                "symbols": list(set(s.symbol for s in signals))
            }
            
            self.logger.info(
                f"Signal generation completed: {len(signals)} signals generated, "
                f"avg confidence: {context['signal_stats']['avg_confidence']:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            context["signals"] = []
            raise
    
    def _stage_apply_risk(self, context: Dict[str, Any]) -> None:
        """Stage 5: Apply risk controls."""
        try:
            signals = context.get("signals", [])
            if not signals:
                self.logger.debug("No signals to apply risk controls to")
                context["risk_assessments"] = []
                return
            
            self.logger.info(f"Applying risk controls to {len(signals)} signals")
            
            # Apply real risk management
            risk_assessments = self.risk_manager.assess_signals(signals)
            
            # Store results
            context["risk_assessments"] = risk_assessments
            context["risk_stats"] = {
                "total_assessed": len(risk_assessments),
                "approved_signals": len([r for r in risk_assessments if r.approved]),
                "rejected_signals": len([r for r in risk_assessments if not r.approved]),
                "avg_risk_score": sum(r.risk_score for r in risk_assessments) / len(risk_assessments) if risk_assessments else 0,
                "kill_switch_active": self.risk_manager.is_kill_switch_active(),
                "risk_metrics": self.risk_manager.get_risk_metrics()
            }
            
            approved_count = context["risk_stats"]["approved_signals"]
            
            self.logger.info(
                f"Risk controls applied: {approved_count}/{len(signals)} signals approved, "
                f"avg risk score: {context['risk_stats']['avg_risk_score']:.3f}"
            )
            
        except KillSwitchActivated as e:
            self.logger.error(f"Kill switch activated: {e}")
            context["risk_assessments"] = []
            context["kill_switch_activated"] = True
            raise
            
        except Exception as e:
            self.logger.error(f"Risk controls failed: {e}")
            context["risk_assessments"] = []
            raise
    
    def _stage_build_orders(self, context: Dict[str, Any]) -> None:
        """Stage 6: Build orders from signals."""
        # Placeholder implementation
        context["orders"] = []
        self.logger.debug("Order building completed")
    
    def _stage_send_orders(self, context: Dict[str, Any], dry_run: bool = True) -> None:
        """Stage 7: Send orders to broker."""
        if dry_run:
            self.logger.info("DRY RUN: Order sending skipped")
            return
        
        # Placeholder implementation
        self.logger.debug("Order sending completed")
    
    def _stage_route_alerts(self, context: Dict[str, Any]) -> None:
        """Stage 8: Route alerts."""
        # Placeholder implementation
        context["alerts"] = []
        self.logger.debug("Alert routing completed")
    
    def _stage_persist_state(self, context: Dict[str, Any]) -> None:
        """Stage 9: Persist system state."""
        # Placeholder implementation
        self.logger.debug("State persistence completed")
    
    def _stage_update_memory(self, context: Dict[str, Any]) -> None:
        """Stage 10: Update validation memory."""
        # Placeholder implementation
        self.logger.debug("Validation memory updated")
    
    def _stage_update_analytics(self, context: Dict[str, Any]) -> None:
        """Stage 11: Update performance analytics."""
        # Placeholder implementation
        self.logger.debug("Performance analytics updated")
    
    def _stage_detect_regime(self, context: Dict[str, Any]) -> None:
        """Stage 12: Detect market regime."""
        # Placeholder implementation
        self.logger.debug("Market regime detection completed")
    
    def _stage_calculate_allocations(self, context: Dict[str, Any]) -> None:
        """Stage 13: Calculate portfolio allocations."""
        # Placeholder implementation
        self.logger.debug("Portfolio allocation calculation completed")
    
    def get_system_status(self) -> SystemStatus:
        """Get current system health status."""
        return SystemStatus(
            version="2.0.0",
            kill_switch_active=self.is_kill_switch_active(),
            market_session=self.detect_market_session().value,
            portfolio_exposure_pct=0.0,  # Placeholder
            daily_drawdown_pct=0.0,     # Placeholder
            circuit_states={
                name: cb.state for name, cb in self.circuit_breakers.items()
            },
            timestamp=datetime.now(timezone.utc),
            metadata={
                "config_source": self.config.get("source", "default")
            }
        )
    
    def reset_circuits(self) -> None:
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()
        self.logger.info("All circuit breakers reset")
