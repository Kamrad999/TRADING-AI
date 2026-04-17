"""
Strategy Abstraction following Freqtrade patterns.
Implements pluggable strategy system with BaseStrategy and concrete implementations.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from ..infrastructure.logging import get_logger
from ..core.models import Signal, SignalDirection, Urgency, MarketRegime, SignalType
from ..signals.multi_factor_model import MultiFactorModel, FactorSignal
from ..events.event_classifier import EventClassification, EventType, ImpactLevel
from ..market.market_microstructure import MicrostructureSignals
from ..portfolio.position import Position


class StrategyType(Enum):
    """Strategy types following Freqtrade classification."""
    NEWS_BASED = "news_based"
    TECHNICAL_BASED = "technical_based"
    HYBRID = "hybrid"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"


class StrategyState(Enum):
    """Strategy states."""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    TESTING = "testing"


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    strategy_name: str
    strategy_type: StrategyType
    enabled: bool = True
    min_confidence: float = 0.6
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = 0.03
    max_positions: int = 5
    cooldown_period: int = 300  # seconds
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Result from strategy execution."""
    strategy_name: str
    signal: Optional[Signal]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class BaseStrategy(ABC):
    """
    Base strategy class following Freqtrade patterns.
    
    All strategies must inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize strategy."""
        self.config = config
        self.logger = get_logger(f"strategy.{config.strategy_name}")
        
        # Strategy state
        self.state = StrategyState.ACTIVE if config.enabled else StrategyState.DISABLED
        self.last_execution = {}
        self.performance_stats = {
            "total_signals": 0,
            "successful_signals": 0,
            "avg_confidence": 0.0,
            "avg_execution_time": 0.0,
            "last_update": datetime.now()
        }
        
        # Multi-factor model integration
        self.multi_factor_model = MultiFactorModel()
        
        self.logger.info(f"Strategy initialized: {config.strategy_name}")
    
    @abstractmethod
    def generate_signal(self, symbol: str, market_data: Dict[str, Any],
                       event_classifications: List[EventClassification] = None,
                       microstructure: Optional[MicrostructureSignals] = None,
                       positions: List[Position] = None) -> StrategyResult:
        """
        Generate trading signal.
        
        Args:
            symbol: Trading symbol
            market_data: Market data including technical indicators
            event_classifications: Event classifications
            microstructure: Market microstructure data
            positions: Current positions
            
        Returns:
            Strategy result with signal
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """
        Validate generated signal.
        
        Args:
            signal: Generated signal
            market_data: Market data
            
        Returns:
            True if signal is valid
        """
        pass
    
    def can_execute(self, symbol: str) -> bool:
        """Check if strategy can execute for symbol."""
        try:
            # Check strategy state
            if self.state != StrategyState.ACTIVE:
                return False
            
            # Check cooldown period
            last_time = self.last_execution.get(symbol)
            if last_time:
                cooldown_elapsed = (datetime.now() - last_time).total_seconds()
                if cooldown_elapsed < self.config.cooldown_period:
                    return False
            
            # Check position limits
            current_positions = len([p for p in positions if p.symbol == symbol]) if positions else 0
            if current_positions >= self.config.max_positions:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check execution ability: {e}")
            return False
    
    def update_performance(self, result: StrategyResult) -> None:
        """Update strategy performance metrics."""
        try:
            self.performance_stats["total_signals"] += 1
            self.performance_stats["last_update"] = datetime.now()
            
            if result.signal:
                self.performance_stats["successful_signals"] += 1
            
            # Update average confidence
            total_signals = self.performance_stats["total_signals"]
            current_avg = self.performance_stats["avg_confidence"]
            new_avg = (current_avg * (total_signals - 1) + result.confidence) / total_signals
            self.performance_stats["avg_confidence"] = new_avg
            
            # Update average execution time
            current_time_avg = self.performance_stats["avg_execution_time"]
            new_time_avg = (current_time_avg * (total_signals - 1) + result.execution_time) / total_signals
            self.performance_stats["avg_execution_time"] = new_time_avg
            
        except Exception as e:
            self.logger.error(f"Failed to update performance: {e}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            "name": self.config.strategy_name,
            "type": self.config.strategy_type.value,
            "state": self.state.value,
            "enabled": self.config.enabled,
            "performance": self.performance_stats,
            "config": {
                "min_confidence": self.config.min_confidence,
                "max_position_size": self.config.max_position_size,
                "stop_loss_pct": self.config.stop_loss_pct,
                "take_profit_pct": self.config.take_profit_pct
            }
        }


class NewsStrategy(BaseStrategy):
    """
    News-based strategy following Freqtrade patterns.
    
    Generates signals based on news sentiment and event classifications.
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # News-specific parameters
        self.impact_weights = {
            ImpactLevel.CRITICAL: 1.0,
            ImpactLevel.HIGH: 0.8,
            ImpactLevel.MEDIUM: 0.6,
            ImpactLevel.LOW: 0.4
        }
        
        # Event type preferences
        self.event_preferences = {
            EventType.MACRO_ECONOMIC: 0.9,
            EventType.CRYPTO_SPECIFIC: 0.8,
            EventType.EARNINGS: 0.7,
            EventType.REGULATORY: 0.85,
            EventType.GEOPOLITICAL: 0.6
        }
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any],
                       event_classifications: List[EventClassification] = None,
                       microstructure: Optional[MicrostructureSignals] = None,
                       positions: List[Position] = None) -> StrategyResult:
        """Generate news-based signal."""
        start_time = datetime.now()
        
        try:
            if not self.can_execute(symbol):
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=0.0,
                    reasoning="Strategy cannot execute",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Analyze events
            if not event_classifications:
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=0.0,
                    reasoning="No events to analyze",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Filter relevant events
            relevant_events = self._filter_relevant_events(symbol, event_classifications)
            
            if not relevant_events:
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=0.0,
                    reasoning="No relevant events",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Calculate signal strength
            signal_strength = self._calculate_news_signal_strength(relevant_events)
            
            if abs(signal_strength) < self.config.min_confidence:
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=abs(signal_strength),
                    reasoning=f"Signal strength {signal_strength:.2f} below threshold {self.config.min_confidence}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Determine direction
            direction = SignalDirection.BUY if signal_strength > 0 else SignalDirection.SELL
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                direction=direction,
                confidence=abs(signal_strength),
                urgency=self._determine_urgency(relevant_events),
                market_regime=self._determine_market_regime(market_data),
                position_size=self._calculate_position_size(signal_strength, market_data),
                execution_priority=self._calculate_execution_priority(relevant_events),
                signal_type=SignalType.NEWS,
                article_id=relevant_events[0].metadata.get("article_id"),
                generated_at=datetime.now(),
                metadata={
                    "strategy": self.config.strategy_name,
                    "events": [event.metadata for event in relevant_events],
                    "signal_strength": signal_strength,
                    "event_count": len(relevant_events)
                }
            )
            
            # Validate signal
            if not self.validate_signal(signal, market_data):
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=signal.confidence,
                    reasoning="Signal validation failed",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Update last execution
            self.last_execution[symbol] = datetime.now()
            
            # Generate reasoning
            reasoning = self._generate_news_reasoning(relevant_events, signal_strength, direction)
            
            result = StrategyResult(
                strategy_name=self.config.strategy_name,
                signal=signal,
                confidence=signal.confidence,
                reasoning=reasoning,
                metadata={
                    "event_count": len(relevant_events),
                    "signal_strength": signal_strength,
                    "top_event": relevant_events[0].event_type.value
                },
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Update performance
            self.update_performance(result)
            
            self.logger.info(f"News signal generated: {symbol} {direction.value} | "
                          f"Confidence: {signal.confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate news signal: {e}")
            return StrategyResult(
                strategy_name=self.config.strategy_name,
                signal=None,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def validate_signal(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Validate news-based signal."""
        try:
            # Check confidence threshold
            if signal.confidence < self.config.min_confidence:
                return False
            
            # Check position size
            if signal.position_size > self.config.max_position_size:
                return False
            
            # Check market conditions
            if market_data.get("volatility", 0.02) > 0.1:  # Very high volatility
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate news signal: {e}")
            return False
    
    def _filter_relevant_events(self, symbol: str, events: List[EventClassification]) -> List[EventClassification]:
        """Filter events relevant to the symbol."""
        relevant_events = []
        
        for event in events:
            # Check if symbol is affected
            if symbol in event.symbols_affected or symbol.lower() in [s.lower() for s in event.symbols_affected]:
                relevant_events.append(event)
        
        return relevant_events
    
    def _calculate_news_signal_strength(self, events: List[EventClassification]) -> float:
        """Calculate signal strength from events."""
        total_strength = 0.0
        
        for event in events:
            # Base strength from impact level
            impact_weight = self.impact_weights.get(event.impact_level, 0.5)
            
            # Adjust by event type preference
            event_weight = self.event_preferences.get(event.event_type, 0.5)
            
            # Adjust by confidence
            confidence_adjustment = event.confidence
            
            # Calculate event strength
            event_strength = impact_weight * event_weight * confidence_adjustment
            
            # Determine direction from event
            if "positive" in event.reasoning.lower():
                event_strength *= 1.0
            elif "negative" in event.reasoning.lower():
                event_strength *= -1.0
            else:
                # Determine from market regime impact
                regime_impact = event.market_regime_impact.get(MarketRegime.RISK_ON, 0)
                event_strength *= 1.0 if regime_impact > 0 else -1.0
            
            total_strength += event_strength
        
        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, total_strength / len(events)))
    
    def _determine_urgency(self, events: List[EventClassification]) -> Urgency:
        """Determine signal urgency from events."""
        max_impact = max([event.impact_level.value for event in events])
        
        if max_impact >= ImpactLevel.HIGH.value:
            return Urgency.HIGH
        elif max_impact >= ImpactLevel.MEDIUM.value:
            return Urgency.MEDIUM
        else:
            return Urgency.LOW
    
    def _determine_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Determine market regime from market data."""
        # Simplified regime detection
        volatility = market_data.get("volatility", 0.02)
        trend = market_data.get("trend", 0)
        
        if volatility > 0.05:
            return MarketRegime.VOLATILE
        elif trend > 0.02:
            return MarketRegime.RISK_ON
        elif trend < -0.02:
            return MarketRegime.RISK_OFF
        else:
            return MarketRegime.SIDEWAYS
    
    def _calculate_position_size(self, signal_strength: float, market_data: Dict[str, Any]) -> float:
        """Calculate position size based on signal strength."""
        base_size = self.config.max_position_size
        
        # Adjust by signal strength
        strength_multiplier = abs(signal_strength)
        
        # Adjust by volatility
        volatility = market_data.get("volatility", 0.02)
        volatility_multiplier = 1.0 / (1.0 + volatility * 10)  # Reduce size in high volatility
        
        position_size = base_size * strength_multiplier * volatility_multiplier
        
        return min(position_size, self.config.max_position_size)
    
    def _calculate_execution_priority(self, events: List[EventClassification]) -> int:
        """Calculate execution priority from events."""
        max_impact = max([event.impact_level.value for event in events])
        
        # Higher priority for higher impact events
        return int(max_impact * 2)
    
    def _generate_news_reasoning(self, events: List[EventClassification], signal_strength: float, direction: SignalDirection) -> str:
        """Generate reasoning for news-based signal."""
        top_event = events[0]
        
        reasoning_parts = [
            f"News-driven {direction.value} signal",
            f"Strength: {signal_strength:.2f}",
            f"Primary event: {top_event.event_type.value} ({top_event.impact_level.name})",
            f"Event confidence: {top_event.confidence:.2f}",
            f"Total events: {len(events)}"
        ]
        
        return " | ".join(reasoning_parts)


class TechnicalStrategy(BaseStrategy):
    """
    Technical analysis strategy following Freqtrade patterns.
    
    Generates signals based on technical indicators and market data.
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Technical indicator weights
        self.indicator_weights = {
            "rsi": 0.25,
            "macd": 0.20,
            "bollinger": 0.20,
            "volume": 0.15,
            "momentum": 0.20
        }
        
        # Technical parameters
        self.rsi_overbought = config.parameters.get("rsi_overbought", 70)
        self.rsi_oversold = config.parameters.get("rsi_oversold", 30)
        self.macd_signal_threshold = config.parameters.get("macd_threshold", 0.001)
        self.bb_threshold = config.parameters.get("bb_threshold", 0.8)
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any],
                       event_classifications: List[EventClassification] = None,
                       microstructure: Optional[MicrostructureSignals] = None,
                       positions: List[Position] = None) -> StrategyResult:
        """Generate technical-based signal."""
        start_time = datetime.now()
        
        try:
            if not self.can_execute(symbol):
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=0.0,
                    reasoning="Strategy cannot execute",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Check required indicators
            required_indicators = ["rsi", "macd", "bb_position", "volume_trend", "price_momentum"]
            missing_indicators = [ind for ind in required_indicators if ind not in market_data]
            
            if missing_indicators:
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=0.0,
                    reasoning=f"Missing indicators: {missing_indicators}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Calculate technical signal
            signal_strength = self._calculate_technical_signal_strength(market_data)
            
            if abs(signal_strength) < self.config.min_confidence:
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=abs(signal_strength),
                    reasoning=f"Signal strength {signal_strength:.2f} below threshold {self.config.min_confidence}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Determine direction
            direction = SignalDirection.BUY if signal_strength > 0 else SignalDirection.SELL
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                direction=direction,
                confidence=abs(signal_strength),
                urgency=self._determine_technical_urgency(market_data),
                market_regime=self._determine_market_regime(market_data),
                position_size=self._calculate_position_size(signal_strength, market_data),
                execution_priority=self._calculate_technical_execution_priority(market_data),
                signal_type=SignalType.TECHNICAL,
                article_id=None,
                generated_at=datetime.now(),
                metadata={
                    "strategy": self.config.strategy_name,
                    "indicators": {
                        "rsi": market_data["rsi"],
                        "macd": market_data["macd"],
                        "bb_position": market_data["bb_position"],
                        "volume_trend": market_data["volume_trend"],
                        "price_momentum": market_data["price_momentum"]
                    },
                    "signal_strength": signal_strength
                }
            )
            
            # Validate signal
            if not self.validate_signal(signal, market_data):
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=signal.confidence,
                    reasoning="Signal validation failed",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Update last execution
            self.last_execution[symbol] = datetime.now()
            
            # Generate reasoning
            reasoning = self._generate_technical_reasoning(market_data, signal_strength, direction)
            
            result = StrategyResult(
                strategy_name=self.config.strategy_name,
                signal=signal,
                confidence=signal.confidence,
                reasoning=reasoning,
                metadata={
                    "indicators_used": required_indicators,
                    "signal_strength": signal_strength
                },
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Update performance
            self.update_performance(result)
            
            self.logger.info(f"Technical signal generated: {symbol} {direction.value} | "
                          f"Confidence: {signal.confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate technical signal: {e}")
            return StrategyResult(
                strategy_name=self.config.strategy_name,
                signal=None,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def validate_signal(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Validate technical-based signal."""
        try:
            # Check confidence threshold
            if signal.confidence < self.config.min_confidence:
                return False
            
            # Check position size
            if signal.position_size > self.config.max_position_size:
                return False
            
            # Check technical indicator consistency
            rsi = market_data.get("rsi", 50)
            if signal.direction == SignalDirection.BUY and rsi > self.rsi_overbought:
                return False
            elif signal.direction == SignalDirection.SELL and rsi < self.rsi_oversold:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate technical signal: {e}")
            return False
    
    def _calculate_technical_signal_strength(self, market_data: Dict[str, Any]) -> float:
        """Calculate signal strength from technical indicators."""
        signal_strength = 0.0
        
        # RSI signal
        rsi = market_data["rsi"]
        if rsi < self.rsi_oversold:
            rsi_signal = (self.rsi_oversold - rsi) / self.rsi_oversold  # Positive for oversold
        elif rsi > self.rsi_overbought:
            rsi_signal = (self.rsi_overbought - rsi) / (100 - self.rsi_overbought)  # Negative for overbought
        else:
            rsi_signal = 0.0
        
        signal_strength += rsi_signal * self.indicator_weights["rsi"]
        
        # MACD signal
        macd = market_data["macd"]
        macd_signal = np.tanh(macd / self.macd_signal_threshold)
        signal_strength += macd_signal * self.indicator_weights["macd"]
        
        # Bollinger Bands signal
        bb_position = market_data["bb_position"]
        if bb_position > self.bb_threshold:
            bb_signal = -(bb_position - self.bb_threshold) / (1.0 - self.bb_threshold)  # Negative
        elif bb_position < (1.0 - self.bb_threshold):
            bb_signal = (self.bb_threshold - bb_position) / self.bb_threshold  # Positive
        else:
            bb_signal = 0.0
        
        signal_strength += bb_signal * self.indicator_weights["bollinger"]
        
        # Volume signal
        volume_trend = market_data["volume_trend"]
        volume_signal = np.tanh(volume_trend)
        signal_strength += volume_signal * self.indicator_weights["volume"]
        
        # Momentum signal
        price_momentum = market_data["price_momentum"]
        momentum_signal = np.tanh(price_momentum * 5)
        signal_strength += momentum_signal * self.indicator_weights["momentum"]
        
        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, signal_strength))
    
    def _determine_technical_urgency(self, market_data: Dict[str, Any]) -> Urgency:
        """Determine signal urgency from technical indicators."""
        rsi = market_data.get("rsi", 50)
        
        if rsi < 20 or rsi > 80:  # Extreme RSI
            return Urgency.HIGH
        elif rsi < 30 or rsi > 70:  # High RSI
            return Urgency.MEDIUM
        else:
            return Urgency.LOW
    
    def _determine_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Determine market regime from market data."""
        volatility = market_data.get("volatility", 0.02)
        trend = market_data.get("trend", 0)
        
        if volatility > 0.05:
            return MarketRegime.VOLATILE
        elif trend > 0.02:
            return MarketRegime.RISK_ON
        elif trend < -0.02:
            return MarketRegime.RISK_OFF
        else:
            return MarketRegime.SIDEWAYS
    
    def _calculate_position_size(self, signal_strength: float, market_data: Dict[str, Any]) -> float:
        """Calculate position size based on signal strength."""
        base_size = self.config.max_position_size
        
        # Adjust by signal strength
        strength_multiplier = abs(signal_strength)
        
        # Adjust by volatility
        volatility = market_data.get("volatility", 0.02)
        volatility_multiplier = 1.0 / (1.0 + volatility * 10)
        
        position_size = base_size * strength_multiplier * volatility_multiplier
        
        return min(position_size, self.config.max_position_size)
    
    def _calculate_technical_execution_priority(self, market_data: Dict[str, Any]) -> int:
        """Calculate execution priority from technical indicators."""
        rsi = market_data.get("rsi", 50)
        
        # Higher priority for extreme RSI levels
        if rsi < 20 or rsi > 80:
            return 3
        elif rsi < 30 or rsi > 70:
            return 2
        else:
            return 1
    
    def _generate_technical_reasoning(self, market_data: Dict[str, Any], signal_strength: float, direction: SignalDirection) -> str:
        """Generate reasoning for technical-based signal."""
        rsi = market_data["rsi"]
        macd = market_data["macd"]
        bb_position = market_data["bb_position"]
        
        reasoning_parts = [
            f"Technical {direction.value} signal",
            f"Strength: {signal_strength:.2f}",
            f"RSI: {rsi:.1f}",
            f"MACD: {macd:.4f}",
            f"BB Position: {bb_position:.2f}"
        ]
        
        return " | ".join(reasoning_parts)


class HybridStrategy(BaseStrategy):
    """
    Hybrid strategy combining news and technical analysis.
    
    Uses multi-factor model to combine different signal sources.
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Hybrid parameters
        self.news_weight = config.parameters.get("news_weight", 0.6)
        self.technical_weight = config.parameters.get("technical_weight", 0.4)
        self.min_consensus = config.parameters.get("min_consensus", 0.5)
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any],
                       event_classifications: List[EventClassification] = None,
                       microstructure: Optional[MicrostructureSignals] = None,
                       positions: List[Position] = None) -> StrategyResult:
        """Generate hybrid signal combining news and technical analysis."""
        start_time = datetime.now()
        
        try:
            if not self.can_execute(symbol):
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=0.0,
                    reasoning="Strategy cannot execute",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Generate multi-factor signal
            factor_signal = self.multi_factor_model.generate_signal(
                symbol, market_data, event_classifications, microstructure, positions
            )
            
            if not factor_signal:
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=0.0,
                    reasoning="Multi-factor model failed to generate signal",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Check consensus threshold
            if factor_signal.confidence < self.min_consensus:
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=factor_signal.confidence,
                    reasoning=f"Signal confidence {factor_signal.confidence:.2f} below consensus threshold {self.min_consensus}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Create hybrid signal
            signal = Signal(
                symbol=symbol,
                direction=factor_signal.direction,
                confidence=factor_signal.confidence,
                urgency=self._determine_hybrid_urgency(factor_signal),
                market_regime=self._determine_market_regime(market_data),
                position_size=self._calculate_position_size(factor_signal.strength.value, market_data),
                execution_priority=self._calculate_hybrid_execution_priority(factor_signal),
                signal_type=SignalType.HYBRID,
                article_id=None,
                generated_at=datetime.now(),
                metadata={
                    "strategy": self.config.strategy_name,
                    "factor_signal": {
                        "strength": factor_signal.strength.value,
                        "factor_count": len(factor_signal.factors),
                        "factor_score": factor_signal.factor_score,
                        "risk_adjusted_score": factor_signal.risk_adjusted_score
                    },
                    "factors": [
                        {
                            "name": f.name,
                            "category": f.category.value,
                            "value": f.value,
                            "weight": f.weight
                        }
                        for f in factor_signal.factors
                    ]
                }
            )
            
            # Validate signal
            if not self.validate_signal(signal, market_data):
                return StrategyResult(
                    strategy_name=self.config.strategy_name,
                    signal=None,
                    confidence=signal.confidence,
                    reasoning="Signal validation failed",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Update last execution
            self.last_execution[symbol] = datetime.now()
            
            # Generate reasoning
            reasoning = self._generate_hybrid_reasoning(factor_signal)
            
            result = StrategyResult(
                strategy_name=self.config.strategy_name,
                signal=signal,
                confidence=signal.confidence,
                reasoning=reasoning,
                metadata={
                    "factor_count": len(factor_signal.factors),
                    "factor_score": factor_signal.factor_score,
                    "risk_adjusted_score": factor_signal.risk_adjusted_score,
                    "expected_return": factor_signal.expected_return,
                    "risk_level": factor_signal.risk_level
                },
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Update performance
            self.update_performance(result)
            
            self.logger.info(f"Hybrid signal generated: {symbol} {signal.direction.value} | "
                          f"Confidence: {signal.confidence:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate hybrid signal: {e}")
            return StrategyResult(
                strategy_name=self.config.strategy_name,
                signal=None,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def validate_signal(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Validate hybrid signal."""
        try:
            # Check confidence threshold
            if signal.confidence < self.config.min_confidence:
                return False
            
            # Check position size
            if signal.position_size > self.config.max_position_size:
                return False
            
            # Check risk level
            risk_level = signal.metadata.get("risk_level", 0.5)
            if risk_level > 0.8:  # Very high risk
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate hybrid signal: {e}")
            return False
    
    def _determine_hybrid_urgency(self, factor_signal: FactorSignal) -> Urgency:
        """Determine urgency from factor signal."""
        if factor_signal.strength in [SignalStrength.VERY_STRONG, SignalStrength.STRONG]:
            return Urgency.HIGH
        elif factor_signal.strength == SignalStrength.MODERATE:
            return Urgency.MEDIUM
        else:
            return Urgency.LOW
    
    def _determine_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Determine market regime from market data."""
        volatility = market_data.get("volatility", 0.02)
        trend = market_data.get("trend", 0)
        
        if volatility > 0.05:
            return MarketRegime.VOLATILE
        elif trend > 0.02:
            return MarketRegime.RISK_ON
        elif trend < -0.02:
            return MarketRegime.RISK_OFF
        else:
            return MarketRegime.SIDEWAYS
    
    def _calculate_position_size(self, signal_strength: int, market_data: Dict[str, Any]) -> float:
        """Calculate position size based on signal strength."""
        base_size = self.config.max_position_size
        
        # Convert signal strength to multiplier
        strength_multipliers = {
            1: 0.2,  # VERY_WEAK
            2: 0.4,  # WEAK
            3: 0.6,  # MODERATE
            4: 0.8,  # STRONG
            5: 1.0   # VERY_STRONG
        }
        
        strength_multiplier = strength_multipliers.get(signal_strength, 0.5)
        
        # Adjust by volatility
        volatility = market_data.get("volatility", 0.02)
        volatility_multiplier = 1.0 / (1.0 + volatility * 10)
        
        position_size = base_size * strength_multiplier * volatility_multiplier
        
        return min(position_size, self.config.max_position_size)
    
    def _calculate_hybrid_execution_priority(self, factor_signal: FactorSignal) -> int:
        """Calculate execution priority from factor signal."""
        return factor_signal.strength.value
    
    def _generate_hybrid_reasoning(self, factor_signal: FactorSignal) -> str:
        """Generate reasoning for hybrid signal."""
        reasoning_parts = [
            f"Hybrid {factor_signal.direction.value} signal",
            f"Strength: {factor_signal.strength.name}",
            f"Confidence: {factor_signal.confidence:.2f}",
            f"Factor score: {factor_signal.factor_score:.3f}",
            f"Risk-adjusted: {factor_signal.risk_adjusted_score:.3f}",
            f"Factors: {len(factor_signal.factors)}"
        ]
        
        return " | ".join(reasoning_parts)


class StrategyManager:
    """
    Strategy manager following Freqtrade patterns.
    
    Manages multiple strategies and coordinates signal generation.
    """
    
    def __init__(self):
        """Initialize strategy manager."""
        self.logger = get_logger("strategy_manager")
        
        # Strategy registry
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("StrategyManager initialized")
    
    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Register a strategy."""
        try:
            self.strategies[strategy.config.strategy_name] = strategy
            self.strategy_configs[strategy.config.strategy_name] = strategy.config
            
            self.logger.info(f"Strategy registered: {strategy.config.strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register strategy: {e}")
    
    def generate_signals(self, symbol: str, market_data: Dict[str, Any],
                        event_classifications: List[EventClassification] = None,
                        microstructure: Optional[MicrostructureSignals] = None,
                        positions: List[Position] = None) -> List[StrategyResult]:
        """Generate signals from all active strategies."""
        try:
            results = []
            
            for strategy_name, strategy in self.strategies.items():
                if strategy.state == StrategyState.ACTIVE:
                    result = strategy.generate_signal(
                        symbol, market_data, event_classifications, microstructure, positions
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to generate signals: {e}")
            return []
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies."""
        try:
            summary = {
                "total_strategies": len(self.strategies),
                "active_strategies": len([s for s in self.strategies.values() if s.state == StrategyState.ACTIVE]),
                "strategies": {}
            }
            
            for name, strategy in self.strategies.items():
                summary["strategies"][name] = strategy.get_strategy_info()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get strategy summary: {e}")
            return {"error": str(e)}
