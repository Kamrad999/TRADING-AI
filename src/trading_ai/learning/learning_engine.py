"""
Self-Learning Trade Memory Engine following FinRL patterns.
Implements adaptive learning from trade outcomes and pattern detection.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json

from ..infrastructure.logging import get_logger
from ..portfolio.position import Position, PositionSide, PositionStatus
from ..core.models import Signal, SignalDirection
from ..events.event_classifier import EventClassification, EventType, ImpactLevel
from ..market.market_microstructure import MicrostructureSignals, LiquidityState


class LearningType(Enum):
    """Types of learning algorithms."""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    ENSEMBLE = "ensemble"


class PatternType(Enum):
    """Types of patterns to detect."""
    WINNING = "winning"
    LOSING = "losing"
    MARKET_REGIME = "market_regime"
    VOLATILITY = "volatility"
    TIME_OF_DAY = "time_of_day"
    CORRELATION = "correlation"


class AdaptationType(Enum):
    """Types of adaptations."""
    SIGNAL_WEIGHTS = "signal_weights"
    CONFIDENCE_THRESHOLDS = "confidence_thresholds"
    STRATEGY_PREFERENCES = "strategy_preferences"
    RISK_PARAMETERS = "risk_parameters"
    TIMING_PREFERENCES = "timing_preferences"


@dataclass
class TradeMemory:
    """Memory of a completed trade for learning."""
    trade_id: str
    symbol: str
    position_side: PositionSide
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl: float
    pnl_percentage: float
    max_unrealized: float
    max_drawdown: float
    duration_hours: float
    strategy: str
    signal_confidence: float
    market_conditions: Dict[str, Any]
    event_classifications: List[EventClassification]
    microstructure_signals: Optional[MicrostructureSignals]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Detected pattern from trade memory."""
    pattern_type: PatternType
    pattern_id: str
    description: str
    confidence: float
    frequency: int
    avg_return: float
    win_rate: float
    conditions: Dict[str, Any]
    recommendations: List[str]
    discovered_at: datetime
    last_updated: datetime


@dataclass
class AdaptationAction:
    """Adaptation action based on learning."""
    adaptation_type: AdaptationType
    target: str
    action: str
    old_value: Any
    new_value: Any
    confidence: float
    reasoning: str
    expected_improvement: float
    applied_at: datetime


class LearningEngine:
    """
    Self-learning trade memory engine following FinRL patterns.
    
    Key features:
    - Trade outcome tracking and analysis
    - Pattern detection and learning
    - Adaptive parameter adjustment
    - Strategy performance optimization
    - Market regime adaptation
    - Reinforcement learning integration
    """
    
    def __init__(self, memory_size: int = 10000, learning_rate: float = 0.01):
        """Initialize learning engine."""
        self.logger = get_logger("learning_engine")
        
        # Configuration
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        
        # Trade memory
        self.trade_memory: deque[TradeMemory] = deque(maxlen=memory_size)
        self.active_trades: Dict[str, TradeMemory] = {}
        
        # Pattern detection
        self.detected_patterns: Dict[str, LearningPattern] = {}
        self.pattern_history: List[LearningPattern] = []
        
        # Adaptation system
        self.adaptation_history: List[AdaptationAction] = []
        self.current_adaptations: Dict[str, Any] = {}
        
        # Learning models
        self._initialize_learning_models()
        
        # Performance tracking
        self.learning_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "patterns_detected": 0,
            "adaptations_applied": 0,
            "learning_progress": 0.0,
            "last_update": datetime.now()
        }
        
        self.logger.info("LearningEngine initialized with FinRL-style adaptive learning")
    
    def _initialize_learning_models(self) -> None:
        """Initialize learning models and algorithms."""
        self.learning_models = {
            # Reinforcement learning parameters
            "reinforcement": {
                "discount_factor": 0.95,
                "exploration_rate": 0.1,
                "learning_rate": self.learning_rate,
                "reward_decay": 0.99
            },
            
            # Pattern detection thresholds
            "pattern_detection": {
                "min_occurrences": 5,
                "min_confidence": 0.6,
                "min_return": 0.02,
                "min_win_rate": 0.6,
                "pattern_window": timedelta(days=30)
            },
            
            # Adaptation parameters
            "adaptation": {
                "min_confidence": 0.7,
                "min_improvement": 0.05,
                "adaptation_frequency": timedelta(hours=6),
                "max_adaptations_per_day": 10
            },
            
            # Feature weights for learning
            "feature_weights": {
                "signal_confidence": 0.3,
                "market_regime": 0.2,
                "volatility": 0.15,
                "time_of_day": 0.1,
                "event_impact": 0.15,
                "liquidity": 0.1
            }
        }
    
    def add_trade_to_memory(self, position: Position, signal: Signal,
                           event_classifications: List[EventClassification] = None,
                           microstructure_signals: Optional[MicrostructureSignals] = None) -> str:
        """
        Add completed trade to memory for learning.
        
        Args:
            position: Completed position
            signal: Original trading signal
            event_classifications: Event classifications during trade
            microstructure_signals: Market microstructure data
            
        Returns:
            Trade memory ID
        """
        try:
            if position.status != PositionStatus.CLOSED:
                self.logger.warning("Cannot add open position to memory")
                return ""
            
            # Create trade memory
            trade_id = f"trade_{position.symbol}_{int(position.exit_time.timestamp())}"
            
            trade_memory = TradeMemory(
                trade_id=trade_id,
                symbol=position.symbol,
                position_side=position.side,
                entry_time=position.entry_time,
                exit_time=position.exit_time,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                quantity=position.quantity,
                realized_pnl=position.realized_pnl,
                pnl_percentage=position.pnl_percentage,
                max_unrealized=position.max_unrealized_pnl,
                max_drawdown=position.max_drawdown,
                duration_hours=(position.exit_time - position.entry_time).total_seconds() / 3600,
                strategy=position.strategy,
                signal_confidence=signal.confidence,
                market_conditions=self._extract_market_conditions(position, signal),
                event_classifications=event_classifications or [],
                microstructure_signals=microstructure_signals,
                metadata={
                    "signal_id": signal.metadata.get("id"),
                    "entry_reason": position.entry_reason,
                    "exit_reason": position.exit_reason,
                    "position_id": position.id
                }
            )
            
            # Add to memory
            self.trade_memory.append(trade_memory)
            
            # Update metrics
            self._update_learning_metrics(trade_memory)
            
            # Trigger pattern detection
            self._trigger_pattern_detection()
            
            # Trigger adaptation if needed
            self._trigger_adaptation_check()
            
            self.logger.info(f"Trade added to memory: {trade_id} | P&L: ${trade_memory.realized_pnl:.2f}")
            
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Failed to add trade to memory: {e}")
            return ""
    
    def start_trade_tracking(self, position: Position, signal: Signal) -> str:
        """Start tracking a new trade."""
        try:
            trade_id = f"active_{position.symbol}_{int(datetime.now().timestamp())}"
            
            # Create temporary trade memory for active tracking
            trade_memory = TradeMemory(
                trade_id=trade_id,
                symbol=position.symbol,
                position_side=position.side,
                entry_time=position.entry_time,
                exit_time=datetime.now(),  # Will be updated when closed
                entry_price=position.entry_price,
                exit_price=position.current_price,
                quantity=position.quantity,
                realized_pnl=position.realized_pnl,
                pnl_percentage=position.pnl_percentage,
                max_unrealized=position.unrealized_pnl,
                max_drawdown=position.max_drawdown,
                duration_hours=0.0,
                strategy=position.strategy,
                signal_confidence=signal.confidence,
                market_conditions=self._extract_market_conditions(position, signal),
                event_classifications=[],
                microstructure_signals=None,
                metadata={"active": True}
            )
            
            self.active_trades[trade_id] = trade_memory
            
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Failed to start trade tracking: {e}")
            return ""
    
    def update_active_trade(self, trade_id: str, position: Position) -> None:
        """Update active trade tracking."""
        try:
            if trade_id not in self.active_trades:
                return
            
            trade_memory = self.active_trades[trade_id]
            
            # Update current values
            trade_memory.exit_price = position.current_price
            trade_memory.realized_pnl = position.realized_pnl
            trade_memory.pnl_percentage = position.pnl_percentage
            trade_memory.max_unrealized = position.unrealized_pnl
            trade_memory.max_drawdown = position.max_drawdown
            trade_memory.duration_hours = (datetime.now() - trade_memory.entry_time).total_seconds() / 3600
            
        except Exception as e:
            self.logger.error(f"Failed to update active trade {trade_id}: {e}")
    
    def complete_active_trade(self, trade_id: str, position: Position, signal: Signal,
                            event_classifications: List[EventClassification] = None,
                            microstructure_signals: Optional[MicrostructureSignals] = None) -> str:
        """Complete active trade and move to memory."""
        try:
            if trade_id not in self.active_trades:
                return ""
            
            # Update final values
            self.update_active_trade(trade_id, position)
            
            # Get trade memory
            trade_memory = self.active_trades[trade_id]
            trade_memory.exit_time = position.exit_time
            trade_memory.metadata["active"] = False
            
            # Add event classifications and microstructure
            if event_classifications:
                trade_memory.event_classifications = event_classifications
            if microstructure_signals:
                trade_memory.microstructure_signals = microstructure_signals
            
            # Move to permanent memory
            self.trade_memory.append(trade_memory)
            
            # Remove from active trades
            del self.active_trades[trade_id]
            
            # Update metrics
            self._update_learning_metrics(trade_memory)
            
            # Trigger learning processes
            self._trigger_pattern_detection()
            self._trigger_adaptation_check()
            
            self.logger.info(f"Active trade completed: {trade_id}")
            
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Failed to complete active trade {trade_id}: {e}")
            return ""
    
    def detect_patterns(self) -> List[LearningPattern]:
        """Detect patterns from trade memory."""
        try:
            new_patterns = []
            
            # Detect different types of patterns
            pattern_detectors = {
                PatternType.WINNING: self._detect_winning_patterns,
                PatternType.LOSING: self._detect_losing_patterns,
                PatternType.MARKET_REGIME: self._detect_market_regime_patterns,
                PatternType.VOLATILITY: self._detect_volatility_patterns,
                PatternType.TIME_OF_DAY: self._detect_time_patterns,
                PatternType.CORRELATION: self._detect_correlation_patterns
            }
            
            for pattern_type, detector in pattern_detectors.items():
                patterns = detector()
                new_patterns.extend(patterns)
            
            # Update detected patterns
            for pattern in new_patterns:
                self.detected_patterns[pattern.pattern_id] = pattern
                self.pattern_history.append(pattern)
            
            self.learning_metrics["patterns_detected"] = len(self.detected_patterns)
            
            self.logger.info(f"Detected {len(new_patterns)} new patterns")
            
            return new_patterns
            
        except Exception as e:
            self.logger.error(f"Failed to detect patterns: {e}")
            return []
    
    def _detect_winning_patterns(self) -> List[LearningPattern]:
        """Detect patterns in winning trades."""
        try:
            winning_trades = [t for t in self.trade_memory if t.realized_pnl > 0]
            
            if len(winning_trades) < self.learning_models["pattern_detection"]["min_occurrences"]:
                return []
            
            patterns = []
            
            # Analyze winning trade characteristics
            avg_confidence = np.mean([t.signal_confidence for t in winning_trades])
            avg_duration = np.mean([t.duration_hours for t in winning_trades])
            avg_return = np.mean([t.pnl_percentage for t in winning_trades])
            
            # Check for high confidence winning pattern
            if avg_confidence > 0.7:
                pattern = LearningPattern(
                    pattern_type=PatternType.WINNING,
                    pattern_id=f"high_confidence_wins_{int(datetime.now().timestamp())}",
                    description="High confidence signals lead to winning trades",
                    confidence=avg_confidence,
                    frequency=len(winning_trades),
                    avg_return=avg_return,
                    win_rate=1.0,  # All are winning by definition
                    conditions={
                        "min_signal_confidence": 0.7,
                        "strategy_distribution": self._calculate_strategy_distribution(winning_trades)
                    },
                    recommendations=[
                        "Increase weight for high confidence signals",
                        "Focus on strategies with high confidence performance"
                    ],
                    discovered_at=datetime.now(),
                    last_updated=datetime.now()
                )
                patterns.append(pattern)
            
            # Check for optimal duration pattern
            if 2 <= avg_duration <= 8:  # 2-8 hours optimal
                pattern = LearningPattern(
                    pattern_type=PatternType.WINNING,
                    pattern_id=f"optimal_duration_wins_{int(datetime.now().timestamp())}",
                    description="Optimal holding period leads to better returns",
                    confidence=0.8,
                    frequency=len(winning_trades),
                    avg_return=avg_return,
                    win_rate=1.0,
                    conditions={
                        "min_duration_hours": 2,
                        "max_duration_hours": 8,
                        "avg_duration": avg_duration
                    },
                    recommendations=[
                        "Target 2-8 hour holding periods",
                        "Consider scaling out after 4 hours"
                    ],
                    discovered_at=datetime.now(),
                    last_updated=datetime.now()
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to detect winning patterns: {e}")
            return []
    
    def _detect_losing_patterns(self) -> List[LearningPattern]:
        """Detect patterns in losing trades."""
        try:
            losing_trades = [t for t in self.trade_memory if t.realized_pnl < 0]
            
            if len(losing_trades) < self.learning_models["pattern_detection"]["min_occurrences"]:
                return []
            
            patterns = []
            
            # Analyze losing trade characteristics
            avg_confidence = np.mean([t.signal_confidence for t in losing_trades])
            avg_drawdown = np.mean([t.max_drawdown for t in losing_trades])
            avg_return = np.mean([t.pnl_percentage for t in losing_trades])
            
            # Check for high drawdown pattern
            if avg_drawdown > 0.05:  # > 5% drawdown
                pattern = LearningPattern(
                    pattern_type=PatternType.LOSING,
                    pattern_id=f"high_drawdown_losses_{int(datetime.now().timestamp())}",
                    description="High drawdown trades lead to losses",
                    confidence=0.8,
                    frequency=len(losing_trades),
                    avg_return=avg_return,
                    win_rate=0.0,  # All are losing by definition
                    conditions={
                        "max_drawdown_threshold": 0.05,
                        "avg_drawdown": avg_drawdown
                    },
                    recommendations=[
                        "Implement tighter stop losses",
                        "Reduce position size in volatile conditions"
                    ],
                    discovered_at=datetime.now(),
                    last_updated=datetime.now()
                )
                patterns.append(pattern)
            
            # Check for low confidence pattern
            if avg_confidence < 0.5:
                pattern = LearningPattern(
                    pattern_type=PatternType.LOSING,
                    pattern_id=f"low_confidence_losses_{int(datetime.now().timestamp())}",
                    description="Low confidence signals lead to losses",
                    confidence=0.7,
                    frequency=len(losing_trades),
                    avg_return=avg_return,
                    win_rate=0.0,
                    conditions={
                        "max_signal_confidence": 0.5,
                        "avg_confidence": avg_confidence
                    },
                    recommendations=[
                        "Increase minimum confidence threshold",
                        "Add confirmation for low confidence signals"
                    ],
                    discovered_at=datetime.now(),
                    last_updated=datetime.now()
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to detect losing patterns: {e}")
            return []
    
    def _detect_market_regime_patterns(self) -> List[LearningPattern]:
        """Detect patterns related to market regimes."""
        try:
            # Group trades by market regime
            regime_trades = defaultdict(list)
            for trade in self.trade_memory:
                regime = trade.market_conditions.get("market_regime", "unknown")
                regime_trades[regime].append(trade)
            
            patterns = []
            
            for regime, trades in regime_trades.items():
                if len(trades) < self.learning_models["pattern_detection"]["min_occurrences"]:
                    continue
                
                avg_return = np.mean([t.pnl_percentage for t in trades])
                win_rate = len([t for t in trades if t.realized_pnl > 0]) / len(trades)
                
                # Check for regime-specific performance
                if abs(avg_return) > 0.03:  # > 3% average return
                    pattern = LearningPattern(
                        pattern_type=PatternType.MARKET_REGIME,
                        pattern_id=f"regime_{regime}_performance_{int(datetime.now().timestamp())}",
                        description=f"{'Strong' if avg_return > 0 else 'Poor'} performance in {regime} regime",
                        confidence=abs(avg_return) / 0.05,  # Normalize to 0-1
                        frequency=len(trades),
                        avg_return=avg_return,
                        win_rate=win_rate,
                        conditions={
                            "market_regime": regime,
                            "min_trades": len(trades)
                        },
                        recommendations=[
                            f"{'Increase' if avg_return > 0 else 'Decrease'} exposure in {regime} regime",
                            f"{'Use more aggressive' if avg_return > 0 else 'Use more conservative'} strategies in {regime}"
                        ],
                        discovered_at=datetime.now(),
                        last_updated=datetime.now()
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to detect market regime patterns: {e}")
            return []
    
    def _detect_volatility_patterns(self) -> List[LearningPattern]:
        """Detect patterns related to volatility."""
        try:
            # Group trades by volatility
            volatility_trades = defaultdict(list)
            for trade in self.trade_memory:
                volatility = trade.market_conditions.get("volatility", 0.02)
                if volatility < 0.01:
                    vol_category = "low"
                elif volatility < 0.03:
                    vol_category = "medium"
                else:
                    vol_category = "high"
                volatility_trades[vol_category].append(trade)
            
            patterns = []
            
            for vol_category, trades in volatility_trades.items():
                if len(trades) < self.learning_models["pattern_detection"]["min_occurrences"]:
                    continue
                
                avg_return = np.mean([t.pnl_percentage for t in trades])
                win_rate = len([t for t in trades if t.realized_pnl > 0]) / len(trades)
                
                # Check for volatility-specific performance
                if abs(avg_return) > 0.02:  # > 2% average return
                    pattern = LearningPattern(
                        pattern_type=PatternType.VOLATILITY,
                        pattern_id=f"volatility_{vol_category}_performance_{int(datetime.now().timestamp())}",
                        description=f"{'Good' if avg_return > 0 else 'Poor'} performance in {vol_category} volatility",
                        confidence=abs(avg_return) / 0.04,
                        frequency=len(trades),
                        avg_return=avg_return,
                        win_rate=win_rate,
                        conditions={
                            "volatility_category": vol_category,
                            "min_trades": len(trades)
                        },
                        recommendations=[
                            f"{'Increase' if avg_return > 0 else 'Decrease'} position size in {vol_category} volatility",
                            f"{'Use tighter' if avg_return < 0 else 'Use wider'} stops in {vol_category} volatility"
                        ],
                        discovered_at=datetime.now(),
                        last_updated=datetime.now()
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to detect volatility patterns: {e}")
            return []
    
    def _detect_time_patterns(self) -> List[LearningPattern]:
        """Detect patterns related to time of day."""
        try:
            # Group trades by time of day
            time_trades = defaultdict(list)
            for trade in self.trade_memory:
                hour = trade.entry_time.hour
                if 6 <= hour < 12:
                    time_category = "morning"
                elif 12 <= hour < 18:
                    time_category = "afternoon"
                elif 18 <= hour < 22:
                    time_category = "evening"
                else:
                    time_category = "night"
                time_trades[time_category].append(trade)
            
            patterns = []
            
            for time_category, trades in time_trades.items():
                if len(trades) < self.learning_models["pattern_detection"]["min_occurrences"]:
                    continue
                
                avg_return = np.mean([t.pnl_percentage for t in trades])
                win_rate = len([t for t in trades if t.realized_pnl > 0]) / len(trades)
                
                # Check for time-specific performance
                if abs(avg_return) > 0.02:  # > 2% average return
                    pattern = LearningPattern(
                        pattern_type=PatternType.TIME_OF_DAY,
                        pattern_id=f"time_{time_category}_performance_{int(datetime.now().timestamp())}",
                        description=f"{'Good' if avg_return > 0 else 'Poor'} performance during {time_category}",
                        confidence=abs(avg_return) / 0.04,
                        frequency=len(trades),
                        avg_return=avg_return,
                        win_rate=win_rate,
                        conditions={
                            "time_category": time_category,
                            "min_trades": len(trades)
                        },
                        recommendations=[
                            f"{'Increase' if avg_return > 0 else 'Decrease'} trading during {time_category}",
                            f"{'Focus' if avg_return > 0 else 'Avoid'} {time_category} trading sessions"
                        ],
                        discovered_at=datetime.now(),
                        last_updated=datetime.now()
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to detect time patterns: {e}")
            return []
    
    def _detect_correlation_patterns(self) -> List[LearningPattern]:
        """Detect correlation patterns between factors."""
        try:
            patterns = []
            
            # Analyze correlation between signal confidence and returns
            if len(self.trade_memory) < 20:
                return patterns
            
            confidences = [t.signal_confidence for t in self.trade_memory]
            returns = [t.pnl_percentage for t in self.trade_memory]
            
            correlation = np.corrcoef(confidences, returns)[0, 1]
            
            if abs(correlation) > 0.3:  # Significant correlation
                pattern = LearningPattern(
                    pattern_type=PatternType.CORRELATION,
                    pattern_id=f"confidence_return_correlation_{int(datetime.now().timestamp())}",
                    description=f"{'Positive' if correlation > 0 else 'Negative'} correlation between confidence and returns",
                    confidence=abs(correlation),
                    frequency=len(self.trade_memory),
                    avg_return=np.mean(returns),
                    win_rate=len([r for r in returns if r > 0]) / len(returns),
                    conditions={
                        "correlation_coefficient": correlation,
                        "sample_size": len(self.trade_memory)
                    },
                    recommendations=[
                        f"{'Trust' if correlation > 0 else 'Question'} signal confidence more",
                        f"{'Increase' if correlation > 0 else 'Decrease'} confidence-based position sizing"
                    ],
                    discovered_at=datetime.now(),
                    last_updated=datetime.now()
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to detect correlation patterns: {e}")
            return []
    
    def generate_adaptations(self) -> List[AdaptationAction]:
        """Generate adaptation actions based on detected patterns."""
        try:
            adaptations = []
            
            # Generate adaptations for each detected pattern
            for pattern in self.detected_patterns.values():
                pattern_adaptations = self._generate_pattern_adaptations(pattern)
                adaptations.extend(pattern_adaptations)
            
            # Generate adaptations based on overall performance
            overall_adaptations = self._generate_overall_adaptations()
            adaptations.extend(overall_adaptations)
            
            # Filter and prioritize adaptations
            prioritized_adaptations = self._prioritize_adaptations(adaptations)
            
            return prioritized_adaptations
            
        except Exception as e:
            self.logger.error(f"Failed to generate adaptations: {e}")
            return []
    
    def _generate_pattern_adaptations(self, pattern: LearningPattern) -> List[AdaptationAction]:
        """Generate adaptations for a specific pattern."""
        adaptations = []
        
        try:
            if pattern.pattern_type == PatternType.WINNING:
                # Adapt to winning patterns
                if "high confidence" in pattern.description.lower():
                    adaptation = AdaptationAction(
                        adaptation_type=AdaptationType.SIGNAL_WEIGHTS,
                        target="high_confidence_weight",
                        action="increase",
                        old_value=1.0,
                        new_value=1.2,
                        confidence=pattern.confidence,
                        reasoning=pattern.description,
                        expected_improvement=pattern.avg_return * 0.1,
                        applied_at=datetime.now()
                    )
                    adaptations.append(adaptation)
            
            elif pattern.pattern_type == PatternType.LOSING:
                # Adapt to losing patterns
                if "high drawdown" in pattern.description.lower():
                    adaptation = AdaptationAction(
                        adaptation_type=AdaptationType.RISK_PARAMETERS,
                        target="stop_loss_pct",
                        action="decrease",
                        old_value=0.05,
                        new_value=0.03,
                        confidence=pattern.confidence,
                        reasoning=pattern.description,
                        expected_improvement=abs(pattern.avg_return) * 0.2,
                        applied_at=datetime.now()
                    )
                    adaptations.append(adaptation)
            
            elif pattern.pattern_type == PatternType.MARKET_REGIME:
                # Adapt to market regime patterns
                regime = pattern.conditions.get("market_regime", "unknown")
                if pattern.avg_return > 0:
                    adaptation = AdaptationAction(
                        adaptation_type=AdaptationType.STRATEGY_PREFERENCES,
                        target=f"regime_{regime}_preference",
                        action="increase",
                        old_value=1.0,
                        new_value=1.3,
                        confidence=pattern.confidence,
                        reasoning=pattern.description,
                        expected_improvement=pattern.avg_return * 0.15,
                        applied_at=datetime.now()
                    )
                    adaptations.append(adaptation)
            
        except Exception as e:
            self.logger.error(f"Failed to generate adaptations for pattern {pattern.pattern_id}: {e}")
        
        return adaptations
    
    def _generate_overall_adaptations(self) -> List[AdaptationAction]:
        """Generate adaptations based on overall performance."""
        adaptations = []
        
        try:
            if len(self.trade_memory) < 10:
                return adaptations
            
            # Calculate overall metrics
            total_trades = len(self.trade_memory)
            winning_trades = len([t for t in self.trade_memory if t.realized_pnl > 0])
            win_rate = winning_trades / total_trades
            avg_return = np.mean([t.pnl_percentage for t in self.trade_memory])
            
            # Adapt based on win rate
            if win_rate < 0.4:  # Low win rate
                adaptation = AdaptationAction(
                    adaptation_type=AdaptationType.CONFIDENCE_THRESHOLDS,
                    target="min_confidence_threshold",
                    action="increase",
                    old_value=0.5,
                    new_value=0.6,
                    confidence=0.8,
                    reasoning=f"Low win rate ({win_rate:.2f}) - increase selectivity",
                    expected_improvement=0.05,
                    applied_at=datetime.now()
                )
                adaptations.append(adaptation)
            
            # Adapt based on average return
            if avg_return < 0:  # Negative average return
                adaptation = AdaptationAction(
                    adaptation_type=AdaptationType.RISK_PARAMETERS,
                    target="position_size_multiplier",
                    action="decrease",
                    old_value=1.0,
                    new_value=0.8,
                    confidence=0.7,
                    reasoning=f"Negative average return ({avg_return:.2%}) - reduce risk",
                    expected_improvement=abs(avg_return) * 0.5,
                    applied_at=datetime.now()
                )
                adaptations.append(adaptation)
            
        except Exception as e:
            self.logger.error(f"Failed to generate overall adaptations: {e}")
        
        return adaptations
    
    def _prioritize_adaptations(self, adaptations: List[AdaptationAction]) -> List[AdaptationAction]:
        """Prioritize adaptations by confidence and expected improvement."""
        try:
            # Sort by confidence and expected improvement
            prioritized = sorted(
                adaptations,
                key=lambda x: (x.confidence * x.expected_improvement),
                reverse=True
            )
            
            # Limit number of adaptations
            max_adaptations = self.learning_models["adaptation"]["max_adaptations_per_day"]
            
            return prioritized[:max_adaptations]
            
        except Exception as e:
            self.logger.error(f"Failed to prioritize adaptations: {e}")
            return adaptations
    
    def apply_adaptation(self, adaptation: AdaptationAction) -> bool:
        """Apply an adaptation action."""
        try:
            # Apply the adaptation
            if adaptation.adaptation_type == AdaptationType.SIGNAL_WEIGHTS:
                self._apply_signal_weight_adaptation(adaptation)
            elif adaptation.adaptation_type == AdaptationType.CONFIDENCE_THRESHOLDS:
                self._apply_confidence_threshold_adaptation(adaptation)
            elif adaptation.adaptation_type == AdaptationType.STRATEGY_PREFERENCES:
                self._apply_strategy_preference_adaptation(adaptation)
            elif adaptation.adaptation_type == AdaptationType.RISK_PARAMETERS:
                self._apply_risk_parameter_adaptation(adaptation)
            elif adaptation.adaptation_type == AdaptationType.TIMING_PREFERENCES:
                self._apply_timing_preference_adaptation(adaptation)
            
            # Record adaptation
            self.adaptation_history.append(adaptation)
            self.current_adaptations[f"{adaptation.adaptation_type.value}_{adaptation.target}"] = adaptation.new_value
            
            # Update metrics
            self.learning_metrics["adaptations_applied"] += 1
            
            self.logger.info(f"Applied adaptation: {adaptation.adaptation_type.value} - {adaptation.target} | "
                          f"Old: {adaptation.old_value} -> New: {adaptation.new_value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply adaptation: {e}")
            return False
    
    def _apply_signal_weight_adaptation(self, adaptation: AdaptationAction) -> None:
        """Apply signal weight adaptation."""
        # Update feature weights
        if adaptation.target in self.learning_models["feature_weights"]:
            self.learning_models["feature_weights"][adaptation.target] = adaptation.new_value
    
    def _apply_confidence_threshold_adaptation(self, adaptation: AdaptationAction) -> None:
        """Apply confidence threshold adaptation."""
        # Update pattern detection thresholds
        if adaptation.target == "min_confidence_threshold":
            self.learning_models["pattern_detection"]["min_confidence"] = adaptation.new_value
    
    def _apply_strategy_preference_adaptation(self, adaptation: AdaptationAction) -> None:
        """Apply strategy preference adaptation."""
        # Store strategy preference
        self.current_adaptations[f"strategy_{adaptation.target}"] = adaptation.new_value
    
    def _apply_risk_parameter_adaptation(self, adaptation: AdaptationAction) -> None:
        """Apply risk parameter adaptation."""
        # Store risk parameter
        self.current_adaptations[f"risk_{adaptation.target}"] = adaptation.new_value
    
    def _apply_timing_preference_adaptation(self, adaptation: AdaptationAction) -> None:
        """Apply timing preference adaptation."""
        # Store timing preference
        self.current_adaptations[f"timing_{adaptation.target}"] = adaptation.new_value
    
    def _extract_market_conditions(self, position: Position, signal: Signal) -> Dict[str, Any]:
        """Extract market conditions from position and signal."""
        return {
            "market_regime": signal.metadata.get("market_regime", "unknown"),
            "volatility": signal.metadata.get("volatility", 0.02),
            "liquidity": signal.metadata.get("liquidity", "medium"),
            "time_of_day": position.entry_time.hour,
            "day_of_week": position.entry_time.weekday()
        }
    
    def _calculate_strategy_distribution(self, trades: List[TradeMemory]) -> Dict[str, float]:
        """Calculate strategy distribution from trades."""
        strategy_counts = defaultdict(int)
        for trade in trades:
            strategy_counts[trade.strategy] += 1
        
        total = len(trades)
        return {strategy: count / total for strategy, count in strategy_counts.items()}
    
    def _update_learning_metrics(self, trade: TradeMemory) -> None:
        """Update learning metrics."""
        self.learning_metrics["total_trades"] += 1
        
        if trade.realized_pnl > 0:
            self.learning_metrics["winning_trades"] += 1
        else:
            self.learning_metrics["losing_trades"] += 1
        
        # Calculate learning progress
        if len(self.trade_memory) > 50:
            recent_trades = list(self.trade_memory)[-50:]
            recent_win_rate = len([t for t in recent_trades if t.realized_pnl > 0]) / len(recent_trades)
            
            older_trades = list(self.trade_memory)[-100:-50] if len(self.trade_memory) > 50 else []
            if older_trades:
                older_win_rate = len([t for t in older_trades if t.realized_pnl > 0]) / len(older_trades)
                self.learning_metrics["learning_progress"] = recent_win_rate - older_win_rate
        
        self.learning_metrics["last_update"] = datetime.now()
    
    def _trigger_pattern_detection(self) -> None:
        """Trigger pattern detection if conditions are met."""
        try:
            # Check if enough trades for pattern detection
            if len(self.trade_memory) >= 20:
                # Check if enough time has passed since last detection
                last_detection = self.learning_metrics.get("last_pattern_detection", datetime.min)
                if datetime.now() - last_detection > timedelta(hours=1):
                    self.detect_patterns()
                    self.learning_metrics["last_pattern_detection"] = datetime.now()
        
        except Exception as e:
            self.logger.error(f"Failed to trigger pattern detection: {e}")
    
    def _trigger_adaptation_check(self) -> None:
        """Trigger adaptation check if conditions are met."""
        try:
            # Check if enough trades for adaptation
            if len(self.trade_memory) >= 10:
                # Check if enough time has passed since last adaptation
                last_adaptation = self.learning_metrics.get("last_adaptation", datetime.min)
                adaptation_frequency = self.learning_models["adaptation"]["adaptation_frequency"]
                
                if datetime.now() - last_adaptation > adaptation_frequency:
                    adaptations = self.generate_adaptations()
                    
                    # Apply high-confidence adaptations
                    for adaptation in adaptations:
                        if adaptation.confidence >= self.learning_models["adaptation"]["min_confidence"]:
                            self.apply_adaptation(adaptation)
                    
                    self.learning_metrics["last_adaptation"] = datetime.now()
        
        except Exception as e:
            self.logger.error(f"Failed to trigger adaptation check: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary."""
        try:
            return {
                "learning_metrics": self.learning_metrics,
                "trade_memory_size": len(self.trade_memory),
                "active_trades": len(self.active_trades),
                "patterns_detected": len(self.detected_patterns),
                "adaptations_applied": len(self.adaptation_history),
                "current_adaptations": self.current_adaptations,
                "recent_patterns": [
                    {
                        "type": p.pattern_type.value,
                        "description": p.description,
                        "confidence": p.confidence,
                        "avg_return": p.avg_return
                    }
                    for p in list(self.detected_patterns.values())[-5:]
                ],
                "recent_adaptations": [
                    {
                        "type": a.adaptation_type.value,
                        "target": a.target,
                        "confidence": a.confidence,
                        "expected_improvement": a.expected_improvement
                    }
                    for a in self.adaptation_history[-5:]
                ]
            }
        
        except Exception as e:
            self.logger.error(f"Failed to get learning summary: {e}")
            return {"error": str(e)}
