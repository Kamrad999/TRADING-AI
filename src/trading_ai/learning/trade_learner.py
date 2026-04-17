"""
Trade Learner following FinRL patterns.
Implements learning from trade data to improve decision making.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import numpy as np
from collections import defaultdict, deque

from ..portfolio.position import Position
from ..strategies.strategy_interface import StrategyOutput
from ..core.models import Signal, SignalDirection
from ..infrastructure.logging import get_logger


@dataclass
class TradeExperience:
    """Trade experience for learning following FinRL patterns."""
    timestamp: datetime
    symbol: str
    action: str
    signal_confidence: float
    market_conditions: Dict[str, Any]
    position_result: Dict[str, Any]
    reward: float
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Learning metrics for tracking improvement."""
    total_experiences: int
    good_trades: int
    bad_trades: int
    win_rate: float
    avg_reward: float
    learning_progress: float
    last_update: datetime


class TradeLearner:
    """
    Trade learner following FinRL patterns.
    
    Key features:
    - Store good/bad trade experiences
    - Analyze patterns in successful trades
    - Adjust strategy weights based on performance
    - Continuous learning from market data
    """
    
    def __init__(self, max_experiences: int = 10000, learning_rate: float = 0.01):
        """Initialize trade learner."""
        self.logger = get_logger("trade_learner")
        
        # Experience storage
        self.max_experiences = max_experiences
        self.experiences: deque[TradeExperience] = deque(maxlen=max_experiences)
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        
        # Strategy weights (adaptive)
        self.strategy_weights = {
            "NewsStrategy": 0.33,
            "TechnicalStrategy": 0.33,
            "HybridStrategy": 0.34
        }
        
        # Market condition weights
        self.market_condition_weights = {
            "bullish": {"NewsStrategy": 0.4, "TechnicalStrategy": 0.3, "HybridStrategy": 0.3},
            "bearish": {"NewsStrategy": 0.3, "TechnicalStrategy": 0.4, "HybridStrategy": 0.3},
            "neutral": {"NewsStrategy": 0.33, "TechnicalStrategy": 0.33, "HybridStrategy": 0.34}
        }
        
        # Performance tracking
        self.metrics = LearningMetrics(
            total_experiences=0,
            good_trades=0,
            bad_trades=0,
            win_rate=0.0,
            avg_reward=0.0,
            learning_progress=0.0,
            last_update=datetime.now()
        )
        
        # Pattern storage
        self.successful_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        
        self.logger.info("TradeLearner initialized with adaptive learning")
    
    def add_trade_experience(self, position: Position, signal: Signal, 
                           market_conditions: Dict[str, Any]) -> None:
        """
        Add a trade experience for learning.
        
        Args:
            position: Closed position with results
            signal: Original trading signal
            market_conditions: Market conditions at trade time
        """
        try:
            # Calculate reward
            reward = self._calculate_reward(position)
            
            # Create experience
            experience = TradeExperience(
                timestamp=position.entry_time,
                symbol=position.symbol,
                action=position.side.value,
                signal_confidence=signal.confidence,
                market_conditions=market_conditions,
                position_result={
                    "pnl": position.realized_pnl,
                    "pnl_pct": position.pnl_percentage,
                    "duration_hours": (position.exit_time - position.entry_time).total_seconds() / 3600 if position.exit_time else 0,
                    "max_drawdown": position.max_drawdown,
                    "entry_price": position.entry_price,
                    "exit_price": position.current_price
                },
                reward=reward,
                strategy=signal.metadata.get("strategy", "Unknown"),
                metadata={
                    "signal_direction": signal.direction.value,
                    "entry_reason": position.entry_reason,
                    "exit_reason": position.exit_reason,
                    "signal_metadata": signal.metadata
                }
            )
            
            # Store experience
            self.experiences.append(experience)
            
            # Update metrics
            self._update_metrics(experience)
            
            # Analyze patterns
            self._analyze_patterns(experience)
            
            # Update weights
            self._update_weights(experience)
            
            self.logger.debug(f"Added trade experience: {position.symbol} {reward:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to add trade experience: {e}")
    
    def _calculate_reward(self, position: Position) -> float:
        """Calculate reward for a trade following FinRL reward shaping."""
        # Base reward from P&L
        pnl_reward = position.realized_pnl / position.entry_value
        
        # Adjust for risk (drawdown penalty)
        drawdown_penalty = -position.max_drawdown * 0.5
        
        # Adjust for trade duration (optimal duration gets bonus)
        duration_hours = (position.exit_time - position.entry_time).total_seconds() / 3600 if position.exit_time else 0
        optimal_duration = 24  # 24 hours optimal
        duration_bonus = -abs(duration_hours - optimal_duration) * 0.01
        
        # Confidence accuracy bonus
        confidence_accuracy = 0.0
        if position.realized_pnl > 0 and position.metadata.get("signal_confidence", 0) > 0.7:
            confidence_accuracy = 0.1
        elif position.realized_pnl < 0 and position.metadata.get("signal_confidence", 0) > 0.7:
            confidence_accuracy = -0.1
        
        # Combined reward
        total_reward = pnl_reward + drawdown_penalty + duration_bonus + confidence_accuracy
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, total_reward))
    
    def _update_metrics(self, experience: TradeExperience) -> None:
        """Update learning metrics."""
        self.metrics.total_experiences += 1
        
        if experience.reward > 0:
            self.metrics.good_trades += 1
        else:
            self.metrics.bad_trades += 1
        
        # Update win rate
        self.metrics.win_rate = self.metrics.good_trades / self.metrics.total_experiences
        
        # Update average reward
        total_reward = sum(exp.reward for exp in self.experiences)
        self.metrics.avg_reward = total_reward / len(self.experiences)
        
        # Update learning progress (improvement over time)
        if len(self.experiences) > 100:
            recent_experiences = list(self.experiences)[-100:]
            recent_avg_reward = sum(exp.reward for exp in recent_experiences) / len(recent_experiences)
            older_experiences = list(self.experiences)[-200:-100]
            older_avg_reward = sum(exp.reward for exp in older_experiences) / len(older_experiences)
            
            self.metrics.learning_progress = recent_avg_reward - older_avg_reward
        
        self.metrics.last_update = datetime.now()
    
    def _analyze_patterns(self, experience: TradeExperience) -> None:
        """Analyze patterns in successful and failed trades."""
        # Extract key features
        features = self._extract_features(experience)
        
        if experience.reward > 0.2:  # Successful trade
            for feature, value in features.items():
                self.successful_patterns[feature].append((value, experience.reward))
        elif experience.reward < -0.2:  # Failed trade
            for feature, value in features.items():
                self.failure_patterns[feature].append((value, experience.reward))
    
    def _extract_features(self, experience: TradeExperience) -> Dict[str, Any]:
        """Extract features from trade experience."""
        features = {}
        
        # Market condition features
        market_regime = experience.market_conditions.get("market_regime", "neutral")
        features["market_regime"] = market_regime
        
        volatility = experience.market_conditions.get("volatility", 0.02)
        features["volatility_bucket"] = "low" if volatility < 0.02 else "medium" if volatility < 0.05 else "high"
        
        # Signal features
        features["signal_confidence_bucket"] = "low" if experience.signal_confidence < 0.5 else "medium" if experience.signal_confidence < 0.8 else "high"
        features["action"] = experience.action
        
        # Time features
        hour = experience.timestamp.hour
        features["time_of_day"] = "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening" if 18 <= hour < 22 else "night"
        
        # Technical features
        rsi = experience.market_conditions.get("rsi", 50)
        features["rsi_bucket"] = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        
        return features
    
    def _update_weights(self, experience: TradeExperience) -> None:
        """Update strategy weights based on experience."""
        if experience.reward > 0.1:  # Good trade
            # Reinforce successful strategy
            strategy = experience.strategy
            if strategy in self.strategy_weights:
                self.strategy_weights[strategy] *= (1 + self.learning_rate * experience.reward)
        elif experience.reward < -0.1:  # Bad trade
            # Penalize unsuccessful strategy
            strategy = experience.strategy
            if strategy in self.strategy_weights:
                self.strategy_weights[strategy] *= (1 - self.learning_rate * abs(experience.reward))
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight
        
        # Update market condition specific weights
        market_regime = experience.market_conditions.get("market_regime", "neutral")
        if market_regime in self.market_condition_weights:
            regime_weights = self.market_condition_weights[market_regime]
            
            if experience.reward > 0.1:
                regime_weights[experience.strategy] *= (1 + self.learning_rate * experience.reward)
            elif experience.reward < -0.1:
                regime_weights[experience.strategy] *= (1 - self.learning_rate * abs(experience.reward))
            
            # Normalize regime weights
            total_regime_weight = sum(regime_weights.values())
            if total_regime_weight > 0:
                for strategy in regime_weights:
                    regime_weights[strategy] /= total_regime_weight
    
    def get_strategy_weights(self, market_regime: str = "neutral") -> Dict[str, float]:
        """Get strategy weights for current market conditions."""
        if market_regime in self.market_condition_weights:
            return self.market_condition_weights[market_regime].copy()
        else:
            return self.strategy_weights.copy()
    
    def get_insights(self) -> Dict[str, Any]:
        """Get learning insights and recommendations."""
        insights = {
            "learning_metrics": {
                "total_experiences": self.metrics.total_experiences,
                "win_rate": self.metrics.win_rate,
                "avg_reward": self.metrics.avg_reward,
                "learning_progress": self.metrics.learning_progress
            },
            "strategy_performance": self.strategy_weights.copy(),
            "successful_patterns": {},
            "failure_patterns": {},
            "recommendations": []
        }
        
        # Analyze successful patterns
        for feature, values in self.successful_patterns.items():
            if len(values) > 10:
                avg_value = np.mean([v[0] for v in values])
                avg_reward = np.mean([v[1] for v in values])
                insights["successful_patterns"][feature] = {
                    "avg_value": avg_value,
                    "avg_reward": avg_reward,
                    "sample_size": len(values)
                }
        
        # Analyze failure patterns
        for feature, values in self.failure_patterns.items():
            if len(values) > 10:
                avg_value = np.mean([v[0] for v in values])
                avg_reward = np.mean([v[1] for v in values])
                insights["failure_patterns"][feature] = {
                    "avg_value": avg_value,
                    "avg_reward": avg_reward,
                    "sample_size": len(values)
                }
        
        # Generate recommendations
        insights["recommendations"] = self._generate_recommendations(insights)
        
        return insights
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on learning insights."""
        recommendations = []
        
        # Strategy recommendations
        best_strategy = max(self.strategy_weights, key=self.strategy_weights.get)
        worst_strategy = min(self.strategy_weights, key=self.strategy_weights.get)
        
        recommendations.append(f"Prefer {best_strategy} (weight: {self.strategy_weights[best_strategy]:.3f})")
        recommendations.append(f"Avoid {worst_strategy} (weight: {self.strategy_weights[worst_strategy]:.3f})")
        
        # Pattern-based recommendations
        successful_patterns = insights["successful_patterns"]
        failure_patterns = insights["failure_patterns"]
        
        # Market regime recommendations
 if "market_regime" in successful_patterns and "market_regime" in failure_patterns:
            success_regime = max(successful_patterns["market_regime"].items(), key=lambda x: x[1]["avg_reward"])
            failure_regime = min(failure_patterns["market_regime"].items(), key=lambda x: x[1]["avg_reward"])
            
            recommendations.append(f"Best performance in {success_regime[0]} regime")
            recommendations.append(f"Worst performance in {failure_regime[0]} regime")
        
        # Confidence recommendations
        if "signal_confidence_bucket" in successful_patterns:
            confidence_bucket = max(successful_patterns["signal_confidence_bucket"].items(), 
                                  key=lambda x: x[1]["avg_reward"])
            recommendations.append(f"Optimal confidence level: {confidence_bucket[0]}")
        
        # Learning progress recommendations
        if self.metrics.learning_progress > 0.05:
            recommendations.append("Learning system showing positive improvement")
        elif self.metrics.learning_progress < -0.05:
            recommendations.append("Learning system showing degradation - consider retraining")
        else:
            recommendations.append("Learning system stable")
        
        return recommendations
    
    def save_learning_state(self, filepath: str) -> None:
        """Save learning state to file."""
        try:
            state = {
                "strategy_weights": self.strategy_weights,
                "market_condition_weights": self.market_condition_weights,
                "metrics": {
                    "total_experiences": self.metrics.total_experiences,
                    "good_trades": self.metrics.good_trades,
                    "bad_trades": self.metrics.bad_trades,
                    "win_rate": self.metrics.win_rate,
                    "avg_reward": self.metrics.avg_reward,
                    "learning_progress": self.metrics.learning_progress
                },
                "successful_patterns": dict(self.successful_patterns),
                "failure_patterns": dict(self.failure_patterns)
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Learning state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save learning state: {e}")
    
    def load_learning_state(self, filepath: str) -> None:
        """Load learning state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.strategy_weights = state.get("strategy_weights", self.strategy_weights)
            self.market_condition_weights = state.get("market_condition_weights", self.market_condition_weights)
            
            metrics_data = state.get("metrics", {})
            self.metrics = LearningMetrics(
                total_experiences=metrics_data.get("total_experiences", 0),
                good_trades=metrics_data.get("good_trades", 0),
                bad_trades=metrics_data.get("bad_trades", 0),
                win_rate=metrics_data.get("win_rate", 0.0),
                avg_reward=metrics_data.get("avg_reward", 0.0),
                learning_progress=metrics_data.get("learning_progress", 0.0),
                last_update=datetime.now()
            )
            
            self.successful_patterns = defaultdict(list, state.get("successful_patterns", {}))
            self.failure_patterns = defaultdict(list, state.get("failure_patterns", {}))
            
            self.logger.info(f"Learning state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load learning state: {e}")
    
    def reset_learning(self) -> None:
        """Reset learning state."""
        self.experiences.clear()
        self.strategy_weights = {
            "NewsStrategy": 0.33,
            "TechnicalStrategy": 0.33,
            "HybridStrategy": 0.34
        }
        self.market_condition_weights = {
            "bullish": {"NewsStrategy": 0.4, "TechnicalStrategy": 0.3, "HybridStrategy": 0.3},
            "bearish": {"NewsStrategy": 0.3, "TechnicalStrategy": 0.4, "HybridStrategy": 0.3},
            "neutral": {"NewsStrategy": 0.33, "TechnicalStrategy": 0.33, "HybridStrategy": 0.34}
        }
        
        self.metrics = LearningMetrics(
            total_experiences=0,
            good_trades=0,
            bad_trades=0,
            win_rate=0.0,
            avg_reward=0.0,
            learning_progress=0.0,
            last_update=datetime.now()
        )
        
        self.successful_patterns.clear()
        self.failure_patterns.clear()
        
        self.logger.info("Learning state reset")
