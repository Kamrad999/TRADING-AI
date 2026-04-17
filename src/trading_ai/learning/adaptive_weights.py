"""
Adaptive Weights system following FinRL patterns.
Dynamically adjusts strategy and agent weights based on performance.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque

from ..infrastructure.logging import get_logger


class AdaptiveWeights:
    """
    Adaptive weights system following FinRL patterns.
    
    Key features:
    - Dynamic weight adjustment based on performance
    - Market regime-specific optimization
    - Decay and regularization to prevent overfitting
    - Continuous learning and adaptation
    """
    
    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.001):
        """Initialize adaptive weights system."""
        self.logger = get_logger("adaptive_weights")
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.min_weight = 0.1
        self.max_weight = 0.6
        
        # Strategy weights
        self.strategy_weights = {
            "NewsStrategy": 0.33,
            "TechnicalStrategy": 0.33,
            "HybridStrategy": 0.34
        }
        
        # Agent weights (for multi-agent system)
        self.agent_weights = {
            "NewsAgent": 0.34,
            "TechnicalAgent": 0.33,
            "RiskAgent": 0.33
        }
        
        # Market regime-specific weights
        self.regime_weights = {
            "bullish": {
                "strategies": {"NewsStrategy": 0.4, "TechnicalStrategy": 0.3, "HybridStrategy": 0.3},
                "agents": {"NewsAgent": 0.4, "TechnicalAgent": 0.3, "RiskAgent": 0.3}
            },
            "bearish": {
                "strategies": {"NewsStrategy": 0.3, "TechnicalStrategy": 0.4, "HybridStrategy": 0.3},
                "agents": {"NewsAgent": 0.3, "TechnicalAgent": 0.4, "RiskAgent": 0.3}
            },
            "neutral": {
                "strategies": {"NewsStrategy": 0.33, "TechnicalStrategy": 0.33, "HybridStrategy": 0.34},
                "agents": {"NewsAgent": 0.34, "TechnicalAgent": 0.33, "RiskAgent": 0.33}
            }
        }
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.weight_history = defaultdict(lambda: deque(maxlen=50))
        self.adaptation_stats = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "avg_improvement": 0.0
        }
        
        self.logger.info("AdaptiveWeights initialized with dynamic learning")
    
    def update_strategy_weights(self, performance_data: Dict[str, float], 
                               market_regime: str = "neutral") -> None:
        """
        Update strategy weights based on performance data.
        
        Args:
            performance_data: Strategy performance metrics
            market_regime: Current market regime
        """
        try:
            # Get current regime weights
            if market_regime in self.regime_weights:
                current_weights = self.regime_weights[market_regime]["strategies"].copy()
            else:
                current_weights = self.strategy_weights.copy()
            
            # Calculate weight adjustments
            adjustments = {}
            for strategy, performance in performance_data.items():
                if strategy in current_weights:
                    # Performance-based adjustment
                    adjustment = self.learning_rate * performance
                    
                    # Apply regularization to prevent extreme weights
                    current_weight = current_weights[strategy]
                    if current_weight > 0.5:  # Penalize high weights
                        adjustment -= self.decay_rate
                    elif current_weight < 0.2:  # Boost low weights
                        adjustment += self.decay_rate
                    
                    adjustments[strategy] = adjustment
            
            # Apply adjustments
            new_weights = current_weights.copy()
            for strategy, adjustment in adjustments.items():
                new_weights[strategy] += adjustment
                # Clamp to valid range
                new_weights[strategy] = max(self.min_weight, min(self.max_weight, new_weights[strategy]))
            
            # Normalize weights
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                for strategy in new_weights:
                    new_weights[strategy] /= total_weight
            
            # Update weights
            if market_regime in self.regime_weights:
                self.regime_weights[market_regime]["strategies"] = new_weights
            else:
                self.strategy_weights = new_weights
            
            # Track performance
            self._track_performance("strategy", performance_data)
            self._track_weights("strategy", new_weights)
            
            # Update adaptation stats
            self._update_adaptation_stats(performance_data)
            
            self.logger.debug(f"Updated strategy weights for {market_regime}: {new_weights}")
            
        except Exception as e:
            self.logger.error(f"Failed to update strategy weights: {e}")
    
    def update_agent_weights(self, performance_data: Dict[str, float], 
                            market_regime: str = "neutral") -> None:
        """
        Update agent weights based on performance data.
        
        Args:
            performance_data: Agent performance metrics
            market_regime: Current market regime
        """
        try:
            # Get current regime weights
            if market_regime in self.regime_weights:
                current_weights = self.regime_weights[market_regime]["agents"].copy()
            else:
                current_weights = self.agent_weights.copy()
            
            # Calculate weight adjustments
            adjustments = {}
            for agent, performance in performance_data.items():
                if agent in current_weights:
                    # Performance-based adjustment
                    adjustment = self.learning_rate * performance
                    
                    # Apply regularization
                    current_weight = current_weights[agent]
                    if current_weight > 0.5:
                        adjustment -= self.decay_rate
                    elif current_weight < 0.2:
                        adjustment += self.decay_rate
                    
                    adjustments[agent] = adjustment
            
            # Apply adjustments
            new_weights = current_weights.copy()
            for agent, adjustment in adjustments.items():
                new_weights[agent] += adjustment
                new_weights[agent] = max(self.min_weight, min(self.max_weight, new_weights[agent]))
            
            # Normalize weights
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                for agent in new_weights:
                    new_weights[agent] /= total_weight
            
            # Update weights
            if market_regime in self.regime_weights:
                self.regime_weights[market_regime]["agents"] = new_weights
            else:
                self.agent_weights = new_weights
            
            # Track performance
            self._track_performance("agent", performance_data)
            self._track_weights("agent", new_weights)
            
            self.logger.debug(f"Updated agent weights for {market_regime}: {new_weights}")
            
        except Exception as e:
            self.logger.error(f"Failed to update agent weights: {e}")
    
    def get_strategy_weights(self, market_regime: str = "neutral") -> Dict[str, float]:
        """Get strategy weights for current market regime."""
        if market_regime in self.regime_weights:
            return self.regime_weights[market_regime]["strategies"].copy()
        else:
            return self.strategy_weights.copy()
    
    def get_agent_weights(self, market_regime: str = "neutral") -> Dict[str, float]:
        """Get agent weights for current market regime."""
        if market_regime in self.regime_weights:
            return self.regime_weights[market_regime]["agents"].copy()
        else:
            return self.agent_weights.copy()
    
    def _track_performance(self, weight_type: str, performance_data: Dict[str, float]) -> None:
        """Track performance history for analysis."""
        timestamp = datetime.now()
        
        for name, performance in performance_data.items():
            key = f"{weight_type}_{name}"
            self.performance_history[key].append((timestamp, performance))
    
    def _track_weights(self, weight_type: str, weights: Dict[str, float]) -> None:
        """Track weight history for analysis."""
        timestamp = datetime.now()
        
        for name, weight in weights.items():
            key = f"{weight_type}_{name}"
            self.weight_history[key].append((timestamp, weight))
    
    def _update_adaptation_stats(self, performance_data: Dict[str, float]) -> None:
        """Update adaptation statistics."""
        self.adaptation_stats["total_adaptations"] += 1
        
        # Calculate improvement
        if performance_data:
            avg_performance = np.mean(list(performance_data.values()))
            if avg_performance > 0.1:
                self.adaptation_stats["successful_adaptations"] += 1
            
            # Update average improvement
            recent_performances = [avg_performance] + list(self.performance_history.get("recent", []))
            self.adaptation_stats["avg_improvement"] = np.mean(recent_performances[:10])
    
    def analyze_weight_stability(self) -> Dict[str, Any]:
        """Analyze weight stability and trends."""
        stability_analysis = {
            "strategy_stability": {},
            "agent_stability": {},
            "regime_stability": {},
            "adaptation_trends": {}
        }
        
        try:
            # Strategy weight stability
            for strategy in self.strategy_weights:
                key = f"strategy_{strategy}"
                if key in self.weight_history and len(self.weight_history[key]) > 10:
                    weights = [w for _, w in self.weight_history[key]]
                    stability_analysis["strategy_stability"][strategy] = {
                        "current_weight": weights[-1],
                        "weight_std": np.std(weights),
                        "weight_range": (min(weights), max(weights)),
                        "trend": "increasing" if weights[-1] > weights[-10] else "decreasing" if weights[-1] < weights[-10] else "stable",
                        "stability_score": 1.0 / (1.0 + np.std(weights))  # Higher is more stable
                    }
            
            # Agent weight stability
            for agent in self.agent_weights:
                key = f"agent_{agent}"
                if key in self.weight_history and len(self.weight_history[key]) > 10:
                    weights = [w for _, w in self.weight_history[key]]
                    stability_analysis["agent_stability"][agent] = {
                        "current_weight": weights[-1],
                        "weight_std": np.std(weights),
                        "weight_range": (min(weights), max(weights)),
                        "trend": "increasing" if weights[-1] > weights[-10] else "decreasing" if weights[-1] < weights[-10] else "stable",
                        "stability_score": 1.0 / (1.0 + np.std(weights))
                    }
            
            # Regime stability
            for regime in self.regime_weights:
                regime_data = self.regime_weights[regime]
                strategy_weights = regime_data["strategies"]
                agent_weights = regime_data["agents"]
                
                stability_analysis["regime_stability"][regime] = {
                    "strategy_entropy": self._calculate_entropy(strategy_weights),
                    "agent_entropy": self._calculate_entropy(agent_weights),
                    "balance_score": self._calculate_balance_score(strategy_weights, agent_weights)
                }
            
            # Adaptation trends
            stability_analysis["adaptation_trends"] = {
                "total_adaptations": self.adaptation_stats["total_adaptations"],
                "success_rate": self.adaptation_stats["successful_adaptations"] / max(1, self.adaptation_stats["total_adaptations"]),
                "avg_improvement": self.adaptation_stats["avg_improvement"],
                "learning_velocity": self._calculate_learning_velocity()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze weight stability: {e}")
        
        return stability_analysis
    
    def _calculate_entropy(self, weights: Dict[str, float]) -> float:
        """Calculate entropy of weight distribution."""
        if not weights:
            return 0.0
        
        entropy = 0.0
        for weight in weights.values():
            if weight > 0:
                entropy -= weight * np.log2(weight)
        
        return entropy
    
    def _calculate_balance_score(self, strategy_weights: Dict[str, float], 
                                agent_weights: Dict[str, float]) -> float:
        """Calculate balance score between strategy and agent weights."""
        # Balance score measures how evenly distributed weights are
        strategy_balance = 1.0 - np.std(list(strategy_weights.values()))
        agent_balance = 1.0 - np.std(list(agent_weights.values()))
        
        return (strategy_balance + agent_balance) / 2.0
    
    def _calculate_learning_velocity(self) -> float:
        """Calculate learning velocity (rate of weight changes)."""
        velocity_scores = []
        
        for key, history in self.weight_history.items():
            if len(history) > 5:
                weights = [w for _, w in history]
                # Calculate rate of change
                recent_changes = np.diff(weights[-5:])
                velocity = np.mean(np.abs(recent_changes))
                velocity_scores.append(velocity)
        
        return np.mean(velocity_scores) if velocity_scores else 0.0
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on weight analysis."""
        recommendations = []
        
        try:
            stability = self.analyze_weight_stability()
            
            # Strategy recommendations
            strategy_stability = stability.get("strategy_stability", {})
            unstable_strategies = [(name, data) for name, data in strategy_stability.items() 
                                  if data["stability_score"] < 0.5]
            
            if unstable_strategies:
                recommendations.append(f"Unstable strategy weights: {[name for name, _ in unstable_strategies]}")
            
            # Agent recommendations
            agent_stability = stability.get("agent_stability", {})
            unstable_agents = [(name, data) for name, data in agent_stability.items() 
                             if data["stability_score"] < 0.5]
            
            if unstable_agents:
                recommendations.append(f"Unstable agent weights: {[name for name, _ in unstable_agents]}")
            
            # Adaptation recommendations
            adaptation_trends = stability.get("adaptation_trends", {})
            success_rate = adaptation_trends.get("success_rate", 0.0)
            
            if success_rate < 0.5:
                recommendations.append("Low adaptation success rate - consider adjusting learning rate")
            elif success_rate > 0.8:
                recommendations.append("High adaptation success rate - learning system effective")
            
            # Balance recommendations
            regime_stability = stability.get("regime_stability", {})
            for regime, data in regime_stability.items():
                if data["balance_score"] < 0.5:
                    recommendations.append(f"Imbalanced weights in {regime} regime")
            
            # Learning velocity recommendations
            learning_velocity = adaptation_trends.get("learning_velocity", 0.0)
            if learning_velocity > 0.1:
                recommendations.append("High learning velocity - weights changing rapidly")
            elif learning_velocity < 0.01:
                recommendations.append("Low learning velocity - system may be stagnating")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to analyze weights for recommendations")
        
        return recommendations
    
    def reset_weights(self) -> None:
        """Reset all weights to default values."""
        self.strategy_weights = {
            "NewsStrategy": 0.33,
            "TechnicalStrategy": 0.33,
            "HybridStrategy": 0.34
        }
        
        self.agent_weights = {
            "NewsAgent": 0.34,
            "TechnicalAgent": 0.33,
            "RiskAgent": 0.33
        }
        
        self.regime_weights = {
            "bullish": {
                "strategies": {"NewsStrategy": 0.4, "TechnicalStrategy": 0.3, "HybridStrategy": 0.3},
                "agents": {"NewsAgent": 0.4, "TechnicalAgent": 0.3, "RiskAgent": 0.3}
            },
            "bearish": {
                "strategies": {"NewsStrategy": 0.3, "TechnicalStrategy": 0.4, "HybridStrategy": 0.3},
                "agents": {"NewsAgent": 0.3, "TechnicalAgent": 0.4, "RiskAgent": 0.3}
            },
            "neutral": {
                "strategies": {"NewsStrategy": 0.33, "TechnicalStrategy": 0.33, "HybridStrategy": 0.34},
                "agents": {"NewsAgent": 0.34, "TechnicalAgent": 0.33, "RiskAgent": 0.33}
            }
        }
        
        # Clear history
        self.performance_history.clear()
        self.weight_history.clear()
        
        # Reset stats
        self.adaptation_stats = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "avg_improvement": 0.0
        }
        
        self.logger.info("All weights reset to default values")
    
    def save_weights(self, filepath: str) -> None:
        """Save current weights to file."""
        try:
            import json
            
            state = {
                "strategy_weights": self.strategy_weights,
                "agent_weights": self.agent_weights,
                "regime_weights": self.regime_weights,
                "adaptation_stats": self.adaptation_stats,
                "learning_rate": self.learning_rate,
                "decay_rate": self.decay_rate
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Adaptive weights saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save weights: {e}")
    
    def load_weights(self, filepath: str) -> None:
        """Load weights from file."""
        try:
            import json
            
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.strategy_weights = state.get("strategy_weights", self.strategy_weights)
            self.agent_weights = state.get("agent_weights", self.agent_weights)
            self.regime_weights = state.get("regime_weights", self.regime_weights)
            self.adaptation_stats = state.get("adaptation_stats", self.adaptation_stats)
            self.learning_rate = state.get("learning_rate", self.learning_rate)
            self.decay_rate = state.get("decay_rate", self.decay_rate)
            
            self.logger.info(f"Adaptive weights loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load weights: {e}")
    
    def get_weight_summary(self) -> Dict[str, Any]:
        """Get comprehensive weight summary."""
        summary = {
            "current_strategy_weights": self.strategy_weights,
            "current_agent_weights": self.agent_weights,
            "regime_specific_weights": self.regime_weights,
            "adaptation_statistics": self.adaptation_stats,
            "learning_parameters": {
                "learning_rate": self.learning_rate,
                "decay_rate": self.decay_rate,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight
            }
        }
        
        return summary
