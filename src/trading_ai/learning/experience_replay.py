"""
Experience Replay system following FinRL patterns.
Efficiently stores and retrieves trade experiences for learning.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import random
import numpy as np
from collections import deque, defaultdict

from .trade_learner import TradeExperience
from ..infrastructure.logging import get_logger


@dataclass
class ReplayBuffer:
    """Experience replay buffer following FinRL patterns."""
    experiences: deque
    max_size: int
    sample_size: int
    priority_weights: Dict[str, float]
    
    def __init__(self, max_size: int = 10000, sample_size: int = 32):
        self.experiences = deque(maxlen=max_size)
        self.max_size = max_size
        self.sample_size = sample_size
        self.priority_weights = {}
        self.logger = get_logger("replay_buffer")
    
    def add(self, experience: TradeExperience) -> None:
        """Add experience to replay buffer."""
        self.experiences.append(experience)
        
        # Update priority weights based on reward magnitude
        if experience.reward > 0.5 or experience.reward < -0.5:
            # High reward experiences get higher priority
            priority = abs(experience.reward)
            self.priority_weights[experience] = priority
    
    def sample(self, strategy: str = None, market_regime: str = None) -> List[TradeExperience]:
        """Sample experiences from replay buffer."""
        if len(self.experiences) < self.sample_size:
            return list(self.experiences)
        
        # Filter experiences if strategy or market regime specified
        filtered_experiences = list(self.experiences)
        
        if strategy:
            filtered_experiences = [exp for exp in filtered_experiences if exp.strategy == strategy]
        
        if market_regime:
            filtered_experiences = [exp for exp in filtered_experiences 
                                  if exp.market_conditions.get("market_regime") == market_regime]
        
        # If not enough filtered experiences, use all
        if len(filtered_experiences) < self.sample_size:
            filtered_experiences = list(self.experiences)
        
        # Weighted sampling based on priority
        if self.priority_weights and len(self.priority_weights) > 0:
            weights = [self.priority_weights.get(exp, 0.1) for exp in filtered_experiences]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                return random.choices(filtered_experiences, weights=weights, k=self.sample_size)
        
        # Random sampling
        return random.sample(filtered_experiences, min(self.sample_size, len(filtered_experiences)))
    
    def get_recent_experiences(self, hours: int = 24) -> List[TradeExperience]:
        """Get recent experiences within specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [exp for exp in self.experiences if exp.timestamp >= cutoff_time]
    
    def get_best_experiences(self, count: int = 10) -> List[TradeExperience]:
        """Get best performing experiences."""
        return sorted(self.experiences, key=lambda x: x.reward, reverse=True)[:count]
    
    def get_worst_experiences(self, count: int = 10) -> List[TradeExperience]:
        """Get worst performing experiences."""
        return sorted(self.experiences, key=lambda x: x.reward)[:count]


class ExperienceReplay:
    """
    Experience Replay system following FinRL patterns.
    
    Key features:
    - Efficient storage and retrieval of trade experiences
    - Prioritized experience replay
    - Stratified sampling by strategy and market conditions
    - Pattern analysis from replay data
    """
    
    def __init__(self, buffer_size: int = 10000, sample_size: int = 32):
        """Initialize experience replay system."""
        self.logger = get_logger("experience_replay")
        
        # Main replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, sample_size)
        
        # Strategy-specific buffers
        self.strategy_buffers = {
            "NewsStrategy": ReplayBuffer(buffer_size // 3, sample_size // 3),
            "TechnicalStrategy": ReplayBuffer(buffer_size // 3, sample_size // 3),
            "HybridStrategy": ReplayBuffer(buffer_size // 3, sample_size // 3)
        }
        
        # Market condition buffers
        self.market_buffers = {
            "bullish": ReplayBuffer(buffer_size // 3, sample_size // 3),
            "bearish": ReplayBuffer(buffer_size // 3, sample_size // 3),
            "neutral": ReplayBuffer(buffer_size // 3, sample_size // 3)
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.replay_stats = {
            "total_replays": 0,
            "successful_replays": 0,
            "avg_improvement": 0.0
        }
        
        self.logger.info(f"ExperienceReplay initialized with buffer size: {buffer_size}")
    
    def add_experience(self, experience: TradeExperience) -> None:
        """Add experience to all relevant buffers."""
        # Add to main buffer
        self.replay_buffer.add(experience)
        
        # Add to strategy-specific buffer
        strategy = experience.strategy
        if strategy in self.strategy_buffers:
            self.strategy_buffers[strategy].add(experience)
        
        # Add to market condition buffer
        market_regime = experience.market_conditions.get("market_regime", "neutral")
        if market_regime in self.market_buffers:
            self.market_buffers[market_regime].add(experience)
    
    def replay_experiences(self, strategy: str = None, market_regime: str = None, 
                          count: int = None) -> List[TradeExperience]:
        """
        Replay experiences for learning.
        
        Args:
            strategy: Filter by strategy
            market_regime: Filter by market regime
            count: Number of experiences to replay
            
        Returns:
            List of experiences for learning
        """
        try:
            # Determine which buffer to use
            if strategy and strategy in self.strategy_buffers:
                buffer = self.strategy_buffers[strategy]
            elif market_regime and market_regime in self.market_buffers:
                buffer = self.market_buffers[market_regime]
            else:
                buffer = self.replay_buffer
            
            # Sample experiences
            experiences = buffer.sample(strategy, market_regime)
            
            # Limit count if specified
            if count and count < len(experiences):
                experiences = experiences[:count]
            
            # Update replay stats
            self.replay_stats["total_replays"] += 1
            
            # Calculate improvement potential
            if experiences:
                avg_reward = np.mean([exp.reward for exp in experiences])
                if avg_reward > 0.1:
                    self.replay_stats["successful_replays"] += 1
                
                # Update performance history
                self.performance_history.append(avg_reward)
                
                # Update average improvement
                if len(self.performance_history) > 0:
                    self.replay_stats["avg_improvement"] = np.mean(self.performance_history)
            
            self.logger.debug(f"Replayed {len(experiences)} experiences (strategy: {strategy}, regime: {market_regime})")
            
            return experiences
            
        except Exception as e:
            self.logger.error(f"Failed to replay experiences: {e}")
            return []
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in replay buffer."""
        patterns = {
            "strategy_performance": {},
            "market_regime_performance": {},
            "reward_distribution": {},
            "temporal_patterns": {},
            "feature_correlations": {}
        }
        
        try:
            # Strategy performance
            for strategy, buffer in self.strategy_buffers.items():
                if len(buffer.experiences) > 0:
                    rewards = [exp.reward for exp in buffer.experiences]
                    patterns["strategy_performance"][strategy] = {
                        "avg_reward": np.mean(rewards),
                        "win_rate": len([r for r in rewards if r > 0]) / len(rewards),
                        "sample_size": len(rewards),
                        "reward_std": np.std(rewards)
                    }
            
            # Market regime performance
            for regime, buffer in self.market_buffers.items():
                if len(buffer.experiences) > 0:
                    rewards = [exp.reward for exp in buffer.experiences]
                    patterns["market_regime_performance"][regime] = {
                        "avg_reward": np.mean(rewards),
                        "win_rate": len([r for r in rewards if r > 0]) / len(rewards),
                        "sample_size": len(rewards),
                        "reward_std": np.std(rewards)
                    }
            
            # Reward distribution
            all_rewards = [exp.reward for exp in self.replay_buffer.experiences]
            if all_rewards:
                patterns["reward_distribution"] = {
                    "mean": np.mean(all_rewards),
                    "std": np.std(all_rewards),
                    "min": np.min(all_rewards),
                    "max": np.max(all_rewards),
                    "quartiles": np.percentile(all_rewards, [25, 50, 75]),
                    "positive_ratio": len([r for r in all_rewards if r > 0]) / len(all_rewards)
                }
            
            # Temporal patterns
            recent_experiences = self.replay_buffer.get_recent_experiences(24)  # Last 24 hours
            if recent_experiences:
                recent_rewards = [exp.reward for exp in recent_experiences]
                patterns["temporal_patterns"]["recent_24h"] = {
                    "avg_reward": np.mean(recent_rewards),
                    "win_rate": len([r for r in recent_rewards if r > 0]) / len(recent_rewards),
                    "sample_size": len(recent_rewards)
                }
            
            # Feature correlations (simplified)
            patterns["feature_correlations"] = self._analyze_feature_correlations()
            
        except Exception as e:
            self.logger.error(f"Failed to analyze patterns: {e}")
        
        return patterns
    
    def _analyze_feature_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between features and rewards."""
        correlations = {}
        
        try:
            experiences = list(self.replay_buffer.experiences)
            if len(experiences) < 50:
                return correlations
            
            # Extract features and rewards
            features = []
            rewards = []
            
            for exp in experiences:
                feature_vector = [
                    exp.signal_confidence,
                    exp.market_conditions.get("volatility", 0.02),
                    exp.market_conditions.get("rsi", 50),
                    exp.position_result.get("duration_hours", 0),
                    1.0 if exp.action == "BUY" else 0.0,  # Action as binary
                    1.0 if exp.strategy == "NewsStrategy" else 0.0,
                    1.0 if exp.strategy == "TechnicalStrategy" else 0.0,
                    1.0 if exp.strategy == "HybridStrategy" else 0.0
                ]
                features.append(feature_vector)
                rewards.append(exp.reward)
            
            # Calculate correlations
            features_array = np.array(features)
            rewards_array = np.array(rewards)
            
            feature_names = [
                "signal_confidence",
                "volatility",
                "rsi",
                "duration_hours",
                "buy_action",
                "news_strategy",
                "technical_strategy",
                "hybrid_strategy"
            ]
            
            for i, feature_name in enumerate(feature_names):
                correlation = np.corrcoef(features_array[:, i], rewards_array)[0, 1]
                correlations[feature_name] = {
                    "correlation": correlation,
                    "strength": abs(correlation),
                    "direction": "positive" if correlation > 0 else "negative"
                }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze feature correlations: {e}")
        
        return correlations
    
    def get_learning_recommendations(self) -> List[str]:
        """Get learning recommendations based on replay analysis."""
        recommendations = []
        
        try:
            patterns = self.analyze_patterns()
            
            # Strategy recommendations
            strategy_perf = patterns.get("strategy_performance", {})
            if strategy_perf:
                best_strategy = max(strategy_perf.items(), key=lambda x: x[1]["avg_reward"])
                worst_strategy = min(strategy_perf.items(), key=lambda x: x[1]["avg_reward"])
                
                recommendations.append(f"Focus on {best_strategy[0]} (avg reward: {best_strategy[1]['avg_reward']:.3f})")
                recommendations.append(f"Reduce {worst_strategy[0]} usage (avg reward: {worst_strategy[1]['avg_reward']:.3f})")
            
            # Market regime recommendations
            regime_perf = patterns.get("market_regime_performance", {})
            if regime_perf:
                best_regime = max(regime_perf.items(), key=lambda x: x[1]["avg_reward"])
                worst_regime = min(regime_perf.items(), key=lambda x: x[1]["avg_reward"])
                
                recommendations.append(f"Best performance in {best_regime[0]} regime")
                recommendations.append(f"Caution in {worst_regime[0]} regime")
            
            # Feature correlation recommendations
            correlations = patterns.get("feature_correlations", {})
            strong_correlations = [(name, data) for name, data in correlations.items() if data["strength"] > 0.3]
            
            if strong_correlations:
                best_feature = max(strong_correlations, key=lambda x: x[1]["strength"])
                recommendations.append(f"Key feature: {best_feature[0]} ({best_feature[1]['direction']} correlation)")
            
            # Performance recommendations
            if self.replay_stats["avg_improvement"] > 0.05:
                recommendations.append("Learning system showing positive improvement")
            elif self.replay_stats["avg_improvement"] < -0.05:
                recommendations.append("Learning system needs adjustment")
            else:
                recommendations.append("Learning system stable")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to analyze patterns for recommendations")
        
        return recommendations
    
    def get_replay_statistics(self) -> Dict[str, Any]:
        """Get comprehensive replay statistics."""
        stats = {
            "buffer_sizes": {
                "total": len(self.replay_buffer.experiences),
                "news_strategy": len(self.strategy_buffers["NewsStrategy"].experiences),
                "technical_strategy": len(self.strategy_buffers["TechnicalStrategy"].experiences),
                "hybrid_strategy": len(self.strategy_buffers["HybridStrategy"].experiences),
                "bullish": len(self.market_buffers["bullish"].experiences),
                "bearish": len(self.market_buffers["bearish"].experiences),
                "neutral": len(self.market_buffers["neutral"].experiences)
            },
            "replay_stats": self.replay_stats.copy(),
            "performance_trend": list(self.performance_history)[-20:] if self.performance_history else []
        }
        
        return stats
    
    def clear_buffers(self) -> None:
        """Clear all replay buffers."""
        self.replay_buffer.experiences.clear()
        self.strategy_buffers["NewsStrategy"].experiences.clear()
        self.strategy_buffers["TechnicalStrategy"].experiences.clear()
        self.strategy_buffers["HybridStrategy"].experiences.clear()
        self.market_buffers["bullish"].experiences.clear()
        self.market_buffers["bearish"].experiences.clear()
        self.market_buffers["neutral"].experiences.clear()
        
        self.performance_history.clear()
        self.replay_stats = {
            "total_replays": 0,
            "successful_replays": 0,
            "avg_improvement": 0.0
        }
        
        self.logger.info("All replay buffers cleared")
    
    def save_replay_state(self, filepath: str) -> None:
        """Save replay state to file."""
        try:
            import pickle
            
            state = {
                "replay_buffer": list(self.replay_buffer.experiences),
                "strategy_buffers": {
                    name: list(buffer.experiences) 
                    for name, buffer in self.strategy_buffers.items()
                },
                "market_buffers": {
                    name: list(buffer.experiences) 
                    for name, buffer in self.market_buffers.items()
                },
                "replay_stats": self.replay_stats,
                "performance_history": list(self.performance_history)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"Replay state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save replay state: {e}")
    
    def load_replay_state(self, filepath: str) -> None:
        """Load replay state from file."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore buffers
            self.replay_buffer.experiences = deque(state["replay_buffer"], maxlen=self.replay_buffer.max_size)
            
            for name, experiences in state["strategy_buffers"].items():
                if name in self.strategy_buffers:
                    self.strategy_buffers[name].experiences = deque(experiences, maxlen=self.strategy_buffers[name].max_size)
            
            for name, experiences in state["market_buffers"].items():
                if name in self.market_buffers:
                    self.market_buffers[name].experiences = deque(experiences, maxlen=self.market_buffers[name].max_size)
            
            # Restore stats
            self.replay_stats = state["replay_stats"]
            self.performance_history = deque(state["performance_history"], maxlen=1000)
            
            self.logger.info(f"Replay state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load replay state: {e}")
