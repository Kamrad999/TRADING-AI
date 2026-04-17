"""
Signal scorer for weighted signal evaluation.
Following patterns from AgentQuant and VectorBT repositories.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import math

from ..core.models import Signal
from ..infrastructure.logging import get_logger


class SignalScorer:
    """
    Signal scorer for weighted signal evaluation.
    
    Following patterns from:
    - AgentQuant: Factor-based scoring system
    - VectorBT: Performance optimization
    - ai-hedge-fund-crypto: Multi-factor consensus scoring
    """
    
    def __init__(self):
        """Initialize signal scorer."""
        self.logger = get_logger("signal_scorer")
        
        # Scoring weights
        self.scoring_weights = {
            "consensus_confidence": 0.3,
            "news_sentiment": 0.25,
            "technical_strength": 0.25,
            "risk_adjustment": 0.15,
            "market_regime": 0.05
        }
        
        # Signal quality thresholds
        self.min_quality_score = 0.5
        self.max_signals_per_batch = 20
        
        self.logger.info("Signal scorer initialized")
    
    def score_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Score and rank signals.
        
        Args:
            signals: List of signals to score
            
        Returns:
            Scored and ranked signals
        """
        try:
            scored_signals = []
            
            for signal in signals:
                score = self._calculate_signal_score(signal)
                if score >= self.min_quality_score:
                    # Add score to metadata
                    signal.metadata["quality_score"] = score
                    scored_signals.append(signal)
            
            # Sort by score
            scored_signals.sort(key=lambda s: s.metadata["quality_score"], reverse=True)
            
            # Limit signals
            if len(scored_signals) > self.max_signals_per_batch:
                scored_signals = scored_signals[:self.max_signals_per_batch]
            
            self.logger.info(f"Scored {len(scored_signals)} signals (quality threshold: {self.min_quality_score})")
            
            return scored_signals
            
        except Exception as e:
            self.logger.error(f"Signal scoring failed: {e}")
            return signals
    
    def _calculate_signal_score(self, signal: Signal) -> float:
        """Calculate comprehensive signal score."""
        try:
            factors = self._extract_signal_factors(signal)
            
            # Calculate weighted score
            score = 0.0
            
            # Consensus confidence
            consensus_confidence = factors.get("consensus_confidence", 0.0)
            score += consensus_confidence * self.scoring_weights["consensus_confidence"]
            
            # News sentiment
            news_sentiment = factors.get("news_sentiment", 0.0)
            score += news_sentiment * self.scoring_weights["news_sentiment"]
            
            # Technical strength
            technical_strength = factors.get("technical_strength", 0.0)
            score += technical_strength * self.scoring_weights["technical_strength"]
            
            # Risk adjustment
            risk_adjustment = factors.get("risk_adjustment", 0.0)
            score += risk_adjustment * self.scoring_weights["risk_adjustment"]
            
            # Market regime
            market_regime = factors.get("market_regime", 0.0)
            score += market_regime * self.scoring_weights["market_regime"]
            
            # Apply quality multipliers
            score = self._apply_quality_multipliers(score, signal, factors)
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating signal score: {e}")
            return 0.0
    
    def _extract_signal_factors(self, signal: Signal) -> Dict[str, float]:
        """Extract factors from signal metadata."""
        factors = {}
        
        # Get signal factors from metadata
        signal_factors = signal.metadata.get("signal_factors", {})
        
        # Extract individual factors
        factors["consensus_confidence"] = signal.confidence
        factors["news_sentiment"] = signal_factors.get("news_sentiment", 0.0)
        factors["technical_strength"] = signal_factors.get("technical_strength", 0.0)
        factors["risk_adjustment"] = signal_factors.get("risk_adjustment", 0.0)
        factors["market_regime"] = signal_factors.get("market_regime", 0.0)
        factors["volatility_adjustment"] = signal_factors.get("volatility_adjustment", 1.0)
        
        # Extract agent decisions
        agent_decisions = signal.metadata.get("agent_decisions", [])
        if agent_decisions:
            # Calculate agent agreement
            actions = [d.get("action", "HOLD") for d in agent_decisions]
            action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
            for action in actions:
                action_counts[action] += 1
            
            # Agent agreement bonus
            max_count = max(action_counts.values())
            total_count = len(agent_decisions)
            agreement_ratio = max_count / total_count if total_count > 0 else 0.0
            factors["agent_agreement"] = agreement_ratio
        else:
            factors["agent_agreement"] = 0.0
        
        return factors
    
    def _apply_quality_multipliers(self, base_score: float, signal: Signal, factors: Dict[str, float]) -> float:
        """Apply quality multipliers to base score."""
        score = base_score
        
        # Volatility adjustment
        volatility_adjustment = factors.get("volatility_adjustment", 1.0)
        score *= volatility_adjustment
        
        # Agent agreement bonus
        agent_agreement = factors.get("agent_agreement", 0.0)
        if agent_agreement > 0.8:  # High agreement
            score *= 1.1
        elif agent_agreement < 0.4:  # Low agreement
            score *= 0.9
        
        # Signal direction bonus (prefer BUY in bull markets, SELL in bear markets)
        market_context = signal.metadata.get("market_context", {})
        market_trend = market_context.get("trend", "neutral")
        
        if market_trend == "bullish" and signal.direction.value == "BUY":
            score *= 1.05
        elif market_trend == "bearish" and signal.direction.value == "SELL":
            score *= 1.05
        
        # Urgency bonus
        if signal.urgency.value == "HIGH":
            score *= 1.02
        elif signal.urgency.value == "LOW":
            score *= 0.98
        
        return score
    
    def calculate_signal_metrics(self, signals: List[Signal]) -> Dict[str, Any]:
        """Calculate signal metrics for batch."""
        if not signals:
            return {
                "total_signals": 0,
                "avg_confidence": 0.0,
                "avg_quality_score": 0.0,
                "signal_distribution": {"BUY": 0, "SELL": 0, "HOLD": 0},
                "urgency_distribution": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            }
        
        # Calculate basic metrics
        total_signals = len(signals)
        avg_confidence = sum(s.confidence for s in signals) / total_signals
        
        # Calculate quality scores
        quality_scores = [s.metadata.get("quality_score", 0.0) for s in signals]
        avg_quality_score = sum(quality_scores) / len(quality_scores)
        
        # Calculate distributions
        signal_distribution = {"BUY": 0, "SELL": 0, "HOLD": 0}
        urgency_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for signal in signals:
            signal_distribution[signal.direction.value] += 1
            urgency_distribution[signal.urgency.value] += 1
        
        # Calculate top symbols
        symbol_counts = {}
        for signal in signals:
            symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1
        
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_signals": total_signals,
            "avg_confidence": avg_confidence,
            "avg_quality_score": avg_quality_score,
            "signal_distribution": signal_distribution,
            "urgency_distribution": urgency_distribution,
            "top_symbols": top_symbols,
            "highest_quality": max(quality_scores) if quality_scores else 0.0,
            "lowest_quality": min(quality_scores) if quality_scores else 0.0
        }
    
    def filter_signals_by_criteria(self, signals: List[Signal], 
                                  min_confidence: float = 0.0,
                                  min_quality: float = 0.0,
                                  symbols: Optional[List[str]] = None,
                                  directions: Optional[List[str]] = None) -> List[Signal]:
        """Filter signals by criteria."""
        filtered_signals = []
        
        for signal in signals:
            # Confidence filter
            if signal.confidence < min_confidence:
                continue
            
            # Quality filter
            quality_score = signal.metadata.get("quality_score", 0.0)
            if quality_score < min_quality:
                continue
            
            # Symbol filter
            if symbols and signal.symbol not in symbols:
                continue
            
            # Direction filter
            if directions and signal.direction.value not in directions:
                continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def rank_signals_by_factor(self, signals: List[Signal], factor: str) -> List[Signal]:
        """Rank signals by specific factor."""
        if factor == "confidence":
            return sorted(signals, key=lambda s: s.confidence, reverse=True)
        elif factor == "quality":
            return sorted(signals, key=lambda s: s.metadata.get("quality_score", 0.0), reverse=True)
        elif factor == "urgency":
            urgency_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            return sorted(signals, key=lambda s: urgency_order.get(s.urgency.value, 1), reverse=True)
        elif factor == "position_size":
            return sorted(signals, key=lambda s: s.position_size, reverse=True)
        else:
            return signals
    
    def calculate_signal_correlation(self, signals: List[Signal]) -> Dict[str, float]:
        """Calculate correlation between signals."""
        if len(signals) < 2:
            return {"correlation": 0.0}
        
        # Group signals by symbol
        symbol_signals = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        # Calculate correlation based on directions and confidence
        correlations = []
        
        for symbol, symbol_signal_list in symbol_signals.items():
            if len(symbol_signal_list) > 1:
                # Calculate direction consistency
                directions = [1 if s.direction.value == "BUY" else -1 if s.direction.value == "SELL" else 0 
                              for s in symbol_signal_list]
                confidences = [s.confidence for s in symbol_signal_list]
                
                # Weighted correlation
                weighted_directions = [d * c for d, c in zip(directions, confidences)]
                avg_direction = sum(weighted_directions) / sum(confidences)
                
                correlations.append(abs(avg_direction))
        
        if correlations:
            avg_correlation = sum(correlations) / len(correlations)
            return {
                "correlation": avg_correlation,
                "signal_count": len(signals),
                "symbol_count": len(symbol_signals)
            }
        
        return {"correlation": 0.0}
    
    def get_signal_recommendations(self, signals: List[Signal], max_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get signal recommendations with reasoning."""
        recommendations = []
        
        # Sort by quality score
        scored_signals = sorted(signals, 
                             key=lambda s: s.metadata.get("quality_score", 0.0), 
                             reverse=True)
        
        for signal in scored_signals[:max_recommendations]:
            recommendation = {
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "confidence": signal.confidence,
                "quality_score": signal.metadata.get("quality_score", 0.0),
                "urgency": signal.urgency.value,
                "position_size": signal.position_size,
                "reasoning": signal.metadata.get("reasoning", ""),
                "market_context": signal.metadata.get("market_context", {}),
                "agent_decisions": signal.metadata.get("agent_decisions", []),
                "recommendation": self._generate_recommendation_text(signal)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_recommendation_text(self, signal: Signal) -> str:
        """Generate recommendation text for signal."""
        quality_score = signal.metadata.get("quality_score", 0.0)
        
        if quality_score > 0.8:
            strength = "Strong"
        elif quality_score > 0.6:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        action = signal.direction.value
        symbol = signal.symbol
        
        return f"{strength} {action} recommendation for {symbol} (Quality: {quality_score:.2f})"
