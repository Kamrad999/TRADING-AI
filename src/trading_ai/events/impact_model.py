"""
Market Impact Model following Qlib and institutional trading patterns.
Predicts market impact and price reactions to classified events.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from .event_classifier import EventClassification, EventType, ImpactLevel, TimeHorizon
from ..infrastructure.logging import get_logger
from ..core.models import MarketRegime


class ImpactDirection(Enum):
    """Direction of market impact."""
    POSITIVE = "positive"    # Price increase expected
    NEGATIVE = "negative"    # Price decrease expected
    NEUTRAL = "neutral"      # No clear direction
    VOLATILE = "volatile"    # Increased volatility expected


class ImpactDuration(Enum):
    """Duration of impact."""
    FLASH = "flash"          # Minutes
    INTRADAY = "intraday"    # Hours
    SWING = "swing"          # Days
    POSITIONAL = "positional"  # Weeks+


@dataclass
class MarketImpact:
    """Predicted market impact from event."""
    direction: ImpactDirection
    magnitude: float  # Expected price move percentage
    duration: ImpactDuration
    volatility_increase: float  # Volatility multiplier
    volume_increase: float     # Volume multiplier
    confidence: float
    affected_assets: List[str]
    time_to_impact: int  # Minutes until impact materializes
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactPrediction:
    """Complete impact prediction for an event."""
    event_classification: EventClassification
    primary_impact: MarketImpact
    secondary_impacts: List[MarketImpact]
    cross_asset_effects: Dict[str, MarketImpact]
    risk_factors: List[str]
    trading_opportunities: List[Dict[str, Any]]
    confidence_score: float
    prediction_timestamp: datetime


class ImpactModel:
    """
    Market impact model following Qlib and institutional trading patterns.
    
    Key features:
    - Multi-dimensional impact prediction
    - Cross-asset effect modeling
    - Volatility and volume forecasting
    - Time-based impact decay
    - Risk factor identification
    """
    
    def __init__(self):
        """Initialize impact model."""
        self.logger = get_logger("impact_model")
        
        # Impact models by event type
        self._initialize_impact_models()
        
        # Historical impact database
        self.impact_history: List[Dict[str, Any]] = []
        
        # Cross-asset correlations
        self._initialize_cross_asset_correlations()
        
        # Volatility models
        self._initialize_volatility_models()
        
        # Performance tracking
        self.prediction_stats = {
            "total_predictions": 0,
            "accuracy_by_type": defaultdict(lambda: {"correct": 0, "total": 0}),
            "direction_accuracy": defaultdict(lambda: {"correct": 0, "total": 0}),
            "magnitude_error": []
        }
        
        self.logger.info("ImpactModel initialized with institutional-grade prediction models")
    
    def _initialize_impact_models(self) -> None:
        """Initialize impact models for different event types."""
        self.impact_models = {
            EventType.MACRO_ECONOMIC: {
                ImpactDirection.POSITIVE: {
                    ImpactLevel.CRITICAL: {"magnitude": 0.08, "duration": ImpactDuration.SWING, "volatility": 2.5},
                    ImpactLevel.HIGH: {"magnitude": 0.04, "duration": ImpactDuration.INTRADAY, "volatility": 2.0},
                    ImpactLevel.MEDIUM: {"magnitude": 0.02, "duration": ImpactDuration.INTRADAY, "volatility": 1.5},
                    ImpactLevel.LOW: {"magnitude": 0.01, "duration": ImpactDuration.FLASH, "volatility": 1.2}
                },
                ImpactDirection.NEGATIVE: {
                    ImpactLevel.CRITICAL: {"magnitude": -0.10, "duration": ImpactDuration.SWING, "volatility": 3.0},
                    ImpactLevel.HIGH: {"magnitude": -0.05, "duration": ImpactDuration.INTRADAY, "volatility": 2.2},
                    ImpactLevel.MEDIUM: {"magnitude": -0.025, "duration": ImpactDuration.INTRADAY, "volatility": 1.6},
                    ImpactLevel.LOW: {"magnitude": -0.012, "duration": ImpactDuration.FLASH, "volatility": 1.3}
                }
            },
            
            EventType.CRYPTO_SPECIFIC: {
                ImpactDirection.POSITIVE: {
                    ImpactLevel.CRITICAL: {"magnitude": 0.15, "duration": ImpactDuration.SWING, "volatility": 3.5},
                    ImpactLevel.HIGH: {"magnitude": 0.08, "duration": ImpactDuration.INTRADAY, "volatility": 2.8},
                    ImpactLevel.MEDIUM: {"magnitude": 0.04, "duration": ImpactDuration.INTRADAY, "volatility": 2.0},
                    ImpactLevel.LOW: {"magnitude": 0.02, "duration": ImpactDuration.FLASH, "volatility": 1.5}
                },
                ImpactDirection.NEGATIVE: {
                    ImpactLevel.CRITICAL: {"magnitude": -0.20, "duration": ImpactDuration.SWING, "volatility": 4.0},
                    ImpactLevel.HIGH: {"magnitude": -0.10, "duration": ImpactDuration.INTRADAY, "volatility": 3.2},
                    ImpactLevel.MEDIUM: {"magnitude": -0.05, "duration": ImpactDuration.INTRADAY, "volatility": 2.2},
                    ImpactLevel.LOW: {"magnitude": -0.025, "duration": ImpactDuration.FLASH, "volatility": 1.6}
                }
            },
            
            EventType.EARNINGS: {
                ImpactDirection.POSITIVE: {
                    ImpactLevel.CRITICAL: {"magnitude": 0.12, "duration": ImpactDuration.INTRADAY, "volatility": 2.8},
                    ImpactLevel.HIGH: {"magnitude": 0.06, "duration": ImpactDuration.INTRADAY, "volatility": 2.2},
                    ImpactLevel.MEDIUM: {"magnitude": 0.03, "duration": ImpactDuration.FLASH, "volatility": 1.8},
                    ImpactLevel.LOW: {"magnitude": 0.015, "duration": ImpactDuration.FLASH, "volatility": 1.4}
                },
                ImpactDirection.NEGATIVE: {
                    ImpactLevel.CRITICAL: {"magnitude": -0.15, "duration": ImpactDuration.SWING, "volatility": 3.2},
                    ImpactLevel.HIGH: {"magnitude": -0.08, "duration": ImpactDuration.INTRADAY, "volatility": 2.5},
                    ImpactLevel.MEDIUM: {"magnitude": -0.04, "duration": ImpactDuration.INTRADAY, "volatility": 2.0},
                    ImpactLevel.LOW: {"magnitude": -0.02, "duration": ImpactDuration.FLASH, "volatility": 1.5}
                }
            },
            
            EventType.REGULATORY: {
                ImpactDirection.POSITIVE: {
                    ImpactLevel.CRITICAL: {"magnitude": 0.25, "duration": ImpactDuration.POSITIONAL, "volatility": 2.5},
                    ImpactLevel.HIGH: {"magnitude": 0.12, "duration": ImpactDuration.SWING, "volatility": 2.0},
                    ImpactLevel.MEDIUM: {"magnitude": 0.06, "duration": ImpactDuration.INTRADAY, "volatility": 1.7},
                    ImpactLevel.LOW: {"magnitude": 0.03, "duration": ImpactDuration.FLASH, "volatility": 1.4}
                },
                ImpactDirection.NEGATIVE: {
                    ImpactLevel.CRITICAL: {"magnitude": -0.30, "duration": ImpactDuration.POSITIONAL, "volatility": 3.5},
                    ImpactLevel.HIGH: {"magnitude": -0.15, "duration": ImpactDuration.SWING, "volatility": 2.8},
                    ImpactLevel.MEDIUM: {"magnitude": -0.08, "duration": ImpactDuration.INTRADAY, "volatility": 2.0},
                    ImpactLevel.LOW: {"magnitude": -0.04, "duration": ImpactDuration.FLASH, "volatility": 1.5}
                }
            }
        }
    
    def _initialize_cross_asset_correlations(self) -> None:
        """Initialize cross-asset correlation models."""
        self.cross_asset_correlations = {
            # Crypto correlations
            "BTC": {
                "ETH": 0.85,
                "SOL": 0.75,
                "AVAX": 0.70,
                "SPY": 0.45,  # S&P 500
                "GLD": 0.30   # Gold
            },
            "ETH": {
                "BTC": 0.85,
                "SOL": 0.80,
                "AVAX": 0.75,
                "SPY": 0.40,
                "GLD": 0.25
            },
            # Stock correlations
            "SPY": {
                "QQQ": 0.92,  # NASDAQ
                "DIA": 0.95,  # Dow Jones
                "BTC": 0.45,
                "GLD": -0.20
            },
            # Macro correlations
            "DXY": {  # Dollar Index
                "BTC": -0.65,
                "GLD": -0.45,
                "SPY": -0.30
            }
        }
    
    def _initialize_volatility_models(self) -> None:
        """Initialize volatility prediction models."""
        self.volatility_models = {
            "base_volatility": {
                "crypto": 0.04,  # 4% daily vol
                "stocks": 0.02,  # 2% daily vol
                "forex": 0.01   # 1% daily vol
            },
            "impact_multipliers": {
                ImpactLevel.CRITICAL: 3.0,
                ImpactLevel.HIGH: 2.0,
                ImpactLevel.MEDIUM: 1.5,
                ImpactLevel.LOW: 1.2
            },
            "time_decay": {
                ImpactDuration.FLASH: 0.5,    # 50% decay in 1 hour
                ImpactDuration.INTRADAY: 0.3,  # 30% decay in 4 hours
                ImpactDuration.SWING: 0.1,     # 10% decay in 1 day
                ImpactDuration.POSITIONAL: 0.05  # 5% decay in 1 week
            }
        }
    
    def predict_impact(self, event_classification: EventClassification, 
                      market_data: Dict[str, Any] = None) -> ImpactPrediction:
        """
        Predict market impact for a classified event.
        
        Args:
            event_classification: Classified event
            market_data: Current market data context
            
        Returns:
            Complete impact prediction
        """
        try:
            # Predict primary impact
            primary_impact = self._predict_primary_impact(event_classification, market_data)
            
            # Predict secondary impacts
            secondary_impacts = self._predict_secondary_impacts(event_classification, primary_impact)
            
            # Predict cross-asset effects
            cross_asset_effects = self._predict_cross_asset_effects(primary_impact)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(event_classification, primary_impact)
            
            # Identify trading opportunities
            trading_opportunities = self._identify_trading_opportunities(primary_impact, cross_asset_effects)
            
            # Calculate overall confidence
            confidence_score = self._calculate_prediction_confidence(event_classification, primary_impact)
            
            # Create prediction
            prediction = ImpactPrediction(
                event_classification=event_classification,
                primary_impact=primary_impact,
                secondary_impacts=secondary_impacts,
                cross_asset_effects=cross_asset_effects,
                risk_factors=risk_factors,
                trading_opportunities=trading_opportunities,
                confidence_score=confidence_score,
                prediction_timestamp=datetime.now()
            )
            
            # Store prediction
            self._store_prediction(prediction)
            
            # Update stats
            self._update_prediction_stats(prediction)
            
            self.logger.info(f"Impact predicted: {primary_impact.direction.value} {primary_impact.magnitude:.2%} | Confidence: {confidence_score:.2f}")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Impact prediction failed: {e}")
            # Return minimal prediction
            return ImpactPrediction(
                event_classification=event_classification,
                primary_impact=MarketImpact(
                    direction=ImpactDirection.NEUTRAL,
                    magnitude=0.0,
                    duration=ImpactDuration.FLASH,
                    volatility_increase=1.0,
                    volume_increase=1.0,
                    confidence=0.1,
                    affected_assets=event_classification.symbols_affected,
                    time_to_impact=0,
                    reasoning=f"Prediction failed: {str(e)}"
                ),
                secondary_impacts=[],
                cross_asset_effects={},
                risk_factors=["Prediction error"],
                trading_opportunities=[],
                confidence_score=0.1,
                prediction_timestamp=datetime.now()
            )
    
    def _predict_primary_impact(self, event_classification: EventClassification, 
                               market_data: Dict[str, Any]) -> MarketImpact:
        """Predict primary market impact."""
        # Determine direction from event content
        direction = self._determine_impact_direction(event_classification)
        
        # Get base impact parameters
        event_type = event_classification.event_type
        impact_level = event_classification.impact_level
        
        if event_type in self.impact_models and direction in self.impact_models[event_type]:
            impact_params = self.impact_models[event_type][direction][impact_level]
        else:
            # Default parameters
            impact_params = {
                "magnitude": 0.02 if direction == ImpactDirection.POSITIVE else -0.02,
                "duration": ImpactDuration.INTRADAY,
                "volatility": 1.5
            }
        
        # Adjust based on market regime
        regime_adjustment = self._calculate_regime_adjustment(event_classification.market_regime_impact)
        
        # Adjust based on current volatility
        volatility_adjustment = self._calculate_volatility_adjustment(market_data)
        
        # Calculate final magnitude
        base_magnitude = impact_params["magnitude"]
        final_magnitude = base_magnitude * (1 + regime_adjustment) * (1 + volatility_adjustment)
        
        # Calculate volume increase
        volume_increase = self._calculate_volume_increase(impact_level, direction)
        
        # Calculate time to impact
        time_to_impact = self._calculate_time_to_impact(event_classification.time_horizon)
        
        # Generate reasoning
        reasoning = self._generate_impact_reasoning(event_classification, direction, impact_params, final_magnitude)
        
        return MarketImpact(
            direction=direction,
            magnitude=final_magnitude,
            duration=impact_params["duration"],
            volatility_increase=impact_params["volatility"],
            volume_increase=volume_increase,
            confidence=event_classification.confidence,
            affected_assets=event_classification.symbols_affected,
            time_to_impact=time_to_impact,
            reasoning=reasoning,
            metadata={
                "base_magnitude": base_magnitude,
                "regime_adjustment": regime_adjustment,
                "volatility_adjustment": volatility_adjustment
            }
        )
    
    def _determine_impact_direction(self, event_classification: EventClassification) -> ImpactDirection:
        """Determine impact direction from event content."""
        content = f"{event_classification.metadata.get('title', '')} {event_classification.metadata.get('content', '')}".lower()
        
        # Positive indicators
        positive_keywords = [
            "bullish", "positive", "growth", "increase", "rise", "surge", "rally",
            "approval", "launch", "partnership", "adoption", "breakthrough", "beat",
            "upgrade", "buy", "strong", "robust", "expansion"
        ]
        
        # Negative indicators
        negative_keywords = [
            "bearish", "negative", "decline", "decrease", "fall", "drop", "crash",
            "rejection", "ban", "restriction", "investigation", "lawsuit", "fine",
            "downgrade", "sell", "weak", "concern", "contraction", "miss"
        ]
        
        # Volatility indicators
        volatility_keywords = [
            "uncertain", "mixed", "volatile", "unclear", "unknown", "pending",
            "wait", "see", "monitor", "cautious", "neutral"
        ]
        
        positive_score = sum(1 for kw in positive_keywords if kw in content)
        negative_score = sum(1 for kw in negative_keywords if kw in content)
        volatility_score = sum(1 for kw in volatility_keywords if kw in content)
        
        if volatility_score > max(positive_score, negative_score):
            return ImpactDirection.VOLATILE
        elif positive_score > negative_score:
            return ImpactDirection.POSITIVE
        elif negative_score > positive_score:
            return ImpactDirection.NEGATIVE
        else:
            return ImpactDirection.NEUTRAL
    
    def _calculate_regime_adjustment(self, regime_impact: Dict[str, float]) -> float:
        """Calculate adjustment based on market regime impact."""
        if not regime_impact:
            return 0.0
        
        # Weight regime impacts
        regime_weights = {
            MarketRegime.RISK_ON: 0.3,
            MarketRegime.RISK_OFF: 0.4,
            MarketRegime.SIDEWAYS: 0.2,
            MarketRegime.VOLATILE: 0.1
        }
        
        total_adjustment = 0.0
        for regime, impact in regime_impact.items():
            weight = regime_weights.get(regime, 0.0)
            total_adjustment += impact * weight
        
        return total_adjustment
    
    def _calculate_volatility_adjustment(self, market_data: Dict[str, Any]) -> float:
        """Calculate adjustment based on current volatility."""
        if not market_data:
            return 0.0
        
        current_volatility = market_data.get("volatility", 0.02)
        base_volatility = self.volatility_models["base_volatility"].get("crypto", 0.02)
        
        # If current volatility is high, reduce impact magnitude
        if current_volatility > base_volatility * 2:
            return -0.3  # Reduce impact by 30%
        elif current_volatility < base_volatility * 0.5:
            return 0.2   # Increase impact by 20%
        
        return 0.0
    
    def _calculate_volume_increase(self, impact_level: ImpactLevel, direction: ImpactDirection) -> float:
        """Calculate expected volume increase."""
        base_multipliers = {
            ImpactLevel.CRITICAL: 4.0,
            ImpactLevel.HIGH: 2.5,
            ImpactLevel.MEDIUM: 1.8,
            ImpactLevel.LOW: 1.3
        }
        
        base_multiplier = base_multipliers.get(impact_level, 1.5)
        
        # Adjust based on direction
        if direction == ImpactDirection.VOLATILE:
            base_multiplier *= 1.5
        elif direction == ImpactDirection.NEUTRAL:
            base_multiplier *= 0.8
        
        return base_multiplier
    
    def _calculate_time_to_impact(self, time_horizon: TimeHorizon) -> int:
        """Calculate time to impact in minutes."""
        horizon_times = {
            TimeHorizon.IMMEDIATE: 5,
            TimeHorizon.SHORT: 30,
            TimeHorizon.MEDIUM: 240,  # 4 hours
            TimeHorizon.LONG: 1440,   # 24 hours
            TimeHorizon.EXTENDED: 10080  # 7 days
        }
        
        return horizon_times.get(time_horizon, 30)
    
    def _generate_impact_reasoning(self, event_classification: EventClassification, direction: ImpactDirection,
                                  impact_params: Dict[str, Any], final_magnitude: float) -> str:
        """Generate reasoning for impact prediction."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Event type: {event_classification.event_type.value}")
        reasoning_parts.append(f"Impact level: {event_classification.impact_level.name}")
        reasoning_parts.append(f"Direction: {direction.value}")
        reasoning_parts.append(f"Expected move: {final_magnitude:.2%}")
        reasoning_parts.append(f"Duration: {impact_params['duration'].value}")
        reasoning_parts.append(f"Volatility multiplier: {impact_params['volatility']:.1f}x")
        
        return " | ".join(reasoning_parts)
    
    def _predict_secondary_impacts(self, event_classification: EventClassification, 
                                  primary_impact: MarketImpact) -> List[MarketImpact]:
        """Predict secondary impacts from primary impact."""
        secondary_impacts = []
        
        # Time-based secondary impacts
        if primary_impact.duration in [ImpactDuration.SWING, ImpactDuration.POSITIONAL]:
            # Follow-up impacts after initial reaction
            follow_up = MarketImpact(
                direction=ImpactDirection.VOLATILE,
                magnitude=primary_impact.magnitude * 0.3,
                duration=ImpactDuration.INTRADAY,
                volatility_increase=1.8,
                volume_increase=1.5,
                confidence=primary_impact.confidence * 0.7,
                affected_assets=primary_impact.affected_assets,
                time_to_impact=primary_impact.time_to_impact + 240,  # 4 hours later
                reasoning="Secondary volatility after initial reaction"
            )
            secondary_impacts.append(follow_up)
        
        # Reversal impacts for strong moves
        if abs(primary_impact.magnitude) > 0.05:
            reversal = MarketImpact(
                direction=ImpactDirection.POSITIVE if primary_impact.direction == ImpactDirection.NEGATIVE else ImpactDirection.NEGATIVE,
                magnitude=primary_impact.magnitude * -0.2,
                duration=ImpactDuration.INTRADAY,
                volatility_increase=1.5,
                volume_increase=1.3,
                confidence=primary_impact.confidence * 0.5,
                affected_assets=primary_impact.affected_assets,
                time_to_impact=primary_impact.time_to_impact + 480,  # 8 hours later
                reasoning="Potential reversal after strong initial move"
            )
            secondary_impacts.append(reversal)
        
        return secondary_impacts
    
    def _predict_cross_asset_effects(self, primary_impact: MarketImpact) -> Dict[str, MarketImpact]:
        """Predict cross-asset effects from primary impact."""
        cross_effects = {}
        
        for asset in primary_impact.affected_assets:
            if asset in self.cross_asset_correlations:
                correlations = self.cross_asset_correlations[asset]
                
                for correlated_asset, correlation in correlations.items():
                    if abs(correlation) > 0.5:  # Only strong correlations
                        # Calculate cross-asset impact
                        cross_magnitude = primary_impact.magnitude * correlation * 0.7
                        cross_direction = ImpactDirection.POSITIVE if cross_magnitude > 0 else ImpactDirection.NEGATIVE
                        
                        cross_effect = MarketImpact(
                            direction=cross_direction,
                            magnitude=abs(cross_magnitude),
                            duration=primary_impact.duration,
                            volatility_increase=1.3,
                            volume_increase=1.2,
                            confidence=primary_impact.confidence * abs(correlation),
                            affected_assets=[correlated_asset],
                            time_to_impact=primary_impact.time_to_impact + 15,  # 15 minutes delay
                            reasoning=f"Cross-asset correlation with {asset} ({correlation:.2f})"
                        )
                        
                        cross_effects[correlated_asset] = cross_effect
        
        return cross_effects
    
    def _identify_risk_factors(self, event_classification: EventClassification, 
                             primary_impact: MarketImpact) -> List[str]:
        """Identify risk factors for the prediction."""
        risk_factors = []
        
        # Event-specific risks
        if event_classification.impact_level == ImpactLevel.CRITICAL:
            risk_factors.append("Critical event - high uncertainty")
        
        if primary_impact.direction == ImpactDirection.VOLATILE:
            risk_factors.append("High volatility expected")
        
        if abs(primary_impact.magnitude) > 0.1:
            risk_factors.append("Large price move expected")
        
        # Market condition risks
        if primary_impact.volatility_increase > 2.5:
            risk_factors.append("Extreme volatility risk")
        
        if primary_impact.volume_increase > 3.0:
            risk_factors.append("Liquidity risk from volume spike")
        
        # Timing risks
        if primary_impact.time_to_impact > 60:
            risk_factors.append("Delayed impact - timing uncertainty")
        
        # Cross-asset risks
        if len(primary_impact.affected_assets) > 5:
            risk_factors.append("Broad market impact")
        
        return risk_factors
    
    def _identify_trading_opportunities(self, primary_impact: MarketImpact, 
                                       cross_asset_effects: Dict[str, MarketImpact]) -> List[Dict[str, Any]]:
        """Identify trading opportunities from impact predictions."""
        opportunities = []
        
        # Primary asset opportunities
        if primary_impact.direction in [ImpactDirection.POSITIVE, ImpactDirection.NEGATIVE]:
            opportunity = {
                "asset": primary_impact.affected_assets[0] if primary_impact.affected_assets else "UNKNOWN",
                "direction": "BUY" if primary_impact.direction == ImpactDirection.POSITIVE else "SELL",
                "timeframe": primary_impact.duration.value,
                "confidence": primary_impact.confidence,
                "expected_return": abs(primary_impact.magnitude),
                "risk_level": "HIGH" if abs(primary_impact.magnitude) > 0.05 else "MEDIUM",
                "reasoning": f"Primary impact: {primary_impact.reasoning}"
            }
            opportunities.append(opportunity)
        
        # Cross-asset opportunities
        for asset, cross_effect in cross_asset_effects.items():
            if cross_effect.direction in [ImpactDirection.POSITIVE, ImpactDirection.NEGATIVE]:
                opportunity = {
                    "asset": asset,
                    "direction": "BUY" if cross_effect.direction == ImpactDirection.POSITIVE else "SELL",
                    "timeframe": cross_effect.duration.value,
                    "confidence": cross_effect.confidence,
                    "expected_return": abs(cross_effect.magnitude),
                    "risk_level": "MEDIUM",
                    "reasoning": f"Cross-asset effect: {cross_effect.reasoning}"
                }
                opportunities.append(opportunity)
        
        return opportunities
    
    def _calculate_prediction_confidence(self, event_classification: EventClassification, 
                                       primary_impact: MarketImpact) -> float:
        """Calculate overall prediction confidence."""
        base_confidence = event_classification.confidence
        
        # Adjust based on impact model confidence
        model_confidence = 0.8  # Base model confidence
        
        # Adjust based on direction clarity
        if primary_impact.direction == ImpactDirection.NEUTRAL:
            model_confidence *= 0.7
        elif primary_impact.direction == ImpactDirection.VOLATILE:
            model_confidence *= 0.6
        
        # Adjust based on magnitude
        if abs(primary_impact.magnitude) < 0.01:
            model_confidence *= 0.8  # Small moves are harder to predict
        
        return min(1.0, base_confidence * model_confidence)
    
    def _store_prediction(self, prediction: ImpactPrediction) -> None:
        """Store prediction in history."""
        self.impact_history.append({
            "timestamp": prediction.prediction_timestamp,
            "prediction": prediction,
            "metadata": {
                "event_type": prediction.event_classification.event_type.value,
                "impact_level": prediction.event_classification.impact_level.value,
                "primary_direction": prediction.primary_impact.direction.value,
                "primary_magnitude": prediction.primary_impact.magnitude,
                "confidence_score": prediction.confidence_score
            }
        })
        
        # Limit history size
        if len(self.impact_history) > 10000:
            self.impact_history.pop(0)
    
    def _update_prediction_stats(self, prediction: ImpactPrediction) -> None:
        """Update prediction statistics."""
        self.prediction_stats["total_predictions"] += 1
        
        event_type = prediction.event_classification.event_type
        self.prediction_stats["accuracy_by_type"][event_type]["total"] += 1
        
        direction = prediction.primary_impact.direction
        self.prediction_stats["direction_accuracy"][direction]["total"] += 1
    
    def get_impact_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get impact prediction summary."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_predictions = [
            entry for entry in self.impact_history 
            if entry["timestamp"] >= cutoff_time
        ]
        
        return {
            "total_predictions": self.prediction_stats["total_predictions"],
            "recent_predictions": len(recent_predictions),
            "high_impact_predictions": len([
                p for p in recent_predictions 
                if abs(p["metadata"]["primary_magnitude"]) > 0.05
            ]),
            "confidence_distribution": self._calculate_confidence_distribution(recent_predictions),
            "direction_distribution": self._calculate_direction_distribution(recent_predictions)
        }
    
    def _calculate_confidence_distribution(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate confidence distribution."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for prediction in predictions:
            confidence = prediction["metadata"]["confidence_score"]
            if confidence > 0.7:
                distribution["high"] += 1
            elif confidence > 0.4:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _calculate_direction_distribution(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate direction distribution."""
        distribution = {"positive": 0, "negative": 0, "neutral": 0, "volatile": 0}
        
        for prediction in predictions:
            direction = prediction["metadata"]["primary_direction"]
            distribution[direction] += 1
        
        return distribution
