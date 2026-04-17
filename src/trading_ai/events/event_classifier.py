"""
Event Classification System following Qlib and FinRL patterns.
Classifies market events and predicts their impact on trading decisions.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import re
from collections import defaultdict

from ..infrastructure.logging import get_logger
from ..core.models import MarketRegime


class EventType(Enum):
    """Event types following Qlib classification."""
    MACRO_ECONOMIC = "macro_economic"
    CRYPTO_SPECIFIC = "crypto_specific"
    EARNINGS = "earnings"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    LIQUIDITY = "liquidity"
    GEOPOLITICAL = "geopolitical"


class ImpactLevel(Enum):
    """Impact levels following institutional trading."""
    CRITICAL = 4  # Market-moving events
    HIGH = 3      # Significant impact
    MEDIUM = 2    # Moderate impact
    LOW = 1       # Minor impact
    NONE = 0      # No impact


class TimeHorizon(Enum):
    """Time horizon for market reaction."""
    IMMEDIATE = "immediate"    # 0-30 minutes
    SHORT = "short"           # 30 minutes - 4 hours
    MEDIUM = "medium"         # 4 hours - 24 hours
    LONG = "long"            # 24 hours - 7 days
    EXTENDED = "extended"     # 7+ days


@dataclass
class EventClassification:
    """Event classification result."""
    event_type: EventType
    impact_level: ImpactLevel
    time_horizon: TimeHorizon
    confidence: float
    symbols_affected: List[str]
    market_regime_impact: Dict[str, float]
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventClassifier:
    """
    Event classifier following Qlib and FinRL patterns.
    
    Key features:
    - Multi-dimensional event classification
    - Impact scoring based on historical patterns
    - Time horizon prediction
    - Symbol-specific impact analysis
    - Market regime correlation
    """
    
    def __init__(self):
        """Initialize event classifier."""
        self.logger = get_logger("event_classifier")
        
        # Classification patterns
        self._initialize_classification_patterns()
        
        # Impact scoring models
        self._initialize_impact_models()
        
        # Historical event database
        self.event_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.classification_stats = {
            "total_classified": 0,
            "accuracy_by_type": defaultdict(lambda: {"correct": 0, "total": 0}),
            "impact_distribution": defaultdict(int),
            "horizon_distribution": defaultdict(int)
        }
        
        self.logger.info("EventClassifier initialized with institutional-grade patterns")
    
    def _initialize_classification_patterns(self) -> None:
        """Initialize classification patterns for event types."""
        self.classification_patterns = {
            EventType.MACRO_ECONOMIC: {
                "keywords": [
                    "inflation", "cpi", "fed", "interest rate", "gdp", "unemployment",
                    "recession", "economic growth", "monetary policy", "quantitative easing",
                    "fomc", "central bank", "economic data", "market volatility"
                ],
                "impact_keywords": {
                    ImpactLevel.CRITICAL: ["rate hike", "rate cut", "recession", "inflation spike"],
                    ImpactLevel.HIGH: ["gdp growth", "unemployment rate", "cpi data"],
                    ImpactLevel.MEDIUM: ["economic indicators", "market sentiment"],
                    ImpactLevel.LOW: ["minor economic news", "market commentary"]
                },
                "time_horizons": {
                    TimeHorizon.IMMEDIATE: ["rate decision", "emergency meeting"],
                    TimeHorizon.SHORT: ["economic data release", "fed statement"],
                    TimeHorizon.MEDIUM: ["gdp report", "inflation data"],
                    TimeHorizon.LONG: ["monetary policy", "economic outlook"]
                }
            },
            
            EventType.CRYPTO_SPECIFIC: {
                "keywords": [
                    "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
                    "defi", "nft", "mining", "hash rate", "whale", "exchange",
                    "binance", "coinbase", "regulation", "sec", "etf"
                ],
                "impact_keywords": {
                    ImpactLevel.CRITICAL: ["sec approval", "etf launch", "exchange hack", "ban"],
                    ImpactLevel.HIGH: ["regulation", "whale movement", "exchange listing"],
                    ImpactLevel.MEDIUM: ["adoption news", "technical developments"],
                    ImpactLevel.LOW: ["market commentary", "price analysis"]
                },
                "time_horizons": {
                    TimeHorizon.IMMEDIATE: ["hack", "exchange failure", "regulation"],
                    TimeHorizon.SHORT: ["whale movement", "large transaction"],
                    TimeHorizon.MEDIUM: ["adoption", "partnership", "listing"],
                    TimeHorizon.LONG: ["regulatory changes", "etf approval"]
                }
            },
            
            EventType.EARNINGS: {
                "keywords": [
                    "earnings", "revenue", "profit", "quarterly", "annual", "eps",
                    "guidance", "forecast", "beat", "miss", "analyst", "estimate"
                ],
                "impact_keywords": {
                    ImpactLevel.CRITICAL: ["bankruptcy", "merger", "acquisition"],
                    ImpactLevel.HIGH: ["earnings beat/miss", "guidance change"],
                    ImpactLevel.MEDIUM: ["quarterly results", "revenue growth"],
                    ImpactLevel.LOW: ["analyst upgrade/downgrade", "price target"]
                },
                "time_horizons": {
                    TimeHorizon.IMMEDIATE: ["earnings surprise", "guidance"],
                    TimeHorizon.SHORT: ["quarterly results", "analyst reaction"],
                    TimeHorizon.MEDIUM: ["annual results", "strategic changes"],
                    TimeHorizon.LONG: ["merger", "acquisition", "restructuring"]
                }
            },
            
            EventType.REGULATORY: {
                "keywords": [
                    "regulation", "sec", "fca", "esma", "compliance", "investigation",
                    "lawsuit", "fine", "penalty", "approval", "rejection", "ban"
                ],
                "impact_keywords": {
                    ImpactLevel.CRITICAL: ["market ban", "major fine", "criminal charges"],
                    ImpactLevel.HIGH: ["regulatory approval", "investigation", "lawsuit"],
                    ImpactLevel.MEDIUM: ["compliance issues", "minor fines"],
                    ImpactLevel.LOW: ["regulatory commentary", "policy discussion"]
                },
                "time_horizons": {
                    TimeHorizon.IMMEDIATE: ["emergency action", "trading halt"],
                    TimeHorizon.SHORT: ["regulatory decision", "fine announcement"],
                    TimeHorizon.MEDIUM: ["investigation results", "compliance deadline"],
                    TimeHorizon.LONG: ["regulatory changes", "policy implementation"]
                }
            },
            
            EventType.GEOPOLITICAL: {
                "keywords": [
                    "war", "conflict", "sanctions", "trade war", "geopolitical",
                    "election", "political", "government", "policy", "tension"
                ],
                "impact_keywords": {
                    ImpactLevel.CRITICAL: ["war", "major sanctions", "government collapse"],
                    ImpactLevel.HIGH: ["election results", "trade sanctions"],
                    ImpactLevel.MEDIUM: ["political tension", "policy changes"],
                    ImpactLevel.LOW: ["political commentary", "diplomatic news"]
                },
                "time_horizons": {
                    TimeHorizon.IMMEDIATE: ["military conflict", "emergency sanctions"],
                    TimeHorizon.SHORT: ["election results", "policy announcements"],
                    TimeHorizon.MEDIUM: ["trade negotiations", "sanctions"],
                    TimeHorizon.LONG: ["geopolitical tensions", "long-term policies"]
                }
            }
        }
    
    def _initialize_impact_models(self) -> None:
        """Initialize impact scoring models."""
        self.impact_models = {
            "keyword_weight": 0.3,
            "sentiment_weight": 0.25,
            "volume_weight": 0.2,
            "timing_weight": 0.15,
            "historical_weight": 0.1
        }
        
        # Historical impact patterns
        self.historical_impacts = {
            EventType.MACRO_ECONOMIC: {
                ImpactLevel.CRITICAL: 0.85,
                ImpactLevel.HIGH: 0.65,
                ImpactLevel.MEDIUM: 0.45,
                ImpactLevel.LOW: 0.25
            },
            EventType.CRYPTO_SPECIFIC: {
                ImpactLevel.CRITICAL: 0.90,
                ImpactLevel.HIGH: 0.70,
                ImpactLevel.MEDIUM: 0.50,
                ImpactLevel.LOW: 0.30
            },
            EventType.EARNINGS: {
                ImpactLevel.CRITICAL: 0.80,
                ImpactLevel.HIGH: 0.60,
                ImpactLevel.MEDIUM: 0.40,
                ImpactLevel.LOW: 0.20
            },
            EventType.REGULATORY: {
                ImpactLevel.CRITICAL: 0.95,
                ImpactLevel.HIGH: 0.75,
                ImpactLevel.MEDIUM: 0.55,
                ImpactLevel.LOW: 0.35
            },
            EventType.GEOPOLITICAL: {
                ImpactLevel.CRITICAL: 0.75,
                ImpactLevel.HIGH: 0.55,
                ImpactLevel.MEDIUM: 0.35,
                ImpactLevel.LOW: 0.15
            }
        }
    
    def classify_event(self, title: str, content: str, timestamp: datetime, 
                      symbols: List[str] = None) -> EventClassification:
        """
        Classify a market event.
        
        Args:
            title: Event title
            content: Event content
            timestamp: Event timestamp
            symbols: Affected symbols
            
        Returns:
            EventClassification result
        """
        try:
            # Combine title and content for analysis
            full_text = f"{title} {content}".lower()
            
            # Classify event type
            event_type = self._classify_event_type(full_text)
            
            # Determine impact level
            impact_level = self._determine_impact_level(full_text, event_type)
            
            # Predict time horizon
            time_horizon = self._predict_time_horizon(full_text, event_type, impact_level)
            
            # Calculate confidence
            confidence = self._calculate_classification_confidence(full_text, event_type, impact_level)
            
            # Identify affected symbols
            affected_symbols = self._identify_affected_symbols(full_text, symbols)
            
            # Calculate market regime impact
            regime_impact = self._calculate_regime_impact(event_type, impact_level, affected_symbols)
            
            # Generate reasoning
            reasoning = self._generate_classification_reasoning(event_type, impact_level, time_horizon, full_text)
            
            # Create classification result
            classification = EventClassification(
                event_type=event_type,
                impact_level=impact_level,
                time_horizon=time_horizon,
                confidence=confidence,
                symbols_affected=affected_symbols,
                market_regime_impact=regime_impact,
                reasoning=reasoning,
                metadata={
                    "title": title,
                    "content": content,
                    "timestamp": timestamp,
                    "classification_timestamp": datetime.now()
                }
            )
            
            # Store in history
            self._store_classification(classification)
            
            # Update stats
            self._update_classification_stats(classification)
            
            self.logger.info(f"Event classified: {event_type.value} | Impact: {impact_level.name} | Horizon: {time_horizon.value}")
            
            return classification
            
        except Exception as e:
            self.logger.error(f"Event classification failed: {e}")
            # Return default classification
            return EventClassification(
                event_type=EventType.SENTIMENT,
                impact_level=ImpactLevel.LOW,
                time_horizon=TimeHorizon.SHORT,
                confidence=0.1,
                symbols_affected=symbols or [],
                market_regime_impact={},
                reasoning=f"Classification failed: {str(e)}"
            )
    
    def _classify_event_type(self, text: str) -> EventType:
        """Classify event type based on text analysis."""
        type_scores = {}
        
        for event_type, patterns in self.classification_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in patterns["keywords"]:
                if keyword in text:
                    score += 1
            
            # Normalize by keyword count
            if patterns["keywords"]:
                score /= len(patterns["keywords"])
            
            type_scores[event_type] = score
        
        # Return type with highest score
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return EventType.SENTIMENT
    
    def _determine_impact_level(self, text: str, event_type: EventType) -> ImpactLevel:
        """Determine impact level based on text analysis."""
        if event_type not in self.classification_patterns:
            return ImpactLevel.MEDIUM
        
        patterns = self.classification_patterns[event_type]
        impact_scores = {}
        
        for impact_level, keywords in patterns["impact_keywords"].items():
            score = 0
            
            for keyword in keywords:
                if keyword in text:
                    score += 2  # Higher weight for impact keywords
            
            # Add historical probability
            historical_prob = self.historical_impacts.get(event_type, {}).get(impact_level, 0.5)
            score += historical_prob * 3
            
            impact_scores[impact_level] = score
        
        if impact_scores:
            return max(impact_scores, key=impact_scores.get)
        
        return ImpactLevel.MEDIUM
    
    def _predict_time_horizon(self, text: str, event_type: EventType, impact_level: ImpactLevel) -> TimeHorizon:
        """Predict time horizon for market reaction."""
        if event_type not in self.classification_patterns:
            return TimeHorizon.SHORT
        
        patterns = self.classification_patterns[event_type]
        horizon_scores = {}
        
        for time_horizon, keywords in patterns["time_horizons"].items():
            score = 0
            
            for keyword in keywords:
                if keyword in text:
                    score += 1
            
            # Adjust based on impact level
            if impact_level == ImpactLevel.CRITICAL:
                if time_horizon in [TimeHorizon.IMMEDIATE, TimeHorizon.SHORT]:
                    score += 2
            elif impact_level == ImpactLevel.HIGH:
                if time_horizon in [TimeHorizon.SHORT, TimeHorizon.MEDIUM]:
                    score += 1
            
            horizon_scores[time_horizon] = score
        
        if horizon_scores:
            return max(horizon_scores, key=horizon_scores.get)
        
        return TimeHorizon.SHORT
    
    def _calculate_classification_confidence(self, text: str, event_type: EventType, impact_level: ImpactLevel) -> float:
        """Calculate confidence in classification."""
        confidence = 0.5  # Base confidence
        
        # Keyword confidence
        if event_type in self.classification_patterns:
            patterns = self.classification_patterns[event_type]
            keyword_matches = sum(1 for kw in patterns["keywords"] if kw in text)
            keyword_confidence = min(1.0, keyword_matches / len(patterns["keywords"]))
            confidence += keyword_confidence * 0.3
        
        # Impact level confidence
        impact_confidence = self.historical_impacts.get(event_type, {}).get(impact_level, 0.5)
        confidence += impact_confidence * 0.2
        
        # Text length confidence (more text = more context)
        text_confidence = min(1.0, len(text) / 1000)
        confidence += text_confidence * 0.1
        
        return min(1.0, confidence)
    
    def _identify_affected_symbols(self, text: str, provided_symbols: List[str]) -> List[str]:
        """Identify symbols affected by the event."""
        affected = set()
        
        # Use provided symbols
        if provided_symbols:
            affected.update(provided_symbols)
        
        # Extract symbols from text
        crypto_symbols = ["btc", "bitcoin", "eth", "ethereum", "bnb", "binance", "sol", "solana"]
        stock_symbols = ["aapl", "apple", "msft", "microsoft", "googl", "google", "amzn", "amazon"]
        
        for symbol in crypto_symbols + stock_symbols:
            if symbol in text:
                affected.add(symbol.upper())
        
        return list(affected)
    
    def _calculate_regime_impact(self, event_type: EventType, impact_level: ImpactLevel, symbols: List[str]) -> Dict[str, float]:
        """Calculate impact on different market regimes."""
        regime_impacts = {}
        
        # Base impacts by event type
        base_impacts = {
            MarketRegime.RISK_ON: 0.1,
            MarketRegime.RISK_OFF: -0.1,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.VOLATILE: 0.2
        }
        
        # Adjust based on event type
        if event_type == EventType.MACRO_ECONOMIC:
            if impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH]:
                regime_impacts[MarketRegime.RISK_OFF] = 0.3
                regime_impacts[MarketRegime.VOLATILE] = 0.4
                regime_impacts[MarketRegime.RISK_ON] = -0.2
        
        elif event_type == EventType.CRYPTO_SPECIFIC:
            regime_impacts[MarketRegime.VOLATILE] = 0.5
            if impact_level == ImpactLevel.CRITICAL:
                regime_impacts[MarketRegime.RISK_OFF] = 0.2
        
        elif event_type == EventType.REGULATORY:
            regime_impacts[MarketRegime.RISK_OFF] = 0.4
            regime_impacts[MarketRegime.VOLATILE] = 0.3
        
        # Apply base impacts
        for regime, impact in base_impacts.items():
            if regime not in regime_impacts:
                regime_impacts[regime] = impact
        
        # Scale by impact level
        impact_multiplier = impact_level.value / 2.0
        for regime in regime_impacts:
            regime_impacts[regime] *= impact_multiplier
        
        return regime_impacts
    
    def _generate_classification_reasoning(self, event_type: EventType, impact_level: ImpactLevel, 
                                         time_horizon: TimeHorizon, text: str) -> str:
        """Generate reasoning for classification."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Classified as {event_type.value.replace('_', ' ').title()}")
        reasoning_parts.append(f"Impact level: {impact_level.name}")
        reasoning_parts.append(f"Expected reaction time: {time_horizon.value}")
        
        # Add key factors
        if event_type in self.classification_patterns:
            patterns = self.classification_patterns[event_type]
            matched_keywords = [kw for kw in patterns["keywords"] if kw in text]
            if matched_keywords:
                reasoning_parts.append(f"Key indicators: {', '.join(matched_keywords[:3])}")
        
        return " | ".join(reasoning_parts)
    
    def _store_classification(self, classification: EventClassification) -> None:
        """Store classification in history."""
        self.event_history.append({
            "timestamp": datetime.now(),
            "classification": classification,
            "metadata": {
                "event_type": classification.event_type.value,
                "impact_level": classification.impact_level.value,
                "time_horizon": classification.time_horizon.value,
                "confidence": classification.confidence
            }
        })
        
        # Limit history size
        if len(self.event_history) > 10000:
            self.event_history.pop(0)
    
    def _update_classification_stats(self, classification: EventClassification) -> None:
        """Update classification statistics."""
        self.classification_stats["total_classified"] += 1
        self.classification_stats["impact_distribution"][classification.impact_level.name] += 1
        self.classification_stats["horizon_distribution"][classification.time_horizon.value] += 1
    
    def get_classification_summary(self) -> Dict[str, Any]:
        """Get classification summary statistics."""
        return {
            "total_classified": self.classification_stats["total_classified"],
            "impact_distribution": dict(self.classification_stats["impact_distribution"]),
            "horizon_distribution": dict(self.classification_stats["horizon_distribution"]),
            "recent_classifications": [
                {
                    "timestamp": entry["timestamp"],
                    "event_type": entry["metadata"]["event_type"],
                    "impact_level": entry["metadata"]["impact_level"],
                    "confidence": entry["metadata"]["confidence"]
                }
                for entry in self.event_history[-10:]
            ]
        }
    
    def get_high_impact_events(self, hours: int = 24) -> List[EventClassification]:
        """Get high impact events from recent history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        high_impact_events = []
        for entry in self.event_history:
            if entry["timestamp"] >= cutoff_time:
                classification = entry["classification"]
                if classification.impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH]:
                    high_impact_events.append(classification)
        
        return high_impact_events
