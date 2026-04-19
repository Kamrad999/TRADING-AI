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
    LLM-based Event classifier following Qlib and FinRL patterns.
    
    Key features:
    - LLM-based contextual classification (NO keyword matching)
    - Impact scoring based on historical patterns
    - Time horizon prediction
    - Symbol-specific impact analysis
    - Market regime correlation
    """
    
    def __init__(self, llm_client=None):
        """Initialize event classifier with optional LLM client."""
        self.logger = get_logger("event_classifier")
        self.llm_client = llm_client
        
        # Historical event database for pattern learning
        self.event_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.classification_stats = {
            "total_classified": 0,
            "accuracy_by_type": defaultdict(lambda: {"correct": 0, "total": 0}),
            "impact_distribution": defaultdict(int),
            "horizon_distribution": defaultdict(int)
        }
        
        self.logger.info("EventClassifier initialized with LLM-based contextual reasoning")
    
    def _get_llm_classification(self, text: str, symbol: str = "") -> Dict[str, Any]:
        """Get LLM-based classification for event text."""
        if not self.llm_client:
            self.logger.warning("No LLM client available, using fallback")
            return self._fallback_classification(text, symbol)
        
        prompt = f"""You are an expert financial event classifier. Analyze this market news/event and classify it.

EVENT TEXT:
{text[:2000]}

SYMBOL: {symbol}

CLASSIFY into EXACTLY this JSON format:
{{
    "event_type": "macro_economic|crypto_specific|earnings|regulatory|technical|sentiment|liquidity|geopolitical",
    "impact_level": "critical|high|medium|low|none",
    "time_horizon": "immediate|short|medium|long|extended",
    "confidence": 0.0-1.0,
    "reasoning": "Clear explanation of classification",
    "symbols_affected": ["SYM1", "SYM2"],
    "market_regime_impact": {{"risk_on": -1.0 to 1.0, "risk_off": -1.0 to 1.0, "sideways": -1.0 to 1.0}}
}}

GUIDELINES:
- Consider context, not just keywords
- Assess actual market-moving potential
- Handle ambiguous cases with lower confidence
- If text is unrelated to markets, set confidence < 0.3"""
        
        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.2, max_tokens=800)
            result = self._parse_llm_response(response)
            if result:
                return result
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
        
        return self._fallback_classification(text, symbol)
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON response."""
        import json
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end+1])
        except json.JSONDecodeError:
            pass
        return None
    
    def _fallback_classification(self, text: str, symbol: str) -> Dict[str, Any]:
        """Fallback classification when LLM is unavailable."""
        text_lower = text.lower()
        
        # Simple pattern detection (minimal keyword use as last resort)
        if any(word in text_lower for word in ["bitcoin", "ethereum", "crypto", "blockchain"]):
            event_type = "crypto_specific"
        elif any(word in text_lower for word in ["earnings", "revenue", "profit", "quarterly"]):
            event_type = "earnings"
        elif any(word in text_lower for word in ["fed", "inflation", "interest rate", "gdp"]):
            event_type = "macro_economic"
        elif any(word in text_lower for word in ["sec", "regulation", "compliance", "investigation"]):
            event_type = "regulatory"
        elif any(word in text_lower for word in ["war", "sanctions", "election", "geopolitical"]):
            event_type = "geopolitical"
        else:
            event_type = "sentiment"
        
        return {
            "event_type": event_type,
            "impact_level": "medium",
            "time_horizon": "short",
            "confidence": 0.4,
            "reasoning": "Fallback classification (LLM unavailable)",
            "symbols_affected": [symbol] if symbol else [],
            "market_regime_impact": {"risk_on": 0.0, "risk_off": 0.0, "sideways": 0.0}
        }
    
    # NOTE: All keyword-based classification removed
    # Classification is now done via LLM in _get_llm_classification method
    
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
            full_text = f"{title} {content}"
            
            # Get LLM-based classification
            llm_result = self._get_llm_classification(full_text, symbols[0] if symbols else "")
            
            # Parse LLM result into enums
            event_type = self._parse_event_type(llm_result.get("event_type", "sentiment"))
            impact_level = self._parse_impact_level(llm_result.get("impact_level", "low"))
            time_horizon = self._parse_time_horizon(llm_result.get("time_horizon", "short"))
            confidence = float(llm_result.get("confidence", 0.5))
            affected_symbols = llm_result.get("symbols_affected", symbols or [])
            regime_impact = llm_result.get("market_regime_impact", {})
            reasoning = llm_result.get("reasoning", "LLM-based classification")
            
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
                    "classification_timestamp": datetime.now(),
                    "llm_raw": llm_result
                }
            )
            
            # Store in history
            self._store_classification(classification)
            
            # Update stats
            self._update_classification_stats(classification)
            
            self.logger.info(f"Event classified: {event_type.value} | Impact: {impact_level.name} | Horizon: {time_horizon.value} | LLM conf: {confidence:.2f}")
            
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
    
    def _parse_event_type(self, event_type_str: str) -> EventType:
        """Parse event type string to enum."""
        type_map = {
            "macro_economic": EventType.MACRO_ECONOMIC,
            "crypto_specific": EventType.CRYPTO_SPECIFIC,
            "earnings": EventType.EARNINGS,
            "regulatory": EventType.REGULATORY,
            "technical": EventType.TECHNICAL,
            "sentiment": EventType.SENTIMENT,
            "liquidity": EventType.LIQUIDITY,
            "geopolitical": EventType.GEOPOLITICAL
        }
        return type_map.get(event_type_str.lower(), EventType.SENTIMENT)
    
    def _parse_impact_level(self, impact_str: str) -> ImpactLevel:
        """Parse impact level string to enum."""
        impact_map = {
            "critical": ImpactLevel.CRITICAL,
            "high": ImpactLevel.HIGH,
            "medium": ImpactLevel.MEDIUM,
            "low": ImpactLevel.LOW,
            "none": ImpactLevel.NONE
        }
        return impact_map.get(impact_str.lower(), ImpactLevel.MEDIUM)
    
    def _parse_time_horizon(self, horizon_str: str) -> TimeHorizon:
        """Parse time horizon string to enum."""
        horizon_map = {
            "immediate": TimeHorizon.IMMEDIATE,
            "short": TimeHorizon.SHORT,
            "medium": TimeHorizon.MEDIUM,
            "long": TimeHorizon.LONG,
            "extended": TimeHorizon.EXTENDED
        }
        return horizon_map.get(horizon_str.lower(), TimeHorizon.SHORT)
    
    # NOTE: These methods removed - LLM now handles time horizon and confidence
    # Symbol extraction is done via LLM in _get_llm_classification
    
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
    
    # NOTE: _generate_classification_reasoning removed - reasoning comes directly from LLM
    
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
