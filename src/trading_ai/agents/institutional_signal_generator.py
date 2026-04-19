"""
Institutional-grade signal generator for TRADING-AI system.
Replaces keyword-based logic with multi-factor signal intelligence.
"""

from __future__ import annotations

import re
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..core.models import Article, Signal, SignalDirection, Urgency, MarketRegime, SignalType
from ..infrastructure.config import config
from ..infrastructure.logging import get_logger


@dataclass
class SignalFactors:
    """Signal factors for confidence calculation."""
    sentiment_score: float  # -1 to +1
    source_credibility: float  # 0 to 1
    event_severity: float  # 0 to 1
    keyword_strength: float  # 0 to 1
    entity_relevance: float  # 0 to 1


@dataclass
class ExtractedEntity:
    """Extracted entity from article."""
    name: str
    symbol: str
    entity_type: str  # crypto, stock, macro
    confidence: float
    mentions: int


@dataclass
class ClassifiedEvent:
    """Classified market event."""
    event_type: str  # earnings, regulation, macro, adoption, technical
    severity: float  # 0 to 1
    description: str
    confidence: float


class InstitutionalSignalGenerator:
    """Institutional-grade signal generator with multi-factor analysis."""
    
    def __init__(self) -> None:
        """Initialize institutional signal generator."""
        self.logger = get_logger("institutional_signal_generator")
        
        # Source credibility weights (higher = more reliable)
        self.source_credibility = {
            "crypto_panic": 0.9,
            "decrypt_media": 0.85,
            "the_block": 0.85,
            "coin_desk": 0.9,
            "coin_telegraph": 0.8,
            "alpha_street": 0.9,
            "benzinga": 0.85,
            "investing_com": 0.8,
            "marketbeat": 0.8,
            "y_finance": 0.85,
            "tech_crunch": 0.75,
            "arstechnica": 0.75,
            "wired": 0.75,
            "federal_reserve": 0.95,
            "cpi_data": 0.95,
        }
        
        # Entity symbol mappings
        self.crypto_mappings = {
            "bitcoin": "BTC", "btc": "BTC", "bitcoin btc": "BTC",
            "ethereum": "ETH", "eth": "ETH", "ethereum eth": "ETH",
            "cardano": "ADA", "ada": "ADA", "cardano ada": "ADA",
            "solana": "SOL", "sol": "SOL", "solana sol": "SOL",
            "polkadot": "DOT", "dot": "DOT", "polkadot dot": "DOT",
            "avalanche": "AVAX", "avax": "AVAX", "avalanche avax": "AVAX",
            "chainlink": "LINK", "link": "LINK", "chainlink link": "LINK",
            "uniswap": "UNI", "uni": "UNI", "uniswap uni": "UNI",
            "polygon": "MATIC", "matic": "MATIC", "polygon matic": "MATIC",
            "dogecoin": "DOGE", "doge": "DOGE", "dogecoin doge": "DOGE",
        }
        
        self.stock_mappings = {
            "apple": "AAPL", "aapl": "AAPL", "apple aapl": "AAPL",
            "microsoft": "MSFT", "msft": "MSFT", "microsoft msft": "MSFT",
            "google": "GOOGL", "googl": "GOOGL", "google googl": "GOOGL",
            "alphabet": "GOOGL", "alphabet googl": "GOOGL",
            "amazon": "AMZN", "amzn": "AMZN", "amazon amzn": "AMZN",
            "tesla": "TSLA", "tsla": "TSLA", "tesla tsla": "TSLA",
            "meta": "META", "meta": "META", "meta meta": "META",
            "facebook": "META", "facebook meta": "META",
            "nvidia": "NVDA", "nvda": "NVDA", "nvidia nvda": "NVDA",
            "netflix": "NFLX", "nflx": "NFLX", "netflix nflx": "NFLX",
            "paypal": "PYPL", "pypl": "PYPL", "paypal pypl": "PYPL",
        }
        
        # Event classification patterns
        self.event_patterns = {
            "earnings": [
                r'\bearnings?\b', r'\bquarterly\b', r'\bq[1-4]\b', r'\brevenue\b',
                r'\bprofit\b', r'\bloss\b', r'\beps\b', r'\bbeat\b', r'\bmiss\b'
            ],
            "regulation": [
                r'\bsec\b', r'\bregulation\b', r'\bregulatory\b', r'\blaw\b',
                r'\bill\b', r'\bcompliance\b', r'\bbanned?\b', r'\bapproved?\b'
            ],
            "macro": [
                r'\bfed\b', r'\bfederal reserve\b', r'\binflation\b', r'\bcpi\b',
                r'\bgdp\b', r'\binterest rates?\b', r'\bmonetary\b', r'\beconomic\b'
            ],
            "adoption": [
                r'\badoption\b', r'\bpartnership\b', r'\bintegration\b', r'\baccept\b',
                r'\bimplement\b', r'\blaunch\b', r'\bdeploy\b', r'\brollout\b'
            ],
            "technical": [
                r'\btechnical\b', r'\btechnical analysis\b', r'\bsupport\b',
                r'\bresistance\b', r'\bbreakout\b', r'\bcorrection\b', r'\brally\b'
            ]
        }
        
        # Sentiment lexicon (simplified VADER-style)
        self.positive_words = {
            'bullish', 'rally', 'surge', 'soar', 'skyrocket', 'boom', 'explosion',
            'breakthrough', 'milestone', 'record', 'beat', 'exceed', 'outperform',
            'rise', 'increase', 'growth', 'gain', 'positive', 'optimistic', 'strong',
            'robust', 'upgrade', 'buy', 'recommend', 'target', 'higher', 'improve',
            'boost', 'support', 'momentum', 'trend', 'adoption', 'partnership',
            'launch', 'success', 'victory', 'win', 'achieve', 'accomplish'
        }
        
        self.negative_words = {
            'bearish', 'crash', 'plunge', 'collapse', 'tumble', 'slump', 'freefall',
            'meltdown', 'crisis', 'panic', 'disaster', 'catastrophe', 'miss', 'disappoint',
            'underperform', 'fall', 'drop', 'decline', 'decrease', 'negative', 'pessimistic',
            'weak', 'downgrade', 'sell', 'avoid', 'risk', 'danger', 'threat', 'concern',
            'warning', 'alert', 'scandal', 'fraud', 'investigation', 'lawsuit', 'ban',
            'prohibit', 'restrict', 'limit', 'cap', 'reduce', 'cut', 'loss', 'failure'
        }
        
        # Dynamic threshold management
        self.min_confidence_threshold = config.MIN_SIGNAL_CONFIDENCE
        self.last_signal_count = 0
        self.threshold_adjustments = 0
        
        self.logger.info("Institutional signal generator initialized")
    
    def generate_signals(self, articles: List[Article]) -> List[Signal]:
        """Generate institutional-grade trading signals from articles."""
        signals = []
        
        self.logger.info(f"Generating institutional signals from {len(articles)} articles")
        
        for article in articles:
            article_signals = self._generate_article_signals(article)
            signals.extend(article_signals)
        
        # Dynamic threshold adjustment
        if len(signals) == 0:
            self._adjust_threshold_downward()
            # Retry with lower threshold
            for article in articles:
                article_signals = self._generate_article_signals(article)
                signals.extend(article_signals)
        
        # Sort by confidence and limit
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        self.last_signal_count = len(signals)
        self.logger.info(
            f"Institutional signal generation completed: {len(signals)} signals generated, "
            f"avg confidence: {sum(s.confidence for s in signals) / len(signals):.3f}" if signals else "No signals generated"
        )
        
        return signals
    
    def _generate_article_signals(self, article: Article) -> List[Signal]:
        """Generate signals from a single article using multi-factor analysis."""
        signals = []
        
        # Extract text for analysis
        text = f"{article.title} {article.content}".lower()
        
        # Multi-factor analysis
        sentiment_score = self._analyze_sentiment(text)
        entities = self._extract_entities(text)
        event = self._classify_event(text)
        keyword_strength = self._calculate_keyword_strength(text)
        
        # Get source credibility
        source_name = article.source.lower().replace(" ", "_")
        source_credibility = self.source_credibility.get(source_name, 0.5)
        
        # Calculate signal factors
        factors = SignalFactors(
            sentiment_score=sentiment_score,
            source_credibility=source_credibility,
            event_severity=event.severity,
            keyword_strength=keyword_strength,
            entity_relevance=self._calculate_entity_relevance(entities)
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(factors)
        
        # Debug output
        self._debug_signal_analysis(article, factors, confidence, entities, event)
        
        # Generate signals for each entity
        for entity in entities:
            if confidence >= self.min_confidence_threshold:
                signal = self._create_signal(article, entity, event, factors, confidence)
                if signal:
                    signals.append(signal)
        
        # Fallback logic - generate low confidence signals if none generated
        if not signals and entities and abs(sentiment_score) > 0.3:
            for entity in entities[:2]:  # Top 2 entities
                signal = self._create_fallback_signal(article, entity, event, factors, sentiment_score)
                if signal:
                    signals.append(signal)
                    break  # Only one fallback signal
        
        return signals
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using NLP-style approach."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Handle negations
        negation_words = {'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor'}
        negated_words = set()
        
        for i, word in enumerate(words):
            if word in negation_words and i + 1 < len(words):
                # Negate the next word
                next_word = words[i + 1]
                if next_word in self.positive_words:
                    negative_count += 1
                elif next_word in self.negative_words:
                    positive_count += 1
        
        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / math.sqrt(total_sentiment_words)
        
        # Normalize to -1 to +1 range
        return max(-1.0, min(1.0, sentiment))
    
    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities with symbol mapping."""
        entities = []
        seen_entities = set()
        
        # Extract crypto entities
        for name, symbol in self.crypto_mappings.items():
            if name in text and symbol not in seen_entities:
                mentions = len(re.findall(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))
                if mentions > 0:
                    entities.append(ExtractedEntity(
                        name=name.title(),
                        symbol=symbol,
                        entity_type="crypto",
                        confidence=0.9,
                        mentions=mentions
                    ))
                    seen_entities.add(symbol)
        
        # Extract stock entities
        for name, symbol in self.stock_mappings.items():
            if name in text and symbol not in seen_entities:
                mentions = len(re.findall(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))
                if mentions > 0:
                    entities.append(ExtractedEntity(
                        name=name.title(),
                        symbol=symbol,
                        entity_type="stock",
                        confidence=0.85,
                        mentions=mentions
                    ))
                    seen_entities.add(symbol)
        
        # Sort by mentions and confidence
        entities.sort(key=lambda e: (e.mentions, e.confidence), reverse=True)
        
        return entities[:3]  # Top 3 entities
    
    def _classify_event(self, text: str) -> ClassifiedEvent:
        """Classify market event from text."""
        event_scores = {}
        
        for event_type, patterns in self.event_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            event_scores[event_type] = score
        
        # Get best event type
        if not event_scores or max(event_scores.values()) == 0:
            return ClassifiedEvent("general", 0.5, "General market news", 0.5)
        
        best_event = max(event_scores, key=event_scores.get)
        severity = min(1.0, event_scores[best_event] / 3.0)  # Normalize severity
        
        descriptions = {
            "earnings": "Earnings announcement or financial results",
            "regulation": "Regulatory action or compliance news",
            "macro": "Macroeconomic indicator or policy change",
            "adoption": "Adoption or partnership announcement",
            "technical": "Technical analysis or price movement"
        }
        
        return ClassifiedEvent(
            event_type=best_event,
            severity=severity,
            description=descriptions.get(best_event, "Market event"),
            confidence=min(1.0, event_scores[best_event] / 2.0)
        )
    
    def _calculate_keyword_strength(self, text: str) -> float:
        """Calculate keyword strength based on market-relevant terms."""
        market_keywords = {
            'bullish', 'bearish', 'rally', 'crash', 'surge', 'plunge', 'earnings',
            'revenue', 'profit', 'loss', 'growth', 'decline', 'adoption', 'partnership',
            'regulation', 'sec', 'fed', 'inflation', 'cpi', 'gdp', 'interest', 'rates'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        keyword_count = sum(1 for word in words if word in market_keywords)
        
        # Normalize by text length
        text_length = len(words)
        if text_length == 0:
            return 0.0
        
        strength = keyword_count / text_length
        return min(1.0, strength * 10)  # Scale to 0-1 range
    
    def _calculate_entity_relevance(self, entities: List[ExtractedEntity]) -> float:
        """Calculate entity relevance score."""
        if not entities:
            return 0.0
        
        # Weight by entity type and confidence
        relevance = 0.0
        for entity in entities:
            type_weight = {"crypto": 0.9, "stock": 0.8, "macro": 0.7}.get(entity.entity_type, 0.5)
            relevance += entity.confidence * type_weight
        
        return min(1.0, relevance / len(entities))
    
    def _calculate_confidence(self, factors: SignalFactors) -> float:
        """Calculate signal confidence using weighted model."""
        confidence = (
            abs(factors.sentiment_score) * 0.4 +
            factors.source_credibility * 0.2 +
            factors.event_severity * 0.2 +
            factors.keyword_strength * 0.2
        )
        
        # Apply entity relevance boost
        confidence *= (1.0 + factors.entity_relevance * 0.2)
        
        return min(1.0, confidence)
    
    def _create_signal(self, article: Article, entity: ExtractedEntity, event: ClassifiedEvent, 
                      factors: SignalFactors, confidence: float) -> Optional[Signal]:
        """Create a signal from analysis."""
        # Determine direction based on sentiment
        if factors.sentiment_score > 0.1:
            direction = SignalDirection.BUY
        elif factors.sentiment_score < -0.1:
            direction = SignalDirection.SELL
        else:
            return None  # Neutral sentiment, no signal
        
        # Determine urgency based on event severity
        if event.severity >= 0.8:
            urgency = Urgency.HIGH
        elif event.severity >= 0.5:
            urgency = Urgency.MEDIUM
        else:
            urgency = Urgency.LOW
        
        # Calculate position size
        position_size = self._calculate_position_size(confidence, event.severity)
        
        # Create reason
        reason_parts = [
            f"{event.event_type.title()} event",
            f"sentiment: {factors.sentiment_score:+.2f}",
            f"source credibility: {factors.source_credibility:.2f}",
            f"event severity: {event.severity:.2f}"
        ]
        reason = " | ".join(reason_parts)
        
        # Calculate position size
        position_size = self._calculate_position_size(confidence, event.severity)
        
        # Create signal
        return Signal(
            symbol=entity.symbol,
            direction=direction,
            confidence=confidence,
            reason=reason,
            urgency=urgency,
            timestamp=article.timestamp,
            article_id=article.url,
            position_size=position_size,
            metadata={
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
                "event_type": event.event_type,
                "sentiment_score": factors.sentiment_score,
                "source_credibility": factors.source_credibility,
                "event_severity": event.severity
            }
        )
    
    def _calculate_position_size(self, confidence: float, event_severity: float) -> float:
        """Calculate position size based on confidence and event severity."""
        base_size = confidence * 0.5  # Base size from confidence
        severity_boost = event_severity * 0.3  # Boost from event severity
        
        position_size = base_size + severity_boost
        return min(1.0, max(0.05, position_size))
    
    def _adjust_threshold_downward(self) -> None:
        """Dynamically adjust threshold downward to avoid zero signals."""
        if self.min_confidence_threshold > 0.15:
            self.min_confidence_threshold *= 0.9
            self.threshold_adjustments += 1
            self.logger.warning(
                f"Adjusted threshold downward to {self.min_confidence_threshold:.3f} "
                f"(adjustment #{self.threshold_adjustments})"
            )
    
    def _debug_signal_analysis(self, article: Article, factors: SignalFactors, 
                               confidence: float, entities: List[ExtractedEntity], 
                               event: ClassifiedEvent) -> None:
        """Debug output for signal analysis."""
        print(f"\n=== SIGNAL ANALYSIS ===")
        print(f"Article: {article.title[:60]}...")
        print(f"Sentiment: {factors.sentiment_score:+.3f}")
        print(f"Source Credibility: {factors.source_credibility:.3f}")
        print(f"Event Severity: {factors.event_severity:.3f} ({event.event_type})")
        print(f"Keyword Strength: {factors.keyword_strength:.3f}")
        print(f"Entity Relevance: {factors.entity_relevance:.3f}")
        print(f"Confidence: {confidence:.3f} (threshold: {self.min_confidence_threshold:.3f})")
        
        if entities:
            print(f"Entities: {[f'{e.name}({e.symbol})' for e in entities[:3]]}")
        else:
            print("Entities: None")
        
        if confidence < self.min_confidence_threshold:
            print(f"REJECTED: Confidence {confidence:.3f} below threshold {self.min_confidence_threshold:.3f}")
        else:
            print(f"ACCEPTED: Signal generated")
        
        print("=" * 50)
