"""
News sentiment strategy using the new strategy interface.
Demonstrates the upgraded architecture with FinBERT-style sentiment analysis.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import re

from .base_strategy import BaseStrategy
from .strategy_interface import StrategyContext, StrategyOutput
from ..core.models import Signal, SignalDirection, Urgency, MarketRegime
from ..infrastructure.logging import get_logger


class NewsSentimentStrategy(BaseStrategy):
    """
    News sentiment strategy using advanced sentiment analysis.
    
    This strategy:
    - Analyzes news sentiment for trading signals
    - Uses multi-factor scoring
    - Implements proper risk management
    - Follows the new strategy interface
    """
    
    def __init__(self, name: str = "news_sentiment", **kwargs):
        """Initialize news sentiment strategy."""
        super().__init__(name, **kwargs)
        
        # Sentiment analysis configuration
        self.sentiment_threshold = kwargs.get("sentiment_threshold", 0.3)
        self.entity_confidence_threshold = kwargs.get("entity_confidence_threshold", 0.7)
        self.max_signals_per_news = kwargs.get("max_signals_per_news", 2)
        
        # Entity mappings (same as institutional signal generator)
        self.crypto_mappings = {
            "bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH",
            "cardano": "ADA", "ada": "ADA", "solana": "SOL", "sol": "SOL",
            "polkadot": "DOT", "dot": "DOT", "avalanche": "AVAX", "avax": "AVAX",
        }
        
        self.stock_mappings = {
            "apple": "AAPL", "aapl": "AAPL", "microsoft": "MSFT", "msft": "MSFT",
            "google": "GOOGL", "googl": "GOOGL", "amazon": "AMZN", "amzn": "AMZN",
            "tesla": "TSLA", "tsla": "TSLA", "meta": "META", "nvidia": "NVDA", "nvda": "NVDA",
            "netflix": "NFLX", "nflx": "NFLX", "paypal": "PYPL", "pypl": "PYPL",
        }
        
        # Sentiment keywords (enhanced from institutional signal generator)
        self.positive_words = {
            'bullish', 'rally', 'surge', 'soar', 'skyrocket', 'boom', 'explosion',
            'breakthrough', 'milestone', 'record', 'beat', 'exceed', 'outperform',
            'rise', 'increase', 'growth', 'gain', 'positive', 'optimistic', 'strong',
            'robust', 'upgrade', 'buy', 'recommend', 'target', 'higher', 'improve',
            'boost', 'support', 'momentum', 'trend', 'adoption', 'partnership',
            'launch', 'success', 'victory', 'win', 'achieve', 'accomplish', 'profit',
            'revenue', 'earnings', 'dividend', 'expansion', 'acquisition', 'merger'
        }
        
        self.negative_words = {
            'bearish', 'crash', 'plunge', 'collapse', 'tumble', 'slump', 'freefall',
            'meltdown', 'crisis', 'panic', 'disaster', 'catastrophe', 'miss', 'disappoint',
            'underperform', 'fall', 'drop', 'decline', 'decrease', 'negative', 'pessimistic',
            'weak', 'downgrade', 'sell', 'avoid', 'risk', 'danger', 'threat', 'concern',
            'warning', 'alert', 'scandal', 'fraud', 'investigation', 'lawsuit', 'ban',
            'prohibit', 'restrict', 'limit', 'cap', 'reduce', 'cut', 'loss', 'failure',
            'bankruptcy', 'default', 'delist', 'suspend', 'fine', 'penalty', 'violation'
        }
        
        self.logger.info("News sentiment strategy initialized")
    
    def analyze(self, context: StrategyContext) -> StrategyOutput:
        """
        Analyze news data and generate trading signals.
        
        Args:
            context: Current market context
            
        Returns:
            Strategy output with signals and adjustments
        """
        signals = []
        position_adjustments = {}
        risk_adjustments = {}
        
        # Process each news article
        for article in context.news_data:
            try:
                article_signals = self._analyze_article(article, context)
                signals.extend(article_signals)
                
                # Limit signals per article
                if len(signals) >= self.max_signals_per_news:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error analyzing article: {e}")
                continue
        
        # Validate signals
        validated_signals = []
        for signal in signals:
            if self.validate_signal(signal, context):
                validated_signals.append(signal)
        
        # Calculate position adjustments
        for signal in validated_signals:
            position_size = self.calculate_position_size(signal, context)
            position_adjustments[signal.symbol] = position_size
        
        # Get risk parameters
        risk_adjustments = self.get_risk_parameters(context)
        
        return StrategyOutput(
            signals=validated_signals,
            position_adjustments=position_adjustments,
            risk_adjustments=risk_adjustments,
            metadata={
                "strategy": self.name,
                "articles_analyzed": len(context.news_data),
                "signals_generated": len(validated_signals),
                "analysis_time": datetime.now().isoformat()
            }
        )
    
    def _analyze_article(self, article: Any, context: StrategyContext) -> List[Signal]:
        """Analyze a single news article."""
        signals = []
        
        # Get article text
        title = getattr(article, 'title', '')
        content = getattr(article, 'content', '')
        text = f"{title} {content}".lower()
        
        # Calculate sentiment
        sentiment_score = self._calculate_sentiment(text)
        
        # Skip if sentiment is weak
        if abs(sentiment_score) < self.sentiment_threshold:
            return signals
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Generate signals for each entity
        for entity in entities:
            if entity['confidence'] >= self.entity_confidence_threshold:
                signal = self._create_signal_from_entity(
                    entity, sentiment_score, article, context
                )
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score using enhanced analysis."""
        import math
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Handle negations
        negation_words = {'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor'}
        negated_words = set()
        
        for i, word in enumerate(words):
            if word in negation_words and i + 1 < len(words):
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
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract trading entities from text."""
        entities = []
        seen_entities = set()
        
        # Extract crypto entities
        for name, symbol in self.crypto_mappings.items():
            if name in text and symbol not in seen_entities:
                mentions = len(re.findall(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))
                if mentions > 0:
                    entities.append({
                        'name': name.title(),
                        'symbol': symbol,
                        'type': 'crypto',
                        'confidence': 0.9,
                        'mentions': mentions
                    })
                    seen_entities.add(symbol)
        
        # Extract stock entities
        for name, symbol in self.stock_mappings.items():
            if name in text and symbol not in seen_entities:
                mentions = len(re.findall(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))
                if mentions > 0:
                    entities.append({
                        'name': name.title(),
                        'symbol': symbol,
                        'type': 'stock',
                        'confidence': 0.85,
                        'mentions': mentions
                    })
                    seen_entities.add(symbol)
        
        # Sort by mentions and confidence
        entities.sort(key=lambda e: (e['mentions'], e['confidence']), reverse=True)
        
        return entities[:3]  # Top 3 entities
    
    def _create_signal_from_entity(self, entity: Dict[str, Any], sentiment_score: float,
                                  article: Any, context: StrategyContext) -> Optional[Signal]:
        """Create signal from entity and sentiment."""
        # Determine direction
        if sentiment_score > 0.1:
            direction = SignalDirection.BUY
        elif sentiment_score < -0.1:
            direction = SignalDirection.SELL
        else:
            return None
        
        # Calculate confidence
        base_confidence = abs(sentiment_score) * 0.6
        entity_boost = entity['confidence'] * 0.3
        market_boost = self._get_market_boost(context)
        
        confidence = min(1.0, base_confidence + entity_boost + market_boost)
        
        # Determine urgency based on sentiment strength
        if abs(sentiment_score) > 0.7:
            urgency = Urgency.HIGH
        elif abs(sentiment_score) > 0.4:
            urgency = Urgency.MEDIUM
        else:
            urgency = Urgency.LOW
        
        # Create reason
        reason_parts = [
            f"{entity['type'].title()} news",
            f"sentiment: {sentiment_score:+.2f}",
            f"entity: {entity['name']}",
            f"mentions: {entity['mentions']}"
        ]
        reason = " | ".join(reason_parts)
        
        # Get article ID
        article_id = getattr(article, 'url', getattr(article, 'id', 'unknown'))
        
        return self.create_signal(
            symbol=entity['symbol'],
            direction=direction,
            confidence=confidence,
            reason=reason,
            urgency=urgency,
            article_id=article_id,
            entity_name=entity['name'],
            entity_type=entity['type'],
            sentiment_score=sentiment_score,
            mentions=entity['mentions']
        )
    
    def _get_market_boost(self, context: StrategyContext) -> float:
        """Get confidence boost based on market conditions."""
        boost = 0.0
        
        # Regime-based boost
        if context.market_regime == MarketRegime.RISK_ON:
            boost += 0.1
        elif context.market_regime == MarketRegime.RISK_OFF:
            boost -= 0.05
        
        # Volatility adjustment
        volatility = context.metadata.get("volatility", 0.02)
        if volatility < 0.02:  # Low volatility = more confidence
            boost += 0.05
        elif volatility > 0.05:  # High volatility = less confidence
            boost -= 0.05
        
        # Trend strength boost
        trend_strength = context.metadata.get("trend_strength", 0.0)
        boost += trend_strength * 0.05
        
        return max(-0.1, min(0.1, boost))
    
    def _is_regime_compatible(self, signal: Signal, regime: MarketRegime) -> bool:
        """Check if signal is compatible with market regime."""
        # News sentiment strategy works in all regimes but adjusts confidence
        return True
    
    def should_execute(self, context: StrategyContext) -> bool:
        """Determine if strategy should execute."""
        if not super().should_execute(context):
            return False
        
        # Check if we have news data
        if not context.news_data:
            return False
        
        # Check market conditions
        if context.market_session in [MarketSession.CLOSED]:
            return False
        
        return True
