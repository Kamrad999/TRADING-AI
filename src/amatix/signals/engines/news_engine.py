"""News-based signal engine for AMATIS.

Generates signals from validated news articles using:
    - Pattern matching (earnings, M&A, macro)
    - Sentiment placeholders
    - Source credibility weighting
    - Ticker extraction

YAML-driven patterns for maintainability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.observability import get_logger, get_metrics
from amatix.data.market.models import Symbol
from amatix.data.news.models import ArticleCategory, NewsArticle
from amatix.signals.engines.base import BaseSignalEngine
from amatix.signals.models import (
    Signal,
    SignalDirection,
    SignalFeature,
    SignalStrength,
    SignalTimeframe,
)

logger = get_logger(__name__)


@dataclass
class PatternConfig:
    """Configuration for a news pattern."""
    name: str
    category: str
    keywords: List[str]
    direction: str  # "long", "short", "neutral"
    confidence_boost: float = 0.0
    description: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PatternConfig:
        return cls(
            name=data["name"],
            category=data["category"],
            keywords=data["keywords"],
            direction=data["direction"],
            confidence_boost=data.get("confidence_boost", 0.0),
            description=data.get("description", ""),
        )


class NewsSignalEngine(BaseSignalEngine):
    """News-based signal generation engine.
    
    Analyzes validated news articles and generates trading signals
    based on pattern matching and source credibility.
    
    Features:
        - YAML-driven pattern configuration
        - Category-based signal direction
        - Source credibility weighting
        - Sentiment-aware confidence (placeholder)
    
    Example:
        >>> engine = NewsSignalEngine(event_bus)
        >>> await engine.initialize(config)
        >>> 
        >>> context = {"articles": [article1, article2]}
        >>> signals = await engine.generate(context)
    """
    
    # Default patterns (can be overridden via YAML)
    DEFAULT_PATTERNS: Dict[str, PatternConfig] = {
        "earnings_beat": PatternConfig(
            name="earnings_beat",
            category="earnings",
            keywords=["beat", "beats", "surpass", "exceeds", "strong earnings"],
            direction="long",
            confidence_boost=0.15,
            description="Earnings beat expectations",
        ),
        "earnings_miss": PatternConfig(
            name="earnings_miss",
            category="earnings",
            keywords=["miss", "misses", "falls short", "disappointing", "weak earnings"],
            direction="short",
            confidence_boost=0.10,
            description="Earnings miss expectations",
        ),
        "upgrade": PatternConfig(
            name="analyst_upgrade",
            category="sentiment",
            keywords=["upgrade", "upgraded", "buy rating", "overweight"],
            direction="long",
            confidence_boost=0.12,
            description="Analyst upgrade",
        ),
        "downgrade": PatternConfig(
            name="analyst_downgrade",
            category="sentiment",
            keywords=["downgrade", "downgraded", "sell rating", "underweight"],
            direction="short",
            confidence_boost=0.10,
            description="Analyst downgrade",
        ),
        "ma_acquisition": PatternConfig(
            name="ma_acquisition",
            category="m_a",
            keywords=["acquisition", "acquires", "buyout", "to be acquired", "takeover"],
            direction="long",
            confidence_boost=0.20,
            description="Acquisition announced",
        ),
        "fed_hawkish": PatternConfig(
            name="fed_hawkish",
            category="macro",
            keywords=["rate hike", "hawkish", "tightening", "higher rates"],
            direction="short",
            confidence_boost=0.10,
            description="Fed hawkish stance",
        ),
        "fed_dovish": PatternConfig(
            name="fed_dovish",
            category="macro",
            keywords=["rate cut", "dovish", "easing", "lower rates"],
            direction="long",
            confidence_boost=0.10,
            description="Fed dovish stance",
        ),
        "crypto_bullish": PatternConfig(
            name="crypto_bullish",
            category="crypto",
            keywords=["bitcoin rally", "crypto surge", "adoption", "etf approval"],
            direction="long",
            confidence_boost=0.12,
            description="Bullish crypto news",
        ),
        "crypto_bearish": PatternConfig(
            name="crypto_bearish",
            category="crypto",
            keywords=["crypto crash", "bitcoin falls", "ban", "regulation crackdown"],
            direction="short",
            confidence_boost=0.10,
            description="Bearish crypto news",
        ),
    }
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        patterns_path: Optional[Path] = None,
    ) -> None:
        """Initialize news signal engine.
        
        Args:
            event_bus: Event bus
            patterns_path: Path to YAML patterns file
        """
        super().__init__(
            name="news",
            version="1.0.0",
            event_bus=event_bus,
        )
        
        # Load patterns
        self._patterns_path = patterns_path
        self._patterns: Dict[str, PatternConfig] = {}
        
        # Compile regex patterns
        self._compiled_patterns: Dict[str, re.Pattern] = {}
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize engine and load patterns."""
        await super().initialize(config)
        
        # Load patterns from YAML or use defaults
        if self._patterns_path and self._patterns_path.exists():
            await self._load_patterns_from_yaml()
        else:
            self._patterns = dict(self.DEFAULT_PATTERNS)
        
        # Compile patterns
        self._compile_patterns()
        
        logger.info(
            "News engine initialized",
            patterns=len(self._patterns),
        )
    
    async def _load_patterns_from_yaml(self) -> None:
        """Load patterns from YAML file."""
        try:
            with open(self._patterns_path) as f:
                data = yaml.safe_load(f)
            
            patterns_data = data.get("patterns", [])
            
            for pattern_data in patterns_data:
                config = PatternConfig.from_dict(pattern_data)
                self._patterns[config.name] = config
            
            logger.info(
                "Patterns loaded from YAML",
                path=str(self._patterns_path),
                count=len(self._patterns),
            )
        
        except Exception as e:
            logger.error(
                "Failed to load patterns from YAML, using defaults",
                error=str(e),
            )
            self._patterns = dict(self.DEFAULT_PATTERNS)
    
    def _compile_patterns(self) -> None:
        """Compile keyword patterns to regex."""
        for name, config in self._patterns.items():
            # Create regex from keywords
            escaped = [re.escape(kw) for kw in config.keywords]
            pattern_str = r'\b(' + '|'.join(escaped) + r')\b'
            self._compiled_patterns[name] = re.compile(
                pattern_str,
                re.IGNORECASE,
            )
    
    async def generate(self, context: Dict[str, Any]) -> List[Signal]:
        """Generate signals from news articles.
        
        Args:
            context: Dictionary with:
                - "articles": List[NewsArticle] validated articles
        
        Returns:
            List of news-based signals
        """
        articles = context.get("articles", [])
        
        if not articles:
            return []
        
        signals: List[Signal] = []
        
        for article in articles:
            # Only process actionable articles
            if not self._is_article_actionable(article):
                continue
            
            # Generate signals for this article
            article_signals = await self._process_article(article)
            signals.extend(article_signals)
        
        # Update stats
        for signal in signals:
            self._track_signal()
        
        get_metrics().counter(
            "news_signals_generated",
            value=len(signals),
        )
        
        return signals
    
    def _is_article_actionable(self, article: NewsArticle) -> bool:
        """Check if article is suitable for signal generation."""
        # Must have tickers
        if not article.tickers:
            return False
        
        # Must be valid
        if not hasattr(article, 'relevance_score') or article.relevance_score < 0.3:
            return False
        
        # Check credibility
        if article.credibility_score < 0.5:
            return False
        
        return True
    
    async def _process_article(self, article: NewsArticle) -> List[Signal]:
        """Process a single article and generate signals."""
        signals: List[Signal] = []
        
        # Match patterns
        text = f"{article.title} {article.body or ''}".lower()
        
        matched_patterns: List[PatternConfig] = []
        
        for pattern_name, pattern_config in self._patterns.items():
            compiled = self._compiled_patterns.get(pattern_name)
            if not compiled:
                continue
            
            if compiled.search(text):
                matched_patterns.append(pattern_config)
        
        if not matched_patterns:
            return []
        
        # Generate signals for each ticker
        for ticker in article.tickers[:3]:  # Max 3 tickers per article
            # Create symbol
            symbol = Symbol(
                base=ticker,
                exchange="NASDAQ",  # Default, could be smarter
                asset_class="equity",
            )
            
            # Generate signal from best matching pattern
            best_pattern = self._select_best_pattern(matched_patterns)
            
            signal = self._create_signal_from_pattern(
                symbol=symbol,
                pattern=best_pattern,
                article=article,
            )
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _select_best_pattern(
        self,
        patterns: List[PatternConfig],
    ) -> PatternConfig:
        """Select the best pattern from matches."""
        # Sort by confidence boost
        return max(patterns, key=lambda p: p.confidence_boost)
    
    def _create_signal_from_pattern(
        self,
        symbol: Symbol,
        pattern: PatternConfig,
        article: NewsArticle,
    ) -> Optional[Signal]:
        """Create a signal from matched pattern."""
        # Determine direction
        direction_map = {
            "long": SignalDirection.LONG,
            "short": SignalDirection.SHORT,
            "neutral": SignalDirection.NEUTRAL,
        }
        
        direction = direction_map.get(pattern.direction)
        if not direction:
            return None
        
        # Calculate confidence
        base_confidence = 0.55
        
        # Add pattern boost
        confidence = base_confidence + pattern.confidence_boost
        
        # Adjust by source credibility
        confidence += article.credibility_score * 0.15
        
        # Adjust by relevance
        confidence += article.relevance_score * 0.10
        
        # Cap at 0.95
        confidence = min(0.95, confidence)
        
        # Determine strength
        if confidence >= 0.85:
            strength = SignalStrength.STRONG
        elif confidence >= 0.70:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Create features
        features = [
            SignalFeature(
                name="pattern_match",
                value=pattern.name,
                weight=0.8,
                category="fundamental",
                description=pattern.description,
            ),
            SignalFeature(
                name="source_credibility",
                value=round(article.credibility_score, 2),
                weight=0.4,
                category="fundamental",
            ),
            SignalFeature(
                name="article_relevance",
                value=round(article.relevance_score, 2),
                weight=0.3,
                category="fundamental",
            ),
            SignalFeature(
                name="matched_keywords",
                value=len(pattern.keywords),
                weight=0.2,
                category="fundamental",
            ),
        ]
        
        # Create signal
        signal = Signal.create(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            source=self.name,
            trace_id=article.article_id,  # Link to article
            strength=strength,
            timeframe=SignalTimeframe.SWING,  # News typically swing trades
            source_version=self.version,
        )
        
        # Add features and rationale
        signal = signal.with_features(features)
        signal.rationale = (
            f"News pattern match: {pattern.description} | "
            f"Source: {article.source} | "
            f"Keywords: {', '.join(pattern.keywords[:3])}"
        )
        signal.tags = ["news", pattern.category, article.category.value if article.category else "general"]
        
        # Link to article
        signal.metadata["article_id"] = str(article.article_id)
        signal.metadata["article_url"] = article.url
        
        return signal
    
    async def health_check(self) -> Dict[str, Any]:
        """Engine health check."""
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "name": self.name,
            "version": self.version,
            "patterns_loaded": len(self._patterns),
            "signals_generated": self._signals_generated,
            "pattern_names": list(self._patterns.keys())[:5],
        }
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get pattern statistics."""
        return {
            name: {
                "category": config.category,
                "direction": config.direction,
                "keywords": len(config.keywords),
            }
            for name, config in self._patterns.items()
        }
