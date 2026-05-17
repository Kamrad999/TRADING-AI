"""News source registry and ratings.

Manages source credibility ratings and feed configurations.
Pre-populated with major financial news sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from amatix.data.news.models import SourceCredibility, SourceRating


@dataclass
class FeedConfig:
    """Configuration for a news feed."""
    name: str
    url: str
    rating: SourceRating
    poll_interval_seconds: int = 300
    priority: int = 50
    enabled: bool = True


class SourceRegistry:
    """Registry of news sources with credibility ratings."""
    
    # Pre-defined source ratings
    SOURCE_RATINGS: Dict[str, SourceRating] = {
        # Tier 1: Premium news services
        "bloomberg": SourceRating(
            name="Bloomberg",
            url="https://www.bloomberg.com",
            credibility=SourceCredibility.TIER_1,
            reliability_score=0.95,
        ),
        "reuters": SourceRating(
            name="Reuters",
            url="https://www.reuters.com",
            credibility=SourceCredibility.TIER_1,
            reliability_score=0.95,
        ),
        "wsj": SourceRating(
            name="Wall Street Journal",
            url="https://www.wsj.com",
            credibility=SourceCredibility.TIER_1,
            reliability_score=0.93,
        ),
        "ft": SourceRating(
            name="Financial Times",
            url="https://www.ft.com",
            credibility=SourceCredibility.TIER_1,
            reliability_score=0.92,
        ),
        "cnbc": SourceRating(
            name="CNBC",
            url="https://www.cnbc.com",
            credibility=SourceCredibility.TIER_2,
            reliability_score=0.85,
        ),
        "marketwatch": SourceRating(
            name="MarketWatch",
            url="https://www.marketwatch.com",
            credibility=SourceCredibility.TIER_2,
            reliability_score=0.82,
        ),
        "seeking_alpha": SourceRating(
            name="Seeking Alpha",
            url="https://seekingalpha.com",
            credibility=SourceCredibility.TIER_2,
            reliability_score=0.78,
            bias_indicator="bullish",
        ),
        "yahoo_finance": SourceRating(
            name="Yahoo Finance",
            url="https://finance.yahoo.com",
            credibility=SourceCredibility.TIER_2,
            reliability_score=0.80,
        ),
        "investopedia": SourceRating(
            name="Investopedia",
            url="https://www.investopedia.com",
            credibility=SourceCredibility.TIER_2,
            reliability_score=0.75,
        ),
        "coindesk": SourceRating(
            name="CoinDesk",
            url="https://www.coindesk.com",
            credibility=SourceCredibility.TIER_2,
            reliability_score=0.80,
        ),
        "cointelegraph": SourceRating(
            name="Cointelegraph",
            url="https://cointelegraph.com",
            credibility=SourceCredibility.TIER_2,
            reliability_score=0.78,
        ),
    }
    
    # Default RSS feeds
    DEFAULT_FEEDS: List[FeedConfig] = [
        FeedConfig(
            name="Yahoo Finance Top Stories",
            url="https://finance.yahoo.com/rss/topstories",
            rating=SOURCE_RATINGS["yahoo_finance"],
            poll_interval_seconds=300,
            priority=40,
        ),
        FeedConfig(
            name="MarketWatch Top Stories",
            url="https://www.marketwatch.com/rss/topstories",
            rating=SOURCE_RATINGS["marketwatch"],
            poll_interval_seconds=300,
            priority=50,
        ),
        FeedConfig(
            name="CoinDesk News",
            url="https://www.coindesk.com/arc/outboundfeeds/rss/",
            rating=SOURCE_RATINGS["coindesk"],
            poll_interval_seconds=300,
            priority=60,
        ),
    ]
    
    def __init__(self) -> None:
        """Initialize source registry."""
        self._sources = dict(self.SOURCE_RATINGS)
        self._feeds = list(self.DEFAULT_FEEDS)
    
    def get_rating(self, source_name: str) -> Optional[SourceRating]:
        """Get rating for a source by name."""
        key = source_name.lower().replace(" ", "_")
        return self._sources.get(key)
    
    def add_rating(self, rating: SourceRating) -> None:
        """Add a new source rating."""
        key = rating.name.lower().replace(" ", "_")
        self._sources[key] = rating
    
    def get_feeds(self, enabled_only: bool = True) -> List[FeedConfig]:
        """Get all configured feeds."""
        if enabled_only:
            return [f for f in self._feeds if f.enabled]
        return self._feeds
    
    def add_feed(self, config: FeedConfig) -> None:
        """Add a new feed configuration."""
        self._feeds.append(config)
    
    def disable_feed(self, name: str) -> bool:
        """Disable a feed by name."""
        for feed in self._feeds:
            if feed.name == name:
                feed.enabled = False
                return True
        return False
    
    def list_sources(self) -> List[str]:
        """List all registered source names."""
        return list(self._sources.keys())
    
    def get_tier_1_sources(self) -> List[SourceRating]:
        """Get all Tier 1 (premium) sources."""
        return [
            s for s in self._sources.values()
            if s.credibility == SourceCredibility.TIER_1
        ]


# Global registry
_global_registry: Optional[SourceRegistry] = None


def get_source_registry() -> SourceRegistry:
    """Get the global source registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SourceRegistry()
    return _global_registry
