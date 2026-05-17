"""News data models for AMATIS.

Immutable dataclasses for news articles with:
    - Content metadata
    - Source credibility
    - Extracted tickers
    - Sentiment placeholders
    - Attribution tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import whenever


class ArticleCategory(Enum):
    """Category of news article."""
    EARNINGS = "earnings"
    MACRO = "macro"
    CRYPTO = "crypto"
    M_A = "m_a"
    REGULATORY = "regulatory"
    COMPANY_NEWS = "company_news"
    GENERAL = "general"


class SourceCredibility(Enum):
    """Credibility tier for news sources."""
    TIER_1 = 1  # Bloomberg, Reuters, WSJ
    TIER_2 = 2  # Established financial media
    TIER_3 = 3  # Blogs, aggregators
    UNKNOWN = 4


@dataclass(frozen=True)
class SourceRating:
    """Rating for a news source."""
    name: str
    url: str
    credibility: SourceCredibility
    reliability_score: float  # 0.0 to 1.0
    average_lag_seconds: Optional[float] = None  # Delay vs primary sources
    bias_indicator: Optional[str] = None  # "bullish", "bearish", "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "credibility": self.credibility.name,
            "reliability_score": self.reliability_score,
        }


@dataclass(frozen=True)
class ExtractedEntity:
    """Entity extracted from article text."""
    text: str
    entity_type: str  # "ticker", "company", "person", "crypto"
    start_pos: int
    end_pos: int
    confidence: float = 1.0


@dataclass
class NewsArticle:
    """News article with full metadata.
    
    Immutable once created. Represents a single news item
    at a point in time with all extracted features.
    """
    # Identity
    article_id: UUID
    url: str
    source: str
    
    # Content
    title: str
    body: Optional[str] = None
    summary: Optional[str] = None
    
    # Timestamps
    published_at: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    collected_at: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    
    # Source info
    source_rating: Optional[SourceRating] = None
    category: Optional[ArticleCategory] = None
    
    # Extracted data
    tickers: List[str] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    
    # Scores
    credibility_score: float = 0.0  # 0.0 to 1.0
    relevance_score: float = 0.0  # 0.0 to 1.0
    sentiment_score: Optional[float] = None  # -1.0 to 1.0 (placeholder for ML)
    
    # Duplicate detection
    content_hash: Optional[str] = None
    duplicate_of: Optional[UUID] = None
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    language: str = "en"
    word_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        url: str,
        source: str,
        title: str,
        body: Optional[str] = None,
        published_at: Optional[datetime] = None,
        source_rating: Optional[SourceRating] = None,
    ) -> NewsArticle:
        """Factory method to create a new article."""
        return cls(
            article_id=uuid4(),
            url=url,
            source=source,
            title=title,
            body=body,
            published_at=published_at or whenever.now().py_datetime(),
            collected_at=whenever.now().py_datetime(),
            source_rating=source_rating,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "article_id": str(self.article_id),
            "url": self.url,
            "source": self.source,
            "title": self.title,
            "published_at": self.published_at.isoformat(),
            "collected_at": self.collected_at.isoformat(),
            "tickers": self.tickers,
            "credibility_score": self.credibility_score,
            "relevance_score": self.relevance_score,
            "sentiment_score": self.sentiment_score,
            "category": self.category.value if self.category else None,
        }
    
    def with_tickers(self, tickers: List[str]) -> NewsArticle:
        """Return new article with tickers added."""
        # Create copy with new tickers
        new_article = NewsArticle(
            article_id=self.article_id,
            url=self.url,
            source=self.source,
            title=self.title,
            body=self.body,
            summary=self.summary,
            published_at=self.published_at,
            collected_at=self.collected_at,
            source_rating=self.source_rating,
            category=self.category,
            tickers=tickers,
            entities=self.entities,
            credibility_score=self.credibility_score,
            relevance_score=self.relevance_score,
            sentiment_score=self.sentiment_score,
            content_hash=self.content_hash,
            duplicate_of=self.duplicate_of,
            similarity_scores=self.similarity_scores,
            language=self.language,
            word_count=self.word_count,
            metadata=self.metadata,
        )
        return new_article
    
    def with_scores(
        self,
        credibility: float,
        relevance: float,
    ) -> NewsArticle:
        """Return new article with updated scores."""
        new_article = NewsArticle(
            article_id=self.article_id,
            url=self.url,
            source=self.source,
            title=self.title,
            body=self.body,
            summary=self.summary,
            published_at=self.published_at,
            collected_at=self.collected_at,
            source_rating=self.source_rating,
            category=self.category,
            tickers=self.tickers,
            entities=self.entities,
            credibility_score=credibility,
            relevance_score=relevance,
            sentiment_score=self.sentiment_score,
            content_hash=self.content_hash,
            duplicate_of=self.duplicate_of,
            similarity_scores=self.similarity_scores,
            language=self.language,
            word_count=self.word_count,
            metadata=self.metadata,
        )
        return new_article


@dataclass
class NewsBatch:
    """Batch of articles for processing."""
    articles: List[NewsArticle]
    source: str
    collected_at: datetime = field(default_factory=lambda: whenever.now().py_datetime())
    
    @property
    def count(self) -> int:
        return len(self.articles)
    
    @property
    def unique_tickers(self) -> Set[str]:
        """Get all unique tickers in batch."""
        tickers: Set[str] = set()
        for article in self.articles:
            tickers.update(article.tickers)
        return tickers


@dataclass
class DuplicateCheckResult:
    """Result of duplicate detection."""
    is_duplicate: bool
    original_article_id: Optional[UUID] = None
    similarity_score: float = 0.0
    method: str = "unknown"  # "hash", "title", "semantic"
