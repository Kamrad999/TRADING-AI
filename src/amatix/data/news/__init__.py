"""AMATIS News Data Pipeline.

Provides news ingestion, parsing, deduplication, and validation
for news-based signal generation.

Components:
    - collector: RSS feed collection
    - parser: Content extraction
    - deduplicator: Duplicate detection
    - validator: Credibility scoring
    - sources: Feed registry
    - models: Data structures
"""

from amatix.data.news.models import NewsArticle, SourceRating

__all__ = [
    "NewsArticle",
    "SourceRating",
]
