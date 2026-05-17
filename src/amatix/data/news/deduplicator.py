"""News deduplication engine.

Detects and filters duplicate news articles using:
    - Exact hash matching
    - Fuzzy title matching
    - Content similarity
    - URL normalization
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from amatix.core.observability import get_logger, get_metrics
from amatix.data.news.models import (
    DuplicateCheckResult,
    NewsArticle,
    SourceCredibility,
)

logger = get_logger(__name__)


@dataclass
class SimilarityScore:
    """Similarity between two articles."""
    article_id: UUID
    score: float  # 0.0 to 1.0
    method: str


class NewsDeduplicator:
    """Deduplicates news articles using multiple methods.
    
    Checks articles against a memory of recently seen articles
    and flags duplicates with confidence scores.
    
    Methods (in order):
        1. Exact URL match
        2. Content hash match
        3. Fuzzy title similarity
        4. Semantic similarity (placeholder for future ML)
    
    Example:
        >>> dedup = NewsDeduplicator()
        >>> 
        >>> # Check if duplicate
        >>> result = await dedup.check(article)
        >>> if result.is_duplicate:
        ...     print(f"Duplicate of {result.original_article_id}")
    """
    
    def __init__(
        self,
        max_memory_hours: float = 24.0,
        title_similarity_threshold: float = 0.85,
        content_similarity_threshold: float = 0.90,
    ) -> None:
        """Initialize deduplicator.
        
        Args:
            max_memory_hours: How long to remember articles
            title_similarity_threshold: Threshold for title match
            content_similarity_threshold: Threshold for content match
        """
        self._max_memory_hours = max_memory_hours
        self._title_threshold = title_similarity_threshold
        self._content_threshold = content_similarity_threshold
        
        # Memory stores
        self._url_index: Dict[str, UUID] = {}
        self._hash_index: Dict[str, UUID] = {}
        self._title_index: Dict[str, Tuple[UUID, datetime]] = {}
        self._article_memory: Dict[UUID, NewsArticle] = {}
        
        # Stats
        self._duplicates_found = 0
        self._articles_checked = 0
    
    async def check(self, article: NewsArticle) -> DuplicateCheckResult:
        """Check if article is a duplicate.
        
        Args:
            article: Article to check
        
        Returns:
            DuplicateCheckResult with similarity info
        """
        self._articles_checked += 1
        
        # 1. Exact URL match
        if article.url in self._url_index:
            original_id = self._url_index[article.url]
            self._duplicates_found += 1
            get_metrics().counter("news_exact_duplicates")
            return DuplicateCheckResult(
                is_duplicate=True,
                original_article_id=original_id,
                similarity_score=1.0,
                method="url",
            )
        
        # 2. Content hash match
        content_hash = self._compute_hash(article)
        if content_hash in self._hash_index:
            original_id = self._hash_index[content_hash]
            self._duplicates_found += 1
            get_metrics().counter("news_hash_duplicates")
            return DuplicateCheckResult(
                is_duplicate=True,
                original_article_id=original_id,
                similarity_score=1.0,
                method="hash",
            )
        
        # 3. Fuzzy title matching
        normalized_title = self._normalize_title(article.title)
        similar = await self._find_similar_title(
            normalized_title,
            article.published_at,
        )
        
        if similar and similar.score >= self._title_threshold:
            self._duplicates_found += 1
            get_metrics().counter("news_title_duplicates")
            return DuplicateCheckResult(
                is_duplicate=True,
                original_article_id=similar.article_id,
                similarity_score=similar.score,
                method="title",
            )
        
        # Not a duplicate - add to memory
        await self._add_to_memory(article, content_hash)
        
        # Clean old entries periodically
        if self._articles_checked % 100 == 0:
            await self._cleanup_old_entries()
        
        return DuplicateCheckResult(
            is_duplicate=False,
            similarity_score=0.0,
            method="none",
        )
    
    def _compute_hash(self, article: NewsArticle) -> str:
        """Compute content hash for article.
        
        Uses title + first 500 chars of body for fingerprinting.
        """
        content = article.title + " " + (article.body or "")[:500]
        
        # Normalize: lowercase, remove extra spaces
        content = content.lower()
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison.
        
        Removes:
            - Punctuation
            - Extra whitespace
            - Common prefixes (Breaking, Exclusive, etc.)
        """
        # Lowercase
        title = title.lower()
        
        # Remove common prefixes
        prefixes = [
            "breaking:",
            "breaking news:",
            "exclusive:",
            "update:",
            "just in:",
            "alert:",
        ]
        for prefix in prefixes:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        # Remove punctuation
        title = re.sub(r'[^\w\s]', '', title)
        
        # Normalize whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Compute Jaccard similarity between two strings.
        
        Uses word sets for comparison.
        """
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _find_similar_title(
        self,
        normalized_title: str,
        published_at: datetime,
    ) -> Optional[SimilarityScore]:
        """Find similar titles in memory.
        
        Only checks articles within time window.
        """
        time_window = timedelta(hours=self._max_memory_hours)
        
        best_match: Optional[SimilarityScore] = None
        
        for title, (article_id, timestamp) in self._title_index.items():
            # Check time window
            if abs(published_at - timestamp) > time_window:
                continue
            
            # Calculate similarity
            similarity = self._jaccard_similarity(normalized_title, title)
            
            if similarity >= self._title_threshold:
                if not best_match or similarity > best_match.score:
                    best_match = SimilarityScore(
                        article_id=article_id,
                        score=similarity,
                        method="title_jaccard",
                    )
        
        return best_match
    
    async def _add_to_memory(
        self,
        article: NewsArticle,
        content_hash: str,
    ) -> None:
        """Add article to deduplication memory."""
        # URL index
        self._url_index[article.url] = article.article_id
        
        # Hash index
        self._hash_index[content_hash] = article.article_id
        
        # Title index
        normalized_title = self._normalize_title(article.title)
        self._title_index[normalized_title] = (
            article.article_id,
            article.published_at,
        )
        
        # Full article memory
        self._article_memory[article.article_id] = article
    
    async def _cleanup_old_entries(self) -> None:
        """Remove entries older than max_memory_hours."""
        cutoff = datetime.utcnow() - timedelta(hours=self._max_memory_hours)
        
        # Clean title index
        expired_titles = [
            title for title, (_, timestamp) in self._title_index.items()
            if timestamp < cutoff
        ]
        for title in expired_titles:
            del self._title_index[title]
        
        # Clean article memory
        expired_ids = [
            aid for aid, article in self._article_memory.items()
            if article.published_at < cutoff
        ]
        for aid in expired_ids:
            del self._article_memory[aid]
        
        if expired_ids or expired_titles:
            logger.debug(
                "Deduplicator cleanup",
                articles_removed=len(expired_ids),
                titles_removed=len(expired_titles),
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplicator statistics."""
        return {
            "articles_checked": self._articles_checked,
            "duplicates_found": self._duplicates_found,
            "duplicate_rate": (
                self._duplicates_found / self._articles_checked
                if self._articles_checked > 0 else 0
            ),
            "memory_size": len(self._article_memory),
            "url_index_size": len(self._url_index),
        }
    
    def clear_memory(self) -> None:
        """Clear all deduplication memory."""
        self._url_index.clear()
        self._hash_index.clear()
        self._title_index.clear()
        self._article_memory.clear()
        
        logger.info("Deduplicator memory cleared")
