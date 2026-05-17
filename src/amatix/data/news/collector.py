"""News collector for RSS feed ingestion.

Collects news from multiple sources:
    - RSS feeds
    - NewsAPI (future)
    - Twitter/X (future)
    - Telegram (future)

Emits NewsArrivedEvent for each article.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
import feedparser

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import EventPriority, EventType
from amatix.core.observability import get_logger, get_metrics
from amatix.data.news.models import NewsArticle, SourceRating
from amatix.data.news.sources import FeedConfig, SourceRegistry

logger = get_logger(__name__)


class NewsCollector:
    """RSS news collector with async polling.
    
    Polls multiple RSS feeds at configured intervals and emits
    events for each article found.
    
    Features:
        - Async feed polling
        - Rate limiting
        - Duplicate filtering by URL
        - Event emission
        - Error handling
    
    Example:
        >>> collector = NewsCollector(event_bus)
        >>> await collector.start()
        >>> # Runs indefinitely, emitting NewsArrivedEvent
        >>> await collector.stop()
    """
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        source_registry: Optional[SourceRegistry] = None,
        http_timeout: float = 30.0,
    ) -> None:
        """Initialize news collector.
        
        Args:
            event_bus: Event bus for article emission
            source_registry: Source registry (creates default if None)
            http_timeout: HTTP request timeout
        """
        self._event_bus = event_bus
        self._registry = source_registry or SourceRegistry()
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        self._http_timeout = http_timeout
        
        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Deduplication
        self._seen_urls: set[str] = set()
        self._max_seen_urls = 100000
        
        # Metrics
        self._articles_collected = 0
        self._feeds_polled = 0
    
    async def start(self) -> None:
        """Start collecting from all configured feeds."""
        if self._running:
            return
        
        self._running = True
        
        # Create HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._http_timeout),
            headers={
                "User-Agent": "AMATIS News Collector/0.1",
            },
        )
        
        # Start polling tasks for each feed
        feeds = self._registry.get_feeds(enabled_only=True)
        
        for feed in feeds:
            task = asyncio.create_task(
                self._poll_loop(feed),
                name=f"poll_{feed.name}",
            )
            self._tasks.append(task)
        
        logger.info(
            "News collector started",
            feeds=len(feeds),
        )
    
    async def stop(self) -> None:
        """Stop all polling tasks."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for completion
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close session
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info(
            "News collector stopped",
            articles_collected=self._articles_collected,
        )
    
    async def _poll_loop(self, feed: FeedConfig) -> None:
        """Poll a single feed indefinitely."""
        while self._running:
            try:
                await self._poll_feed(feed)
                
            except Exception as e:
                logger.error(
                    "Feed poll error",
                    feed=feed.name,
                    error=str(e),
                )
                get_metrics().counter("news_poll_errors", labels={"feed": feed.name})
            
            # Wait before next poll
            await asyncio.sleep(feed.poll_interval_seconds)
    
    async def _poll_feed(self, feed: FeedConfig) -> None:
        """Poll a single RSS feed."""
        if not self._session:
            return
        
        logger.debug("Polling feed", feed=feed.name, url=feed.url)
        
        # Fetch feed
        async with self._session.get(feed.url) as response:
            response.raise_for_status()
            content = await response.text()
        
        # Parse RSS
        parsed = feedparser.parse(content)
        
        self._feeds_polled += 1
        
        # Process entries
        articles: List[NewsArticle] = []
        
        for entry in parsed.entries:
            # Skip if already seen
            url = entry.get("link", "")
            if not url:
                continue
            
            if url in self._seen_urls:
                continue
            
            # Create article
            article = self._create_article(entry, feed.rating)
            
            if article:
                articles.append(article)
                self._seen_urls.add(url)
                
                # Maintain seen URL cache size
                if len(self._seen_urls) > self._max_seen_urls:
                    # Clear half of cache
                    self._seen_urls = set(list(self._seen_urls)[self._max_seen_urls//2:])
        
        # Emit articles
        for article in articles:
            await self._emit_article(article, feed.name)
        
        logger.debug(
            "Feed polled",
            feed=feed.name,
            new_articles=len(articles),
            total_entries=len(parsed.entries),
        )
        
        get_metrics().counter(
            "news_articles_collected",
            value=len(articles),
            labels={"feed": feed.name},
        )
    
    def _create_article(
        self,
        entry: Dict[str, Any],
        rating: SourceRating,
    ) -> Optional[NewsArticle]:
        """Create NewsArticle from RSS entry."""
        try:
            url = entry.get("link", "")
            title = entry.get("title", "").strip()
            
            if not title:
                return None
            
            # Parse published date
            published = datetime.utcnow()
            if "published_parsed" in entry:
                import time
                published = datetime(*entry.published_parsed[:6])
            elif "updated_parsed" in entry:
                import time
                published = datetime(*entry.updated_parsed[:6])
            
            # Get summary/body
            summary = entry.get("summary", "")
            body = entry.get("content", [{}])[0].get("value", "") if "content" in entry else None
            
            # Use summary if no body
            body = body or summary
            
            # Create article
            article = NewsArticle.create(
                url=url,
                source=rating.name,
                title=title,
                body=body,
                published_at=published,
                source_rating=rating,
            )
            
            return article
            
        except Exception as e:
            logger.warning(
                "Failed to create article from entry",
                error=str(e),
            )
            return None
    
    async def _emit_article(
        self,
        article: NewsArticle,
        feed_name: str,
    ) -> None:
        """Emit article to event bus."""
        await self._event_bus.emit_new(
            EventType.NEWS_ARRIVED,
            {
                "article_id": str(article.article_id),
                "url": article.url,
                "source": article.source,
                "title": article.title,
                "published_at": article.published_at.isoformat(),
            },
            priority=EventPriority.NORMAL,
            source="news_collector",
            correlation_id=feed_name,
        )
        
        self._articles_collected += 1
    
    async def collect_single(
        self,
        url: str,
        rating: Optional[SourceRating] = None,
    ) -> List[NewsArticle]:
        """Manually collect from a single URL.
        
        For testing or one-off collections.
        """
        if not self._session:
            raise RuntimeError("Collector not started")
        
        async with self._session.get(url) as response:
            response.raise_for_status()
            content = await response.text()
        
        parsed = feedparser.parse(content)
        
        articles = []
        for entry in parsed.entries:
            rating = rating or SourceRating(
                name=urlparse(url).netloc,
                url=url,
                credibility=SourceCredibility.UNKNOWN,
                reliability_score=0.5,
            )
            
            article = self._create_article(entry, rating)
            if article:
                articles.append(article)
        
        return articles
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "running": self._running,
            "feeds": len(self._registry.get_feeds()),
            "articles_collected": self._articles_collected,
            "feeds_polled": self._feeds_polled,
            "seen_urls_cache_size": len(self._seen_urls),
        }
