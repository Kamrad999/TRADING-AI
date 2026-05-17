"""News article validation and credibility scoring.

Scores articles on:
    - Source credibility
    - Content quality
    - Spam probability
    - Relevance to trading
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from amatix.core.observability import get_logger, get_metrics
from amatix.data.news.models import (
    ArticleCategory,
    NewsArticle,
    SourceCredibility,
)

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of article validation."""
    is_valid: bool
    credibility_score: float  # 0.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    spam_score: float  # 0.0 to 1.0 (lower is better)
    category: Optional[ArticleCategory] = None
    tickers: List[str] = None
    rejection_reasons: List[str] = None
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = []
        if self.rejection_reasons is None:
            self.rejection_reasons = []


class NewsValidator:
    """Validates and scores news articles.
    
    Performs multi-layer validation:
        1. Source credibility check
        2. Content quality analysis
        3. Spam detection
        4. Trading relevance scoring
        5. Ticker extraction
    
    Example:
        >>> validator = NewsValidator()
        >>> result = await validator.validate(article)
        >>> if result.is_valid:
        ...     print(f"Credibility: {result.credibility_score}")
    """
    
    # Trading-relevant keywords
    TRADING_KEYWORDS: Set[str] = {
        "earnings", "revenue", "profit", "loss", "guidance",
        "merger", "acquisition", "buyout", "deal",
        "fed", "interest rate", "inflation", "gdp", "jobs",
        "bitcoin", "crypto", "blockchain", "etf",
        "upgrade", "downgrade", "price target",
        "dividend", "split", "buyback",
        "sec", "investigation", "lawsuit",
    }
    
    # Spam indicators
    SPAM_PATTERNS: List[str] = [
        r"click here",
        r"subscribe now",
        r"limited time",
        r"act now",
        r"\$\$\$",
        r"make money fast",
        r"guaranteed profit",
        r"100% free",
    ]
    
    # Ticker patterns
    TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')
    
    def __init__(self) -> None:
        """Initialize validator."""
        self._spam_regex = re.compile(
            '|'.join(self.SPAM_PATTERNS),
            re.IGNORECASE
        )
        
        # Common words that are not tickers
        self._non_tickers: Set[str] = {
            "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU",
            "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT",
            "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "ITS",
            "MAY", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY",
            "WHO", "BOY", "DID", "SHE", "USE", "HER", "EOD",
            "CEO", "CFO", "CTO", "COO", "IPO", "EPS", "GDP",
            "FED", "SEC", "ETF", "ESG", "AI", "ML", "API",
        }
    
    async def validate(self, article: NewsArticle) -> ValidationResult:
        """Validate and score an article.
        
        Args:
            article: Article to validate
        
        Returns:
            ValidationResult with scores and classification
        """
        rejection_reasons: List[str] = []
        
        # 1. Source credibility
        credibility = self._score_credibility(article)
        
        # 2. Content quality
        content_score = self._score_content_quality(article)
        
        # 3. Spam detection
        spam_score = self._detect_spam(article)
        if spam_score > 0.7:
            rejection_reasons.append("High spam probability")
        
        # 4. Relevance
        relevance, category = self._score_relevance(article)
        
        # 5. Extract tickers
        tickers = self._extract_tickers(article)
        
        # Determine validity
        is_valid = (
            credibility >= 0.3 and  # At least some credibility
            spam_score < 0.5 and     # Not spammy
            relevance >= 0.1         # Some relevance to trading
        )
        
        if credibility < 0.3:
            rejection_reasons.append("Low source credibility")
        if relevance < 0.1:
            rejection_reasons.append("Low trading relevance")
        
        result = ValidationResult(
            is_valid=is_valid,
            credibility_score=credibility,
            relevance_score=relevance,
            spam_score=spam_score,
            category=category,
            tickers=tickers,
            rejection_reasons=rejection_reasons,
        )
        
        # Emit metrics
        get_metrics().counter(
            "news_validated",
            labels={
                "source": article.source,
                "valid": str(is_valid),
                "category": category.value if category else "unknown",
            },
        )
        
        return result
    
    def _score_credibility(self, article: NewsArticle) -> float:
        """Score source credibility (0.0 to 1.0)."""
        if not article.source_rating:
            return 0.3  # Unknown source
        
        rating = article.source_rating
        
        # Base credibility from tier
        tier_scores = {
            SourceCredibility.TIER_1: 0.95,
            SourceCredibility.TIER_2: 0.75,
            SourceCredibility.TIER_3: 0.50,
            SourceCredibility.UNKNOWN: 0.30,
        }
        
        base_score = tier_scores.get(rating.credibility, 0.30)
        
        # Adjust by reliability score
        adjusted = (base_score + rating.reliability_score) / 2
        
        return adjusted
    
    def _score_content_quality(self, article: NewsArticle) -> float:
        """Score content quality."""
        score = 0.5
        
        # Check title length
        if len(article.title) < 20:
            score -= 0.1
        if len(article.title) > 150:
            score -= 0.05
        
        # Check body length
        if article.body:
            word_count = len(article.body.split())
            if word_count < 50:
                score -= 0.1
            if word_count > 100:
                score += 0.1
            
            # Store word count
            article.word_count = word_count
        
        # Check for all caps (shouting)
        if article.title.isupper():
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _detect_spam(self, article: NewsArticle) -> float:
        """Detect spam probability (0.0 to 1.0, higher = more spammy)."""
        text = f"{article.title} {article.body or ''}".lower()
        
        # Check spam patterns
        spam_matches = len(self._spam_regex.findall(text))
        
        # Base spam score
        spam_score = min(1.0, spam_matches * 0.2)
        
        # Check for excessive punctuation
        exclamation_count = text.count("!")
        if exclamation_count > 3:
            spam_score += 0.1
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.5:
            spam_score += 0.1
        
        return min(1.0, spam_score)
    
    def _score_relevance(
        self,
        article: NewsArticle,
    ) -> tuple[float, Optional[ArticleCategory]]:
        """Score trading relevance and detect category."""
        text = f"{article.title} {article.body or ''}".lower()
        
        # Count trading keywords
        keyword_matches = sum(
            1 for kw in self.TRADING_KEYWORDS if kw in text
        )
        
        # Base relevance score
        relevance = min(1.0, keyword_matches * 0.15)
        
        # Detect category
        category = self._detect_category(text)
        
        # Boost relevance for specific categories
        if category == ArticleCategory.EARNINGS:
            relevance = min(1.0, relevance + 0.3)
        elif category in [ArticleCategory.MACRO, ArticleCategory.M_A]:
            relevance = min(1.0, relevance + 0.2)
        
        return relevance, category
    
    def _detect_category(self, text: str) -> Optional[ArticleCategory]:
        """Detect article category from content."""
        text = text.lower()
        
        # Category patterns
        category_patterns = {
            ArticleCategory.EARNINGS: [
                "earnings", "eps", "revenue", "profit", "loss",
                "guidance", "quarterly results", "beat", "miss",
            ],
            ArticleCategory.MACRO: [
                "fed", "fomc", "interest rate", "inflation", "gdp",
                "jobs report", "unemployment", "cpi", "ppi", "nfp",
            ],
            ArticleCategory.CRYPTO: [
                "bitcoin", "ethereum", "crypto", "blockchain", "defi",
                "nft", "altcoin", "btc", "eth",
            ],
            ArticleCategory.M_A: [
                "merger", "acquisition", "buyout", "takeover",
                "deal", "acquired", "merging",
            ],
            ArticleCategory.REGULATORY: [
                "sec", "investigation", "lawsuit", "regulatory",
                "compliance", "fine", "settlement",
            ],
        }
        
        # Score each category
        best_category = None
        best_score = 0
        
        for category, patterns in category_patterns.items():
            score = sum(1 for p in patterns if p in text)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category if best_score > 0 else ArticleCategory.GENERAL
    
    def _extract_tickers(self, article: NewsArticle) -> List[str]:
        """Extract potential stock tickers from text."""
        text = f"{article.title} {article.body or ''}"
        
        # Find potential tickers
        matches = self.TICKER_PATTERN.findall(text)
        
        # Filter valid tickers
        tickers: Set[str] = set()
        for match in matches:
            if (
                len(match) >= 1 and
                len(match) <= 5 and
                match not in self._non_tickers and
                match.isalpha() and
                match.isupper()
            ):
                tickers.add(match)
        
        return sorted(tickers)
    
    def add_non_ticker(self, word: str) -> None:
        """Add a word to the non-ticker list."""
        self._non_tickers.add(word.upper())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return {
            "trading_keywords": len(self.TRADING_KEYWORDS),
            "spam_patterns": len(self.SPAM_PATTERNS),
            "non_ticker_words": len(self._non_tickers),
        }
