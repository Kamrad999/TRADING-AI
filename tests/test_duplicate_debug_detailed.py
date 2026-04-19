#!/usr/bin/env python3
"""
Detailed duplicate filter debugging to understand 100% duplicate rate.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.validation.duplicate_filter import DuplicateFilter
from trading_ai.core.models import Article
from trading_ai.core.orchestrator import PipelineOrchestrator


def debug_duplicate_filter_detailed():
    """Debug duplicate filter with detailed logging."""
    print("🔍 DETAILED DUPLICATE FILTER DEBUG")
    print("=" * 60)
    
    # Clear ALL state
    data_dir = Path("./data")
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir()
    print("🗑️  All state cleared")
    
    # Test 1: Manual articles
    print("\n📝 TEST 1: Manual articles")
    filter = DuplicateFilter()
    
    manual_articles = [
        Article(
            title=f"Test Article {i}",
            content=f"Content for article {i}",
            source=f"Source {i}",
            timestamp=datetime.now(timezone.utc),
            url=f"https://test{i}.com/article{i}",
            metadata={}
        )
        for i in range(5)
    ]
    
    print(f"   Processing {len(manual_articles)} manual articles...")
    unique_manual = filter.filter_duplicates(manual_articles)
    print(f"   Result: {len(unique_manual)} unique, {len(manual_articles) - len(unique_manual)} duplicates")
    
    # Test 2: Single RSS fetch
    print("\n📡 TEST 2: Single RSS fetch")
    filter2 = DuplicateFilter()  # Fresh instance
    
    try:
        orchestrator = PipelineOrchestrator()
        
        # Get articles from just one source
        sources = list(orchestrator.source_registry.sources.values())
        if sources:
            source = sources[0]  # First source only
            print(f"   Fetching from: {source.name}")
            articles, metadata = orchestrator.news_collector.fetch_feed(source.url)
            print(f"   Fetched {len(articles)} articles")
            
            if articles:
                print(f"   First article: {articles[0].title[:50]}...")
                print(f"   First URL: {articles[0].url}")
                
                # Test duplicate filtering
                unique_rss = filter2.filter_duplicates(articles)
                print(f"   Result: {len(unique_rss)} unique, {len(articles) - len(unique_rss)} duplicates")
                
                # Show first few duplicates if any
                if len(unique_rss) < len(articles):
                    print(f"   ❌ DUPLICATES DETECTED!")
                    for i, article in enumerate(articles[:3]):
                        is_duplicate = filter2._is_duplicate(article)
                        print(f"      Article {i+1}: {is_duplicate}")
                        print(f"         Title: {article.title[:40]}...")
                        print(f"         URL: {article.url}")
                        
                        # Check URL hash
                        url_hash = filter2._generate_url_hash(article.url)
                        print(f"         URL hash: {url_hash}")
                        print(f"         Hash exists: {url_hash in filter2.url_hashes}")
                        
                        if url_hash in filter2.url_hashes:
                            print(f"         ❌ URL HASH DUPLICATE!")
                        else:
                            # Check title similarity
                            for seen_id, seen_data in filter2.seen_articles.items():
                                seen_title = filter2._normalize_title(seen_data["title"])
                                article_title = filter2._normalize_title(article.title)
                                from difflib import SequenceMatcher
                                similarity = SequenceMatcher(None, article_title, seen_title).ratio()
                                if similarity >= filter2.title_similarity_threshold:
                                    print(f"         ❌ TITLE SIMILARITY: {similarity:.3f} (threshold: {filter2.title_similarity_threshold})")
                                    print(f"         Seen title: {seen_title[:40]}...")
                                    break
            else:
                print("   No articles fetched")
                
    except Exception as e:
        print(f"   ❌ RSS fetch failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Lower thresholds
    print("\n🎛️  TEST 3: Lower thresholds")
    filter3 = DuplicateFilter()
    filter3.title_similarity_threshold = 0.8  # Lower threshold
    filter3.timestamp_window_hours = 1  # Shorter window
    
    print(f"   Thresholds: similarity={filter3.title_similarity_threshold}, window={filter3.timestamp_window_hours}h")
    
    # Test with same articles again
    unique_lowered = filter3.filter_duplicates(manual_articles)
    print(f"   Result: {len(unique_lowered)} unique, {len(manual_articles) - len(unique_lowered)} duplicates")
    
    return len(unique_manual) > 0 or len(unique_rss) > 0 or len(unique_lowered) > 0


if __name__ == "__main__":
    success = debug_duplicate_filter_detailed()
    
    if success:
        print("\n✅ DUPLICATE FILTER DEBUG COMPLETED")
        print("   System can generate unique articles")
        sys.exit(0)
    else:
        print("\n❌ DUPLICATE FILTER DEBUG FAILED")
        print("   System cannot generate unique articles")
        sys.exit(1)
