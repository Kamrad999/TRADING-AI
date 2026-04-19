#!/usr/bin/env python3
"""
Debug duplicate filter behavior to understand 100% duplicate rate.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.validation.duplicate_filter import DuplicateFilter
from trading_ai.core.models import Article


def debug_duplicate_filter():
    """Debug duplicate filter with real RSS data."""
    print("🔍 DEBUGGING DUPLICATE FILTER")
    print("=" * 50)
    
    # Clear state
    import os
    state_file = Path("./data/state.json")
    if state_file.exists():
        os.remove(state_file)
        print("🗑️  Cleared existing state file")
    
    # Initialize filter
    filter = DuplicateFilter()
    
    # Get real RSS articles
    print("\n📡 Fetching real RSS articles...")
    orchestrator = None
    try:
        from trading_ai.core.orchestrator import PipelineOrchestrator
        orchestrator = PipelineOrchestrator()
        
        # Just run the collection stage
        articles = []
        for source in orchestrator.source_registry.sources.values():
            try:
                print(f"   Fetching from {source.name}...")
                source_articles = orchestrator.news_collector.fetch_feed(source.url)[0]
                articles.extend(source_articles)
                print(f"   Got {len(source_articles)} articles from {source.name}")
            except Exception as e:
                print(f"   Error fetching {source.name}: {e}")
        
        print(f"\n📊 Total articles fetched: {len(articles)}")
        
        # Show first few articles
        print("\n📝 First 5 articles:")
        for i, article in enumerate(articles[:5]):
            print(f"   {i+1}. {article.title[:60]}...")
            print(f"      Source: {article.source}")
            print(f"      URL: {article.url}")
            print()
        
        # Test duplicate filtering
        print("🔍 Testing duplicate filtering...")
        unique_articles = filter.filter_duplicates(articles)
        
        print(f"\n📊 RESULTS:")
        print(f"   Total articles: {len(articles)}")
        print(f"   Unique articles: {len(unique_articles)}")
        print(f"   Duplicates removed: {len(articles) - len(unique_articles)}")
        print(f"   Duplicate rate: {(len(articles) - len(unique_articles)) / len(articles) * 100:.1f}%")
        
        # Show some duplicate examples
        if len(unique_articles) < len(articles):
            print("\n❌ DUPLICATE EXAMPLES:")
            duplicate_count = 0
            for article in articles:
                if filter._is_duplicate(article):
                    duplicate_count += 1
                    if duplicate_count <= 3:
                        print(f"   {duplicate_count}. {article.title[:60]}...")
                        print(f"      Source: {article.source}")
                        print(f"      URL: {article.url}")
                        
                        # Check similarity
                        for seen_id, seen_data in filter.seen_articles.items():
                            seen_title = filter._normalize_title(seen_data["title"])
                            article_title = filter._normalize_title(article.title)
                            similarity = SequenceMatcher(None, article_title, seen_title).ratio()
                            if similarity >= filter.title_similarity_threshold:
                                print(f"      Similarity: {similarity:.3f} (threshold: {filter.title_similarity_threshold})")
                                print(f"      Seen title: {seen_title[:50]}...")
                                break
                        print()
                    if duplicate_count >= 3:
                        break
        
        return len(unique_articles) > 0
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    from difflib import SequenceMatcher
    
    success = debug_duplicate_filter()
    
    if success:
        print("✅ DUPLICATE FILTER DEBUG COMPLETED")
        sys.exit(0)
    else:
        print("❌ DUPLICATE FILTER DEBUG FAILED")
        sys.exit(1)
