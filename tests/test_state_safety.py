#!/usr/bin/env python3
"""
State safety test to prevent contamination between runs.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.validation.duplicate_filter import DuplicateFilter


def test_state_isolation():
    """Test that state doesn't contaminate between runs."""
    print("🧪 Testing state isolation...")
    
    # Clear any existing state
    state_file = Path("./data/state.json")
    if state_file.exists():
        os.remove(state_file)
        print("🗑️  Cleared existing state file")
    
    # First run - should process all articles
    print("\n📊 First run (fresh state):")
    filter1 = DuplicateFilter()
    
    from trading_ai.core.models import Article
    test_articles = [
        Article(
            title=f"Test Article {i}",
            content=f"Content for article {i}",
            source="Test Source",
            timestamp=datetime.now(timezone.utc),
            url=f"https://test.com/article{i}",
            metadata={}
        )
        for i in range(10)
    ]
    
    unique1 = filter1.filter_duplicates(test_articles)
    print(f"  Articles processed: {len(test_articles)}")
    print(f"  Unique articles: {len(unique1)}")
    
    # Second run - should also process all articles (different URLs)
    print("\n📊 Second run (different articles):")
    filter2 = DuplicateFilter()
    
    test_articles2 = [
        Article(
            title=f"Different Test Article {i}",
            content=f"Different content for article {i}",
            source="Different Source",
            timestamp=datetime.now(timezone.utc),
            url=f"https://different.com/article{i}",
            metadata={}
        )
        for i in range(10)
    ]
    
    unique2 = filter2.filter_duplicates(test_articles2)
    print(f"  Articles processed: {len(test_articles2)}")
    print(f"  Unique articles: {len(unique2)}")
    
    # Check state contamination
    stats = filter2.get_duplicate_stats()
    print(f"\n📈 Final state stats:")
    print(f"  Seen articles: {stats['seen_articles_count']}")
    print(f"  URL hashes: {stats['url_hashes_cached']}")
    print(f"  Recent articles: {stats['recent_articles_count']}")
    
    # Verify no contamination
    if len(unique1) == 10 and len(unique2) == 10:
        print("✅ State isolation test PASSED")
        return True
    else:
        print("❌ State isolation test FAILED")
        print(f"  Expected: 10 unique in both runs")
        print(f"  Got: {len(unique1)} and {len(unique2)}")
        return False


def test_state_reset():
    """Test state reset functionality."""
    print("\n🧪 Testing state reset...")
    
    filter = DuplicateFilter()
    
    # Add some articles to populate state
    from trading_ai.core.models import Article
    test_article = Article(
        title="Test Article",
        content="Test content",
        source="Test Source",
        timestamp=datetime.now(timezone.utc),
        url="https://test.com/article",
        metadata={}
    )
    
    filter.filter_duplicates([test_article])
    
    # Check state has entries
    stats_before = filter.get_duplicate_stats()
    print(f"  Before reset: {stats_before['seen_articles_count']} seen articles")
    
    # Reset state
    filter.reset_duplicate_state()
    
    # Check state is empty
    stats_after = filter.get_duplicate_stats()
    print(f"  After reset: {stats_after['seen_articles_count']} seen articles")
    
    if stats_after['seen_articles_count'] == 0:
        print("✅ State reset test PASSED")
        return True
    else:
        print("❌ State reset test FAILED")
        return False


if __name__ == "__main__":
    print("🛡️  STATE SAFETY VALIDATION")
    print("=" * 50)
    
    test1_passed = test_state_isolation()
    test2_passed = test_state_reset()
    
    print(f"\n🎯 OVERALL RESULT:")
    if test1_passed and test2_passed:
        print("✅ ALL STATE SAFETY TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ STATE SAFETY TESTS FAILED")
        sys.exit(1)
