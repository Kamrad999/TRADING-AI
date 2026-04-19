#!/usr/bin/env python3
"""
Debug signal generation to understand why 0 signals are generated.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.agents.signal_generator import SignalGenerator
from trading_ai.core.models import Article


def debug_signal_generation():
    """Debug signal generation with real article data."""
    print("🔍 SIGNAL GENERATION DEBUG")
    print("=" * 50)
    
    # Clear state
    data_dir = Path("./data")
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir()
    
    # Create signal generator
    generator = SignalGenerator()
    
    # Test 1: Manual article with clear keywords
    print("\n📝 TEST 1: Manual article with clear keywords")
    manual_article = Article(
        title="Bitcoin surges to new all-time high as institutional buying accelerates",
        content="Bitcoin reached $75,000 today as major institutional investors announced massive BTC purchases. Market analysts predict continued upward momentum.",
        source="Test Source",
        timestamp=datetime.now(timezone.utc),
        url="https://test.com/bitcoin-surge",
        metadata={"validation": {"confidence_score": 0.8}}
    )
    
    print(f"   Article: {manual_article.title[:60]}...")
    
    # Test keyword scoring
    text = f"{manual_article.title} {manual_article.content}".lower()
    print(f"   Text length: {len(text)} characters")
    
    bullish_score = generator._calculate_keyword_score(text, generator.bullish_keywords)
    bearish_score = generator._calculate_keyword_score(text, generator.bearish_keywords)
    
    print(f"   Bullish score: {bullish_score:.3f}")
    print(f"   Bearish score: {bearish_score:.3f}")
    print(f"   Max score: {max(bullish_score, bearish_score):.3f}")
    
    # Test threshold
    if max(bullish_score, bearish_score) < 0.1:
        print("   ❌ BELOW THRESHOLD - No signal generated")
    else:
        print("   ✅ ABOVE THRESHOLD - Signal should be generated")
    
    # Test 2: Real RSS article
    print("\n📡 TEST 2: Real RSS article")
    from trading_ai.core.orchestrator import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator()
    sources = list(orchestrator.source_registry.sources.values())
    
    if sources:
        # Get articles from multiple sources
        all_articles = []
        for source in sources[:3]:  # Test first 3 sources
            try:
                articles, metadata = orchestrator.news_collector.fetch_feed(source.url)
                if articles:
                    all_articles.extend(articles)
                    print(f"   Fetched {len(articles)} from {source.name}")
            except Exception as e:
                print(f"   Error fetching {source.name}: {e}")
        
        if all_articles:
            # Test first few articles
            for i, article in enumerate(all_articles[:3]):
                print(f"\n   Article {i+1}: {article.title[:60]}...")
                print(f"      Source: {article.source}")
                
                # Test keyword scoring
                text = f"{article.title} {article.content}".lower()
                bullish_score = generator._calculate_keyword_score(text, generator.bullish_keywords)
                bearish_score = generator._calculate_keyword_score(text, generator.bearish_keywords)
                
                print(f"      Bullish score: {bullish_score:.3f}")
                print(f"      Bearish score: {bearish_score:.3f}")
                print(f"      Max score: {max(bullish_score, bearish_score):.3f}")
                
                # Show found keywords
                found_keywords = generator._get_found_keywords(text)
                if found_keywords:
                    print(f"      Found keywords: {list(found_keywords.keys())[:5]}")
                else:
                    print(f"      Found keywords: None")
                
                # Test threshold
                if max(bullish_score, bearish_score) < 0.1:
                    print("      ❌ BELOW THRESHOLD")
                else:
                    print("      ✅ ABOVE THRESHOLD")
    
    # Test 3: Show sample keywords
    print(f"\n🔤 SAMPLE KEYWORDS:")
    print(f"   Bullish keywords: {list(generator.bullish_keywords.keys())[:10]}")
    print(f"   Bearish keywords: {list(generator.bearish_keywords.keys())[:10]}")
    
    return True


if __name__ == "__main__":
    success = debug_signal_generation()
    
    if success:
        print("\n✅ SIGNAL GENERATION DEBUG COMPLETED")
        sys.exit(0)
    else:
        print("\n❌ SIGNAL GENERATION DEBUG FAILED")
        sys.exit(1)
