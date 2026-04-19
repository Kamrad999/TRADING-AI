#!/usr/bin/env python3
"""
Simple source validation for TRADING-AI.
"""

import sys
import requests
import time
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_sources():
    """Test key RSS sources for TRADING-AI."""
    print("🚀 RSS SOURCE VALIDATION")
    print("=" * 50)
    
    # Key sources to test
    sources = [
        # CRYPTO SOURCES (High Priority)
        {"name": "crypto_panic", "url": "https://cryptopanic.com/news/rss", "category": "crypto_news"},
        {"name": "decrypt_media", "url": "https://decrypt.co/feed", "category": "crypto_news"},
        {"name": "the_block", "url": "https://theblock.co/rss", "category": "crypto_news"},
        {"name": "coin_desk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "category": "crypto_news"},
        {"name": "coin_telegraph", "url": "https://cointelegraph.com/rss", "category": "crypto_news"},
        
        # STOCK/EQUITY SOURCES
        {"name": "alpha_street", "url": "https://seekingalpha.com/feed.xml", "category": "market_data"},
        {"name": "benzinga", "url": "https://www.benzinga.com/feed", "category": "financial_news"},
        {"name": "investing_com", "url": "https://www.investing.com/rss/news.rss", "category": "financial_news"},
        {"name": "marketbeat", "url": "https://www.marketbeat.com/rss", "category": "financial_news"},
        {"name": "y_finance", "url": "https://finance.yahoo.com/news/rss", "category": "financial_news"},
        
        # TECH SOURCES
        {"name": "tech_crunch", "url": "https://techcrunch.com/feed/", "category": "tech_news"},
        {"name": "arstechnica", "url": "https://arstechnica.com/rss", "category": "tech_news"},
        {"name": "wired", "url": "https://www.wired.com/feed/rss", "category": "tech_news"},
        
        # MACRO SOURCES
        {"name": "federal_reserve", "url": "https://www.federalreserve.gov/feeds/pressreleases.xml", "category": "economic_data"},
        {"name": "cpi_data", "url": "https://www.bls.gov/rss/cpi.rss", "category": "economic_data"},
    ]
    
    print(f"Testing {len(sources)} key sources...")
    
    successful_sources = []
    for i, source in enumerate(sources):
        print(f"\n📡 [{i+1}/{len(sources)}] {source['name']}")
        
        try:
            start_time = time.time()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml'
            }
            response = requests.get(source['url'], timeout=10, headers=headers)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    import feedparser
                    feed = feedparser.parse(response.content)
                    article_count = len(feed.entries)
                    
                    print(f"   ✅ SUCCESS - {article_count} articles")
                    print(f"   Response Time: {response_time:.2f}s")
                    
                    # Check signal relevance
                    signal_keywords = 0
                    for entry in feed.entries[:5]:
                        if hasattr(entry, 'title'):
                            title = entry.title.lower()
                            if any(keyword in title for keyword in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'stock', 'market', 'earnings', 'fed', 'inflation']):
                                signal_keywords += 1
                    
                    signal_ratio = signal_keywords / len(feed.entries[:5]) if len(feed.entries[:5]) > 0 else 0
                    print(f"   Signal Relevance: {signal_ratio*100:.0f}%")
                    
                    # Grade
                    if article_count > 50:
                        grade = "A"
                    elif article_count > 25:
                        grade = "B"
                    elif article_count > 10:
                        grade = "C"
                    else:
                        grade = "D"
                    
                    successful_sources.append({
                        "name": source['name'],
                        "url": source['url'],
                        "category": source['category'],
                        "grade": grade,
                        "articles": article_count,
                        "signal_relevance": signal_ratio
                    })
                    
                except Exception as e:
                    print(f"   ⚠️  RSS parsing error: {e}")
                    
            else:
                print(f"   ❌ FAILED - HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ ERROR - {e}")
    
    print(f"\n\n📊 VALIDATION SUMMARY:")
    print(f"   Total Sources: {len(sources)}")
    print(f"   Successful: {len(successful_sources)}")
    print(f"   Success Rate: {len(successful_sources)/len(sources)*100:.1f}%")
    
    print(f"\n✅ OPTIMIZED SOURCES FOR IMPLEMENTATION:")
    for source in successful_sources:
        print(f"   {source['name']} ({source['grade']}) - {source['url']}")
    
    return successful_sources


if __name__ == "__main__":
    successful_sources = test_sources()
    
    print(f"\n🎯 FINAL RESULT: {len(successful_sources)} high-quality sources ready for implementation")
