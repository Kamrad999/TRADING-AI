#!/usr/bin/env python3
"""
Comprehensive RSS source audit for TRADING-AI system.
"""

import sys
import requests
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.infrastructure.source_registry import SourceRegistry


def test_rss_source(url: str, name: str) -> Dict[str, Any]:
    """Test RSS source accessibility and content."""
    result = {
        "name": name,
        "url": url,
        "status": "UNKNOWN",
        "response_code": None,
        "response_time": None,
        "content_length": 0,
        "article_count": 0,
        "sample_titles": [],
        "error_message": None
    }
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response_time = time.time() - start_time
        
        result["response_code"] = response.status_code
        result["response_time"] = round(response_time, 2)
        
        if response.status_code == 200:
            result["status"] = "SUCCESS"
            result["content_length"] = len(response.content)
            
            # Try to parse RSS content
            try:
                import feedparser
                feed = feedparser.parse(response.content)
                result["article_count"] = len(feed.entries)
                
                # Get sample titles
                for entry in feed.entries[:3]:
                    if hasattr(entry, 'title'):
                        result["sample_titles"].append(entry.title[:80])
                
            except Exception as e:
                result["error_message"] = f"RSS parsing error: {str(e)}"
                
        else:
            result["status"] = "FAILED"
            result["error_message"] = f"HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        result["status"] = "TIMEOUT"
        result["error_message"] = "Request timeout"
    except Exception as e:
        result["status"] = "ERROR"
        result["error_message"] = str(e)
    
    return result


def audit_all_sources():
    """Audit all RSS sources in the registry."""
    print("🔍 RSS SOURCE AUDIT - PHASE 1 DISCOVERY")
    print("=" * 60)
    
    # Initialize source registry
    registry = SourceRegistry()
    sources = registry.get_sources(enabled_only=False)
    
    print(f"Found {len(sources)} total sources")
    
    # Test each source
    results = []
    for source in sources:
        print(f"\n📡 Testing: {source.name}")
        print(f"   URL: {source.url}")
        print(f"   Category: {source.category}")
        
        result = test_rss_source(source.url, source.name)
        result["category"] = source.category
        result["priority"] = source.priority
        results.append(result)
        
        print(f"   Status: {result['status']}")
        print(f"   Response Code: {result['response_code']}")
        print(f"   Response Time: {result['response_time']}s")
        print(f"   Articles: {result['article_count']}")
        
        if result['sample_titles']:
            print(f"   Sample Titles:")
            for title in result['sample_titles']:
                print(f"      - {title}")
        
        if result['error_message']:
            print(f"   Error: {result['error_message']}")
    
    return results


def analyze_sources(results: List[Dict[str, Any]]) -> None:
    """Analyze source quality and classify them."""
    print("\n\n📊 RSS SOURCE AUDIT - PHASE 2 QUALITY ANALYSIS")
    print("=" * 60)
    
    # Categorize results
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] in ['FAILED', 'TIMEOUT', 'ERROR']]
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total Sources: {len(results)}")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Success Rate: {len(successful)/len(results)*100:.1f}%")
    
    # Analyze by category
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = {"total": 0, "successful": 0, "articles": 0}
        categories[cat]["total"] += 1
        if result['status'] == 'SUCCESS':
            categories[cat]["successful"] += 1
        categories[cat]["articles"] += result['article_count']
    
    print(f"\n📂 BY CATEGORY:")
    for cat, stats in categories.items():
        success_rate = (stats['successful'] / stats['total']) * 100 if stats['total'] > 0 else 0
        avg_articles = stats['articles'] / stats['successful'] if stats['successful'] > 0 else 0
        print(f"   {cat}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%) - {avg_articles:.1f} avg articles")
    
    # Detailed analysis
    print(f"\n🔍 DETAILED ANALYSIS:")
    for result in results:
        print(f"\n📰 {result['name']} ({result['category']})")
        print(f"   Status: {result['status']}")
        print(f"   Response Time: {result['response_time']}s")
        print(f"   Articles: {result['article_count']}")
        
        # Quality assessment
        if result['status'] == 'SUCCESS':
            if result['article_count'] > 20:
                signal_relevance = "HIGH"
                frequency = "HIGH"
            elif result['article_count'] > 10:
                signal_relevance = "MEDIUM"
                frequency = "MEDIUM"
            else:
                signal_relevance = "LOW"
                frequency = "LOW"
            
            # Check for signal-relevant content
            signal_keywords = 0
            for title in result['sample_titles']:
                if any(keyword in title.lower() for keyword in ['bitcoin', 'stock', 'market', 'earnings', 'fed', 'inflation', 'crypto']):
                    signal_keywords += 1
            
            if signal_keywords > len(result['sample_titles']) * 0.7:
                noise_level = "LOW"
            elif signal_keywords > len(result['sample_titles']) * 0.3:
                noise_level = "MEDIUM"
            else:
                noise_level = "HIGH"
            
            print(f"   Signal Relevance: {signal_relevance}")
            print(f"   Frequency: {frequency}")
            print(f"   Noise Level: {noise_level}")
            print(f"   Signal Keywords: {signal_keywords}/{len(result['sample_titles'])}")
        else:
            print(f"   Error: {result['error_message']}")


def classify_and_recommend(results: List[Dict[str, Any]]) -> None:
    """Classify sources and provide recommendations."""
    print("\n\n🎯 RSS SOURCE AUDIT - PHASE 3-5 CLASSIFICATION & RECOMMENDATIONS")
    print("=" * 60)
    
    # Classification
    keep_sources = []
    replace_sources = []
    remove_sources = []
    
    for result in results:
        if result['status'] == 'SUCCESS' and result['article_count'] > 10:
            keep_sources.append(result['name'])
        elif result['status'] == 'SUCCESS' and result['article_count'] > 0:
            replace_sources.append(result['name'])
        else:
            remove_sources.append(result['name'])
    
    print(f"\n📋 CLASSIFICATION:")
    print(f"   KEEP ({len(keep_sources)}): {', '.join(keep_sources)}")
    print(f"   REPLACE ({len(replace_sources)}): {', '.join(replace_sources)}")
    print(f"   REMOVE ({len(remove_sources)}): {', '.join(remove_sources)}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    print(f"\n🚀 NEW RECOMMENDED SOURCES:")
    
    # Crypto sources (high priority)
    crypto_sources = [
        {"name": "crypto_panic", "url": "https://cryptopanic.com/news/rss", "category": "crypto_news"},
        {"name": "decrypt_media", "url": "https://decrypt.co/feed", "category": "crypto_news"},
        {"name": "the_block", "url": "https://theblock.co/rss", "category": "crypto_news"},
        {"name": "coin_telegraph", "url": "https://cointelegraph.com/rss", "category": "crypto_news"},
        {"name": "bitcoin_magazine", "url": "https://bitcoinmagazine.com/feed", "category": "crypto_news"},
    ]
    
    # Stock sources (free accessible)
    stock_sources = [
        {"name": "alpha_street", "url": "https://seekingalpha.com/feed.xml", "category": "market_data"},
        {"name": "benzinga", "url": "https://www.benzinga.com/feed", "category": "financial_news"},
        {"name": "investing_com", "url": "https://www.investing.com/rss/news.rss", "category": "financial_news"},
        {"name": "marketbeat", "url": "https://www.marketbeat.com/rss", "category": "financial_news"},
    ]
    
    # Macro sources
    macro_sources = [
        {"name": "federal_reserve", "url": "https://www.federalreserve.gov/feeds/pressreleases.xml", "category": "economic_data"},
        {"name": "cpi_data", "url": "https://www.bls.gov/rss/cpi.rss", "category": "economic_data"},
        {"name": "treasury_reports", "url": "https://home.treasury.gov/data.xml", "category": "economic_data"},
    ]
    
    print(f"\n🪙 CRYPTO SOURCES:")
    for source in crypto_sources:
        print(f"   {source['name']}: {source['url']}")
    
    print(f"\n📈 STOCK SOURCES:")
    for source in stock_sources:
        print(f"   {source['name']}: {source['url']}")
    
    print(f"\n🏛️  MACRO SOURCES:")
    for source in macro_sources:
        print(f"   {source['name']}: {source['url']}")
    
    # Final optimized list
    print(f"\n🎯 FINAL OPTIMIZED SOURCE LIST:")
    all_recommended = crypto_sources + stock_sources + macro_sources
    
    for source in all_recommended:
        print(f"   {source['name']} ({source['category']}): {source['url']}")


if __name__ == "__main__":
    print("🔍 COMPREHENSIVE RSS SOURCE AUDIT")
    print("=" * 60)
    
    # Phase 1-2: Discovery and Analysis
    results = audit_all_sources()
    
    # Phase 3-5: Classification and Recommendations
    classify_and_recommend(results)
