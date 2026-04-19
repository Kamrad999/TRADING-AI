#!/usr/bin/env python3
"""
Final source validation and optimization for TRADING-AI.
"""

import sys
import requests
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_source(url: str, name: str, category: str) -> Dict[str, Any]:
    """Test a single RSS source."""
    try:
        start_time = time.time()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            # Parse RSS content
            try:
                import feedparser
                feed = feedparser.parse(response.content)
                article_count = len(feed.entries)
                
                # Check signal relevance
                signal_keywords = 0
                for entry in feed.entries[:10]:
                    if hasattr(entry, 'title'):
                        title = entry.title.lower()
                        if any(keyword in title for keyword in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'stock', 'market', 'earnings', 'fed', 'inflation', 'trading', 'surge', 'crash', 'rally']):
                            signal_keywords += 1
                
                # Calculate freshness
                now = datetime.now(timezone.utc)
                total_age_hours = 0
                for entry in feed.entries[:5]:
                    if hasattr(entry, 'published'):
                        pub_time = entry.published
                        if isinstance(pub_time, str):
                            try:
                                pub_time = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                            except:
                                continue
                        else:
                            pub_time = pub_time
                        
                        age_hours = (now - pub_time).total_seconds() / 3600
                        total_age_hours += age_hours
                
                avg_age_hours = total_age_hours / len(feed.entries[:5]) if len(feed.entries[:5]) > 0 else 0
                
                # Calculate scores
                reliability_score = 40.0  # Success
                signal_density_score = 30.0 if article_count > 50 else 25.0 if article_count > 25 else 20.0 if article_count > 10 else 10.0 if article_count > 0 else 0.0
                latency_score = 15.0 if response_time < 1.0 else 12.0 if response_time < 2.0 else 8.0 if response_time < 5.0 else 0.0
                freshness_score = 15.0 if avg_age_hours < 6.0 else 12.0 if avg_age_hours < 12.0 else 8.0 if avg_age_hours < 24.0 else 0.0
                total_score = reliability_score + signal_density_score + latency_score + freshness_score
                
                # Grade
                if total_score >= 80.0:
                    grade = "A"
                elif total_score >= 70.0:
                    grade = "B"
                elif total_score >= 60.0:
                    grade = "C"
                else:
                    grade = "D"
                
                return {
                    "name": name,
                    "url": url,
                    "category": category,
                    "status": "SUCCESS",
                    "article_count": article_count,
                    "signal_keywords": signal_keywords,
                    "avg_age_hours": avg_age_hours,
                    "response_time": round(response_time, 2),
                    "total_score": total_score,
                    "grade": grade
                }
                
        else:
            return {
                "name": name,
                "url": url,
                "category": category,
                "status": "FAILED",
                "article_count": 0,
                "signal_keywords": 0,
                "avg_age_hours": None,
                "response_time": round(response_time, 2),
                "total_score": 0.0,
                "grade": "F"
            }
            
    except Exception as e:
        return {
            "name": name,
            "url": url,
            "category": category,
            "status": "ERROR",
            "article_count": 0,
            "signal_keywords": 0,
            "avg_age_hours": None,
            "response_time": None,
            "total_score": 0.0,
            "grade": "F"
        }


def main():
    """Main source validation function."""
    print("🚀 COMPREHENSIVE SOURCE VALIDATION & OPTIMIZATION")
    print("=" * 60)
    
    # High-quality sources to test
    sources = [
        # CRYPTO SOURCES
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
    
    print(f"Testing {len(sources)} high-quality sources...")
    
    # Test each source
    results = []
    for i, source in enumerate(sources):
        print(f"\n📡 [{i+1}/{len(sources)}] Testing: {source['name']}")
        
        result = test_source(source['url'], source['name'], source['category'])
        results.append(result)
        
        print(f"   Status: {result['status']}")
        print(f"   Articles: {result['article_count']}")
        print(f"   Signal Keywords: {result['signal_keywords']}")
        print(f"   Response Time: {result['response_time']}s")
        print(f"   Score: {result['total_score']:.1f} (Grade: {result['grade']})")
    
    # Sort by score
    results.sort(key=lambda x: x["total_score"], reverse=True)
    
    # Auto-selection
    print(f"\n\n🎯 AUTO-SELECTION RESULTS:")
    grade_a = [r for r in results if r["grade"] == "A"]
    grade_b = [r for r in results if r["grade"] == "B"]
    grade_c = [r for r in results if r["grade"] == "C"]
    grade_df = [r for r in results if r["grade"] in ["D", "F"]]
    
    print(f"   Grade A (KEEP): {len(grade_a)} sources")
    print(f"   Grade B (KEEP): {len(grade_b)} sources")
    print(f"   Grade C (CONSIDER): {len(grade_c)} sources")
    print(f"   Grade D-F (REMOVE): {len(grade_df)} sources")
    
    # Top sources
    top_sources = grade_a + grade_b + grade_c[:8]  # Keep A, B, and top 8 C
    print(f"\n🏆 TOP SOURCES (Total: {len(top_sources)}):")
    
    # Generate final optimized list
    optimized_sources = []
    for source in top_sources:
        optimized_sources.append({
            "name": source["name"],
            "url": source["url"],
            "category": source["category"],
            "priority": 1,
            "enabled": True
        })
    
    # Display results
    print(f"{'Name':<20} {'Category':<15} {'Articles':<8} {'Score':<6} {'Grade':<5}")
    print("-" * 80)
    
    for source in top_sources:
        print(f"{source['name']:<19} {source['category']:<14} {source['article_count']:<8} {source['total_score']:<6.1f} {source['grade']:<5}")
    
    print(f"\n🎯 FINAL OPTIMIZED SOURCE LIST:")
    for source in optimized_sources:
        print(f"   {source['name']} ({source['category']}): {source['url']}")
    
    print(f"\n✅ SOURCE OPTIMIZATION COMPLETE")
    print(f"📊 SUMMARY: {len(top_sources)} sources selected for implementation")
    
    return optimized_sources


if __name__ == "__main__":
    main()
