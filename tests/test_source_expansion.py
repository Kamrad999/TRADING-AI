#!/usr/bin/env python3
"""
Source expansion discovery for TRADING-AI system.
"""

import sys
import requests
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_source_candidate(url: str, name: str, category: str) -> Dict[str, Any]:
    """Test a candidate RSS source."""
    result = {
        "name": name,
        "url": url,
        "category": category,
        "status": "UNKNOWN",
        "response_code": None,
        "response_time": None,
        "content_length": 0,
        "article_count": 0,
        "sample_titles": [],
        "error_message": None,
        "signal_relevance": "UNKNOWN",
        "freshness_hours": None
    }
    
    try:
        start_time = time.time()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response_time = time.time() - start_time
        
        result["response_code"] = response.status_code
        result["response_time"] = round(response_time, 2)
        
        if response.status_code == 200:
            result["status"] = "SUCCESS"
            result["content_length"] = len(response.content)
            
            # Parse RSS content
            try:
                import feedparser
                feed = feedparser.parse(response.content)
                result["article_count"] = len(feed.entries)
                
                # Get sample titles and check freshness
                now = datetime.now(timezone.utc)
                total_age_hours = 0
                signal_keywords = 0
                
                for entry in feed.entries[:5]:
                    if hasattr(entry, 'title'):
                        title = entry.title[:80]
                        result["sample_titles"].append(title)
                        
                        # Check for signal-relevant keywords
                        title_lower = title.lower()
                        if any(keyword in title_lower for keyword in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'stock', 'market', 'earnings', 'fed', 'inflation', 'trading']):
                            signal_keywords += 1
                        
                        # Calculate freshness
                        if hasattr(entry, 'published'):
                            pub_time = entry.published
                            if isinstance(pub_time, str):
                                try:
                                    pub_time = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                                except:
                                    try:
                                        from dateutil import parser
                                        pub_time = parser.parse(pub_time)
                                    except:
                                        continue
                            else:
                                pub_time = pub_time
                            
                            age_hours = (now - pub_time).total_seconds() / 3600
                            total_age_hours += age_hours
                
                # Calculate metrics
                avg_age_hours = total_age_hours / len(feed.entries[:5]) if len(feed.entries[:5]) > 0 else 0
                result["freshness_hours"] = round(avg_age_hours, 1)
                
                # Signal relevance assessment
                signal_ratio = signal_keywords / len(feed.entries[:5]) if len(feed.entries[:5]) > 0 else 0
                if signal_ratio > 0.8:
                    result["signal_relevance"] = "HIGH"
                elif signal_ratio > 0.5:
                    result["signal_relevance"] = "MEDIUM"
                elif signal_ratio > 0:
                    result["signal_relevance"] = "LOW"
                else:
                    result["signal_relevance"] = "NONE"
                
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


def discover_high_quality_sources():
    """Discover 30-50 high-quality free RSS sources."""
    print("🔍 PHASE 1 - SOURCE EXPANSION")
    print("=" * 60)
    
    # High-quality source candidates
    source_candidates = [
        # CRYPTO SOURCES (High Priority)
        {"name": "crypto_panic", "url": "https://cryptopanic.com/news/rss", "category": "crypto_news"},
        {"name": "decrypt_media", "url": "https://decrypt.co/feed", "category": "crypto_news"},
        {"name": "the_block", "url": "https://theblock.co/rss", "category": "crypto_news"},
        {"name": "bitcoin_magazine", "url": "https://bitcoinmagazine.com/feed", "category": "crypto_news"},
        {"name": "crypto_news", "url": "https://cryptonews.com/feed", "category": "crypto_news"},
        {"name": "coin_desk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "category": "crypto_news"},
        {"name": "coin_telegraph", "url": "https://cointelegraph.com/rss", "category": "crypto_news"},
        {"name": "crypto_slate", "url": "https://cryptoslate.com/feed", "category": "crypto_news"},
        {"name": "amb_crypto", "url": "https://ambcrypto.com/feed", "category": "crypto_news"},
        {"name": "blockworks", "url": "https://blockworks.co/news/rss", "category": "crypto_news"},
        {"name": "the_defiant", "url": "https://thedefiant.io/feed", "category": "crypto_news"},
        
        # STOCK/EQUITY SOURCES
        {"name": "alpha_street", "url": "https://seekingalpha.com/feed.xml", "category": "market_data"},
        {"name": "benzinga", "url": "https://www.benzinga.com/feed", "category": "financial_news"},
        {"name": "investing_com", "url": "https://www.investing.com/rss/news.rss", "category": "financial_news"},
        {"name": "marketbeat", "url": "https://www.marketbeat.com/rss", "category": "financial_news"},
        {"name": "street_insider", "url": "https://seekingalpha.com/api/sa/combined?id=352765&count=10", "category": "market_data"},
        {"name": "y_finance", "url": "https://finance.yahoo.com/news/rss", "category": "financial_news"},
        {"name": "tip_ranks", "url": "https://www.tipranks.com/rss", "category": "market_data"},
        {"name": "finviz", "url": "https://finviz.com/news_rss.ashx", "category": "financial_news"},
        {"name": "stock_twits", "url": "https://www.stocktwits.com/messages.rss", "category": "market_data"},
        
        # MACRO/ECONOMIC SOURCES
        {"name": "federal_reserve", "url": "https://www.federalreserve.gov/feeds/pressreleases.xml", "category": "economic_data"},
        {"name": "cpi_data", "url": "https://www.bls.gov/rss/cpi.rss", "category": "economic_data"},
        {"name": "treasury_reports", "url": "https://home.treasury.gov/data.xml", "category": "economic_data"},
        {"name": "bea_gdp", "url": "https://www.bea.gov/news/rss", "category": "economic_data"},
        {"name": "durable_goods", "url": "https://www.census.gov/manufacturing/rss", "category": "economic_data"},
        {"name": "jobless_claims", "url": "https://www.dol.gov/rss/newsreleases.xml", "category": "economic_data"},
        {"name": "retail_sales", "url": "https://www.census.gov/retail/rss", "category": "economic_data"},
        
        # COMMODITY SOURCES
        {"name": "oil_price", "url": "https://oilprice.com/rss/latest", "category": "commodity_news"},
        {"name": "gold_price", "url": "https://www.goldprice.org/rss", "category": "commodity_news"},
        {"name": "kitco_news", "url": "https://www.kitco.com/news/rss", "category": "commodity_news"},
        {"name": "silver_price", "url": "https://www.silverprice.org/rss", "category": "commodity_news"},
        
        # FOREX SOURCES
        {"name": "forex_factory", "url": "https://www.forexfactory.com/news.rss", "category": "forex_news"},
        {"name": "daily_fx", "url": "https://www.dailyfx.com/rss", "category": "forex_news"},
        {"name": "fx_empire", "url": "https://www.fxempire.com/rss", "category": "forex_news"},
        {"name": "investing_fx", "url": "https://www.investing.com/rss/forex_1.rss", "category": "forex_news"},
        
        # TECH/SECTOR SOURCES
        {"name": "tech_crunch", "url": "https://techcrunch.com/feed/", "category": "tech_news"},
        {"name": "venture_beat", "url": "https://venturebeat.com/rss", "category": "tech_news"},
        {"name": "ars_technica", "url": "https://arstechnica.com/rss", "category": "tech_news"},
        {"name": "wired", "url": "https://www.wired.com/feed/rss", "category": "tech_news"},
        {"name": "the_verge", "url": "https://www.theverge.com/rss/index.xml", "category": "tech_news"},
    ]
    
    print(f"Testing {len(source_candidates)} high-quality source candidates...")
    
    # Test each source
    results = []
    successful_sources = []
    
    for i, source in enumerate(source_candidates):
        print(f"\n📡 [{i+1}/{len(source_candidates)}] Testing: {source['name']}")
        print(f"   URL: {source['url']}")
        print(f"   Category: {source['category']}")
        
        result = test_source_candidate(source['url'], source['name'], source['category'])
        result["priority"] = 1  # All high priority
        
        if result["status"] == "SUCCESS":
            successful_sources.append(result)
        
        results.append(result)
        
        print(f"   Status: {result['status']}")
        print(f"   Response Code: {result['response_code']}")
        print(f"   Response Time: {result['response_time']}s")
        print(f"   Articles: {result['article_count']}")
        print(f"   Signal Relevance: {result['signal_relevance']}")
        print(f"   Freshness: {result['freshness_hours']}h")
        
        if result['sample_titles']:
            print(f"   Sample Titles:")
            for title in result['sample_titles'][:2]:
                print(f"      - {title}")
        
        if result['error_message']:
            print(f"   Error: {result['error_message']}")
    
    return results, successful_sources


if __name__ == "__main__":
    print("🚀 HIGH-QUALITY SOURCE DISCOVERY")
    print("=" * 60)
    
    results, successful_sources = discover_high_quality_sources()
    
    # Summary
    print(f"\n\n📊 DISCOVERY SUMMARY:")
    print(f"   Total Candidates: {len(results)}")
    print(f"   Successful: {len(successful_sources)}")
    print(f"   Success Rate: {len(successful_sources)/len(results)*100:.1f}%")
    
    # Category breakdown
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
    
    # High-quality sources for next phase
    print(f"\n✅ HIGH-QUALITY SOURCES FOR NEXT PHASE:")
    for source in successful_sources:
        print(f"   {source['name']} ({source['category']}): {source['url']}")
