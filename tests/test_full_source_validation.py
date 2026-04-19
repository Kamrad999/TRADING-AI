#!/usr/bin/env python3
"""
Full source validation with live testing and scoring.
"""

import sys
import requests
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_single_source(url: str, name: str, category: str) -> Dict[str, Any]:
    """Test a single RSS source comprehensively."""
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
        "signal_relevance": "UNKNOWN",
        "freshness_hours": None,
        "error_message": None
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
                
                for entry in feed.entries[:10]:
                    if hasattr(entry, 'title'):
                        title = entry.title[:80]
                        result["sample_titles"].append(title)
                        
                        # Check for signal-relevant keywords
                        title_lower = title.lower()
                        if any(keyword in title_lower for keyword in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'stock', 'market', 'earnings', 'fed', 'inflation', 'trading', 'surge', 'crash', 'rally']):
                            signal_keywords += 1
                        
                        # Calculate freshness
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
                
                # Calculate metrics
                avg_age_hours = total_age_hours / len(feed.entries[:10]) if len(feed.entries[:10]) > 0 else 0
                result["freshness_hours"] = round(avg_age_hours, 1)
                
                # Signal relevance assessment
                signal_ratio = signal_keywords / len(feed.entries[:10]) if len(feed.entries[:10]) > 0 else 0
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


def validate_top_sources():
    """Validate top sources with comprehensive scoring."""
    print("🔍 COMPREHENSIVE SOURCE VALIDATION")
    print("=" * 60)
    
    # Top source candidates
    top_sources = [
        # CRYPTO SOURCES (High Priority)
        {"name": "crypto_panic", "url": "https://cryptopanic.com/news/rss", "category": "crypto_news"},
        {"name": "decrypt_media", "url": "https://decrypt.co/feed", "category": "crypto_news"},
        {"name": "the_block", "url": "https://theblock.co/rss", "category": "crypto_news"},
        {"name": "bitcoin_magazine", "url": "https://bitcoinmagazine.com/feed", "category": "crypto_news"},
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
        {"name": "y_finance", "url": "https://finance.yahoo.com/news/rss", "category": "financial_news"},
        {"name": "finviz", "url": "https://finviz.com/news_rss.ashx", "category": "financial_news"},
        {"name": "street_insider", "url": "https://seekingalpha.com/api/sa/combined?id=352765&count=10", "category": "market_data"},
        
        # TECH SOURCES
        {"name": "tech_crunch", "url": "https://techcrunch.com/feed/", "category": "tech_news"},
        {"name": "arstechnica", "url": "https://arstechnica.com/rss", "category": "tech_news"},
        {"name": "wired", "url": "https://www.wired.com/feed/rss", "category": "tech_news"},
        {"name": "the_verge", "url": "https://www.theverge.com/rss/index.xml", "category": "tech_news"},
        
        # MACRO SOURCES
        {"name": "federal_reserve", "url": "https://www.federalreserve.gov/feeds/pressreleases.xml", "category": "economic_data"},
        {"name": "cpi_data", "url": "https://www.bls.gov/rss/cpi.rss", "category": "economic_data"},
        {"name": "treasury_reports", "url": "https://home.treasury.gov/data.xml", "category": "economic_data"},
        {"name": "bea_gdp", "url": "https://www.bea.gov/news/rss", "category": "economic_data"},
    ]
    
    print(f"Testing {len(top_sources)} top sources...")
    
    # Test each source
    results = []
    for i, source in enumerate(top_sources):
        print(f"\n📡 [{i+1}/{len(top_sources)}] Testing: {source['name']}")
        
        result = test_single_source(source['url'], source['name'], source['category'])
        results.append(result)
        
        print(f"   Status: {result['status']}")
        print(f"   Articles: {result['article_count']}")
        print(f"   Signal Relevance: {result['signal_relevance']}")
        print(f"   Freshness: {result['freshness_hours']}h")
        print(f"   Response Time: {result['response_time']}s")
        
        if result['sample_titles']:
            print(f"   Sample Titles:")
            for title in result['sample_titles'][:2]:
                print(f"      - {title}")
    
    return results


def calculate_final_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate final scores for all sources."""
    scored_sources = []
    
    for result in results:
        score = {
            "name": result["name"],
            "url": result["url"],
            "category": result["category"],
            "total_score": 0.0,
            "reliability_score": 0.0,
            "signal_density_score": 0.0,
            "latency_score": 0.0,
            "freshness_score": 0.0,
            "final_grade": "F"
        }
        
        # Reliability Score (40% of total)
        if result.get("status") == "SUCCESS":
            score["reliability_score"] = 40.0
        else:
            score["reliability_score"] = 0.0
        
        # Signal Density Score (30% of total)
        article_count = result.get("article_count", 0)
        if article_count > 50:
            score["signal_density_score"] = 30.0
        elif article_count > 25:
            score["signal_density_score"] = 25.0
        elif article_count > 10:
            score["signal_density_score"] = 20.0
        elif article_count > 0:
            score["signal_density_score"] = 10.0
        else:
            score["signal_density_score"] = 0.0
        
        # Latency Score (15% of total)
        response_time = result.get("response_time", 10.0)
        if response_time < 1.0:
            score["latency_score"] = 15.0
        elif response_time < 2.0:
            score["latency_score"] = 12.0
        elif response_time < 5.0:
            score["latency_score"] = 8.0
        else:
            score["latency_score"] = 0.0
        
        # Freshness Score (15% of total)
        fresh_hours = result.get("freshness_hours", 24.0)
        if fresh_hours < 6.0:
            score["freshness_score"] = 15.0
        elif fresh_hours < 12.0:
            score["freshness_score"] = 12.0
        elif fresh_hours < 24.0:
            score["freshness_score"] = 8.0
        else:
            score["freshness_score"] = 0.0
        
        # Calculate total score
        score["total_score"] = (
            score["reliability_score"] + 
            score["signal_density_score"] + 
            score["latency_score"] + 
            score["freshness_score"]
        )
        
        # Grade assignment
        if score["total_score"] >= 80.0:
            score["final_grade"] = "A"
        elif score["total_score"] >= 70.0:
            score["final_grade"] = "B"
        elif score["total_score"] >= 60.0:
            score["final_grade"] = "C"
        elif score["total_score"] >= 50.0:
            score["final_grade"] = "D"
        else:
            score["final_grade"] = "F"
        
        scored_sources.append(score)
    
    # Sort by total score
    scored_sources.sort(key=lambda x: x["total_score"], reverse=True)
    return scored_sources


if __name__ == "__main__":
    print("🚀 COMPREHENSIVE SOURCE VALIDATION & SCORING")
    print("=" * 60)
    
    # Phase 2-3: Validation and Scoring
    results = validate_top_sources()
    scored_sources = calculate_final_scores(results)
    
    # Phase 4: Auto-Selection
    print(f"\n\n🎯 AUTO-SELECTION RESULTS:")
    print("=" * 60)
    
    # Grade A sources (keep)
    grade_a = [s for s in scored_sources if s["final_grade"] == "A"]
    # Grade B sources (keep)
    grade_b = [s for s in scored_sources if s["final_grade"] == "B"]
    # Grade C sources (consider)
    grade_c = [s for s in scored_sources if s["final_grade"] == "C"]
    # Grade D-F sources (remove)
    grade_df = [s for s in scored_sources if s["final_grade"] in ["D", "F"]]
    
    print(f"   Grade A (KEEP): {len(grade_a)} sources")
    print(f"   Grade B (KEEP): {len(grade_b)} sources")
    print(f"   Grade C (CONSIDER): {len(grade_c)} sources")
    print(f"   Grade D-F (REMOVE): {len(grade_df)} sources")
    
    # Top 15-25 sources
    top_sources = grade_a + grade_b + grade_c[:8]  # Keep A, B, and top 8 C
    print(f"\n🏆 TOP SOURCES (Total: {len(top_sources)}):")
    
    print(f"{'Name':<20} {'Category':<15} {'Grade':<5} {'Score':<6} {'Rel':<5} {'Density':<8} {'Latency':<9} {'Freshness':<10}")
    print("-" * 100)
    
    for source in top_sources:
        print(f"{source['name']:<19} {source['category']:<14} {source['final_grade']:<5} {source['total_score']:<6.1f} {source['reliability_score']:<5.1f} {source['signal_density_score']:<8.1f} {source['latency_score']:<9.1f} {source['freshness_score']:<10.1f}")
    
    print(f"\n✅ READY FOR PHASE 5-6 IMPLEMENTATION")
    print(f"🎯 FINAL OPTIMIZED SOURCE LIST: {len(top_sources)} sources")
    
    # Generate optimized source list for implementation
    optimized_sources = []
    for source in top_sources:
        optimized_sources.append({
            "name": source["name"],
            "url": source["url"],
            "category": source["category"],
            "priority": 1,
            "enabled": True
        })
    
    print(f"\n🎯 FINAL OPTIMIZED SOURCES:")
    for source in optimized_sources:
        print(f"   {source['name']} ({source['category']}): {source['url']}")
    
    return optimized_sources
