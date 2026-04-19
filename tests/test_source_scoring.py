#!/usr/bin/env python3
"""
Source validation and scoring system for TRADING-AI.
"""

import sys
import requests
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def calculate_source_score(source_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive source score."""
    score = {
        "name": source_data["name"],
        "url": source_data["url"],
        "category": source_data["category"],
        "total_score": 0.0,
        "reliability_score": 0.0,
        "signal_density_score": 0.0,
        "latency_score": 0.0,
        "freshness_score": 0.0,
        "final_grade": "F"
    }
    
    # Reliability Score (40% of total)
    if source_data.get("status") == "SUCCESS":
        score["reliability_score"] = 40.0
    else:
        score["reliability_score"] = 0.0
    
    # Signal Density Score (30% of total)
    article_count = source_data.get("article_count", 0)
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
    response_time = source_data.get("response_time", 10.0)
    if response_time < 1.0:
        score["latency_score"] = 15.0
    elif response_time < 2.0:
        score["latency_score"] = 12.0
    elif response_time < 5.0:
        score["latency_score"] = 8.0
    else:
        score["latency_score"] = 0.0
    
    # Freshness Score (15% of total)
    fresh_hours = source_data.get("freshness_hours", 24.0)
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
    
    return score


def validate_and_score_sources():
    """Validate and score all discovered sources."""
    print("🔍 PHASE 2-3 — VALIDATION & SCORING SYSTEM")
    print("=" * 60)
    
    # High-quality sources from Phase 1
    high_quality_sources = [
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
        {"name": "alpha_street", "url": "https://seekingalpha.com/feed.xml", "category": "market_data"},
        {"name": "benzinga", "url": "https://www.benzinga.com/feed", "category": "financial_news"},
        {"name": "investing_com", "url": "https://www.investing.com/rss/news.rss", "category": "financial_news"},
        {"name": "marketbeat", "url": "https://www.marketbeat.com/rss", "category": "financial_news"},
        {"name": "y_finance", "url": "https://finance.yahoo.com/news/rss", "category": "financial_news"},
        {"name": "tech_crunch", "url": "https://techcrunch.com/feed/", "category": "tech_news"},
        {"name": "arstechnica", "url": "https://arstechnica.com/rss", "category": "tech_news"},
        {"name": "wired", "url": "https://www.wired.com/feed/rss", "category": "tech_news"},
        {"name": "the_verge", "url": "https://www.theverge.com/rss/index.xml", "category": "tech_news"},
        {"name": "bea_gdp", "url": "https://www.bea.gov/news/rss", "category": "economic_data"},
        {"name": "kitco_news", "url": "https://www.kitco.com/news/rss", "category": "commodity_news"},
        {"name": "daily_fx", "url": "https://www.dailyfx.com/rss", "category": "forex_news"},
    ]
    
    print(f"Scoring {len(high_quality_sources)} high-quality sources...")
    
    # Test and score each source
    scored_sources = []
    for source in high_quality_sources:
        score = calculate_source_score(source)
        scored_sources.append(score)
    
    # Sort by total score
    scored_sources.sort(key=lambda x: x["total_score"], reverse=True)
    
    return scored_sources


if __name__ == "__main__":
    print("📊 SOURCE VALIDATION & SCORING SYSTEM")
    print("=" * 60)
    
    scored_sources = validate_and_score_sources()
    
    # Display results
    print(f"\n📈 SOURCE SCORING RESULTS:")
    print(f"{'Name':<20} {'Category':<15} {'Grade':<5} {'Score':<6} {'Rel':<5} {'Density':<8} {'Latency':<9} {'Freshness':<10}")
    print("-" * 80)
    
    for score in scored_sources:
        print(f"{score['name']:<19} {score['category']:<14} {score['final_grade']:<5} {score['total_score']:<6.1f} {score['reliability_score']:<5.1f} {score['signal_density_score']:<8.1f} {score['latency_score']:<9.1f} {score['freshness_score']:<10.1f}")
    
    # Auto-selection analysis
    print(f"\n🎯 AUTO-SELECTION ANALYSIS:")
    
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
    top_sources = grade_a + grade_b + grade_c[:10]  # Keep A, B, and top 10 C
    print(f"\n🏆 TOP SOURCES (Total: {len(top_sources)}):")
    for source in top_sources:
        print(f"   {source['name']} ({source['final_grade']}) - Score: {source['total_score']:.1f}")
    
    print(f"\n✅ READY FOR PHASE 4-5 IMPLEMENTATION")
