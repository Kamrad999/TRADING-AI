#!/usr/bin/env python3
"""
Test keyword matching directly to debug signal generation.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.agents.signal_generator import SignalGenerator
from trading_ai.core.models import Article
from datetime import datetime, timezone


def test_keyword_matching():
    """Test keyword matching directly."""
    print("🔍 KEYWORD MATCHING DEBUG")
    print("=" * 50)
    
    generator = SignalGenerator()
    
    # Test text with clear keywords
    test_text = "bitcoin surges to new all-time high as institutional buying accelerates"
    print(f"📝 Test text: {test_text}")
    
    # Test bullish scoring
    bullish_score = generator._calculate_keyword_score(test_text, generator.bullish_keywords)
    print(f"📈 Bullish score: {bullish_score:.3f}")
    
    # Test bearish scoring  
    bearish_score = generator._calculate_keyword_score(test_text, generator.bearish_keywords)
    print(f"📉 Bearish score: {bearish_score:.3f}")
    
    # Show found keywords
    found_keywords = generator._get_found_keywords(test_text)
    print(f"🔤 Found keywords: {found_keywords}")
    
    # Test each keyword individually
    print(f"\n🔍 INDIVIDUAL KEYWORD TESTS:")
    for keyword, impact in generator.bullish_keywords.items():
        pattern = r'\b' + keyword + r'\b'
        matches = __import__('re').findall(pattern, test_text, __import__('re').IGNORECASE)
        print(f"   '{keyword}': {len(matches)} matches (impact: {impact})")
    
    # Test with sample article
    print(f"\n📰 SAMPLE ARTICLE TEST:")
    article = Article(
        title="Bitcoin surges to new all-time high",
        content="Bitcoin reached $75,000 as institutional buying accelerates",
        source="Test Source",
        timestamp=datetime.now(timezone.utc),
        url="https://test.com/bitcoin-surge",
        metadata={"validation": {"confidence_score": 0.8}}
    )
    
    signals = generator.generate_signals([article])
    print(f"   Generated signals: {len(signals)}")
    
    for signal in signals:
        print(f"   Signal: {signal.symbol} {signal.action} (confidence: {signal.confidence:.2f})")
        print(f"   Reason: {signal.reason}")


if __name__ == "__main__":
    test_keyword_matching()
