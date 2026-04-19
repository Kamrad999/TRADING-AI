#!/usr/bin/env python3
"""
Debug regex pattern matching for keywords.
"""

import re

def test_regex_patterns():
    """Test regex patterns for keyword matching."""
    print("🔍 REGEX PATTERN DEBUG")
    print("=" * 40)
    
    test_text = "bitcoin surges to new all-time high"
    keyword = "surge"
    
    # Test current pattern
    pattern1 = r'\b' + re.escape(keyword) + r'\b'
    matches1 = re.findall(pattern1, test_text, re.IGNORECASE)
    print(f"📝 Pattern 1: {pattern1}")
    print(f"   Matches: {matches1}")
    print(f"   Count: {len(matches1)}")
    
    # Test alternative pattern
    pattern2 = r'\b' + re.escape(keyword) + r'\b'
    matches2 = re.findall(pattern2, test_text.lower(), re.IGNORECASE)
    print(f"\n📝 Pattern 2 (lowercase): {pattern2}")
    print(f"   Matches: {matches2}")
    print(f"   Count: {len(matches2)}")
    
    # Test simple search
    pattern3 = keyword
    matches3 = test_text.lower().count(keyword.lower())
    print(f"\n📝 Simple search: {keyword}")
    print(f"   Count: {matches3}")
    
    # Test with word boundaries manually
    words = test_text.lower().split()
    matches4 = [i for i, word in enumerate(words) if word == keyword.lower()]
    print(f"\n📝 Manual word split: {keyword}")
    print(f"   Count: {len(matches4)}")
    print(f"   Word positions: {[i for i, word in enumerate(words) if word == keyword.lower()]}")
    
    # Show the actual text
    print(f"\n📝 Original text: {test_text}")
    print(f"📝 Lowercase text: {test_text.lower()}")
    print(f"📝 Split words: {words}")


if __name__ == "__main__":
    test_regex_patterns()
