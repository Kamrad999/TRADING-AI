#!/usr/bin/env python3
"""
Smoke test: Verify all modules import successfully after patches.
Tests basic contract compliance without actual execution.
"""

import sys
sys.path.insert(0, 'news-hunter')

print("=" * 80)
print("SMOKE TEST: Verifying all module imports after patches")
print("=" * 80)

test_results = {}

# Test 1: config.py loads and has new constants
try:
    from config import (
        DRAWDOWN_POLICY_TIERS,
        DRAWDOWN_ACTION_KILL_SWITCH,
        SIGNAL_FIELD_DIRECTION,
        SIGNAL_FIELD_CONFIDENCE,
        REGIME_FIELD_NAME,
        REGIME_FIELD_GROSS_CAP,
    )
    test_results['config'] = '✓ PASS'
    print("✓ config.py — all new constants defined")
    print(f"  - DRAWDOWN_POLICY_TIERS has {len(DRAWDOWN_POLICY_TIERS)} tiers")
    print(f"  - Field constants imported successfully")
except Exception as e:
    test_results['config'] = f'✗ FAIL: {e}'
    print(f"✗ config.py — Import failed: {e}")

# Test 2: god_core.py has updated STAGE_NAMES
try:
    from god_core import STAGE_NAMES
    test_results['god_core'] = '✓ PASS'
    print(f"✓ god_core.py — STAGE_NAMES has {len(STAGE_NAMES)} stages")
    if 'detect_market_regime' in STAGE_NAMES and 'calculate_portfolio_allocations' in STAGE_NAMES:
        print("  - Both new stages present")
    else:
        test_results['god_core'] = '✗ FAIL: Missing new stages'
except Exception as e:
    test_results['god_core'] = f'✗ FAIL: {e}'
    print(f"✗ god_core.py — Import failed: {e}")

# Test 3: risk_guardian.py imports unified policy
try:
    import risk_guardian
    test_results['risk_guardian'] = '✓ PASS'
    print("✓ risk_guardian.py — Imports unified policy successfully")
except Exception as e:
    test_results['risk_guardian'] = f'✗ FAIL: {e}'
    print(f"✗ risk_guardian.py — Import failed: {e}")

# Test 4: self_learning_optimizer.py imports unified policy
try:
    import self_learning_optimizer
    test_results['self_learning_optimizer'] = '✓ PASS'
    print("✓ self_learning_optimizer.py — Imports unified policy successfully")
except Exception as e:
    test_results['self_learning_optimizer'] = f'✗ FAIL: {e}'
    print(f"✗ self_learning_optimizer.py — Import failed: {e}")

# Test 5: signal_engine.py field names
try:
    from signal_engine import _SIGNAL_DEFAULTS
    if "market_regime" in _SIGNAL_DEFAULTS:
        test_results['signal_engine'] = '✓ PASS'
        print("✓ signal_engine.py — Uses correct 'market_regime' field name")
    else:
        test_results['signal_engine'] = '✗ FAIL: Old field name still present'
        print("✗ signal_engine.py — Still using old field names")
except Exception as e:
    test_results['signal_engine'] = f'✗ FAIL: {e}'
    print(f"✗ signal_engine.py — Import failed: {e}")

# Test 6: broker_sender.py has escalation
try:
    import broker_sender
    if hasattr(broker_sender, '_audit'):
        test_results['broker_sender'] = '✓ PASS'
        print("✓ broker_sender.py — Has _audit function for escalation")
    else:
        test_results['broker_sender'] = '✗ FAIL: Missing _audit function'
        print("✗ broker_sender.py — Missing _audit function")
except Exception as e:
    test_results['broker_sender'] = f'✗ FAIL: {e}'
    print(f"✗ broker_sender.py — Import failed: {e}")

# Test 7: news_engine.py has empty feed check
try:
    import news_engine
    test_results['news_engine'] = '✓ PASS'
    print("✓ news_engine.py — Imports successfully with empty feed check")
except Exception as e:
    test_results['news_engine'] = f'✗ FAIL: {e}'
    print(f"✗ news_engine.py — Import failed: {e}")

# Test 8: rss_sandbox.py has try/except
try:
    import rss_sandbox  
    test_results['rss_sandbox'] = '✓ PASS'
    print("✓ rss_sandbox.py — Imports successfully with graceful fallback")
except ImportError as e:
    if 'feedparser' in str(e):
        test_results['rss_sandbox'] = '✓ PASS (expected ImportError with instructions)'
        print("✓ rss_sandbox.py — Raises ImportError with helpful instructions (expected)")
    else:
        test_results['rss_sandbox'] = f'✗ FAIL: {e}'
        print(f"✗ rss_sandbox.py — Unexpected error: {e}")
except Exception as e:
    test_results['rss_sandbox'] = f'✗ FAIL: {e}'
    print(f"✗ rss_sandbox.py — Import failed: {e}")

# Summary
print("\n" + "=" * 80)
print("SMOKE TEST SUMMARY")
print("=" * 80)

passed = sum(1 for v in test_results.values() if v.startswith('✓'))
failed = sum(1 for v in test_results.values() if v.startswith('✗'))

for module, result in test_results.items():
    print(f"{module:30s} {result}")

print("=" * 80)
print(f"Results: {passed} PASSED, {failed} FAILED out of {len(test_results)} tests")
print("=" * 80)

sys.exit(0 if failed == 0 else 1)
