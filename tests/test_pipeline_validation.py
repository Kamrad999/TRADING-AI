#!/usr/bin/env python3
"""
End-to-end pipeline validation test.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.core.orchestrator import PipelineOrchestrator


def test_full_pipeline():
    """Test complete pipeline functionality."""
    print("🚀 Testing full pipeline functionality...")
    
    # Clear any existing state
    state_file = Path("./data/state.json")
    if state_file.exists():
        os.remove(state_file)
        print("🗑️  Cleared existing state file")
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    print("\n📊 Running pipeline (dry run)...")
    
    try:
        # Run full pipeline
        result = orchestrator.run_pipeline(dry_run=True)
        
        print(f"✅ Pipeline completed successfully!")
        print(f"   Status: {result.status}")
        print(f"   Articles processed: {result.articles_processed}")
        print(f"   Signals generated: {result.signals_generated}")
        print(f"   Orders sent: {result.orders_sent}")
        print(f"   Alerts sent: {result.alerts_sent}")
        print(f"   Pipeline latency: {result.pipeline_latency_ms:.1f}ms")
        print(f"   Pipeline ID: {result.pipeline_id}")
        print(f"   Error message: {result.error_message}")
        
        # Check for critical issues
        if result.articles_processed > 0 and result.status == PipelineStatus.SUCCESS:
            print("❌ CRITICAL: No articles processed")
            return False
        
        if result.signals_generated == 0:
            print("⚠️  WARNING: No signals generated (may be normal)")
        
        # Get system status
        status = orchestrator.get_system_status()
        print(f"\n📈 System Status:")
        print(f"   Kill switch active: {status.kill_switch_active}")
        print(f"   Market session: {status.market_session}")
        print(f"   Portfolio exposure: {status.portfolio_exposure_pct:.1f}%")
        print(f"   Daily drawdown: {status.daily_drawdown_pct:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_quality():
    """Test signal generation quality."""
    print("\n🎯 Testing signal quality...")
    
    orchestrator = PipelineOrchestrator()
    
    try:
        # Run pipeline to generate signals
        result = orchestrator.run_pipeline(dry_run=True)
        
        if result.success and result.signals_generated > 0:
            print(f"✅ Generated {result.signals_generated} signals")
            
            # Check signal structure in metadata
            if 'signals' in result.metadata:
                signals = result.metadata['signals']
                for i, signal in enumerate(signals[:3]):  # Check first 3 signals
                    print(f"   Signal {i+1}: {signal.get('symbol', 'N/A')} - {signal.get('action', 'N/A')} (confidence: {signal.get('confidence', 'N/A')})")
            
            return True
        else:
            print("⚠️  No signals generated (may be normal for current market data)")
            return True
            
    except Exception as e:
        print(f"❌ Signal quality test failed: {e}")
        return False


def test_duplicate_filter_behavior():
    """Test duplicate filter behavior."""
    print("\n🔍 Testing duplicate filter behavior...")
    
    from trading_ai.validation.duplicate_filter import DuplicateFilter
    from trading_ai.core.models import Article
    
    # Clear state
    filter = DuplicateFilter()
    
    # Test with identical articles
    article1 = Article(
        title="Test Article",
        content="Test content",
        source="Test Source",
        timestamp=datetime.now(timezone.utc),
        url="https://test.com/article1",
        metadata={}
    )
    
    article2 = Article(
        title="Test Article",
        content="Test content",
        source="Test Source",
        timestamp=datetime.now(timezone.utc),
        url="https://test.com/article2",  # Different URL
        metadata={}
    )
    
    # First should pass
    unique1 = filter.filter_duplicates([article1])
    print(f"   First article unique: {len(unique1) == 1}")
    
    # Second should be duplicate (same title)
    unique2 = filter.filter_duplicates([article2])
    print(f"   Duplicate article filtered: {len(unique2) == 0}")
    
    # Test with different title
    article3 = Article(
        title="Different Article",
        content="Different content",
        source="Test Source",
        timestamp=datetime.now(timezone.utc),
        url="https://test.com/article3",
        metadata={}
    )
    
    unique3 = filter.filter_duplicates([article3])
    print(f"   Different article unique: {len(unique3) == 1}")
    
    stats = filter.get_duplicate_stats()
    print(f"   Final stats: {stats['seen_articles_count']} seen, {stats['url_hashes_cached']} URLs")
    
    return len(unique1) == 1 and len(unique2) == 0 and len(unique3) == 1


if __name__ == "__main__":
    print("🔧 PIPELINE VALIDATION")
    print("=" * 50)
    
    test1_passed = test_full_pipeline()
    test2_passed = test_signal_quality()
    test3_passed = test_duplicate_filter_behavior()
    
    print(f"\n🎯 OVERALL RESULT:")
    if test1_passed and test2_passed and test3_passed:
        print("✅ ALL PIPELINE TESTS PASSED")
        print("🚀 SYSTEM READY FOR PAPER TRADING")
        sys.exit(0)
    else:
        print("❌ PIPELINE TESTS FAILED")
        print("🔧 FIXES REQUIRED BEFORE PAPER TRADING")
        sys.exit(1)
