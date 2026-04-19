#!/usr/bin/env python3
"""
Final comprehensive signal validation test with working RSS feeds.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.core.orchestrator import PipelineOrchestrator
from trading_ai.agents.signal_generator import SignalGenerator
from trading_ai.core.models import Article


def run_final_signal_validation(num_runs=5):
    """Run final signal validation test with comprehensive analysis."""
    print(f"🎯 FINAL SIGNAL VALIDATION TEST ({num_runs} runs)")
    print("=" * 60)
    
    # Clear ALL state completely
    data_dir = Path("./data")
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir()
    print("🗑️  All state cleared completely")
    
    # Track results across runs
    all_results = []
    all_signals = []
    
    print(f"🚀 Running {num_runs} pipeline executions with comprehensive analysis...")
    
    for run_num in range(num_runs):
        print(f"\n--- RUN {run_num + 1}/{num_runs} ---")
        
        # Clear state before each run
        if run_num > 0:
            if data_dir.exists():
                shutil.rmtree(data_dir)
            data_dir.mkdir()
            print("   🗑️  State cleared for fresh run")
        
        try:
            orchestrator = PipelineOrchestrator()
            
            # Adjust thresholds for signal generation
            orchestrator.duplicate_filter.title_similarity_threshold = 0.8  # Lower threshold
            orchestrator.duplicate_filter.timestamp_window_hours = 1  # Shorter window
            
            print(f"   📋 Adjusted thresholds: similarity={orchestrator.duplicate_filter.title_similarity_threshold}, window={orchestrator.duplicate_filter.timestamp_window_hours}h")
            
            # Run pipeline
            result = orchestrator.run_pipeline(dry_run=True)
            
            # Collect metrics
            run_data = {
                "run": run_num + 1,
                "status": result.status,
                "articles_processed": result.articles_processed,
                "articles_validated": len(result.metadata.get("validated_articles", [])),
                "signals_generated": result.signals_generated,
                "pipeline_latency_ms": result.pipeline_latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "similarity_threshold": orchestrator.duplicate_filter.title_similarity_threshold,
                "time_window_hours": orchestrator.duplicate_filter.timestamp_window_hours
            }
            
            all_results.append(run_data)
            
            # Collect signals if any
            if result.signals_generated > 0:
                signals = result.metadata.get("signals", [])
                for signal in signals:
                    signal_data = {
                        "run": run_num + 1,
                        "symbol": signal.get("symbol", "UNKNOWN"),
                        "action": signal.get("action", "UNKNOWN"),
                        "confidence": signal.get("confidence", 0.0),
                        "reason": signal.get("reason", "No reason"),
                        "timestamp": signal.get("timestamp", datetime.now(timezone.utc).isoformat())
                    }
                    all_signals.append(signal_data)
            
            # Print run summary
            print(f"   Status: {result.status}")
            print(f"   Articles processed: {result.articles_processed}")
            print(f"   Articles validated: {len(result.metadata.get('validated_articles', []))}")
            print(f"   Signals generated: {result.signals_generated}")
            print(f"   Pipeline latency: {result.pipeline_latency_ms:.1f}ms")
            
            if result.signals_generated > 0:
                print(f"   ✅ SIGNALS GENERATED!")
                signals = result.metadata.get("signals", [])
                for i, signal in enumerate(signals[:3]):  # Show first 3
                    print(f"      Signal {i+1}: {signal.get('symbol')} {signal.get('action')} (confidence: {signal.get('confidence', 0.0):.2f})")
                    print(f"         Reason: {signal.get('reason', 'No reason')}")
            else:
                print(f"   ⚠️  No signals generated")
                
        except Exception as e:
            print(f"   ❌ Run failed: {e}")
            run_data = {
                "run": run_num + 1,
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            all_results.append(run_data)
    
    # Analysis
    print(f"\n📊 FINAL ANALYSIS OF {num_runs} RUNS")
    print("=" * 60)
    
    # Calculate statistics
    successful_runs = [r for r in all_results if r["status"] != "FAILED"]
    failed_runs = [r for r in all_results if r["status"] == "FAILED"]
    
    if successful_runs:
        total_articles = sum(r["articles_processed"] for r in successful_runs)
        total_validated = sum(r["articles_validated"] for r in successful_runs)
        total_signals = sum(r["signals_generated"] for r in successful_runs)
        avg_latency = sum(r["pipeline_latency_ms"] for r in successful_runs) / len(successful_runs)
        
        print(f"✅ Successful runs: {len(successful_runs)}/{num_runs}")
        print(f"❌ Failed runs: {len(failed_runs)}/{num_runs}")
        print(f"📰 Total articles processed: {total_articles}")
        print(f"✅ Total articles validated: {total_validated}")
        print(f"🎯 Total signals generated: {total_signals}")
        print(f"⚡ Average pipeline latency: {avg_latency:.1f}ms")
        
        # Signal analysis
        if all_signals:
            print(f"\n🎯 SIGNAL ANALYSIS ({len(all_signals)} total signals):")
            
            # Group by symbol
            symbols = {}
            for signal in all_signals:
                symbol = signal["symbol"]
                if symbol not in symbols:
                    symbols[symbol] = []
                symbols[symbol].append(signal)
            
            print(f"   Unique symbols: {len(symbols)}")
            for symbol, signals_list in sorted(symbols.items()):
                avg_confidence = sum(s["confidence"] for s in signals_list) / len(signals_list)
                actions = [s["action"] for s in signals_list]
                buy_count = actions.count("BUY")
                sell_count = actions.count("SELL")
                
                print(f"   {symbol}: {len(signals_list)} signals (BUY: {buy_count}, SELL: {sell_count})")
                print(f"      Average confidence: {avg_confidence:.2f}")
                
                # Show sample signals
                for signal in signals_list[:2]:
                    print(f"      Sample: {signal['action']} @ confidence {signal['confidence']:.2f}")
                    print(f"         Reason: {signal['reason']}")
        else:
            print(f"\n❌ NO SIGNALS GENERATED ACROSS ALL RUNS")
    
    # Test signal generation with sample data
    print(f"\n🧪 TESTING SIGNAL GENERATION WITH SAMPLE DATA")
    generator = SignalGenerator()
    
    # Create test articles with clear signals
    test_articles = [
        Article(
            title="Bitcoin surges to new all-time high as institutional buying accelerates",
            content="Bitcoin reached $75,000 today as major institutional investors announced massive BTC purchases. Market analysts predict continued upward momentum.",
            source="Test Source",
            timestamp=datetime.now(timezone.utc),
            url="https://test.com/bitcoin-surge",
            metadata={"validation": {"confidence_score": 0.8}}
        ),
        Article(
            title="Stock market crashes as recession fears grow",
            content="Major stock indices fell sharply today as investors panicked over potential economic recession. Dow Jones dropped 1000 points in trading.",
            source="Test Source",
            timestamp=datetime.now(timezone.utc),
            url="https://test.com/market-crash",
            metadata={"validation": {"confidence_score": 0.8}}
        ),
        Article(
            title="Federal Reserve signals interest rate hike coming",
            content="Fed officials indicated that interest rates may rise next month to combat inflation. Markets reacted negatively to the news.",
            source="Test Source",
            timestamp=datetime.now(timezone.utc),
            url="https://test.com/fed-signals",
            metadata={"validation": {"confidence_score": 0.8}}
        )
    ]
    
    test_signals = generator.generate_signals(test_articles)
    print(f"   Test articles: {len(test_articles)}")
    print(f"   Generated signals: {len(test_signals)}")
    
    if test_signals:
        print(f"   ✅ SIGNAL GENERATION WORKING!")
        for i, signal in enumerate(test_signals):
            print(f"      Signal {i+1}: {signal.symbol} {signal.direction} (confidence: {signal.confidence:.2f})")
            print(f"         Reason: {signal.reason}")
    else:
        print(f"   ❌ SIGNAL GENERATION BROKEN")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    signal_rate = total_signals / max(total_validated, 1) if successful_runs else 0
    test_signal_rate = len(test_signals) / len(test_articles)
    
    if signal_rate == 0 and test_signal_rate == 0:
        print("   🔧 CRITICAL: No signals generated in any scenario")
        print("   → Signal generation logic is broken")
        print("   → Check keyword matching")
        print("   → Verify confidence calculation")
        verdict = "BROKEN"
    elif signal_rate == 0 and test_signal_rate > 0:
        print("   ⚠️  WARNING: No signals from RSS but test data works")
        print("   → RSS feeds may be blocked/empty")
        print("   → Check feed accessibility")
        verdict = "RSS_ISSUE"
    elif signal_rate < 0.05:  # Less than 5% signal rate
        print("   ⚠️  WARNING: Very low signal generation rate")
        print("   → Lower similarity threshold further (try 0.7)")
        print("   → Review validation criteria")
        verdict = "NEEDS_ADJUSTMENT"
    elif signal_rate < 0.2:  # 5-20% signal rate
        print("   ✅ GOOD: Reasonable signal generation rate")
        print("   → Monitor signal quality")
        print("   → Current thresholds working well")
        verdict = "READY"
    else:  # High signal rate (>20%)
        print("   ⚠️  WARNING: High signal generation rate")
        print("   → May generate too many false signals")
        print("   → Consider raising validation threshold")
        verdict = "NEEDS_TUNING"
    
    print(f"\n🎯 FINAL VERDICT: {verdict}")
    
    return verdict, {
        "total_runs": num_runs,
        "successful_runs": len(successful_runs),
        "total_articles": total_articles if successful_runs else 0,
        "total_validated": total_validated if successful_runs else 0,
        "total_signals": total_signals,
        "signal_rate": signal_rate,
        "test_signal_rate": test_signal_rate,
        "verdict": verdict
    }


if __name__ == "__main__":
    print("🎯 TRADING-AI FINAL SIGNAL VALIDATION")
    print("=" * 60)
    
    verdict, metrics = run_final_signal_validation(num_runs=5)
    
    # Save results
    results_file = Path("signal_validation_final_results.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    if verdict in ["READY", "NEEDS_TUNING"]:
        print("\n🚀 SYSTEM READY FOR PAPER TRADING")
        sys.exit(0)
    else:
        print("\n❌ SYSTEM NEEDS FIXES BEFORE PAPER TRADING")
        sys.exit(1)
