#!/usr/bin/env python3
"""
Realistic signal validation test for TRADING-AI.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.core.orchestrator import PipelineOrchestrator


def run_signal_validation_test(num_runs=10):
    """Run multiple pipeline executions to test signal generation."""
    print(f"🎯 SIGNAL VALIDATION TEST ({num_runs} runs)")
    print("=" * 60)
    
    # Clear state for clean test
    data_dir = Path("./data")
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir()
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Track results across runs
    all_results = []
    all_signals = []
    
    print(f"🚀 Running {num_runs} pipeline executions...")
    
    for run_num in range(num_runs):
        print(f"\n--- RUN {run_num + 1}/{num_runs} ---")
        
        try:
            # Run pipeline
            result = orchestrator.run_pipeline(dry_run=True)
            
            # Collect metrics
            run_data = {
                "run": run_num + 1,
                "status": result.status,
                "articles_processed": result.articles_processed,
                "articles_validated": result.metadata.get("validated_articles", 0),
                "signals_generated": result.signals_generated,
                "pipeline_latency_ms": result.pipeline_latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
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
            print(f"   Articles validated: {result.metadata.get('validated_articles', 0)}")
            print(f"   Signals generated: {result.signals_generated}")
            print(f"   Pipeline latency: {result.pipeline_latency_ms:.1f}ms")
            
            if result.signals_generated > 0:
                print(f"   ✅ SIGNALS GENERATED!")
                signals = result.metadata.get("signals", [])
                for i, signal in enumerate(signals[:3]):  # Show first 3
                    print(f"      Signal {i+1}: {signal.get('symbol')} {signal.get('action')} (confidence: {signal.get('confidence', 0.0):.2f})")
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
    print(f"\n📊 ANALYSIS OF {num_runs} RUNS")
    print("=" * 60)
    
    # Calculate statistics
    successful_runs = [r for r in all_results if r["status"] != "FAILED"]
    failed_runs = [r for r in all_results if r["status"] == "FAILED"]
    
    if successful_runs:
        total_articles = sum(r["articles_processed"] for r in successful_runs)
        total_validated = sum(len(r.get("articles_validated", [])) for r in successful_runs)
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
        else:
            print(f"\n❌ NO SIGNALS GENERATED ACROSS ALL RUNS")
            print("   System may be too strict or market conditions not suitable")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    signal_rate = total_signals / max(total_validated, 1) if successful_runs else 0
    if signal_rate == 0:
        print("   🔧 CRITICAL: No signals generated")
        print("   → Lower validation thresholds")
        print("   → Check signal generation logic")
        print("   → Test with different market data")
        verdict = "BROKEN"
    elif signal_rate < 0.1:  # Less than 10% signal rate
        print("   ⚠️  WARNING: Very low signal generation rate")
        print("   → Consider lowering similarity threshold")
        print("   → Review validation criteria")
        verdict = "NEEDS_ADJUSTMENT"
    elif signal_rate < 0.3:  # 10-30% signal rate
        print("   ✅ GOOD: Reasonable signal generation rate")
        print("   → Monitor signal quality")
        verdict = "READY"
    else:  # High signal rate (>30%)
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
        "verdict": verdict
    }


if __name__ == "__main__":
    print("🎯 TRADING-AI SIGNAL VALIDATION")
    print("=" * 60)
    
    verdict, metrics = run_signal_validation_test(num_runs=10)
    
    # Save results
    results_file = Path("signal_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    if verdict in ["READY", "NEEDS_TUNING"]:
        print("\n🚀 SYSTEM READY FOR PAPER TRADING")
        sys.exit(0)
    else:
        print("\n❌ SYSTEM NEEDS FIXES BEFORE PAPER TRADING")
        sys.exit(1)
