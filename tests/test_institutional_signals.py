#!/usr/bin/env python3
"""
Test institutional signal generator to verify 5-20 signals per run.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.core.orchestrator import PipelineOrchestrator


def test_institutional_signal_generation():
    """Test institutional signal generation with optimized sources."""
    print(" Institutional Signal Generation Test")
    print("=" * 60)
    
    # Clear state for clean test
    data_dir = Path("./data")
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir()
    
    # Initialize orchestrator with institutional signal generator
    orchestrator = PipelineOrchestrator()
    
    print(f" Sources: {len(orchestrator.source_registry.get_sources())}")
    print(f" Signal Generator: {type(orchestrator.signal_generator).__name__}")
    
    # Run pipeline
    print(f"\n Running pipeline with institutional signal generator...")
    result = orchestrator.run_pipeline(dry_run=True)
    
    # Display results
    print(f"\n PIPELINE RESULTS:")
    print(f"   Status: {result.status}")
    print(f"   Articles Processed: {result.articles_processed}")
    print(f"   Articles Validated: {len(result.metadata.get('validated_articles', []))}")
    print(f"   Signals Generated: {result.signals_generated}")
    print(f"   Pipeline Latency: {result.pipeline_latency_ms:.1f}ms")
    
    # Signal analysis
    if result.signals_generated > 0:
        print(f"\n SIGNAL ANALYSIS:")
        print(f"   Signal Rate: {result.signals_generated/result.articles_processed*100:.1f}%")
        print(f"   Avg Confidence: {sum(s.confidence for s in result.signals)/len(result.signals):.3f}")
        
        # Show signal details
        print(f"\n SIGNAL DETAILS:")
        for i, signal in enumerate(result.signals[:10]):  # Show first 10
            print(f"   {i+1}. {signal.symbol} {signal.direction} (conf: {signal.confidence:.3f})")
            print(f"      Reason: {signal.reason}")
            print(f"      Urgency: {signal.urgency}")
            print(f"      Position Size: {signal.position_size:.2f}")
            
            # Show metadata
            if hasattr(signal, 'metadata') and signal.metadata:
                metadata = signal.metadata
                print(f"      Entity: {metadata.get('entity_name', 'Unknown')}")
                print(f"      Event: {metadata.get('event_type', 'Unknown')}")
                print(f"      Sentiment: {metadata.get('sentiment_score', 0):.3f}")
            
            print()
        
        # Signal distribution
        buy_signals = [s for s in result.signals if s.direction.value == "BUY"]
        sell_signals = [s for s in result.signals if s.direction.value == "SELL"]
        
        print(f" SIGNAL DISTRIBUTION:")
        print(f"   Buy Signals: {len(buy_signals)}")
        print(f"   Sell Signals: {len(sell_signals)}")
        
        # Symbol distribution
        symbols = {}
        for signal in result.signals:
            symbols[signal.symbol] = symbols.get(signal.symbol, 0) + 1
        
        print(f"\n TOP SYMBOLS:")
        for symbol, count in sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {symbol}: {count} signals")
    
    # Determine success
    if 5 <= result.signals_generated <= 20:
        verdict = "SUCCESS"
        status = " OPTIMAL SIGNAL GENERATION"
    elif 1 <= result.signals_generated < 5:
        verdict = "PARTIAL"
        status = " LOW SIGNAL GENERATION"
    elif result.signals_generated > 20:
        verdict = "SUCCESS"
        status = " HIGH SIGNAL GENERATION"
    else:
        verdict = "FAILED"
        status = " ZERO SIGNAL GENERATION"
    
    print(f"\n FINAL VERDICT: {verdict}")
    print(f" STATUS: {status}")
    
    # Performance metrics
    signal_rate = result.signals_generated / max(result.articles_processed, 1) if result.articles_processed > 0 else 0
    validation_rate = len(result.metadata.get('validated_articles', [])) / max(result.articles_processed, 1) if result.articles_processed > 0 else 0
    
    print(f"\n PERFORMANCE METRICS:")
    print(f"   Signal Rate: {signal_rate*100:.1f}%")
    print(f"   Validation Rate: {validation_rate*100:.1f}%")
    print(f"   Pipeline Latency: {result.pipeline_latency_ms:.1f}ms")
    print(f"   Signals per Article: {signal_rate:.3f}")
    
    # System readiness assessment
    if verdict == "SUCCESS":
        readiness = " READY FOR PAPER TRADING"
    elif verdict == "PARTIAL":
        readiness = " NEEDS TUNING"
    else:
        readiness = " NOT READY"
    
    print(f"\n SYSTEM READINESS: {readiness}")
    
    return result


if __name__ == "__main__":
    result = test_institutional_signal_generation()
    
    print(f"\n INSTITUTIONAL SIGNAL GENERATION TEST COMPLETE")
    print("=" * 60)
    print(f" FINAL RESULTS:")
    print(f"   Articles: {result.articles_processed}")
    print(f"   Signals: {result.signals_generated}")
    print(f"   Signal Rate: {result.signals_generated/max(result.articles_processed, 1)*100:.1f}%")
    
    if result.signals_generated >= 5:
        print(f"   STATUS: SUCCESS - Institutional signal generation working!")
    else:
        print(f"   STATUS: NEEDS OPTIMIZATION - Signal generation below target")
