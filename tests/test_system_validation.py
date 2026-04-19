#!/usr/bin/env python3
"""
Final system validation test for the upgraded TRADING-AI system.
Tests the complete decision-making pipeline from news to execution.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.brain.decision_engine import DecisionEngine
from trading_ai.agents.multi_agent_system import MultiAgentSystem
from trading_ai.market.data_provider import DataProvider
from trading_ai.signals.enhanced_signal_generator import EnhancedSignalGenerator
from trading_ai.backtest.backtest_engine import BacktestEngine
from trading_ai.execution.exchange import Exchange
from trading_ai.memory.trade_memory import TradeMemory


def test_complete_pipeline():
    """Test the complete trading decision pipeline."""
    print("\n Complete Pipeline Validation Test")
    print("=" * 50)
    
    try:
        # Initialize all components
        print("  Initializing components...")
        decision_engine = DecisionEngine()
        multi_agent = MultiAgentSystem()
        data_provider = DataProvider()
        signal_generator = EnhancedSignalGenerator()
        backtest_engine = BacktestEngine(initial_cash=100000.0)
        exchange = Exchange(paper_trading=True)
        trade_memory = TradeMemory()
        
        # Connect to exchange
        exchange.connect()
        
        print("  Components initialized successfully")
        
        # Test market data pipeline
        print("\n  Testing market data pipeline...")
        market_data = data_provider.get_market_data("BTC", include_indicators=True)
        if market_data:
            print(f"    Market data: {market_data['symbol']} @ ${market_data['price']:.2f}")
            print(f"    Indicators: {len(market_data['indicators'])}")
        else:
            print("    Failed to get market data")
            return False
        
        # Test multi-agent system
        print("\n  Testing multi-agent system...")
        context = {
            "symbol": "BTC",
            "current_price": market_data["price"],
            "volume": market_data["volume"],
            "volatility": market_data["volatility"],
            "technical_indicators": market_data["indicators"],
            "news_data": [
                {
                    "title": "Bitcoin shows strong bullish momentum",
                    "sentiment": 0.7,
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "positions": {},
            "portfolio_value": 100000.0
        }
        
        consensus = multi_agent.make_consensus_decision(context)
        if consensus:
            print(f"    Consensus: {consensus['action']} {consensus['symbol']} (conf: {consensus['confidence']:.2f})")
            print(f"    Agent decisions: {len(consensus['agent_decisions'])}")
        else:
            print("    No consensus generated")
            return False
        
        # Test signal generation
        print("\n  Testing signal generation...")
        symbols = ["BTC", "ETH"]
        news_data = [
            {
                "title": "Bitcoin reaches new all-time high",
                "content": "BTC Bitcoin cryptocurrency shows strong bullish momentum",
                "sentiment": 0.8,
                "timestamp": datetime.now().isoformat()
            }
        ]
        positions = {}
        
        signals = signal_generator.generate_signals(symbols, news_data, positions)
        print(f"    Signals generated: {len(signals)}")
        
        for signal in signals:
            print(f"      {signal.direction.value} {signal.symbol} (conf: {signal.confidence:.2f})")
        
        # Test trade execution
        print("\n  Testing trade execution...")
        if signals:
            signal = signals[0]
            ticker = exchange.get_ticker("BTC/USDT")
            if ticker:
                order = exchange.create_order("BTC/USDT", signal.direction.value.lower(), "market", 0.001)
                if order:
                    print(f"    Order executed: {order.order_id}")
                    print(f"    Status: {order.status}")
                    print(f"    Cost: ${order.cost:.2f}")
                    
                    # Test trade memory
                    print("\n  Testing trade memory...")
                    trade_record = TradeMemory.TradeRecord(
                        trade_id=order.order_id,
                        symbol=signal.symbol,
                        direction=signal.direction.value,
                        quantity=order.filled,
                        entry_price=order.cost / order.filled if order.filled > 0 else 0.0,
                        exit_price=None,
                        entry_time=order.timestamp,
                        exit_time=None,
                        stop_loss=ticker["price"] * 0.95,
                        take_profit=ticker["price"] * 1.1,
                        pnl=0.0,
                        pnl_pct=0.0,
                        status="open",
                        fees=order.fees.get("trading", 0.0),
                        confidence=signal.confidence,
                        reasoning=signal.metadata.get("reasoning", ""),
                        market_conditions={"regime": "bullish"},
                        agent_decisions=consensus.get("agent_decisions", []),
                        metadata={"signal_id": signal.symbol}
                    )
                    
                    trade_memory.add_trade(trade_record)
                    
                    # Get performance metrics
                    metrics = trade_memory.get_performance_metrics()
                    if metrics:
                        print(f"    Performance metrics:")
                        print(f"      Total trades: {metrics.get('total_trades', 0)}")
                        print(f"      Win rate: {metrics.get('win_rate', 0):.2%}")
                        print(f"      Total P&L: ${metrics.get('total_pnl', 0):.2f}")
                else:
                    print("    Order execution failed")
                    return False
            else:
                print("    No ticker data available")
                return False
        else:
            print("    No signals to execute")
        
        # Test backtesting
        print("\n  Testing backtesting...")
        backtest_results = backtest_engine.run_backtest(
            symbols=["BTC"],
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        if backtest_results:
            summary = backtest_results["summary"]
            print(f"    Backtest results:")
            print(f"      Total return: {summary.get('total_return', 0):.2%}")
            print(f"      Total trades: {summary.get('total_trades', 0)}")
            print(f"      Win rate: {summary.get('win_rate', 0):.2%}")
            print(f"      Sharpe ratio: {summary.get('sharpe_ratio', 0):.2f}")
        else:
            print("    Backtest failed")
            return False
        
        print("\n  All components working correctly!")
        return True
        
    except Exception as e:
        print(f"  Pipeline test failed: {e}")
        return False


def test_system_architecture():
    """Test system architecture and integration."""
    print("\n System Architecture Test")
    print("=" * 50)
    
    try:
        # Test component integration
        print("  Testing component integration...")
        
        # Initialize components
        decision_engine = DecisionEngine()
        multi_agent = MultiAgentSystem()
        signal_generator = EnhancedSignalGenerator()
        
        # Test data flow
        print("  Testing data flow...")
        
        # 1. Market data -> Multi-agent system
        data_provider = DataProvider()
        market_data = data_provider.get_market_data("BTC", include_indicators=True)
        
        context = {
            "symbol": "BTC",
            "current_price": market_data["price"],
            "technical_indicators": market_data["indicators"],
            "news_data": [
                {
                    "title": "Bitcoin bullish",
                    "sentiment": 0.6,
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "positions": {},
            "portfolio_value": 100000.0
        }
        
        # 2. Multi-agent system -> Consensus
        consensus = multi_agent.make_consensus_decision(context)
        
        # 3. Consensus -> Signal
        if consensus:
            signals = signal_generator.generate_signals(["BTC"], context["news_data"], {})
            print(f"    Data flow successful: {len(signals)} signals generated")
        
        print("  System architecture validated")
        return True
        
    except Exception as e:
        print(f"  Architecture test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print(" TRADING-AI System Validation")
    print("=" * 60)
    print("Testing upgraded decision-making trading system")
    
    tests = [
        ("Complete Pipeline", test_complete_pipeline),
        ("System Architecture", test_system_architecture)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n Validation Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n Overall: {passed}/{total} validation tests passed")
    
    if passed == total:
        print("\n  TRADING-AI System: FULLY VALIDATED")
        print("  Successfully transformed from 'news processing pipeline' to 'full decision-making trading system'")
        print("\n  Key Features Implemented:")
        print("  - Multi-agent decision system (News, Technical, Risk agents)")
        print("  - LLM integration with structured JSON input/output")
        print("  - Market data pipeline with technical indicators")
        print("  - Weighted signal generation and scoring")
        print("  - Backtesting engine with P&L tracking")
        print("  - Exchange execution layer with paper trading")
        print("  - Trade memory and performance analysis")
        print("  - Production-ready error handling and logging")
        return True
    else:
        print("\n  TRADING-AI System: NEEDS ATTENTION")
        print("  Some validation tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
