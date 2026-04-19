#!/usr/bin/env python3
"""
Comprehensive test suite for the new decision-making trading system.
Tests all components: decision engine, multi-agent system, backtesting, execution, memory.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_ai.brain.decision_engine import DecisionEngine
from trading_ai.brain.llm_client import LLMClient
from trading_ai.brain.market_context import MarketContext
from trading_ai.agents.multi_agent_system import MultiAgentSystem
from trading_ai.market.data_provider import DataProvider
from trading_ai.signals.enhanced_signal_generator import EnhancedSignalGenerator
from trading_ai.backtest.backtest_engine import BacktestEngine
from trading_ai.execution.exchange import Exchange
from trading_ai.memory.trade_memory import TradeMemory, TradeRecord


def test_llm_client():
    """Test LLM client functionality."""
    print("\n Testing LLM Client")
    print("=" * 50)
    
    try:
        llm_client = LLMClient()
        
        # Test trading decision
        market_context = {
            "symbol": "BTC",
            "news_summary": "Bitcoin reaches new all-time high",
            "sentiment_score": 0.8,
            "current_price": 50000.0,
            "market_trend": "bullish",
            "agent_type": "test"
        }
        
        decision = llm_client.make_trading_decision(market_context)
        
        if decision:
            print(f"  LLM Decision: {decision.action} {decision.symbol}")
            print(f"  Confidence: {decision.confidence:.2f}")
            print(f"  Entry: ${decision.entry:.2f}")
            print(f"  Stop Loss: ${decision.stop_loss:.2f}")
            print(f"  Take Profit: ${decision.take_profit:.2f}")
            print(f"  Reasoning: {decision.reasoning}")
            return True
        else:
            print("  No decision generated")
            return False
            
    except Exception as e:
        print(f"  LLM Client test failed: {e}")
        return False


def test_decision_engine():
    """Test decision engine functionality."""
    print("\n Testing Decision Engine")
    print("=" * 50)
    
    try:
        decision_engine = DecisionEngine()
        
        # Test consensus decision
        market_data = {
            "price": 50000.0,
            "volume": 1000000.0,
            "volatility": 0.03,
            "indicators": {
                "rsi": 65.0,
                "macd": 0.5,
                "sma_20": 48000.0,
                "sma_50": 45000.0
            }
        }
        
        news_data = [
            {
                "title": "Bitcoin surges",
                "sentiment": 0.7,
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        positions = {"BTC": 1.0}
        
        consensus = decision_engine.make_decision("BTC", market_data, news_data, positions)
        
        if consensus:
            print(f"  Consensus Action: {consensus.action}")
            print(f"  Consensus Confidence: {consensus.confidence:.2f}")
            print(f"  Agent Scores: {len(consensus.agent_scores)}")
            print(f"  Reasoning: {consensus.reasoning[:100]}...")
            
            # Test signal conversion
            signal = decision_engine.convert_to_signal(consensus)
            if signal:
                print(f"  Signal: {signal.direction.value} {signal.symbol} (conf: {signal.confidence:.2f})")
                return True
            else:
                print("  Signal conversion failed")
                return False
        else:
            print("  No consensus generated")
            return False
            
    except Exception as e:
        print(f"  Decision Engine test failed: {e}")
        return False


def test_multi_agent_system():
    """Test multi-agent system."""
    print("\n Testing Multi-Agent System")
    print("=" * 50)
    
    try:
        multi_agent = MultiAgentSystem()
        
        # Test consensus decision
        context = {
            "symbol": "BTC",
            "current_price": 50000.0,
            "volume": 1000000.0,
            "volatility": 0.03,
            "market_trend": "bullish",
            "news_data": [
                {
                    "title": "Bitcoin bullish",
                    "sentiment": 0.6,
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "technical_indicators": {
                "rsi": 60.0,
                "macd": 0.3,
                "sma_20": 48000.0,
                "sma_50": 45000.0
            },
            "positions": {"BTC": 0.5},
            "portfolio_value": 100000.0
        }
        
        consensus = multi_agent.make_consensus_decision(context)
        
        if consensus:
            print(f"  Consensus: {consensus['action']} {consensus['symbol']}")
            print(f"  Confidence: {consensus['confidence']:.2f}")
            print(f"  Agent Decisions: {len(consensus['agent_decisions'])}")
            
            for agent_decision in consensus['agent_decisions']:
                print(f"    {agent_decision['agent']}: {agent_decision['action']} (conf: {agent_decision['confidence']:.2f})")
            
            return True
        else:
            print("  No consensus generated")
            return False
            
    except Exception as e:
        print(f"  Multi-Agent System test failed: {e}")
        return False


def test_market_data_pipeline():
    """Test market data pipeline."""
    print("\n Testing Market Data Pipeline")
    print("=" * 50)
    
    try:
        data_provider = DataProvider()
        
        # Test price data
        price_data = data_provider.get_current_price("BTC")
        
        if price_data:
            print(f"  BTC Price: ${price_data.price:.2f}")
            print(f"  Volume: {price_data.volume:,.0f}")
            print(f"  Change 24h: {price_data.change_pct_24h:.2%}")
            print(f"  Indicators: {len(price_data.indicators)}")
        else:
            print("  No price data available")
            return False
        
        # Test market data
        market_data = data_provider.get_market_data("BTC", include_indicators=True)
        
        if market_data:
            print(f"  Market Data: {market_data['symbol']}")
            print(f"  Price: ${market_data['price']:.2f}")
            print(f"  Volatility: {market_data['volatility']:.3f}")
            print(f"  Trend: {market_data['trend']}")
            print(f"  Indicators: {len(market_data['indicators'])}")
            return True
        else:
            print("  No market data available")
            return False
            
    except Exception as e:
        print(f"  Market Data Pipeline test failed: {e}")
        return False


def test_enhanced_signal_generator():
    """Test enhanced signal generator."""
    print("\n Testing Enhanced Signal Generator")
    print("=" * 50)
    
    try:
        signal_generator = EnhancedSignalGenerator()
        
        # Test signal generation
        symbols = ["BTC", "ETH"]
        news_data = [
            {
                "title": "Bitcoin reaches new heights",
                "content": "BTC Bitcoin cryptocurrency sees strong bullish momentum",
                "sentiment": 0.7,
                "timestamp": datetime.now().isoformat()
            },
            {
                "title": "Ethereum network upgrade",
                "content": "ETH Ethereum completes major upgrade",
                "sentiment": 0.5,
                "timestamp": datetime.now().isoformat()
            }
        ]
        positions = {"BTC": 1.0, "ETH": 10.0}
        
        signals = signal_generator.generate_signals(symbols, news_data, positions)
        
        print(f"  Signals Generated: {len(signals)}")
        
        for signal in signals:
            print(f"    {signal.direction.value} {signal.symbol}")
            print(f"      Confidence: {signal.confidence:.2f}")
            print(f"      Position Size: {signal.position_size:.2f}")
            print(f"      Urgency: {signal.urgency.value}")
            print(f"      Reasoning: {signal.metadata.get('reasoning', 'No reasoning')[:50]}...")
        
        return len(signals) > 0
        
    except Exception as e:
        print(f"  Enhanced Signal Generator test failed: {e}")
        return False


def test_backtest_engine():
    """Test backtesting engine."""
    print("\n Testing Backtest Engine")
    print("=" * 50)
    
    try:
        backtest_engine = BacktestEngine(initial_cash=100000.0)
        
        # Test backtest
        symbols = ["BTC", "ETH"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        results = backtest_engine.run_backtest(symbols, start_date, end_date)
        
        if results:
            summary = results.get("summary", {})
            print(f"  Initial Cash: ${summary.get('initial_cash', 0):,.2f}")
            print(f"  Final Value: ${summary.get('final_value', 0):,.2f}")
            print(f"  Total Return: {summary.get('total_return', 0):.2%}")
            print(f"  Total Trades: {summary.get('total_trades', 0)}")
            print(f"  Win Rate: {summary.get('win_rate', 0):.2%}")
            print(f"  Max Drawdown: {summary.get('max_drawdown', 0):.2%}")
            print(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
            
            return True
        else:
            print("  No backtest results available")
            return False
            
    except Exception as e:
        print(f"  Backtest Engine test failed: {e}")
        return False


def test_exchange_integration():
    """Test exchange integration."""
    print("\n Testing Exchange Integration")
    print("=" * 50)
    
    try:
        exchange = Exchange(paper_trading=True)
        
        # Test connection
        connected = exchange.connect()
        print(f"  Connected: {connected}")
        
        if connected:
            # Test balance
            balance = exchange.get_balance()
            print(f"  Balance: {balance}")
            
            # Test ticker
            ticker = exchange.get_ticker("BTC/USDT")
            if ticker:
                print(f"  BTC/USDT: ${ticker['price']:.2f}")
                
                # Test order creation
                order = exchange.create_order("BTC/USDT", "buy", "market", 0.001)
                if order:
                    print(f"  Order Created: {order.order_id}")
                    print(f"    Status: {order.status}")
                    print(f"    Filled: {order.filled}")
                    print(f"    Cost: ${order.cost:.2f}")
                    
                    # Test order history
                    history = exchange.get_order_history()
                    print(f"  Order History: {len(history)} orders")
                    
                    # Test positions
                    positions = exchange.get_positions()
                    print(f"  Positions: {len(positions)} positions")
                    
                    return True
                else:
                    print("  Order creation failed")
                    return False
            else:
                print("  No ticker data available")
                return False
        else:
            print("  Connection failed")
            return False
            
    except Exception as e:
        print(f"  Exchange Integration test failed: {e}")
        return False


def test_trade_memory():
    """Test trade memory system."""
    print("\n Testing Trade Memory System")
    print("=" * 50)
    
    try:
        trade_memory = TradeMemory()
        
        # Create test trade record
        trade_record = TradeRecord(
            trade_id="test_001",
            symbol="BTC",
            direction="BUY",
            quantity=0.1,
            entry_price=50000.0,
            exit_price=52000.0,
            entry_time=datetime.now() - timedelta(hours=24),
            exit_time=datetime.now() - timedelta(hours=12),
            stop_loss=47500.0,
            take_profit=55000.0,
            pnl=200.0,
            pnl_pct=0.04,
            status="closed",
            fees=10.0,
            confidence=0.8,
            reasoning="Strong bullish momentum",
            market_conditions={"regime": "bullish", "volatility": 0.03},
            agent_decisions=[
                {"agent": "news_agent", "action": "BUY", "confidence": 0.8},
                {"agent": "technical_agent", "action": "BUY", "confidence": 0.7}
            ],
            metadata={"test": True}
        )
        
        # Add trade to memory
        added = trade_memory.add_trade(trade_record)
        print(f"  Trade Added: {added}")
        
        if added:
            # Test trade history
            history = trade_memory.get_trade_history()
            print(f"  Trade History: {len(history)} trades")
            
            # Test performance metrics
            metrics = trade_memory.get_performance_metrics()
            if metrics:
                print(f"  Performance Metrics:")
                print(f"    Total Trades: {metrics.get('total_trades', 0)}")
                print(f"    Win Rate: {metrics.get('win_rate', 0):.2%}")
                print(f"    Total P&L: ${metrics.get('total_pnl', 0):.2f}")
                print(f"    Avg Win: ${metrics.get('avg_win', 0):.2f}")
                print(f"    Avg Loss: ${metrics.get('avg_loss', 0):.2f}")
            
            # Test learning insights
            insights = trade_memory.get_learning_insights()
            print(f"  Learning Insights: {len(insights)} categories")
            
            return True
        else:
            print("  Failed to add trade")
            return False
            
    except Exception as e:
        print(f"  Trade Memory test failed: {e}")
        return False


def test_system_integration():
    """Test full system integration."""
    print("\n Testing System Integration")
    print("=" * 50)
    
    try:
        # Initialize components
        decision_engine = DecisionEngine()
        signal_generator = EnhancedSignalGenerator()
        exchange = Exchange(paper_trading=True)
        trade_memory = TradeMemory()
        
        # Connect to exchange
        exchange.connect()
        
        # Generate signals
        symbols = ["BTC"]
        news_data = [
            {
                "title": "Bitcoin breaks resistance",
                "content": "BTC Bitcoin shows strong bullish momentum",
                "sentiment": 0.8,
                "timestamp": datetime.now().isoformat()
            }
        ]
        positions = {}
        
        signals = signal_generator.generate_signals(symbols, news_data, positions)
        
        if signals:
            signal = signals[0]
            print(f"  Generated Signal: {signal.direction.value} {signal.symbol}")
            
            # Execute trade
            ticker = exchange.get_ticker("BTC/USDT")
            if ticker:
                order = exchange.create_order("BTC/USDT", signal.direction.value.lower(), "market", 0.001)
                
                if order:
                    print(f"  Order Executed: {order.order_id}")
                    
                    # Create trade record
                    trade_record = TradeRecord(
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
                        agent_decisions=signal.metadata.get("agent_decisions", []),
                        metadata={"signal_id": signal.symbol}
                    )
                    
                    # Add to memory
                    trade_memory.add_trade(trade_record)
                    
                    # Get account summary
                    summary = exchange.get_account_summary()
                    print(f"  Account Summary:")
                    print(f"    Total Value: ${summary['total_value']:.2f}")
                    print(f"    Positions: {len(summary['positions'])}")
                    print(f"    Open Orders: {summary['open_orders']}")
                    
                    return True
                else:
                    print("  Order execution failed")
                    return False
            else:
                print("  No ticker data")
                return False
        else:
            print("  No signals generated")
            return False
            
    except Exception as e:
        print(f"  System Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print(" Decision-Making Trading System Test Suite")
    print("=" * 60)
    
    tests = [
        ("LLM Client", test_llm_client),
        ("Decision Engine", test_decision_engine),
        ("Multi-Agent System", test_multi_agent_system),
        ("Market Data Pipeline", test_market_data_pipeline),
        ("Enhanced Signal Generator", test_enhanced_signal_generator),
        ("Backtest Engine", test_backtest_engine),
        ("Exchange Integration", test_exchange_integration),
        ("Trade Memory", test_trade_memory),
        ("System Integration", test_system_integration)
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
    print(f"\n Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("  All tests PASSED - System ready for production!")
        return True
    else:
        print("  Some tests FAILED - Review implementation")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n Decision-Making Trading System: FULLY OPERATIONAL")
        print(f" System transformed from 'news processing pipeline' to 'full decision-making trading system'")
    else:
        print(f"\n Decision-Making Trading System: NEEDS ATTENTION")
        print(f" Some components require fixes before production deployment")
