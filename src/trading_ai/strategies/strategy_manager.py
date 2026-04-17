"""
Strategy manager following Freqtrade patterns.
Manages multiple strategies and coordinates execution.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

from .strategy_interface import IStrategy, StrategyContext, StrategyOutput
from .market_data_pipeline import MarketDataPipeline, MarketState
from ..core.models import Signal, MarketSession, MarketRegime
from ..infrastructure.logging import get_logger


class StrategyManager:
    """
    Strategy manager following Freqtrade patterns.
    
    Responsibilities:
    - Manage multiple strategies
    - Coordinate strategy execution
    - Handle strategy lifecycle
    - Aggregate strategy outputs
    - Manage strategy performance
    """
    
    def __init__(self, market_data_pipeline: MarketDataPipeline):
        """Initialize strategy manager."""
        self.logger = get_logger("strategy_manager")
        self.market_data_pipeline = market_data_pipeline
        self.strategies: Dict[str, IStrategy] = {}
        self.active_strategies: Dict[str, IStrategy] = {}
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.execution_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Configuration
        self.max_concurrent_strategies = 4
        self.strategy_timeout = 30  # seconds
        self.performance_window = 100  # number of executions
        
        self.logger.info("Strategy manager initialized")
    
    def register_strategy(self, strategy: IStrategy) -> None:
        """
        Register a strategy with the manager.
        
        Args:
            strategy: Strategy instance to register
        """
        with self.execution_lock:
            self.strategies[strategy.name] = strategy
            self.strategy_performance[strategy.name] = {}
            
        self.logger.info(f"Registered strategy: {strategy.name}")
    
    def enable_strategy(self, strategy_name: str) -> None:
        """
        Enable a strategy.
        
        Args:
            strategy_name: Name of strategy to enable
        """
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            strategy.enabled = True
            self.active_strategies[strategy_name] = strategy
            self.logger.info(f"Enabled strategy: {strategy_name}")
        else:
            self.logger.warning(f"Strategy not found: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str) -> None:
        """
        Disable a strategy.
        
        Args:
            strategy_name: Name of strategy to disable
        """
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            strategy.enabled = False
            self.active_strategies.pop(strategy_name, None)
            self.logger.info(f"Disabled strategy: {strategy_name}")
        else:
            self.logger.warning(f"Strategy not found: {strategy_name}")
    
    def execute_strategies(self, portfolio_value: float, available_cash: float, 
                          positions: Dict[str, float], news_data: List[Any]) -> Dict[str, StrategyOutput]:
        """
        Execute all active strategies.
        
        Args:
            portfolio_value: Current portfolio value
            available_cash: Available cash
            positions: Current positions
            news_data: News data
            
        Returns:
            Dictionary of strategy outputs
        """
        if not self.active_strategies:
            self.logger.info("No active strategies to execute")
            return {}
        
        # Create strategy context
        context = self._create_strategy_context(
            portfolio_value, available_cash, positions, news_data
        )
        
        # Execute strategies in parallel
        strategy_outputs = {}
        
        with self.execution_lock:
            # Filter strategies that should execute
            executable_strategies = {
                name: strategy for name, strategy in self.active_strategies.items()
                if strategy.should_execute(context)
            }
            
            if not executable_strategies:
                self.logger.info("No strategies ready for execution")
                return {}
            
            self.logger.info(f"Executing {len(executable_strategies)} strategies")
            
            # Execute strategies
            futures = {}
            for name, strategy in executable_strategies.items():
                future = self.executor.submit(
                    self._execute_single_strategy, 
                    strategy, 
                    context
                )
                futures[name] = future
            
            # Collect results
            for name, future in futures.items():
                try:
                    output = future.result(timeout=self.strategy_timeout)
                    strategy_outputs[name] = output
                    
                    # Update strategy performance
                    self._update_strategy_performance(name, output)
                    
                except Exception as e:
                    self.logger.error(f"Strategy {name} execution failed: {e}")
                    # Disable strategy on repeated failures
                    self._handle_strategy_failure(name)
        
        return strategy_outputs
    
    def get_all_signals(self, portfolio_value: float, available_cash: float,
                        positions: Dict[str, float], news_data: List[Any]) -> List[Signal]:
        """
        Get all signals from active strategies.
        
        Args:
            portfolio_value: Current portfolio value
            available_cash: Available cash
            positions: Current positions
            news_data: News data
            
        Returns:
            List of all signals
        """
        strategy_outputs = self.execute_strategies(
            portfolio_value, available_cash, positions, news_data
        )
        
        # Aggregate all signals
        all_signals = []
        for output in strategy_outputs.values():
            all_signals.extend(output.signals)
        
        # Sort by confidence
        all_signals.sort(key=lambda s: s.confidence, reverse=True)
        
        return all_signals
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, float]:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            Performance metrics
        """
        return self.strategy_performance.get(strategy_name, {})
    
    def get_all_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all strategies.
        
        Returns:
            Dictionary of strategy performance metrics
        """
        return self.strategy_performance.copy()
    
    def _create_strategy_context(self, portfolio_value: float, available_cash: float,
                                 positions: Dict[str, float], news_data: List[Any]) -> StrategyContext:
        """Create strategy context for execution."""
        market_state = self.market_data_pipeline.get_market_state()
        
        # Get market data for all symbols
        market_data = {}
        for symbol in positions.keys():
            latest_price = self.market_data_pipeline.get_latest_price(symbol)
            if latest_price:
                market_data[symbol] = {
                    "price": latest_price,
                    "indicators": self.market_data_pipeline.get_indicators(symbol)
                }
        
        return StrategyContext(
            current_time=datetime.now(),
            market_session=market_state.session,
            market_regime=market_state.regime,
            portfolio_value=portfolio_value,
            available_cash=available_cash,
            positions=positions,
            market_data=market_data,
            news_data=news_data,
            metadata={
                "volatility": market_state.volatility,
                "trend_strength": market_state.trend_strength,
                "market_sentiment": market_state.market_sentiment,
                "liquidity_score": market_state.liquidity_score
            }
        )
    
    def _execute_single_strategy(self, strategy: IStrategy, context: StrategyContext) -> StrategyOutput:
        """Execute a single strategy safely."""
        try:
            return strategy.analyze(context)
        except Exception as e:
            self.logger.error(f"Strategy {strategy.name} analysis failed: {e}")
            # Return empty output
            return StrategyOutput(
                signals=[],
                position_adjustments={},
                risk_adjustments={},
                metadata={"error": str(e)}
            )
    
    def _update_strategy_performance(self, strategy_name: str, output: StrategyOutput) -> None:
        """Update strategy performance metrics."""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {}
        
        performance = self.strategy_performance[strategy_name]
        
        # Update basic metrics
        performance["signal_count"] = len(output.signals)
        performance["last_execution"] = datetime.now().isoformat()
        
        # Calculate confidence metrics
        if output.signals:
            confidences = [s.confidence for s in output.signals]
            performance["avg_confidence"] = sum(confidences) / len(confidences)
            performance["max_confidence"] = max(confidences)
            performance["min_confidence"] = min(confidences)
        else:
            performance["avg_confidence"] = 0.0
            performance["max_confidence"] = 0.0
            performance["min_confidence"] = 0.0
        
        # Update execution count
        performance["execution_count"] = performance.get("execution_count", 0) + 1
        
        # Store recent outputs for rolling metrics
        if "recent_outputs" not in performance:
            performance["recent_outputs"] = []
        
        performance["recent_outputs"].append({
            "timestamp": datetime.now(),
            "signal_count": len(output.signals),
            "avg_confidence": performance["avg_confidence"]
        })
        
        # Keep only recent outputs
        if len(performance["recent_outputs"]) > self.performance_window:
            performance["recent_outputs"] = performance["recent_outputs"][-self.performance_window:]
        
        # Calculate rolling metrics
        recent_outputs = performance["recent_outputs"]
        if len(recent_outputs) >= 10:
            recent_avg_confidence = sum(o["avg_confidence"] for o in recent_outputs[-10:]) / 10
            performance["rolling_avg_confidence"] = recent_avg_confidence
    
    def _handle_strategy_failure(self, strategy_name: str) -> None:
        """Handle strategy execution failure."""
        performance = self.strategy_performance.get(strategy_name, {})
        failure_count = performance.get("failure_count", 0) + 1
        performance["failure_count"] = failure_count
        
        # Disable strategy after too many failures
        if failure_count >= 3:
            self.logger.warning(f"Disabling strategy {strategy_name} due to repeated failures")
            self.disable_strategy(strategy_name)
    
    def cleanup(self) -> None:
        """Cleanup strategy manager resources."""
        self.executor.shutdown(wait=True)
        
        # Cleanup all strategies
        for strategy in self.strategies.values():
            try:
                strategy.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up strategy {strategy.name}: {e}")
        
        self.strategies.clear()
        self.active_strategies.clear()
        self.strategy_performance.clear()
        
        self.logger.info("Strategy manager cleaned up")
