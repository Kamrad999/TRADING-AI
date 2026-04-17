"""
Strategy interface following Freqtrade/Jesse patterns.
Clean abstraction for trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ..core.models import Signal, MarketSession, MarketRegime


@dataclass
class StrategyContext:
    """Context provided to strategies during execution."""
    current_time: datetime
    market_session: MarketSession
    market_regime: MarketRegime
    portfolio_value: float
    available_cash: float
    positions: Dict[str, float]  # symbol -> quantity
    market_data: Dict[str, Any]  # symbol -> market data
    news_data: List[Any]  # news articles
    metadata: Dict[str, Any]


@dataclass
class StrategyOutput:
    """Output from strategy execution."""
    signals: List[Signal]
    position_adjustments: Dict[str, float]  # symbol -> target position
    risk_adjustments: Dict[str, float]  # symbol -> risk parameters
    metadata: Dict[str, Any]


class IStrategy(ABC):
    """
    Base strategy interface following Freqtrade/Jesse patterns.
    
    All strategies must implement these methods to ensure
    compatibility with the trading engine.
    """
    
    def __init__(self, name: str, **kwargs):
        """Initialize strategy with configuration."""
        self.name = name
        self.config = kwargs
        self.enabled = True
        self.last_execution = None
        self.performance_metrics = {}
    
    @abstractmethod
    def analyze(self, context: StrategyContext) -> StrategyOutput:
        """
        Analyze market conditions and generate trading decisions.
        
        Args:
            context: Current market context and data
            
        Returns:
            StrategyOutput with signals and position adjustments
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Signal, context: StrategyContext) -> bool:
        """
        Validate if signal should be executed.
        
        Args:
            signal: Generated signal
            context: Current market context
            
        Returns:
            True if signal should be executed
        """
        pass
    
    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        """
        Calculate position size for signal.
        
        Args:
            signal: Trading signal
            context: Current market context
            
        Returns:
            Position size (0-1 representing portfolio fraction)
        """
        # Default implementation - can be overridden
        base_size = signal.confidence * 0.1  # 10% max per signal
        return min(1.0, max(0.01, base_size))
    
    def get_risk_parameters(self, context: StrategyContext) -> Dict[str, Any]:
        """
        Get risk parameters for current market conditions.
        
        Args:
            context: Current market context
            
        Returns:
            Risk parameters dictionary
        """
        return {
            "max_position_size": 0.1,
            "stop_loss": 0.05,
            "take_profit": 0.15,
            "max_drawdown": 0.2,
            "leverage": 1.0
        }
    
    def update_performance(self, metrics: Dict[str, float]) -> None:
        """
        Update strategy performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
        """
        self.performance_metrics.update(metrics)
    
    def should_execute(self, context: StrategyContext) -> bool:
        """
        Determine if strategy should execute in current conditions.
        
        Args:
            context: Current market context
            
        Returns:
            True if strategy should execute
        """
        if not self.enabled:
            return False
        
        # Check market conditions
        if context.market_session == MarketSession.CLOSED:
            return False
        
        # Check cooldown period
        if self.last_execution:
            cooldown_minutes = self.config.get("cooldown_minutes", 5)
            if (context.current_time - self.last_execution).total_seconds() < cooldown_minutes * 60:
                return False
        
        return True
    
    def on_signal_executed(self, signal: Signal, execution_result: Dict[str, Any]) -> None:
        """
        Called when signal is executed.
        
        Args:
            signal: Executed signal
            execution_result: Execution details
        """
        self.last_execution = datetime.now()
    
    def cleanup(self) -> None:
        """Cleanup strategy resources."""
        pass
