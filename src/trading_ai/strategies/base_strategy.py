"""
Base strategy implementation following Backtrader/Freqtrade patterns.
Provides common functionality for trading strategies.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from .strategy_interface import IStrategy, StrategyContext, StrategyOutput
from ..core.models import Signal, SignalDirection, Urgency, MarketRegime, SignalType, MarketSession
from ..infrastructure.logging import get_logger


class BaseStrategy(IStrategy):
    """
    Base strategy implementation following Backtrader/Freqtrade patterns.
    
    Provides common functionality:
    - Signal validation
    - Position sizing
    - Risk parameter calculation
    - Performance tracking
    """
    
    def __init__(self, name: str, **kwargs):
        """Initialize base strategy."""
        super().__init__(name, **kwargs)
        self.logger = get_logger(f"strategy.{name}")
        
        # Strategy configuration
        self.min_confidence = kwargs.get("min_confidence", 0.3)
        self.max_signals_per_execution = kwargs.get("max_signals_per_execution", 5)
        self.position_sizing_method = kwargs.get("position_sizing_method", "fixed")
        self.fixed_position_size = kwargs.get("fixed_position_size", 0.1)
        
        # Risk parameters
        self.max_position_size = kwargs.get("max_position_size", 0.2)
        self.stop_loss_pct = kwargs.get("stop_loss_pct", 0.05)
        self.take_profit_pct = kwargs.get("take_profit_pct", 0.15)
        self.max_drawdown = kwargs.get("max_drawdown", 0.2)
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.total_pnl = 0.0
        
        self.logger.info(f"Base strategy {name} initialized with config: {kwargs}")
    
    def validate_signal(self, signal: Signal, context: StrategyContext) -> bool:
        """
        Validate signal against market conditions and risk parameters.
        
        Args:
            signal: Generated signal
            context: Current market context
            
        Returns:
            True if signal should be executed
        """
        # Check minimum confidence
        if signal.confidence < self.min_confidence:
            self.logger.debug(f"Signal rejected: confidence {signal.confidence} below minimum {self.min_confidence}")
            return False
        
        # Check market regime compatibility
        if not self._is_regime_compatible(signal, context.market_regime):
            self.logger.debug(f"Signal rejected: not compatible with market regime {context.market_regime}")
            return False
        
        # Check position limits
        current_position = context.positions.get(signal.symbol, 0)
        if not self._is_position_allowed(signal, current_position, context):
            self.logger.debug(f"Signal rejected: position limits exceeded for {signal.symbol}")
            return False
        
        # Check available cash
        if signal.direction == SignalDirection.BUY and context.available_cash <= 0:
            self.logger.debug("Signal rejected: no available cash for BUY signal")
            return False
        
        # Check market session
        if not self._is_session_compatible(context.market_session):
            self.logger.debug(f"Signal rejected: not compatible with market session {context.market_session}")
            return False
        
        return True
    
    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        """
        Calculate position size using configured method.
        
        Args:
            signal: Trading signal
            context: Current market context
            
        Returns:
            Position size (0-1 representing portfolio fraction)
        """
        if self.position_sizing_method == "fixed":
            return self.fixed_position_size
        
        elif self.position_sizing_method == "confidence":
            # Size based on signal confidence
            base_size = signal.confidence * 0.2  # 20% max for 100% confidence
            return min(self.max_position_size, max(0.01, base_size))
        
        elif self.position_sizing_method == "volatility":
            # Size based on market volatility
            volatility = context.metadata.get("volatility", 0.02)
            vol_adjusted_size = self.fixed_position_size * (0.02 / max(volatility, 0.01))
            return min(self.max_position_size, max(0.01, vol_adjusted_size))
        
        elif self.position_sizing_method == "kelly":
            # Kelly criterion (simplified)
            win_rate = self._get_estimated_win_rate(signal.symbol)
            avg_win = self._get_estimated_avg_win(signal.symbol)
            avg_loss = self._get_estimated_avg_loss(signal.symbol)
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
                return min(self.max_position_size, max(0.01, kelly_fraction))
            else:
                return self.fixed_position_size
        
        else:
            return self.fixed_position_size
    
    def get_risk_parameters(self, context: StrategyContext) -> Dict[str, Any]:
        """
        Get risk parameters for current market conditions.
        
        Args:
            context: Current market context
            
        Returns:
            Risk parameters dictionary
        """
        # Adjust risk based on market regime
        regime_multiplier = self._get_regime_risk_multiplier(context.market_regime)
        
        # Adjust based on volatility
        volatility_multiplier = self._get_volatility_risk_multiplier(
            context.metadata.get("volatility", 0.02)
        )
        
        combined_multiplier = regime_multiplier * volatility_multiplier
        
        return {
            "max_position_size": self.max_position_size * combined_multiplier,
            "stop_loss": self.stop_loss_pct * combined_multiplier,
            "take_profit": self.take_profit_pct * combined_multiplier,
            "max_drawdown": self.max_drawdown,
            "leverage": 1.0,
            "regime_multiplier": regime_multiplier,
            "volatility_multiplier": volatility_multiplier
        }
    
    def create_signal(self, symbol: str, direction: SignalDirection, confidence: float,
                     reason: str, urgency: Urgency = Urgency.MEDIUM, **metadata) -> Signal:
        """
        Create a signal with proper structure.
        
        Args:
            symbol: Trading symbol
            direction: Signal direction
            confidence: Signal confidence
            reason: Signal reason
            urgency: Signal urgency
            **metadata: Additional metadata
            
        Returns:
            Signal object
        """
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            urgency=urgency,
            market_regime=MarketRegime.SIDEWAYS,  # Will be updated by strategy
            position_size=0.0,  # Will be calculated by position sizer
            execution_priority=1,
            signal_type=SignalType.NEWS,
            article_id=metadata.get("article_id"),
            generated_at=datetime.now(),
            metadata={
                "strategy": self.name,
                "reason": reason,
                **metadata
            }
        )
    
    def _is_regime_compatible(self, signal: Signal, regime: MarketRegime) -> bool:
        """Check if signal is compatible with market regime."""
        # Default implementation - can be overridden
        if regime == MarketRegime.VOLATILE:
            # Be more cautious in volatile markets
            return signal.confidence >= self.min_confidence * 1.2
        
        return True
    
    def _is_position_allowed(self, signal: Signal, current_position: float, context: StrategyContext) -> bool:
        """Check if position is allowed based on current position."""
        # Default implementation - can be overridden
        
        if signal.direction == SignalDirection.BUY:
            # Check if we already have maximum position
            if current_position >= context.portfolio_value * self.max_position_size:
                return False
        
        elif signal.direction == SignalDirection.SELL:
            # Check if we have position to sell
            if current_position <= 0:
                return False
        
        return True
    
    def _is_session_compatible(self, session: MarketSession) -> bool:
        """Check if signal is compatible with market session."""
        # Default implementation - can be overridden
        return session in [MarketSession.REGULAR, MarketSession.CRYPTO_24_7]
    
    def _get_regime_risk_multiplier(self, regime: MarketRegime) -> float:
        """Get risk multiplier based on market regime."""
        multipliers = {
            MarketRegime.RISK_ON: 1.2,      # More risk in risk-on markets
            MarketRegime.RISK_OFF: 0.6,     # Less risk in risk-off markets
            MarketRegime.SIDEWAYS: 1.0,     # Normal risk in sideways markets
            MarketRegime.VOLATILE: 0.5,     # Much less risk in volatile markets
        }
        return multipliers.get(regime, 1.0)
    
    def _get_volatility_risk_multiplier(self, volatility: float) -> float:
        """Get risk multiplier based on volatility."""
        # Higher volatility = lower risk
        if volatility > 0.05:  # Very high volatility
            return 0.5
        elif volatility > 0.03:  # High volatility
            return 0.7
        elif volatility > 0.02:  # Normal volatility
            return 1.0
        else:  # Low volatility
            return 1.1
    
    def _get_estimated_win_rate(self, symbol: str) -> float:
        """Get estimated win rate for symbol."""
        # Default implementation - should be overridden with historical data
        return 0.55  # Conservative estimate
    
    def _get_estimated_avg_win(self, symbol: str) -> float:
        """Get estimated average win for symbol."""
        # Default implementation - should be overridden with historical data
        return 0.08  # 8% average win
    
    def _get_estimated_avg_loss(self, symbol: str) -> float:
        """Get estimated average loss for symbol."""
        # Default implementation - should be overridden with historical data
        return 0.04  # 4% average loss
    
    def update_performance(self, metrics: Dict[str, float]) -> None:
        """Update strategy performance metrics."""
        super().update_performance(metrics)
        
        # Update internal tracking
        if "signal_count" in metrics:
            self.total_signals += metrics["signal_count"]
        
        if "successful_signals" in metrics:
            self.successful_signals += metrics["successful_signals"]
        
        if "pnl" in metrics:
            self.total_pnl += metrics["pnl"]
        
        # Log performance update
        if self.total_signals > 0:
            win_rate = self.successful_signals / self.total_signals
            self.logger.info(
                f"Strategy {self.name} performance: "
                f"win_rate={win_rate:.2%}, "
                f"total_pnl={self.total_pnl:.2%}, "
                f"signals={self.total_signals}"
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the strategy."""
        win_rate = self.successful_signals / max(self.total_signals, 1)
        
        return {
            "name": self.name,
            "total_signals": self.total_signals,
            "successful_signals": self.successful_signals,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "last_execution": self.last_execution,
            "enabled": self.enabled,
            "config": {
                "min_confidence": self.min_confidence,
                "max_position_size": self.max_position_size,
                "position_sizing_method": self.position_sizing_method
            }
        }
