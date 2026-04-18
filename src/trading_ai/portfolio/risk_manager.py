"""
Risk Manager following Jesse patterns.
Handles risk management for portfolio positions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..infrastructure.logging import get_logger
from .position import Position, PositionSide


class RiskLevel(Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskConfig:
    """Risk configuration."""
    max_position_size: float = 0.1  # 10% of portfolio
    max_risk_per_trade: float = 0.02  # 2% risk per trade
    max_drawdown: float = 0.15  # 15% max drawdown
    max_correlation: float = 0.7  # Max correlation between positions
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit


class RiskManager:
    """
    Risk manager following Jesse patterns.
    
    Manages portfolio risk through position sizing, correlation checks,
    and drawdown protection.
    """
    
    def __init__(self, portfolio_value: float = 100000.0, config: Optional[RiskConfig] = None):
        """Initialize risk manager."""
        self.logger = get_logger("risk_manager")
        
        self.portfolio_value = portfolio_value
        self.config = config or RiskConfig()
        
        # Risk tracking
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_value = portfolio_value
        
        # Position risk tracking
        self.position_risk: Dict[str, float] = {}
        self.total_risk = 0.0
        
        self.logger.info(f"RiskManager initialized with ${portfolio_value:,.2f} portfolio")
    
    def check_position_risk(self, symbol: str, position_size: float, 
                           stop_loss_pct: float) -> Dict[str, Any]:
        """
        Check if position meets risk criteria.
        
        Args:
            symbol: Trading symbol
            position_size: Position size in units
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Risk check result
        """
        try:
            # Calculate position value
            position_value = position_size * self._get_current_price(symbol)
            
            # Check position size limit
            position_pct = position_value / self.portfolio_value
            if position_pct > self.config.max_position_size:
                return {
                    "approved": False,
                    "reason": f"Position size {position_pct:.2%} exceeds limit {self.config.max_position_size:.2%}",
                    "risk_level": RiskLevel.HIGH.value
                }
            
            # Calculate risk amount
            risk_amount = position_value * stop_loss_pct
            risk_pct = risk_amount / self.portfolio_value
            
            if risk_pct > self.config.max_risk_per_trade:
                return {
                    "approved": False,
                    "reason": f"Risk per trade {risk_pct:.2%} exceeds limit {self.config.max_risk_per_trade:.2%}",
                    "risk_level": RiskLevel.HIGH.value
                }
            
            # Check total portfolio risk
            total_risk_with_new = self.total_risk + risk_pct
            if total_risk_with_new > 0.06:  # 6% total portfolio risk
                return {
                    "approved": False,
                    "reason": f"Total portfolio risk {total_risk_with_new:.2%} too high",
                    "risk_level": RiskLevel.HIGH.value
                }
            
            # Check drawdown limit
            if self.current_drawdown > self.config.max_drawdown:
                return {
                    "approved": False,
                    "reason": f"Drawdown {self.current_drawdown:.2%} exceeds limit {self.config.max_drawdown:.2%}",
                    "risk_level": RiskLevel.EXTREME.value
                }
            
            return {
                "approved": True,
                "reason": "Position meets risk criteria",
                "risk_level": RiskLevel.LOW.value,
                "position_pct": position_pct,
                "risk_pct": risk_pct
            }
            
        except Exception as e:
            self.logger.error(f"Risk check failed: {e}")
            return {
                "approved": False,
                "reason": f"Risk check error: {str(e)}",
                "risk_level": RiskLevel.HIGH.value
            }
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update portfolio value and track drawdown."""
        try:
            # Update peak value
            if new_value > self.peak_value:
                self.peak_value = new_value
            
            # Calculate drawdown
            self.current_drawdown = (self.peak_value - new_value) / self.peak_value
            
            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                self.logger.warning(f"New max drawdown: {self.max_drawdown:.2%}")
            
            self.portfolio_value = new_value
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio value: {e}")
    
    def register_position(self, position: Position) -> None:
        """Register position for risk tracking."""
        try:
            # Calculate position risk
            risk_amount = position.entry_value * (position.stop_loss / position.entry_price if position.stop_loss else 0.05)
            risk_pct = risk_amount / self.portfolio_value
            
            self.position_risk[position.id] = risk_pct
            self.total_risk = sum(self.position_risk.values())
            
            self.logger.debug(f"Position registered: {position.id} | Risk: {risk_pct:.2%}")
            
        except Exception as e:
            self.logger.error(f"Failed to register position: {e}")
    
    def unregister_position(self, position_id: str) -> None:
        """Unregister position from risk tracking."""
        try:
            if position_id in self.position_risk:
                del self.position_risk[position_id]
                self.total_risk = sum(self.position_risk.values())
            
        except Exception as e:
            self.logger.error(f"Failed to unregister position: {e}")
    
    def check_correlation(self, symbol: str, existing_positions: List[Position]) -> Dict[str, Any]:
        """Check correlation with existing positions."""
        try:
            # Simplified correlation check
            # In production, would use actual correlation matrix
            
            similar_symbols = [p.symbol for p in existing_positions if p.symbol == symbol]
            
            if len(similar_symbols) >= 2:
                return {
                    "approved": False,
                    "reason": f"Too many positions in {symbol} ({len(similar_symbols)})",
                    "correlation": 1.0
                }
            
            return {
                "approved": True,
                "reason": "Correlation check passed",
                "correlation": 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Correlation check failed: {e}")
            return {
                "approved": False,
                "reason": f"Correlation check error: {str(e)}",
                "correlation": 1.0
            }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary."""
        return {
            "portfolio_value": self.portfolio_value,
            "peak_value": self.peak_value,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "total_risk": self.total_risk,
            "position_count": len(self.position_risk),
            "config": {
                "max_position_size": self.config.max_position_size,
                "max_risk_per_trade": self.config.max_risk_per_trade,
                "max_drawdown": self.config.max_drawdown
            }
        }
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        # In production, would get from market data provider
        price_map = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "AAPL": 150.0,
            "MSFT": 300.0,
            "SPY": 400.0,
            "QQQ": 350.0
        }
        return price_map.get(symbol, 100.0)
