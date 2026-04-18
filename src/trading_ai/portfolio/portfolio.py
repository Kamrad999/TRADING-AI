"""
Portfolio management following Jesse patterns.
Manages overall portfolio state and performance.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

from ..infrastructure.logging import get_logger
from .position import Position, PositionSide, PositionStatus
from .risk_manager import RiskManager, RiskConfig


@dataclass
class PortfolioConfig:
    """Portfolio configuration."""
    initial_value: float = 100000.0
    base_currency: str = "USD"
    max_positions: int = 10
    risk_config: RiskConfig = field(default_factory=RiskConfig)


class Portfolio:
    """
    Portfolio manager following Jesse patterns.
    
    Manages portfolio state, positions, and overall performance.
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """Initialize portfolio."""
        self.logger = get_logger("portfolio")
        
        self.config = config or PortfolioConfig()
        
        # Portfolio state
        self.initial_value = self.config.initial_value
        self.current_value = self.config.initial_value
        self.available_cash = self.config.initial_value
        
        # Position management
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Risk management
        self.risk_manager = RiskManager(
            portfolio_value=self.current_value,
            config=self.config.risk_config
        )
        
        # Performance tracking
        self.total_pnl = 0.0
        self.total_pnl_pct = 0.0
        self.daily_pnl = 0.0
        
        # History
        self.value_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"Portfolio initialized with ${self.initial_value:,.2f}")
    
    def update_value(self, new_value: float) -> None:
        """Update portfolio value."""
        try:
            old_value = self.current_value
            self.current_value = new_value
            
            # Calculate P&L
            self.daily_pnl = new_value - old_value
            self.total_pnl = new_value - self.initial_value
            self.total_pnl_pct = (self.total_pnl / self.initial_value) * 100
            
            # Update risk manager
            self.risk_manager.update_portfolio_value(new_value)
            
            # Record history
            self.value_history.append({
                "timestamp": datetime.now(),
                "value": new_value,
                "pnl": self.total_pnl,
                "pnl_pct": self.total_pnl_pct
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio value: {e}")
    
    def add_position(self, position: Position) -> bool:
        """Add position to portfolio."""
        try:
            # Check position limits
            if len(self.positions) >= self.config.max_positions:
                self.logger.warning(f"Max positions reached: {self.config.max_positions}")
                return False
            
            # Check risk
            risk_check = self.risk_manager.check_position_risk(
                position.symbol,
                position.quantity,
                0.05  # Default stop loss
            )
            
            if not risk_check["approved"]:
                self.logger.warning(f"Position rejected: {risk_check['reason']}")
                return False
            
            # Add position
            self.positions[position.id] = position
            
            # Update cash
            position_value = position.entry_value
            self.available_cash -= position_value
            
            # Register with risk manager
            self.risk_manager.register_position(position)
            
            self.logger.info(f"Position added: {position.id} | {position.symbol} | ${position_value:,.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add position: {e}")
            return False
    
    def remove_position(self, position_id: str) -> Optional[Position]:
        """Remove position from portfolio."""
        try:
            if position_id not in self.positions:
                return None
            
            position = self.positions[position_id]
            
            # Update cash
            self.available_cash += position.current_value
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position_id]
            
            # Unregister from risk manager
            self.risk_manager.unregister_position(position_id)
            
            self.logger.info(f"Position removed: {position_id}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Failed to remove position: {e}")
            return None
    
    def update_positions(self, price_updates: Dict[str, float]) -> None:
        """Update all positions with new prices."""
        try:
            total_value = self.available_cash
            
            for position in self.positions.values():
                if position.symbol in price_updates:
                    position.update_price(price_updates[position.symbol])
                
                total_value += position.current_value
            
            self.update_value(total_value)
            
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get portfolio position summary."""
        try:
            open_positions = len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
            closed_positions = len(self.closed_positions)
            
            unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
            realized_pnl = sum(p.realized_pnl for p in self.closed_positions)
            
            return {
                "total_value": self.current_value,
                "initial_value": self.initial_value,
                "available_cash": self.available_cash,
                "total_pnl": self.total_pnl,
                "total_pnl_pct": self.total_pnl_pct,
                "daily_pnl": self.daily_pnl,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "open_positions": open_positions,
                "closed_positions": closed_positions,
                "total_positions": open_positions + closed_positions
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get position summary: {e}")
            return {}
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get portfolio risk summary."""
        return self.risk_manager.get_risk_summary()
    
    def get_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation by symbol."""
        try:
            allocation = {}
            
            for position in self.positions.values():
                if position.status == PositionStatus.OPEN:
                    pct = (position.current_value / self.current_value) * 100
                    allocation[position.symbol] = pct
            
            # Cash allocation
            cash_pct = (self.available_cash / self.current_value) * 100
            allocation["CASH"] = cash_pct
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Failed to get allocation: {e}")
            return {}
