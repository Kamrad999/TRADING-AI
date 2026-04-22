"""
Position management following Jesse patterns.
Handles position lifecycle, P&L tracking, and risk management.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from ..infrastructure.logging import get_logger


class PositionStatus(Enum):
    """Position status following Jesse patterns."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


class PositionSide(Enum):
    """Position side following Jesse patterns."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """
    Position class following Jesse architecture.
    
    Handles complete position lifecycle:
    - Entry and exit management
    - Stop loss and take profit
    - P&L calculation
    - Risk tracking
    """
    
    # Core position data
    id: str
    symbol: str
    side: PositionSide
    status: PositionStatus
    entry_price: float
    current_price: float
    quantity: float
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Financial tracking
    entry_value: float = field(init=False)
    current_value: float = field(init=False)
    unrealized_pnl: float = field(init=False)
    realized_pnl: float = field(init=False)
    pnl_percentage: float = field(init=False)
    
    # Timing
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Metadata
    strategy: str = ""
    entry_reason: str = ""
    exit_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Risk tracking
    max_unrealized_pnl: float = 0.0
    min_unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    
    def __post_init__(self):
        """Initialize calculated fields."""
        self.logger = get_logger(f"position.{self.symbol}")
        self._update_financials()
    
    def update_price(self, new_price: float) -> None:
        """Update position price and recalculate P&L."""
        self.current_price = new_price
        self.last_updated = datetime.now()
        self._update_financials()
        self._track_extremes()
        
        # Check for stop loss / take profit
        self._check_exit_conditions()
    
    def _update_financials(self) -> None:
        """Update financial calculations."""
        # Entry and current values
        self.entry_value = self.entry_price * self.quantity
        self.current_value = self.current_price * self.quantity
        
        # P&L calculations
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity
        
        # P&L percentage
        if self.entry_value > 0:
            self.pnl_percentage = (self.unrealized_pnl / self.entry_value) * 100
        else:
            self.pnl_percentage = 0.0
    
    def _track_extremes(self) -> None:
        """Track P&L extremes for risk analysis."""
        if self.unrealized_pnl > self.max_unrealized_pnl:
            self.max_unrealized_pnl = self.unrealized_pnl
        
        if self.unrealized_pnl < self.min_unrealized_pnl:
            self.min_unrealized_pnl = self.unrealized_pnl
        
        # Calculate max drawdown
        if self.max_unrealized_pnl > 0:
            drawdown = (self.max_unrealized_pnl - self.unrealized_pnl) / self.max_unrealized_pnl
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def _check_exit_conditions(self) -> None:
        """Check if position should be closed based on exit conditions."""
        if self.status != PositionStatus.OPEN:
            return
        
        # DEBUG: Log stop loss check
        if self.stop_loss:
            should_sl = self._should_stop_loss()
            self.logger.debug(
                f"SL Check {self.symbol}: price={self.current_price:.2f}, "
                f"SL={self.stop_loss:.2f}, side={self.side.value}, "
                f"should_trigger={should_sl}"
            )
        
        # Check stop loss
        if self.stop_loss and self._should_stop_loss():
            self.logger.info(
                f"🛑 STOP LOSS TRIGGERED: {self.symbol} {self.side.value} "
                f"at ${self.current_price:.2f} (SL: ${self.stop_loss:.2f})"
            )
            self.close("stop_loss")
        
        # Check take profit
        elif self.take_profit and self._should_take_profit():
            self.logger.info(f"Take profit triggered for {self.symbol} at {self.current_price}")
            self.close("take_profit")
        
        # Check trailing stop
        elif self.trailing_stop and self._should_trailing_stop():
            self.logger.info(f"Trailing stop triggered for {self.symbol} at {self.current_price}")
            self.close("trailing_stop")
    
    def _should_stop_loss(self) -> bool:
        """Check if stop loss should be triggered."""
        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss
    
    def _should_take_profit(self) -> bool:
        """Check if take profit should be triggered."""
        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        else:  # SHORT
            return self.current_price <= self.take_profit
    
    def _should_trailing_stop(self) -> bool:
        """Check if trailing stop should be triggered."""
        if self.side == PositionSide.LONG:
            return self.current_price <= self.trailing_stop
        else:  # SHORT
            return self.current_price >= self.trailing_stop
    
    def close(self, reason: str = "manual") -> float:
        """
        Close position and return realized P&L.
        
        Args:
            reason: Reason for closing position
            
        Returns:
            Realized P&L
        """
        if self.status == PositionStatus.CLOSED:
            return self.realized_pnl
        
        self.exit_time = datetime.now()
        self.exit_reason = reason
        self.status = PositionStatus.CLOSED
        
        # Calculate realized P&L
        self.realized_pnl = self.unrealized_pnl
        
        self.logger.info(
            f"Position {self.id} closed: {self.symbol} {self.side.value} | "
            f"P&L: ${self.realized_pnl:.2f} ({self.pnl_percentage:.2f}%) | "
            f"Reason: {reason}"
        )
        
        return self.realized_pnl
    
    def close_partial(self, quantity: float, reason: str = "partial") -> float:
        """
        Partially close position.
        
        Args:
            quantity: Quantity to close
            reason: Reason for partial close
            
        Returns:
            Realized P&L from partial close
        """
        if self.status != PositionStatus.OPEN:
            return 0.0
        
        if quantity >= self.quantity:
            return self.close(reason)
        
        # Calculate P&L for partial close
        if self.side == PositionSide.LONG:
            partial_pnl = (self.current_price - self.entry_price) * quantity
        else:  # SHORT
            partial_pnl = (self.entry_price - self.current_price) * quantity
        
        # Update position
        self.quantity -= quantity
        self.realized_pnl += partial_pnl
        self.status = PositionStatus.PARTIALLY_CLOSED
        
        # Recalculate entry value
        self.entry_value = self.entry_price * self.quantity
        self._update_financials()
        
        self.logger.info(
            f"Position {self.id} partially closed: {self.symbol} {self.side.value} | "
            f"Quantity: {quantity} | P&L: ${partial_pnl:.2f} | "
            f"Remaining: {self.quantity}"
        )
        
        return partial_pnl
    
    def update_stop_loss(self, new_stop_loss: float) -> None:
        """Update stop loss price."""
        old_stop_loss = self.stop_loss
        self.stop_loss = new_stop_loss
        
        self.logger.debug(
            f"Stop loss updated for {self.symbol}: ${old_stop_loss} -> ${new_stop_loss}"
        )
    
    def update_take_profit(self, new_take_profit: float) -> None:
        """Update take profit price."""
        old_take_profit = self.take_profit
        self.take_profit = new_take_profit
        
        self.logger.debug(
            f"Take profit updated for {self.symbol}: ${old_take_profit} -> ${new_take_profit}"
        )
    
    def update_trailing_stop(self, trailing_distance: float) -> None:
        """
        Update trailing stop based on current price.
        
        Args:
            trailing_distance: Distance from current price for trailing stop
        """
        if self.side == PositionSide.LONG:
            new_trailing_stop = self.current_price - trailing_distance
            # Only move trailing stop up (for long positions)
            if self.trailing_stop is None or new_trailing_stop > self.trailing_stop:
                self.trailing_stop = new_trailing_stop
        else:  # SHORT
            new_trailing_stop = self.current_price + trailing_distance
            # Only move trailing stop down (for short positions)
            if self.trailing_stop is None or new_trailing_stop < self.trailing_stop:
                self.trailing_stop = new_trailing_stop
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics for the position."""
        return {
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "pnl_percentage": self.pnl_percentage,
            "max_unrealized_pnl": self.max_unrealized_pnl,
            "min_unrealized_pnl": self.min_unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "risk_reward_ratio": self._calculate_risk_reward_ratio(),
            "time_in_position": (datetime.now() - self.entry_time).total_seconds() / 3600,  # hours
            "volatility_risk": self._calculate_volatility_risk()
        }
    
    def _calculate_risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        if not self.stop_loss or not self.take_profit:
            return 0.0
        
        if self.side == PositionSide.LONG:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
        else:  # SHORT
            risk = abs(self.stop_loss - self.entry_price)
            reward = abs(self.entry_price - self.take_profit)
        
        return reward / risk if risk > 0 else 0.0
    
    def _calculate_volatility_risk(self) -> float:
        """Calculate volatility-based risk score."""
        # Simplified volatility risk based on price movement
        price_volatility = abs(self.current_price - self.entry_price) / self.entry_price
        return min(1.0, price_volatility * 10)  # Normalize to 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "status": self.status.value,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "entry_value": self.entry_value,
            "current_value": self.current_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "pnl_percentage": self.pnl_percentage,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "last_updated": self.last_updated.isoformat(),
            "strategy": self.strategy,
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
            "metadata": self.metadata,
            "max_unrealized_pnl": self.max_unrealized_pnl,
            "min_unrealized_pnl": self.min_unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "risk_metrics": self.get_risk_metrics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create position from dictionary."""
        position = cls(
            id=data["id"],
            symbol=data["symbol"],
            side=PositionSide(data["side"]),
            status=PositionStatus(data["status"]),
            entry_price=data["entry_price"],
            current_price=data["current_price"],
            quantity=data["quantity"],
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            trailing_stop=data.get("trailing_stop"),
            strategy=data.get("strategy", ""),
            entry_reason=data.get("entry_reason", ""),
            metadata=data.get("metadata", {})
        )
        
        # Restore timing
        if data.get("entry_time"):
            position.entry_time = datetime.fromisoformat(data["entry_time"])
        if data.get("exit_time"):
            position.exit_time = datetime.fromisoformat(data["exit_time"])
        if data.get("last_updated"):
            position.last_updated = datetime.fromisoformat(data["last_updated"])
        
        # Restore financials
        position.entry_value = data.get("entry_value", 0.0)
        position.current_value = data.get("current_value", 0.0)
        position.unrealized_pnl = data.get("unrealized_pnl", 0.0)
        position.realized_pnl = data.get("realized_pnl", 0.0)
        position.pnl_percentage = data.get("pnl_percentage", 0.0)
        
        # Restore tracking
        position.max_unrealized_pnl = data.get("max_unrealized_pnl", 0.0)
        position.min_unrealized_pnl = data.get("min_unrealized_pnl", 0.0)
        position.max_drawdown = data.get("max_drawdown", 0.0)
        
        return position
