"""
Position Manager following Jesse patterns.
Handles position lifecycle, opening/closing positions, and risk management.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from .position import Position, PositionSide, PositionStatus
from ..infrastructure.logging import get_logger
from ..core.models import Signal, SignalDirection


@dataclass
class PositionRequest:
    """Request to open a new position."""
    symbol: str
    direction: SignalDirection
    quantity: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ""
    reason: str = ""
    metadata: Dict[str, Any] = None


class PositionManager:
    """
    Position Manager following Jesse architecture.
    
    Handles complete position lifecycle:
    - Opening positions from signals
    - Managing open positions
    - Closing positions (manual, stop loss, take profit)
    - Risk management and position sizing
    """
    
    def __init__(self, initial_balance: float = 100000.0):
        """Initialize position manager."""
        self.logger = get_logger("position_manager")
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Portfolio tracking
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.total_pnl = 0.0
        self.total_commissions = 0.0
        
        # Position counters
        self.position_counter = 0
        self.total_positions_opened = 0
        self.total_positions_closed = 0
        
        # Risk management
        self.max_positions = 10
        self.max_position_size = 0.2  # 20% of portfolio
        self.min_position_size = 0.01  # 1% of portfolio
        self.commission_rate = 0.001  # 0.1%
        
        self.logger.info(f"PositionManager initialized with balance: ${initial_balance:,.2f}")
    
    def open_position(self, request: PositionRequest) -> Optional[Position]:
        """
        Open a new position from a signal.
        
        Args:
            request: Position request with all parameters
            
        Returns:
            New position if successful, None otherwise
        """
        try:
            # Validate request
            if not self._validate_position_request(request):
                return None
            
            # Check position limits
            if not self._check_position_limits(request):
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(request)
            if position_size <= 0:
                return None
            
            # Create position
            position_id = f"pos_{self.position_counter}"
            self.position_counter += 1
            
            position = Position(
                id=position_id,
                symbol=request.symbol,
                side=PositionSide.LONG if request.direction == SignalDirection.BUY else PositionSide.SHORT,
                status=PositionStatus.OPEN,
                entry_price=request.entry_price,
                current_price=request.entry_price,
                quantity=position_size,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                strategy=request.strategy,
                entry_reason=request.reason,
                metadata=request.metadata or {}
            )
            
            # Add to tracking
            self.positions[position_id] = position
            self.open_positions[position_id] = position
            self.total_positions_opened += 1
            
            # Update balance (subtract commission)
            commission = position.entry_value * self.commission_rate
            self.current_balance -= commission
            self.total_commissions += commission
            
            self.logger.info(
                f"Position opened: {position_id} | {request.symbol} {request.direction.value} | "
                f"Size: {position_size:.6f} @ ${request.entry_price:.2f} | "
                f"Value: ${position.entry_value:.2f} | "
                f"SL: ${request.stop_loss} | TP: ${request.take_profit}"
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"Failed to open position: {e}")
            return None
    
    def close_position(self, position_id: str, reason: str = "manual") -> Optional[float]:
        """
        Close an existing position.
        
        Args:
            position_id: Position ID to close
            reason: Reason for closing
            
        Returns:
            Realized P&L if successful, None otherwise
        """
        try:
            position = self.positions.get(position_id)
            if not position:
                self.logger.warning(f"Position {position_id} not found")
                return None
            
            if position.status == PositionStatus.CLOSED:
                self.logger.warning(f"Position {position_id} already closed")
                return position.realized_pnl
            
            # Close position
            realized_pnl = position.close(reason)
            
            # Update balance
            self.current_balance += realized_pnl
            self.total_pnl += realized_pnl
            
            # Add commission for closing
            commission = position.current_value * self.commission_rate
            self.current_balance -= commission
            self.total_commissions += commission
            
            # Move to closed positions
            if position_id in self.open_positions:
                del self.open_positions[position_id]
            
            self.closed_positions.append(position)
            self.total_positions_closed += 1
            
            self.logger.info(
                f"Position closed: {position_id} | P&L: ${realized_pnl:.2f} | "
                f"Balance: ${self.current_balance:.2f} | Reason: {reason}"
            )
            
            return realized_pnl
            
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {e}")
            return None
    
    def close_all_positions(self, reason: str = "shutdown") -> Dict[str, float]:
        """
        Close all open positions.
        
        Args:
            reason: Reason for closing all positions
            
        Returns:
            Dictionary of position IDs and their P&L
        """
        results = {}
        
        # Copy list to avoid modification during iteration
        open_position_ids = list(self.open_positions.keys())
        
        for position_id in open_position_ids:
            pnl = self.close_position(position_id, reason)
            if pnl is not None:
                results[position_id] = pnl
        
        self.logger.info(f"Closed {len(results)} positions with reason: {reason}")
        return results
    
    def update_prices(self, price_updates: Dict[str, float]) -> None:
        """
        Update prices for all positions.
        
        Args:
            price_updates: Dictionary of symbol -> new price
        """
        for position_id, position in self.open_positions.items():
            if position.symbol in price_updates:
                position.update_price(price_updates[position.symbol])
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        return self.positions.get(position_id)
    
    def get_open_position(self, symbol: str) -> Optional[Position]:
        """Get open position for a symbol."""
        for position in self.open_positions.values():
            if position.symbol == symbol and position.status == PositionStatus.OPEN:
                return position
        return None
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self.positions.values())
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.open_positions.values())
    
    def get_closed_positions(self, limit: Optional[int] = None) -> List[Position]:
        """Get closed positions."""
        positions = sorted(self.closed_positions, key=lambda p: p.exit_time or datetime.min, reverse=True)
        return positions[:limit] if limit else positions
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        open_positions = self.get_open_positions()
        total_open_value = sum(p.current_value for p in open_positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)
        
        return {
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "total_pnl": self.total_pnl,
            "total_commissions": self.total_commissions,
            "total_return": (self.current_balance - self.initial_balance) / self.initial_balance,
            "open_positions": len(open_positions),
            "total_positions_opened": self.total_positions_opened,
            "total_positions_closed": self.total_positions_closed,
            "total_open_value": total_open_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "portfolio_value": self.current_balance + total_open_value,
            "win_rate": self._calculate_win_rate(),
            "avg_win": self._calculate_average_win(),
            "avg_loss": self._calculate_average_loss(),
            "max_drawdown": self._calculate_max_drawdown(),
            "sharpe_ratio": self._calculate_sharpe_ratio()
        }
    
    def _validate_position_request(self, request: PositionRequest) -> bool:
        """Validate position request."""
        if request.quantity <= 0:
            self.logger.error("Invalid quantity: must be positive")
            return False
        
        if request.entry_price <= 0:
            self.logger.error("Invalid entry price: must be positive")
            return False
        
        if request.stop_loss and request.stop_loss <= 0:
            self.logger.error("Invalid stop loss: must be positive")
            return False
        
        if request.take_profit and request.take_profit <= 0:
            self.logger.error("Invalid take profit: must be positive")
            return False
        
        return True
    
    def _check_position_limits(self, request: PositionRequest) -> bool:
        """Check if position request respects limits."""
        # Check max positions
        if len(self.open_positions) >= self.max_positions:
            self.logger.warning(f"Max positions reached: {self.max_positions}")
            return False
        
        # Check if already have position for this symbol
        existing_position = self.get_open_position(request.symbol)
        if existing_position:
            self.logger.warning(f"Already have open position for {request.symbol}")
            return False
        
        # Check position size
        position_value = request.quantity * request.entry_price
        max_allowed = self.current_balance * self.max_position_size
        
        if position_value > max_allowed:
            self.logger.warning(f"Position too large: ${position_value:.2f} > ${max_allowed:.2f}")
            return False
        
        return True
    
    def _calculate_position_size(self, request: PositionRequest) -> float:
        """Calculate optimal position size."""
        # Base position size from request
        base_size = request.quantity
        
        # Adjust for portfolio constraints
        position_value = base_size * request.entry_price
        max_allowed = self.current_balance * self.max_position_size
        
        if position_value > max_allowed:
            base_size = max_allowed / request.entry_price
        
        # Ensure minimum position size
        min_value = self.current_balance * self.min_position_size
        if base_size * request.entry_price < min_value:
            return 0.0
        
        return base_size
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        if not self.closed_positions:
            return 0.0
        
        winning_positions = [p for p in self.closed_positions if p.realized_pnl > 0]
        return len(winning_positions) / len(self.closed_positions)
    
    def _calculate_average_win(self) -> float:
        """Calculate average winning trade."""
        winning_positions = [p for p in self.closed_positions if p.realized_pnl > 0]
        
        if not winning_positions:
            return 0.0
        
        return sum(p.realized_pnl for p in winning_positions) / len(winning_positions)
    
    def _calculate_average_loss(self) -> float:
        """Calculate average losing trade."""
        losing_positions = [p for p in self.closed_positions if p.realized_pnl < 0]
        
        if not losing_positions:
            return 0.0
        
        return sum(p.realized_pnl for p in losing_positions) / len(losing_positions)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.closed_positions:
            return 0.0
        
        max_drawdown = 0.0
        for position in self.closed_positions:
            max_drawdown = max(max_drawdown, position.max_drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)."""
        if not self.closed_positions:
            return 0.0
        
        # Simplified Sharpe ratio calculation
        returns = [p.realized_pnl / self.initial_balance for p in self.closed_positions]
        
        if not returns:
            return 0.0
        
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        
        if variance == 0:
            return 0.0
        
        # Assuming risk-free rate of 2% annually
        risk_free_rate = 0.02
        return (avg_return - risk_free_rate) / (variance ** 0.5)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary."""
        open_positions = self.get_open_positions()
        
        return {
            "total_exposure": sum(p.current_value for p in open_positions),
            "max_position_size": self.max_position_size,
            "current_utilization": sum(p.current_value for p in open_positions) / self.current_balance,
            "positions_at_risk": len([p for p in open_positions if p.unrealized_pnl < 0]),
            "total_unrealized_loss": sum(p.unrealized_pnl for p in open_positions if p.unrealized_pnl < 0),
            "stop_loss_coverage": len([p for p in open_positions if p.stop_loss]) / len(open_positions) if open_positions else 0,
            "take_profit_coverage": len([p for p in open_positions if p.take_profit]) / len(open_positions) if open_positions else 0
        }
