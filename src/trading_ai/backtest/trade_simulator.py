"""
Trade simulator for backtesting.
Following patterns from VectorBT and AgentQuant repositories.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

from ..core.models import Signal, SignalDirection
from ..infrastructure.logging import get_logger


@dataclass
class Trade:
    """Trade record."""
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_pct: float
    status: str  # "open", "closed", "stopped", "profit_taken"
    fees: float
    metadata: Dict[str, Any]


@dataclass
class Portfolio:
    """Portfolio state."""
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    open_trades: List[Trade]
    closed_trades: List[Trade]
    total_value: float
    peak_value: float
    drawdown: float
    max_drawdown: float


class TradeSimulator:
    """
    Trade simulator for backtesting.
    
    Following patterns from:
    - VectorBT: Vectorized backtesting
    - AgentQuant: Walk-forward validation
    - Backtrader: Trade execution simulation
    """
    
    def __init__(self, initial_cash: float = 100000.0, commission_rate: float = 0.001):
        """Initialize trade simulator."""
        self.logger = get_logger("trade_simulator")
        
        # Portfolio configuration
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = 0.0005  # 0.05% slippage
        
        # Portfolio state
        self.portfolio = Portfolio(
            cash=initial_cash,
            positions={},
            open_trades=[],
            closed_trades=[],
            total_value=initial_cash,
            peak_value=initial_cash,
            drawdown=0.0,
            max_drawdown=0.0
        )
        
        # Trade configuration
        self.default_stop_loss = 0.05  # 5%
        self.default_take_profit = 0.15  # 15%
        self.max_position_size = 0.2  # 20% of portfolio
        
        # Price data cache
        self.price_cache: Dict[str, List[Tuple[datetime, float]]] = {}
        
        self.logger.info(f"Trade simulator initialized with ${initial_cash:,.2f}")
    
    def execute_signal(self, signal: Signal, current_price: float, timestamp: datetime) -> Optional[Trade]:
        """
        Execute trading signal.
        
        Args:
            signal: Trading signal
            current_price: Current price
            timestamp: Execution timestamp
            
        Returns:
            Executed trade or None
        """
        try:
            # Calculate position size
            position_value = self.portfolio.total_value * signal.position_size
            quantity = position_value / current_price
            
            # Apply slippage
            entry_price = self._apply_slippage(current_price, signal.direction)
            
            # Calculate fees
            fees = position_value * self.commission_rate
            
            # Check if enough cash
            if signal.direction == SignalDirection.BUY:
                required_cash = position_value + fees
                if self.portfolio.cash < required_cash:
                    self.logger.warning(f"Insufficient cash for {signal.symbol}: need ${required_cash:.2f}, have ${self.portfolio.cash:.2f}")
                    return None
                
                # Execute buy
                self.portfolio.cash -= required_cash
                self.portfolio.positions[signal.symbol] = self.portfolio.positions.get(signal.symbol, 0) + quantity
                
            else:  # SELL
                current_position = self.portfolio.positions.get(signal.symbol, 0)
                if current_position <= 0:
                    self.logger.warning(f"No position to sell for {signal.symbol}")
                    return None
                
                # Execute sell (use full position)
                quantity = min(quantity, current_position)
                position_value = quantity * entry_price
                fees = position_value * self.commission_rate
                
                self.portfolio.cash += position_value - fees
                self.portfolio.positions[signal.symbol] = current_position - quantity
            
            # Calculate stop loss and take profit
            if signal.direction == SignalDirection.BUY:
                stop_loss = entry_price * (1 - self.default_stop_loss)
                take_profit = entry_price * (1 + self.default_take_profit)
            else:  # SELL
                stop_loss = entry_price * (1 + self.default_stop_loss)
                take_profit = entry_price * (1 - self.default_take_profit)
            
            # Create trade
            trade = Trade(
                symbol=signal.symbol,
                direction=signal.direction.value,
                quantity=quantity,
                entry_price=entry_price,
                exit_price=None,
                entry_time=timestamp,
                exit_time=None,
                stop_loss=stop_loss,
                take_profit=take_profit,
                pnl=0.0,
                pnl_pct=0.0,
                status="open",
                fees=fees,
                metadata={
                    "signal_confidence": signal.confidence,
                    "signal_reasoning": signal.metadata.get("reasoning", ""),
                    "signal_urgency": signal.urgency.value,
                    "market_regime": signal.market_regime.value
                }
            )
            
            self.portfolio.open_trades.append(trade)
            
            # Update portfolio value
            self._update_portfolio_value(timestamp)
            
            self.logger.info(f"Executed {signal.direction.value} {signal.symbol}: {quantity:.4f} @ ${entry_price:.2f}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Failed to execute signal: {e}")
            return None
    
    def update_trades(self, current_prices: Dict[str, float], timestamp: datetime) -> List[Trade]:
        """
        Update open trades and check for exits.
        
        Args:
            current_prices: Current prices for all symbols
            timestamp: Current timestamp
            
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for trade in self.portfolio.open_trades[:]:
            if trade.symbol not in current_prices:
                continue
            
            current_price = current_prices[trade.symbol]
            
            # Check stop loss
            if trade.direction == "BUY" and current_price <= trade.stop_loss:
                self._close_trade(trade, current_price, timestamp, "stopped")
                closed_trades.append(trade)
            elif trade.direction == "SELL" and current_price >= trade.stop_loss:
                self._close_trade(trade, current_price, timestamp, "stopped")
                closed_trades.append(trade)
            
            # Check take profit
            elif trade.direction == "BUY" and current_price >= trade.take_profit:
                self._close_trade(trade, current_price, timestamp, "profit_taken")
                closed_trades.append(trade)
            elif trade.direction == "SELL" and current_price <= trade.take_profit:
                self._close_trade(trade, current_price, timestamp, "profit_taken")
                closed_trades.append(trade)
        
        # Update portfolio value
        self._update_portfolio_value(timestamp)
        
        return closed_trades
    
    def close_all_trades(self, current_prices: Dict[str, float], timestamp: datetime) -> List[Trade]:
        """Close all open trades."""
        closed_trades = []
        
        for trade in self.portfolio.open_trades[:]:
            if trade.symbol in current_prices:
                current_price = current_prices[trade.symbol]
                self._close_trade(trade, current_price, timestamp, "manual_close")
                closed_trades.append(trade)
        
        return closed_trades
    
    def _close_trade(self, trade: Trade, exit_price: float, timestamp: datetime, exit_reason: str):
        """Close a trade."""
        # Apply slippage
        exit_price = self._apply_slippage(exit_price, "BUY" if trade.direction == "SELL" else "SELL")
        
        # Calculate P&L
        if trade.direction == "BUY":
            pnl = (exit_price - trade.entry_price) * trade.quantity
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:  # SELL
            pnl = (trade.entry_price - exit_price) * trade.quantity
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
        
        # Calculate exit fees
        exit_value = trade.quantity * exit_price
        exit_fees = exit_value * self.commission_rate
        pnl -= exit_fees
        
        # Update trade
        trade.exit_price = exit_price
        trade.exit_time = timestamp
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.status = exit_reason
        trade.fees += exit_fees
        
        # Update portfolio
        if trade.direction == "BUY":
            self.portfolio.cash += exit_value - exit_fees
            self.portfolio.positions[trade.symbol] = self.portfolio.positions.get(trade.symbol, 0) - trade.quantity
        else:  # SELL
            self.portfolio.cash += exit_value - exit_fees
            self.portfolio.positions[trade.symbol] = self.portfolio.positions.get(trade.symbol, 0) + trade.quantity
        
        # Move to closed trades
        self.portfolio.open_trades.remove(trade)
        self.portfolio.closed_trades.append(trade)
        
        self.logger.info(f"Closed {trade.direction} {trade.symbol}: P&L ${pnl:.2f} ({pnl_pct:.2%}) - {exit_reason}")
    
    def _apply_slippage(self, price: float, direction: str) -> float:
        """Apply slippage to price."""
        if direction == "BUY":
            return price * (1 + self.slippage_rate)
        else:  # SELL
            return price * (1 - self.slippage_rate)
    
    def _update_portfolio_value(self, timestamp: datetime):
        """Update portfolio value and drawdown."""
        # Calculate total value
        total_value = self.portfolio.cash
        
        for symbol, quantity in self.portfolio.positions.items():
            if quantity != 0 and symbol in self.price_cache:
                # Get latest price
                latest_price = self.price_cache[symbol][-1][1] if self.price_cache[symbol] else 0.0
                total_value += quantity * latest_price
        
        self.portfolio.total_value = total_value
        
        # Update peak value and drawdown
        if total_value > self.portfolio.peak_value:
            self.portfolio.peak_value = total_value
            self.portfolio.drawdown = 0.0
        else:
            self.portfolio.drawdown = (self.portfolio.peak_value - total_value) / self.portfolio.peak_value
            self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, self.portfolio.drawdown)
    
    def add_price_data(self, symbol: str, timestamp: datetime, price: float):
        """Add price data for simulation."""
        if symbol not in self.price_cache:
            self.price_cache[symbol] = []
        
        self.price_cache[symbol].append((timestamp, price))
        
        # Keep only recent data (last 1000 points)
        if len(self.price_cache[symbol]) > 1000:
            self.price_cache[symbol] = self.price_cache[symbol][-1000:]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        total_trades = len(self.portfolio.open_trades) + len(self.portfolio.closed_trades)
        winning_trades = [t for t in self.portfolio.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.portfolio.closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.portfolio.closed_trades) if self.portfolio.closed_trades else 0.0
        
        total_pnl = sum(t.pnl for t in self.portfolio.closed_trades)
        total_fees = sum(t.fees for t in self.portfolio.closed_trades)
        
        # Calculate average trade metrics
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        # Profit factor
        gross_wins = sum(t.pnl for t in winning_trades)
        gross_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        return {
            "total_value": self.portfolio.total_value,
            "cash": self.portfolio.cash,
            "positions": self.portfolio.positions.copy(),
            "open_trades": len(self.portfolio.open_trades),
            "closed_trades": len(self.portfolio.closed_trades),
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "net_pnl": total_pnl - total_fees,
            "peak_value": self.portfolio.peak_value,
            "current_drawdown": self.portfolio.drawdown,
            "max_drawdown": self.portfolio.max_drawdown,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "return_pct": (self.portfolio.total_value - self.initial_cash) / self.initial_cash
        }
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history."""
        all_trades = self.portfolio.open_trades + self.portfolio.closed_trades
        
        return [
            {
                "symbol": trade.symbol,
                "direction": trade.direction,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "stop_loss": trade.stop_loss,
                "take_profit": trade.take_profit,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "status": trade.status,
                "fees": trade.fees,
                "duration": (trade.exit_time - trade.entry_time).total_seconds() / 3600 if trade.exit_time else None,
                "metadata": trade.metadata
            }
            for trade in all_trades
        ]
    
    def reset(self):
        """Reset simulator to initial state."""
        self.portfolio = Portfolio(
            cash=self.initial_cash,
            positions={},
            open_trades=[],
            closed_trades=[],
            total_value=self.initial_cash,
            peak_value=self.initial_cash,
            drawdown=0.0,
            max_drawdown=0.0
        )
        self.price_cache.clear()
        
        self.logger.info("Trade simulator reset")
