"""
Exchange integration with ccxt support.
Following patterns from ccxt and ai-trade repositories.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import json

from ..infrastructure.logging import get_logger
from ..infrastructure.config import config


@dataclass
class Order:
    """Order structure."""
    symbol: str
    side: str  # "buy" or "sell"
    type: str  # "market" or "limit"
    amount: float
    price: Optional[float]
    status: str
    order_id: Optional[str]
    timestamp: datetime
    filled: float
    remaining: float
    cost: float
    fees: Dict[str, float]


@dataclass
class Position:
    """Position structure."""
    symbol: str
    side: str  # "long" or "short"
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    percentage: float
    timestamp: datetime


class Exchange:
    """
    Exchange integration with ccxt support.
    
    Following patterns from:
    - ccxt: Unified exchange API
    - ai-trade: Order management system
    - MetaAPI: Real-time execution
    """
    
    def __init__(self, exchange_name: str = "binance", paper_trading: bool = True):
        """Initialize exchange connection."""
        self.logger = get_logger("exchange")
        
        self.exchange_name = exchange_name
        self.paper_trading = paper_trading
        
        # Configuration
        self.api_key = config.get("EXCHANGE_API_KEY", "")
        self.api_secret = config.get("EXCHANGE_API_SECRET", "")
        self.sandbox = config.get("EXCHANGE_SANDBOX", True)
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        
        # Trading configuration
        self.default_order_type = "market"
        self.commission_rate = 0.001  # 0.1%
        self.min_order_amount = 0.001  # Minimum order size
        
        # Mock data for paper trading
        self.mock_balances = {
            "BTC": 1.0,
            "ETH": 10.0,
            "USDT": 100000.0
        }
        
        self.mock_prices = {
            "BTC/USDT": 50000.0,
            "ETH/USDT": 3000.0,
            "BTC/ETH": 16.67
        }
        
        self.logger.info(f"Exchange initialized: {exchange_name} (paper_trading={paper_trading})")
    
    def connect(self) -> bool:
        """Connect to exchange."""
        try:
            if self.paper_trading:
                self.logger.info("Connected to paper trading exchange")
                return True
            else:
                # In production, implement real ccxt connection
                self.logger.info(f"Connected to {self.exchange_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            return False
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        try:
            if self.paper_trading:
                return self.mock_balances.copy()
            else:
                # In production, use ccxt to get real balance
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker information."""
        try:
            if self.paper_trading:
                price = self.mock_prices.get(symbol, 0.0)
                
                return {
                    "symbol": symbol,
                    "price": price,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "volume": 1000000.0,
                    "timestamp": datetime.now(),
                    "bid": price * 0.999,
                    "ask": price * 1.001
                }
            else:
                # In production, use ccxt to get real ticker
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None
    
    def create_order(self, symbol: str, side: str, order_type: str, 
                    amount: float, price: Optional[float] = None) -> Optional[Order]:
        """Create order."""
        try:
            # Validate order
            if not self._validate_order(symbol, side, order_type, amount, price):
                return None
            
            # Generate order ID
            order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.orders)}"
            
            # Get current price
            ticker = self.get_ticker(symbol)
            if not ticker:
                return None
            
            current_price = ticker["price"]
            
            # Determine execution price
            if order_type == "market":
                execution_price = current_price
            else:
                execution_price = price or current_price
            
            # Calculate cost
            cost = amount * execution_price
            fees = cost * self.commission_rate
            
            # Create order
            order = Order(
                symbol=symbol,
                side=side,
                type=order_type,
                amount=amount,
                price=price,
                status="open",
                order_id=order_id,
                timestamp=datetime.now(),
                filled=0.0,
                remaining=amount,
                cost=0.0,
                fees={"trading": fees}
            )
            
            # Execute order immediately for paper trading
            if self.paper_trading:
                order = self._execute_order(order, execution_price)
            
            self.orders[order_id] = order
            
            self.logger.info(f"Order created: {side} {amount} {symbol} @ {execution_price}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to create order: {e}")
            return None
    
    def _execute_order(self, order: Order, execution_price: float) -> Order:
        """Execute order (paper trading)."""
        try:
            # Update order
            order.filled = order.amount
            order.remaining = 0.0
            order.cost = order.filled * execution_price
            order.status = "closed"
            
            # Update balances
            if order.side == "buy":
                # Deduct cost from quote currency
                base_currency, quote_currency = order.symbol.split("/")
                self.mock_balances[quote_currency] -= order.cost + order.fees["trading"]
                self.mock_balances[base_currency] += order.filled
                
                # Create/update position
                if base_currency in self.positions:
                    pos = self.positions[base_currency]
                    # Update average entry price
                    total_cost = pos.amount * pos.entry_price + order.cost
                    total_amount = pos.amount + order.filled
                    pos.entry_price = total_cost / total_amount
                    pos.amount += order.filled
                    pos.current_price = execution_price
                    pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.amount
                    pos.percentage = pos.unrealized_pnl / (pos.entry_price * pos.amount)
                else:
                    self.positions[base_currency] = Position(
                        symbol=base_currency,
                        side="long",
                        amount=order.filled,
                        entry_price=execution_price,
                        current_price=execution_price,
                        unrealized_pnl=0.0,
                        percentage=0.0,
                        timestamp=datetime.now()
                    )
                
            else:  # sell
                # Add proceeds to quote currency
                base_currency, quote_currency = order.symbol.split("/")
                proceeds = order.cost - order.fees["trading"]
                self.mock_balances[quote_currency] += proceeds
                self.mock_balances[base_currency] -= order.filled
                
                # Update or close position
                if base_currency in self.positions:
                    pos = self.positions[base_currency]
                    pos.amount -= order.filled
                    pos.current_price = execution_price
                    
                    if pos.amount <= 0:
                        # Position closed
                        realized_pnl = (execution_price - pos.entry_price) * order.filled
                        pos.unrealized_pnl = realized_pnl
                        pos.percentage = realized_pnl / (pos.entry_price * order.filled)
                        pos.amount = 0
                    else:
                        # Position partially closed
                        pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.amount
                        pos.percentage = pos.unrealized_pnl / (pos.entry_price * pos.amount)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to execute order: {e}")
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            if order_id in self.orders:
                order = self.orders[order_id]
                if order.status == "open":
                    order.status = "cancelled"
                    self.logger.info(f"Order cancelled: {order_id}")
                    return True
                else:
                    self.logger.warning(f"Cannot cancel order {order_id}: status is {order.status}")
                    return False
            else:
                self.logger.warning(f"Order not found: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        return self.orders.get(order_id)
    
    def get_open_orders(self) -> List[Order]:
        """Get open orders."""
        return [order for order in self.orders.values() if order.status == "open"]
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        # Update position values
        for symbol, position in self.positions.items():
            ticker = self.get_ticker(f"{symbol}/USDT")
            if ticker:
                position.current_price = ticker["price"]
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.amount
                position.percentage = position.unrealized_pnl / (position.entry_price * position.amount)
        
        return [pos for pos in self.positions.values() if pos.amount > 0]
    
    def _validate_order(self, symbol: str, side: str, order_type: str, 
                       amount: float, price: Optional[float]) -> bool:
        """Validate order parameters."""
        # Check side
        if side not in ["buy", "sell"]:
            return False
        
        # Check order type
        if order_type not in ["market", "limit"]:
            return False
        
        # Check amount
        if amount <= 0:
            return False
        
        # Check minimum order amount
        if amount < self.min_order_amount:
            return False
        
        # Check limit order price
        if order_type == "limit" and (price is None or price <= 0):
            return False
        
        # Check balance for buy orders
        if side == "buy":
            ticker = self.get_ticker(symbol)
            if not ticker:
                return False
            
            cost = amount * (price or ticker["price"])
            fees = cost * self.commission_rate
            total_cost = cost + fees
            
            base_currency, quote_currency = symbol.split("/")
            available_balance = self.mock_balances.get(quote_currency, 0.0)
            
            if available_balance < total_cost:
                return False
        
        # Check position for sell orders
        if side == "sell":
            base_currency, quote_currency = symbol.split("/")
            available_position = self.mock_balances.get(base_currency, 0.0)
            
            if available_position < amount:
                return False
        
        return True
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history."""
        all_orders = list(self.orders.values())
        all_orders.sort(key=lambda o: o.timestamp, reverse=True)
        return all_orders[:limit]
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history."""
        trades = []
        
        for order in self.orders.values():
            if order.status == "closed":
                trade = {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.type,
                    "amount": order.filled,
                    "price": order.cost / order.filled if order.filled > 0 else 0.0,
                    "cost": order.cost,
                    "fees": order.fees,
                    "timestamp": order.timestamp
                }
                trades.append(trade)
        
        trades.sort(key=lambda t: t["timestamp"], reverse=True)
        return trades[:limit]
    
    def update_prices(self, price_updates: Dict[str, float]):
        """Update prices for paper trading."""
        self.mock_prices.update(price_updates)
        
        # Update position values
        for symbol, position in self.positions.items():
            ticker = self.get_ticker(f"{symbol}/USDT")
            if ticker:
                position.current_price = ticker["price"]
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.amount
                position.percentage = position.unrealized_pnl / (position.entry_price * position.amount)
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary."""
        positions = self.get_positions()
        
        total_value = sum(pos.unrealized_pnl for pos in positions)
        total_value += self.mock_balances.get("USDT", 0.0)
        
        return {
            "total_value": total_value,
            "balances": self.mock_balances.copy(),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "amount": pos.amount,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "percentage": pos.percentage
                }
                for pos in positions
            ],
            "open_orders": len(self.get_open_orders()),
            "total_orders": len(self.orders)
        }
    
    def disconnect(self):
        """Disconnect from exchange."""
        try:
            self.logger.info(f"Disconnected from {self.exchange_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to disconnect: {e}")
