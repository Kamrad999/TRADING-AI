"""Order class — pure data structure following Zipline/Backtrader patterns.

Order represents intent to trade. No execution logic here.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order lifecycle status — Backtrader/Zipline pattern."""
    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Order — strict dataclass representing trading intent.
    
    SIGNED QUANTITY MODEL:
    - quantity > 0: BUY order
    - quantity < 0: SELL order
    - No OrderSide enum needed
    
    Fields:
        id: Unique order identifier
        symbol: Trading symbol
        quantity: Signed quantity (positive=buy, negative=sell)
        order_type: Order type ("market" only for now)
        timestamp: When order was created
        filled_quantity: How much has been filled
        avg_fill_price: Average fill price
        status: Current order status
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
    """
    id: str
    symbol: str
    quantity: float  # SIGNED: positive=buy, negative=sell
    order_type: str  # "market" only for now
    timestamp: datetime
    
    # Execution tracking
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    
    # State
    status: OrderStatus = field(default=OrderStatus.CREATED)
    
    # Optional risk parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def fill(self, quantity: float, price: float) -> None:
        """
        Record a fill for this order.
        
        Args:
            quantity: Signed fill quantity (positive for buy fill, negative for sell fill)
            price: Fill price
        
        Updates:
            - filled_quantity: cumulative filled amount
            - avg_fill_price: weighted average of all fills
            - status: PARTIALLY_FILLED or FILLED based on completion
        """
        # Record pre-fill state for weighted average calculation
        old_filled_qty = self.filled_quantity
        new_filled_qty = old_filled_qty + quantity
        
        # SAFETY CHECK: never exceed original order quantity
        assert abs(new_filled_qty) <= abs(self.quantity), \
            f"Fill would exceed order quantity: {new_filled_qty} vs {self.quantity}"
        
        # Update filled quantity
        self.filled_quantity = new_filled_qty
        
        # Update weighted average fill price
        # Formula: new_avg = (old_avg * old_qty + price * fill_qty) / new_total_qty
        if new_filled_qty != 0:
            total_value = (self.avg_fill_price * old_filled_qty) + (price * quantity)
            self.avg_fill_price = total_value / new_filled_qty
        else:
            # Edge case: net zero quantity (shouldn't happen in normal trading)
            self.avg_fill_price = 0.0
        
        # Update status based on fill completion
        if abs(self.filled_quantity) == abs(self.quantity):
            # Complete fill
            self.status = OrderStatus.FILLED
        elif abs(self.filled_quantity) > 0:
            # Partial fill
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self) -> None:
        """
        Cancel this order.
        
        Raises:
            RuntimeError: If order is already FILLED
        """
        if self.status == OrderStatus.FILLED:
            raise RuntimeError(f"Cannot cancel filled order {self.id}")
        
        self.status = OrderStatus.CANCELED
