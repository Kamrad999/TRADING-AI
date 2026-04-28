"""Position core - Clean implementation following Backtrader patterns.

Key principles:
- Position = signed quantity * price
- SINGLE PnL formula: (current_price - entry_price) * quantity
- Works for BOTH long and short automatically
- NO logic inside position except math
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class PositionSide(Enum):
    """Position side - LONG or SHORT."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position status - OPEN or CLOSED."""
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Position:
    """
    Position class - PURE math, no side effects.
    
    Quantity is SIGNED:
    - Positive = LONG
    - Negative = SHORT
    
    PnL formula (SINGLE formula for ALL cases):
        pnl = (exit_price - entry_price) * quantity
    
    Examples:
    - Long: qty=+1, entry=100, exit=120 → (120-100)*+1 = +20 ✓
    - Short: qty=-1, entry=100, exit=120 → (120-100)*-1 = -20 ✓
    """
    
    # Core attributes (REQUIRED)
    symbol: str
    entry_price: float
    quantity: float  # POSITIVE = long, NEGATIVE = short
    
    # Current state
    current_price: float = field(default=0.0)
    
    # Exit tracking
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_time: Optional[datetime] = None
    
    # Risk parameters (set by external system)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Internal state
    is_open: bool = field(default=True, init=False)
    entry_time: datetime = field(default_factory=datetime.now, init=False)
    
    def __post_init__(self):
        """Initialize current price if not set."""
        if self.current_price == 0.0:
            self.current_price = self.entry_price
    
    def update_price(self, price: float) -> None:
        """Update current price. ONLY updates price — NO hidden logic."""
        self.current_price = price
    
    def get_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL. SINGLE formula works for long AND short."""
        if not self.is_open:
            return self.get_realized_pnl()
        return (self.current_price - self.entry_price) * self.quantity
    
    def get_realized_pnl(self) -> float:
        """Get realized PnL (after close)."""
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.quantity
    
    def close(self, price: Optional[float] = None, reason: str = "manual") -> float:
        """Close position at given price. Returns realized PnL."""
        if not self.is_open:
            return self.get_realized_pnl()
        
        close_price = price if price is not None else self.current_price
        
        self.exit_price = close_price
        self.exit_reason = reason
        self.exit_time = datetime.now()
        self.current_price = close_price
        self.is_open = False
        
        return self.get_realized_pnl()
    
    @property
    def is_long(self) -> bool:
        """True if position is long (quantity > 0)."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """True if position is short (quantity < 0)."""
        return self.quantity < 0
    
    def add(self, quantity: float, price: float, timestamp: datetime) -> None:
        """
        Add to existing position (Backtrader pattern).
        Updates quantity and recalculates average entry price.
        
        Formula: new_entry = (old_value + new_value) / total_quantity
        """
        if self.is_open:
            # Calculate weighted average entry price
            old_value = self.entry_price * self.quantity
            new_value = price * quantity
            total_quantity = self.quantity + quantity
            
            if total_quantity != 0:
                self.entry_price = (old_value + new_value) / total_quantity
            self.quantity = total_quantity
        else:
            # Position was closed, treat as new entry
            self.entry_price = price
            self.quantity = quantity
            self.is_open = True
            self.exit_price = None
            self.exit_reason = None
            self.exit_time = None
        
        self.current_price = price
        self.entry_time = timestamp
