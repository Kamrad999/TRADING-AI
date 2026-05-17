"""Portfolio manager for AMATIS.

Institutional-grade portfolio tracking and management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import EventType
from amatix.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Portfolio position with full tracking."""
    symbol: str
    side: str  # "long", "short", "flat"
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Optional[Decimal] = None
    market_value: Optional[Decimal] = None
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    
    # Metadata
    sector: Optional[str] = None
    asset_class: str = "equity"  # equity, crypto, forex, etc.
    opened_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None


@dataclass
class PortfolioState:
    """Complete portfolio state snapshot."""
    timestamp: datetime
    
    # Values
    cash: Decimal
    total_value: Decimal
    buying_power: Decimal
    
    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # Exposure
    gross_exposure: Decimal = field(default_factory=lambda: Decimal("0"))
    net_exposure: Decimal = field(default_factory=lambda: Decimal("0"))
    long_exposure: Decimal = field(default_factory=lambda: Decimal("0"))
    short_exposure: Decimal = field(default_factory=lambda: Decimal("0"))
    
    # Risk metrics
    daily_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    current_drawdown: float = 0.0
    
    # Allocation
    sector_exposure: Dict[str, Decimal] = field(default_factory=dict)
    asset_class_exposure: Dict[str, Decimal] = field(default_factory=dict)


class PortfolioManager:
    """Institutional-grade portfolio manager.
    
    Responsibilities:
        - Track all positions
        - Calculate exposures
        - Monitor P&L
        - Enforce limits
        - Generate snapshots
    
    Event-driven updates:
        - ORDER_FILLED → Update position
        - POSITION_UPDATE → Recalculate exposure
        - PRICE_UPDATE → Update unrealized P&L
    """
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        initial_cash: Decimal = Decimal("100000"),
    ) -> None:
        self._event_bus = event_bus
        self._cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._snapshot_history: List[PortfolioState] = []
        
        # Configuration
        self._max_gross_exposure = Decimal("2.0")  # 200%
        self._max_net_exposure = Decimal("1.5")  # 150%
        self._max_concentration = Decimal("0.2")  # 20% per position
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """Subscribe to portfolio-relevant events."""
        
        @self._event_bus.on(EventType.ORDER_FILLED)
        async def on_fill(event):
            await self._handle_fill(event)
        
        @self._event_bus.on(EventType.POSITION_UPDATED)
        async def on_position_update(event):
            await self._handle_position_update(event)
    
    async def _handle_fill(self, event) -> None:
        """Process order fill and update position."""
        symbol = event.payload.get("symbol")
        side = event.payload.get("side")
        filled_qty = Decimal(str(event.payload.get("filled_quantity", "0")))
        filled_price = Decimal(str(event.payload.get("filled_price", "0")))
        
        if symbol not in self._positions:
            # New position
            self._positions[symbol] = Position(
                symbol=symbol,
                side="long" if side == "buy" else "short",
                quantity=filled_qty,
                avg_entry_price=filled_price,
                current_price=filled_price,
                opened_at=datetime.utcnow(),
            )
        else:
            # Update existing position
            pos = self._positions[symbol]
            
            if side == "buy":
                # Adding to long
                if pos.side == "long":
                    # Average up
                    total_cost = pos.quantity * pos.avg_entry_price + filled_qty * filled_price
                    pos.quantity += filled_qty
                    pos.avg_entry_price = total_cost / pos.quantity
                else:
                    # Reducing short
                    if filled_qty >= pos.quantity:
                        # Flip to long
                        remaining = filled_qty - pos.quantity
                        pos.realized_pnl += self._calculate_close_pnl(
                            pos, pos.quantity, filled_price
                        )
                        pos.side = "long"
                        pos.quantity = remaining
                        pos.avg_entry_price = filled_price
                    else:
                        # Still short
                        pos.quantity -= filled_qty
                        pos.realized_pnl += self._calculate_close_pnl(
                            pos, filled_qty, filled_price
                        )
            else:
                # Sell
                if pos.side == "long":
                    if filled_qty >= pos.quantity:
                        # Close long
                        pos.realized_pnl += self._calculate_close_pnl(
                            pos, pos.quantity, filled_price
                        )
                        pos.side = "flat"
                        pos.quantity = Decimal("0")
                    else:
                        # Partial close
                        pos.quantity -= filled_qty
                        pos.realized_pnl += self._calculate_close_pnl(
                            pos, filled_qty, filled_price
                        )
                else:
                    # Adding to short
                    total_cost = pos.quantity * pos.avg_entry_price + filled_qty * filled_price
                    pos.quantity += filled_qty
                    pos.avg_entry_price = total_cost / pos.quantity
            
            pos.current_price = filled_price
            pos.last_updated = datetime.utcnow()
        
        # Update cash
        fill_cost = filled_qty * filled_price
        if side == "buy":
            self._cash -= fill_cost
        else:
            self._cash += fill_cost
        
        # Emit position update
        await self._event_bus.emit_new(
            EventType.POSITION_UPDATED,
            {
                "symbol": symbol,
                "quantity": str(self._positions[symbol].quantity),
                "avg_price": str(self._positions[symbol].avg_entry_price),
            },
            source="portfolio_manager",
        )
        
        logger.info(
            "Position updated",
            symbol=symbol,
            side=side,
            quantity=str(filled_qty),
            cash=str(self._cash),
        )
    
    def _calculate_close_pnl(
        self,
        position: Position,
        quantity: Decimal,
        close_price: Decimal,
    ) -> Decimal:
        """Calculate P&L for closing part of position."""
        if position.side == "long":
            return quantity * (close_price - position.avg_entry_price)
        else:
            return quantity * (position.avg_entry_price - close_price)
    
    async def _handle_position_update(self, event) -> None:
        """Handle position update event."""
        # Recalculate portfolio state
        await self._recalculate_state()
    
    async def _recalculate_state(self) -> None:
        """Recalculate all portfolio metrics."""
        # Calculate exposures
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")
        sector_exposure: Dict[str, Decimal] = {}
        asset_exposure: Dict[str, Decimal] = {}
        
        for symbol, pos in self._positions.items():
            if pos.side == "flat" or pos.quantity == 0:
                continue
            
            exposure = pos.quantity * (pos.current_price or pos.avg_entry_price)
            
            if pos.side == "long":
                long_exposure += exposure
            else:
                short_exposure += exposure
            
            # Sector exposure
            sector = pos.sector or "unknown"
            sector_exposure[sector] = sector_exposure.get(sector, Decimal("0")) + exposure
            
            # Asset class exposure
            asset_exposure[pos.asset_class] = asset_exposure.get(
                pos.asset_class, Decimal("0")
            ) + exposure
        
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        # Check limits
        portfolio_value = self._cash + gross_exposure
        
        if portfolio_value > 0:
            gross_pct = gross_exposure / portfolio_value
            net_pct = abs(net_exposure) / portfolio_value
            
            if gross_pct > self._max_gross_exposure:
                logger.warning(
                    "Gross exposure limit exceeded",
                    exposure_pct=float(gross_pct),
                    limit=float(self._max_gross_exposure),
                )
            
            if net_pct > self._max_net_exposure:
                logger.warning(
                    "Net exposure limit exceeded",
                    exposure_pct=float(net_pct),
                    limit=float(self._max_net_exposure),
                )
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all non-flat positions."""
        return [
            pos for pos in self._positions.values()
            if pos.side != "flat" and pos.quantity > 0
        ]
    
    def get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value."""
        position_value = sum(
            pos.quantity * (pos.current_price or pos.avg_entry_price)
            for pos in self._positions.values()
            if pos.side != "flat"
        )
        return self._cash + position_value
    
    def get_exposure(self) -> Dict[str, Decimal]:
        """Get exposure metrics."""
        long_exp = Decimal("0")
        short_exp = Decimal("0")
        
        for pos in self._positions.values():
            if pos.side == "long":
                long_exp += pos.quantity * (pos.current_price or pos.avg_entry_price)
            elif pos.side == "short":
                short_exp += pos.quantity * (pos.current_price or pos.avg_entry_price)
        
        return {
            "gross": long_exp + short_exp,
            "net": long_exp - short_exp,
            "long": long_exp,
            "short": short_exp,
        }
    
    def can_add_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
    ) -> tuple[bool, Optional[str]]:
        """Check if adding position would violate constraints.
        
        Returns:
            (allowed, reason) tuple
        """
        portfolio_value = self.get_portfolio_value()
        if portfolio_value == 0:
            return True, None
        
        # Check concentration
        position_value = quantity * price
        concentration = position_value / portfolio_value
        
        if concentration > self._max_concentration:
            return False, f"Concentration {concentration:.1%} exceeds max {self._max_concentration:.1%}"
        
        # Check exposure
        exposure = self.get_exposure()
        new_gross = exposure["gross"] + position_value
        gross_pct = new_gross / portfolio_value
        
        if gross_pct > self._max_gross_exposure:
            return False, f"Gross exposure {gross_pct:.1%} would exceed max"
        
        return True, None
    
    def get_state(self) -> PortfolioState:
        """Get current portfolio state snapshot."""
        exposure = self.get_exposure()
        
        return PortfolioState(
            timestamp=datetime.utcnow(),
            cash=self._cash,
            total_value=self.get_portfolio_value(),
            buying_power=self._cash * Decimal("2"),  # Assuming 2:1 margin
            positions=self._positions.copy(),
            gross_exposure=exposure["gross"],
            net_exposure=exposure["net"],
            long_exposure=exposure["long"],
            short_exposure=exposure["short"],
        )
    
    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update market price for position."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = price
            
            # Recalculate unrealized P&L
            if pos.side == "long":
                pos.unrealized_pnl = pos.quantity * (price - pos.avg_entry_price)
            elif pos.side == "short":
                pos.unrealized_pnl = pos.quantity * (pos.avg_entry_price - price)
            
            pos.market_value = pos.quantity * price
            pos.last_updated = datetime.utcnow()
