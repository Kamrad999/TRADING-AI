"""Realistic Execution Simulator for AMATIS.

Institutional-grade execution modeling with:
    - Slippage based on order size and volatility
    - Spread widening under stress
    - Delayed and partial fills
    - Liquidity exhaustion
    - Volatility-based execution degradation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from amatix.core.observability import get_logger
from amatix.data.models import OHLCV, Quote

logger = get_logger(__name__)


class SlippageModelType(Enum):
    """Types of slippage models."""
    FIXED = auto()           # Fixed bps slippage
    LINEAR = auto()          # Linear with size
    SQUARE_ROOT = auto()     # Square root with size (institutional standard)
    VOLATILITY_ADJUSTED = auto()  # Adjusted for market volatility


@dataclass
class SlippageModel:
    """Slippage model for realistic fill prices.
    
    Based on institutional models from:
        - Almgren-Chriss market impact model
        - Square root law of market impact
    """
    model_type: SlippageModelType = SlippageModelType.SQUARE_ROOT
    base_bps: Decimal = field(default_factory=lambda: Decimal("5"))  # 5 bps base
    volatility_factor: Decimal = field(default_factory=lambda: Decimal("0.5"))
    max_slippage_bps: Decimal = field(default_factory=lambda: Decimal("100"))  # 1%
    
    def calculate_slippage(
        self,
        order_size: Decimal,
        avg_daily_volume: Decimal,
        volatility: Decimal,  # annualized
        side: str,
    ) -> Decimal:
        """Calculate slippage in price terms.
        
        Args:
            order_size: Order quantity
            adv: Average daily volume
            volatility: Annualized volatility (e.g., 0.25 for 25%)
            side: "buy" or "sell"
        
        Returns:
            Slippage as percentage (e.g., 0.001 for 10 bps)
        """
        if avg_daily_volume == 0:
            return Decimal("0")
        
        # Participation rate
        participation = order_size / avg_daily_volume
        
        if self.model_type == SlippageModelType.FIXED:
            slippage_bps = self.base_bps
        
        elif self.model_type == SlippageModelType.LINEAR:
            # Linear with participation
            slippage_bps = self.base_bps * (Decimal("1") + participation * Decimal("10"))
        
        elif self.model_type == SlippageModelType.SQUARE_ROOT:
            # Square root model (Almgren-Chriss inspired)
            # Impact ~ sqrt(participation)
            import math
            sqrt_participation = Decimal(str(math.sqrt(float(participation))))
            slippage_bps = self.base_bps * (Decimal("1") + sqrt_participation * Decimal("5"))
        
        elif self.model_type == SlippageModelType.VOLATILITY_ADJUSTED:
            # Adjust for volatility
            vol_adj = Decimal("1") + volatility * self.volatility_factor
            slippage_bps = self.base_bps * vol_adj * (Decimal("1") + participation * Decimal("5"))
        
        else:
            slippage_bps = self.base_bps
        
        # Cap at max
        slippage_bps = min(slippage_bps, self.max_slippage_bps)
        
        # Convert bps to percentage
        return slippage_bps / Decimal("10000")
    
    def apply_to_price(
        self,
        base_price: Decimal,
        slippage_pct: Decimal,
        side: str,
    ) -> Decimal:
        """Apply slippage to base price.
        
        Buy orders: pay more (slippage added)
        Sell orders: receive less (slippage subtracted)
        """
        if side == "buy":
            return base_price * (Decimal("1") + slippage_pct)
        else:
            return base_price * (Decimal("1") - slippage_pct)


@dataclass
class FillSimulation:
    """Result of fill simulation."""
    filled_quantity: Decimal
    filled_price: Decimal
    commission: Decimal
    slippage: Decimal
    fill_time: datetime
    partial_fill: bool
    rejection_reason: Optional[str] = None


@dataclass
class MarketCondition:
    """Current market conditions for execution."""
    symbol: str
    quote: Optional[Quote] = None
    volatility: Decimal = field(default_factory=lambda: Decimal("0.20"))
    spread_bps: Decimal = field(default_factory=lambda: Decimal("1"))
    adv_30d: Decimal = field(default_factory=lambda: Decimal("1000000"))
    liquidity_score: Decimal = field(default_factory=lambda: Decimal("100"))
    
    # Stress conditions
    circuit_breaker_active: bool = False
    halted: bool = False
    wide_spreads: bool = False


class ExecutionSimulator:
    """Realistic execution simulation for backtesting.
    
    NOT idealized. Models real-world execution challenges:
        - Slippage increases with size
        - Spreads widen under stress
        - Fills can be delayed or partial
        - Liquidity can be exhausted
        - Volatility degrades execution
    
    Usage:
        simulator = ExecutionSimulator(
            slippage_model=SlippageModel(SlippageModelType.SQUARE_ROOT),
            base_latency_ms=50,
        )
        
        fill = await simulator.simulate_fill(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("1000"),
            order_type="market",
            market_condition=market,
        )
    """
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        base_latency_ms: float = 50.0,
        partial_fill_rate: float = 0.1,
        rejection_rate: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        self.slippage_model = slippage_model or SlippageModel()
        self.base_latency_ms = base_latency_ms
        self.partial_fill_rate = partial_fill_rate
        self.rejection_rate = rejection_rate
        
        # Random state for reproducibility
        self._rng = random.Random(seed)
        
        # Statistics
        self._total_orders = 0
        self._total_fills = 0
        self._total_rejections = 0
        self._partial_fills = 0
        self._total_slippage = Decimal("0")
        self._total_commission = Decimal("0")
    
    async def simulate_fill(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        market_condition: MarketCondition,
        limit_price: Optional[Decimal] = None,
        time_in_force: str = "day",
    ) -> FillSimulation:
        """Simulate order execution with realistic conditions.
        
        Returns:
            FillSimulation with results or rejection
        """
        self._total_orders += 1
        
        # Check for rejection conditions
        rejection_reason = self._check_rejection(
            symbol, side, quantity, order_type, market_condition, limit_price
        )
        
        if rejection_reason:
            self._total_rejections += 1
            return FillSimulation(
                filled_quantity=Decimal("0"),
                filled_price=Decimal("0"),
                commission=Decimal("0"),
                slippage=Decimal("0"),
                fill_time=datetime.utcnow(),
                partial_fill=False,
                rejection_reason=rejection_reason,
            )
        
        # Get base price
        if market_condition.quote:
            if side == "buy":
                base_price = market_condition.quote.ask or market_condition.quote.price
            else:
                base_price = market_condition.quote.bid or market_condition.quote.price
        else:
            base_price = Decimal("100")  # Default fallback
        
        # Calculate slippage
        slippage_pct = self.slippage_model.calculate_slippage(
            order_size=quantity,
            avg_daily_volume=market_condition.adv_30d,
            volatility=market_condition.volatility,
            side=side,
        )
        
        # Apply additional spread if wide
        if market_condition.wide_spreads:
            slippage_pct *= Decimal("2")
        
        # Adjust for liquidity
        liquidity_impact = self._calculate_liquidity_impact(quantity, market_condition)
        slippage_pct *= (Decimal("1") + liquidity_impact)
        
        # Calculate filled price
        filled_price = self.slippage_model.apply_to_price(base_price, slippage_pct, side)
        
        # Determine fill quantity
        filled_quantity = self._calculate_fill_quantity(
            quantity, market_condition, order_type, limit_price, filled_price
        )
        
        partial = filled_quantity < quantity
        if partial:
            self._partial_fills += 1
        
        # Calculate commission (institutional tiered pricing)
        commission = self._calculate_commission(filled_quantity, filled_price)
        
        # Record statistics
        self._total_fills += 1
        self._total_slippage += slippage_pct * filled_quantity * filled_price
        self._total_commission += commission
        
        return FillSimulation(
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            commission=commission,
            slippage=slippage_pct,
            fill_time=datetime.utcnow(),
            partial_fill=partial,
        )
    
    def _check_rejection(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        market_condition: MarketCondition,
        limit_price: Optional[Decimal],
    ) -> Optional[str]:
        """Check if order should be rejected."""
        # Market halted
        if market_condition.halted:
            return "market_halted"
        
        # Circuit breaker
        if market_condition.circuit_breaker_active:
            return "circuit_breaker_active"
        
        # Limit price check
        if order_type == "limit" and limit_price:
            if market_condition.quote:
                if side == "buy" and limit_price < market_condition.quote.bid:
                    # Limit too low, won't fill
                    if self._rng.random() < 0.8:  # 80% chance of no fill
                        return "limit_too_low"
        
        # Random rejection (simulates broker issues)
        if self._rng.random() < self.rejection_rate:
            return "broker_rejection"
        
        return None
    
    def _calculate_liquidity_impact(
        self,
        quantity: Decimal,
        market_condition: MarketCondition,
    ) -> Decimal:
        """Calculate liquidity impact on slippage."""
        if market_condition.liquidity_score <= 0:
            return Decimal("1")  # Double slippage if no liquidity
        
        # Higher impact if participation is high relative to liquidity
        participation = quantity / market_condition.adv_30d
        if participation > Decimal("0.01"):  # >1% of ADV
            return Decimal("0.5")  # 50% more slippage
        
        return Decimal("0")
    
    def _calculate_fill_quantity(
        self,
        quantity: Decimal,
        market_condition: MarketCondition,
        order_type: str,
        limit_price: Optional[Decimal],
        market_price: Decimal,
    ) -> Decimal:
        """Determine how much of order gets filled."""
        # Check for partial fill
        if self._rng.random() < self.partial_fill_rate:
            # Partial fill - random percentage 20-80%
            fill_pct = Decimal(str(self._rng.uniform(0.2, 0.8)))
            return (quantity * fill_pct).quantize(Decimal("1"))
        
        # Limit order may not fill if limit crossed
        if order_type == "limit" and limit_price:
            # Check if limit prevents fill
            if market_condition.liquidity_score < 50:
                # Low liquidity, may not fill completely
                return (quantity * Decimal("0.7")).quantize(Decimal("1"))
        
        return quantity
    
    def _calculate_commission(
        self,
        quantity: Decimal,
        price: Decimal,
    ) -> Decimal:
        """Calculate commission based on tiered pricing."""
        notional = quantity * price
        
        # Tiered rates (institutional)
        if notional < Decimal("10000"):
            rate = Decimal("0.005")  # 50 bps for small
        elif notional < Decimal("100000"):
            rate = Decimal("0.003")  # 30 bps
        else:
            rate = Decimal("0.001")  # 10 bps for large
        
        min_commission = Decimal("1.00")
        commission = max(notional * rate, min_commission)
        
        return commission
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution simulation statistics."""
        if self._total_fills == 0:
            return {
                "total_orders": self._total_orders,
                "total_fills": 0,
                "fill_rate": 0.0,
            }
        
        fill_rate = self._total_fills / self._total_orders
        avg_slippage = (
            self._total_slippage / self._total_fills if self._total_fills > 0 else 0
        )
        
        return {
            "total_orders": self._total_orders,
            "total_fills": self._total_fills,
            "total_rejections": self._total_rejections,
            "partial_fills": self._partial_fills,
            "fill_rate": round(fill_rate, 3),
            "rejection_rate": round(self._total_rejections / self._total_orders, 3),
            "partial_fill_rate": round(self._partial_fills / self._total_fills, 3),
            "avg_slippage_bps": round(float(avg_slippage) * 10000, 2),
            "total_commission": str(self._total_commission),
        }
    
    def reset_statistics(self) -> None:
        """Reset simulation statistics."""
        self._total_orders = 0
        self._total_fills = 0
        self._total_rejections = 0
        self._partial_fills = 0
        self._total_slippage = Decimal("0")
        self._total_commission = Decimal("0")
