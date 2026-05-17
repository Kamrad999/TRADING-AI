"""Backtesting Engine for AMATIS.

Event-driven backtesting with:
    - Historical replay
    - Simulated execution
    - Slippage modeling
    - Performance attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import Event, EventType
from amatix.core.observability import get_logger
from amatix.interfaces import Order, Signal
from amatix.signals.models import SignalDirection

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Decimal("100000")
    
    # Simulation settings
    slippage_bps: float = 5.0  # 5 basis points
    commission_bps: float = 1.0  # 1 basis point
    latency_ms: float = 100.0  # Execution latency
    
    # Risk settings
    enable_risk: bool = True
    max_position_pct: float = 0.20
    
    # Data
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1D"


@dataclass
class BacktestResult:
    """Results from backtest run."""
    config: BacktestConfig
    
    # Performance
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    volatility: float
    var_95: float
    calmar_ratio: float
    
    # Signals
    signals_generated: int
    signals_executed: int
    
    # Series data
    equity_curve: List[tuple]  # (timestamp, value)
    drawdown_series: List[tuple]
    trade_history: List[Dict]


class BacktestEngine:
    """Event-driven backtesting engine.
    
    Replays historical events and simulates trading:
        1. Load historical market data and signals
        2. Replay events chronologically
        3. Simulate order execution
        4. Track portfolio state
        5. Calculate performance metrics
    
    Example:
        >>> engine = BacktestEngine(event_bus)
        >>> 
        >>> config = BacktestConfig(
        ...     start_date=datetime(2023, 1, 1),
        ...     end_date=datetime(2023, 12, 31),
        ... )
        >>> 
        >>> result = await engine.run(config, historical_events)
    """
    
    def __init__(self, event_bus: HardenedEventBusV2) -> None:
        """Initialize backtest engine.
        
        Args:
            event_bus: Event bus for replay
        """
        self._event_bus = event_bus
        self._config: Optional[BacktestConfig] = None
        
        # Portfolio simulation
        self._cash: Decimal = Decimal("0")
        self._positions: Dict[str, Dict] = {}
        self._equity_curve: List[tuple] = []
        
        # Trade tracking
        self._trades: List[Dict] = []
        self._signals_processed = 0
    
    async def run(
        self,
        config: BacktestConfig,
        events: List[Event],
    ) -> BacktestResult:
        """Run backtest simulation.
        
        Args:
            config: Backtest configuration
            events: Chronological list of historical events
        
        Returns:
            BacktestResult with performance metrics
        """
        logger.info(
            "Starting backtest",
            start=config.start_date,
            end=config.end_date,
            events=len(events),
        )
        
        self._config = config
        self._cash = config.initial_capital
        self._positions = {}
        self._equity_curve = []
        self._trades = []
        self._signals_processed = 0
        
        # Sort events chronologically
        sorted_events = sorted(events, key=lambda e: e.context.timestamp)
        
        # Replay events
        for event in sorted_events:
            await self._process_event(event)
        
        # Calculate results
        result = self._calculate_results()
        
        logger.info(
            "Backtest complete",
            total_return=f"{result.total_return:.2%}",
            sharpe=result.sharpe_ratio,
            trades=result.total_trades,
        )
        
        return result
    
    async def _process_event(self, event: Event) -> None:
        """Process a single historical event."""
        if event.event_type == EventType.MARKET_DATA_RECEIVED:
            # Update price data
            symbol = event.payload.get("symbol")
            price = event.payload.get("price")
            if symbol and price:
                await self._update_price(symbol, Decimal(str(price)))
        
        elif event.event_type == EventType.SIGNAL_GENERATED:
            # Process signal
            await self._process_signal(event)
        
        elif event.event_type == EventType.ORDER_FILLED:
            # Update positions
            await self._process_fill(event)
        
        # Record equity
        equity = self._calculate_equity()
        self._equity_curve.append((event.context.timestamp, equity))
    
    async def _process_signal(self, event: Event) -> None:
        """Process trading signal."""
        self._signals_processed += 1
        
        # In backtest, we assume signals become orders immediately
        # (In production, risk engine would filter)
        symbol = event.payload.get("symbol")
        direction = event.payload.get("direction")
        confidence = event.payload.get("confidence", 0.5)
        
        # Skip low confidence
        if confidence < 0.6:
            return
        
        # Calculate position size (simplified)
        position_value = self._cash * Decimal("0.10")  # 10% per position
        
        # Get current price (would look up from market data)
        price = Decimal("100")  # Placeholder
        
        quantity = position_value / price
        
        # Simulate order
        await self._simulate_order(symbol, direction, quantity, price)
    
    async def _simulate_order(
        self,
        symbol: str,
        direction: str,
        quantity: Decimal,
        price: Decimal,
    ) -> None:
        """Simulate order execution with slippage."""
        if self._config is None:
            return
        
        # Apply slippage
        slippage = Decimal(str(self._config.slippage_bps / 10000))
        
        if direction == "long":
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)
        
        # Apply commission
        commission = fill_price * quantity * Decimal(str(self._config.commission_bps / 10000))
        
        # Record trade
        trade = {
            "symbol": symbol,
            "direction": direction,
            "quantity": float(quantity),
            "price": float(fill_price),
            "commission": float(commission),
            "timestamp": datetime.utcnow(),
        }
        self._trades.append(trade)
        
        # Update cash
        cost = fill_price * quantity + commission
        if direction == "long":
            self._cash -= cost
            # Add to positions
            if symbol not in self._positions:
                self._positions[symbol] = {"qty": 0, "cost": 0}
            self._positions[symbol]["qty"] += float(quantity)
            self._positions[symbol]["cost"] += float(cost)
        else:
            # Short - credit cash
            self._cash += fill_price * quantity - commission
    
    async def _update_price(self, symbol: str, price: Decimal) -> None:
        """Update position mark-to-market."""
        if symbol in self._positions:
            self._positions[symbol]["current_price"] = float(price)
    
    async def _process_fill(self, event: Event) -> None:
        """Process order fill (already handled in simulation)."""
        pass  # Fills are simulated inline
    
    def _calculate_equity(self) -> Decimal:
        """Calculate total equity."""
        equity = self._cash
        
        for symbol, pos in self._positions.items():
            qty = pos.get("qty", 0)
            price = pos.get("current_price", pos.get("cost", 0))
            equity += Decimal(str(qty)) * Decimal(str(price))
        
        return equity
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest performance metrics."""
        if self._config is None or not self._equity_curve:
            raise ValueError("Backtest not run")
        
        # Extract equity series
        timestamps = [t for t, _ in self._equity_curve]
        values = [v for _, v in self._equity_curve]
        
        initial = self._config.initial_capital
        final = values[-1] if values else initial
        
        # Returns
        total_return = float((final - initial) / initial)
        
        # Annualized (simplified - assumes 1 year)
        annualized_return = total_return
        
        # Calculate drawdowns
        peak = initial
        max_dd = 0.0
        dd_series = []
        
        for v in values:
            if v > peak:
                peak = v
            dd = float((peak - v) / peak) if peak > 0 else 0
            max_dd = max(max_dd, dd)
            dd_series.append((timestamps[dd_series.index] if len(dd_series) < len(timestamps) else datetime.utcnow(), dd))
        
        # Sharpe (simplified)
        returns = []
        for i in range(1, len(values)):
            r = (values[i] - values[i-1]) / values[i-1]
            returns.append(r)
        
        if returns:
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            sharpe = (avg_return / std_dev) * (252 ** 0.5) if std_dev > 0 else 0
        else:
            sharpe = 0
        
        # Trade stats
        winning = sum(1 for t in self._trades if t.get("pnl", 0) > 0)
        total = len(self._trades)
        
        return BacktestResult(
            config=self._config,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=total,
            winning_trades=winning,
            losing_trades=total - winning,
            win_rate=winning / total if total > 0 else 0,
            avg_profit=0,  # Would calculate from trades
            avg_loss=0,
            profit_factor=0,
            volatility=std_dev * (252 ** 0.5) if 'std_dev' in locals() else 0,
            var_95=0,
            calmar_ratio=annualized_return / max_dd if max_dd > 0 else 0,
            signals_generated=self._signals_processed,
            signals_executed=len(self._trades),
            equity_curve=self._equity_curve,
            drawdown_series=dd_series,
            trade_history=self._trades,
        )
