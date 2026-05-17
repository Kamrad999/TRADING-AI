"""Core interfaces for AMATIS components.

This module defines the abstract base classes (ABCs) that establish
contracts between all major system components. These interfaces enable:
    - Component swapping without system changes
    - Clean testing with mocks
    - Clear dependency boundaries
    - Future distributed architecture preparation

Design principles:
    - Explicit is better than implicit
    - ABCs prevent tight coupling
    - Async-first for I/O operations
    - Type-safe with generics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
)

# =============================================================================
# Domain Primitives (Re-exported from canonical locations)
# =============================================================================

# Re-export all domain models from their canonical locations
# This maintains backward compatibility while consolidating definitions
from amatix.data.market.models import (
    DataSource,
    OHLCV,
    OrderBookLevel,
    OrderBookSnapshot,
    Quote,
    Symbol,
    Tick,
    Trade,
    TradeSide,
)
from amatix.signals.models import (
    Signal,
    SignalDirection,
    SignalFeature,
    SignalStrength,
    SignalTimeframe,
)

# Define interface-specific enums (these don't conflict)
class OrderSide(Enum):
    """Order direction."""
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    """Order types supported by execution engines."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()


class OrderStatus(Enum):
    """Lifecycle states of an order."""
    PENDING = auto()
    SUBMITTED = auto()
    ACCEPTED = auto()
    PARTIAL_FILL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class PositionSide(Enum):
    """Position direction."""
    LONG = auto()
    SHORT = auto()
    FLAT = auto()


# Order, Execution, Position are defined here (they are execution-layer, not data-layer)
# These don't conflict with data/market/models.py
from dataclasses import dataclass, field


@dataclass
class Order:
    """Order request."""
    symbol: Symbol
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    
    # Type-specific fields
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    
    # Identifiers (set by execution engine)
    order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    
    def __post_init__(self):
        # Validation
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT orders require limit_price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP orders require stop_price")


@dataclass
class Execution:
    """Order execution / fill."""
    order_id: str
    symbol: Symbol
    side: OrderSide
    filled_quantity: Decimal
    filled_price: Decimal
    commission: Decimal
    timestamp: datetime
    
    # Remaining quantity after this fill
    remaining_quantity: Decimal


@dataclass
class Position:
    """Current position in an instrument."""
    symbol: Symbol
    side: PositionSide
    quantity: Decimal
    avg_entry_price: Decimal
    
    # Current P&L
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    
    # Timestamps
    opened_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None


# =============================================================================
# Data Provider Interface
# =============================================================================

class DataProvider(ABC):
    """Abstract interface for market data providers.
    
    Implementations connect to exchanges, data vendors, or
    internal databases to provide uniform market data access.
    
    Example implementations:
        - AlpacaDataProvider
        - PolygonDataProvider
        - YahooFinanceProvider
        - DatabaseDataProvider (for backtesting)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and metrics."""
        pass
    
    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to data source.
        
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connection resources."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if provider is currently connected."""
        pass
    
    @abstractmethod
    async def get_price(self, symbol: Symbol) -> Decimal:
        """Get current market price (last trade).
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            Current price
        
        Raises:
            DataUnavailableError: If price not available
        """
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: Symbol) -> Quote:
        """Get current bid/ask quote.
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            Current quote
        """
        pass
    
    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: Symbol,
        timeframe: str,  # "1m", "5m", "1h", "1d"
        limit: int = 100,
    ) -> List[OHLCV]:
        """Get historical OHLCV bars.
        
        Args:
            symbol: Instrument symbol
            timeframe: Bar timeframe
            limit: Number of bars to return
        
        Returns:
            List of OHLCV bars, newest last
        """
        pass
    
    @abstractmethod
    async def subscribe_quotes(
        self,
        symbols: List[Symbol],
        callback: Any,  # Callable[[Quote], Awaitable[None]]
    ) -> None:
        """Subscribe to real-time quote updates.
        
        Args:
            symbols: Symbols to subscribe to
            callback: Async function called on each quote
        """
        pass
    
    @abstractmethod
    async def subscribe_trades(
        self,
        symbols: List[Symbol],
        callback: Any,  # Callable[[OHLCV], Awaitable[None]]
    ) -> None:
        """Subscribe to real-time trade updates.
        
        Args:
            symbols: Symbols to subscribe to
            callback: Async function called on each trade
        """
        pass


# =============================================================================
# Signal Engine Interface
# =============================================================================

class SignalEngine(ABC):
    """Abstract interface for signal generation engines.
    
    Signal engines analyze market data, news, or other inputs
to generate trading signals with confidence scores.
    
    Design principles:
        - Pure functions: No side effects in generate()
        - Composable: Multiple engines can feed one portfolio
        - Observable: All decisions emit events
        - Testable: Deterministic with same inputs
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name/identifier."""
        pass
    
    @property
    @abstractmethod
    def supported_asset_classes(self) -> List[str]:
        """Asset classes this engine supports."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize engine with configuration.
        
        Args:
            config: Engine-specific configuration dict
        """
        pass
    
    @abstractmethod
    async def generate(self, context: Dict[str, Any]) -> List[Signal]:
        """Generate signals from input context.
        
        Args:
            context: Dictionary containing:
                - market_data: Dict[Symbol, OHLCV]
                - news: List[NewsItem] (optional)
                - regime: MarketRegime (optional)
                - portfolio: PortfolioState (optional)
        
        Returns:
            List of generated signals (may be empty)
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.
        
        Returns:
            Dict with keys:
                - status: "healthy" | "degraded" | "unhealthy"
                - latency_ms: Average generation latency
                - last_signal_time: ISO timestamp or null
                - error_count: Recent error count
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass


# =============================================================================
# Risk Engine Interface
# =============================================================================

@dataclass
class RiskAssessment:
    """Result of risk evaluation."""
    approved: bool
    max_position_size: Decimal
    risk_score: float  # 0.0 (safe) to 1.0 (dangerous)
    
    # If not approved
    rejection_reasons: List[str] = None
    
    # Risk decomposition
    portfolio_heat: float = 0.0
    concentration_risk: float = 0.0
    drawdown_proximity: float = 0.0
    
    def __post_init__(self):
        if self.rejection_reasons is None:
            object.__setattr__(self, 'rejection_reasons', [])


class RiskEngine(ABC):
    """Abstract interface for risk management systems.
    
    The risk engine has FINAL AUTHORITY over all trading decisions.
    No trade can execute without risk engine approval.
    
    Responsibilities:
        - Pre-trade risk checks
        - Position limit enforcement
        - Drawdown monitoring
        - Kill switch activation
        - Risk attribution
    
    Design principle: BLOCK EARLY, BLOCK OFTEN
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name/identifier."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        pass
    
    @abstractmethod
    async def assess_order(
        self,
        order: Order,
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> RiskAssessment:
        """Assess if an order can be executed.
        
        This is the CRITICAL PATH for all trading decisions.
        
        Args:
            order: The proposed order
            portfolio_state: Current portfolio (positions, P&L, exposure)
            market_state: Current market conditions (regime, volatility)
        
        Returns:
            RiskAssessment with approval decision
        """
        pass
    
    @abstractmethod
    async def assess_signal(
        self,
        signal: Signal,
        portfolio_state: Dict[str, Any],
    ) -> RiskAssessment:
        """Assess a signal before conversion to order.
        
        Early rejection saves computation and reduces latency.
        
        Args:
            signal: The generated signal
            portfolio_state: Current portfolio state
        
        Returns:
            RiskAssessment (may approve with reduced size)
        """
        pass
    
    @abstractmethod
    async def update_portfolio_state(self, state: Dict[str, Any]) -> None:
        """Update internal portfolio tracking.
        
        Must be called whenever portfolio changes.
        
        Args:
            state: Current portfolio state dict
        """
        pass
    
    @abstractmethod
    async def check_kill_switch(self) -> bool:
        """Check if kill switch should be activated.
        
        Returns:
            True if kill switch is ACTIVE (trading blocked)
        """
        pass
    
    @abstractmethod
    async def get_risk_metrics(self) -> Dict[str, float]:
        """Return current risk metrics.
        
        Returns:
            Dict with keys:
                - portfolio_heat: 0.0 to 1.0
                - max_drawdown_pct: Current DD from peak
                - daily_var: Value at Risk
                - sharpe_ratio: If available
        """
        pass
    
    @abstractmethod
    async def manual_kill(self, reason: str) -> None:
        """Manually activate kill switch.
        
        Args:
            reason: Human-readable reason for kill
        """
        pass
    
    @abstractmethod
    async def manual_reset(self, auth_token: str) -> bool:
        """Manually reset kill switch (requires auth).
        
        Args:
            auth_token: Authentication token for reset
        
        Returns:
            True if reset successful
        """
        pass


# =============================================================================
# Execution Engine Interface
# =============================================================================

class ExecutionEngine(ABC):
    """Abstract interface for order execution.
    
    Execution engines connect to brokers/exchanges to:
        - Submit orders
        - Monitor order status
        - Handle fills
        - Manage positions
    
    Implementations:
        - AlpacaExecution
        - InteractiveBrokersExecution
        - BinanceExecution
        - PaperTradingExecution (simulation)
    
    Design principles:
        - Async-first: All operations are async
        - Circuit breaker protected: Automatic failover
        - Idempotent: Same order_id = same order
        - Observable: All actions emit events
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name/identifier."""
        pass
    
    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Connected broker name."""
        pass
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to broker API.
        
        Raises:
            AuthenticationError: If credentials invalid
            ConnectionError: If cannot connect
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check connection status."""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit order to broker.
        
        Args:
            order: Order to submit
        
        Returns:
            order_id: Internal order ID
        
        Raises:
            OrderRejectedError: If broker rejects
            ConnectionError: If cannot connect
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: Order to cancel
        
        Returns:
            True if cancellation accepted
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status.
        
        Args:
            order_id: Order to check
        
        Returns:
            Current OrderStatus
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all current positions.
        
        Returns:
            List of Position objects
        """
        pass
    
    @abstractmethod
    async def get_position(self, symbol: Symbol) -> Optional[Position]:
        """Get position for specific symbol.
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            Position or None if flat
        """
        pass
    
    @abstractmethod
    async def get_account_value(self) -> Decimal:
        """Get total account value.
        
        Returns:
            Total account value in base currency
        """
        pass
    
    @abstractmethod
    async def get_cash(self) -> Decimal:
        """Get available cash.
        
        Returns:
            Available cash for trading
        """
        pass


# =============================================================================
# Strategy Interface
# =============================================================================

class Strategy(ABC):
    """Abstract base class for trading strategies.
    
    Strategies implement specific trading logic:
        - Momentum strategies
        - Mean reversion
        - News-based strategies
        - ML-driven strategies
    
    Strategies MUST NOT:
        - Execute trades directly (go through risk/engine)
        - Access broker APIs directly
        - Maintain state outside of provided context
    
    Strategies MAY:
        - Generate multiple signals
        - Request specific data
        - Maintain internal indicators
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name/identifier."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Strategy version for tracking."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize strategy with configuration."""
        pass
    
    @abstractmethod
    async def on_market_data(self, ohlcv: OHLCV) -> Optional[Signal]:
        """Process new market data bar.
        
        Args:
            ohlcv: New OHLCV bar
        
        Returns:
            Signal or None
        """
        pass
    
    @abstractmethod
    async def on_signal(self, signal: Signal) -> Optional[Signal]:
        """Process signal from another strategy (for stacking).
        
        Args:
            signal: Signal from upstream strategy
        
        Returns:
            Modified signal or None to block
        """
        pass
    
    @abstractmethod
    async def on_fill(self, execution: Execution) -> None:
        """Process execution fill (for position-aware strategies).
        
        Args:
            execution: Fill information
        """
        pass
    
    @abstractmethod
    def get_required_data(self) -> Dict[str, Any]:
        """Declare data requirements.
        
        Returns:
            Dict with:
                - symbols: List[Symbol]
                - timeframes: List[str]
                - lookback_bars: int
                - indicators: List[str] (optional)
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return strategy health status."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass


# =============================================================================
# Portfolio Manager Interface
# =============================================================================

class PortfolioManager(ABC):
    """Abstract interface for portfolio management.
    
    Responsible for:
        - Position sizing
        - Capital allocation
        - Portfolio construction
        - Rebalancing
    
    The portfolio manager works with signals and risk assessments
to determine final position sizes.
    """
    
    @abstractmethod
    async def calculate_position_size(
        self,
        signal: Signal,
        risk_assessment: RiskAssessment,
        account_value: Decimal,
    ) -> Decimal:
        """Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            risk_assessment: Risk engine assessment
            account_value: Total account value
        
        Returns:
            Position size in units (shares, contracts, etc.)
        """
        pass
    
    @abstractmethod
    async def get_target_allocations(self) -> Dict[Symbol, float]:
        """Get target portfolio allocations.
        
        Returns:
            Dict mapping symbols to target weights (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    async def generate_rebalance_orders(
        self,
        current_positions: List[Position],
    ) -> List[Order]:
        """Generate orders to rebalance portfolio.
        
        Args:
            current_positions: Current holdings
        
        Returns:
            List of orders to execute
        """
        pass


# =============================================================================
# Agent Interface (Future Multi-Agent Support)
# =============================================================================

class Agent(ABC):
    """Abstract base for autonomous trading agents.
    
    Agents are higher-level constructs that:
        - Coordinate multiple strategies
        - Adapt to market regimes
        - Learn from performance
        - Make meta-decisions
    
    This is a forward-looking interface for Phase 2+.
    """
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique agent identifier."""
        pass
    
    @abstractmethod
    async def perceive(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment state and form beliefs."""
        pass
    
    @abstractmethod
    async def decide(self, beliefs: Dict[str, Any]) -> List[Signal]:
        """Make decisions based on beliefs."""
        pass
    
    @abstractmethod
    async def act(self, decisions: List[Signal]) -> List[Order]:
        """Convert decisions to orders (via risk/portfolio)."""
        pass
    
    @abstractmethod
    async def learn(
        self,
        decisions: List[Signal],
        outcomes: List[Dict[str, Any]],
    ) -> None:
        """Update agent based on outcomes."""
        pass


# =============================================================================
# Factory Functions
# =============================================================================

def create_symbol(ticker: str, exchange: Optional[str] = None) -> Symbol:
    """Factory function to create symbols."""
    return Symbol(ticker=ticker.upper(), exchange=exchange)


# Type aliases for common return types
TDataProvider = TypeVar("TDataProvider", bound=DataProvider)
TSignalEngine = TypeVar("TSignalEngine", bound=SignalEngine)
TRiskEngine = TypeVar("TRiskEngine", bound=RiskEngine)
TExecutionEngine = TypeVar("TExecutionEngine", bound=ExecutionEngine)
TStrategy = TypeVar("TStrategy", bound=Strategy)
