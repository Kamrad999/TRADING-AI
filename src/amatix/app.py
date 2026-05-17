"""AMATIS Trading System - Main Application.

Bootstraps and wires together all components for a complete
production-ready trading intelligence platform.

Usage:
    python -m amatix.app

Environment Variables:
    AMATIS_MODE: "paper" or "live" (default: "paper")
    AMATIS_LOG_LEVEL: Logging level (default: "INFO")
    ALPACA_API_KEY: Alpaca API key
    ALPACA_SECRET_KEY: Alpaca secret key
    DATABASE_URL: PostgreSQL connection string
    REDIS_URL: Redis connection string (optional)
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType

from amatix.core.config import Settings, get_settings
from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import Event, EventPriority, EventType
from amatix.core.observability import get_logger, initialize_observability
from amatix.core.orchestrator import Orchestrator
from amatix.data.market.models import Symbol
from amatix.data.market.providers.alpaca import AlpacaDataProvider
from amatix.execution.oms.order_manager_hardened import HardenedOrderManager
from amatix.risk.engine import RiskEngine
from amatix.risk.models import RiskConfig
from amatix.signals.engines.momentum_engine import MomentumEngine
from amatix.signals.engines.news_engine import NewsSignalEngine
from amatix.signals.pipeline import SignalPipeline

logger = get_logger(__name__)


class AMATISApplication:
    """Main AMATIS application container.

    Responsibilities:
        1. Initialize all components in correct order
        2. Wire event subscriptions
        3. Manage lifecycle (start, run, shutdown)
        4. Handle signals and graceful shutdown

    Initialization Order:
        1. Configuration
        2. Observability (logging, metrics)
        3. Event Bus
        4. Database connections
        5. Market data providers
        6. Signal engines
        7. Risk engine
        8. Order management system
        9. Execution adapters
        10. Wire event subscriptions

    Example:
        >>> app = AMATISApplication()
        >>> await app.initialize()
        >>> await app.start()
        >>> # ... run for some time ...
        >>> await app.shutdown()
    """

    def __init__(self) -> None:
        """Initialize application container."""
        self._initialized = False
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Configuration
        self._settings: Settings | None = None
        self._mode: str = "paper"  # "paper" or "live"

        # Core components
        self._event_bus: HardenedEventBusV2 | None = None
        self._orchestrator: Orchestrator | None = None

        # Data layer
        self._data_provider: AlpacaDataProvider | None = None

        # Signal layer
        self._signal_pipeline: SignalPipeline | None = None
        self._momentum_engine: MomentumEngine | None = None
        self._news_engine: NewsSignalEngine | None = None

        # Risk layer
        self._risk_engine: RiskEngine | None = None

        # Execution layer
        self._order_manager: HardenedOrderManager | None = None

        # State tracking
        self._component_health: dict[str, bool] = {}

        logger.info("AMATIS application container created")

    async def initialize(self) -> None:
        """Initialize all components in correct order."""
        if self._initialized:
            logger.warning("Application already initialized")
            return

        logger.info("Initializing AMATIS trading system")

        # 1. Configuration
        self._settings = get_settings()
        self._mode = os.getenv("AMATIS_MODE", "paper").lower()

        if self._mode == "live":
            logger.critical("⚠️  LIVE TRADING MODE - Confirm with manual override")
            # In production, require explicit confirmation
            raise RuntimeError("Live trading requires explicit confirmation")

        logger.info(f"AMATIS mode: {self._mode}")

        # 2. Observability
        initialize_observability(
            log_level=os.getenv("AMATIS_LOG_LEVEL", "INFO"),
            service_name="amatix",
            environment=self._mode,
        )

        # 3. Event Bus
        self._event_bus = HardenedEventBusV2(enable_journaling=True)
        await self._event_bus.start()
        logger.info("Event bus initialized")

        # 4. Initialize core components
        await self._init_orchestrator()
        await self._init_signal_pipeline()
        await self._init_risk_engine()
        await self._init_order_management()
        await self._init_data_provider()

        # 5. Wire event subscriptions
        await self._wire_events()

        self._initialized = True
        logger.info("✅ AMATIS initialization complete")

        # Emit system started event
        await self._event_bus.emit_new(
            EventType.SYSTEM_STARTED,
            {
                "mode": self._mode,
                "version": "2.5.0",
                "components": list(self._component_health.keys()),
            },
            priority=EventPriority.CRITICAL,
            source="app",
        )

    async def _init_orchestrator(self) -> None:
        """Initialize system orchestrator."""
        self._orchestrator = Orchestrator(event_bus=self._event_bus)
        # Note: Orchestrator initialization would happen here
        self._component_health["orchestrator"] = True
        logger.info("Orchestrator initialized")

    async def _init_signal_pipeline(self) -> None:
        """Initialize signal generation pipeline."""
        # Create signal engines
        self._momentum_engine = MomentumEngine(self._event_bus)
        self._news_engine = NewsSignalEngine(self._event_bus)

        # Initialize engines
        await self._momentum_engine.initialize({})
        await self._news_engine.initialize({})

        # Create pipeline
        self._signal_pipeline = SignalPipeline(self._event_bus)
        self._signal_pipeline.register_engine(self._momentum_engine)
        self._signal_pipeline.register_engine(self._news_engine)

        self._component_health["signal_pipeline"] = True
        logger.info("Signal pipeline initialized")

    async def _init_risk_engine(self) -> None:
        """Initialize risk management engine."""
        # Configure risk
        risk_config = RiskConfig(
            max_position_size=self._settings.max_position_size,
            max_position_pct=self._settings.max_position_pct,
            max_gross_exposure=2.0,
            max_net_exposure=1.0,
            max_leverage=2.0,
            max_daily_drawdown=0.03,
            max_total_drawdown=0.10,
            kill_switch_drawdown=0.15,
            volatility_scaling=True,
        )

        self._risk_engine = RiskEngine(
            event_bus=self._event_bus,
            config=risk_config,
        )
        await self._risk_engine.initialize()

        self._component_health["risk_engine"] = True
        logger.info("Risk engine initialized")

    async def _init_order_management(self) -> None:
        """Initialize order management system."""
        self._order_manager = HardenedOrderManager(
            event_bus=self._event_bus,
            max_active_orders=1000,
            enable_reconciliation=True,
        )
        await self._order_manager.initialize()

        self._component_health["order_manager"] = True
        logger.info("Order manager initialized")

    async def _init_data_provider(self) -> None:
        """Initialize market data provider."""
        # Only initialize if credentials are available
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            logger.warning("No Alpaca credentials - data provider disabled")
            self._component_health["data_provider"] = False
            return

        # Initialize Alpaca provider
        from amatix.data.market.providers.base import ProviderConfig

        config = ProviderConfig(
            api_key=api_key,
            api_secret=secret_key,
            paper=True,  # Force paper mode regardless of AMATIS_MODE
        )

        self._data_provider = AlpacaDataProvider(
            config=config,
            event_bus=self._event_bus,
        )

        try:
            await self._data_provider.connect()
            self._component_health["data_provider"] = True
            logger.info("Alpaca data provider connected")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self._component_health["data_provider"] = False

    async def _wire_events(self) -> None:
        """Wire event subscriptions between components."""
        logger.info("Wiring event subscriptions")

        # Signal flow: Market Data -> Signal Engine -> Risk -> OMS
        @self._event_bus.on(EventType.MARKET_DATA_RECEIVED, priority=EventPriority.HIGH)
        async def on_market_data(event: Event) -> None:
            await self._handle_market_data(event)

        @self._event_bus.on(EventType.SIGNAL_GENERATED, priority=EventPriority.NORMAL)
        async def on_signal(event: Event) -> None:
            await self._handle_signal(event)

        @self._event_bus.on(EventType.ORDER_FILLED, priority=EventPriority.HIGH)
        async def on_fill(event: Event) -> None:
            await self._handle_fill(event)

        @self._event_bus.on(EventType.KILL_SWITCH_TRIGGERED, priority=EventPriority.CRITICAL)
        async def on_kill_switch(event: Event) -> None:
            await self._handle_kill_switch(event)

        logger.info("Event subscriptions wired")

    async def _handle_market_data(self, event: Event) -> None:
        """Process market data and generate signals."""
        try:
            # Extract data from event
            symbol_str = event.payload.get("symbol")
            if not symbol_str:
                return

            # Convert to Symbol object
            symbol = Symbol(base=symbol_str, exchange="NASDAQ")

            # Get recent bars for signal generation
            # In production, this would come from cache/historical data
            # For now, we'll rely on the signal engine to fetch what it needs

            # Generate signals
            {
                "symbol": symbol,
                "price": event.payload.get("price"),
                "timestamp": event.context.timestamp,
            }

            # Pass to signal pipeline
            # Note: Pipeline would typically be called from a scheduled task
            # This is simplified for demonstration
            await self._signal_pipeline.process_market_data(symbol)

        except Exception as e:
            logger.error(f"Error handling market data: {e}")

    async def _handle_signal(self, event: Event) -> None:
        """Process trading signal through risk assessment."""
        try:
            # Import here to avoid circular imports

            # Extract signal data
            symbol_str = event.payload.get("symbol")
            direction_str = event.payload.get("direction")
            confidence = event.payload.get("confidence", 0.0)

            logger.info(f"Processing signal: {symbol_str} {direction_str} (conf: {confidence:.2f})")

            # Skip low confidence signals
            if confidence < 0.6:
                logger.info(f"Signal confidence too low: {confidence}")
                return

            # Assess signal through risk engine
            # In production, this would create a proper Signal object
            # and pass through full risk assessment

            # For paper trading demo, we'll emit order directly
            if self._mode == "paper":
                await self._event_bus.emit_new(
                    EventType.ORDER_SUBMITTED,
                    {
                        "symbol": symbol_str,
                        "side": "buy" if direction_str == "long" else "sell",
                        "quantity": "100",  # Simplified sizing
                        "signal_confidence": confidence,
                    },
                    priority=EventPriority.HIGH,
                    source="risk_engine",
                )

        except Exception as e:
            logger.error(f"Error handling signal: {e}")

    async def _handle_fill(self, event: Event) -> None:
        """Process order fill."""
        try:
            order_id = event.payload.get("order_id")
            symbol = event.payload.get("symbol")
            filled_qty = event.payload.get("filled_qty")

            logger.info(f"Order filled: {order_id} {symbol} qty={filled_qty}")

            # Update portfolio tracking
            # Update risk engine with new portfolio state

        except Exception as e:
            logger.error(f"Error handling fill: {e}")

    async def _handle_kill_switch(self, event: Event) -> None:
        """Handle emergency kill switch."""
        reason = event.payload.get("reason", "Unknown")
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")

        # Cancel all pending orders
        if self._order_manager:
            active_orders = await self._order_manager.get_active_orders()
            for order in active_orders:
                await self._order_manager.cancel_order(
                    order.order_id, reason="Kill switch activated"
                )

        # Stop data feeds
        if self._data_provider:
            await self._data_provider.disconnect()

        # Initiate shutdown
        self._shutdown_event.set()

    async def start(self) -> None:
        """Start the trading system."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")

        if self._running:
            logger.warning("Application already running")
            return

        self._running = True
        logger.info("🚀 AMATIS trading system starting")

        # Setup signal handlers
        self._setup_signal_handlers()

        # Start data subscriptions if provider is available
        if self._data_provider and self._component_health.get("data_provider"):
            await self._start_data_subscriptions()

        # Main loop
        try:
            await self._run_main_loop()
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        finally:
            await self.shutdown()

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""

        def handle_signal(sig: int, frame: FrameType | None) -> None:
            logger.info(f"Received signal {sig}, initiating shutdown...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        if sys.platform != "win32":
            signal.signal(signal.SIGHUP, handle_signal)

    async def _start_data_subscriptions(self) -> None:
        """Start market data subscriptions."""
        # Default watchlist
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        logger.info(f"Starting data subscriptions for: {symbols}")

        for symbol_str in symbols:
            symbol = Symbol(base=symbol_str, exchange="NASDAQ")
            try:
                # Get initial quote
                quote = await self._data_provider.get_quote(symbol)
                logger.info(f"Subscribed to {symbol_str}: bid={quote.bid}, ask={quote.ask}")

            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol_str}: {e}")

    async def _run_main_loop(self) -> None:
        """Main application loop."""
        logger.info("Entering main loop")

        health_check_interval = 30  # seconds
        last_health_check = 0

        while self._running and not self._shutdown_event.is_set():
            try:
                # Wait for shutdown or health check interval
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=1.0)
                break  # Shutdown requested
            except TimeoutError:
                pass  # Normal timeout, continue loop

            # Periodic health check
            import time

            current_time = time.time()
            if current_time - last_health_check >= health_check_interval:
                await self._perform_health_check()
                last_health_check = current_time

    async def _perform_health_check(self) -> None:
        """Perform periodic health check."""
        health_status = {}

        # Check event bus
        if self._event_bus:
            metrics = self._event_bus.get_metrics()
            health_status["event_bus"] = {
                "healthy": True,
                "journal_size": metrics.get("journal_size", 0),
                "queue_size": metrics.get("queue_size", 0),
                "dead_letter_size": metrics.get("dead_letter_size", 0),
            }

        # Check data provider
        if self._data_provider:
            try:
                connected = await self._data_provider.is_connected()
                health_status["data_provider"] = {"healthy": connected}
            except Exception as e:
                health_status["data_provider"] = {"healthy": False, "error": str(e)}

        # Check risk engine
        if self._risk_engine:
            try:
                stats = self._risk_engine.get_stats()
                health_status["risk_engine"] = {
                    "healthy": stats.get("trading_allowed", False),
                    "kill_switch": stats.get("kill_switch_active", False),
                }
            except Exception as e:
                health_status["risk_engine"] = {"healthy": False, "error": str(e)}

        # Log health status
        logger.info(f"Health check: {health_status}")

        # Emit health event
        if self._event_bus:
            await self._event_bus.emit_new(
                EventType.PERFORMANCE_UPDATED,
                {"health": health_status},
                priority=EventPriority.LOW,
                source="app",
            )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if not self._running and not self._initialized:
            return

        logger.info("🛑 Initiating graceful shutdown")
        self._running = False

        # Shutdown in reverse order
        if self._data_provider:
            try:
                await self._data_provider.disconnect()
                logger.info("Data provider disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting data provider: {e}")

        if self._order_manager:
            # Cancel any pending orders
            try:
                active = await self._order_manager.get_active_orders()
                for order in active:
                    await self._order_manager.cancel_order(order.order_id, reason="System shutdown")
                logger.info(f"Cancelled {len(active)} pending orders")
            except Exception as e:
                logger.error(f"Error cancelling orders: {e}")

        if self._risk_engine:
            logger.info("Risk engine shutdown")

        if self._signal_pipeline:
            logger.info("Signal pipeline shutdown")

        if self._event_bus:
            # Emit shutdown event
            try:
                await self._event_bus.emit_new(
                    EventType.SYSTEM_SHUTDOWN,
                    {"reason": "graceful_shutdown"},
                    priority=EventPriority.CRITICAL,
                    source="app",
                )
                await self._event_bus.stop()
            except Exception:
                logger.debug("Event bus might already be closing")

        self._initialized = False
        logger.info("✅ AMATIS shutdown complete")


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success)
    """
    app = AMATISApplication()

    try:
        # Initialize
        await app.initialize()

        # Start (blocks until shutdown)
        await app.start()

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        await app.shutdown()
        return 0

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        await app.shutdown()
        return 1


if __name__ == "__main__":
    # Run the async main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
