"""Market Regime Scenarios for Replay Validation.

Institutional-grade market condition simulation:
    - Bull markets
    - Bear markets
    - Sideways regimes
    - High volatility
    - Low liquidity
    - News shocks
    - Flash crash conditions
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.event_models import Event, EventContext, EventPriority, EventType


class MarketRegimeType(Enum):
    """Types of market regimes."""
    BULL_TREND = auto()
    BEAR_TREND = auto()
    SIDEWAYS = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    LOW_LIQUIDITY = auto()
    FLASH_CRASH = auto()
    GAP_UP_OPEN = auto()
    GAP_DOWN_OPEN = auto()
    NEWS_SHOCK = auto()


@dataclass
class RegimeParameters:
    """Parameters defining a market regime."""
    drift_daily: Decimal  # Expected daily return
    volatility_annual: Decimal  # Annualized volatility
    trend_strength: Decimal  # 0-1, strength of directional move
    liquidity_factor: Decimal  # 0-1, relative to normal
    gap_probability: Decimal  # Probability of overnight gap
    shock_probability: Decimal  # Probability of intraday shock
    mean_reversion: Decimal  # Mean reversion strength
    
    # Market microstructure
    spread_bps_normal: Decimal = field(default_factory=lambda: Decimal("1"))
    spread_bps_stress: Decimal = field(default_factory=lambda: Decimal("10"))
    volume_factor: Decimal = field(default_factory=lambda: Decimal("1"))  # Multiplier


@dataclass
class MarketRegime:
    """A market regime with complete configuration."""
    name: str
    regime_type: MarketRegimeType
    parameters: RegimeParameters
    description: str
    duration_days: int = 30
    
    # Risk characteristics
    expected_max_drawdown: Decimal = field(default_factory=lambda: Decimal("0.10"))
    expected_sharpe: Decimal = field(default_factory=lambda: Decimal("0.5"))
    
    def get_expected_behavior(self) -> Dict[str, Any]:
        """Get expected system behavior under this regime."""
        return {
            "regime": self.name,
            "expected_volatility": float(self.parameters.volatility_annual),
            "expected_trend": float(self.parameters.drift_daily * 252),  # Annualized
            "liquidity_stress": self.parameters.liquidity_factor < Decimal("0.5"),
            "risk_engine_should_be_active": self.parameters.volatility_annual > Decimal("0.25"),
            "kill_switch_proximity": self.expected_max_drawdown > Decimal("0.15"),
        }


class RegimeGenerator:
    """Generate realistic market data for different regimes.
    
    Creates synthetic OHLCV data that exhibits characteristics of
    specific market regimes for replay validation.
    """
    
    # Predefined regime templates
    REGIMES = {
        MarketRegimeType.BULL_TREND: RegimeParameters(
            drift_daily=Decimal("0.001"),  # ~25% annual
            volatility_annual=Decimal("0.15"),
            trend_strength=Decimal("0.8"),
            liquidity_factor=Decimal("1.0"),
            gap_probability=Decimal("0.05"),
            shock_probability=Decimal("0.02"),
            mean_reversion=Decimal("0.2"),
        ),
        MarketRegimeType.BEAR_TREND: RegimeParameters(
            drift_daily=Decimal("-0.001"),  # ~-25% annual
            volatility_annual=Decimal("0.25"),
            trend_strength=Decimal("0.7"),
            liquidity_factor=Decimal("0.9"),
            gap_probability=Decimal("0.08"),
            shock_probability=Decimal("0.05"),
            mean_reversion=Decimal("0.3"),
        ),
        MarketRegimeType.SIDEWAYS: RegimeParameters(
            drift_daily=Decimal("0.0001"),  # ~2.5% annual
            volatility_annual=Decimal("0.12"),
            trend_strength=Decimal("0.1"),
            liquidity_factor=Decimal("1.0"),
            gap_probability=Decimal("0.02"),
            shock_probability=Decimal("0.01"),
            mean_reversion=Decimal("0.8"),
        ),
        MarketRegimeType.HIGH_VOLATILITY: RegimeParameters(
            drift_daily=Decimal("0.0005"),
            volatility_annual=Decimal("0.50"),
            trend_strength=Decimal("0.4"),
            liquidity_factor=Decimal("0.7"),
            gap_probability=Decimal("0.15"),
            shock_probability=Decimal("0.10"),
            mean_reversion=Decimal("0.3"),
            spread_bps_stress=Decimal("25"),
        ),
        MarketRegimeType.LOW_LIQUIDITY: RegimeParameters(
            drift_daily=Decimal("0.0003"),
            volatility_annual=Decimal("0.20"),
            trend_strength=Decimal("0.3"),
            liquidity_factor=Decimal("0.3"),
            gap_probability=Decimal("0.10"),
            shock_probability=Decimal("0.08"),
            mean_reversion=Decimal("0.4"),
            spread_bps_normal=Decimal("5"),
            spread_bps_stress=Decimal("50"),
        ),
        MarketRegimeType.FLASH_CRASH: RegimeParameters(
            drift_daily=Decimal("-0.005"),
            volatility_annual=Decimal("0.80"),
            trend_strength=Decimal("0.9"),
            liquidity_factor=Decimal("0.1"),
            gap_probability=Decimal("0.30"),
            shock_probability=Decimal("0.30"),
            mean_reversion=Decimal("0.1"),
            spread_bps_stress=Decimal("100"),
        ),
    }
    
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
    
    def generate_regime(
        self,
        regime_type: MarketRegimeType,
        symbols: List[str],
        start_date: datetime,
        days: int = 30,
    ) -> MarketRegime:
        """Generate a complete market regime configuration."""
        params = self.REGIMES.get(regime_type, self.REGIMES[MarketRegimeType.SIDEWAYS])
        
        return MarketRegime(
            name=regime_type.name,
            regime_type=regime_type,
            parameters=params,
            description=self._get_description(regime_type),
            duration_days=days,
            expected_max_drawdown=self._estimate_drawdown(params),
        )
    
    def generate_market_data_events(
        self,
        regime: MarketRegime,
        symbols: List[str],
        start_date: datetime,
        bars_per_day: int = 78,  # 5-min bars
    ) -> List[Event]:
        """Generate synthetic market data events for regime.
        
        Uses geometric Brownian motion with regime-specific parameters.
        """
        events = []
        current_prices = {s: Decimal("100") for s in symbols}  # Start at $100
        
        total_bars = regime.duration_days * bars_per_day
        
        for bar in range(total_bars):
            current_date = start_date + timedelta(
                days=bar // bars_per_day,
                minutes=(bar % bars_per_day) * (390 // bars_per_day),  # Trading hours
            )
            
            for symbol in symbols:
                price = self._generate_bar(
                    current_prices[symbol],
                    regime.parameters,
                    bar,
                    total_bars,
                )
                
                # Create OHLCV
                volatility = float(regime.parameters.volatility_annual) / 16  # Daily
                ohlcv = self._create_ohlcv(price, volatility, symbol, current_date)
                
                event = Event(
                    event_type=EventType.MARKET_DATA_RECEIVED,
                    payload={
                        "symbol": symbol,
                        "timestamp": current_date.isoformat(),
                        "open": str(ohlcv.open),
                        "high": str(ohlcv.high),
                        "low": str(ohlcv.low),
                        "close": str(ohlcv.close),
                        "volume": str(ohlcv.volume),
                        "regime": regime.regime_type.name,
                    },
                    context=EventContext(
                        trace_id=__import__('uuid').uuid4(),
                        source_component="regime_generator",
                        timestamp=current_date,
                    ),
                    priority=EventPriority.HIGH,
                )
                
                events.append(event)
                current_prices[symbol] = ohlcv.close
        
        return events
    
    def _generate_bar(
        self,
        current_price: Decimal,
        params: RegimeParameters,
        bar_index: int,
        total_bars: int,
    ) -> Decimal:
        """Generate next price bar using regime-aware model."""
        import math
        
        # GBM parameters
        mu = float(params.drift_daily) / 78  # Per bar
        sigma = float(params.volatility_annual) / 16 / math.sqrt(78)  # Per bar
        
        # Add trend component
        trend = float(params.trend_strength) * mu
        
        # Random shock
        z = self._rng.gauss(0, 1)
        
        # Mean reversion
        if bar_index > 0 and params.mean_reversion > 0:
            # Pull back toward moving average
            pass  # Simplified
        
        # Calculate return
        dt = 1.0 / 78
        ret = (mu + trend) * dt + sigma * math.sqrt(dt) * z
        
        # Shock events
        if self._rng.random() < float(params.shock_probability):
            ret += self._rng.gauss(0, sigma * 5)  # 5-sigma shock
        
        # Gap events (overnight)
        if bar_index % 78 == 0 and self._rng.random() < float(params.gap_probability):
            ret += self._rng.gauss(0, sigma * 3)  # Gap
        
        new_price = current_price * Decimal(str(math.exp(ret)))
        return new_price.quantize(Decimal("0.01"))
    
    def _create_ohlcv(
        self,
        close: Decimal,
        volatility: float,
        symbol: str,
        timestamp: datetime,
    ) -> Any:
        """Create OHLCV from close price and volatility."""
        from amatix.data.models import OHLCV
        
        # Generate realistic OHLC around close
        range_pct = Decimal(str(volatility * 2))
        
        high = close * (Decimal("1") + range_pct)
        low = close * (Decimal("1") - range_pct)
        open_price = low + (high - low) * Decimal(str(self._rng.random()))
        
        return OHLCV(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price.quantize(Decimal("0.01")),
            high=high.quantize(Decimal("0.01")),
            low=low.quantize(Decimal("0.01")),
            close=close,
            volume=Decimal(str(int(self._rng.gauss(1000000, 300000)))),
        )
    
    def _get_description(self, regime_type: MarketRegimeType) -> str:
        """Get human-readable description."""
        descriptions = {
            MarketRegimeType.BULL_TREND: "Sustained upward trend with normal volatility",
            MarketRegimeType.BEAR_TREND: "Sustained downward trend with elevated volatility",
            MarketRegimeType.SIDEWAYS: "Range-bound with mean reversion",
            MarketRegimeType.HIGH_VOLATILITY: "Elevated volatility without clear trend",
            MarketRegimeType.LOW_LIQUIDITY: "Thin markets with wide spreads",
            MarketRegimeType.FLASH_CRASH: "Extreme stress with liquidity evaporation",
        }
        return descriptions.get(regime_type, "Unknown regime")
    
    def _estimate_drawdown(self, params: RegimeParameters) -> Decimal:
        """Estimate expected max drawdown for regime."""
        vol = params.volatility_annual
        trend = abs(params.drift_daily * 252)
        
        # Simplified drawdown estimate
        if params.regime_type == MarketRegimeType.BULL_TREND:
            return Decimal("0.10")
        elif params.regime_type == MarketRegimeType.BEAR_TREND:
            return vol * Decimal("0.8")
        elif params.regime_type == MarketRegimeType.FLASH_CRASH:
            return Decimal("0.30")
        else:
            return vol * Decimal("0.5")


class ScenarioBuilder:
    """Build complex multi-regime scenarios."""
    
    def __init__(self) -> None:
        self._phases: List[Tuple[MarketRegimeType, int]] = []
    
    def add_phase(
        self,
        regime_type: MarketRegimeType,
        days: int,
    ) -> "ScenarioBuilder":
        """Add a regime phase."""
        self._phases.append((regime_type, days))
        return self
    
    def build_scenario(
        self,
        name: str,
        symbols: List[str],
        start_date: datetime,
        seed: Optional[int] = None,
    ) -> Tuple[str, List[Event]]:
        """Build complete scenario with regime transitions."""
        generator = RegimeGenerator(seed)
        all_events = []
        
        current_date = start_date
        
        for regime_type, days in self._phases:
            regime = generator.generate_regime(
                regime_type, symbols, current_date, days
            )
            
            events = generator.generate_market_data_events(
                regime, symbols, current_date, days
            )
            
            all_events.extend(events)
            current_date += timedelta(days=days)
        
        return name, all_events
    
    @staticmethod
    def create_stress_test() -> "ScenarioBuilder":
        """Create standard stress test scenario."""
        return (
            ScenarioBuilder()
            .add_phase(MarketRegimeType.BULL_TREND, 10)
            .add_phase(MarketRegimeType.HIGH_VOLATILITY, 5)
            .add_phase(MarketRegimeType.FLASH_CRASH, 3)
            .add_phase(MarketRegimeType.LOW_LIQUIDITY, 7)
            .add_phase(MarketRegimeType.BEAR_TREND, 10)
        )
    
    @staticmethod
    def create_chop_test() -> "ScenarioBuilder":
        """Create chop/range-bound test."""
        return (
            ScenarioBuilder()
            .add_phase(MarketRegimeType.SIDEWAYS, 15)
            .add_phase(MarketRegimeType.HIGH_VOLATILITY, 5)
            .add_phase(MarketRegimeType.SIDEWAYS, 15)
        )
