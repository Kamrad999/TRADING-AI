"""Momentum/Techncial signal engine for AMATIS.

Generates signals based on:
    - EMA crossovers
    - RSI levels
    - Volume spikes
    - Price breakouts

Consumes market data events, emits momentum signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

import whenever

from amatix.core.event_bus_v2 import HardenedEventBusV2
from amatix.core.observability import get_logger, get_metrics
from amatix.data.market.models import OHLCV, Symbol
from amatix.signals.engines.base import BaseSignalEngine
from amatix.signals.models import (
    Signal,
    SignalDirection,
    SignalFeature,
    SignalStrength,
    SignalTimeframe,
)

logger = get_logger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    volume_ma_period: int = 20
    volume_spike_threshold: float = 2.0  # Multiple of average


class MomentumEngine(BaseSignalEngine):
    """Technical momentum signal engine.
    
    Generates LONG/SHORT signals based on:
        - EMA crossovers (bullish/bearish)
        - RSI overbought/oversold
        - Volume confirmation
    
    Example:
        >>> engine = MomentumEngine(event_bus)
        >>> await engine.initialize(config)
        >>> 
        >>> context = {"bars": recent_bars, "symbol": symbol}
        >>> signals = await engine.generate(context)
    """
    
    def __init__(
        self,
        event_bus: HardenedEventBusV2,
        indicator_config: Optional[IndicatorConfig] = None,
    ) -> None:
        """Initialize momentum engine.
        
        Args:
            event_bus: Event bus
            indicator_config: Indicator parameters
        """
        super().__init__(
            name="momentum",
            version="1.0.0",
            event_bus=event_bus,
        )
        self._indicator_config = indicator_config or IndicatorConfig()
        
        # Historical data cache for indicator calculation
        self._bar_history: Dict[str, List[OHLCV]] = {}
        self._max_history = 100
        
        # Last signal tracking (prevent signal spam)
        self._last_signal_time: Dict[str, float] = {}
        self._min_signal_interval = 300  # 5 minutes
    
    async def generate(self, context: Dict[str, Any]) -> List[Signal]:
        """Generate momentum signals from market data.
        
        Args:
            context: Dictionary with:
                - "bars": List[OHLCV] recent price bars
                - "symbol": Symbol being analyzed
        
        Returns:
            List of momentum signals (may be empty)
        """
        bars = context.get("bars", [])
        symbol = context.get("symbol")
        
        if not bars or not symbol:
            return []
        
        # Need minimum bars for indicators
        min_bars = max(
            self._indicator_config.ema_slow,
            self._indicator_config.rsi_period,
        ) + 5
        
        if len(bars) < min_bars:
            logger.debug(
                "Insufficient bars for momentum calculation",
                symbol=str(symbol),
                have=len(bars),
                need=min_bars,
            )
            return []
        
        signals: List[Signal] = []
        
        # Calculate indicators
        prices = [float(bar.close) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]
        
        # EMA Crossover
        ema_signal = self._check_ema_crossover(
            prices,
            symbol,
            bars[-1].close,
        )
        if ema_signal:
            signals.append(ema_signal)
        
        # RSI Signal
        rsi_signal = self._check_rsi(
            prices,
            symbol,
            bars[-1].close,
        )
        if rsi_signal:
            signals.append(rsi_signal)
        
        # Volume Spike
        volume_signal = self._check_volume_spike(
            volumes,
            prices,
            symbol,
        )
        if volume_signal:
            signals.append(volume_signal)
        
        # Filter by cooldown
        filtered = self._apply_cooldown(signals, str(symbol))
        
        for signal in filtered:
            self._track_signal()
        
        return filtered
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return []
        
        multiplier = 2.0 / (period + 1)
        ema = [sum(prices[:period]) / period]  # Start with SMA
        
        for price in prices[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate average gains/losses
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _check_ema_crossover(
        self,
        prices: List[float],
        symbol: Symbol,
        current_price: Decimal,
    ) -> Optional[Signal]:
        """Check for EMA crossover signal."""
        config = self._indicator_config
        
        ema_fast = self._calculate_ema(prices, config.ema_fast)
        ema_slow = self._calculate_ema(prices, config.ema_slow)
        
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return None
        
        # Check for crossover
        prev_fast = ema_fast[-2]
        prev_slow = ema_slow[-2]
        curr_fast = ema_fast[-1]
        curr_slow = ema_slow[-1]
        
        # Bullish crossover: fast crosses above slow
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return self._create_signal(
                symbol=symbol,
                direction=SignalDirection.LONG,
                confidence=0.70,
                strength=SignalStrength.MODERATE,
                features=[
                    SignalFeature(
                        name="ema_crossover",
                        value=f"{config.ema_fast}/{config.ema_slow}",
                        weight=0.8,
                        category="technical",
                        description=f"EMA{config.ema_fast} crossed above EMA{config.ema_slow}",
                    ),
                    SignalFeature(
                        name="ema_fast",
                        value=round(curr_fast, 2),
                        weight=0.3,
                        category="technical",
                    ),
                    SignalFeature(
                        name="ema_slow",
                        value=round(curr_slow, 2),
                        weight=0.3,
                        category="technical",
                    ),
                ],
                rationale=f"Bullish EMA crossover: {config.ema_fast}-period EMA crossed above {config.ema_slow}-period EMA",
            )
        
        # Bearish crossover: fast crosses below slow
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return self._create_signal(
                symbol=symbol,
                direction=SignalDirection.SHORT,
                confidence=0.65,
                strength=SignalStrength.MODERATE,
                features=[
                    SignalFeature(
                        name="ema_crossover",
                        value=f"{config.ema_fast}/{config.ema_slow}",
                        weight=0.8,
                        category="technical",
                        description=f"EMA{config.ema_fast} crossed below EMA{config.ema_slow}",
                    ),
                ],
                rationale=f"Bearish EMA crossover: {config.ema_fast}-period EMA crossed below {config.ema_slow}-period EMA",
            )
        
        return None
    
    def _check_rsi(
        self,
        prices: List[float],
        symbol: Symbol,
        current_price: Decimal,
    ) -> Optional[Signal]:
        """Check for RSI signal."""
        config = self._indicator_config
        
        rsi = self._calculate_rsi(prices, config.rsi_period)
        
        # Oversold - potential buy signal
        if rsi < config.rsi_oversold:
            return self._create_signal(
                symbol=symbol,
                direction=SignalDirection.LONG,
                confidence=0.60,
                strength=SignalStrength.WEAK,
                features=[
                    SignalFeature(
                        name="rsi",
                        value=round(rsi, 2),
                        weight=0.7,
                        category="technical",
                        description=f"RSI at {round(rsi, 1)} (oversold)",
                    ),
                ],
                rationale=f"RSI oversold at {round(rsi, 1)} (< {config.rsi_oversold})",
            )
        
        # Overbought - potential sell signal
        if rsi > config.rsi_overbought:
            return self._create_signal(
                symbol=symbol,
                direction=SignalDirection.SHORT,
                confidence=0.55,
                strength=SignalStrength.WEAK,
                features=[
                    SignalFeature(
                        name="rsi",
                        value=round(rsi, 2),
                        weight=0.7,
                        category="technical",
                        description=f"RSI at {round(rsi, 1)} (overbought)",
                    ),
                ],
                rationale=f"RSI overbought at {round(rsi, 1)} (> {config.rsi_overbought})",
            )
        
        return None
    
    def _check_volume_spike(
        self,
        volumes: List[float],
        prices: List[float],
        symbol: Symbol,
    ) -> Optional[Signal]:
        """Check for volume spike with price movement."""
        config = self._indicator_config
        
        if len(volumes) < config.volume_ma_period + 1:
            return None
        
        # Calculate volume MA
        recent_volumes = volumes[-config.volume_ma_period:]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = volumes[-1]
        
        # Check for spike
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio < config.volume_spike_threshold:
            return None
        
        # Determine direction from price
        price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
        
        if price_change > 0.005:  # 0.5% up move
            return self._create_signal(
                symbol=symbol,
                direction=SignalDirection.LONG,
                confidence=0.55,
                strength=SignalStrength.WEAK,
                features=[
                    SignalFeature(
                        name="volume_spike",
                        value=round(volume_ratio, 2),
                        weight=0.6,
                        category="technical",
                        description=f"Volume {round(volume_ratio, 1)}x average",
                    ),
                    SignalFeature(
                        name="price_change",
                        value=round(price_change * 100, 2),
                        weight=0.4,
                        category="technical",
                        description=f"Price up {round(price_change * 100, 2)}%",
                    ),
                ],
                rationale=f"Volume spike ({round(volume_ratio, 1)}x avg) with bullish price action",
            )
        elif price_change < -0.005:  # 0.5% down move
            return self._create_signal(
                symbol=symbol,
                direction=SignalDirection.SHORT,
                confidence=0.50,
                strength=SignalStrength.WEAK,
                features=[
                    SignalFeature(
                        name="volume_spike",
                        value=round(volume_ratio, 2),
                        weight=0.6,
                        category="technical",
                    ),
                    SignalFeature(
                        name="price_change",
                        value=round(price_change * 100, 2),
                        weight=0.4,
                        category="technical",
                    ),
                ],
                rationale=f"Volume spike ({round(volume_ratio, 1)}x avg) with bearish price action",
            )
        
        return None
    
    def _create_signal(
        self,
        symbol: Symbol,
        direction: SignalDirection,
        confidence: float,
        strength: SignalStrength,
        features: List[SignalFeature],
        rationale: str,
    ) -> Signal:
        """Create a signal with standard settings."""
        signal = Signal.create(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            source=self.name,
            trace_id=uuid4(),
            strength=strength,
            timeframe=SignalTimeframe.INTRADAY,
            source_version=self.version,
        )
        
        # Add features and rationale
        signal = signal.with_features(features)
        signal.rationale = rationale
        signal.tags = ["momentum", "technical"]
        
        return signal
    
    def _apply_cooldown(
        self,
        signals: List[Signal],
        symbol_key: str,
    ) -> List[Signal]:
        """Apply cooldown to prevent signal spam."""
        now = whenever.now().py_datetime().timestamp()
        
        last_time = self._last_signal_time.get(symbol_key, 0)
        
        if now - last_time < self._min_signal_interval:
            # Within cooldown, only allow strongest signal
            if signals:
                strongest = max(signals, key=lambda s: s.confidence)
                self._last_signal_time[symbol_key] = now
                return [strongest]
            return []
        
        # Outside cooldown, allow all
        if signals:
            self._last_signal_time[symbol_key] = now
        
        return signals
    
    async def health_check(self) -> Dict[str, Any]:
        """Engine health check."""
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "name": self.name,
            "version": self.version,
            "signals_generated": self._signals_generated,
            "indicator_config": {
                "ema_fast": self._indicator_config.ema_fast,
                "ema_slow": self._indicator_config.ema_slow,
                "rsi_period": self._indicator_config.rsi_period,
            },
        }
