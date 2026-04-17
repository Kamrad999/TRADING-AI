"""
Technical analysis-based trading strategy following Freqtrade/Jesse patterns.
Analyzes technical indicators and generates trading signals based on price action.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_strategy import BaseStrategy
from .strategy_interface import StrategyContext, StrategyOutput
from ..core.models import Signal, SignalDirection, Urgency, MarketRegime, SignalType
from ..infrastructure.logging import get_logger


class TechnicalStrategy(BaseStrategy):
    """
    Technical analysis strategy based on price indicators.
    
    Following patterns from:
    - Freqtrade: Technical analysis and indicator system
    - Jesse: Position lifecycle and technical signals
    - Backtrader: Strategy class architecture
    """
    
    def __init__(self, **kwargs):
        """Initialize technical strategy."""
        super().__init__("TechnicalStrategy", **kwargs)
        
        # Technical indicator thresholds
        self.rsi_oversold = kwargs.get("rsi_oversold", 30)
        self.rsi_overbought = kwargs.get("rsi_overbought", 70)
        self.rsi_neutral_min = kwargs.get("rsi_neutral_min", 40)
        self.rsi_neutral_max = kwargs.get("rsi_neutral_max", 60)
        
        # MACD settings
        self.macd_bullish_threshold = kwargs.get("macd_bullish_threshold", 0.01)
        self.macd_bearish_threshold = kwargs.get("macd_bearish_threshold", -0.01)
        
        # SMA settings
        self.sma_trend_threshold = kwargs.get("sma_trend_threshold", 0.02)
        self.sma_cross_threshold = kwargs.get("sma_cross_threshold", 0.01)
        
        # Bollinger Bands
        self.bb_width_threshold = kwargs.get("bb_width_threshold", 0.04)
        self.bb_position_threshold = kwargs.get("bb_position_threshold", 0.8)
        
        # ATR for volatility
        self.atr_multiplier_stop = kwargs.get("atr_multiplier_stop", 2.0)
        self.atr_multiplier_target = kwargs.get("atr_multiplier_target", 3.0)
        
        # Signal weights
        self.rsi_weight = kwargs.get("rsi_weight", 0.3)
        self.macd_weight = kwargs.get("macd_weight", 0.3)
        self.sma_weight = kwargs.get("sma_weight", 0.2)
        self.bb_weight = kwargs.get("bb_weight", 0.2)
        
        self.logger.info(f"TechnicalStrategy initialized with RSI thresholds: {self.rsi_oversold}/{self.rsi_overbought}")
    
    def execute(self, context: StrategyContext) -> StrategyOutput:
        """
        Execute technical strategy.
        
        Args:
            context: Current market context
            
        Returns:
            Strategy output with signals
        """
        signals = []
        
        try:
            # Process each symbol
            for symbol in context.symbols:
                # Get technical indicators
                indicators = self._get_symbol_indicators(symbol, context)
                if not indicators:
                    continue
                
                # Analyze technical conditions
                technical_analysis = self._analyze_technicals(indicators)
                
                # Generate signal based on technical analysis
                signal = self._generate_technical_signal(symbol, technical_analysis, context)
                if signal and self.validate_signal(signal, context):
                    signals.append(signal)
            
            self.logger.info(f"TechnicalStrategy generated {len(signals)} signals")
            
            return StrategyOutput(
                signals=signals,
                metadata={
                    "strategy": "TechnicalStrategy",
                    "processed_symbols": len(context.symbols),
                    "indicators_used": ["RSI", "MACD", "SMA", "Bollinger Bands", "ATR"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"TechnicalStrategy execution failed: {e}")
            return StrategyOutput(signals=signals, metadata={"strategy": "TechnicalStrategy", "error": str(e)})
    
    def _get_symbol_indicators(self, symbol: str, context: StrategyContext) -> Optional[Dict[str, float]]:
        """Get technical indicators for a symbol."""
        # Get indicators from market data
        market_data = context.metadata.get("market_data", {})
        symbol_data = market_data.get(symbol, {})
        
        if not symbol_data:
            return None
        
        indicators = symbol_data.get("indicators", {})
        if not indicators:
            return None
        
        # Extract key indicators
        return {
            "rsi": indicators.get("rsi", 50.0),
            "macd": indicators.get("macd", 0.0),
            "macd_signal": indicators.get("macd_signal", 0.0),
            "macd_histogram": indicators.get("macd_histogram", 0.0),
            "sma_20": indicators.get("sma_20", 0.0),
            "sma_50": indicators.get("sma_50", 0.0),
            "ema_12": indicators.get("ema_12", 0.0),
            "ema_26": indicators.get("ema_26", 0.0),
            "bollinger_upper": indicators.get("bollinger_upper", 0.0),
            "bollinger_middle": indicators.get("bollinger_middle", 0.0),
            "bollinger_lower": indicators.get("bollinger_lower", 0.0),
            "atr": indicators.get("atr", 0.0),
            "current_price": symbol_data.get("price", 0.0),
            "volume": symbol_data.get("volume", 0.0)
        }
    
    def _analyze_technicals(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Analyze technical indicators and generate signals."""
        analysis = {
            "rsi_signal": 0.0,
            "macd_signal": 0.0,
            "sma_signal": 0.0,
            "bb_signal": 0.0,
            "overall_signal": 0.0,
            "strength": 0.0,
            "conditions": []
        }
        
        # RSI Analysis
        rsi = indicators["rsi"]
        if rsi < self.rsi_oversold:
            analysis["rsi_signal"] = 1.0
            analysis["conditions"].append(f"RSI oversold ({rsi:.1f})")
        elif rsi > self.rsi_overbought:
            analysis["rsi_signal"] = -1.0
            analysis["conditions"].append(f"RSI overbought ({rsi:.1f})")
        elif self.rsi_neutral_min <= rsi <= self.rsi_neutral_max:
            analysis["rsi_signal"] = 0.0
        else:
            # Weak signals in neutral zone
            if rsi < self.rsi_neutral_min:
                analysis["rsi_signal"] = 0.3
                analysis["conditions"].append(f"RSI mildly oversold ({rsi:.1f})")
            else:
                analysis["rsi_signal"] = -0.3
                analysis["conditions"].append(f"RSI mildly overbought ({rsi:.1f})")
        
        # MACD Analysis
        macd = indicators["macd"]
        macd_signal_line = indicators["macd_signal"]
        macd_histogram = indicators["macd_histogram"]
        
        if macd > macd_signal_line + self.macd_bullish_threshold:
            analysis["macd_signal"] = 1.0
            analysis["conditions"].append(f"MACD bullish ({macd:.4f} vs {macd_signal_line:.4f})")
        elif macd < macd_signal_line + self.macd_bearish_threshold:
            analysis["macd_signal"] = -1.0
            analysis["conditions"].append(f"MACD bearish ({macd:.4f} vs {macd_signal_line:.4f})")
        else:
            analysis["macd_signal"] = 0.0
        
        # MACD histogram momentum
        if macd_histogram > 0:
            analysis["macd_signal"] *= 1.2  # Boost bullish signal
        elif macd_histogram < 0:
            analysis["macd_signal"] *= 1.2  # Boost bearish signal
        
        # SMA Analysis
        current_price = indicators["current_price"]
        sma_20 = indicators["sma_20"]
        sma_50 = indicators["sma_50"]
        
        if sma_20 > 0 and sma_50 > 0:
            # Price vs SMA
            if current_price > sma_20 > sma_50:
                analysis["sma_signal"] = 1.0
                analysis["conditions"].append(f"Price above both SMAs ({current_price:.2f} > {sma_20:.2f} > {sma_50:.2f})")
            elif current_price < sma_20 < sma_50:
                analysis["sma_signal"] = -1.0
                analysis["conditions"].append(f"Price below both SMAs ({current_price:.2f} < {sma_20:.2f} < {sma_50:.2f})")
            else:
                # Mixed signals
                if current_price > sma_20:
                    analysis["sma_signal"] = 0.3
                    analysis["conditions"].append(f"Price above SMA20 but below SMA50")
                elif current_price < sma_20:
                    analysis["sma_signal"] = -0.3
                    analysis["conditions"].append(f"Price below SMA20 but above SMA50")
                else:
                    analysis["sma_signal"] = 0.0
        
        # Bollinger Bands Analysis
        bb_upper = indicators["bollinger_upper"]
        bb_middle = indicators["bollinger_middle"]
        bb_lower = indicators["bollinger_lower"]
        
        if bb_upper > 0 and bb_lower > 0:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            if bb_position < (1 - self.bb_position_threshold):
                analysis["bb_signal"] = 1.0
                analysis["conditions"].append(f"Price near lower Bollinger Band ({bb_position:.2f})")
            elif bb_position > self.bb_position_threshold:
                analysis["bb_signal"] = -1.0
                analysis["conditions"].append(f"Price near upper Bollinger Band ({bb_position:.2f})")
            else:
                analysis["bb_signal"] = 0.0
            
            # Squeeze detection
            if bb_width < self.bb_width_threshold:
                analysis["conditions"].append(f"Bollinger Band squeeze detected ({bb_width:.3f})")
        
        # Calculate weighted overall signal
        analysis["overall_signal"] = (
            analysis["rsi_signal"] * self.rsi_weight +
            analysis["macd_signal"] * self.macd_weight +
            analysis["sma_signal"] * self.sma_weight +
            analysis["bb_signal"] * self.bb_weight
        )
        
        # Calculate signal strength
        signal_components = [
            abs(analysis["rsi_signal"]),
            abs(analysis["macd_signal"]),
            abs(analysis["sma_signal"]),
            abs(analysis["bb_signal"])
        ]
        analysis["strength"] = sum(signal_components) / len(signal_components)
        
        return analysis
    
    def _generate_technical_signal(self, symbol: str, technical_analysis: Dict[str, Any], 
                                  context: StrategyContext) -> Optional[Signal]:
        """Generate trading signal based on technical analysis."""
        overall_signal = technical_analysis["overall_signal"]
        strength = technical_analysis["strength"]
        conditions = technical_analysis["conditions"]
        
        # Determine signal direction
        if overall_signal > 0.3:
            direction = SignalDirection.BUY
            confidence = min(0.9, strength * abs(overall_signal))
            urgency = Urgency.HIGH if overall_signal > 0.7 else Urgency.MEDIUM
            reason = f"Bullish technical signals: {'; '.join(conditions)}"
        elif overall_signal < -0.3:
            direction = SignalDirection.SELL
            confidence = min(0.9, strength * abs(overall_signal))
            urgency = Urgency.HIGH if overall_signal < -0.7 else Urgency.MEDIUM
            reason = f"Bearish technical signals: {'; '.join(conditions)}"
        else:
            # Weak or conflicting signals - generate HOLD signal
            direction = SignalDirection.HOLD
            confidence = 0.3
            urgency = Urgency.LOW
            reason = f"Weak/conflicting technical signals: {'; '.join(conditions)}"
        
        # Adjust confidence based on market regime
        if context.market_regime == MarketRegime.VOLATILE:
            confidence *= 0.7  # Reduce confidence in volatile markets
        elif context.market_regime == MarketRegime.RISK_ON and direction == SignalDirection.BUY:
            confidence *= 1.1  # Boost confidence for buys in risk-on markets
        elif context.market_regime == MarketRegime.RISK_OFF and direction == SignalDirection.SELL:
            confidence *= 1.1  # Boost confidence for sells in risk-off markets
        
        # Create signal
        signal = self.create_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            reason=reason,
            urgency=urgency,
            signal_type=SignalType.TECHNICAL,
            technical_score=overall_signal,
            technical_strength=strength,
            technical_conditions=conditions,
            strategy_specific={
                "rsi_signal": technical_analysis["rsi_signal"],
                "macd_signal": technical_analysis["macd_signal"],
                "sma_signal": technical_analysis["sma_signal"],
                "bb_signal": technical_analysis["bb_signal"]
            }
        )
        
        return signal
    
    def get_risk_parameters(self, context: StrategyContext) -> Dict[str, Any]:
        """Get risk parameters specific to technical strategy."""
        base_params = super().get_risk_parameters(context)
        
        # Get ATR for dynamic stops
        indicators = context.metadata.get("market_data", {}).get(context.symbols[0] if context.symbols else "", {}).get("indicators", {})
        atr = indicators.get("atr", 0.02)
        
        if atr > 0:
            # Use ATR-based stops and targets
            base_params["stop_loss"] = (atr * self.atr_multiplier_stop) / context.metadata.get("current_price", 50000)
            base_params["take_profit"] = (atr * self.atr_multiplier_target) / context.metadata.get("current_price", 50000)
        
        return base_params
