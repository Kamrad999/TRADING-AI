"""
Technical indicators for market data analysis.
Following patterns from ai-trade and AgentQuant repositories.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import math

from .models import MarketData
from ..infrastructure.logging import get_logger


class TechnicalIndicators:
    """
    Technical indicators for market analysis.
    
    Following patterns from:
    - ai-trade: technicalindicators npm package
    - AgentQuant: Feature Engine with indicators
    - VectorBT: Vectorized indicator calculations
    """
    
    def __init__(self):
        """Initialize technical indicators."""
        self.logger = get_logger("technical_indicators")
        
        # Default periods
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.sma_short = 20
        self.sma_long = 50
        self.ema_short = 12
        self.ema_long = 26
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        self.atr_period = 14
        
        self.logger.info("Technical indicators initialized")
    
    def calculate_all(self, ohlc_data: List[MarketData]) -> Dict[str, float]:
        """
        Calculate all technical indicators.
        
        Args:
            ohlc_data: List of OHLC data
            
        Returns:
            Dictionary of all indicators
        """
        try:
            if len(ohlc_data) < self.rsi_period:
                return {}
            
            # Extract price data
            closes = [d.close_price for d in ohlc_data]
            highs = [d.high_price for d in ohlc_data]
            lows = [d.low_price for d in ohlc_data]
            volumes = [d.volume for d in ohlc_data]
            
            # Calculate indicators
            indicators = {}
            
            # RSI
            indicators["rsi"] = self.calculate_rsi(closes)
            
            # MACD
            macd_data = self.calculate_macd(closes)
            indicators.update(macd_data)
            
            # Moving Averages
            indicators["sma_20"] = self.calculate_sma(closes, self.sma_short)
            indicators["sma_50"] = self.calculate_sma(closes, self.sma_long)
            indicators["ema_12"] = self.calculate_ema(closes, self.ema_short)
            indicators["ema_26"] = self.calculate_ema(closes, self.ema_long)
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(closes)
            indicators.update(bb_data)
            
            # ATR
            indicators["atr"] = self.calculate_atr(highs, lows, closes)
            
            # Volume indicators
            indicators["volume_sma"] = self.calculate_sma(volumes, 20)
            
            # Current price for reference
            indicators["current_price"] = closes[-1] if closes else 0.0
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def calculate_rsi(self, closes: List[float], period: int = None) -> Optional[float]:
        """
        Calculate Relative Strength Index.
        
        Args:
            closes: List of closing prices
            period: RSI period
            
        Returns:
            RSI value
        """
        try:
            if period is None:
                period = self.rsi_period
            
            if len(closes) < period + 1:
                return None
            
            # Calculate price changes
            gains = []
            losses = []
            
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0.0)
                else:
                    gains.append(0.0)
                    losses.append(abs(change))
            
            # Calculate average gains and losses
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            # Calculate RSI
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None
    
    def calculate_macd(self, closes: List[float]) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            closes: List of closing prices
            
        Returns:
            MACD indicators
        """
        try:
            if len(closes) < self.macd_slow:
                return {}
            
            # Calculate EMAs
            ema_fast = self.calculate_ema(closes, self.macd_fast)
            ema_slow = self.calculate_ema(closes, self.macd_slow)
            
            if ema_fast is None or ema_slow is None:
                return {}
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line (EMA of MACD)
            macd_history = []
            for i in range(len(closes)):
                if i >= self.macd_slow - 1:
                    fast_ema = self.calculate_ema(closes[:i+1], self.macd_fast)
                    slow_ema = self.calculate_ema(closes[:i+1], self.macd_slow)
                    if fast_ema and slow_ema:
                        macd_history.append(fast_ema - slow_ema)
            
            signal_line = None
            if len(macd_history) >= self.macd_signal:
                signal_line = self.calculate_ema(macd_history, self.macd_signal)
            
            # Histogram
            histogram = None
            if signal_line is not None:
                histogram = macd_line - signal_line
            
            return {
                "macd": macd_line,
                "macd_signal": signal_line if signal_line is not None else 0.0,
                "macd_histogram": histogram if histogram is not None else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return {}
    
    def calculate_sma(self, data: List[float], period: int) -> Optional[float]:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: List of data points
            period: SMA period
            
        Returns:
            SMA value
        """
        try:
            if len(data) < period:
                return None
            
            return sum(data[-period:]) / period
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return None
    
    def calculate_ema(self, data: List[float], period: int) -> Optional[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: List of data points
            period: EMA period
            
        Returns:
            EMA value
        """
        try:
            if len(data) < period:
                return None
            
            # Calculate multiplier
            multiplier = 2.0 / (period + 1.0)
            
            # Calculate EMA
            ema = data[0]
            for price in data[1:]:
                ema = (price * multiplier) + (ema * (1.0 - multiplier))
            
            return ema
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return None
    
    def calculate_bollinger_bands(self, closes: List[float]) -> Dict[str, float]:
        """
        Calculate Bollinger Bands.
        
        Args:
            closes: List of closing prices
            
        Returns:
            Bollinger Bands indicators
        """
        try:
            if len(closes) < self.bollinger_period:
                return {}
            
            # Calculate SMA
            sma = self.calculate_sma(closes, self.bollinger_period)
            if sma is None:
                return {}
            
            # Calculate standard deviation
            recent_closes = closes[-self.bollinger_period:]
            mean = sum(recent_closes) / len(recent_closes)
            variance = sum((x - mean) ** 2 for x in recent_closes) / len(recent_closes)
            std_dev = math.sqrt(variance)
            
            # Calculate bands
            upper_band = sma + (self.bollinger_std * std_dev)
            lower_band = sma - (self.bollinger_std * std_dev)
            
            # Calculate bandwidth and position
            bandwidth = (upper_band - lower_band) / sma
            position = (closes[-1] - lower_band) / (upper_band - lower_band)
            
            return {
                "bollinger_upper": upper_band,
                "bollinger_middle": sma,
                "bollinger_lower": lower_band,
                "bollinger_bandwidth": bandwidth,
                "bollinger_position": position
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = None) -> Optional[float]:
        """
        Calculate Average True Range.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            period: ATR period
            
        Returns:
            ATR value
        """
        try:
            if period is None:
                period = self.atr_period
            
            if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
                return None
            
            # Calculate True Range
            true_ranges = []
            
            for i in range(1, len(closes)):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i-1])
                low_close = abs(lows[i] - closes[i-1])
                
                true_range = max(high_low, high_close, low_close)
                true_ranges.append(true_range)
            
            # Calculate ATR
            if len(true_ranges) >= period:
                atr = sum(true_ranges[-period:]) / period
                return atr
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return None
    
    def calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            Stochastic indicators
        """
        try:
            if len(highs) < k_period or len(lows) < k_period or len(closes) < k_period:
                return {}
            
            # Calculate %K
            k_values = []
            for i in range(k_period - 1, len(closes)):
                highest_high = max(highs[i - k_period + 1:i + 1])
                lowest_low = min(lows[i - k_period + 1:i + 1])
                
                if highest_high - lowest_low == 0:
                    k_percent = 50.0
                else:
                    k_percent = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100.0
                
                k_values.append(k_percent)
            
            # Calculate %D (SMA of %K)
            d_percent = None
            if len(k_values) >= d_period:
                d_percent = sum(k_values[-d_period:]) / d_period
            
            return {
                "stochastic_k": k_values[-1] if k_values else 50.0,
                "stochastic_d": d_percent if d_percent is not None else 50.0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return {}
    
    def calculate_williams_r(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """
        Calculate Williams %R.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            period: Williams %R period
            
        Returns:
            Williams %R value
        """
        try:
            if len(highs) < period or len(lows) < period or len(closes) < period:
                return None
            
            # Get recent data
            recent_highs = highs[-period:]
            recent_lows = lows[-period:]
            current_close = closes[-1]
            
            # Calculate Williams %R
            highest_high = max(recent_highs)
            lowest_low = min(recent_lows)
            
            if highest_high - lowest_low == 0:
                return -50.0
            
            williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100.0
            
            return williams_r
            
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return None
    
    def calculate_cci(self, highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> Optional[float]:
        """
        Calculate Commodity Channel Index.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            period: CCI period
            
        Returns:
            CCI value
        """
        try:
            if len(highs) < period or len(lows) < period or len(closes) < period:
                return None
            
            # Calculate typical price
            typical_prices = []
            for i in range(len(closes)):
                tp = (highs[i] + lows[i] + closes[i]) / 3.0
                typical_prices.append(tp)
            
            # Calculate SMA of typical prices
            sma_tp = self.calculate_sma(typical_prices, period)
            if sma_tp is None:
                return None
            
            # Calculate mean deviation
            recent_tp = typical_prices[-period:]
            mean_deviation = sum(abs(tp - sma_tp) for tp in recent_tp) / period
            
            # Calculate CCI
            if mean_deviation == 0:
                return 0.0
            
            cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)
            
            return cci
            
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return None
    
    def get_signal_strength(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """
        Get signal strength from indicators.
        
        Args:
            indicators: Dictionary of indicators
            
        Returns:
            Signal strength for each indicator
        """
        signals = {}
        
        # RSI signals
        rsi = indicators.get("rsi", 50.0)
        if rsi < 30:
            signals["rsi"] = 1.0  # Strong buy
        elif rsi < 40:
            signals["rsi"] = 0.5  # Buy
        elif rsi > 70:
            signals["rsi"] = -1.0  # Strong sell
        elif rsi > 60:
            signals["rsi"] = -0.5  # Sell
        else:
            signals["rsi"] = 0.0  # Neutral
        
        # MACD signals
        macd = indicators.get("macd", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)
        if macd > macd_signal:
            signals["macd"] = 0.5  # Buy
        else:
            signals["macd"] = -0.5  # Sell
        
        # SMA signals
        current_price = indicators.get("current_price", 0.0)
        sma_20 = indicators.get("sma_20", 0.0)
        sma_50 = indicators.get("sma_50", 0.0)
        
        if current_price > sma_20 > sma_50:
            signals["sma"] = 1.0  # Strong buy
        elif current_price > sma_20 and sma_20 < sma_50:
            signals["sma"] = 0.5  # Buy
        elif current_price < sma_20 < sma_50:
            signals["sma"] = -1.0  # Strong sell
        elif current_price < sma_20 and sma_20 > sma_50:
            signals["sma"] = -0.5  # Sell
        else:
            signals["sma"] = 0.0  # Neutral
        
        return signals
