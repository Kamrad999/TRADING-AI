"""
Market data pipeline following VectorBT/Jesse patterns.
Handles price data, technical indicators, and market state.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from ..infrastructure.logging import get_logger
from ..core.models import MarketSession, MarketRegime


@dataclass
class MarketData:
    """Market data for a single symbol."""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None


@dataclass
class TechnicalIndicators:
    """Technical indicators for market analysis."""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = None
    volume_sma: Optional[float] = None


@dataclass
class MarketState:
    """Current market state."""
    session: MarketSession
    regime: MarketRegime
    volatility: float
    trend_strength: float
    market_sentiment: float
    liquidity_score: float


class MarketDataPipeline:
    """
    Market data pipeline following VectorBT/Jesse patterns.
    
    Handles:
    - Price data collection and processing
    - Technical indicator calculation
    - Market state detection
    - Data quality validation
    """
    
    def __init__(self):
        """Initialize market data pipeline."""
        self.logger = get_logger("market_data_pipeline")
        self.data_cache: Dict[str, List[MarketData]] = {}
        self.indicator_cache: Dict[str, TechnicalIndicators] = {}
        self.market_state = MarketState(
            session=MarketSession.CLOSED,
            regime=MarketRegime.SIDEWAYS,
            volatility=0.0,
            trend_strength=0.0,
            market_sentiment=0.0,
            liquidity_score=0.0
        )
        
        # Configuration
        self.min_data_points = 50  # Minimum data points for indicators
        self.max_cache_size = 1000  # Maximum data points to cache
        
        self.logger.info("Market data pipeline initialized")
    
    def add_market_data(self, data: MarketData) -> None:
        """
        Add market data point to pipeline.
        
        Args:
            data: Market data point
        """
        symbol = data.symbol
        
        if symbol not in self.data_cache:
            self.data_cache[symbol] = []
        
        self.data_cache[symbol].append(data)
        
        # Maintain cache size
        if len(self.data_cache[symbol]) > self.max_cache_size:
            self.data_cache[symbol] = self.data_cache[symbol][-self.max_cache_size:]
        
        # Update indicators for this symbol
        self._update_indicators(symbol)
        
        # Update market state
        self._update_market_state()
    
    def get_market_data(self, symbol: str, count: Optional[int] = None) -> List[MarketData]:
        """
        Get market data for symbol.
        
        Args:
            symbol: Trading symbol
            count: Number of data points to return
            
        Returns:
            List of market data points
        """
        data = self.data_cache.get(symbol, [])
        
        if count is not None:
            return data[-count:]
        
        return data
    
    def get_indicators(self, symbol: str) -> TechnicalIndicators:
        """
        Get technical indicators for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Technical indicators
        """
        return self.indicator_cache.get(symbol, TechnicalIndicators())
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest price or None if no data
        """
        data = self.data_cache.get(symbol, [])
        return data[-1].close_price if data else None
    
    def calculate_returns(self, symbol: str, periods: int = 1) -> List[float]:
        """
        Calculate returns for symbol.
        
        Args:
            symbol: Trading symbol
            periods: Number of periods for returns
            
        Returns:
            List of returns
        """
        data = self.get_market_data(symbol)
        if len(data) < periods + 1:
            return []
        
        prices = [d.close_price for d in data]
        returns = []
        
        for i in range(periods, len(prices)):
            ret = (prices[i] - prices[i - periods]) / prices[i - periods]
            returns.append(ret)
        
        return returns
    
    def _update_indicators(self, symbol: str) -> None:
        """Update technical indicators for symbol."""
        data = self.data_cache.get(symbol, [])
        
        if len(data) < self.min_data_points:
            return
        
        # Extract price and volume data
        closes = [d.close_price for d in data]
        highs = [d.high_price for d in data]
        lows = [d.low_price for d in data]
        volumes = [d.volume for d in data]
        
        # Calculate indicators
        indicators = TechnicalIndicators()
        
        # Simple Moving Averages
        indicators.sma_20 = self._calculate_sma(closes, 20)
        indicators.sma_50 = self._calculate_sma(closes, 50)
        
        # Exponential Moving Averages
        indicators.ema_12 = self._calculate_ema(closes, 12)
        indicators.ema_26 = self._calculate_ema(closes, 26)
        
        # RSI
        indicators.rsi = self._calculate_rsi(closes, 14)
        
        # MACD
        ema_12 = self._calculate_ema(closes, 12)
        ema_26 = self._calculate_ema(closes, 26)
        indicators.macd = ema_12 - ema_26
        indicators.macd_signal = self._calculate_ema([indicators.macd] * len(closes), 9)
        
        # Bollinger Bands
        if indicators.sma_20:
            std_20 = self._calculate_std(closes[-20:], 20)
            indicators.bollinger_upper = indicators.sma_20 + 2 * std_20
            indicators.bollinger_lower = indicators.sma_20 - 2 * std_20
        
        # Average True Range
        indicators.atr = self._calculate_atr(highs, lows, closes, 14)
        
        # Volume SMA
        indicators.volume_sma = self._calculate_sma(volumes, 20)
        
        self.indicator_cache[symbol] = indicators
    
    def _calculate_sma(self, data: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return None
        return sum(data[-period:]) / period
    
    def _calculate_ema(self, data: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, closes: List[float], period: int) -> Optional[float]:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_std(self, data: List[float], period: int) -> float:
        """Calculate standard deviation."""
        if len(data) < period:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance ** 0.5
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> Optional[float]:
        """Calculate Average True Range."""
        if len(highs) < period + 1:
            return None
        
        true_ranges = []
        
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        return sum(true_ranges[-period:]) / period
    
    def _update_market_state(self) -> None:
        """Update overall market state."""
        if not self.data_cache:
            return
        
        # Calculate market session based on current time
        current_time = datetime.now()
        self.market_state.session = self._determine_market_session(current_time)
        
        # Calculate market regime based on overall trend
        self.market_state.regime = self._determine_market_regime()
        
        # Calculate volatility
        self.market_state.volatility = self._calculate_market_volatility()
        
        # Calculate trend strength
        self.market_state.trend_strength = self._calculate_trend_strength()
        
        # Calculate market sentiment (simplified)
        self.market_state.market_sentiment = self._calculate_market_sentiment()
        
        # Calculate liquidity score
        self.market_state.liquidity_score = self._calculate_liquidity_score()
    
    def _determine_market_session(self, current_time: datetime) -> MarketSession:
        """Determine current market session."""
        # Simplified logic - in production, use exchange-specific hours
        hour = current_time.hour
        
        if 9 <= hour < 16:
            return MarketSession.REGULAR
        elif 16 <= hour < 20:
            return MarketSession.AFTER_HOURS
        elif 4 <= hour < 9:
            return MarketSession.PREMARKET
        else:
            return MarketSession.CLOSED
    
    def _determine_market_regime(self) -> MarketRegime:
        """Determine current market regime."""
        if not self.data_cache:
            return MarketRegime.SIDEWAYS
        
        # Get overall market trend
        major_symbols = ['BTC', 'ETH', 'SPY', 'QQQ']
        trend_scores = []
        
        for symbol in major_symbols:
            if symbol in self.indicator_cache:
                indicators = self.indicator_cache[symbol]
                if indicators.sma_20 and indicators.sma_50:
                    current_price = self.get_latest_price(symbol)
                    if current_price:
                        trend_score = (current_price - indicators.sma_20) / indicators.sma_20
                        trend_scores.append(trend_score)
        
        if not trend_scores:
            return MarketRegime.SIDEWAYS
        
        avg_trend = sum(trend_scores) / len(trend_scores) if trend_scores else 0.0
        
        if avg_trend > 0.02:
            return MarketRegime.RISK_ON
        elif avg_trend < -0.02:
            return MarketRegime.RISK_OFF
        else:
            return MarketRegime.SIDEWAYS
    
    def _calculate_market_volatility(self) -> float:
        """Calculate overall market volatility."""
        volatility_scores = []
        
        for symbol, indicators in self.indicator_cache.items():
            if indicators.atr and self.get_latest_price(symbol):
                current_price = self.get_latest_price(symbol)
                if current_price > 0:
                    vol_score = indicators.atr / current_price
                    volatility_scores.append(vol_score)
        
        return sum(volatility_scores) / len(volatility_scores) if volatility_scores else 0.0
    
    def _calculate_trend_strength(self) -> float:
        """Calculate overall trend strength."""
        trend_strengths = []
        
        for symbol, indicators in self.indicator_cache.items():
            if indicators.sma_20 and indicators.sma_50 and indicators.bollinger_upper and indicators.bollinger_lower:
                current_price = self.get_latest_price(symbol)
                if current_price:
                    # Calculate distance from moving averages
                    sma_distance = abs(current_price - indicators.sma_20) / indicators.sma_20
                    bb_position = (current_price - indicators.bollinger_lower) / (indicators.bollinger_upper - indicators.bollinger_lower)
                    trend_strengths.append((sma_distance + abs(bb_position - 0.5)) / 2)
        
        return sum(trend_strengths) / len(trend_strengths) if trend_strengths else 0.0
    
    def _calculate_market_sentiment(self) -> float:
        """Calculate market sentiment (simplified)."""
        # In production, use news sentiment, options data, etc.
        return 0.0  # Neutral for now
    
    def _calculate_liquidity_score(self) -> float:
        """Calculate market liquidity score."""
        liquidity_scores = []
        
        for symbol, data_list in self.data_cache.items():
            if data_list:
                latest_data = data_list[-1]
                if latest_data.spread and latest_data.volume:
                    # Lower spread and higher volume = higher liquidity
                    spread_score = 1.0 / (1.0 + latest_data.spread)
                    volume_score = min(1.0, latest_data.volume / 1000000)  # Normalize volume
                    liquidity_scores.append((spread_score + volume_score) / 2)
        
        return sum(liquidity_scores) / len(liquidity_scores) if liquidity_scores else 0.5
    
    def get_market_state(self) -> MarketState:
        """Get current market state."""
        return self.market_state
    
    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        self.data_cache.clear()
        self.indicator_cache.clear()
        self.logger.info("Market data pipeline cleaned up")
