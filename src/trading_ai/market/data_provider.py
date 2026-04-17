"""
Data provider for market data and technical indicators.
Following patterns from ai-trade and AgentQuant repositories.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .models import MarketData, PriceData
from .technical_indicators import TechnicalIndicators
from ..infrastructure.logging import get_logger
from ..infrastructure.config import config


class DataProvider:
    """
    Data provider for market data and technical indicators.
    
    Following patterns from:
    - ai-trade: Yahoo Finance API integration
    - AgentQuant: yfinance data pipeline
    - ai-hedge-fund-crypto: Multi-timeframe data processing
    """
    
    def __init__(self):
        """Initialize data provider."""
        self.logger = get_logger("data_provider")
        
        # Configuration
        self.data_source = config.get("DATA_SOURCE", "yfinance")
        self.update_interval = config.get("DATA_UPDATE_INTERVAL", 60)  # seconds
        self.max_history_days = config.get("MAX_HISTORY_DAYS", 30)
        
        # Technical indicators
        self.technical_indicators = TechnicalIndicators()
        
        # Data cache
        self.price_cache: Dict[str, PriceData] = {}
        self.ohlc_cache: Dict[str, List[MarketData]] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # Supported symbols
        self.crypto_symbols = ["BTC", "ETH", "ADA", "SOL", "DOT", "AVAX", "LINK", "UNI", "MATIC", "DOGE"]
        self.stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "PYPL"]
        
        self.logger.info(f"Data provider initialized with source: {self.data_source}")
    
    def get_current_price(self, symbol: str) -> Optional[PriceData]:
        """
        Get current price data for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price data with indicators
        """
        try:
            # Check cache
            if symbol in self.price_cache:
                last_update = self.last_update.get(symbol, datetime.min)
                if (datetime.now() - last_update).seconds < self.update_interval:
                    return self.price_cache[symbol]
            
            # Fetch fresh data
            price_data = self._fetch_price_data(symbol)
            
            if price_data:
                # Cache the data
                self.price_cache[symbol] = price_data
                self.last_update[symbol] = datetime.now()
                
                self.logger.debug(f"Updated price data for {symbol}: {price_data.price}")
                return price_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get price data for {symbol}: {e}")
            return None
    
    def get_ohlc_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[MarketData]:
        """
        Get OHLC data for symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of data points
            
        Returns:
            List of OHLC data
        """
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            # Check cache
            if cache_key in self.ohlc_cache:
                last_update = self.last_update.get(cache_key, datetime.min)
                if (datetime.now() - last_update).seconds < self.update_interval:
                    return self.ohlc_cache[cache_key][-limit:]
            
            # Fetch fresh data
            ohlc_data = self._fetch_ohlc_data(symbol, timeframe, limit)
            
            if ohlc_data:
                # Cache the data
                self.ohlc_cache[cache_key] = ohlc_data
                self.last_update[cache_key] = datetime.now()
                
                self.logger.debug(f"Updated OHLC data for {symbol} {timeframe}: {len(ohlc_data)} points")
                return ohlc_data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get OHLC data for {symbol}: {e}")
            return []
    
    def get_market_data(self, symbol: str, include_indicators: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive market data for symbol.
        
        Args:
            symbol: Trading symbol
            include_indicators: Whether to include technical indicators
            
        Returns:
            Market data dictionary
        """
        try:
            # Get current price
            price_data = self.get_current_price(symbol)
            if not price_data:
                return None
            
            # Get OHLC data for indicators
            ohlc_data = self.get_ohlc_data(symbol, "1h", 100)
            
            # Calculate indicators
            indicators = {}
            if include_indicators and ohlc_data:
                indicators = self.technical_indicators.calculate_all(ohlc_data)
            
            # Build market data
            market_data = {
                "symbol": symbol,
                "price": price_data.price,
                "volume": price_data.volume,
                "change_24h": price_data.change_24h,
                "change_pct_24h": price_data.change_pct_24h,
                "high_24h": price_data.high_24h,
                "low_24h": price_data.low_24h,
                "timestamp": price_data.timestamp,
                "indicators": indicators,
                "ohlc_data": ohlc_data,
                "liquidity": self._assess_liquidity(price_data),
                "volatility": self._calculate_volatility(ohlc_data),
                "trend": self._determine_trend(indicators)
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, List[MarketData]]:
        """
        Get multi-timeframe data for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of timeframe data
        """
        timeframes = ["5m", "15m", "1h", "4h", "1d"]
        data = {}
        
        for timeframe in timeframes:
            ohlc_data = self.get_ohlc_data(symbol, timeframe, 100)
            if ohlc_data:
                data[timeframe] = ohlc_data
        
        return data
    
    def _fetch_price_data(self, symbol: str) -> Optional[PriceData]:
        """Fetch price data from data source."""
        try:
            # Mock implementation - replace with actual API call
            # Following ai-trade pattern with Yahoo Finance API
            
            # Generate mock data for testing
            base_price = self._get_base_price(symbol)
            
            # Add some random variation
            import random
            variation = random.uniform(-0.05, 0.05)
            current_price = base_price * (1 + variation)
            
            # Calculate 24h change
            change_24h = random.uniform(-0.1, 0.1) * base_price
            change_pct_24h = change_24h / base_price
            
            # High/low
            high_24h = current_price * random.uniform(1.0, 1.05)
            low_24h = current_price * random.uniform(0.95, 1.0)
            
            # Volume
            volume = random.uniform(100000, 10000000)
            
            # Calculate indicators
            indicators = self._calculate_simple_indicators(current_price)
            
            return PriceData(
                symbol=symbol,
                price=current_price,
                volume=volume,
                change_24h=change_24h,
                change_pct_24h=change_pct_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                indicators=indicators,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
    
    def _fetch_ohlc_data(self, symbol: str, timeframe: str, limit: int) -> List[MarketData]:
        """Fetch OHLC data from data source."""
        try:
            # Mock implementation - replace with actual API call
            base_price = self._get_base_price(symbol)
            
            # Generate mock OHLC data
            ohlc_data = []
            current_time = datetime.now()
            
            # Timeframe in minutes
            timeframe_minutes = {
                "1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440
            }
            
            minutes = timeframe_minutes.get(timeframe, 60)
            
            import random
            current_price = base_price
            
            for i in range(limit):
                # Generate OHLC data
                open_price = current_price
                
                # Random walk
                change = random.uniform(-0.02, 0.02)
                high_price = open_price * (1 + random.uniform(0, 0.03))
                low_price = open_price * (1 - random.uniform(0, 0.03))
                close_price = open_price * (1 + change)
                
                # Ensure high/low are correct
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                volume = random.uniform(1000, 100000)
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=current_time - timedelta(minutes=minutes * (limit - i)),
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=volume
                )
                
                ohlc_data.append(market_data)
                current_price = close_price
            
            return ohlc_data
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLC data for {symbol}: {e}")
            return []
    
    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol."""
        crypto_prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "ADA": 1.0,
            "SOL": 100.0,
            "DOT": 20.0,
            "AVAX": 30.0,
            "LINK": 15.0,
            "UNI": 8.0,
            "MATIC": 0.8,
            "DOGE": 0.1
        }
        
        stock_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "AMZN": 3000.0,
            "TSLA": 800.0,
            "META": 200.0,
            "NVDA": 400.0,
            "NFLX": 350.0,
            "PYPL": 100.0
        }
        
        return crypto_prices.get(symbol, stock_prices.get(symbol, 100.0))
    
    def _calculate_simple_indicators(self, price: float) -> Dict[str, float]:
        """Calculate simple indicators for current price."""
        # Mock indicators - in production, calculate from historical data
        return {
            "rsi": 50.0,
            "macd": 0.0,
            "sma_20": price * 0.98,
            "sma_50": price * 0.95,
            "ema_12": price * 0.99,
            "ema_26": price * 0.97,
            "bollinger_upper": price * 1.02,
            "bollinger_lower": price * 0.98,
            "atr": price * 0.02
        }
    
    def _assess_liquidity(self, price_data: PriceData) -> str:
        """Assess market liquidity."""
        if price_data.volume > 1000000:
            return "high"
        elif price_data.volume > 100000:
            return "medium"
        else:
            return "low"
    
    def _calculate_volatility(self, ohlc_data: List[MarketData]) -> float:
        """Calculate price volatility."""
        if len(ohlc_data) < 2:
            return 0.0
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(ohlc_data)):
            prev_close = ohlc_data[i-1].close_price
            curr_close = ohlc_data[i].close_price
            if prev_close > 0:
                ret = (curr_close - prev_close) / prev_close
                returns.append(ret)
        
        if not returns:
            return 0.0
        
        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        return volatility
    
    def _determine_trend(self, indicators: Dict[str, float]) -> str:
        """Determine market trend from indicators."""
        if not indicators:
            return "neutral"
        
        # Check SMA trend
        sma_20 = indicators.get("sma_20", 0.0)
        sma_50 = indicators.get("sma_50", 0.0)
        current_price = indicators.get("current_price", 0.0)
        
        if sma_20 and sma_50 and current_price:
            if current_price > sma_20 > sma_50:
                return "bullish"
            elif current_price < sma_20 < sma_50:
                return "bearish"
        
        # Check MACD
        macd = indicators.get("macd", 0.0)
        if macd > 0.1:
            return "bullish"
        elif macd < -0.1:
            return "bearish"
        
        return "neutral"
    
    def cleanup(self) -> None:
        """Cleanup data provider resources."""
        self.price_cache.clear()
        self.ohlc_cache.clear()
        self.last_update.clear()
        self.logger.info("Data provider cleaned up")
