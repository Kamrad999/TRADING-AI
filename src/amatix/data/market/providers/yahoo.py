"""Yahoo Finance fallback provider for AMATIS.

Provides:
    - Historical OHLCV data
    - No real-time streaming (use Alpaca for that)
    - Good for backtesting and research

Note: Uses yfinance library. Not for high-frequency trading.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import yfinance as yf

from amatix.core.observability import get_logger, get_metrics
from amatix.data.market.models import OHLCV, Quote, Symbol, DataSource
from amatix.data.market.normalizer import SymbolNormalizer
from amatix.data.market.providers.base import BaseDataProvider, ProviderConfig

logger = get_logger(__name__)


class YahooDataProvider(BaseDataProvider):
    """Yahoo Finance data provider.
    
    Use for:
        - Fallback historical data
        - Backtesting
        - Research
        - Assets not on Alpaca
    
    Limitations:
        - No real-time streaming
        - Rate limited by Yahoo
        - Not for production trading
    
    Example:
        >>> provider = YahooDataProvider(config, event_bus)
        >>> await provider.connect()
        >>> bars = await provider.get_ohlcv(Symbol("AAPL", "NASDAQ"), "1D", 100)
    """
    
    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        event_bus: Optional[Any] = None,
    ) -> None:
        """Initialize Yahoo provider."""
        super().__init__(config or ProviderConfig(), event_bus)
        self._normalizer = SymbolNormalizer()
    
    @property
    def name(self) -> str:
        """Provider name."""
        return "yahoo"
    
    @property
    def source(self) -> DataSource:
        """Data source identifier."""
        return DataSource.YAHOO
    
    async def _connect_internal(self) -> None:
        """Yahoo doesn't require connection."""
        logger.info("Yahoo provider initialized (no connection needed)")
    
    async def _disconnect_internal(self) -> None:
        """No cleanup needed."""
        pass
    
    async def _get_price_internal(self, symbol: Symbol) -> Decimal:
        """Get current price from Yahoo."""
        quote = await self._get_quote_internal(symbol)
        return quote.mid
    
    async def _get_quote_internal(self, symbol: Symbol) -> Quote:
        """Get current quote from Yahoo."""
        # Yahoo format
        yahoo_symbol = self._normalizer.to_provider_format(symbol, "yahoo")
        
        try:
            # Use synchronous yfinance (runs in thread)
            import asyncio
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: yf.Ticker(yahoo_symbol)
            )
            
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ticker.info
            )
            
            bid = Decimal(str(info.get("bid", 0)))
            ask = Decimal(str(info.get("ask", 0)))
            
            # If no bid/ask, use last price
            if bid == 0 and ask == 0:
                last_price = Decimal(str(info.get("regularMarketPrice", 0)))
                bid = last_price
                ask = last_price
            
            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                bid_size=Decimal("0"),
                ask_size=Decimal("0"),
                timestamp=datetime.utcnow(),
                source=self.source,
            )
            
        except Exception as e:
            logger.error(
                "Yahoo quote fetch failed",
                symbol=str(symbol),
                error=str(e),
            )
            raise
    
    async def _get_ohlcv_internal(
        self,
        symbol: Symbol,
        timeframe: str,
        limit: int,
    ) -> List[OHLCV]:
        """Get historical bars from Yahoo."""
        yahoo_symbol = self._normalizer.to_provider_format(symbol, "yahoo")
        
        # Map timeframe to yfinance period/interval
        period, interval = self._map_timeframe(timeframe, limit)
        
        try:
            import asyncio
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: yf.Ticker(yahoo_symbol)
            )
            
            hist = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ticker.history(period=period, interval=interval)
            )
            
            if hist.empty:
                logger.warning(
                    "No historical data from Yahoo",
                    symbol=yahoo_symbol,
                )
                return []
            
            # Convert to OHLCV
            bars = []
            for timestamp, row in hist.iterrows():
                bars.append(OHLCV(
                    symbol=symbol,
                    timestamp=timestamp.to_pydatetime(),
                    open=Decimal(str(row.get("Open", 0))),
                    high=Decimal(str(row.get("High", 0))),
                    low=Decimal(str(row.get("Low", 0))),
                    close=Decimal(str(row.get("Close", 0))),
                    volume=Decimal(str(row.get("Volume", 0))),
                    source=self.source,
                ))
            
            # Return only requested limit (Yahoo may return more)
            bars = bars[-limit:] if len(bars) > limit else bars
            
            get_metrics().counter(
                "yahoo_ohlcv_fetched",
                labels={"symbol": str(symbol), "timeframe": timeframe},
            )
            
            return bars
            
        except Exception as e:
            logger.error(
                "Yahoo OHLCV fetch failed",
                symbol=str(symbol),
                timeframe=timeframe,
                error=str(e),
            )
            raise
    
    async def _subscribe_quotes_internal(self, symbols: List[Symbol]) -> None:
        """Yahoo doesn't support streaming."""
        logger.warning("Yahoo provider does not support streaming quotes")
    
    async def _subscribe_trades_internal(self, symbols: List[Symbol]) -> None:
        """Yahoo doesn't support streaming."""
        logger.warning("Yahoo provider does not support streaming trades")
    
    def _map_timeframe(self, timeframe: str, limit: int) -> tuple[str, str]:
        """Map AMATIS timeframe to yfinance period/interval.
        
        Returns:
            (period, interval)
        """
        # Map timeframes to yfinance parameters
        if timeframe in ["1m", "5m", "15m", "30m", "1h"]:
            # Intraday - limited history
            period = "5d"  # Max for intraday
            interval = timeframe
        elif timeframe == "1D":
            # Daily - calculate period from limit
            period = f"{limit + 50}d"  # Extra for weekends/holidays
            interval = "1d"
        elif timeframe == "1W":
            period = f"{limit * 2}wk"
            interval = "1wk"
        elif timeframe == "1M":
            period = f"{limit * 2}mo"
            interval = "1mo"
        else:
            # Default
            period = "1y"
            interval = "1d"
        
        return period, interval
    
    async def get_company_info(self, symbol: Symbol) -> Dict[str, Any]:
        """Get company information (Yahoo-specific).
        
        Args:
            symbol: Company symbol
        
        Returns:
            Dictionary of company info
        """
        yahoo_symbol = self._normalizer.to_provider_format(symbol, "yahoo")
        
        try:
            import asyncio
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: yf.Ticker(yahoo_symbol)
            )
            
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ticker.info
            )
            
            return {
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
                "employees": info.get("fullTimeEmployees"),
                "website": info.get("website", ""),
                "description": info.get("longBusinessSummary", ""),
            }
            
        except Exception as e:
            logger.error(
                "Yahoo company info fetch failed",
                symbol=str(symbol),
                error=str(e),
            )
            return {}
