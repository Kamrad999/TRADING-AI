"""Symbol normalization across exchanges and asset classes.

Handles the messy reality of ticker symbols:
    - AAPL vs AAPL.NASDAQ
    - BTC vs BTC/USD vs BTC-USD
    - EUR/USD vs EURUSD

Provides canonical normalization for consistent internal representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from amatix.data.market.models import Symbol


class SymbolNormalizer:
    """Normalizes symbols across exchanges and data sources.
    
    The goal is to convert various symbol formats into a consistent
    internal representation that works across all AMATIS components.
    
    Supported formats:
        - Equity: AAPL, AAPL.NASDAQ
        - Crypto: BTC/USD, BTC-USD, BTCUSDT
        - Forex: EUR/USD, EURUSD
    
    Example:
        >>> normalizer = SymbolNormalizer()
        >>> sym = normalizer.normalize("AAPL", "NASDAQ", "equity")
        >>> str(sym)
        'AAPL'
        >>> 
        >>> sym = normalizer.normalize("BTC/USD", "BINANCE", "crypto")
        >>> str(sym)
        'BTC/USD'
    """
    
    # Exchange-specific suffixes
    EXCHANGE_SUFFIXES: Dict[str, str] = {
        "NASDAQ": "",
        "NYSE": "",
        "NYSE_ARCA": "",
        "BATS": "",
        "IEX": "",
        "TSX": ".TO",
        "LSE": ".L",
        "TSE": ".T",
        "HKEX": ".HK",
    }
    
    # Asset class mapping
    ASSET_CLASSES: Set[str] = {"equity", "crypto", "forex", "futures", "option"}
    
    def __init__(self) -> None:
        """Initialize normalizer."""
        self._cache: Dict[str, Symbol] = {}
    
    def normalize(
        self,
        symbol: str,
        exchange: str,
        asset_class: str = "equity",
        quote_currency: Optional[str] = None,
    ) -> Symbol:
        """Normalize a symbol to canonical form.
        
        Args:
            symbol: Raw symbol string
            exchange: Exchange identifier
            asset_class: Type of asset
            quote_currency: For pairs (crypto/forex)
        
        Returns:
            Normalized Symbol object
        """
        cache_key = f"{symbol}:{exchange}:{asset_class}:{quote_currency}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Parse and normalize
        base, parsed_quote = self._parse_symbol(symbol, asset_class)
        
        # Use provided quote_currency if available
        final_quote = quote_currency or parsed_quote
        
        # Normalize exchange
        normalized_exchange = self._normalize_exchange(exchange)
        
        result = Symbol(
            base=base,
            exchange=normalized_exchange,
            asset_class=asset_class.lower(),
            quote_currency=final_quote.upper() if final_quote else None,
        )
        
        self._cache[cache_key] = result
        return result
    
    def _parse_symbol(
        self,
        symbol: str,
        asset_class: str,
    ) -> tuple[str, Optional[str]]:
        """Parse symbol into base and quote components.
        
        Returns:
            (base, quote_currency or None)
        """
        symbol = symbol.strip().upper()
        
        if asset_class == "crypto":
            # Handle various crypto formats
            # BTC/USD, BTC-USD, BTCUSD, BTCUSDT
            for separator in ["/", "-"]:
                if separator in symbol:
                    parts = symbol.split(separator, 1)
                    return parts[0], parts[1]
            
            # Check for common quote currencies
            quote_currencies = ["USDT", "USD", "USDC", "BTC", "ETH", "EUR", "GBP"]
            for quote in quote_currencies:
                if symbol.endswith(quote):
                    return symbol[:-len(quote)], quote
            
            return symbol, None
        
        elif asset_class == "forex":
            # EUR/USD or EURUSD
            if "/" in symbol:
                parts = symbol.split("/", 1)
                return parts[0], parts[1]
            elif len(symbol) == 6:
                # Standard forex format: EURUSD
                return symbol[:3], symbol[3:]
            
            return symbol, None
        
        elif asset_class == "equity":
            # Remove exchange suffixes
            for suffix in self.EXCHANGE_SUFFIXES.values():
                if suffix and symbol.endswith(suffix):
                    return symbol[:-len(suffix)], None
            
            return symbol, None
        
        else:
            # Unknown asset class, return as-is
            return symbol, None
    
    def _normalize_exchange(self, exchange: str) -> str:
        """Normalize exchange identifier."""
        exchange_map = {
            "NASDAQ": "NASDAQ",
            "NYSE": "NYSE",
            "BINANCE": "BINANCE",
            "COINBASE": "COINBASE",
            "KRAKEN": "KRAKEN",
            "ALPACA": "ALPACA",
            "FOREX": "FOREX",
            "OANDA": "OANDA",
        }
        
        normalized = exchange.upper().strip()
        return exchange_map.get(normalized, normalized)
    
    def to_provider_format(
        self,
        symbol: Symbol,
        provider: str,
    ) -> str:
        """Convert Symbol to provider-specific format.
        
        Args:
            symbol: Normalized symbol
            provider: Provider name (alpaca, binance, etc.)
        
        Returns:
            Provider-formatted symbol string
        """
        provider_lower = provider.lower()
        
        if provider_lower == "alpaca":
            return self._to_alpaca(symbol)
        elif provider_lower == "binance":
            return self._to_binance(symbol)
        elif provider_lower == "yahoo":
            return self._to_yahoo(symbol)
        elif provider_lower == "polygon":
            return self._to_polygon(symbol)
        else:
            # Default to canonical format
            return str(symbol)
    
    def _to_alpaca(self, symbol: Symbol) -> str:
        """Convert to Alpaca format."""
        if symbol.asset_class == "crypto":
            if symbol.quote_currency:
                return f"{symbol.base}/{symbol.quote_currency}"
            return symbol.base
        return symbol.base
    
    def _to_binance(self, symbol: Symbol) -> str:
        """Convert to Binance format."""
        if symbol.quote_currency:
            return f"{symbol.base}{symbol.quote_currency}"
        return symbol.base
    
    def _to_yahoo(self, symbol: Symbol) -> str:
        """Convert to Yahoo Finance format."""
        if symbol.asset_class == "crypto":
            quote = symbol.quote_currency or "USD"
            return f"{symbol.base}-{quote}"
        elif symbol.asset_class == "forex":
            quote = symbol.quote_currency or "USD"
            return f"{symbol.base}{quote}=X"
        
        # Add exchange suffix for non-US equities
        suffix = self.EXCHANGE_SUFFIXES.get(symbol.exchange, "")
        return f"{symbol.base}{suffix}"
    
    def _to_polygon(self, symbol: Symbol) -> str:
        """Convert to Polygon.io format."""
        return symbol.base
    
    def batch_normalize(
        self,
        symbols: List[str],
        exchange: str,
        asset_class: str = "equity",
    ) -> List[Symbol]:
        """Normalize multiple symbols efficiently."""
        return [
            self.normalize(s, exchange, asset_class)
            for s in symbols
        ]
    
    def get_supported_exchanges(self) -> List[str]:
        """Get list of supported exchange identifiers."""
        return list(self.EXCHANGE_SUFFIXES.keys()) + [
            "BINANCE", "COINBASE", "KRAKEN", "ALPACA", "FOREX"
        ]
    
    def clear_cache(self) -> None:
        """Clear normalization cache."""
        self._cache.clear()


# Global normalizer instance
_global_normalizer: Optional[SymbolNormalizer] = None


def get_normalizer() -> SymbolNormalizer:
    """Get the global symbol normalizer."""
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = SymbolNormalizer()
    return _global_normalizer


def normalize_symbol(
    symbol: str,
    exchange: str,
    asset_class: str = "equity",
    quote_currency: Optional[str] = None,
) -> Symbol:
    """Convenience function for one-off normalization."""
    return get_normalizer().normalize(symbol, exchange, asset_class, quote_currency)
