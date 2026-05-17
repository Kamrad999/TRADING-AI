"""Tests for symbol normalizer."""

from amatix.data.market.models import Symbol
from amatix.data.market.normalizer import SymbolNormalizer


class TestSymbolNormalizer:
    """Symbol normalizer tests."""

    def test_equity_normalization(self):
        """Test equity symbol normalization."""
        normalizer = SymbolNormalizer()

        sym = normalizer.normalize("AAPL", "NASDAQ", "equity")
        assert sym.base == "AAPL"
        assert sym.asset_class == "equity"

    def test_crypto_normalization(self):
        """Test crypto symbol normalization."""
        normalizer = SymbolNormalizer()

        # BTC/USD format
        sym = normalizer.normalize("BTC/USD", "BINANCE", "crypto")
        assert sym.base == "BTC"
        assert sym.quote_currency == "USD"

        # BTCUSD format
        sym = normalizer.normalize("BTCUSD", "BINANCE", "crypto")
        assert sym.base == "BTC"
        assert sym.quote_currency == "USD"

    def test_forex_normalization(self):
        """Test forex symbol normalization."""
        normalizer = SymbolNormalizer()

        sym = normalizer.normalize("EUR/USD", "FOREX", "forex")
        assert sym.base == "EUR"
        assert sym.quote_currency == "USD"

    def test_provider_format_alpaca(self):
        """Test Alpaca format conversion."""
        normalizer = SymbolNormalizer()

        sym = Symbol("AAPL", "NASDAQ", "equity")
        assert normalizer.to_provider_format(sym, "alpaca") == "AAPL"

        sym = Symbol("BTC", "BINANCE", "crypto", "USD")
        assert normalizer.to_provider_format(sym, "alpaca") == "BTC/USD"

    def test_provider_format_yahoo(self):
        """Test Yahoo format conversion."""
        normalizer = SymbolNormalizer()

        sym = Symbol("AAPL", "NASDAQ", "equity")
        assert normalizer.to_provider_format(sym, "yahoo") == "AAPL"

        sym = Symbol("BTC", "BINANCE", "crypto", "USD")
        assert normalizer.to_provider_format(sym, "yahoo") == "BTC-USD"

    def test_provider_format_binance(self):
        """Test Binance format conversion."""
        normalizer = SymbolNormalizer()

        sym = Symbol("BTC", "BINANCE", "crypto", "USDT")
        assert normalizer.to_provider_format(sym, "binance") == "BTCUSDT"
