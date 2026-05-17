"""Tests for market data models.
"""

from datetime import datetime
from decimal import Decimal

import pytest

from amatix.data.market.models import (
    OHLCV,
    OrderBookLevel,
    OrderBookSnapshot,
    Quote,
    Symbol,
    Tick,
    Trade,
    TradeSide,
    DataSource,
)


class TestSymbol:
    """Symbol dataclass tests."""
    
    def test_symbol_creation(self):
        """Test basic symbol creation."""
        sym = Symbol("AAPL", "NASDAQ", "equity")
        assert sym.base == "AAPL"
        assert sym.exchange == "NASDAQ"
        assert sym.asset_class == "equity"
    
    def test_symbol_normalization(self):
        """Test symbol normalization to uppercase."""
        sym = Symbol("aapl", "nasdaq", "equity")
        assert sym.base == "AAPL"
        assert sym.exchange == "NASDAQ"
    
    def test_crypto_symbol(self):
        """Test crypto symbol with quote currency."""
        sym = Symbol("BTC", "BINANCE", "crypto", "USD")
        assert sym.canonical == "BTC/USD"
        assert str(sym) == "BTC/USD"


class TestQuote:
    """Quote dataclass tests."""
    
    def test_quote_mid(self):
        """Test mid price calculation."""
        quote = Quote(
            symbol=Symbol("AAPL", "NASDAQ"),
            bid=Decimal("150.00"),
            ask=Decimal("150.10"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            timestamp=datetime.utcnow(),
            source=DataSource.ALPACA,
        )
        assert quote.mid == Decimal("150.05")


class TestOHLCV:
    """OHLCV bar tests."""
    
    def test_ohlcv_properties(self):
        """Test OHLCV calculated properties."""
        bar = OHLCV(
            symbol=Symbol("AAPL", "NASDAQ"),
            timestamp=datetime.utcnow(),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("153.00"),
            volume=Decimal("1000000"),
            source=DataSource.ALPACA,
        )
        
        assert bar.range == Decimal("6.00")
        assert bar.change == Decimal("3.00")
        assert bar.is_green is True
        assert bar.is_red is False
    
    def test_ohlcv_to_dict(self):
        """Test OHLCV serialization."""
        bar = OHLCV(
            symbol=Symbol("AAPL", "NASDAQ"),
            timestamp=datetime.utcnow(),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("153.00"),
            volume=Decimal("1000000"),
            source=DataSource.ALPACA,
        )
        
        d = bar.to_dict()
        assert "symbol" in d
        assert "open" in d
        assert "close" in d


class TestTrade:
    """Trade dataclass tests."""
    
    def test_trade_value(self):
        """Test trade value calculation."""
        trade = Trade(
            symbol=Symbol("AAPL", "NASDAQ"),
            price=Decimal("150.00"),
            size=Decimal("100"),
            timestamp=datetime.utcnow(),
            side=TradeSide.BUY,
            source=DataSource.ALPACA,
        )
        assert trade.value == Decimal("15000.00")


class TestOrderBook:
    """Order book snapshot tests."""
    
    def test_order_book_best_prices(self):
        """Test best bid/ask extraction."""
        snapshot = OrderBookSnapshot(
            symbol=Symbol("AAPL", "NASDAQ"),
            timestamp=datetime.utcnow(),
            bids=[
                OrderBookLevel(Decimal("150.00"), Decimal("100")),
                OrderBookLevel(Decimal("149.99"), Decimal("200")),
            ],
            asks=[
                OrderBookLevel(Decimal("150.10"), Decimal("100")),
                OrderBookLevel(Decimal("150.11"), Decimal("200")),
            ],
            source=DataSource.ALPACA,
        )
        
        assert snapshot.best_bid.price == Decimal("150.00")
        assert snapshot.best_ask.price == Decimal("150.10")
        assert snapshot.mid == Decimal("150.05")
