"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║   ███████╗██╗  ██╗███████╗ ██████╗██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗         ║
║   ██╔════╝╚██╗██╔╝██╔════╝██╔════╝██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║         ║
║   █████╗   ╚███╔╝ █████╗  ██║     ██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║         ║
║   ██╔══╝   ██╔██╗ ██╔══╝  ██║     ██║   ██║   ██║   ██║██║   ██║██║╚██╗██║         ║
║   ███████╗██╔╝ ██╗███████╗╚██████╗╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║         ║
║   ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝         ║
║                                                                                      ║
║   ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗                                      ║
║   ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝                                      ║
║   ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗                                        ║
║   ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝                                        ║
║   ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗                                      ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝                                      ║
║                                                                                      ║
║   ┌──────────────────────────────────────────────────────────────────────────────┐   ║
║   │               ⚡  E X E C U T I O N   B R I D G E  ⚡                        │   ║
║   │           Pipeline Stage 6 — Broker Order Construction Engine                │   ║
║   │   alert_router → [YOU] → alpaca_adapter / ibkr_adapter / binance_adapter     │   ║
║   └──────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                      ║
║   Module  : execution_bridge.py                                                      ║
║   Version : 1.0.0                                                                    ║
║   Role    : Convert routed execution candidates into broker-ready order payloads     ║
║   Brokers : Alpaca · IBKR · Binance  (adapter-ready)                                ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import math
import sys
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable

# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1 ▸ CONSTANTS & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

BRIDGE_VERSION = "1.0.0"
BRIDGE_BUILD   = "MONSTER-TRADING-AI"

# ── Confidence thresholds ────────────────────────────────────────────────────
CONFIDENCE_AGGRESSIVE_FLOOR = 0.90   # confidence > 0.90 → aggressive sizing
CONFIDENCE_STANDARD_FLOOR   = 0.80   # confidence 0.80–0.90 → standard sizing
# below CONFIDENCE_STANDARD_FLOOR → REJECTED

# ── Position sizing (fraction of notional portfolio) ─────────────────────────
AGGRESSIVE_SIZE_PCT = 0.05    # 5% of portfolio per position
STANDARD_SIZE_PCT   = 0.025   # 2.5% of portfolio per position
DEFAULT_PORTFOLIO_VALUE = 100_000.0   # overridable via set_portfolio_value()

# ── Risk management ──────────────────────────────────────────────────────────
MAX_CONCURRENT_POSITIONS = 5         # hard cap on live positions
TICKER_COOLDOWN_MINUTES  = 15        # same ticker blocked for N minutes
MAX_RISK_PER_TICKER_PCT  = 0.10      # max 10% of portfolio exposed to one ticker

# ── SL / TP templates by asset class ─────────────────────────────────────────
#   (stop_loss_pct, take_profit_pct)  — expressed as decimal fractions
SL_TP_TEMPLATES: dict[str, tuple[float, float]] = {
    "CRYPTO":  (0.06, 0.12),   # wider — volatile
    "MACRO":   (0.015, 0.03),  # tight — macro / index
    "EQUITY":  (0.03, 0.06),   # standard equities
    "FOREX":   (0.01, 0.02),   # tight — FX
    "DEFAULT": (0.03, 0.06),   # fallback
}

# ── Paper trade mode (global toggle) ─────────────────────────────────────────
_PAPER_TRADE_MODE: bool = True   # safe default — production sets to False


def enable_live_trading() -> None:
    """Switch bridge to LIVE mode.  Call explicitly to prevent accidents."""
    global _PAPER_TRADE_MODE
    _PAPER_TRADE_MODE = False


def enable_paper_trading() -> None:
    """Switch bridge back to paper/simulation mode."""
    global _PAPER_TRADE_MODE
    _PAPER_TRADE_MODE = True


# ── Portfolio value (used for position sizing) ────────────────────────────────
_PORTFOLIO_VALUE: float = DEFAULT_PORTFOLIO_VALUE


def set_portfolio_value(value: float) -> None:
    global _PORTFOLIO_VALUE
    if value <= 0:
        raise ValueError("Portfolio value must be positive")
    _PORTFOLIO_VALUE = value


# ── Required fields from alert_router.py ─────────────────────────────────────
REQUIRED_ROUTED_FIELDS = {
    "delivery_channel",
    "execution_candidate",
    "alert_priority",
    "formatted_alert",
}

REQUIRED_SIGNAL_FIELDS = {
    "signal_type",
    "confidence_score",
    "ticker",
}


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2 ▸ ENUMS & STATUS CODES
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionStatus(str, Enum):
    QUEUED        = "QUEUED"        # accepted, awaiting broker send
    PAPER_QUEUED  = "PAPER_QUEUED"  # paper trade queued
    REJECTED      = "REJECTED"      # did not pass bridge checks
    COOLDOWN      = "COOLDOWN"      # ticker in cooldown window
    CAP_EXCEEDED  = "CAP_EXCEEDED"  # max positions or risk cap hit
    MALFORMED     = "MALFORMED"     # could not parse entry


class SizingTier(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"   # confidence > 0.90
    STANDARD   = "STANDARD"     # confidence 0.80–0.90
    REJECTED   = "REJECTED"     # below floor


class AssetClass(str, Enum):
    CRYPTO  = "CRYPTO"
    MACRO   = "MACRO"
    EQUITY  = "EQUITY"
    FOREX   = "FOREX"
    DEFAULT = "DEFAULT"


class BrokerAdapter(str, Enum):
    ALPACA  = "alpaca"
    IBKR    = "ibkr"
    BINANCE = "binance"
    PAPER   = "paper"


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3 ▸ COOLDOWN REGISTRY  (O(1) lookup via dict)
# ──────────────────────────────────────────────────────────────────────────────

class CooldownRegistry:
    """
    Thread-unsafe, in-process cooldown tracker.
    For multi-process deployments, swap _store for a Redis backend
    by subclassing and overriding get / set / clear.
    """

    def __init__(self, cooldown_minutes: int = TICKER_COOLDOWN_MINUTES) -> None:
        self._store: dict[str, datetime] = {}       # ticker → last execution time
        self._cooldown = timedelta(minutes=cooldown_minutes)

    def is_cooled_down(self, ticker: str) -> bool:
        """True = ticker is free to trade.  O(1)."""
        last = self._store.get(ticker)
        if last is None:
            return True
        return (datetime.now(timezone.utc) - last) >= self._cooldown

    def record_execution(self, ticker: str) -> None:
        self._store[ticker] = datetime.now(timezone.utc)

    def remaining_seconds(self, ticker: str) -> float:
        last = self._store.get(ticker)
        if last is None:
            return 0.0
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        remaining = self._cooldown.total_seconds() - elapsed
        return max(0.0, remaining)

    def clear(self, ticker: str | None = None) -> None:
        if ticker:
            self._store.pop(ticker, None)
        else:
            self._store.clear()

    def snapshot(self) -> dict[str, str]:
        return {t: dt.isoformat() for t, dt in self._store.items()}


# Module-level singleton
_COOLDOWN_REGISTRY = CooldownRegistry()


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 4 ▸ POSITION LEDGER  (tracks concurrent live positions)
# ──────────────────────────────────────────────────────────────────────────────

class PositionLedger:
    """
    Tracks open positions for concurrency cap enforcement.
    In production, replace _positions with a database-backed store.
    """

    def __init__(self, max_positions: int = MAX_CONCURRENT_POSITIONS) -> None:
        self._positions: dict[str, dict] = {}   # ticker → position metadata
        self._max = max_positions

    @property
    def count(self) -> int:
        return len(self._positions)

    @property
    def is_full(self) -> bool:
        return self.count >= self._max

    def has_position(self, ticker: str) -> bool:
        return ticker in self._positions

    def exposure_pct(self, ticker: str) -> float:
        pos = self._positions.get(ticker)
        return pos["size_pct"] if pos else 0.0

    def open_position(self, ticker: str, size_pct: float, order_id: str) -> None:
        self._positions[ticker] = {
            "ticker":    ticker,
            "size_pct":  size_pct,
            "order_id":  order_id,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }

    def close_position(self, ticker: str) -> None:
        self._positions.pop(ticker, None)

    def snapshot(self) -> list[dict]:
        return list(self._positions.values())

    def clear(self) -> None:
        self._positions.clear()


# Module-level singleton
_POSITION_LEDGER = PositionLedger()


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 5 ▸ ASSET CLASS CLASSIFIER  (O(1) via lookup sets)
# ──────────────────────────────────────────────────────────────────────────────

# Crypto symbols — extend as needed
_CRYPTO_SYMBOLS = frozenset({
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
    "DOT", "MATIC", "LINK", "LTC", "BCH", "ATOM", "UNI",
    "BTCUSD", "ETHUSD", "SOLUSD",
})

# Macro / index instruments
_MACRO_SYMBOLS = frozenset({
    "SPY", "QQQ", "IVV", "VTI", "DIA", "GLD", "SLV", "TLT",
    "VIX", "ES", "NQ", "YM", "RTY", "CL", "GC", "SI",
    "SPX", "NDX", "RUT", "DJIA",
})

# Forex pairs (6-char or slash-separated)
_FOREX_PATTERNS = frozenset({
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
    "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
})


def classify_asset(ticker: str) -> AssetClass:
    """O(1) asset class lookup."""
    t = ticker.upper().replace("/", "").replace("-", "")
    if t in _CRYPTO_SYMBOLS or t.endswith("USDT") or t.endswith("BTC"):
        return AssetClass.CRYPTO
    if t in _MACRO_SYMBOLS:
        return AssetClass.MACRO
    if t in _FOREX_PATTERNS or (len(t) == 6 and t.isalpha()):
        return AssetClass.FOREX
    return AssetClass.EQUITY


def get_sl_tp(asset_class: AssetClass) -> tuple[float, float]:
    """Return (stop_loss_pct, take_profit_pct) for the asset class."""
    return SL_TP_TEMPLATES.get(asset_class.value, SL_TP_TEMPLATES["DEFAULT"])


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 6 ▸ POSITION SIZER
# ──────────────────────────────────────────────────────────────────────────────

def compute_sizing_tier(confidence: float) -> SizingTier:
    """O(1) tier classification."""
    if confidence > CONFIDENCE_AGGRESSIVE_FLOOR:
        return SizingTier.AGGRESSIVE
    if confidence >= CONFIDENCE_STANDARD_FLOOR:
        return SizingTier.STANDARD
    return SizingTier.REJECTED


def compute_position_size(
    confidence: float,
    asset_class: AssetClass,
    ticker: str,
) -> tuple[SizingTier, float, float]:
    """
    Returns (tier, size_pct, notional_usd).
    Applies risk-cap guard: never exceed MAX_RISK_PER_TICKER_PCT
    after accounting for existing exposure.
    """
    tier = compute_sizing_tier(confidence)
    if tier == SizingTier.REJECTED:
        return tier, 0.0, 0.0

    base_pct = AGGRESSIVE_SIZE_PCT if tier == SizingTier.AGGRESSIVE else STANDARD_SIZE_PCT

    # Crypto: slightly larger to compensate for wider SL
    if asset_class == AssetClass.CRYPTO:
        base_pct *= 1.20
    # Macro: slightly smaller (tighter risk)
    elif asset_class == AssetClass.MACRO:
        base_pct *= 0.80

    # Clamp to risk cap (existing + new ≤ MAX_RISK_PER_TICKER_PCT)
    existing = _POSITION_LEDGER.exposure_pct(ticker)
    available_cap = max(0.0, MAX_RISK_PER_TICKER_PCT - existing)
    size_pct = min(base_pct, available_cap)

    notional_usd = round(_PORTFOLIO_VALUE * size_pct, 2)
    size_pct     = round(size_pct, 6)

    return tier, size_pct, notional_usd


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 7 ▸ BROKER ADAPTER FORMATTERS
# ──────────────────────────────────────────────────────────────────────────────

class AlpacaAdapter:
    """
    Alpaca Markets REST API v2 order payload.
    Plugin hook: alpaca_sender.py → requests.post(ALPACA_URL, json=payload, headers=auth)
    Docs: https://docs.alpaca.markets/reference/postorder
    """

    @staticmethod
    def build(
        ticker: str,
        signal_type: str,
        notional_usd: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        order_id: str,
        paper: bool,
    ) -> dict:
        side     = "buy" if signal_type == "BUY" else "sell"
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        return {
            "_adapter":          "alpaca",
            "_base_url":         f"{base_url}/v2/orders",
            "symbol":            ticker,
            "notional":          str(notional_usd),   # dollar-based fractional
            "side":              side,
            "type":              "market",
            "time_in_force":     "day",
            "order_class":       "bracket",
            "stop_loss":         {"stop_price_pct": round(stop_loss_pct * 100, 2)},
            "take_profit":       {"limit_price_pct": round(take_profit_pct * 100, 2)},
            "client_order_id":   order_id,
            "extended_hours":    False,
        }


class IBKRAdapter:
    """
    Interactive Brokers Client Portal API order payload.
    Plugin hook: ibkr_sender.py → requests.post(IBKR_GATEWAY_URL, json=payload)
    Docs: https://www.interactivebrokers.com/api/doc.html#tag/Order/paths/~1iserver~1account~1{accountId}~1orders/post
    """

    @staticmethod
    def build(
        ticker: str,
        signal_type: str,
        notional_usd: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        order_id: str,
        paper: bool,
    ) -> dict:
        side = "BUY" if signal_type == "BUY" else "SELL"
        return {
            "_adapter":        "ibkr",
            "_endpoint":       "/iserver/account/{accountId}/orders",
            "acctId":          "PAPER_ACCOUNT" if paper else "{IBKR_ACCOUNT_ID}",
            "conid":           "{CONID_LOOKUP_REQUIRED}",   # must resolve via /iserver/secdef/search
            "secType":         "STK",
            "orderType":       "MKT",
            "listingExchange": "SMART",
            "outsideRTH":      False,
            "side":            side,
            "cashQty":         notional_usd,
            "tif":             "DAY",
            "referenceId":     order_id,
            "stopPrice":       round(stop_loss_pct * 100, 2),     # pct — caller resolves to price
            "profitTarget":    round(take_profit_pct * 100, 2),   # pct — caller resolves to price
            "isPaperOrder":    paper,
        }


class BinanceAdapter:
    """
    Binance Spot API order payload.
    Plugin hook: binance_sender.py → requests.post(BINANCE_URL, data=payload, headers=auth)
    Docs: https://binance-docs.github.io/apidocs/spot/en/#new-order-trade
    """

    @staticmethod
    def build(
        ticker: str,
        signal_type: str,
        notional_usd: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        order_id: str,
        paper: bool,
    ) -> dict:
        side = "BUY" if signal_type == "BUY" else "SELL"
        symbol = ticker.upper().replace("/", "").replace("-", "")
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        return {
            "_adapter":            "binance",
            "_base_url":           "https://testnet.binance.vision/api/v3/order" if paper
                                   else "https://api.binance.com/api/v3/order",
            "symbol":              symbol,
            "side":                side,
            "type":                "MARKET",
            "quoteOrderQty":       notional_usd,   # USDT amount
            "newClientOrderId":    order_id,
            "newOrderRespType":    "FULL",
            "stopLossPct":         round(stop_loss_pct * 100, 2),     # informational
            "takeProfitPct":       round(take_profit_pct * 100, 2),   # informational
            "_paper":              paper,
        }


class PaperAdapter:
    """Internal paper trade record — no real broker endpoint."""

    @staticmethod
    def build(
        ticker: str,
        signal_type: str,
        notional_usd: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        order_id: str,
    ) -> dict:
        return {
            "_adapter":       "paper",
            "_base_url":      "internal://paper_ledger",
            "ticker":         ticker,
            "side":           signal_type,
            "notional_usd":   notional_usd,
            "stop_loss_pct":  stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "order_id":       order_id,
            "paper":          True,
            "simulated_at":   datetime.now(timezone.utc).isoformat(),
        }


# Dispatch table: adapter_name → callable  (O(1) selection)
_ADAPTER_DISPATCH: dict[str, Callable] = {
    BrokerAdapter.ALPACA:  AlpacaAdapter.build,
    BrokerAdapter.IBKR:    IBKRAdapter.build,
    BrokerAdapter.BINANCE: BinanceAdapter.build,
}

# Active adapter — change with set_broker_adapter()
_ACTIVE_ADAPTER: str = BrokerAdapter.ALPACA


def set_broker_adapter(adapter: str) -> None:
    """
    Switch the active broker adapter.
    Valid values: 'alpaca', 'ibkr', 'binance'
    """
    global _ACTIVE_ADAPTER
    if adapter not in _ADAPTER_DISPATCH:
        raise ValueError(f"Unknown adapter '{adapter}'. Choose from: {list(_ADAPTER_DISPATCH)}")
    _ACTIVE_ADAPTER = adapter


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 8 ▸ ORDER BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def _extract_signal(routed: dict) -> dict:
    """
    Pull raw_signal out of formatted_alert (set by alert_router.py).
    Falls back to top-level keys for forward/backward compatibility.
    """
    formatted = routed.get("formatted_alert", {})
    raw = formatted.get("raw_signal") if isinstance(formatted, dict) else None
    if raw and isinstance(raw, dict):
        return raw
    # Fallback: some callers may pass a flat dict
    return routed


def _build_order(
    routed: dict,
    order_id: str,
) -> dict:
    """
    Core order construction.  Returns the 7 required output fields.
    Does NOT perform gate checks (those happen in build_execution_orders).
    """
    paper    = _PAPER_TRADE_MODE
    signal   = _extract_signal(routed)

    ticker       = str(signal.get("ticker", "UNKNOWN")).upper()
    signal_type  = str(signal.get("signal_type", "BUY")).upper()
    confidence   = float(signal.get("confidence_score", 0.0))
    asset_class  = classify_asset(ticker)
    sl_pct, tp_pct = get_sl_tp(asset_class)

    tier, size_pct, notional_usd = compute_position_size(confidence, asset_class, ticker)

    if tier == SizingTier.REJECTED:
        return {
            "broker_payload":    {},
            "position_size":     0.0,
            "stop_loss_pct":     sl_pct,
            "take_profit_pct":   tp_pct,
            "paper_trade":       paper,
            "execution_status":  ExecutionStatus.REJECTED,
            "execution_reason":  f"Confidence {confidence:.2%} below minimum {CONFIDENCE_STANDARD_FLOOR:.0%} floor",
        }

    # Select adapter
    if paper:
        broker_payload = PaperAdapter.build(
            ticker, signal_type, notional_usd, sl_pct, tp_pct, order_id
        )
    else:
        adapter_fn     = _ADAPTER_DISPATCH[_ACTIVE_ADAPTER]
        broker_payload = adapter_fn(
            ticker, signal_type, notional_usd, sl_pct, tp_pct, order_id, paper
        )

    exec_status = ExecutionStatus.PAPER_QUEUED if paper else ExecutionStatus.QUEUED
    exec_reason = (
        f"{'PAPER' if paper else 'LIVE'} {signal_type} order built — "
        f"{tier.value} sizing ({size_pct:.2%} of portfolio / ${notional_usd:,.2f}) | "
        f"asset_class={asset_class.value} | "
        f"SL={sl_pct:.1%} TP={tp_pct:.1%} | "
        f"adapter={_ACTIVE_ADAPTER if not paper else 'paper'}"
    )

    # Register execution in cooldown + ledger
    _COOLDOWN_REGISTRY.record_execution(ticker)
    _POSITION_LEDGER.open_position(ticker, size_pct, order_id)

    return {
        "broker_payload":    broker_payload,
        "position_size":     size_pct,
        "stop_loss_pct":     sl_pct,
        "take_profit_pct":   tp_pct,
        "paper_trade":       paper,
        "execution_status":  exec_status,
        "execution_reason":  exec_reason,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 9 ▸ GATE CHECKS  (O(1) each)
# ──────────────────────────────────────────────────────────────────────────────

class _Reject(Exception):
    """Raised to short-circuit gate pipeline with a status + reason."""
    def __init__(self, status: ExecutionStatus, reason: str) -> None:
        self.status = status
        self.reason = reason


def _gate_channel(routed: dict) -> None:
    if routed.get("delivery_channel") != "execution_queue":
        raise _Reject(
            ExecutionStatus.REJECTED,
            f"delivery_channel='{routed.get('delivery_channel')}' — only 'execution_queue' accepted",
        )


def _gate_exec_candidate(routed: dict) -> None:
    if not routed.get("execution_candidate", False):
        raise _Reject(
            ExecutionStatus.REJECTED,
            "execution_candidate=False — signal not flagged for execution",
        )


def _gate_human_review(routed: dict) -> None:
    if routed.get("requires_human_confirmation", False):
        raise _Reject(
            ExecutionStatus.REJECTED,
            "requires_human_confirmation=True — must be cleared by analyst before execution",
        )


def _gate_confidence(signal: dict) -> None:
    confidence = float(signal.get("confidence_score", 0.0))
    if confidence < CONFIDENCE_STANDARD_FLOOR:
        raise _Reject(
            ExecutionStatus.REJECTED,
            f"Confidence {confidence:.2%} below minimum {CONFIDENCE_STANDARD_FLOOR:.0%}",
        )


def _gate_cooldown(ticker: str) -> None:
    if not _COOLDOWN_REGISTRY.is_cooled_down(ticker):
        remaining = _COOLDOWN_REGISTRY.remaining_seconds(ticker)
        raise _Reject(
            ExecutionStatus.COOLDOWN,
            f"Ticker '{ticker}' in cooldown — {remaining:.0f}s remaining "
            f"({TICKER_COOLDOWN_MINUTES}min window)",
        )


def _gate_position_cap() -> None:
    if _POSITION_LEDGER.is_full:
        raise _Reject(
            ExecutionStatus.CAP_EXCEEDED,
            f"Max concurrent positions ({MAX_CONCURRENT_POSITIONS}) reached — "
            "close a position before opening new ones",
        )


def _gate_risk_cap(ticker: str, confidence: float, asset_class: AssetClass) -> None:
    _, size_pct, _ = compute_position_size(confidence, asset_class, ticker)
    if size_pct <= 0.0:
        existing = _POSITION_LEDGER.exposure_pct(ticker)
        raise _Reject(
            ExecutionStatus.CAP_EXCEEDED,
            f"Risk cap exceeded for '{ticker}' — existing exposure {existing:.1%} "
            f"≥ max {MAX_RISK_PER_TICKER_PCT:.0%}",
        )


def _run_all_gates(routed: dict) -> dict:
    """
    Run all gate checks in priority order.
    Returns the extracted signal dict on success.
    Raises _Reject on any failure.
    """
    _gate_channel(routed)
    _gate_exec_candidate(routed)
    _gate_human_review(routed)

    signal     = _extract_signal(routed)
    confidence = float(signal.get("confidence_score", 0.0))
    ticker     = str(signal.get("ticker", "")).upper()

    _gate_confidence(signal)
    _gate_cooldown(ticker)
    _gate_position_cap()

    asset_class = classify_asset(ticker)
    _gate_risk_cap(ticker, confidence, asset_class)

    return signal


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 10 ▸ PUBLIC API  +  PLUGIN REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

# Plugin registry for broker_sender.py and friends
_PLUGIN_REGISTRY: list[Callable[[dict], None]] = []


def register_plugin(fn: Callable[[dict], None]) -> None:
    """
    Register a downstream plugin.

    Example (alpaca_sender.py):
        from execution_bridge import register_plugin

        def send_to_alpaca(order: dict):
            if order["execution_status"] in ("QUEUED", "PAPER_QUEUED"):
                payload = order["broker_payload"]
                requests.post(payload["_base_url"], json=payload, headers=HEADERS)

        register_plugin(send_to_alpaca)
    """
    _PLUGIN_REGISTRY.append(fn)


def build_execution_orders(routed_articles: list[dict]) -> list[dict]:
    """
    Convert routed execution candidates into broker-ready order payloads.

    Parameters
    ----------
    routed_articles : list[dict]
        Output from alert_router.route_alerts().
        Only entries with delivery_channel == 'execution_queue' are processed.

    Returns
    -------
    list[dict]
        One order record per valid candidate.  Each contains:

        broker_payload    : dict  — adapter-specific order object
        position_size     : float — fraction of portfolio (e.g. 0.05 = 5%)
        stop_loss_pct     : float — SL distance as decimal (e.g. 0.03 = 3%)
        take_profit_pct   : float — TP distance as decimal (e.g. 0.06 = 6%)
        paper_trade       : bool  — True if running in paper/sim mode
        execution_status  : str   — QUEUED | PAPER_QUEUED | REJECTED | COOLDOWN | CAP_EXCEEDED
        execution_reason  : str   — plain-English explanation
    """
    results: list[dict] = []

    for idx, routed in enumerate(routed_articles):
        order_id = str(uuid.uuid4())

        # ── Structural validation + auto-enrichment ──────────────────────────
        try:
            if not isinstance(routed, dict):
                raise _Reject(ExecutionStatus.MALFORMED, f"Expected dict, got {type(routed).__name__}")
            
            # Auto-create missing alert_router fields if not present
            # (allows pipeline to work even if alert_router hasn't run yet)
            if "delivery_channel" not in routed:
                confidence = float(routed.get("confidence_score", 0.0))
                # High-confidence signals → execution queue; others → human review
                routed["delivery_channel"] = "execution_queue" if confidence >= 0.75 else "human_review"
            
            if "execution_candidate" not in routed:
                confidence = float(routed.get("confidence_score", 0.0))
                routed["execution_candidate"] = confidence >= 0.75
            
            if "alert_priority" not in routed:
                confidence = float(routed.get("confidence_score", 0.0))
                urgency = routed.get("urgency", "LOW")
                # Map based on confidence + urgency
                if confidence >= 0.90 or urgency == "CRITICAL":
                    routed["alert_priority"] = "CRITICAL"
                elif confidence >= 0.80 or urgency == "HIGH":
                    routed["alert_priority"] = "HIGH"
                elif confidence >= 0.65 or urgency == "MEDIUM":
                    routed["alert_priority"] = "MEDIUM"
                else:
                    routed["alert_priority"] = "LOW"
            
            if "formatted_alert" not in routed:
                # Generate minimal formatted_alert structure
                routed["formatted_alert"] = {
                    "raw_signal": routed,
                    "signal_type": routed.get("signal_type", routed.get("signal_direction", "NO_TRADE")),
                    "confidence": float(routed.get("confidence_score", 0.0)),
                }
            
        except _Reject as exc:
            _warn(f"[SKIP] Entry {idx} — {exc.status}: {exc.reason}")
            continue
        except Exception as exc:  # noqa: BLE001
            _warn(f"[SKIP] Entry {idx} — unexpected: {exc}")
            continue

        # ── Gate pipeline ────────────────────────────────────────────────────
        try:
            _run_all_gates(routed)
        except _Reject as exc:
            # Non-execution-queue entries are noisy — log only exec-queue rejections
            if routed.get("delivery_channel") == "execution_queue":
                _warn(f"[GATE] Entry {idx} — {exc.status}: {exc.reason}")
            reject_record = {
                "broker_payload":    {},
                "position_size":     0.0,
                "stop_loss_pct":     0.0,
                "take_profit_pct":   0.0,
                "paper_trade":       _PAPER_TRADE_MODE,
                "execution_status":  exc.status,
                "execution_reason":  exc.reason,
                # preserve routing context for downstream audit
                "_ticker":           _extract_signal(routed).get("ticker"),
                "_order_id":         order_id,
            }
            results.append(reject_record)
            continue

        # ── Build order ──────────────────────────────────────────────────────
        try:
            order = _build_order(routed, order_id)
            order["_order_id"] = order_id
            order["_ticker"]   = _extract_signal(routed).get("ticker")
        except Exception as exc:  # noqa: BLE001
            _warn(f"[BUILD] Entry {idx} — order construction failed: {exc}\n{traceback.format_exc()}")
            results.append({
                "broker_payload":    {},
                "position_size":     0.0,
                "stop_loss_pct":     0.0,
                "take_profit_pct":   0.0,
                "paper_trade":       _PAPER_TRADE_MODE,
                "execution_status":  ExecutionStatus.MALFORMED,
                "execution_reason":  f"Order build error: {exc}",
                "_order_id":         order_id,
                "_ticker":           None,
            })
            continue

        # ── Fire plugins ─────────────────────────────────────────────────────
        for plugin in _PLUGIN_REGISTRY:
            try:
                plugin(order)
            except Exception as exc:  # noqa: BLE001
                _warn(f"Plugin '{plugin.__name__}' raised: {exc}")

        results.append(order)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 10b ▸ HELPERS & TERMINAL DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

def _warn(msg: str) -> None:
    print(f"\033[1;33m⚠  EXEC_BRIDGE WARNING ▸ {msg}\033[0m", file=sys.stderr)


def print_execution_summary(orders: list[dict]) -> None:
    """Rich terminal summary banner — call after build_execution_orders()."""
    PURPLE = "\033[1;35m"
    GREEN  = "\033[1;32m"
    RED    = "\033[1;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[1;36m"
    RESET  = "\033[0m"

    queued      = [o for o in orders if o["execution_status"] in (ExecutionStatus.QUEUED, ExecutionStatus.PAPER_QUEUED)]
    rejected    = [o for o in orders if o["execution_status"] == ExecutionStatus.REJECTED]
    cooldowns   = [o for o in orders if o["execution_status"] == ExecutionStatus.COOLDOWN]
    cap_hits    = [o for o in orders if o["execution_status"] == ExecutionStatus.CAP_EXCEEDED]
    paper_count = sum(1 for o in queued if o.get("paper_trade"))
    live_count  = len(queued) - paper_count
    total_notional = sum(
        o.get("broker_payload", {}).get("notional", 0) or
        o.get("broker_payload", {}).get("cashQty", 0) or
        o.get("broker_payload", {}).get("quoteOrderQty", 0) or
        (_PORTFOLIO_VALUE * o.get("position_size", 0))
        for o in queued
    )

    print(f"\n{PURPLE}{'═'*80}{RESET}")
    print(f"{PURPLE}  ⚡  EXECUTION BRIDGE — ORDER SUMMARY{RESET}")
    print(f"{PURPLE}{'═'*80}{RESET}")
    print(f"  Mode          : {'📋 PAPER' if _PAPER_TRADE_MODE else '🔴 LIVE'}")
    print(f"  Adapter       : {_ACTIVE_ADAPTER.upper()}")
    print(f"  Portfolio     : ${_PORTFOLIO_VALUE:>12,.2f}")
    print(f"  Open Positions: {_POSITION_LEDGER.count}/{MAX_CONCURRENT_POSITIONS}")
    print()
    print(f"  {GREEN}✅ Queued       : {len(queued):>4}  "
          f"(live={live_count}  paper={paper_count}){RESET}")
    print(f"  {YELLOW}🚫 Rejected     : {len(rejected):>4}{RESET}")
    print(f"  {CYAN}⏳ Cooldown     : {len(cooldowns):>4}{RESET}")
    print(f"  {RED}🔒 Cap Exceeded : {len(cap_hits):>4}{RESET}")
    print(f"  {'─'*50}")
    print(f"  💰 Total Notional : ${total_notional:>12,.2f}")
    print()

    if queued:
        print(f"  {'─'*78}")
        print(f"  {'ORDER ID':<38} {'TICKER':<8} {'SIDE':<5} {'SIZE':>6}  {'SL':>5}  {'TP':>5}  STATUS")
        print(f"  {'─'*78}")
        for o in queued:
            oid    = o.get("_order_id", "?")[:36]
            ticker = o.get("_ticker", "???")
            bp     = o.get("broker_payload", {})
            side   = bp.get("side", bp.get("Side", "?")).upper()[:4]
            size   = f"{o['position_size']:.2%}"
            sl     = f"{o['stop_loss_pct']:.1%}"
            tp     = f"{o['take_profit_pct']:.1%}"
            status = o["execution_status"]
            colour = GREEN if "QUEUED" in status else YELLOW
            print(f"  {colour}{oid:<38} {ticker:<8} {side:<5} {size:>6}  {sl:>5}  {tp:>5}  {status}{RESET}")

    print(f"{PURPLE}{'═'*80}{RESET}\n")


# ──────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

def _make_routed(
    ticker: str,
    signal_type: str,
    confidence: float,
    channel: str = "execution_queue",
    exec_candidate: bool = True,
    requires_human: bool = False,
    verdict: str = "VERIFIED",
    signal_strength: str = "CRITICAL",
) -> dict:
    """Build a minimal routed_article dict as alert_router.py would produce."""
    return {
        "delivery_channel":             channel,
        "alert_priority":               "CRITICAL",
        "requires_human_confirmation":  requires_human,
        "execution_candidate":          exec_candidate,
        "router_reason":                "smoke-test fixture",
        "formatted_alert": {
            "raw_signal": {
                "ticker":           ticker,
                "signal_type":      signal_type,
                "signal_strength":  signal_strength,
                "confidence_score": confidence,
                "verdict":          verdict,
                "title":            f"Smoke test: {ticker}",
                "source":           "smoke_test",
                "url":              "https://smoke.test/",
            }
        },
    }


def _smoke_test() -> None:
    GREEN  = "\033[1;32m"
    RED    = "\033[1;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[1;36m"
    PURPLE = "\033[1;35m"
    RESET  = "\033[0m"

    print(f"\n{PURPLE}{'▓'*80}")
    print("  🧪  EXECUTION BRIDGE — SMOKE TEST SUITE")
    print(f"{'▓'*80}{RESET}\n")

    # Reset singletons between runs
    _COOLDOWN_REGISTRY.clear()
    _POSITION_LEDGER.clear()
    enable_paper_trading()
    set_portfolio_value(100_000)

    tests: list[tuple[str, dict | list, ExecutionStatus, str | None]] = [
        # (label, input, expected_status, optional_note)

        # ── Happy paths ───────────────────────────────────────────────────────
        ("AGGRESSIVE BUY SPY — MACRO SL/TP (conf=0.95)",
         _make_routed("SPY",  "BUY",  0.95),
         ExecutionStatus.PAPER_QUEUED, "SL=1.5% TP=3.0% (MACRO)"),

        ("STANDARD SELL NVDA — EQUITY SL/TP (conf=0.83)",
         _make_routed("NVDA", "SELL", 0.83),
         ExecutionStatus.PAPER_QUEUED, "SL=3.0% TP=6.0% (EQUITY)"),

        ("AGGRESSIVE BUY BTC — CRYPTO SL/TP (conf=0.91)",
         _make_routed("BTC",  "BUY",  0.91),
         ExecutionStatus.PAPER_QUEUED, "SL=6.0% TP=12.0% (CRYPTO)"),

        ("AGGRESSIVE BUY EURUSD — FOREX SL/TP (conf=0.93)",
         _make_routed("EURUSD", "BUY", 0.93),
         ExecutionStatus.PAPER_QUEUED, "SL=1.0% TP=2.0% (FOREX)"),

        ("STANDARD BUY TSLA — EQUITY (conf=0.82)",
         _make_routed("TSLA", "BUY",  0.82),
         ExecutionStatus.PAPER_QUEUED, None),

        # ── Gate: confidence too low ──────────────────────────────────────────
        ("REJECT — conf=0.75 below 0.80 floor",
         _make_routed("AMZN", "BUY",  0.75),
         ExecutionStatus.REJECTED, None),

        # ── Gate: wrong delivery channel ─────────────────────────────────────
        ("REJECT — channel=telegram (not execution_queue)",
         _make_routed("AAPL", "BUY",  0.92, channel="telegram"),
         ExecutionStatus.REJECTED, None),

        # ── Gate: execution_candidate=False ──────────────────────────────────
        ("REJECT — execution_candidate=False",
         _make_routed("META", "BUY",  0.92, exec_candidate=False),
         ExecutionStatus.REJECTED, None),

        # ── Gate: requires_human_confirmation ────────────────────────────────
        ("REJECT — requires_human_confirmation=True",
         _make_routed("GOOG", "BUY",  0.92, requires_human=True),
         ExecutionStatus.REJECTED, None),

        # ── Gate: cooldown (re-use BTC which we already traded) ──────────────
        ("COOLDOWN — BTC re-submitted immediately",
         _make_routed("BTC",  "BUY",  0.93),
         ExecutionStatus.COOLDOWN, "BTC cooldown from earlier test"),

        # ── Gate: max positions (5) — fill 5, then try a 6th ─────────────────
        # At this point ledger already has: SPY NVDA BTC EURUSD TSLA = 5 positions
        ("CAP_EXCEEDED — 6th position (GLD) blocked",
         _make_routed("GLD",  "BUY",  0.91),
         ExecutionStatus.CAP_EXCEEDED, None),

        # ── Malformed / safe-skip ─────────────────────────────────────────────
        ("SKIP — non-dict entry",
         "not-a-dict",   # type: ignore[arg-type]
         None, "should be silently skipped"),

        ("SKIP — missing required fields",
         {"delivery_channel": "execution_queue"},
         None, "should be silently skipped"),
    ]

    passed = 0
    failed = 0

    for label, fixture, expected_status, note in tests:
        is_skip = expected_status is None
        inp     = fixture if isinstance(fixture, list) else [fixture]

        results = build_execution_orders(inp)  # type: ignore[arg-type]

        if is_skip:
            ok = len(results) == 0
        else:
            ok = len(results) == 1 and results[0]["execution_status"] == expected_status

        status_str = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{status_str}]  {label}")

        if not ok and not is_skip:
            got = results[0]["execution_status"] if results else "no result"
            print(f"          {RED}↳ expected={expected_status}  got={got}{RESET}")
        elif ok and not is_skip and results:
            detail_parts = [f"status={results[0]['execution_status']}"]
            if results[0].get("position_size"):
                detail_parts.append(f"size={results[0]['position_size']:.2%}")
                detail_parts.append(f"SL={results[0]['stop_loss_pct']:.1%}")
                detail_parts.append(f"TP={results[0]['take_profit_pct']:.1%}")
            if note:
                detail_parts.append(note)
            print(f"          {CYAN}↳ {' | '.join(detail_parts)}{RESET}")
        elif is_skip and ok:
            print(f"          {CYAN}↳ correctly skipped{RESET}")

        if ok:
            passed += 1
        else:
            failed += 1

    # ── Plugin test ────────────────────────────────────────────────────────────
    print(f"\n  {YELLOW}── Plugin Registry Test ────────────────────────────{RESET}")
    _COOLDOWN_REGISTRY.clear()
    _POSITION_LEDGER.clear()
    plugin_calls = []

    def _mock_plugin(order: dict) -> None:
        plugin_calls.append(order["execution_status"])

    register_plugin(_mock_plugin)
    build_execution_orders([_make_routed("MSFT", "BUY", 0.93)])
    ok = len(plugin_calls) == 1 and plugin_calls[0] == ExecutionStatus.PAPER_QUEUED
    print(f"  [{'PASS' if ok else 'FAIL'}]  Plugin fired — status: {plugin_calls}")

    if ok:
        passed += 1
    else:
        failed += 1

    # ── Adapter switching test ─────────────────────────────────────────────────
    print(f"\n  {YELLOW}── Adapter Switching Test ──────────────────────────{RESET}")
    _COOLDOWN_REGISTRY.clear()
    _POSITION_LEDGER.clear()
    enable_paper_trading()

    for adapter in (BrokerAdapter.ALPACA, BrokerAdapter.IBKR, BrokerAdapter.BINANCE):
        set_broker_adapter(adapter)
        # paper mode always uses PaperAdapter, so test live mode briefly
        enable_live_trading()
        results = build_execution_orders([_make_routed("ETH", "BUY", 0.93)])
        enable_paper_trading()
        _COOLDOWN_REGISTRY.clear()
        _POSITION_LEDGER.clear()
        if results and results[0].get("broker_payload", {}).get("_adapter") == adapter:
            print(f"  [PASS]  Adapter '{adapter}' payload built correctly")
            passed += 1
        else:
            got = results[0].get("broker_payload", {}).get("_adapter") if results else "none"
            print(f"  [FAIL]  Adapter '{adapter}' — got '{got}'")
            failed += 1

    set_broker_adapter(BrokerAdapter.ALPACA)

    # ── Print summary ──────────────────────────────────────────────────────────
    _COOLDOWN_REGISTRY.clear()
    _POSITION_LEDGER.clear()
    _PLUGIN_REGISTRY.clear()

    total = passed + failed
    print(f"\n{PURPLE}{'═'*80}")
    print(f"  🏁  SMOKE TEST RESULTS — {passed}/{total} passed   "
          f"{'✅ ALL CLEAR' if failed == 0 else '❌ FAILURES DETECTED'}")
    print(f"{'═'*80}{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _startup_banner() -> None:
    print("""
\033[1;32m╔══════════════════════════════════════════════════════════════════════════╗
║        ⚡  MONSTER TRADING AI — EXECUTION BRIDGE  v1.0.0              ║
║           Pipeline Stage 6 — Broker Order Construction Engine         ║
╚══════════════════════════════════════════════════════════════════════════╝\033[0m
""")


if __name__ == "__main__":
    if "--smoke" in sys.argv or "-s" in sys.argv:
        _smoke_test()
    else:
        _startup_banner()
        print("Usage:")
        print("  python execution_bridge.py --smoke     # run smoke test")
        print()
        print("API usage:")
        print("  from execution_bridge import build_execution_orders")
        print("  orders = build_execution_orders(routed_articles)")
        print()
        print("Configuration:")
        print("  from execution_bridge import (")
        print("      enable_live_trading, set_broker_adapter, set_portfolio_value,")
        print("      register_plugin, print_execution_summary")
        print("  )")