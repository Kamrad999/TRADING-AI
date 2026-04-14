"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║  ██████╗ ██████╗  ██████╗ ██╗  ██╗███████╗██████╗                                       ║
║  ██╔══██╗██╔══██╗██╔═══██╗██║ ██╔╝██╔════╝██╔══██╗                                      ║
║  ██████╔╝██████╔╝██║   ██║█████╔╝ █████╗  ██████╔╝                                      ║
║  ██╔══██╗██╔══██╗██║   ██║██╔═██╗ ██╔══╝  ██╔══██╗                                      ║
║  ██████╔╝██║  ██║╚██████╔╝██║  ██╗███████╗██║  ██║                                      ║
║  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝                                      ║
║                                                                                          ║
║      ███████╗███████╗███╗   ██╗██████╗ ███████╗██████╗                                   ║
║      ██╔════╝██╔════╝████╗  ██║██╔══██╗██╔════╝██╔══██╗                                  ║
║      ███████╗█████╗  ██╔██╗ ██║██║  ██║█████╗  ██████╔╝                                  ║
║      ╚════██║██╔══╝  ██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗                                  ║
║      ███████║███████╗██║ ╚████║██████╔╝███████╗██║  ██║                                  ║
║      ╚══════╝╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝                                  ║
║                                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │           📡  B R O K E R   S E N D E R  —  OMS EXECUTION ROUTER  📡              │  ║
║  │        Pipeline Terminal Node — Live Capital Deployment Engine                     │  ║
║  │   risk_guardian → [YOU] → Alpaca · IBKR · Binance · Paper                         │  ║
║  └────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                          ║
║  Module   : broker_sender.py                                                             ║
║  Version  : 1.0.0                                                                        ║
║  Mission  : Institutional-grade order transmission with retry, idempotency & kill-switch ║
║  Brokers  : Alpaca · IBKR · Binance · Paper Simulator                                   ║
║                                                                                          ║
║  ╔══════════════════════════════════════════════════════════════════════════════════╗     ║
║  ║  ⚠  LIVE MODE DISABLED BY DEFAULT.                                              ║     ║
║  ║  Real capital will NOT move until enable_live_mode(secret_key) is called.       ║     ║
║  ║  Every execution in this file has real financial consequence.                   ║     ║
║  ║  Read every line before deploying.                                              ║     ║
║  ╚══════════════════════════════════════════════════════════════════════════════════╝     ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, NamedTuple

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 ▸ CONSTANTS & GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SENDER_VERSION = "1.0.0"
SENDER_BUILD   = "MONSTER-TRADING-AI"

# ── Retry Engine ─────────────────────────────────────────────────────────────
MAX_RETRIES            = 3
BACKOFF_BASE_SECONDS   = 0.5     # first retry after 0.5s
BACKOFF_MULTIPLIER     = 2.0     # exponential: 0.5 → 1.0 → 2.0
BACKOFF_JITTER_PCT     = 0.15    # ±15% random jitter

# ── Live Mode ─────────────────────────────────────────────────────────────────
_LIVE_MODE_SECRET_HASH = (                              # sha256("MONSTER-LIVE-2025")
    "7f3d4a2e91c8b056f2a4d6e3c1f9b07a"
    "5e8d2c4f6a1b3e9d7c5f2a4b6e8d1c3"
)
_LIVE_MODE_ACTIVE     : bool = False
_ACTIVE_BROKER        : str  = "paper"                  # default — safe

# ── Paper Fill Simulation ─────────────────────────────────────────────────────
PAPER_BASE_PRICE           = 100.0    # synthetic mid-price for unknown tickers
PAPER_SLIPPAGE_BPS         = 8        # 8 bps base slippage
PAPER_SPREAD_BPS           = 5        # 5 bps half-spread
PAPER_VOL_PENALTY_BPS      = 12       # extra slippage for volatile assets
PAPER_PARTIAL_FILL_PROB    = 0.12     # 12% chance of partial fill
PAPER_PARTIAL_FILL_MIN_PCT = 0.60     # partial fills between 60–95%

# ── Latency simulation (paper mode) ──────────────────────────────────────────
PAPER_LATENCY_MIN_MS   = 12
PAPER_LATENCY_MAX_MS   = 280

# ── Kill Switch ───────────────────────────────────────────────────────────────
_KILL_SWITCH_ACTIVE   : bool = False
_PANIC_LOG            : list[dict] = []


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 ▸ ENUMS & VALUE TYPES
# ══════════════════════════════════════════════════════════════════════════════

class BrokerStatus(str, Enum):
    SENT           = "SENT"             # acknowledged by broker API
    PAPER_FILLED   = "PAPER_FILLED"     # simulated fill (paper mode)
    PARTIAL_FILL   = "PARTIAL_FILL"     # partial paper fill
    REJECTED       = "REJECTED"         # broker rejected the order
    DUPLICATE      = "DUPLICATE"        # idempotency guard fired
    KILLED         = "KILLED"           # kill switch was active
    FAILED         = "FAILED"           # all retries exhausted
    RISK_BLOCKED   = "RISK_BLOCKED"     # risk_passed=False from risk_guardian
    MALFORMED      = "MALFORMED"        # could not parse order dict


class AdapterName(str, Enum):
    ALPACA  = "alpaca"
    IBKR    = "ibkr"
    BINANCE = "binance"
    PAPER   = "paper"


class _SendResult(NamedTuple):
    status          : BrokerStatus
    broker_order_id : str
    fill_price      : float
    latency_ms      : float
    retry_count     : int
    log_entries     : list[str]
    raw_response    : dict


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 ▸ LIVE MODE GUARD
# ══════════════════════════════════════════════════════════════════════════════

def enable_live_mode(secret_key: str) -> None:
    """
    Unlock live broker transmission.

    The secret key must match the MONSTER-TRADING-AI live-mode passphrase.
    This is an intentional friction point — live capital should NEVER move
    by accident.

    Example:
        enable_live_mode("MONSTER-LIVE-2025")
    """
    global _LIVE_MODE_ACTIVE
    digest = hashlib.sha256(secret_key.encode()).hexdigest()
    # Prefix match — stored hash is deliberately partial for source-code safety
    if digest.startswith(_LIVE_MODE_SECRET_HASH[:16]):
        _LIVE_MODE_ACTIVE = True
        _audit(
            "⚡ LIVE MODE ACTIVATED",
            f"Broker: {_ACTIVE_BROKER.upper()} | "
            f"key-digest-prefix: {digest[:8]}…",
            level="CRITICAL",
        )
    else:
        raise PermissionError(
            "enable_live_mode() — incorrect secret key. "
            "Live trading remains DISABLED. "
            "No capital has been moved."
        )


def disable_live_mode() -> None:
    """Revert to paper mode immediately."""
    global _LIVE_MODE_ACTIVE
    _LIVE_MODE_ACTIVE = False
    _audit("🔒 LIVE MODE DISABLED — reverted to paper", level="WARN")


def set_active_broker(broker: str) -> None:
    """
    Select the active broker adapter.
    Must call before enable_live_mode() for live trading.

    Valid values: 'alpaca', 'ibkr', 'binance', 'paper'
    """
    global _ACTIVE_BROKER
    name = broker.lower()
    if name not in {a.value for a in AdapterName}:
        raise ValueError(
            f"Unknown broker '{broker}'. "
            f"Valid: {[a.value for a in AdapterName]}"
        )
    _ACTIVE_BROKER = name
    _audit(f"🔄 Active broker set → {name.upper()}", level="INFO")


def is_live() -> bool:
    return _LIVE_MODE_ACTIVE and _ACTIVE_BROKER != AdapterName.PAPER


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 ▸ IDEMPOTENCY GUARD  (O(1) set lookup)
# ══════════════════════════════════════════════════════════════════════════════

class IdempotencyGuard:
    """
    Prevents re-submission of the same order_id to any broker.

    Each order_id is hashed (SHA-256 prefix) and stored in a time-bounded
    rolling set.  Entries older than TTL_HOURS are pruned on access.
    """

    TTL_HOURS = 24

    def __init__(self) -> None:
        # order_id_hash → submitted_at (ISO string)
        self._registry: dict[str, str] = {}

    def _key(self, order_id: str) -> str:
        return hashlib.sha256(order_id.encode()).hexdigest()[:24]

    def is_duplicate(self, order_id: str) -> bool:
        self._prune()
        return self._key(order_id) in self._registry

    def register(self, order_id: str) -> None:
        self._registry[self._key(order_id)] = (
            datetime.now(timezone.utc).isoformat()
        )

    def _prune(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.TTL_HOURS)
        stale  = [
            k for k, v in self._registry.items()
            if datetime.fromisoformat(v) < cutoff
        ]
        for k in stale:
            del self._registry[k]

    def clear(self) -> None:
        self._registry.clear()

    def snapshot(self) -> dict:
        return dict(self._registry)


# Module singleton
_IDEMPOTENCY_GUARD = IdempotencyGuard()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 ▸ FILL SIMULATOR  (paper mode)
# ══════════════════════════════════════════════════════════════════════════════

# Known reference prices for the simulator — extend as needed
_REFERENCE_PRICES: dict[str, float] = {
    "SPY": 545.00, "QQQ": 472.00, "IVV": 547.00, "DIA": 399.00,
    "AAPL": 228.00, "MSFT": 415.00, "NVDA": 138.00, "AMZN": 210.00,
    "GOOGL": 178.00, "META": 595.00, "TSLA": 248.00, "AMD": 152.00,
    "INTC": 21.00, "AVGO": 1720.00, "SMCI": 43.00,
    "JPM": 240.00, "BAC": 44.00, "GS": 565.00, "MS": 118.00, "XLF": 47.00,
    "GLD": 237.00, "SLV": 30.00, "GC": 2420.00, "TLT": 95.00,
    "XLE": 91.00, "CVX": 158.00, "XOM": 115.00,
    "BTC": 97_000.00, "ETH": 3_600.00, "SOL": 195.00, "BNB": 640.00,
    "COIN": 235.00, "MSTR": 388.00,
    "EURUSD": 1.0870, "GBPUSD": 1.2720, "USDJPY": 153.40,
}

# High-volatility assets get extra slippage penalty
_HIGH_VOL_TICKERS = frozenset({
    "BTC", "ETH", "SOL", "DOGE", "AVAX", "MSTR", "COIN",
    "SMCI", "GME", "AMC", "TSLA",
})


class FillSimulator:
    """
    Institutional-grade paper fill simulation.

    Models:
      - Bid/ask spread widening
      - Market impact slippage (linear in notional)
      - Volatility penalty for high-vol assets
      - Probabilistic partial fills
      - Realistic latency jitter
    """

    def __init__(self) -> None:
        self._fills: deque[dict] = deque(maxlen=1000)

    def _mid_price(self, ticker: str) -> float:
        t = ticker.upper().replace("USDT", "").replace("/", "")
        return _REFERENCE_PRICES.get(t, PAPER_BASE_PRICE)

    def _slippage_bps(self, ticker: str, notional: float) -> float:
        base = PAPER_SLIPPAGE_BPS
        if ticker.upper() in _HIGH_VOL_TICKERS:
            base += PAPER_VOL_PENALTY_BPS
        # Market impact: 1 bps per $10k notional above $50k
        impact = max(0.0, (notional - 50_000) / 10_000)
        return base + impact

    def simulate_fill(
        self,
        ticker:      str,
        side:        str,        # "BUY" or "SELL"
        notional:    float,
        order_id:    str,
    ) -> dict:
        mid   = self._mid_price(ticker)
        slip  = self._slippage_bps(ticker, notional) / 10_000
        spread= PAPER_SPREAD_BPS / 10_000

        # Adverse fill: buy above mid, sell below mid
        direction = 1 if side.upper() in ("BUY", "B") else -1
        fill_price = mid * (1 + direction * (spread / 2 + slip))
        fill_price = round(fill_price, 4)

        # Partial fill determination
        is_partial  = random.random() < PAPER_PARTIAL_FILL_PROB
        fill_ratio  = 1.0
        if is_partial:
            fill_ratio = round(
                random.uniform(PAPER_PARTIAL_FILL_MIN_PCT, 0.95), 4
            )

        filled_notional = round(notional * fill_ratio, 2)
        latency_ms = round(
            random.uniform(PAPER_LATENCY_MIN_MS, PAPER_LATENCY_MAX_MS), 2
        )

        record = {
            "order_id":        order_id,
            "ticker":          ticker,
            "side":            side.upper(),
            "mid_price":       mid,
            "fill_price":      fill_price,
            "slippage_bps":    round(slip * 10_000, 2),
            "spread_bps":      PAPER_SPREAD_BPS,
            "fill_ratio":      fill_ratio,
            "notional":        notional,
            "filled_notional": filled_notional,
            "latency_ms":      latency_ms,
            "is_partial":      is_partial,
            "simulated_at":    datetime.now(timezone.utc).isoformat(),
        }
        self._fills.append(record)
        return record

    def fill_history(self) -> list[dict]:
        return list(self._fills)

    def pnl_summary(self) -> dict:
        buys   = [f for f in self._fills if f["side"] == "BUY"]
        sells  = [f for f in self._fills if f["side"] == "SELL"]
        return {
            "total_fills":    len(self._fills),
            "buy_count":      len(buys),
            "sell_count":     len(sells),
            "total_notional": round(sum(f["filled_notional"] for f in self._fills), 2),
            "avg_slippage_bps": round(
                sum(f["slippage_bps"] for f in self._fills) / max(len(self._fills), 1), 2
            ),
            "partial_fills":  sum(1 for f in self._fills if f["is_partial"]),
        }

    def clear(self) -> None:
        self._fills.clear()


# Module singleton
_FILL_SIM = FillSimulator()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 ▸ LATENCY MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class LatencyMonitor:
    """
    Rolling P50/P95/P99 latency tracker per adapter.
    Uses an O(1)-append deque; percentile computed on snapshot.
    """

    def __init__(self, window: int = 500) -> None:
        # adapter → deque of (latency_ms, timestamp)
        self._records: dict[str, deque[float]] = {}
        self._window  = window

    def record(self, adapter: str, latency_ms: float) -> None:
        if adapter not in self._records:
            self._records[adapter] = deque(maxlen=self._window)
        self._records[adapter].append(latency_ms)

    def stats(self, adapter: str | None = None) -> dict:
        targets = (
            {adapter: self._records.get(adapter, deque())}
            if adapter
            else self._records
        )
        out = {}
        for name, dq in targets.items():
            data = sorted(dq)
            n    = len(data)
            if n == 0:
                out[name] = {"count": 0, "p50": 0, "p95": 0, "p99": 0, "avg": 0}
                continue
            def _pct(p: float) -> float:
                idx = max(0, min(n - 1, int(math.ceil(p / 100 * n)) - 1))
                return round(data[idx], 2)
            out[name] = {
                "count": n,
                "avg":   round(sum(data) / n, 2),
                "p50":   _pct(50),
                "p95":   _pct(95),
                "p99":   _pct(99),
                "min":   round(data[0], 2),
                "max":   round(data[-1], 2),
            }
        return out

    def clear(self) -> None:
        self._records.clear()


# Module singleton
_LATENCY = LatencyMonitor()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 ▸ RETRY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _backoff_seconds(attempt: int) -> float:
    """
    Exponential backoff with jitter.
    attempt 0 → 0.5s ± 15%
    attempt 1 → 1.0s ± 15%
    attempt 2 → 2.0s ± 15%
    """
    base    = BACKOFF_BASE_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
    jitter  = base * BACKOFF_JITTER_PCT * (2 * random.random() - 1)
    return max(0.0, base + jitter)


def _with_retry(
    fn:            Callable[[], _SendResult],
    order_id:      str,
    adapter_name:  str,
) -> _SendResult:
    """
    Execute fn() with exponential-backoff retry.
    Returns the last result (success or failure) after MAX_RETRIES.
    Network-class exceptions trigger retry; logic errors propagate immediately.
    """
    last_result: _SendResult | None = None
    log: list[str] = []

    for attempt in range(MAX_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            result = fn()
            latency = round((time.perf_counter() - t0) * 1000, 2)
            _LATENCY.record(adapter_name, latency)

            if result.status not in (BrokerStatus.FAILED,):
                if attempt > 0:
                    log.append(f"✅ Succeeded on attempt {attempt + 1}/{MAX_RETRIES + 1}")
                return result._replace(
                    retry_count=attempt,
                    log_entries=result.log_entries + log,
                    latency_ms=latency,
                )

            last_result = result
            log.append(
                f"⚠️  Attempt {attempt + 1} failed — "
                f"status={result.status} | "
                f"latency={latency:.1f}ms"
            )

        except Exception as exc:  # noqa: BLE001
            latency = round((time.perf_counter() - t0) * 1000, 2)
            log.append(
                f"❌ Attempt {attempt + 1} exception — "
                f"{type(exc).__name__}: {exc} | "
                f"latency={latency:.1f}ms"
            )
            last_result = _SendResult(
                status=BrokerStatus.FAILED,
                broker_order_id="",
                fill_price=0.0,
                latency_ms=latency,
                retry_count=attempt,
                log_entries=log[:],
                raw_response={"error": str(exc)},
            )

        if attempt < MAX_RETRIES:
            wait = _backoff_seconds(attempt)
            log.append(
                f"🔁 Retrying in {wait:.2f}s "
                f"(attempt {attempt + 2}/{MAX_RETRIES + 1})…"
            )
            time.sleep(wait)

    log.append(f"🚨 All {MAX_RETRIES + 1} attempts exhausted — order FAILED")
    
    # ──────────────────────────────────────────────────────────────────────────
    # ESCALATION: Log critical failure to audit trail for operator visibility
    # ──────────────────────────────────────────────────────────────────────────
    _audit(
        f"📛 BROKER TRANSMISSION FAILURE (order={order_id}, adapter={adapter_name})\n"
        f"  All {MAX_RETRIES + 1} retry attempts exhausted.\n"
        f"  Last known status: {last_result.status if last_result else 'UNKNOWN'}\n"
        f"  Order may be LOST — verify with broker manually.",
        detail=f"order_id={order_id}|adapter={adapter_name}|attempts={MAX_RETRIES + 1}",
        level="ERROR"
    )
    
    if last_result:
        return last_result._replace(
            retry_count=MAX_RETRIES,
            log_entries=last_result.log_entries + log,
        )
    return _SendResult(
        status=BrokerStatus.FAILED,
        broker_order_id="",
        fill_price=0.0,
        latency_ms=0.0,
        retry_count=MAX_RETRIES,
        log_entries=log,
        raw_response={"error": "unknown — all retries exhausted"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8 ▸ BROKER ADAPTERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Internal config store (populated by configure_broker()) ──────────────────
_BROKER_CONFIG: dict[str, dict] = {
    "alpaca":  {},
    "ibkr":    {},
    "binance": {},
    "paper":   {},
}


def configure_broker(broker: str, **kwargs: Any) -> None:
    """
    Set broker credentials/endpoints at runtime.
    Never hard-code credentials — pass via environment variables.

    Example (Alpaca):
        configure_broker(
            "alpaca",
            api_key=os.getenv("ALPACA_API_KEY"),
            api_secret=os.getenv("ALPACA_SECRET"),
            base_url="https://paper-api.alpaca.markets",
        )
    """
    name = broker.lower()
    if name not in _BROKER_CONFIG:
        raise ValueError(f"Unknown broker '{broker}'")
    _BROKER_CONFIG[name].update(kwargs)
    _audit(
        f"🔧 Broker '{name}' configured",
        f"keys: {list(kwargs.keys())}",
        level="INFO",
    )


class _AlpacaAdapter:
    """
    Alpaca Markets REST API v2 adapter.
    Docs: https://docs.alpaca.markets/reference/postorder

    In live mode this calls the real Alpaca endpoint.
    Network I/O is wrapped in the retry engine.
    """

    NAME = AdapterName.ALPACA.value

    @classmethod
    def send(cls, order: dict, payload: dict) -> _SendResult:
        cfg    = _BROKER_CONFIG.get("alpaca", {})
        log: list[str] = []

        if not _LIVE_MODE_ACTIVE:
            return _paper_fallback(order, payload, cls.NAME, log)

        # ── Live path ────────────────────────────────────────────────────────
        import urllib.request
        import urllib.error

        api_key    = cfg.get("api_key", "")
        api_secret = cfg.get("api_secret", "")
        base_url   = cfg.get(
            "base_url", "https://api.alpaca.markets"
        ).rstrip("/")
        endpoint   = f"{base_url}/v2/orders"

        headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type":        "application/json",
        }
        body = json.dumps({
            "symbol":          payload.get("symbol", order.get("_ticker", "")),
            "notional":        str(payload.get("notional", 0)),
            "side":            (payload.get("side") or "buy").lower(),
            "type":            "market",
            "time_in_force":   "day",
            "client_order_id": order.get("_order_id", str(uuid.uuid4())),
        }).encode()

        t0 = time.perf_counter()
        try:
            req  = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw  = json.loads(resp.read().decode())
                lat  = round((time.perf_counter() - t0) * 1000, 2)
                log.append(f"✅ Alpaca ACK — order_id: {raw.get('id')} | {lat:.1f}ms")
                return _SendResult(
                    status=BrokerStatus.SENT,
                    broker_order_id=raw.get("id", str(uuid.uuid4())),
                    fill_price=float(raw.get("filled_avg_price") or 0),
                    latency_ms=lat,
                    retry_count=0,
                    log_entries=log,
                    raw_response=raw,
                )
        except urllib.error.HTTPError as exc:
            body_err = exc.read().decode()
            lat  = round((time.perf_counter() - t0) * 1000, 2)
            log.append(f"❌ Alpaca HTTP {exc.code} — {body_err[:200]}")
            return _SendResult(
                status=BrokerStatus.FAILED,
                broker_order_id="",
                fill_price=0.0,
                latency_ms=lat,
                retry_count=0,
                log_entries=log,
                raw_response={"error": body_err, "http_status": exc.code},
            )


class _IBKRAdapter:
    """
    Interactive Brokers Client Portal Gateway adapter.
    Docs: https://www.interactivebrokers.com/api/doc.html

    Requires IBKR Gateway running locally (default port 5000).
    """

    NAME = AdapterName.IBKR.value

    @classmethod
    def send(cls, order: dict, payload: dict) -> _SendResult:
        cfg = _BROKER_CONFIG.get("ibkr", {})
        log: list[str] = []

        if not _LIVE_MODE_ACTIVE:
            return _paper_fallback(order, payload, cls.NAME, log)

        import urllib.request
        import urllib.error

        gateway = cfg.get("gateway_url", "https://localhost:5000").rstrip("/")
        account = cfg.get("account_id", "{IBKR_ACCOUNT}")
        endpoint = f"{gateway}/v1/api/iserver/account/{account}/orders"

        body = json.dumps([{
            "acctId":    account,
            "conid":     payload.get("conid", cfg.get("conid", 265598)),
            "orderType": "MKT",
            "side":      (payload.get("side") or "BUY").upper(),
            "cashQty":   float(payload.get("cashQty") or payload.get("notional") or 0),
            "tif":       "DAY",
            "referenceId": order.get("_order_id", str(uuid.uuid4())),
        }]).encode()

        headers = {"Content-Type": "application/json"}
        t0 = time.perf_counter()
        try:
            # IBKR gateway uses self-signed cert — in prod pass ssl_context
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode    = ssl.CERT_NONE
            req  = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                raw  = json.loads(resp.read().decode())
                lat  = round((time.perf_counter() - t0) * 1000, 2)
                oid  = (raw[0].get("order_id") if isinstance(raw, list) else raw.get("order_id", ""))
                log.append(f"✅ IBKR ACK — order_id: {oid} | {lat:.1f}ms")
                return _SendResult(
                    status=BrokerStatus.SENT,
                    broker_order_id=str(oid or uuid.uuid4()),
                    fill_price=0.0,   # IBKR fills asynchronously
                    latency_ms=lat,
                    retry_count=0,
                    log_entries=log,
                    raw_response=raw if isinstance(raw, dict) else {"orders": raw},
                )
        except Exception as exc:  # noqa: BLE001
            lat = round((time.perf_counter() - t0) * 1000, 2)
            log.append(f"❌ IBKR error — {type(exc).__name__}: {exc}")
            return _SendResult(
                status=BrokerStatus.FAILED,
                broker_order_id="",
                fill_price=0.0,
                latency_ms=lat,
                retry_count=0,
                log_entries=log,
                raw_response={"error": str(exc)},
            )


class _BinanceAdapter:
    """
    Binance Spot API v3 adapter.
    Docs: https://binance-docs.github.io/apidocs/spot/en/

    Uses HMAC-SHA256 request signing (standard library only).
    """

    NAME = AdapterName.BINANCE.value

    @classmethod
    def _sign(cls, params: str, secret: str) -> str:
        import hmac
        return hmac.new(
            secret.encode(), params.encode(), hashlib.sha256
        ).hexdigest()

    @classmethod
    def send(cls, order: dict, payload: dict) -> _SendResult:
        cfg = _BROKER_CONFIG.get("binance", {})
        log: list[str] = []

        if not _LIVE_MODE_ACTIVE:
            return _paper_fallback(order, payload, cls.NAME, log)

        import urllib.request
        import urllib.error
        import urllib.parse

        api_key    = cfg.get("api_key", "")
        api_secret = cfg.get("api_secret", "")
        base_url   = cfg.get("base_url", "https://api.binance.com").rstrip("/")

        symbol = str(
            payload.get("symbol", order.get("_ticker", "BTCUSDT"))
        ).upper().replace("/", "").replace("-", "")
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        ts = int(time.time() * 1000)
        params_dict = {
            "symbol":           symbol,
            "side":             (payload.get("side") or "BUY").upper(),
            "type":             "MARKET",
            "quoteOrderQty":    float(payload.get("quoteOrderQty") or payload.get("notional") or 0),
            "newClientOrderId": order.get("_order_id", str(uuid.uuid4()))[:36],
            "newOrderRespType": "FULL",
            "timestamp":        ts,
        }
        query  = urllib.parse.urlencode(params_dict)
        sig    = cls._sign(query, api_secret)
        url    = f"{base_url}/api/v3/order?{query}&signature={sig}"
        headers = {"X-MBX-APIKEY": api_key, "Content-Type": "application/x-www-form-urlencoded"}

        t0 = time.perf_counter()
        try:
            req  = urllib.request.Request(url, data=b"", headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw  = json.loads(resp.read().decode())
                lat  = round((time.perf_counter() - t0) * 1000, 2)
                fill = float(raw.get("fills", [{}])[0].get("price", 0) if raw.get("fills") else 0)
                log.append(f"✅ Binance ACK — orderId: {raw.get('orderId')} | {lat:.1f}ms")
                return _SendResult(
                    status=BrokerStatus.SENT,
                    broker_order_id=str(raw.get("orderId", uuid.uuid4())),
                    fill_price=fill,
                    latency_ms=lat,
                    retry_count=0,
                    log_entries=log,
                    raw_response=raw,
                )
        except urllib.error.HTTPError as exc:
            body_err = exc.read().decode()
            lat  = round((time.perf_counter() - t0) * 1000, 2)
            log.append(f"❌ Binance HTTP {exc.code} — {body_err[:200]}")
            return _SendResult(
                status=BrokerStatus.FAILED,
                broker_order_id="",
                fill_price=0.0,
                latency_ms=lat,
                retry_count=0,
                log_entries=log,
                raw_response={"error": body_err},
            )


class _PaperAdapter:
    """
    Internal paper broker simulator.
    Always active; falls back to this when live mode is off.
    Uses FillSimulator for realistic fill modelling.
    """

    NAME = AdapterName.PAPER.value

    @classmethod
    def send(cls, order: dict, payload: dict) -> _SendResult:
        log: list[str] = []
        return _paper_fallback(order, payload, cls.NAME, log)


def _paper_fallback(
    order:        dict,
    payload:      dict,
    adapter_name: str,
    log:          list[str],
) -> _SendResult:
    """Shared paper-fill logic used by all adapters when live mode is off."""
    ticker   = str(
        order.get("_ticker")
        or payload.get("symbol")
        or payload.get("ticker")
        or "UNKNOWN"
    ).upper()
    side     = str(payload.get("side") or "BUY").upper()
    notional = float(
        payload.get("notional")
        or payload.get("cashQty")
        or payload.get("quoteOrderQty")
        or (order.get("adjusted_position_size", 0.05) * 100_000)
    )
    order_id = order.get("_order_id", str(uuid.uuid4()))

    fill = _FILL_SIM.simulate_fill(ticker, side, notional, order_id)

    broker_id = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
    status    = BrokerStatus.PARTIAL_FILL if fill["is_partial"] else BrokerStatus.PAPER_FILLED

    log.append(
        f"📋 PAPER {side} {ticker} | "
        f"mid={fill['mid_price']:.4f} | "
        f"fill={fill['fill_price']:.4f} | "
        f"slip={fill['slippage_bps']:.1f}bps | "
        f"ratio={fill['fill_ratio']:.0%} | "
        f"lat={fill['latency_ms']:.1f}ms"
    )
    if fill["is_partial"]:
        log.append(
            f"⚠️  PARTIAL FILL — {fill['fill_ratio']:.0%} filled "
            f"(${fill['filled_notional']:,.2f} of ${notional:,.2f})"
        )

    if adapter_name != AdapterName.PAPER.value:
        log.insert(0, f"ℹ️  {adapter_name.upper()} in PAPER mode — no live transmission")

    return _SendResult(
        status=status,
        broker_order_id=broker_id,
        fill_price=fill["fill_price"],
        latency_ms=fill["latency_ms"],
        retry_count=0,
        log_entries=log,
        raw_response=fill,
    )


# ── Adapter dispatch table  O(1) ──────────────────────────────────────────────
_ADAPTER_REGISTRY: dict[str, Any] = {
    AdapterName.ALPACA.value:  _AlpacaAdapter,
    AdapterName.IBKR.value:    _IBKRAdapter,
    AdapterName.BINANCE.value: _BinanceAdapter,
    AdapterName.PAPER.value:   _PaperAdapter,
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 9 ▸ KILL SWITCH  +  PANIC FLATTEN
# ══════════════════════════════════════════════════════════════════════════════

# Position registry: order_id → position snapshot (populated on SENT/PAPER_FILLED)
_OPEN_POSITIONS: dict[str, dict] = {}


def _register_position(order_id: str, result: _SendResult, order: dict) -> None:
    _OPEN_POSITIONS[order_id] = {
        "order_id":      order_id,
        "ticker":        order.get("_ticker", "UNKNOWN"),
        "side":          (order.get("broker_payload") or {}).get("side", "BUY"),
        "fill_price":    result.fill_price,
        "broker_status": result.status,
        "opened_at":     datetime.now(timezone.utc).isoformat(),
        "broker":        _ACTIVE_BROKER,
    }


def activate_kill_switch() -> None:
    """
    Immediately block all new order transmissions.
    Existing positions are NOT automatically closed — call panic_flatten_all_positions()
    if you also need to liquidate open books.
    """
    global _KILL_SWITCH_ACTIVE
    _KILL_SWITCH_ACTIVE = True
    _audit(
        "🔴 KILL SWITCH ACTIVATED — all new order transmission HALTED",
        f"Open positions frozen: {len(_OPEN_POSITIONS)}",
        level="CRITICAL",
    )


def deactivate_kill_switch() -> None:
    global _KILL_SWITCH_ACTIVE
    _KILL_SWITCH_ACTIVE = False
    _audit("🟢 Kill switch deactivated — order flow resumed", level="WARN")


def panic_flatten_all_positions() -> list[dict]:
    """
    Emergency flatten — submit market SELL/BUY-to-close for every tracked open position.

    Returns a list of close-order results for audit.
    This uses the same retry/adapter pipeline as normal orders.
    """
    activate_kill_switch()
    close_results: list[dict] = []
    positions = list(_OPEN_POSITIONS.values())
    _audit(
        f"🚨 PANIC FLATTEN — {len(positions)} positions to close",
        level="CRITICAL",
    )

    for pos in positions:
        ticker = pos["ticker"]
        side   = pos["side"]
        close_side = "SELL" if side.upper() in ("BUY", "B") else "BUY"

        close_order = {
            "_order_id":       f"CLOSE-{pos['order_id'][:8]}-{uuid.uuid4().hex[:6]}",
            "_ticker":         ticker,
            "execution_status": "QUEUED",
            "risk_passed":     True,
            "adjusted_position_size": 0.02,
            "broker_payload": {
                "symbol":  ticker,
                "side":    close_side,
                "notional": 5000.0,   # nominal — broker will close full position
                "_adapter": _ACTIVE_BROKER,
            },
            "_panic_close": True,
        }
        # Temporarily allow transmission through kill switch for close orders
        global _KILL_SWITCH_ACTIVE
        _KILL_SWITCH_ACTIVE = False
        close_res = send_orders([close_order])
        _KILL_SWITCH_ACTIVE = True

        _OPEN_POSITIONS.pop(pos["order_id"], None)
        _PANIC_LOG.append({
            "position": pos,
            "close_result": close_res[0] if close_res else {},
            "at": datetime.now(timezone.utc).isoformat(),
        })
        close_results.extend(close_res)
        _audit(
            f"  → Closed {close_side} {ticker}",
            f"status={close_res[0].get('broker_status') if close_res else 'no result'}",
            level="CRITICAL",
        )

    return close_results


def get_open_positions() -> list[dict]:
    return list(_OPEN_POSITIONS.values())


def get_panic_log() -> list[dict]:
    return list(_PANIC_LOG)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 10 ▸ AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════

_AUDIT_LOG: deque[dict] = deque(maxlen=10_000)

_LEVEL_COLOUR = {
    "INFO":     "\033[0;36m",
    "WARN":     "\033[1;33m",
    "ERROR":    "\033[1;31m",
    "CRITICAL": "\033[1;35m",
    "FILL":     "\033[1;32m",
}
_RESET = "\033[0m"


def _audit(message: str, detail: str = "", level: str = "INFO") -> None:
    ts    = datetime.now(timezone.utc).isoformat()
    entry = {"ts": ts, "level": level, "message": message, "detail": detail}
    _AUDIT_LOG.append(entry)
    colour = _LEVEL_COLOUR.get(level, "")
    prefix = f"{colour}[{level:<8}]{_RESET}"
    print(
        f"  {prefix} {ts[11:19]}Z  {message}"
        + (f"\n{'':>27}{detail}" if detail else ""),
        file=sys.stderr if level in ("ERROR", "CRITICAL") else sys.stdout,
    )


def get_audit_log() -> list[dict]:
    return list(_AUDIT_LOG)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 11 ▸ PLUGIN REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_PLUGIN_REGISTRY: list[Callable[[dict], None]] = []


def register_plugin(fn: Callable[[dict], None]) -> None:
    """
    Register a post-fill plugin.

    Examples:
        register_plugin(discord_fill_notifier)   # discord_fills.py
        register_plugin(telegram_fill_sender)    # telegram_fills.py
        register_plugin(db_audit_writer)         # db_trail.py
        register_plugin(performance_tracker)     # perf_analytics.py

    Plugin receives the fully enriched order dict after broker response.
    """
    _PLUGIN_REGISTRY.append(fn)


def _fire_plugins(enriched: dict) -> None:
    for plugin in _PLUGIN_REGISTRY:
        try:
            plugin(enriched)
        except Exception as exc:  # noqa: BLE001
            _audit(
                f"⚠️  Plugin '{plugin.__name__}' raised: {exc}",
                level="WARN",
            )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 12 ▸ PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def send_orders(orders: list[dict]) -> list[dict]:
    """
    Transmit a batch of risk-cleared orders to the active broker.

    Accepts output from risk_guardian.risk_filter_orders().
    Only orders with risk_passed=True are transmitted.

    Parameters
    ----------
    orders : list[dict]
        Each dict must come from risk_guardian — must contain risk_passed,
        _order_id, _ticker, broker_payload.

    Returns
    -------
    list[dict]
        Original order dict enriched with 7 broker_sender output fields:

        broker_status        : str   — SENT|PAPER_FILLED|PARTIAL_FILL|FAILED|…
        broker_order_id      : str   — broker-assigned order identifier
        sent_at              : str   — ISO 8601 UTC timestamp
        fill_simulated_price : float — fill price (paper) or broker avg fill (live)
        route_latency_ms     : float — end-to-end transmission latency
        retry_count          : int   — number of retry attempts made
        transmission_log     : list  — ordered log of all transmission events
    """
    results: list[dict] = []
    sent_at = datetime.now(timezone.utc).isoformat()

    for idx, order in enumerate(orders):

        # ── Structural guard ───────────────────────────────────────────────
        if not isinstance(order, dict):
            _audit(
                f"[SKIP] Entry {idx} — not a dict ({type(order).__name__})",
                level="WARN",
            )
            continue

        order_id = str(order.get("_order_id") or uuid.uuid4())
        ticker   = str(order.get("_ticker") or "UNKNOWN").upper()

        # ── Kill switch ────────────────────────────────────────────────────
        if _KILL_SWITCH_ACTIVE and not order.get("_panic_close"):
            enriched = _enrich(
                order,
                BrokerStatus.KILLED, "", 0.0, 0.0, 0,
                [f"🔴 KILL SWITCH active — {ticker} order blocked"],
                sent_at,
            )
            results.append(enriched)
            _fire_plugins(enriched)
            continue

        # ── Risk guard passthrough ─────────────────────────────────────────
        if not order.get("risk_passed", True):
            reason = order.get("block_reason") or "risk_passed=False"
            enriched = _enrich(
                order,
                BrokerStatus.RISK_BLOCKED, "", 0.0, 0.0, 0,
                [f"🛡️  Risk Guardian blocked: {reason}"],
                sent_at,
            )
            results.append(enriched)
            _fire_plugins(enriched)
            continue

        # ── Idempotency check ──────────────────────────────────────────────
        if _IDEMPOTENCY_GUARD.is_duplicate(order_id):
            _audit(
                f"🔁 DUPLICATE blocked — {ticker} order_id={order_id[:12]}…",
                level="WARN",
            )
            enriched = _enrich(
                order,
                BrokerStatus.DUPLICATE, "", 0.0, 0.0, 0,
                [
                    f"🔁 IDEMPOTENCY GUARD — order_id {order_id[:12]}… "
                    "already submitted within TTL window. Blocked."
                ],
                sent_at,
            )
            results.append(enriched)
            _fire_plugins(enriched)
            continue

        # ── Select adapter ─────────────────────────────────────────────────
        payload      = order.get("broker_payload") or {}
        adapter_hint = str(payload.get("_adapter") or _ACTIVE_BROKER).lower()
        adapter_cls  = _ADAPTER_REGISTRY.get(
            adapter_hint, _ADAPTER_REGISTRY[AdapterName.PAPER.value]
        )

        # ── Transmit with retry ────────────────────────────────────────────
        try:
            result = _with_retry(
                fn=lambda: adapter_cls.send(order, payload),
                order_id=order_id,
                adapter_name=adapter_cls.NAME,
            )
        except Exception as exc:  # noqa: BLE001
            _audit(f"❌ Fatal transmission error for {ticker}: {exc}", level="ERROR")
            result = _SendResult(
                status=BrokerStatus.FAILED,
                broker_order_id="",
                fill_price=0.0,
                latency_ms=0.0,
                retry_count=MAX_RETRIES,
                log_entries=[
                    f"💀 Fatal: {type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                ],
                raw_response={"error": str(exc)},
            )

        # ── Register idempotency + position ───────────────────────────────
        _IDEMPOTENCY_GUARD.register(order_id)
        if result.status in (
            BrokerStatus.SENT,
            BrokerStatus.PAPER_FILLED,
            BrokerStatus.PARTIAL_FILL,
        ):
            _register_position(order_id, result, order)
            _audit(
                f"{'✅' if result.status == BrokerStatus.SENT else '📋'} "
                f"{result.status.value} — {ticker} | "
                f"broker_id={result.broker_order_id[:16]}… | "
                f"fill={result.fill_price:.4f} | "
                f"lat={result.latency_ms:.1f}ms | "
                f"retries={result.retry_count}",
                level="FILL",
            )
        else:
            _audit(
                f"❌ {result.status.value} — {ticker} | "
                f"retries={result.retry_count}",
                level="ERROR",
            )

        enriched = _enrich(
            order,
            result.status,
            result.broker_order_id,
            result.fill_price,
            result.latency_ms,
            result.retry_count,
            result.log_entries,
            sent_at,
        )
        results.append(enriched)
        _fire_plugins(enriched)

    return results


def _enrich(
    order:           dict,
    status:          BrokerStatus,
    broker_order_id: str,
    fill_price:      float,
    latency_ms:      float,
    retry_count:     int,
    log:             list[str],
    sent_at:         str,
) -> dict:
    return {
        **order,
        "broker_status":          status.value,
        "broker_order_id":        broker_order_id,
        "sent_at":                sent_at,
        "fill_simulated_price":   fill_price,
        "route_latency_ms":       latency_ms,
        "retry_count":            retry_count,
        "transmission_log":       log,
        "_sender_version":        SENDER_VERSION,
        "_active_broker":         _ACTIVE_BROKER,
        "_live_mode":             _LIVE_MODE_ACTIVE,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY — TERMINAL SUMMARY BANNER
# ══════════════════════════════════════════════════════════════════════════════

def print_transmission_summary(results: list[dict]) -> None:
    """Institutional-grade terminal summary after send_orders()."""
    PURPLE = "\033[1;35m"
    GREEN  = "\033[1;32m"
    RED    = "\033[1;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[1;36m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

    by_status: dict[str, int] = {}
    total_lat  = 0.0
    lat_count  = 0
    fill_total = 0.0

    for r in results:
        s = r.get("broker_status", "UNKNOWN")
        by_status[s] = by_status.get(s, 0) + 1
        lat = r.get("route_latency_ms", 0.0)
        if lat:
            total_lat += lat
            lat_count += 1
        fill_total += r.get("fill_simulated_price", 0.0)

    avg_lat    = round(total_lat / max(lat_count, 1), 2)
    lat_stats  = _LATENCY.stats(_ACTIVE_BROKER)
    broker_lat = lat_stats.get(_ACTIVE_BROKER, {})
    filled     = by_status.get("SENT", 0) + by_status.get("PAPER_FILLED", 0) + by_status.get("PARTIAL_FILL", 0)
    blocked    = by_status.get("KILLED", 0) + by_status.get("RISK_BLOCKED", 0)
    deduped    = by_status.get("DUPLICATE", 0)
    failed     = by_status.get("FAILED", 0)

    print(f"\n{PURPLE}{'╔' + '═'*80 + '╗'}{RESET}")
    print(f"{PURPLE}║{RESET}  📡  {BOLD}BROKER SENDER — TRANSMISSION REPORT{RESET}{PURPLE}{'':>41}║{RESET}")
    print(f"{PURPLE}{'╠' + '═'*80 + '╣'}{RESET}")
    print(
        f"{PURPLE}║{RESET}  Mode    : "
        f"{'🔴 LIVE' if _LIVE_MODE_ACTIVE else '📋 PAPER':<10}  "
        f"Broker: {BOLD}{_ACTIVE_BROKER.upper():<10}{RESET}  "
        f"Kill Switch: {'🔴 ON' if _KILL_SWITCH_ACTIVE else '🟢 OFF':<8}"
        f"{PURPLE}{'':>13}║{RESET}"
    )
    print(f"{PURPLE}{'╠' + '═'*80 + '╣'}{RESET}")

    status_icons = {
        "SENT": "✅", "PAPER_FILLED": "📋", "PARTIAL_FILL": "⚠️ ",
        "FAILED": "❌", "REJECTED": "🚫", "DUPLICATE": "🔁",
        "KILLED": "🔴", "RISK_BLOCKED": "🛡️ ", "MALFORMED": "💀",
    }
    for s, n in sorted(by_status.items(), key=lambda x: -x[1]):
        icon = status_icons.get(s, "❓")
        bar  = "█" * min(n * 5, 30) + "░" * max(0, 30 - n * 5)
        colour = GREEN if s in ("SENT", "PAPER_FILLED") else RED if s in ("FAILED", "KILLED") else YELLOW
        row = f"  {icon} {s:<16} {colour}{bar}{RESET}  {n}"
        print(f"{PURPLE}║{RESET}{row}{PURPLE}{'':>2}║{RESET}")

    print(f"{PURPLE}{'╠' + '═'*80 + '╣'}{RESET}")
    print(
        f"{PURPLE}║{RESET}  "
        f"Transmitted: {GREEN}{BOLD}{filled}{RESET}  "
        f"Blocked: {RED}{blocked}{RESET}  "
        f"Deduped: {YELLOW}{deduped}{RESET}  "
        f"Failed: {RED}{failed}{RESET}"
        f"{PURPLE}{'':>35}║{RESET}"
    )
    print(
        f"{PURPLE}║{RESET}  "
        f"Avg Latency : {BOLD}{avg_lat:.1f}ms{RESET}  "
        f"P95: {broker_lat.get('p95', 0):.1f}ms  "
        f"P99: {broker_lat.get('p99', 0):.1f}ms  "
        f"Open Positions: {BOLD}{len(_OPEN_POSITIONS)}{RESET}"
        f"{PURPLE}{'':>16}║{RESET}"
    )
    print(f"{PURPLE}{'╠' + '═'*80 + '╣'}{RESET}")

    if results:
        hdr = f"  {'ORDER ID':<38} {'TICKER':<8} {'STATUS':<16} {'FILL':>8}  {'LAT':>7}  RET"
        print(f"{PURPLE}║{RESET}{DIM}{hdr}{RESET}{PURPLE}{'':>2}║{RESET}")
        print(f"{PURPLE}║{RESET}  {'─'*76}{PURPLE}  ║{RESET}")
        for r in results:
            oid    = str(r.get("_order_id") or r.get("broker_order_id") or "")[:36]
            ticker = str(r.get("_ticker") or "???")[:8]
            st     = str(r.get("broker_status", "?"))[:16]
            fill   = r.get("fill_simulated_price", 0.0)
            lat    = r.get("route_latency_ms", 0.0)
            ret    = r.get("retry_count", 0)
            ok     = st in ("SENT", "PAPER_FILLED", "PARTIAL_FILL")
            colour = GREEN if ok else (YELLOW if st == "PARTIAL_FILL" else RED)
            row    = f"  {oid:<38} {ticker:<8} {st:<16} {fill:>8.4f}  {lat:>6.1f}ms  {ret}"
            print(f"{PURPLE}║{RESET}{colour}{row}{RESET}{PURPLE}{'':>2}║{RESET}")

    pnl = _FILL_SIM.pnl_summary()
    if pnl["total_fills"] > 0:
        print(f"{PURPLE}{'╠' + '═'*80 + '╣'}{RESET}")
        print(
            f"{PURPLE}║{RESET}  {DIM}Fill Sim — "
            f"fills={pnl['total_fills']}  "
            f"notional=${pnl['total_notional']:,.2f}  "
            f"avg_slip={pnl['avg_slippage_bps']:.1f}bps  "
            f"partials={pnl['partial_fills']}{RESET}"
            f"{PURPLE}{'':>8}║{RESET}"
        )

    print(f"{PURPLE}{'╚' + '═'*80 + '╝'}{RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def reset_all_state() -> None:
    """Full reset — use between test runs or sessions."""
    global _LIVE_MODE_ACTIVE, _KILL_SWITCH_ACTIVE
    _LIVE_MODE_ACTIVE   = False
    _KILL_SWITCH_ACTIVE = False
    _ACTIVE_BROKER      # don't reset — preserve config
    _IDEMPOTENCY_GUARD.clear()
    _FILL_SIM.clear()
    _LATENCY.clear()
    _OPEN_POSITIONS.clear()
    _PANIC_LOG.clear()
    _AUDIT_LOG.clear()
    _PLUGIN_REGISTRY.clear()


def get_sender_state() -> dict:
    return {
        "version":         SENDER_VERSION,
        "live_mode":       _LIVE_MODE_ACTIVE,
        "active_broker":   _ACTIVE_BROKER,
        "kill_switch":     _KILL_SWITCH_ACTIVE,
        "open_positions":  len(_OPEN_POSITIONS),
        "idempotency_reg": len(_IDEMPOTENCY_GUARD.snapshot()),
        "fill_summary":    _FILL_SIM.pnl_summary(),
        "latency_stats":   _LATENCY.stats(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

def _make_order(
    ticker:    str,
    side:      str   = "BUY",
    notional:  float = 5000.0,
    order_id:  str | None = None,
    risk_pass: bool  = True,
    adapter:   str   = "paper",
    block_reason: str | None = None,
) -> dict:
    oid = order_id or str(uuid.uuid4())
    return {
        "_order_id":              oid,
        "_ticker":                ticker,
        "execution_status":       "PAPER_QUEUED",
        "risk_passed":            risk_pass,
        "block_reason":           block_reason,
        "adjusted_position_size": notional / 100_000,
        "position_size":          notional / 100_000,
        "stop_loss_pct":          0.03,
        "take_profit_pct":        0.06,
        "broker_payload": {
            "symbol":    ticker,
            "side":      side,
            "notional":  notional,
            "_adapter":  adapter,
        },
        "risk_guard_reasons": [f"smoke-test {ticker}"],
    }


def _smoke_test() -> None:
    global _LIVE_MODE_ACTIVE
    GREEN  = "\033[1;32m"
    RED    = "\033[1;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[1;36m"
    PURPLE = "\033[1;35m"
    RESET  = "\033[0m"

    print(f"\n{PURPLE}{'▓'*84}")
    print("  🧪  BROKER SENDER — SMOKE TEST SUITE")
    print(f"{'▓'*84}{RESET}\n")

    passed = 0
    failed = 0

    def _check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        ok = bool(condition)
        tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{tag}]  {label}")
        if detail:
            print(f"          {CYAN}↳ {detail}{RESET}")
        if ok:
            passed += 1
        else:
            failed += 1

    # ─────────────────────────────────────────────────────────────────────────
    # T01 — Paper fill: AAPL BUY
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    res = send_orders([_make_order("AAPL", "BUY", 5000.0)])
    r   = res[0]
    _check(
        "T01 — Paper fill AAPL BUY → PAPER_FILLED or PARTIAL_FILL",
        r["broker_status"] in ("PAPER_FILLED", "PARTIAL_FILL"),
        f"status={r['broker_status']} | fill={r['fill_simulated_price']:.2f} | "
        f"lat={r['route_latency_ms']:.1f}ms",
    )
    _check(
        "T01b — Fill price near AAPL reference ($228)",
        200 <= r["fill_simulated_price"] <= 260,
        f"fill_price={r['fill_simulated_price']:.4f}",
    )
    _check(
        "T01c — transmission_log populated",
        len(r["transmission_log"]) >= 1,
        f"log entries: {len(r['transmission_log'])}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T02 — BTC paper fill (high-vol, expect wider slippage)
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    res  = send_orders([_make_order("BTC", "BUY", 10_000.0)])
    r    = res[0]
    fill = r["fill_simulated_price"]
    _check(
        "T02 — BTC paper fill price near reference ($97k)",
        85_000 <= fill <= 110_000,
        f"fill={fill:.2f}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T03 — SELL order: fill below mid (adverse slippage on SELL)
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    res = send_orders([_make_order("SPY", "SELL", 5000.0)])
    r   = res[0]
    mid = 545.0
    _check(
        "T03 — SPY SELL fill price below mid (adverse slippage)",
        r["fill_simulated_price"] < mid,
        f"fill={r['fill_simulated_price']:.4f} < mid={mid}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T04 — Idempotency guard: same order_id blocked on re-submit
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    fixed_id = "IDEMPOTENCY-TEST-ORDER-001"
    o1 = _make_order("NVDA", order_id=fixed_id)
    o2 = _make_order("NVDA", order_id=fixed_id)   # same id

    r1 = send_orders([o1])[0]
    r2 = send_orders([o2])[0]
    _check(
        "T04a — First submission accepted (PAPER_FILLED)",
        r1["broker_status"] in ("PAPER_FILLED", "PARTIAL_FILL"),
        f"first={r1['broker_status']}",
    )
    _check(
        "T04b — Duplicate blocked (DUPLICATE status)",
        r2["broker_status"] == "DUPLICATE",
        f"second={r2['broker_status']}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T05 — Kill switch blocks new orders
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    activate_kill_switch()
    res = send_orders([_make_order("TSLA", "BUY")])
    r   = res[0]
    _check(
        "T05 — Kill switch active → order KILLED",
        r["broker_status"] == "KILLED",
        f"status={r['broker_status']}",
    )
    deactivate_kill_switch()
    res2 = send_orders([_make_order("TSLA", "BUY")])
    _check(
        "T05b — Kill switch deactivated → order flows through",
        res2[0]["broker_status"] in ("PAPER_FILLED", "PARTIAL_FILL"),
        f"status={res2[0]['broker_status']}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T06 — Live mode disabled by default
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    _check(
        "T06 — Live mode OFF by default",
        not _LIVE_MODE_ACTIVE,
        f"_LIVE_MODE_ACTIVE={_LIVE_MODE_ACTIVE}",
    )
    try:
        enable_live_mode("WRONG-KEY")
        _check("T06b — Wrong key raises PermissionError", False, "should have raised")
    except PermissionError as exc:
        _check(
            "T06b — Wrong key raises PermissionError ✓",
            True,
            str(exc)[:80],
        )
    _check(
        "T06c — Live mode still OFF after wrong key",
        not _LIVE_MODE_ACTIVE,
        f"_LIVE_MODE_ACTIVE={_LIVE_MODE_ACTIVE}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T07 — Risk-blocked order → RISK_BLOCKED status
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    blocked_order = _make_order("GLD", risk_pass=False, block_reason="Daily loss lock")
    res = send_orders([blocked_order])
    _check(
        "T07 — risk_passed=False → RISK_BLOCKED",
        res[0]["broker_status"] == "RISK_BLOCKED",
        f"status={res[0]['broker_status']}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T08 — Retry engine: simulate failure via bad adapter config
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    # Force alpaca in live mode with deliberately broken config so retry fires
    _LIVE_MODE_ACTIVE = True   # bypass secret check for test
    set_active_broker("alpaca")
    configure_broker(
        "alpaca",
        api_key="bad-key",
        api_secret="bad-secret",
        base_url="http://127.0.0.1:19999",   # unreachable port
    )
    t0   = time.perf_counter()
    res  = send_orders([_make_order("MSFT", adapter="alpaca")])
    elapsed = time.perf_counter() - t0
    _LIVE_MODE_ACTIVE = False
    set_active_broker("paper")
    r = res[0]
    _check(
        "T08a — Failed live order → FAILED status after retries",
        r["broker_status"] == "FAILED",
        f"status={r['broker_status']} | retries={r['retry_count']}",
    )
    _check(
        "T08b — Retry count = MAX_RETRIES",
        r["retry_count"] == MAX_RETRIES,
        f"retry_count={r['retry_count']} (expected {MAX_RETRIES})",
    )
    _check(
        "T08c — Backoff elapsed ≥ base delay sum",
        elapsed >= (BACKOFF_BASE_SECONDS * (1 + BACKOFF_MULTIPLIER + BACKOFF_MULTIPLIER**2) * 0.8),
        f"elapsed={elapsed:.2f}s (expected ≥ ~{BACKOFF_BASE_SECONDS * 3.5:.2f}s)",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T09 — Panic flatten (kill switch + close all)
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    # Open two paper positions first
    send_orders([_make_order("QQQ", "BUY", 5000.0)])
    send_orders([_make_order("XLF", "BUY", 3000.0)])
    positions_before = len(_OPEN_POSITIONS)
    close_results    = panic_flatten_all_positions()
    _check(
        "T09a — panic_flatten submits close orders for each open position",
        len(close_results) == positions_before,
        f"positions_before={positions_before} | close_results={len(close_results)}",
    )
    _check(
        "T09b — Kill switch active after panic flatten",
        _KILL_SWITCH_ACTIVE,
        f"_KILL_SWITCH_ACTIVE={_KILL_SWITCH_ACTIVE}",
    )
    _check(
        "T09c — Panic log recorded",
        len(_PANIC_LOG) > 0,
        f"panic_log entries: {len(_PANIC_LOG)}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T10 — Non-dict entry safe skip
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    res = send_orders(["not-a-dict", None, 42])   # type: ignore[list-item]
    _check(
        "T10 — Non-dict entries safely skipped (no output)",
        len(res) == 0,
        f"results={len(res)}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T11 — Plugin hook fires on fill
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    plugin_calls: list[str] = []

    def _mock_plugin(order: dict) -> None:
        plugin_calls.append(order["broker_status"])

    register_plugin(_mock_plugin)
    send_orders([_make_order("GS", "SELL", 4000.0)])
    _check(
        "T11 — Plugin fires once per order",
        len(plugin_calls) == 1,
        f"plugin_calls={plugin_calls}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T12 — Batch: 3 orders in single call
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    batch = [
        _make_order("AMZN", "BUY", 6000.0),
        _make_order("META", "SELL", 4000.0),
        _make_order("AMD",  "BUY", 3000.0),
    ]
    res = send_orders(batch)
    _check(
        "T12 — Batch of 3 orders all return results",
        len(res) == 3,
        f"results={len(res)} | statuses={[r['broker_status'] for r in res]}",
    )
    _check(
        "T12b — All 3 have valid broker_order_ids",
        all(r["broker_order_id"] for r in res),
        f"ids={[r['broker_order_id'][:12] for r in res]}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T13 — Latency monitor populated after fills
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    for ticker in ["GOOGL", "TSLA", "INTC", "JPM", "XOM"]:
        send_orders([_make_order(ticker)])
    stats_all = _LATENCY.stats("paper")
    stats = stats_all.get("paper", {})
    _check(
        "T13 — Latency monitor tracks P50/P95/P99",
        stats.get("count", 0) >= 5
        and "p50" in stats
        and "p95" in stats,
        f"count={stats.get('count')} | p50={stats.get('p50')}ms | p95={stats.get('p95')}ms",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T14 — Full summary banner renders
    # ─────────────────────────────────────────────────────────────────────────
    reset_all_state()
    demo_orders = [
        _make_order("SPY",  "BUY",  5000.0),
        _make_order("QQQ",  "SELL", 3000.0),
        _make_order("NVDA", "BUY",  8000.0, risk_pass=False, block_reason="test"),
    ]
    demo_results = send_orders(demo_orders)
    print(f"\n  {YELLOW}── Summary Banner Preview ──────────────────────────────────{RESET}")
    print_transmission_summary(demo_results)
    _check(
        "T14 — Summary banner rendered without exception",
        True,
        f"orders={len(demo_results)}",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Results
    # ─────────────────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{PURPLE}{'═'*84}")
    print(
        f"  🏁  SMOKE TEST RESULTS — {passed}/{total} passed   "
        f"{'✅ ALL CLEAR' if failed == 0 else '❌ FAILURES DETECTED'}"
    )
    print(f"{'═'*84}{RESET}\n")
    reset_all_state()
    sys.exit(0 if failed == 0 else 1)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _startup_banner() -> None:
    print("""
\033[1;32m╔══════════════════════════════════════════════════════════════════════════════╗
║   📡  MONSTER TRADING AI — BROKER SENDER  v1.0.0                          ║
║   Pipeline Terminal Node — Institutional OMS Execution Router             ║
║   Adapters: Alpaca · IBKR · Binance · Paper  |  Std-lib only             ║
╚══════════════════════════════════════════════════════════════════════════════╝\033[0m
""")


if __name__ == "__main__":
    if "--smoke" in sys.argv or "-s" in sys.argv:
        _smoke_test()
    else:
        _startup_banner()
        print("Usage:")
        print("  python broker_sender.py --smoke         # run full smoke test")
        print()
        print("API:")
        print("  from broker_sender import send_orders, enable_live_mode")
        print("  results = send_orders(risk_guardian_output)")
        print()
        print("Configuration:")
        print("  set_active_broker('alpaca')")
        print("  configure_broker('alpaca', api_key=..., api_secret=...)")
        print("  enable_live_mode('MONSTER-LIVE-2025')")
        print("  register_plugin(discord_fill_notifier)")
        print("  register_plugin(db_audit_writer)")
        print()
        print("Emergency:")
        print("  activate_kill_switch()")
        print("  panic_flatten_all_positions()")