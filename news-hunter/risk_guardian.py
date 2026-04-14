"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                       ║
║   ██████╗ ██╗███████╗██╗  ██╗    ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ ██╗       ║
║   ██╔══██╗██║██╔════╝██║ ██╔╝   ██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗██║       ║
║   ██████╔╝██║███████╗█████╔╝    ██║  ███╗██║   ██║███████║██████╔╝██║  ██║██║       ║
║   ██╔══██╗██║╚════██║██╔═██╗    ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║██║       ║
║   ██║  ██║██║███████║██║  ██╗   ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝██║       ║
║   ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝       ║
║                                                                                       ║
║        ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ ██╗ █████╗ ███╗   ██╗               ║
║       ██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗██║██╔══██╗████╗  ██║               ║
║       ██║  ███╗██║   ██║███████║██████╔╝██║  ██║██║███████║██╔██╗ ██║               ║
║       ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║██║██╔══██║██║╚██╗██║               ║
║       ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝██║██║  ██║██║ ╚████║               ║
║        ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝               ║
║                                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────────────────────┐  ║
║  │            🛡️  R I S K   G U A R D I A N  —  CAPITAL FIREWALL  🛡️              │  ║
║  │         Pipeline Stage 7 — Final Kill-Switch Before Broker Execution            │  ║
║  │   execution_bridge → [YOU] → broker_sender                                      │  ║
║  └─────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                       ║
║  Module   : risk_guardian.py                                                          ║
║  Version  : 1.0.0                                                                     ║
║  Mission  : "Even if a trade is valid, should capital be deployed RIGHT NOW?"         ║
║  Layers   : 12  |  Checks: 7  |  Complexity: O(1) per order                          ║
║                                                                                       ║
║  ╔═══════════════════════════════════════════════════════════════════════════════╗    ║
║  ║  WARNING: This module controls live capital deployment.                       ║    ║
║  ║  Every line of logic here is portfolio survival logic.                        ║    ║
║  ║  Block early. Block often. Let the alpha prove itself.                        ║    ║
║  ╚═══════════════════════════════════════════════════════════════════════════════╝    ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import sys
import textwrap
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, NamedTuple

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 ▸ CONSTANTS & GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GUARDIAN_VERSION = "1.0.0"
GUARDIAN_BUILD   = "MONSTER-TRADING-AI"

# ── Layer 1: Daily Loss Lock ──────────────────────────────────────────────────
# Read from unified config.DRAWDOWN_POLICY_TIERS (never hardcode locally)
from config import DRAWDOWN_POLICY_TIERS, DRAWDOWN_ACTION_KILL_SWITCH

# ── Layer 2: Correlation Cluster ─────────────────────────────────────────────
MAX_CLUSTER_POSITIONS       = 2       # max active trades within one cluster

# ── Layer 3: Macro Event Freeze ──────────────────────────────────────────────
MACRO_FREEZE_MINUTES_BEFORE = 30      # block N mins before known macro event
MACRO_FREEZE_MINUTES_AFTER  = 15      # block N mins after macro event

# ── Layer 4: Whipsaw Protection ──────────────────────────────────────────────
WHIPSAW_WINDOW_MINUTES      = 60      # BUY→SELL→BUY within this window → block
WHIPSAW_MIN_FLIPS           = 2       # direction changes needed to trigger block

# ── Layer 5: News Cascade ────────────────────────────────────────────────────
CASCADE_WINDOW_MINUTES      = 10      # rolling window for theme counting
CASCADE_HEADLINE_THRESHOLD  = 3       # > N same-theme headlines → reduce size
CASCADE_SIZE_REDUCTION      = 0.50    # multiply position by 0.50

# ── Layer 6: Volatility Shock ────────────────────────────────────────────────
VOLATILITY_EXTREME_MULT     = 0.30    # EXTREME regime → keep only 30%
VOLATILITY_HIGH_MULT        = 0.60    # HIGH regime → keep only 60%
VOLATILITY_ELEVATED_MULT    = 0.80    # ELEVATED → keep 80%
VOLATILITY_NORMAL_MULT      = 1.00    # NORMAL → full size

# ── Layer 7: Confidence Decay ────────────────────────────────────────────────
CONFIDENCE_DECAY_HALF_LIFE_MINUTES = 20   # half-life in minutes
CONFIDENCE_DECAY_FLOOR             = 0.40  # never decay below this multiplier

# ── Position size multiplier bounds ──────────────────────────────────────────
SIZE_MULT_MIN          = 0.0    # full block
SIZE_MULT_MAX          = 1.25   # exceptional conviction bonus
SIZE_MULT_CONVICTION   = 1.25   # awarded when risk_score ≥ 0.92

# ── Risk score thresholds ─────────────────────────────────────────────────────
RISK_SCORE_PASS_FLOOR  = 0.40   # below this → always block
RISK_SCORE_WARN_FLOOR  = 0.60   # 0.40–0.60 → heavy reduction


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 ▸ ENUMS & VALUE TYPES
# ══════════════════════════════════════════════════════════════════════════════

class RiskVerdict(str, Enum):
    PASS            = "PASS"
    BLOCK           = "BLOCK"
    REDUCED         = "REDUCED"        # passed but position shrunk
    CONVICTION      = "CONVICTION"     # exceptional — size bonus granted


class VolatilityRegime(str, Enum):
    NORMAL   = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH     = "HIGH"
    EXTREME  = "EXTREME"


class MacroEventType(str, Enum):
    FOMC          = "FOMC"
    CPI           = "CPI"
    NFP           = "NFP"
    POWELL_SPEECH = "POWELL_SPEECH"
    MEGA_EARNINGS = "MEGA_EARNINGS"
    CUSTOM        = "CUSTOM"


class CorrelationCluster(str, Enum):
    CRYPTO_PAIR   = "CRYPTO_PAIR"         # BTC + COIN + MSTR
    SEMI_QQQ      = "SEMI_QQQ"            # NVDA + AMD + QQQ
    GOLD_DXY      = "GOLD_DXY"            # XAU + DXY (inverse)
    ENERGY        = "ENERGY"              # XLE + CVX + XOM
    FINANCIALS    = "FINANCIALS"          # XLF + JPM + BAC + GS
    RATE_SENS     = "RATE_SENSITIVE"      # TLT + IEF + MBS proxies
    UNCORRELATED  = "UNCORRELATED"


class _RiskCheckResult(NamedTuple):
    passed:     bool
    multiplier: float     # 0.0 → block, 1.0 → unchanged, <1.0 → reduce
    reasons:    list[str]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 ▸ DAILY LOSS LEDGER
# ══════════════════════════════════════════════════════════════════════════════

class DailyLossLedger:
    """
    Tracks realised + unrealised P&L for the current trading day.
    Reset automatically at UTC midnight.
    Thread-unsafe; swap _store for Redis for multi-process deployment.
    """

    def __init__(self) -> None:
        self._day:           str   = self._today()
        self._realised_loss: float = 0.0     # cumulative loss (positive = loss)
        self._portfolio_val: float = 100_000.0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day           = today
            self._realised_loss = 0.0

    def set_portfolio_value(self, value: float) -> None:
        self._portfolio_val = value

    def record_loss(self, loss_usd: float) -> None:
        """Record a realised loss (positive value = money lost)."""
        self._roll_day()
        self._realised_loss += max(0.0, loss_usd)

    def record_gain(self, gain_usd: float) -> None:
        """Offset losses with a realised gain."""
        self._roll_day()
        self._realised_loss = max(0.0, self._realised_loss - gain_usd)

    @property
    def drawdown_pct(self) -> float:
        self._roll_day()
        if self._portfolio_val <= 0:
            return 0.0
        return self._realised_loss / self._portfolio_val

    def check(self) -> _RiskCheckResult:
        """Check current drawdown against unified policy tiers."""
        dd = self.drawdown_pct
        
        # Query unified policy — first matching tier wins
        for threshold, action, multiplier in DRAWDOWN_POLICY_TIERS:
            if dd >= threshold:
                if action == DRAWDOWN_ACTION_KILL_SWITCH:
                    return _RiskCheckResult(
                        passed=False,
                        multiplier=0.0,
                        reasons=[
                            f"🔒 DAILY LOSS LOCK — drawdown {dd:.2%} ≥ "
                            f"policy threshold {threshold:.2%}. "
                            "All new trades blocked until next session."
                        ],
                    )
                else:
                    return _RiskCheckResult(
                        passed=True,
                        multiplier=multiplier,
                        reasons=[
                            f"⚠️  Daily drawdown warning {dd:.2%} [{action}] — "
                            f"position size reduced to {multiplier:.0%}"
                        ],
                    )
        
        # Fallback (shouldn't reach — (0.0, NORMAL, 1.0) should always match)
        return _RiskCheckResult(passed=True, multiplier=1.0, reasons=[])

    def reset(self) -> None:
        self._realised_loss = 0.0
        self._day = self._today()

    def snapshot(self) -> dict:
        return {
            "date":           self._day,
            "realised_loss":  self._realised_loss,
            "drawdown_pct":   self.drawdown_pct,
            "portfolio_val":  self._portfolio_val,
        }


# Module singleton
_DAILY_LEDGER = DailyLossLedger()


def record_daily_loss(loss_usd: float) -> None:
    _DAILY_LEDGER.record_loss(loss_usd)


def record_daily_gain(gain_usd: float) -> None:
    _DAILY_LEDGER.record_gain(gain_usd)


def set_portfolio_value(value: float) -> None:
    _DAILY_LEDGER.set_portfolio_value(value)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 ▸ CORRELATION CLUSTER REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# O(1) ticker → cluster lookup
_TICKER_CLUSTER_MAP: dict[str, CorrelationCluster] = {
    # Crypto proxy cluster
    "BTC":    CorrelationCluster.CRYPTO_PAIR,
    "ETH":    CorrelationCluster.CRYPTO_PAIR,
    "COIN":   CorrelationCluster.CRYPTO_PAIR,
    "MSTR":   CorrelationCluster.CRYPTO_PAIR,
    "BTCUSD": CorrelationCluster.CRYPTO_PAIR,
    "ETHUSD": CorrelationCluster.CRYPTO_PAIR,
    "GBTC":   CorrelationCluster.CRYPTO_PAIR,
    # Semiconductor / tech cluster
    "NVDA":   CorrelationCluster.SEMI_QQQ,
    "AMD":    CorrelationCluster.SEMI_QQQ,
    "QQQ":    CorrelationCluster.SEMI_QQQ,
    "INTC":   CorrelationCluster.SEMI_QQQ,
    "MU":     CorrelationCluster.SEMI_QQQ,
    "SMCI":   CorrelationCluster.SEMI_QQQ,
    "AVGO":   CorrelationCluster.SEMI_QQQ,
    # Gold / Dollar inverse cluster
    "GLD":    CorrelationCluster.GOLD_DXY,
    "GC":     CorrelationCluster.GOLD_DXY,
    "XAU":    CorrelationCluster.GOLD_DXY,
    "SLV":    CorrelationCluster.GOLD_DXY,
    "DXY":    CorrelationCluster.GOLD_DXY,
    "UUP":    CorrelationCluster.GOLD_DXY,
    # Energy cluster
    "XLE":    CorrelationCluster.ENERGY,
    "CVX":    CorrelationCluster.ENERGY,
    "XOM":    CorrelationCluster.ENERGY,
    "COP":    CorrelationCluster.ENERGY,
    "OXY":    CorrelationCluster.ENERGY,
    "USO":    CorrelationCluster.ENERGY,
    # Financials cluster
    "XLF":    CorrelationCluster.FINANCIALS,
    "JPM":    CorrelationCluster.FINANCIALS,
    "BAC":    CorrelationCluster.FINANCIALS,
    "GS":     CorrelationCluster.FINANCIALS,
    "MS":     CorrelationCluster.FINANCIALS,
    "C":      CorrelationCluster.FINANCIALS,
    # Rate-sensitive
    "TLT":    CorrelationCluster.RATE_SENS,
    "IEF":    CorrelationCluster.RATE_SENS,
    "SHY":    CorrelationCluster.RATE_SENS,
    "BND":    CorrelationCluster.RATE_SENS,
    "AGG":    CorrelationCluster.RATE_SENS,
}


def classify_cluster(ticker: str) -> CorrelationCluster:
    """O(1) cluster lookup."""
    return _TICKER_CLUSTER_MAP.get(ticker.upper(), CorrelationCluster.UNCORRELATED)


class CorrelationGuard:
    """
    Tracks how many positions are open per cluster.
    Blocks new trades when a cluster is saturated.
    """

    def __init__(self, max_per_cluster: int = MAX_CLUSTER_POSITIONS) -> None:
        # cluster → set of active order_ids
        self._active: dict[str, set[str]] = {}
        self._max = max_per_cluster

    def cluster_count(self, cluster: CorrelationCluster) -> int:
        return len(self._active.get(cluster.value, set()))

    def is_saturated(self, cluster: CorrelationCluster) -> bool:
        if cluster == CorrelationCluster.UNCORRELATED:
            return False
        return self.cluster_count(cluster) >= self._max

    def register(self, cluster: CorrelationCluster, order_id: str) -> None:
        key = cluster.value
        if key not in self._active:
            self._active[key] = set()
        self._active[key].add(order_id)

    def deregister(self, cluster: CorrelationCluster, order_id: str) -> None:
        self._active.get(cluster.value, set()).discard(order_id)

    def check(self, ticker: str) -> _RiskCheckResult:
        cluster = classify_cluster(ticker)
        if self.is_saturated(cluster):
            count = self.cluster_count(cluster)
            return _RiskCheckResult(
                passed=False,
                multiplier=0.0,
                reasons=[
                    f"🔗 CORRELATION CLUSTER BLOCK — '{cluster.value}' already has "
                    f"{count}/{self._max} active positions. Adding '{ticker}' would "
                    "over-concentrate correlated risk."
                ],
            )
        if cluster != CorrelationCluster.UNCORRELATED:
            count = self.cluster_count(cluster)
            return _RiskCheckResult(
                passed=True,
                multiplier=1.0,
                reasons=[
                    f"✅ Cluster '{cluster.value}' {count}/{self._max} — slot available"
                ],
            )
        return _RiskCheckResult(passed=True, multiplier=1.0, reasons=[])

    def snapshot(self) -> dict:
        return {k: list(v) for k, v in self._active.items()}

    def clear(self) -> None:
        self._active.clear()


# Module singleton
_CORRELATION_GUARD = CorrelationGuard()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 ▸ MACRO EVENT CALENDAR + FREEZE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class MacroEvent(NamedTuple):
    event_type:  MacroEventType
    scheduled:   datetime       # timezone-aware UTC
    label:       str
    tickers_affected: frozenset[str]   # empty = affects all


class MacroCalendar:
    """
    O(1) freeze check — events stored in a sorted list;
    check only scans within a rolling ±FREEZE window.
    """

    def __init__(self) -> None:
        self._events: list[MacroEvent] = []

    def add_event(
        self,
        event_type: MacroEventType,
        scheduled: datetime,
        label: str,
        tickers_affected: frozenset[str] | None = None,
    ) -> None:
        if scheduled.tzinfo is None:
            raise ValueError("MacroEvent.scheduled must be timezone-aware (UTC)")
        self._events.append(MacroEvent(
            event_type=event_type,
            scheduled=scheduled,
            label=label,
            tickers_affected=tickers_affected or frozenset(),
        ))

    def check(self, ticker: str) -> _RiskCheckResult:
        now     = datetime.now(timezone.utc)
        ticker  = ticker.upper()
        freeze_before = timedelta(minutes=MACRO_FREEZE_MINUTES_BEFORE)
        freeze_after  = timedelta(minutes=MACRO_FREEZE_MINUTES_AFTER)

        for event in self._events:
            # Affects all, or specifically this ticker
            affects = (
                not event.tickers_affected
                or ticker in event.tickers_affected
            )
            if not affects:
                continue

            delta_to_event  = event.scheduled - now     # positive = future
            delta_from_event = now - event.scheduled     # positive = past

            in_pre_window  = timedelta(0) <= delta_to_event  <= freeze_before
            in_post_window = timedelta(0) <= delta_from_event <= freeze_after

            if in_pre_window:
                mins = int(delta_to_event.total_seconds() / 60)
                return _RiskCheckResult(
                    passed=False,
                    multiplier=0.0,
                    reasons=[
                        f"📅 MACRO FREEZE — '{event.label}' "
                        f"({event.event_type.value}) in {mins} min. "
                        f"Pre-event blackout: {MACRO_FREEZE_MINUTES_BEFORE} min before event."
                    ],
                )
            if in_post_window:
                mins = int(delta_from_event.total_seconds() / 60)
                return _RiskCheckResult(
                    passed=False,
                    multiplier=0.0,
                    reasons=[
                        f"📅 MACRO FREEZE — post-event cooldown after '{event.label}' "
                        f"({event.event_type.value}), {mins} min ago. "
                        f"Settling for {MACRO_FREEZE_MINUTES_AFTER} min."
                    ],
                )
        return _RiskCheckResult(passed=True, multiplier=1.0, reasons=[])

    def clear(self) -> None:
        self._events.clear()

    def upcoming(self, window_minutes: int = 120) -> list[MacroEvent]:
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=window_minutes)
        return [e for e in self._events if now <= e.scheduled <= cutoff]

    def snapshot(self) -> list[dict]:
        return [
            {
                "type":     e.event_type.value,
                "label":    e.label,
                "at_utc":   e.scheduled.isoformat(),
                "tickers":  list(e.tickers_affected),
            }
            for e in self._events
        ]


# Module singleton
_MACRO_CALENDAR = MacroCalendar()


def schedule_macro_event(
    event_type: MacroEventType | str,
    scheduled_utc: datetime,
    label: str,
    tickers_affected: frozenset[str] | None = None,
) -> None:
    """
    Register an upcoming macro event in the freeze calendar.

    Example:
        schedule_macro_event(
            MacroEventType.FOMC,
            datetime(2025, 7, 30, 14, 0, tzinfo=timezone.utc),
            "FOMC Rate Decision July 2025",
        )
    """
    if isinstance(event_type, str):
        event_type = MacroEventType(event_type)
    _MACRO_CALENDAR.add_event(event_type, scheduled_utc, label, tickers_affected)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 ▸ WHIPSAW DETECTION REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class WhipsawGuard:
    """
    Detects rapid direction reversal: BUY→SELL→BUY (or SELL→BUY→SELL)
    within a rolling WHIPSAW_WINDOW_MINUTES window.
    """

    def __init__(
        self,
        window_minutes: int = WHIPSAW_WINDOW_MINUTES,
        min_flips:      int = WHIPSAW_MIN_FLIPS,
    ) -> None:
        # ticker → deque of (timestamp, direction)
        self._history: dict[str, deque[tuple[datetime, str]]] = {}
        self._window   = timedelta(minutes=window_minutes)
        self._min_flips = min_flips

    def _prune(self, ticker: str) -> None:
        """Remove stale entries outside the rolling window.  O(k) where k ≤ window depth."""
        cutoff = datetime.now(timezone.utc) - self._window
        dq = self._history.get(ticker)
        if dq:
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def record_trade(self, ticker: str, direction: str) -> None:
        ticker = ticker.upper()
        if ticker not in self._history:
            self._history[ticker] = deque()
        self._prune(ticker)
        self._history[ticker].append((datetime.now(timezone.utc), direction.upper()))

    def check(self, ticker: str, proposed_direction: str) -> _RiskCheckResult:
        ticker = ticker.upper()
        self._prune(ticker)
        history = list(self._history.get(ticker, []))

        if len(history) < self._min_flips:
            return _RiskCheckResult(passed=True, multiplier=1.0, reasons=[])

        # Count direction flips within the window (including proposed)
        directions = [d for _, d in history] + [proposed_direction.upper()]
        flips = sum(
            1 for i in range(1, len(directions))
            if directions[i] != directions[i - 1]
        )

        if flips >= self._min_flips:
            oldest_ts = history[0][0].strftime("%H:%M:%S UTC")
            return _RiskCheckResult(
                passed=False,
                multiplier=0.0,
                reasons=[
                    f"🌀 WHIPSAW BLOCK — '{ticker}' flipped direction "
                    f"{flips}x in the last {WHIPSAW_WINDOW_MINUTES} min "
                    f"(since {oldest_ts}). Suspected noise / stop-hunt. "
                    "Trade blocked to prevent chasing."
                ],
            )
        return _RiskCheckResult(passed=True, multiplier=1.0, reasons=[])

    def clear(self, ticker: str | None = None) -> None:
        if ticker:
            self._history.pop(ticker.upper(), None)
        else:
            self._history.clear()

    def snapshot(self) -> dict:
        return {
            t: [(ts.isoformat(), d) for ts, d in dq]
            for t, dq in self._history.items()
        }


# Module singleton
_WHIPSAW_GUARD = WhipsawGuard()


def record_executed_trade(ticker: str, direction: str) -> None:
    """
    Call this AFTER a trade is executed so the whipsaw guard can track history.
    """
    _WHIPSAW_GUARD.record_trade(ticker, direction)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 ▸ NEWS CASCADE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

# O(1) keyword → theme cluster (first match wins)
_THEME_KEYWORDS: dict[str, str] = {
    "fed":         "MONETARY_POLICY",
    "fomc":        "MONETARY_POLICY",
    "rate":        "MONETARY_POLICY",
    "inflation":   "MONETARY_POLICY",
    "cpi":         "MONETARY_POLICY",
    "powell":      "MONETARY_POLICY",
    "bank":        "BANKING_CRISIS",
    "collapse":    "BANKING_CRISIS",
    "fdic":        "BANKING_CRISIS",
    "recession":   "MACRO_RISK",
    "gdp":         "MACRO_RISK",
    "nfp":         "MACRO_RISK",
    "jobs":        "MACRO_RISK",
    "war":         "GEOPOLITICAL",
    "sanction":    "GEOPOLITICAL",
    "oil":         "ENERGY_SHOCK",
    "crude":       "ENERGY_SHOCK",
    "opec":        "ENERGY_SHOCK",
    "crypto":      "CRYPTO_NEWS",
    "bitcoin":     "CRYPTO_NEWS",
    "btc":         "CRYPTO_NEWS",
    "hack":        "CYBER_EVENT",
    "breach":      "CYBER_EVENT",
    "earn":        "EARNINGS",
    "revenue":     "EARNINGS",
    "eps":         "EARNINGS",
    "guidance":    "EARNINGS",
}


def _extract_theme(text: str) -> str:
    """O(k) where k = len(text.split()) — classify headline into a theme."""
    lower = text.lower()
    for keyword, theme in _THEME_KEYWORDS.items():
        if keyword in lower:
            return theme
    return "GENERAL"


class NewsCascadeMonitor:
    """
    Rolling window counter per theme.
    > CASCADE_HEADLINE_THRESHOLD same-theme headlines in CASCADE_WINDOW_MINUTES
    → reduce position size by CASCADE_SIZE_REDUCTION.
    """

    def __init__(
        self,
        window_minutes: int = CASCADE_WINDOW_MINUTES,
        threshold:      int = CASCADE_HEADLINE_THRESHOLD,
    ) -> None:
        # theme → deque of timestamps
        self._buckets: dict[str, deque[datetime]] = {}
        self._window    = timedelta(minutes=window_minutes)
        self._threshold = threshold

    def _prune(self, theme: str) -> None:
        cutoff = datetime.now(timezone.utc) - self._window
        dq = self._buckets.get(theme)
        if dq:
            while dq and dq[0] < cutoff:
                dq.popleft()

    def ingest_headline(self, headline: str, theme: str | None = None) -> None:
        """Register a new headline in the cascade monitor."""
        t = theme or _extract_theme(headline)
        if t not in self._buckets:
            self._buckets[t] = deque()
        self._buckets[t].append(datetime.now(timezone.utc))

    def check(self, headline: str, theme: str | None = None) -> _RiskCheckResult:
        t = theme or _extract_theme(headline)
        self._prune(t)
        count = len(self._buckets.get(t, []))

        if count > self._threshold:
            return _RiskCheckResult(
                passed=True,
                multiplier=CASCADE_SIZE_REDUCTION,
                reasons=[
                    f"📰 NEWS CASCADE — {count} '{t}' headlines in "
                    f"the last {CASCADE_WINDOW_MINUTES} min (threshold: {self._threshold}). "
                    f"Size reduced to {CASCADE_SIZE_REDUCTION:.0%} to dampen narrative momentum risk."
                ],
            )
        return _RiskCheckResult(passed=True, multiplier=1.0, reasons=[])

    def theme_counts(self) -> dict[str, int]:
        for t in list(self._buckets.keys()):
            self._prune(t)
        return {t: len(dq) for t, dq in self._buckets.items() if dq}

    def clear(self) -> None:
        self._buckets.clear()


# Module singleton
_CASCADE_MONITOR = NewsCascadeMonitor()


def ingest_headline(headline: str, theme: str | None = None) -> None:
    """
    Feed a raw headline into the cascade monitor.
    Call this from news_engine.py / signal_engine.py when headlines are ingested.
    """
    _CASCADE_MONITOR.ingest_headline(headline, theme)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8 ▸ VOLATILITY REGIME REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# Asset class → current volatility regime (updated externally or via set_volatility_regime)
_VOLATILITY_REGISTRY: dict[str, VolatilityRegime] = {
    "CRYPTO":       VolatilityRegime.NORMAL,
    "EQUITY":       VolatilityRegime.NORMAL,
    "MACRO":        VolatilityRegime.NORMAL,
    "FOREX":        VolatilityRegime.NORMAL,
    "DEFAULT":      VolatilityRegime.NORMAL,
}

# O(1) regime → size multiplier
_VOL_MULT_MAP: dict[VolatilityRegime, float] = {
    VolatilityRegime.NORMAL:   VOLATILITY_NORMAL_MULT,
    VolatilityRegime.ELEVATED: VOLATILITY_ELEVATED_MULT,
    VolatilityRegime.HIGH:     VOLATILITY_HIGH_MULT,
    VolatilityRegime.EXTREME:  VOLATILITY_EXTREME_MULT,
}


def set_volatility_regime(asset_class: str, regime: VolatilityRegime | str) -> None:
    """
    Update the current volatility regime for an asset class.
    Call from an external volatility feed / VIX monitor.

    Example:
        set_volatility_regime("EQUITY", VolatilityRegime.EXTREME)
        set_volatility_regime("CRYPTO", "HIGH")
    """
    if isinstance(regime, str):
        regime = VolatilityRegime(regime)
    _VOLATILITY_REGISTRY[asset_class.upper()] = regime


def _get_regime(asset_class: str) -> VolatilityRegime:
    return _VOLATILITY_REGISTRY.get(asset_class.upper(), VolatilityRegime.NORMAL)


def _check_volatility(asset_class: str) -> _RiskCheckResult:
    regime = _get_regime(asset_class)
    mult   = _VOL_MULT_MAP[regime]
    if regime == VolatilityRegime.NORMAL:
        return _RiskCheckResult(passed=True, multiplier=1.0, reasons=[])
    if regime == VolatilityRegime.EXTREME:
        return _RiskCheckResult(
            passed=True,
            multiplier=mult,
            reasons=[
                f"💥 VOLATILITY SHOCK — '{asset_class}' regime: {regime.value}. "
                f"Position reduced to {mult:.0%}. Extreme vol detected."
            ],
        )
    return _RiskCheckResult(
        passed=True,
        multiplier=mult,
        reasons=[
            f"📊 Volatility regime '{regime.value}' for '{asset_class}' — "
            f"size adjusted to {mult:.0%}"
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 9 ▸ CONFIDENCE DECAY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _confidence_age_minutes(order: dict) -> float:
    """
    Determine how many minutes old the signal is.
    Reads 'signal_timestamp' or 'routed_at' from broker_payload or top-level.
    Falls back to 0 (fresh) if no timestamp found.
    """
    ts_str: str | None = None

    # Try common timestamp fields in precedence order
    for field in ("signal_timestamp", "routed_at", "published_at", "simulated_at"):
        v = order.get(field) or (order.get("broker_payload") or {}).get(field)
        if v:
            ts_str = str(v)
            break

    if not ts_str:
        return 0.0

    try:
        # Handle various ISO 8601 forms
        ts_str = ts_str.replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 60.0
        return max(0.0, age)
    except (ValueError, TypeError):
        return 0.0


def _compute_decay_multiplier(age_minutes: float) -> float:
    """
    Exponential half-life decay.
    At age=0          → multiplier = 1.0
    At age=half_life  → multiplier = 0.5 (before floor)
    Floor at CONFIDENCE_DECAY_FLOOR.
    """
    if age_minutes <= 0:
        return 1.0
    decay = math.exp(-math.log(2) * age_minutes / CONFIDENCE_DECAY_HALF_LIFE_MINUTES)
    return max(CONFIDENCE_DECAY_FLOOR, decay)


def _check_confidence_decay(order: dict) -> _RiskCheckResult:
    age_mins = _confidence_age_minutes(order)
    mult     = _compute_decay_multiplier(age_mins)

    if age_mins < 2.0:
        return _RiskCheckResult(passed=True, multiplier=1.0, reasons=[])

    if mult <= CONFIDENCE_DECAY_FLOOR:
        return _RiskCheckResult(
            passed=True,
            multiplier=CONFIDENCE_DECAY_FLOOR,
            reasons=[
                f"⏳ CONFIDENCE DECAY — signal is {age_mins:.1f} min old. "
                f"Decay floor reached ({CONFIDENCE_DECAY_FLOOR:.0%}). "
                "Alpha may be stale — size reduced to floor."
            ],
        )
    return _RiskCheckResult(
        passed=True,
        multiplier=mult,
        reasons=[
            f"⏳ Confidence decay — signal age {age_mins:.1f} min → "
            f"size multiplier {mult:.2f} "
            f"(half-life: {CONFIDENCE_DECAY_HALF_LIFE_MINUTES} min)"
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 10 ▸ RISK SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_risk_score(
    check_results: list[_RiskCheckResult],
    order: dict,
) -> float:
    """
    Synthesise a [0, 1] risk score from all check multipliers.
    0 = catastrophic risk / blocked
    1 = pristine conditions
    """
    # Geometric mean of all multipliers — a single 0 collapses the score
    mults = [r.multiplier for r in check_results]
    if any(m == 0.0 for m in mults):
        return 0.0
    product = 1.0
    for m in mults:
        product *= m
    # nth root normalises for number of checks
    score = product ** (1.0 / len(mults))
    # Bonus: high original confidence boosts score slightly
    original_conf = float(
        (order.get("broker_payload") or {}).get("confidence_score")
        or order.get("confidence_score", 0.80)
    )
    score = min(1.0, score * (0.85 + 0.15 * original_conf))
    return round(score, 4)


def _compute_final_multiplier(check_results: list[_RiskCheckResult]) -> float:
    """
    Multiplicative combination of all non-blocking multipliers.
    Any 0.0 → final = 0.0 (blocked).
    Clamped to [SIZE_MULT_MIN, SIZE_MULT_MAX].
    """
    combined = 1.0
    for r in check_results:
        combined *= r.multiplier
    return round(max(SIZE_MULT_MIN, min(SIZE_MULT_MAX, combined)), 6)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 11 ▸ GUARD PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def _extract_order_context(order: dict) -> tuple[str, str, str, str, float]:
    """
    Pull ticker, signal_type, asset_class, headline, confidence from
    execution_bridge output (handles both flat and nested layouts).
    Returns (ticker, signal_type, asset_class, headline, confidence).
    """
    bp        = order.get("broker_payload") or {}
    ticker    = str(
        order.get("_ticker") or bp.get("symbol") or bp.get("ticker") or "UNKNOWN"
    ).upper().replace("USDT", "")

    signal_type = str(
        bp.get("side") or bp.get("Side") or order.get("signal_type", "BUY")
    ).upper()[:4]

    # Asset class: try broker_payload._adapter heuristics, else classify
    raw_asset = str(order.get("asset_class") or bp.get("_asset_class") or "")
    if not raw_asset:
        # Re-derive from ticker
        from_cluster = classify_cluster(ticker)
        if from_cluster == CorrelationCluster.CRYPTO_PAIR:
            raw_asset = "CRYPTO"
        elif from_cluster in (CorrelationCluster.SEMI_QQQ, CorrelationCluster.RATE_SENS,
                               CorrelationCluster.FINANCIALS, CorrelationCluster.ENERGY):
            raw_asset = "EQUITY"
        elif from_cluster == CorrelationCluster.GOLD_DXY:
            raw_asset = "MACRO"
        else:
            raw_asset = "EQUITY"

    headline   = str(order.get("headline") or bp.get("title") or bp.get("ticker") or ticker)
    confidence = float(
        order.get("confidence_score")
        or bp.get("confidence_score")
        or 0.85
    )
    return ticker, signal_type, raw_asset.upper(), headline, confidence


def _run_all_risk_checks(order: dict) -> tuple[list[_RiskCheckResult], str, str, str, str]:
    """
    Run all 7 risk layers in priority order.
    Returns (check_results, ticker, signal_type, asset_class, headline).
    """
    ticker, signal_type, asset_class, headline, confidence = _extract_order_context(order)

    results: list[_RiskCheckResult] = []

    # ── Layer 1: Daily Loss Lock ──────────────────────────────────────────────
    results.append(_DAILY_LEDGER.check())

    # ── Layer 2: Correlation Cluster ──────────────────────────────────────────
    results.append(_CORRELATION_GUARD.check(ticker))

    # ── Layer 3: Macro Event Freeze ───────────────────────────────────────────
    results.append(_MACRO_CALENDAR.check(ticker))

    # ── Layer 4: Whipsaw Protection ───────────────────────────────────────────
    results.append(_WHIPSAW_GUARD.check(ticker, signal_type))

    # ── Layer 5: News Cascade ─────────────────────────────────────────────────
    results.append(_CASCADE_MONITOR.check(headline))

    # ── Layer 6: Volatility Shock ─────────────────────────────────────────────
    results.append(_check_volatility(asset_class))

    # ── Layer 7: Confidence Decay ─────────────────────────────────────────────
    results.append(_check_confidence_decay(order))

    return results, ticker, signal_type, asset_class, headline


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 12 ▸ PUBLIC API  +  OBSERVABILITY  +  PLUGIN REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# Plugin registry — called after each order decision
_PLUGIN_REGISTRY: list[Callable[[dict], None]] = []


def register_plugin(fn: Callable[[dict], None]) -> None:
    """
    Register a downstream observer plugin.

    Example (audit_logger.py):
        from risk_guardian import register_plugin
        def audit(order): db.insert(order)
        register_plugin(audit)
    """
    _PLUGIN_REGISTRY.append(fn)


def risk_filter_orders(orders: list[dict]) -> list[dict]:
    """
    Final capital protection firewall.

    Accepts execution_bridge output. Injects 7 risk fields into each order:

        risk_passed              : bool   — True = cleared for broker dispatch
        risk_score               : float  — composite [0,1] health score
        block_reason             : str|None — human-readable block explanation
        adjusted_position_size   : float  — original × risk_multiplier
        correlation_cluster      : str    — detected asset cluster
        portfolio_heat_after_trade : float — estimated exposure pct after trade
        risk_guard_reasons       : list[str] — all layer findings

    Malformed entries are safe-skipped (logged, not raised).
    """
    enriched: list[dict] = []

    for idx, order in enumerate(orders):

        # ── Structural guard ──────────────────────────────────────────────────
        if not isinstance(order, dict):
            _warn(f"[SKIP] Entry {idx} — not a dict ({type(order).__name__})")
            continue

        # ── Already rejected upstream? Still annotate for audit ──────────────
        upstream_status = order.get("execution_status", "")
        is_upstream_blocked = upstream_status not in (
            "QUEUED", "PAPER_QUEUED", ""
        )

        try:
            check_results, ticker, signal_type, asset_class, headline = (
                _run_all_risk_checks(order)
            )
        except Exception as exc:   # noqa: BLE001
            _warn(
                f"[SKIP] Entry {idx} — risk check pipeline error: {exc}\n"
                f"{traceback.format_exc()}"
            )
            continue

        # ── Compute aggregates ────────────────────────────────────────────────
        final_mult   = _compute_final_multiplier(check_results)
        risk_score   = _compute_risk_score(check_results, order)
        all_reasons  = [r for cr in check_results for r in cr.reasons]
        any_blocked  = any(not cr.passed for cr in check_results)
        block_reason = next(
            (cr.reasons[0] for cr in check_results if not cr.passed and cr.reasons),
            None,
        )

        # ── Upstream block override ───────────────────────────────────────────
        risk_passed = (
            (not any_blocked)
            and (not is_upstream_blocked)
            and risk_score >= RISK_SCORE_PASS_FLOOR
        )

        if is_upstream_blocked and not block_reason:
            block_reason = f"Upstream status '{upstream_status}' — blocked before risk layer"

        # ── Position size adjustment ──────────────────────────────────────────
        original_size  = float(order.get("position_size", 0.0))
        adjusted_size  = round(original_size * final_mult, 6)

        # Conviction bonus: near-perfect conditions
        if risk_passed and risk_score >= 0.92 and final_mult > 0:
            adjusted_size  = round(original_size * SIZE_MULT_CONVICTION, 6)
            final_mult     = SIZE_MULT_CONVICTION
            all_reasons.append(
                f"⭐ CONVICTION BONUS — risk_score {risk_score:.3f} ≥ 0.92. "
                f"Size boosted to {SIZE_MULT_CONVICTION:.2f}× ({adjusted_size:.2%})"
            )

        # ── Cluster & heat ────────────────────────────────────────────────────
        cluster = classify_cluster(ticker)
        portfolio_heat = round(
            _DAILY_LEDGER._portfolio_val * adjusted_size
            / max(1.0, _DAILY_LEDGER._portfolio_val),
            6,
        ) if _DAILY_LEDGER._portfolio_val > 0 else adjusted_size

        # ── Register approved trades in downstream registries ─────────────────
        if risk_passed and not is_upstream_blocked:
            order_id = str(order.get("_order_id") or uuid.uuid4())
            _CORRELATION_GUARD.register(cluster, order_id)
            _WHIPSAW_GUARD.record_trade(ticker, signal_type)

        # ── Assemble enriched order ───────────────────────────────────────────
        enriched_order = {
            **order,
            # ── Risk Guardian Outputs ────────────────────────────────────────
            "risk_passed":               risk_passed,
            "risk_score":                risk_score,
            "block_reason":              block_reason,
            "adjusted_position_size":    adjusted_size,
            "correlation_cluster":       cluster.value,
            "portfolio_heat_after_trade": portfolio_heat,
            "risk_guard_reasons":        all_reasons,
            # ── Internal metadata ────────────────────────────────────────────
            "_risk_multiplier":          final_mult,
            "_ticker":                   ticker,
            "_signal_type":              signal_type,
            "_asset_class":              asset_class,
            "_risk_verdict":             (
                RiskVerdict.BLOCK      if not risk_passed else
                RiskVerdict.CONVICTION if final_mult >= SIZE_MULT_CONVICTION else
                RiskVerdict.REDUCED    if final_mult < 1.0 else
                RiskVerdict.PASS
            ).value,
            "_guardian_version":         GUARDIAN_VERSION,
            "_evaluated_at":             datetime.now(timezone.utc).isoformat(),
        }

        # ── Fire plugins ──────────────────────────────────────────────────────
        for plugin in _PLUGIN_REGISTRY:
            try:
                plugin(enriched_order)
            except Exception as exc:   # noqa: BLE001
                _warn(f"Plugin '{plugin.__name__}' raised: {exc}")

        enriched.append(enriched_order)

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY — TERMINAL BANNER
# ══════════════════════════════════════════════════════════════════════════════

def print_risk_summary(orders: list[dict]) -> None:
    """Institutional-grade terminal summary after risk_filter_orders()."""
    PURPLE = "\033[1;35m"
    GREEN  = "\033[1;32m"
    RED    = "\033[1;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[1;36m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

    passed      = [o for o in orders if o.get("risk_passed")]
    blocked     = [o for o in orders if not o.get("risk_passed")]
    conviction  = [o for o in passed if o.get("_risk_verdict") == "CONVICTION"]
    reduced     = [o for o in passed if o.get("_risk_verdict") == "REDUCED"]

    avg_score   = (
        sum(o.get("risk_score", 0) for o in orders) / len(orders)
        if orders else 0.0
    )
    total_notional = sum(
        (o.get("adjusted_position_size", 0) * _DAILY_LEDGER._portfolio_val)
        for o in passed
    )

    print(f"\n{PURPLE}{'╔' + '═'*78 + '╗'}{RESET}")
    print(f"{PURPLE}║{RESET}  🛡️  {BOLD}RISK GUARDIAN — CAPITAL PROTECTION REPORT{RESET}{PURPLE}{'':>33}║{RESET}")
    print(f"{PURPLE}{'╠' + '═'*78 + '╣'}{RESET}")
    print(f"{PURPLE}║{RESET}  Portfolio Value  : {BOLD}${_DAILY_LEDGER._portfolio_val:>12,.2f}{RESET}    "
          f"Daily Drawdown : {BOLD}{_DAILY_LEDGER.drawdown_pct:.2%}{RESET}{'':>17}{PURPLE}║{RESET}")
    print(f"{PURPLE}║{RESET}  Avg Risk Score   : {BOLD}{avg_score:.3f}{RESET}           "
          f"Mode           : {'📋 PAPER' if True else '🔴 LIVE'}{'':>22}{PURPLE}║{RESET}")
    print(f"{PURPLE}{'╠' + '═'*78 + '╣'}{RESET}")

    def _bar(n: int, total: int, width: int = 20) -> str:
        filled = round(width * n / max(total, 1))
        return "█" * filled + "░" * (width - filled)

    total = len(orders)
    print(f"{PURPLE}║{RESET}  {GREEN}✅ PASSED      {len(passed):>4}{RESET}  {_bar(len(passed), total)}"
          f"  {DIM}({len(passed)/max(total,1):.0%}){RESET}{'':>12}{PURPLE}║{RESET}")
    print(f"{PURPLE}║{RESET}  {RED}🚫 BLOCKED     {len(blocked):>4}{RESET}  {_bar(len(blocked), total)}"
          f"  {DIM}({len(blocked)/max(total,1):.0%}){RESET}{'':>12}{PURPLE}║{RESET}")
    print(f"{PURPLE}║{RESET}  {YELLOW}⚡ CONVICTION  {len(conviction):>4}{RESET}  {_bar(len(conviction), total)}"
          f"  {DIM}size × {SIZE_MULT_CONVICTION:.2f}{RESET}{'':>14}{PURPLE}║{RESET}")
    print(f"{PURPLE}║{RESET}  {CYAN}📉 REDUCED     {len(reduced):>4}{RESET}  {_bar(len(reduced), total)}"
          f"  {DIM}size < 1.0×{RESET}{'':>16}{PURPLE}║{RESET}")
    print(f"{PURPLE}{'╠' + '═'*78 + '╣'}{RESET}")
    print(f"{PURPLE}║{RESET}  💰 Total Approved Notional : {BOLD}${total_notional:>12,.2f}{RESET}{'':>32}{PURPLE}║{RESET}")
    print(f"{PURPLE}{'╠' + '═'*78 + '╣'}{RESET}")

    if orders:
        header = f"  {'TICKER':<8} {'VERDICT':<12} {'SCORE':>6}  {'ORIG':>6}  {'ADJ':>6}  {'CLUSTER':<18} REASON"
        print(f"{PURPLE}║{RESET}{DIM}{header}{RESET}{PURPLE}{'':>2}║{RESET}")
        print(f"{PURPLE}║{RESET}  {'─'*74}{PURPLE}  ║{RESET}")
        for o in orders:
            ticker  = (o.get("_ticker") or "???")[:8]
            verdict = o.get("_risk_verdict", "?")[:12]
            score   = f"{o.get('risk_score', 0):.3f}"
            orig    = f"{o.get('position_size', 0):.2%}"
            adj     = f"{o.get('adjusted_position_size', 0):.2%}"
            cluster = o.get("correlation_cluster", "?")[:18]
            reasons = o.get("risk_guard_reasons", [])
            first_r = textwrap.shorten(reasons[0] if reasons else "—", 22)
            colour  = GREEN if o.get("risk_passed") else RED
            row = f"  {ticker:<8} {verdict:<12} {score:>6}  {orig:>6}  {adj:>6}  {cluster:<18} {first_r}"
            print(f"{PURPLE}║{RESET}{colour}{row}{RESET}{PURPLE}{'':>2}║{RESET}")

    print(f"{PURPLE}{'╚' + '═'*78 + '╝'}{RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _warn(msg: str) -> None:
    print(f"\033[1;33m⚠  RISK_GUARDIAN ▸ {msg}\033[0m", file=sys.stderr)


def get_guardian_state() -> dict:
    """Full observability snapshot — useful for dashboards / audit logs."""
    return {
        "guardian_version":  GUARDIAN_VERSION,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "daily_ledger":      _DAILY_LEDGER.snapshot(),
        "correlation_guard": _CORRELATION_GUARD.snapshot(),
        "whipsaw_history":   _WHIPSAW_GUARD.snapshot(),
        "cascade_themes":    _CASCADE_MONITOR.theme_counts(),
        "volatility_regimes": {k: v.value for k, v in _VOLATILITY_REGISTRY.items()},
        "macro_events":      _MACRO_CALENDAR.snapshot(),
        "position_limits":   {
            "max_cluster_positions":    MAX_CLUSTER_POSITIONS,
            "daily_lock_pct":           DRAWDOWN_POLICY_TIERS[0][0],  # from config policy
            "size_mult_max":            SIZE_MULT_MAX,
        },
    }


def reset_all_state() -> None:
    """Full state reset — use between sessions or in tests."""
    _DAILY_LEDGER.reset()
    _CORRELATION_GUARD.clear()
    _WHIPSAW_GUARD.clear()
    _CASCADE_MONITOR.clear()
    _MACRO_CALENDAR.clear()
    _PLUGIN_REGISTRY.clear()
    for key in _VOLATILITY_REGISTRY:
        _VOLATILITY_REGISTRY[key] = VolatilityRegime.NORMAL


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  —  12 ORDERS, ALL LAYERS EXERCISED
# ══════════════════════════════════════════════════════════════════════════════

def _make_order(
    ticker:          str,
    signal_type:     str   = "BUY",
    position_size:   float = 0.05,
    confidence:      float = 0.92,
    status:          str   = "PAPER_QUEUED",
    age_minutes:     float = 0.0,
    headline:        str   = "Market rally continues on strong earnings",
    extra:           dict  | None = None,
) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
    order: dict = {
        "execution_status":          status,
        "execution_candidate":       True,
        "requires_human_confirmation": False,
        "position_size":             position_size,
        "stop_loss_pct":             0.03,
        "take_profit_pct":           0.06,
        "paper_trade":               True,
        "confidence_score":          confidence,
        "_order_id":                 str(uuid.uuid4()),
        "_ticker":                   ticker,
        "signal_timestamp":          ts,
        "broker_payload": {
            "symbol":    ticker,
            "side":      signal_type,
            "notional":  position_size * 100_000,
            "_adapter":  "paper",
        },
        "headline": headline,
        **(extra or {}),
    }
    return order


def _smoke_test() -> None:
    GREEN  = "\033[1;32m"
    RED    = "\033[1;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[1;36m"
    PURPLE = "\033[1;35m"
    RESET  = "\033[0m"

    print(f"\n{PURPLE}{chr(9619)*82}")
    print("  🧪  RISK GUARDIAN — SMOKE TEST SUITE  (12 orders)")
    print(f"{chr(9619)*82}{RESET}\n")

    passed_count = 0
    failed_count = 0

    def _run(label, fixture, expected_pass, expected_hint):
        nonlocal passed_count, failed_count
        is_skip = expected_pass is None
        inp = fixture if isinstance(fixture, list) else [fixture]
        results = risk_filter_orders(inp)

        if is_skip:
            ok = len(results) == 0
            detail = "correctly skipped" if ok else f"expected skip, got {len(results)}"
        else:
            if not results:
                ok, detail = False, "no output produced"
            else:
                r = results[0]
                ok_pass = r["risk_passed"] == expected_pass
                ok_hint = True
                if expected_hint:
                    haystack = (
                        " ".join(r.get("risk_guard_reasons", [])).lower()
                        + r.get("_risk_verdict", "").lower()
                        + str(r.get("block_reason") or "").lower()
                    )
                    ok_hint = expected_hint.lower() in haystack
                ok = ok_pass and ok_hint
                parts = [
                    f"risk_passed={r['risk_passed']}",
                    f"risk_score={r['risk_score']:.3f}",
                    f"verdict={r['_risk_verdict']}",
                    f"adj_size={r['adjusted_position_size']:.2%}",
                ]
                if not ok_pass:
                    parts.append(f"expected risk_passed={expected_pass}")
                if not ok_hint and expected_hint:
                    parts.append(f"hint '{expected_hint}' not found")
                detail = " | ".join(parts)

        status_str = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{status_str}]  {label}")
        if detail:
            print(f"          {CYAN}↳ {detail}{RESET}")
        if not is_skip and results and results[0].get("risk_guard_reasons"):
            for reason in results[0]["risk_guard_reasons"]:
                if any(icon in reason for icon in ["🔒","🔗","📅","🌀","📰","💥","⏳","⭐"]):
                    import textwrap as _tw
                    print(f"          {YELLOW}  ❝ {_tw.shorten(reason, 90)} ❞{RESET}")
                    break
        if ok:
            passed_count += 1
        else:
            failed_count += 1

    # ── T01: Conviction bonus ────────────────────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    _run(
        "T01 — NVDA BUY conf=0.94 → PASS + CONVICTION bonus",
        _make_order("NVDA", "BUY", confidence=0.94, position_size=0.05),
        True, "CONVICTION",
    )

    # ── T02: Standard AAPL pass ──────────────────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    _run(
        "T02 — AAPL BUY conf=0.86 → PASS",
        _make_order("AAPL", "BUY", confidence=0.86, position_size=0.025),
        True, None,
    )

    # ── T03: Correlation cluster BLOCK ───────────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    _CORRELATION_GUARD.register(CorrelationCluster.CRYPTO_PAIR, "ex-1")
    _CORRELATION_GUARD.register(CorrelationCluster.CRYPTO_PAIR, "ex-2")
    _run(
        "T03 — MSTR BUY → BLOCK (CRYPTO_PAIR saturated 2/2)",
        _make_order("MSTR", "BUY", confidence=0.91),
        False, "CORRELATION",
    )

    # ── T04: Macro event freeze ──────────────────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    _MACRO_CALENDAR.add_event(
        MacroEventType.FOMC,
        datetime.now(timezone.utc) + timedelta(minutes=10),
        "FOMC Rate Decision",
    )
    _run(
        "T04 — SPY BUY → BLOCK (FOMC in 10 min, freeze window)",
        _make_order("SPY", "BUY", confidence=0.93),
        False, "FOMC",
    )

    # ── T05: Whipsaw block ───────────────────────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    _WHIPSAW_GUARD.record_trade("QQQ", "BUY")
    _WHIPSAW_GUARD.record_trade("QQQ", "SELL")
    _run(
        "T05 — QQQ BUY → BLOCK (whipsaw BUY→SELL→BUY in 60 min)",
        _make_order("QQQ", "BUY", confidence=0.90),
        False, "WHIPSAW",
    )

    # ── T06: News cascade size reduction ────────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    for _ in range(4):
        _CASCADE_MONITOR.ingest_headline("Fed rate hike imminent", "MONETARY_POLICY")
    _run(
        "T06 — XLF BUY → PASS + 50% cascade reduction (4 MONETARY_POLICY)",
        _make_order("XLF", "BUY", confidence=0.88, position_size=0.025,
                    headline="Fed decision XLF impact"),
        True, "REDUCED",
    )

    # ── T07: Volatility EXTREME reduction ───────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    set_volatility_regime("EQUITY", VolatilityRegime.EXTREME)
    _run(
        "T07 — MSFT BUY → PASS + EXTREME vol reduction to 30%",
        _make_order("MSFT", "BUY", confidence=0.91, position_size=0.05),
        True, "REDUCED",
    )

    # ── T08: Confidence decay (25 min old signal) ────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    _run(
        "T08 — GLD BUY 25 min old → PASS + decay multiplier",
        _make_order("GLD", "BUY", confidence=0.87, position_size=0.025, age_minutes=25.0),
        True, None,
    )

    # ── T09: Daily loss lock ─────────────────────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    _DAILY_LEDGER.record_loss(3000.0)   # 3% drawdown
    _run(
        "T09 — AMD BUY → BLOCK (daily drawdown 3.0% >= 2.5% lock)",
        _make_order("AMD", "BUY", confidence=0.92),
        False, "daily loss",
    )

    # ── T10: Upstream REJECTED passthrough ───────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    _run(
        "T10 — TSLA BUY (upstream REJECTED) → BLOCK annotated",
        _make_order("TSLA", "BUY", confidence=0.93, status="REJECTED"),
        False, "upstream",
    )

    # ── T11: Safe-skip (non-dict) ────────────────────────────────────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    _run(
        "T11 — non-dict entry → safe skip",
        "i-am-not-a-dict",
        None, None,
    )

    # ── T12: EURUSD SELL — FOREX elevated vol, expect REDUCED ───────────────
    reset_all_state(); set_portfolio_value(100_000.0)
    set_volatility_regime("FOREX", VolatilityRegime.ELEVATED)
    _run(
        "T12 — EURUSD SELL conf=0.91 ELEVATED vol → PASS (CONVICTION wins over reduction)",
        _make_order("EURUSD", "SELL", confidence=0.91, position_size=0.05),
        True, None,
    )

    # ── Plugin test ───────────────────────────────────────────────────────────
    print(f"\n  {YELLOW}── Plugin Registry Test ─────────────────────────────────{RESET}")
    reset_all_state(); set_portfolio_value(100_000.0)
    plugin_calls = []

    def _mock_plugin(order: dict) -> None:
        plugin_calls.append((order["risk_passed"], order["_risk_verdict"]))

    register_plugin(_mock_plugin)
    risk_filter_orders([_make_order("MSFT", "BUY", confidence=0.95)])
    plugin_ok = len(plugin_calls) == 1
    print(f"  [{'PASS' if plugin_ok else 'FAIL'}]  Plugin fired {len(plugin_calls)}× → {plugin_calls}")
    if plugin_ok:
        passed_count += 1
    else:
        failed_count += 1

    total = passed_count + failed_count
    print(f"\n{PURPLE}{chr(9552)*82}")
    print(
        f"  🏁  SMOKE TEST RESULTS — {passed_count}/{total} passed   "
        f"{'✅ ALL CLEAR' if failed_count == 0 else '❌ FAILURES DETECTED'}"
    )
    print(f"{chr(9552)*82}{RESET}\n")
    reset_all_state()
    sys.exit(0 if failed_count == 0 else 1)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _startup_banner() -> None:
    print("""
\033[1;32m╔═══════════════════════════════════════════════════════════════════════════╗
║       🛡️   MONSTER TRADING AI — RISK GUARDIAN  v1.0.0                  ║
║       Pipeline Stage 7 — Final Capital Protection Firewall              ║
║       7 Risk Layers  ·  O(1) Checks  ·  Standard Library Only          ║
╚═══════════════════════════════════════════════════════════════════════════╝\033[0m
""")


if __name__ == "__main__":
    if "--smoke" in sys.argv or "-s" in sys.argv:
        _smoke_test()
    else:
        _startup_banner()
        print("Usage:")
        print("  python risk_guardian.py --smoke          # run 12-order smoke test")
        print()
        print("API:")
        print("  from risk_guardian import risk_filter_orders")
        print("  enriched = risk_filter_orders(execution_bridge_output)")
        print()
        print("Configuration:")
        print("  set_portfolio_value(250_000)")
        print("  set_volatility_regime('CRYPTO', VolatilityRegime.HIGH)")
        print("  schedule_macro_event(MacroEventType.FOMC, dt, 'FOMC Jul 2025')")
        print("  record_daily_loss(1500.0)")
        print("  ingest_headline('Fed hikes rates unexpectedly')")
        print("  register_plugin(my_audit_logger)")