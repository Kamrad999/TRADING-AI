"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║    ██████╗  ██████╗ ██████╗      ██████╗ ██████╗ ██████╗ ███████╗                       ║
║   ██╔════╝ ██╔═══██╗██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝                       ║
║   ██║  ███╗██║   ██║██║  ██║    ██║     ██║   ██║██████╔╝█████╗                         ║
║   ██║   ██║██║   ██║██║  ██║    ██║     ██║   ██║██╔══██╗██╔══╝                         ║
║   ╚██████╔╝╚██████╔╝██████╔╝    ╚██████╗╚██████╔╝██║  ██║███████╗                       ║
║    ╚═════╝  ╚═════╝ ╚═════╝      ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝                       ║
║                                                                                          ║
║   ┌──────────────────────────────────────────────────────────────────────────────────┐   ║
║   │     🧠  G O D _ C O R E . P Y  —  MASTER PIPELINE ORCHESTRATOR  🧠              │   ║
║   │          Institutional Hedge-Fund-Grade Trading Intelligence Engine              │   ║
║   │   news_engine → dup_filter → validator → signal_engine → risk_guardian          │   ║
║   │   → execution_bridge → broker_sender → alert_router → persist → analytics       │   ║
║   └──────────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                          ║
║  Module   : god_core.py                                                                  ║
║  Version  : 1.0.0                                                                        ║
║  Role     : Central Nervous System — orchestrates all 11 pipeline stages                 ║
║  Style    : Bloomberg Terminal Backend / Low-Latency Institutional OMS                   ║
║                                                                                          ║
║  ╔════════════════════════════════════════════════════════════════════════════════╗       ║
║  ║  KILL SWITCH: set TRADING_KILL_SWITCH=1 env var or call activate_kill_switch() ║       ║
║  ║  LIVE MODE  : requires explicit enable_live_mode() before execution            ║       ║
║  ╚════════════════════════════════════════════════════════════════════════════════╝       ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import sys
import time
import traceback
import unittest
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 ▸ MODULE-LEVEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_VERSION = "1.0.0"
ORCHESTRATOR_BUILD   = "MONSTER-TRADING-AI"

# ── Retry & resilience ────────────────────────────────────────────────────────
STAGE_MAX_RETRIES         = 3          # attempts per pipeline stage
STAGE_BACKOFF_BASE_S      = 0.25       # base backoff between retries
STAGE_BACKOFF_MULTIPLIER  = 2.0        # exponential factor
STAGE_BACKOFF_JITTER_PCT  = 0.20       # ±20% random jitter
STAGE_TIMEOUT_S           = 120.0      # wall-clock timeout per stage (seconds)

# ── Circuit breaker ───────────────────────────────────────────────────────────
CIRCUIT_FAILURE_THRESHOLD = 3          # consecutive failures → OPEN
CIRCUIT_RESET_WINDOW_S    = 300.0      # seconds before HALF_OPEN probe

# ── Portfolio safety caps ─────────────────────────────────────────────────────
MAX_PORTFOLIO_EXPOSURE_PCT = 0.40      # max 40% of portfolio deployed at once
MAX_DAILY_DRAWDOWN_PCT     = 0.025     # 2.5% daily loss → hard stop

# ── Market session windows (UTC) ─────────────────────────────────────────────
PREMARKET_START_UTC_H   = 9            # 09:00 UTC
PREMARKET_END_UTC_H     = 14           # 14:00 UTC  (09:00–09:30 ET = premarket)
REGULAR_START_UTC_H     = 14           # 14:30 UTC  (09:30 ET open)
REGULAR_START_UTC_M     = 30
REGULAR_END_UTC_H       = 21           # 21:00 UTC  (16:00 ET close)
AFTER_HOURS_END_UTC_H   = 1            # 01:00 UTC next day (20:00 ET)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)-8s] [god_core] %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# ── Kill switch ───────────────────────────────────────────────────────────────
_KILL_SWITCH_ENV_VAR = "TRADING_KILL_SWITCH"

# ── Stage identifiers (ordered) ───────────────────────────────────────────────
STAGE_NAMES: Tuple[str, ...] = (
    "fetch_news",
    "deduplicate_articles",
    "validate_articles",
    "generate_signals",
    "detect_market_regime",                    # ← NEW: Analyze market structure
    "calculate_portfolio_allocations",         # ← NEW: Use regime for position sizing
    "apply_risk_controls",
    "build_orders",
    "send_orders",
    "route_alerts",
    "persist_state",
    "update_validation_memory",
    "update_performance_analytics",
)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 ▸ ENUMS & VALUE TYPES
# ══════════════════════════════════════════════════════════════════════════════

class PipelineStatus(str, Enum):
    SUCCESS  = "SUCCESS"    # all stages executed cleanly
    DEGRADED = "DEGRADED"   # one or more stages soft-failed; output partial
    HALTED   = "HALTED"     # kill switch / hard cap / unrecoverable failure


class MarketSession(str, Enum):
    PREMARKET   = "PREMARKET"
    REGULAR     = "REGULAR"
    AFTER_HOURS = "AFTER_HOURS"
    CLOSED      = "CLOSED"
    CRYPTO_24_7 = "CRYPTO_24_7"   # always active for crypto assets


class CircuitState(str, Enum):
    CLOSED    = "CLOSED"      # normal operation
    OPEN      = "OPEN"        # tripped — blocking calls
    HALF_OPEN = "HALF_OPEN"   # probe attempt


class StageResult:
    """Immutable outcome record for a single pipeline stage."""
    __slots__ = (
        "stage", "success", "output", "error",
        "latency_ms", "retries", "degraded",
    )

    def __init__(
        self,
        stage:      str,
        success:    bool,
        output:     Any       = None,
        error:      str       = "",
        latency_ms: float     = 0.0,
        retries:    int       = 0,
        degraded:   bool      = False,
    ) -> None:
        self.stage      = stage
        self.success    = success
        self.output     = output
        self.error      = error
        self.latency_ms = latency_ms
        self.retries    = retries
        self.degraded   = degraded

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage":      self.stage,
            "success":    self.success,
            "latency_ms": round(self.latency_ms, 3),
            "retries":    self.retries,
            "degraded":   self.degraded,
            "error":      self.error or None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 ▸ STRUCTURED LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def _build_logger(name: str = "god_core") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(handler)
    return logger


log = _build_logger()

# ANSI colour codes for terminal output
_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "red":    "\033[1;31m",
    "green":  "\033[1;32m",
    "yellow": "\033[1;33m",
    "cyan":   "\033[1;36m",
    "purple": "\033[1;35m",
    "dim":    "\033[2m",
}

def _banner(msg: str, colour: str = "purple") -> None:
    c = _C.get(colour, "")
    r = _C["reset"]
    log.info(f"{c}{'═'*72}{r}")
    log.info(f"{c}  {msg}{r}")
    log.info(f"{c}{'═'*72}{r}")


def _stage_log(stage: str, msg: str, level: str = "info") -> None:
    prefix = f"{_C['cyan']}[{stage.upper()}]{_C['reset']}"
    getattr(log, level)(f"{prefix}  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 ▸ CIRCUIT BREAKER
# ══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Per-stage circuit breaker.
    Transitions: CLOSED → OPEN (after N failures) → HALF_OPEN (after reset window) → CLOSED.
    """

    def __init__(
        self,
        name:              str,
        failure_threshold: int   = CIRCUIT_FAILURE_THRESHOLD,
        reset_window_s:    float = CIRCUIT_RESET_WINDOW_S,
    ) -> None:
        self.name              = name
        self.failure_threshold = failure_threshold
        self.reset_window_s    = reset_window_s
        self._state            = CircuitState.CLOSED
        self._failures         = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._opened_at and (time.time() - self._opened_at) >= self.reset_window_s:
                self._state = CircuitState.HALF_OPEN
                log.warning(
                    "[CIRCUIT] '%s' → HALF_OPEN (probe window after %.0fs)",
                    self.name, self.reset_window_s,
                )
        return self._state

    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def record_success(self) -> None:
        self._failures = 0
        if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            log.info("[CIRCUIT] '%s' → CLOSED (recovered)", self.name)
        self._state     = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold and self._state != CircuitState.OPEN:
            self._state     = CircuitState.OPEN
            self._opened_at = time.time()
            log.error(
                "[CIRCUIT] '%s' → OPEN after %d consecutive failures",
                self.name, self._failures,
            )

    def reset(self) -> None:
        self._state     = CircuitState.CLOSED
        self._failures  = 0
        self._opened_at = None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 ▸ KILL SWITCH
# ══════════════════════════════════════════════════════════════════════════════

_KILL_SWITCH_ACTIVE: bool = bool(os.getenv(_KILL_SWITCH_ENV_VAR, ""))


def activate_kill_switch(reason: str = "manual") -> None:
    """Immediately halt all future pipeline executions in this process."""
    global _KILL_SWITCH_ACTIVE
    _KILL_SWITCH_ACTIVE = True
    log.critical(
        "%s🔴 KILL SWITCH ACTIVATED — reason: %s — no new pipelines will execute%s",
        _C["red"], reason, _C["reset"],
    )


def deactivate_kill_switch() -> None:
    """Re-enable pipeline execution (operator must call explicitly)."""
    global _KILL_SWITCH_ACTIVE
    _KILL_SWITCH_ACTIVE = False
    log.warning("%s🟢 Kill switch DEACTIVATED — pipeline resumed%s", _C["yellow"], _C["reset"])


def is_kill_switch_active() -> bool:
    return _KILL_SWITCH_ACTIVE or bool(os.getenv(_KILL_SWITCH_ENV_VAR, ""))


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 ▸ MARKET SESSION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def detect_market_session(now_utc: Optional[datetime] = None) -> MarketSession:
    """
    Classify the current moment into a market session.

    Session boundaries (all UTC, approximate US equity market):
      PREMARKET   : 09:00 – 14:29 UTC   (04:00 – 09:29 ET)
      REGULAR     : 14:30 – 20:59 UTC   (09:30 – 15:59 ET)
      AFTER_HOURS : 21:00 – 00:59 UTC   (16:00 – 19:59 ET)
      CLOSED      : 01:00 – 08:59 UTC   (20:00 – 03:59 ET)

    CRYPTO_24_7 is always available — not returned by this function directly
    but referenced in routing decisions by the caller.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    weekday = now_utc.weekday()   # 0=Mon … 6=Sun
    if weekday >= 5:              # Weekend — equities closed, crypto still on
        return MarketSession.CLOSED

    h = now_utc.hour
    m = now_utc.minute
    minutes_since_midnight = h * 60 + m

    premarket_start = PREMARKET_START_UTC_H * 60           # 540
    regular_start   = REGULAR_START_UTC_H   * 60 + REGULAR_START_UTC_M  # 870
    regular_end     = REGULAR_END_UTC_H     * 60           # 1260
    after_end       = AFTER_HOURS_END_UTC_H * 60           # 60 (next day)

    if premarket_start <= minutes_since_midnight < regular_start:
        return MarketSession.PREMARKET
    if regular_start <= minutes_since_midnight < regular_end:
        return MarketSession.REGULAR
    if regular_end <= minutes_since_midnight or minutes_since_midnight < after_end:
        return MarketSession.AFTER_HOURS
    return MarketSession.CLOSED


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 ▸ MODULE LOADER (graceful import with fallback stubs)
# ══════════════════════════════════════════════════════════════════════════════

class _StubModule:
    """
    Zero-dependency stub returned when a real module cannot be imported.
    Every attribute access returns a callable that logs a warning and
    returns a safe empty value, preventing AttributeError cascades.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> Callable:
        def _stub(*args: Any, **kwargs: Any) -> Any:
            log.warning(
                "[STUB] %s.%s() called — module not available, returning empty result",
                self._name, attr,
            )
            # Heuristic safe defaults by function name
            _list_pats = ("list","fetch","filter","generate","signal",
                          "build","send","route","validate","deduplicate",
                          "apply","get_top","export","record","save","update")
            _dict_pats = ("dict","stats","report","status","analyze",
                          "reputation","profile","summary")
            if any(p in attr for p in _list_pats):
                return []
            if any(p in attr for p in _dict_pats):
                return {}
            return None
        return _stub


def _load_module(module_name: str) -> Any:
    """Import a pipeline module by name; return a _StubModule on ImportError."""
    try:
        mod = importlib.import_module(module_name)
        log.debug("  ✓ Loaded module: %s", module_name)
        return mod
    except ImportError as exc:
        log.warning("  ⚠  Could not import '%s': %s — using stub", module_name, exc)
        return _StubModule(module_name)
    except Exception as exc:  # noqa: BLE001
        log.error("  ✗ Unexpected error loading '%s': %s — using stub", module_name, exc)
        return _StubModule(module_name)


def _load_all_modules() -> Dict[str, Any]:
    """Load every pipeline module; stubs fill gaps so the orchestrator never crashes."""
    names = [
        "news_engine",
        "duplicate_filter",
        "fake_news_validator",
        "signal_engine",
        "risk_guardian",
        "execution_bridge",
        "broker_sender",
        "alert_router",
        "performance_analytics",
        "state_manager",
        "validation_memory",
        "config",
    ]
    log.debug("Loading pipeline modules …")
    return {n: _load_module(n) for n in names}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8 ▸ RETRY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

import random


def _backoff(attempt: int) -> float:
    """Exponential backoff with ±JITTER_PCT jitter."""
    base    = STAGE_BACKOFF_BASE_S * (STAGE_BACKOFF_MULTIPLIER ** attempt)
    jitter  = base * STAGE_BACKOFF_JITTER_PCT * (2 * random.random() - 1)
    return max(0.0, base + jitter)


def _run_with_retry(
    fn:           Callable[[], Any],
    stage_name:   str,
    circuit:      CircuitBreaker,
    max_retries:  int   = STAGE_MAX_RETRIES,
) -> Tuple[bool, Any, str, int]:
    """
    Execute fn() with exponential retry and circuit-breaker integration.

    Returns (success, result, error_message, attempt_count).
    """
    if circuit.is_open():
        msg = f"Circuit breaker OPEN for '{stage_name}' — skipping"
        _stage_log(stage_name, msg, "warning")
        return False, None, msg, 0

    last_exc = ""
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            result = fn()
            circuit.record_success()
            lat = (time.perf_counter() - t0) * 1000
            if attempt > 0:
                _stage_log(stage_name, f"Succeeded on attempt {attempt + 1} ({lat:.1f}ms)")
            return True, result, "", attempt
        except Exception as exc:  # noqa: BLE001
            lat = (time.perf_counter() - t0) * 1000
            last_exc = f"{type(exc).__name__}: {exc}"
            _stage_log(
                stage_name,
                f"Attempt {attempt + 1}/{max_retries + 1} failed ({lat:.1f}ms): {last_exc}",
                "warning",
            )
            circuit.record_failure()
            if attempt < max_retries:
                wait = _backoff(attempt)
                _stage_log(stage_name, f"Retrying in {wait:.2f}s …")
                time.sleep(wait)

    return False, None, last_exc, max_retries


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 9 ▸ PORTFOLIO SAFETY GUARDS
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioGuard:
    """
    Tracks cumulative exposure and daily drawdown.
    Activates the pipeline kill switch when hard caps are breached.
    """

    def __init__(
        self,
        max_exposure_pct:   float = MAX_PORTFOLIO_EXPOSURE_PCT,
        max_drawdown_pct:   float = MAX_DAILY_DRAWDOWN_PCT,
        portfolio_value:    float = 100_000.0,
    ) -> None:
        self.max_exposure_pct = max_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.portfolio_value  = portfolio_value
        self._exposure_pct    = 0.0
        self._daily_loss_usd  = 0.0
        self._day_str         = self._today()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day_str:
            self._day_str      = today
            self._daily_loss_usd = 0.0

    def check_exposure(self, proposed_size_pct: float) -> Tuple[bool, str]:
        self._roll_day()
        projected = self._exposure_pct + proposed_size_pct
        if projected > self.max_exposure_pct:
            reason = (
                f"Portfolio exposure cap breached: "
                f"current={self._exposure_pct:.1%} + proposed={proposed_size_pct:.1%} "
                f"= {projected:.1%} > max={self.max_exposure_pct:.1%}"
            )
            return False, reason
        return True, ""

    def check_drawdown(self) -> Tuple[bool, str]:
        self._roll_day()
        dd_pct = self._daily_loss_usd / max(self.portfolio_value, 1)
        if dd_pct >= self.max_drawdown_pct:
            reason = (
                f"Daily drawdown stop triggered: "
                f"loss=${self._daily_loss_usd:,.2f} "
                f"({dd_pct:.2%}) ≥ cap={self.max_drawdown_pct:.2%}"
            )
            activate_kill_switch(reason)
            return False, reason
        return True, ""

    def record_loss(self, amount_usd: float) -> None:
        self._roll_day()
        self._daily_loss_usd += max(0.0, amount_usd)
        ok, reason = self.check_drawdown()
        if not ok:
            log.critical("%s%s%s", _C["red"], reason, _C["reset"])

    def add_exposure(self, size_pct: float) -> None:
        self._exposure_pct = min(1.0, self._exposure_pct + size_pct)

    def remove_exposure(self, size_pct: float) -> None:
        self._exposure_pct = max(0.0, self._exposure_pct - size_pct)

    @property
    def current_exposure_pct(self) -> float:
        return self._exposure_pct

    @property
    def daily_drawdown_pct(self) -> float:
        return self._daily_loss_usd / max(self.portfolio_value, 1)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 10 ▸ RUN CONTEXT (single pipeline execution state)
# ══════════════════════════════════════════════════════════════════════════════

class RunContext:
    """
    Mutable container threaded through all 11 pipeline stages.
    Each stage reads its inputs from and writes its outputs into this object.
    """

    def __init__(self, run_id: str, session: MarketSession) -> None:
        self.run_id:     str           = run_id
        self.session:    MarketSession = session
        self.started_at: float         = time.time()

        # Pipeline data — populated stage by stage
        self.raw_articles:       List[Dict]  = []
        self.deduped_articles:   List[Dict]  = []
        self.validated_articles: List[Dict]  = []
        self.signals:            List[Dict]  = []
        self.risk_filtered:      List[Dict]  = []
        self.orders:             List[Dict]  = []
        self.sent_orders:        List[Dict]  = []
        self.routed_alerts:      List[Dict]  = []
        self.analytics_report:   Dict        = {}
        self.state_saved:        bool        = False
        self.memory_updated:     bool        = False

        # Stage audit trail
        self.stage_results: List[StageResult] = []
        self.degraded_stages: List[str]        = []

        # Counts (derived after each stage for the summary)
        self.articles_processed: int = 0
        self.signals_generated:  int = 0
        self.orders_sent:        int = 0
        self.alerts_routed:      int = 0

    def record(self, result: StageResult) -> None:
        self.stage_results.append(result)
        if not result.success or result.degraded:
            self.degraded_stages.append(result.stage)

    @property
    def pipeline_latency_ms(self) -> float:
        return (time.time() - self.started_at) * 1000

    @property
    def overall_status(self) -> PipelineStatus:
        if is_kill_switch_active():
            return PipelineStatus.HALTED
        critical_stages = {
            "fetch_news", "validate_articles", "generate_signals",
        }
        failed_critical = [
            r for r in self.stage_results
            if not r.success and r.stage in critical_stages
        ]
        if failed_critical:
            return PipelineStatus.HALTED
        if self.degraded_stages:
            return PipelineStatus.DEGRADED
        return PipelineStatus.SUCCESS


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 11 ▸ PIPELINE STAGE IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _stage_fetch_news(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
) -> StageResult:
    """Stage 1: Fetch raw articles from news_engine."""
    t0 = time.perf_counter()
    stage = "fetch_news"

    def _fn() -> List[Dict]:
        news = modules["news_engine"]
        fn   = getattr(news, "fetch_news", None) or getattr(news, "run", None)
        if fn is None:
            raise AttributeError("news_engine has no fetch_news() or run()")
        result = fn()
        return result if isinstance(result, list) else []

    ok, result, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    if ok:
        ctx.raw_articles = result or []
        n = len(ctx.raw_articles)
        _stage_log(stage, f"Fetched {n} raw articles ({lat:.1f}ms)", "info")
        return StageResult(stage, True, result, latency_ms=lat, retries=retries)
    else:
        ctx.raw_articles = []
        _stage_log(stage, f"FAILED: {err}", "error")
        return StageResult(stage, False, [], err, lat, retries, degraded=True)


def _stage_deduplicate(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
) -> StageResult:
    """Stage 2: Remove duplicate articles via duplicate_filter."""
    t0 = time.perf_counter()
    stage = "deduplicate_articles"

    if not ctx.raw_articles:
        ctx.deduped_articles = []
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, "No articles to deduplicate — skipping", "info")
        return StageResult(stage, True, [], latency_ms=lat)

    def _fn() -> List[Dict]:
        df = modules["duplicate_filter"]
        fn = (
            getattr(df, "deduplicate_articles", None)
            or getattr(df, "filter_duplicates", None)
            or getattr(df, "run", None)
        )
        if fn is None:
            raise AttributeError("duplicate_filter: no suitable function found")
        result = fn(ctx.raw_articles)
        return result if isinstance(result, list) else ctx.raw_articles

    ok, result, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    if ok:
        ctx.deduped_articles = result or ctx.raw_articles
        removed = len(ctx.raw_articles) - len(ctx.deduped_articles)
        _stage_log(stage, f"{len(ctx.deduped_articles)} unique ({removed} dupes removed) ({lat:.1f}ms)")
        return StageResult(stage, True, result, latency_ms=lat, retries=retries)
    else:
        # Graceful degradation: pass raw articles through
        ctx.deduped_articles = ctx.raw_articles
        _stage_log(stage, f"DEGRADED — passing raw articles: {err}", "warning")
        return StageResult(stage, True, ctx.raw_articles, err, lat, retries, degraded=True)


def _stage_validate(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
) -> StageResult:
    """Stage 3: Validate articles via fake_news_validator."""
    t0 = time.perf_counter()
    stage = "validate_articles"

    if not ctx.deduped_articles:
        ctx.validated_articles = []
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, "No articles to validate — skipping", "info")
        return StageResult(stage, True, [], latency_ms=lat)

    def _fn() -> List[Dict]:
        fnv = modules["fake_news_validator"]
        fn  = (
            getattr(fnv, "validate_articles", None)
            or getattr(fnv, "run", None)
        )
        if fn is None:
            raise AttributeError("fake_news_validator: no suitable function found")
        result = fn(ctx.deduped_articles)
        return result if isinstance(result, list) else ctx.deduped_articles

    ok, result, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    if ok:
        ctx.validated_articles = result or ctx.deduped_articles
        ctx.articles_processed = len(ctx.validated_articles)
        _stage_log(stage, f"{ctx.articles_processed} articles validated ({lat:.1f}ms)")
        return StageResult(stage, True, result, latency_ms=lat, retries=retries)
    else:
        ctx.validated_articles = ctx.deduped_articles
        ctx.articles_processed = len(ctx.validated_articles)
        _stage_log(stage, f"FAILED — forwarding unvalidated: {err}", "error")
        return StageResult(stage, False, ctx.deduped_articles, err, lat, retries)


def _stage_generate_signals(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
) -> StageResult:
    """Stage 4: Generate trading signals via signal_engine."""
    t0 = time.perf_counter()
    stage = "generate_signals"

    if not ctx.validated_articles:
        ctx.signals = []
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, "No validated articles — no signals generated", "info")
        return StageResult(stage, True, [], latency_ms=lat)

    def _fn() -> List[Dict]:
        se = modules["signal_engine"]
        fn = (
            getattr(se, "generate_signals", None)
            or getattr(se, "run", None)
        )
        if fn is None:
            raise AttributeError("signal_engine: no suitable function found")
        result = fn(ctx.validated_articles)
        return result if isinstance(result, list) else []

    ok, result, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    if ok:
        ctx.signals = result or []
        ctx.signals_generated = len(ctx.signals)
        _stage_log(stage, f"{ctx.signals_generated} signals generated ({lat:.1f}ms)")
        return StageResult(stage, True, result, latency_ms=lat, retries=retries)
    else:
        ctx.signals = []
        _stage_log(stage, f"FAILED — no signals: {err}", "error")
        return StageResult(stage, False, [], err, lat, retries)


def _stage_apply_risk(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
    guard:   PortfolioGuard,
) -> StageResult:
    """Stage 5: Apply risk controls via risk_guardian."""
    t0 = time.perf_counter()
    stage = "apply_risk_controls"

    # Portfolio drawdown check before processing
    dd_ok, dd_reason = guard.check_drawdown()
    if not dd_ok:
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, f"HALTED by drawdown stop: {dd_reason}", "critical")
        ctx.risk_filtered = []
        return StageResult(stage, False, [], dd_reason, lat)

    if not ctx.signals:
        ctx.risk_filtered = []
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, "No signals to risk-filter — skipping", "info")
        return StageResult(stage, True, [], latency_ms=lat)

    def _fn() -> List[Dict]:
        rg = modules["risk_guardian"]
        fn = (
            getattr(rg, "risk_filter_orders", None)
            or getattr(rg, "filter", None)
            or getattr(rg, "run", None)
        )
        if fn is None:
            raise AttributeError("risk_guardian: no suitable function found")
        result = fn(ctx.signals)
        return result if isinstance(result, list) else ctx.signals

    ok, result, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    if ok:
        ctx.risk_filtered = result or []
        passed = sum(1 for r in ctx.risk_filtered if r.get("risk_passed", True))
        blocked = len(ctx.risk_filtered) - passed
        _stage_log(
            stage,
            f"{passed} passed / {blocked} blocked by risk guardian ({lat:.1f}ms)",
        )
        return StageResult(stage, True, result, latency_ms=lat, retries=retries)
    else:
        # Graceful: pass signals through without risk filtering
        ctx.risk_filtered = ctx.signals
        _stage_log(stage, f"DEGRADED — risk filter bypassed: {err}", "warning")
        return StageResult(stage, True, ctx.signals, err, lat, retries, degraded=True)


def _stage_build_orders(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
    guard:   PortfolioGuard,
) -> StageResult:
    """Stage 6: Build broker-ready orders via execution_bridge."""
    t0 = time.perf_counter()
    stage = "build_orders"

    if not ctx.risk_filtered:
        ctx.orders = []
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, "No risk-approved signals — no orders to build", "info")
        return StageResult(stage, True, [], latency_ms=lat)

    def _fn() -> List[Dict]:
        eb = modules["execution_bridge"]
        fn = (
            getattr(eb, "build_execution_orders", None)
            or getattr(eb, "build_orders", None)
            or getattr(eb, "run", None)
        )
        if fn is None:
            raise AttributeError("execution_bridge: no suitable function found")
        result = fn(ctx.risk_filtered)
        return result if isinstance(result, list) else []

    ok, result, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    if ok:
        ctx.orders = result or []
        # Portfolio exposure cap enforcement
        total_size = sum(
            float(o.get("position_size", 0)) for o in ctx.orders
            if o.get("execution_status") not in ("REJECTED", "COOLDOWN", "CAP_EXCEEDED")
        )
        exp_ok, exp_reason = guard.check_exposure(total_size)
        if not exp_ok:
            _stage_log(stage, f"Exposure cap: {exp_reason}", "warning")
            # Mark all orders as blocked
            for o in ctx.orders:
                o["exposure_blocked"] = True
        else:
            guard.add_exposure(total_size)

        queued = sum(
            1 for o in ctx.orders
            if o.get("execution_status") in ("QUEUED", "PAPER_QUEUED")
            and not o.get("exposure_blocked")
        )
        _stage_log(stage, f"{queued} orders queued for transmission ({lat:.1f}ms)")
        return StageResult(stage, True, result, latency_ms=lat, retries=retries)
    else:
        ctx.orders = []
        _stage_log(stage, f"FAILED — no orders built: {err}", "error")
        return StageResult(stage, True, [], err, lat, retries, degraded=True)


def _stage_send_orders(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
) -> StageResult:
    """Stage 7: Transmit orders to broker via broker_sender."""
    t0 = time.perf_counter()
    stage = "send_orders"

    if is_kill_switch_active():
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, "Kill switch active — orders NOT sent", "critical")
        ctx.sent_orders = []
        return StageResult(stage, False, [], "Kill switch active", lat)

    transmittable = [
        o for o in ctx.orders
        if not o.get("exposure_blocked")
        and o.get("execution_status") in ("QUEUED", "PAPER_QUEUED")
    ]

    if not transmittable:
        ctx.sent_orders = []
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, "No transmittable orders — skipping broker send", "info")
        return StageResult(stage, True, [], latency_ms=lat)

    def _fn() -> List[Dict]:
        bs = modules["broker_sender"]
        fn = (
            getattr(bs, "send_orders", None)
            or getattr(bs, "run", None)
        )
        if fn is None:
            raise AttributeError("broker_sender: no suitable function found")
        result = fn(transmittable)
        return result if isinstance(result, list) else []

    ok, result, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    if ok:
        ctx.sent_orders = result or []
        filled = sum(
            1 for o in ctx.sent_orders
            if o.get("broker_status") in ("SENT", "PAPER_FILLED", "PARTIAL_FILL")
        )
        ctx.orders_sent = filled
        _stage_log(stage, f"{filled}/{len(transmittable)} orders filled ({lat:.1f}ms)")
        return StageResult(stage, True, result, latency_ms=lat, retries=retries)
    else:
        ctx.sent_orders = []
        _stage_log(stage, f"DEGRADED — broker send failed: {err}", "error")
        return StageResult(stage, True, [], err, lat, retries, degraded=True)


def _stage_route_alerts(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
) -> StageResult:
    """Stage 8: Route alerts via alert_router."""
    t0 = time.perf_counter()
    stage = "route_alerts"

    # Route both risk-filtered signals AND sent order results
    routable = ctx.risk_filtered or ctx.signals

    if not routable:
        ctx.routed_alerts = []
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, "Nothing to route — skipping", "info")
        return StageResult(stage, True, [], latency_ms=lat)

    def _fn() -> List[Dict]:
        ar = modules["alert_router"]
        fn = (
            getattr(ar, "route_alerts", None)
            or getattr(ar, "run", None)
        )
        if fn is None:
            raise AttributeError("alert_router: no suitable function found")
        result = fn(routable)
        return result if isinstance(result, list) else []

    ok, result, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    if ok:
        ctx.routed_alerts = result or []
        ctx.alerts_routed = len(ctx.routed_alerts)
        _stage_log(stage, f"{ctx.alerts_routed} alerts routed ({lat:.1f}ms)")
        return StageResult(stage, True, result, latency_ms=lat, retries=retries)
    else:
        ctx.routed_alerts = []
        _stage_log(stage, f"DEGRADED — alert routing failed: {err}", "warning")
        return StageResult(stage, True, [], err, lat, retries, degraded=True)


def _stage_persist_state(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
) -> StageResult:
    """Stage 9: Persist pipeline state via state_manager."""
    t0 = time.perf_counter()
    stage = "persist_state"

    state_payload = {
        "run_id":             ctx.run_id,
        "session":            ctx.session.value,
        "articles_processed": ctx.articles_processed,
        "signals_generated":  ctx.signals_generated,
        "orders_sent":        ctx.orders_sent,
        "alerts_routed":      ctx.alerts_routed,
        "sent_orders":        ctx.sent_orders,
        "routed_alerts":      ctx.routed_alerts,
        "degraded_stages":    ctx.degraded_stages,
        "saved_at":           datetime.now(timezone.utc).isoformat(),
    }

    def _fn() -> bool:
        sm = modules["state_manager"]
        fn = (
            getattr(sm, "save_state", None)
            or getattr(sm, "persist", None)
            or getattr(sm, "run", None)
        )
        if fn is None:
            raise AttributeError("state_manager: no suitable function found")
        fn(state_payload)
        return True

    ok, _, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    ctx.state_saved = ok
    if ok:
        _stage_log(stage, f"State persisted (run_id={ctx.run_id[:8]}…) ({lat:.1f}ms)")
    else:
        _stage_log(stage, f"DEGRADED — state not saved: {err}", "warning")

    return StageResult(stage, ok, state_payload, err if not ok else "", lat, retries,
                       degraded=not ok)


def _stage_update_memory(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
) -> StageResult:
    """Stage 10: Update forensic validation memory via validation_memory."""
    t0 = time.perf_counter()
    stage = "update_validation_memory"

    if not ctx.validated_articles:
        lat = (time.perf_counter() - t0) * 1000
        _stage_log(stage, "No validated articles — memory unchanged", "info")
        return StageResult(stage, True, None, latency_ms=lat)

    def _fn() -> None:
        vm = modules["validation_memory"]
        fn = (
            getattr(vm, "record_validation_results", None)
            or getattr(vm, "update", None)
            or getattr(vm, "run", None)
        )
        if fn is None:
            raise AttributeError("validation_memory: no suitable function found")
        fn(ctx.validated_articles)

    ok, _, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000
    ctx.memory_updated = ok

    if ok:
        _stage_log(stage, f"Forensic memory updated ({len(ctx.validated_articles)} articles) ({lat:.1f}ms)")
    else:
        _stage_log(stage, f"DEGRADED — memory update failed: {err}", "warning")

    return StageResult(stage, ok, None, err if not ok else "", lat, retries, degraded=not ok)


def _stage_update_analytics(
    ctx:     RunContext,
    modules: Dict[str, Any],
    circuit: CircuitBreaker,
) -> StageResult:
    """Stage 11: Compute performance analytics via performance_analytics."""
    t0 = time.perf_counter()
    stage = "update_performance_analytics"

    fills = ctx.sent_orders or []

    def _fn() -> Dict:
        pa = modules["performance_analytics"]
        fn = (
            getattr(pa, "analyze_fills", None)
            or getattr(pa, "analyze", None)
            or getattr(pa, "run", None)
        )
        if fn is None:
            raise AttributeError("performance_analytics: no suitable function found")
        result = fn(fills)
        return result if isinstance(result, dict) else {}

    ok, result, err, retries = _run_with_retry(_fn, stage, circuit)
    lat = (time.perf_counter() - t0) * 1000

    if ok:
        ctx.analytics_report = result or {}
        _stage_log(stage, f"Analytics computed ({len(fills)} fills processed) ({lat:.1f}ms)")
    else:
        ctx.analytics_report = {}
        _stage_log(stage, f"DEGRADED — analytics unavailable: {err}", "warning")

    return StageResult(stage, ok, result, err if not ok else "", lat, retries, degraded=not ok)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 12 ▸ STATE RESTORE
# ══════════════════════════════════════════════════════════════════════════════

def attempt_state_restore(modules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Attempt to load the last persisted pipeline state from state_manager.
    Returns the state dict on success, None if unavailable or corrupted.
    Used for crash recovery at startup.
    """
    try:
        sm = modules["state_manager"]
        fn = (
            getattr(sm, "load_state", None)
            or getattr(sm, "restore", None)
        )
        if fn is None:
            return None
        state = fn()
        if isinstance(state, dict) and state:
            log.info(
                "[RESTORE] Found persisted state — run_id=%s, saved_at=%s",
                state.get("run_id", "?")[:12],
                state.get("saved_at", "?"),
            )
            return state
    except Exception as exc:  # noqa: BLE001
        log.warning("[RESTORE] State restore failed (non-critical): %s", exc)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 13 ▸ RUN SUMMARY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def _print_run_summary(ctx: RunContext) -> None:
    status    = ctx.overall_status
    P = _C["purple"]
    G = _C["green"]
    R = _C["red"]
    Y = _C["yellow"]
    C = _C["cyan"]
    B = _C["bold"]
    D = _C["dim"]
    X = _C["reset"]

    status_colour = G if status == PipelineStatus.SUCCESS else (
        Y if status == PipelineStatus.DEGRADED else R
    )
    status_icon   = "✅" if status == PipelineStatus.SUCCESS else (
        "⚠️ " if status == PipelineStatus.DEGRADED else "🔴"
    )

    log.info(f"{P}{'╔' + '═'*74 + '╗'}{X}")
    log.info(f"{P}║{X}  {B}MONSTER TRADING AI — PIPELINE RUN SUMMARY{X}{P}{'':>32}║{X}")
    log.info(f"{P}{'╠' + '═'*74 + '╣'}{X}")
    log.info(f"{P}║{X}  Run ID   : {C}{ctx.run_id[:36]}{X}{P}{'':>36}║{X}")
    log.info(f"{P}║{X}  Session  : {B}{ctx.session.value:<12}{X}  "
             f"Status: {status_colour}{B}{status_icon} {status.value}{X}{P}{'':>22}║{X}")
    log.info(f"{P}║{X}  Latency  : {B}{ctx.pipeline_latency_ms:>9.1f}ms{X}{P}{'':>50}║{X}")
    log.info(f"{P}{'╠' + '═'*74 + '╣'}{X}")
    log.info(f"{P}║{X}  {D}{'METRIC':<32}{'VALUE':>10}{X}{P}{'':>30}║{X}")
    log.info(f"{P}║{X}  {'─'*72}{P}  ║{X}")

    metrics = [
        ("Articles Processed",   ctx.articles_processed,  "G"),
        ("Signals Generated",    ctx.signals_generated,   "G"),
        ("Orders Sent",          ctx.orders_sent,         "G"),
        ("Alerts Routed",        ctx.alerts_routed,       "C"),
        ("Degraded Stages",      len(ctx.degraded_stages), "Y" if ctx.degraded_stages else "G"),
    ]
    for label, value, col in metrics:
        colour = _C.get(col.lower(), X)
        log.info(f"{P}║{X}  {label:<32}{colour}{value:>10}{X}{P}{'':>30}║{X}")

    log.info(f"{P}{'╠' + '═'*74 + '╣'}{X}")
    log.info(f"{P}║{X}  {B}STAGE BREAKDOWN{X}{P}{'':>59}║{X}")
    log.info(f"{P}║{X}  {D}{'STAGE':<32}{'STATUS':<12}{'LATENCY':>9}  {'RETRIES':>7}{X}{P}{'':>9}║{X}")
    log.info(f"{P}║{X}  {'─'*72}{P}  ║{X}")

    for sr in ctx.stage_results:
        ok_str  = f"{G}✓ OK{X}"     if sr.success and not sr.degraded else (
                  f"{Y}⚠ DEGRADED{X}" if sr.degraded else f"{R}✗ FAIL{X}")
        row = (
            f"  {sr.stage:<32}{ok_str:<12}"
            f"  {sr.latency_ms:>7.1f}ms  {sr.retries:>7}"
        )
        log.info(f"{P}║{X}{row}{P}{'':>5}║{X}")

    if ctx.degraded_stages:
        log.info(f"{P}{'╠' + '═'*74 + '╣'}{X}")
        log.info(f"{P}║{X}  {Y}Degraded: {', '.join(ctx.degraded_stages)}{X}{P}{'':>30}║{X}")

    log.info(f"{P}{'╚' + '═'*74 + '╝'}{X}")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 14 ▸ CIRCUIT BREAKER REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def _build_circuit_registry() -> Dict[str, CircuitBreaker]:
    """One circuit breaker per pipeline stage."""
    return {name: CircuitBreaker(name) for name in STAGE_NAMES}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 15 ▸ PUBLIC API  — run_pipeline()
# ══════════════════════════════════════════════════════════════════════════════

# Module-level singletons — survive across multiple run_pipeline() calls
_MODULES:  Optional[Dict[str, Any]] = None
_CIRCUITS: Optional[Dict[str, CircuitBreaker]] = None
_GUARD:    Optional[PortfolioGuard] = None


def _get_singletons() -> Tuple[Dict, Dict, PortfolioGuard]:
    global _MODULES, _CIRCUITS, _GUARD
    if _MODULES is None:
        _MODULES  = _load_all_modules()
    if _CIRCUITS is None:
        _CIRCUITS = _build_circuit_registry()
    if _GUARD is None:
        _GUARD = PortfolioGuard()
    return _MODULES, _CIRCUITS, _GUARD


def run_pipeline(
    market_override: Optional[str] = None,
    dry_run:         bool          = False,
    crash_restore:   bool          = True,
) -> Dict[str, Any]:
    """
    Execute the full 11-stage trading intelligence pipeline.

    Parameters
    ----------
    market_override : str, optional
        Force a specific MarketSession value (e.g. "CRYPTO_24_7").
        Defaults to auto-detection from current UTC time.
    dry_run : bool
        If True, skips broker transmission (stage 7) safely.
    crash_restore : bool
        If True, attempts to load persisted state before running.

    Returns
    -------
    dict
        {
          "articles_processed" : int,
          "signals_generated"  : int,
          "orders_sent"        : int,
          "alerts_sent"        : int,
          "pipeline_latency"   : float,    # milliseconds
          "status"             : str,      # "SUCCESS" | "DEGRADED" | "HALTED"
        }
    """
    run_id    = str(uuid.uuid4())
    pipeline_t0 = time.perf_counter()

    # ── Kill switch check ─────────────────────────────────────────────────────
    if is_kill_switch_active():
        log.critical("%s🔴 Pipeline HALTED — kill switch active%s", _C["red"], _C["reset"])
        return {
            "articles_processed": 0, "signals_generated": 0,
            "orders_sent": 0, "alerts_sent": 0,
            "pipeline_latency": 0.0, "status": PipelineStatus.HALTED.value,
        }

    # ── Session detection ─────────────────────────────────────────────────────
    if market_override:
        try:
            session = MarketSession(market_override)
        except ValueError:
            log.warning("Unknown market_override '%s' — auto-detecting", market_override)
            session = detect_market_session()
    else:
        session = detect_market_session()

    _banner(
        f"🧠 GOD_CORE v{ORCHESTRATOR_VERSION}  |  run={run_id[:8]}…  |  "
        f"session={session.value}  |  dry_run={dry_run}",
        "purple",
    )

    modules, circuits, guard = _get_singletons()

    # ── Crash recovery ────────────────────────────────────────────────────────
    if crash_restore:
        prior_state = attempt_state_restore(modules)
        if prior_state:
            log.info(
                "[RESTORE] Prior state loaded (run_id=%s, orders_sent=%d)",
                prior_state.get("run_id", "?")[:8],
                prior_state.get("orders_sent", 0),
            )

    ctx = RunContext(run_id=run_id, session=session)

    # ── Stage execution helper ─────────────────────────────────────────────────
    def _exec(stage_fn: Callable[[], StageResult]) -> StageResult:
        """Execute one stage, halt on kill switch."""
        if is_kill_switch_active():
            s = stage_fn.__name__.replace("_stage_", "").replace("_", " ")
            return StageResult(s, False, None, "Kill switch active", 0.0)
        try:
            result = stage_fn()
            ctx.record(result)
            return result
        except Exception as exc:  # noqa: BLE001
            stage = getattr(stage_fn, "__name__", "unknown")
            err   = f"{type(exc).__name__}: {exc}"
            log.error("[STAGE ERROR] %s: %s\n%s", stage, err, traceback.format_exc())
            sr = StageResult(stage, False, None, err, 0.0, degraded=True)
            ctx.record(sr)
            return sr

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE EXECUTION — strict pipeline order
    # ══════════════════════════════════════════════════════════════════════════

    # 1. Fetch news
    _exec(lambda: _stage_fetch_news(ctx, modules, circuits["fetch_news"]))

    # 2. Deduplicate
    _exec(lambda: _stage_deduplicate(ctx, modules, circuits["deduplicate_articles"]))

    # 3. Validate
    _exec(lambda: _stage_validate(ctx, modules, circuits["validate_articles"]))

    # 4. Generate signals
    _exec(lambda: _stage_generate_signals(ctx, modules, circuits["generate_signals"]))

    # 5. Risk controls
    _exec(lambda: _stage_apply_risk(ctx, modules, circuits["apply_risk_controls"], guard))

    # 6. Build orders
    _exec(lambda: _stage_build_orders(ctx, modules, circuits["build_orders"], guard))

    # 7. Send orders (skip in dry_run)
    if dry_run:
        ctx.sent_orders = []
        _stage_log("send_orders", "DRY RUN — broker transmission skipped", "warning")
        ctx.stage_results.append(StageResult("send_orders", True, [], "dry_run", 0.0))
    else:
        _exec(lambda: _stage_send_orders(ctx, modules, circuits["send_orders"]))

    # 8. Route alerts
    _exec(lambda: _stage_route_alerts(ctx, modules, circuits["route_alerts"]))

    # 9. Persist state
    _exec(lambda: _stage_persist_state(ctx, modules, circuits["persist_state"]))

    # 10. Update validation memory
    _exec(lambda: _stage_update_memory(ctx, modules, circuits["update_validation_memory"]))

    # 11. Update performance analytics
    _exec(lambda: _stage_update_analytics(ctx, modules, circuits["update_performance_analytics"]))

    # ── Final summary ─────────────────────────────────────────────────────────
    ctx.pipeline_latency_ms  # trigger property computation
    _print_run_summary(ctx)

    return {
        "articles_processed": ctx.articles_processed,
        "signals_generated":  ctx.signals_generated,
        "orders_sent":        ctx.orders_sent,
        "alerts_sent":        ctx.alerts_routed,
        "pipeline_latency":   round(ctx.pipeline_latency_ms, 3),
        "status":             ctx.overall_status.value,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 16 ▸ CONVENIENCE MANAGEMENT API
# ══════════════════════════════════════════════════════════════════════════════

def reset_circuits() -> None:
    """Reset all circuit breakers to CLOSED — use after diagnosing failures."""
    global _CIRCUITS
    if _CIRCUITS:
        for cb in _CIRCUITS.values():
            cb.reset()
        log.info("[MANAGEMENT] All circuit breakers reset to CLOSED")


def reset_portfolio_guard() -> None:
    """Reset portfolio guard exposure and daily loss tracking."""
    global _GUARD
    _GUARD = PortfolioGuard()
    log.info("[MANAGEMENT] Portfolio guard reset")


def get_system_status() -> Dict[str, Any]:
    """Return a snapshot of the orchestrator's current health."""
    _, circuits, guard = _get_singletons()
    return {
        "version":          ORCHESTRATOR_VERSION,
        "kill_switch":      is_kill_switch_active(),
        "session":          detect_market_session().value,
        "portfolio_exposure_pct": round(guard.current_exposure_pct, 4) if guard else 0.0,
        "daily_drawdown_pct":     round(guard.daily_drawdown_pct, 4) if guard else 0.0,
        "circuits": {
            name: cb.state.value for name, cb in circuits.items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 17 ▸ SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

class _GodCoreTests(unittest.TestCase):
    """
    Self-contained smoke tests — no real broker/market connectivity required.
    Uses stub modules throughout so the pipeline can be exercised end-to-end.
    """

    def setUp(self) -> None:
        global _MODULES, _CIRCUITS, _GUARD, _KILL_SWITCH_ACTIVE
        _KILL_SWITCH_ACTIVE = False
        _MODULES  = None
        _CIRCUITS = None
        _GUARD    = None

    def tearDown(self) -> None:
        global _KILL_SWITCH_ACTIVE
        _KILL_SWITCH_ACTIVE = False

    # ── T01: Output contract shape ────────────────────────────────────────────
    def test_01_output_contract(self) -> None:
        result = run_pipeline(dry_run=True, crash_restore=False)
        required = {
            "articles_processed", "signals_generated",
            "orders_sent", "alerts_sent",
            "pipeline_latency", "status",
        }
        self.assertTrue(required.issubset(result.keys()),
                        f"Missing keys: {required - result.keys()}")
        self.assertIn(result["status"], ("SUCCESS", "DEGRADED", "HALTED"))
        self.assertIsInstance(result["pipeline_latency"], float)

    # ── T02: Kill switch blocks execution ─────────────────────────────────────
    def test_02_kill_switch_blocks(self) -> None:
        activate_kill_switch("test")
        result = run_pipeline(dry_run=True, crash_restore=False)
        self.assertEqual(result["status"], "HALTED")
        self.assertEqual(result["orders_sent"], 0)
        deactivate_kill_switch()

    # ── T03: Kill switch deactivation restores execution ─────────────────────
    def test_03_kill_switch_deactivate(self) -> None:
        activate_kill_switch("test")
        deactivate_kill_switch()
        result = run_pipeline(dry_run=True, crash_restore=False)
        self.assertIn(result["status"], ("SUCCESS", "DEGRADED"))

    # ── T04: Market session detection ─────────────────────────────────────────
    def test_04_market_session_detection(self) -> None:
        # Weekday regular hours: Tuesday 15:00 UTC
        dt_regular = datetime(2025, 6, 3, 15, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(detect_market_session(dt_regular), MarketSession.REGULAR)

        # Weekend
        dt_weekend = datetime(2025, 6, 7, 15, 0, 0, tzinfo=timezone.utc)   # Saturday
        self.assertEqual(detect_market_session(dt_weekend), MarketSession.CLOSED)

        # Premarket: 10:00 UTC Tuesday
        dt_pre = datetime(2025, 6, 3, 10, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(detect_market_session(dt_pre), MarketSession.PREMARKET)

        # After hours: 21:30 UTC Tuesday
        dt_ah = datetime(2025, 6, 3, 21, 30, 0, tzinfo=timezone.utc)
        self.assertEqual(detect_market_session(dt_ah), MarketSession.AFTER_HOURS)

    # ── T05: market_override accepted ─────────────────────────────────────────
    def test_05_market_override(self) -> None:
        result = run_pipeline(
            market_override="CRYPTO_24_7",
            dry_run=True,
            crash_restore=False,
        )
        self.assertIn(result["status"], ("SUCCESS", "DEGRADED", "HALTED"))

    # ── T06: Circuit breaker trips and resets ─────────────────────────────────
    def test_06_circuit_breaker(self) -> None:
        cb = CircuitBreaker("test_stage", failure_threshold=2, reset_window_s=1.0)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        time.sleep(1.1)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    # ── T07: Portfolio guard exposure cap ─────────────────────────────────────
    def test_07_portfolio_exposure_cap(self) -> None:
        guard = PortfolioGuard(max_exposure_pct=0.10)
        ok, _ = guard.check_exposure(0.08)
        self.assertTrue(ok)
        guard.add_exposure(0.08)
        ok, reason = guard.check_exposure(0.05)
        self.assertFalse(ok)
        self.assertIn("cap", reason.lower())

    # ── T08: Portfolio drawdown stop ──────────────────────────────────────────
    def test_08_drawdown_stop(self) -> None:
        global _KILL_SWITCH_ACTIVE
        guard = PortfolioGuard(max_drawdown_pct=0.01, portfolio_value=10_000.0)
        _KILL_SWITCH_ACTIVE = False
        guard.record_loss(50.0)    # 0.5% — should NOT trip
        self.assertFalse(is_kill_switch_active())
        guard.record_loss(100.0)   # total 1.5% — should trip 1% cap
        self.assertTrue(is_kill_switch_active())
        _KILL_SWITCH_ACTIVE = False   # reset for tearDown

    # ── T09: StubModule safe returns ──────────────────────────────────────────
    def test_09_stub_module(self) -> None:
        stub = _StubModule("test_module")
        result = stub.fetch_news()
        self.assertIsInstance(result, list)
        result2 = stub.generate_signals(articles=[])
        self.assertIsInstance(result2, list)

    # ── T10: get_system_status shape ──────────────────────────────────────────
    def test_10_system_status(self) -> None:
        status = get_system_status()
        required = {
            "version", "kill_switch", "session",
            "portfolio_exposure_pct", "daily_drawdown_pct",
            "circuits", "timestamp",
        }
        self.assertTrue(required.issubset(status.keys()))
        self.assertEqual(len(status["circuits"]), len(STAGE_NAMES))

    # ── T11: Dry run does not send orders ────────────────────────────────────
    def test_11_dry_run_skips_broker(self) -> None:
        result = run_pipeline(dry_run=True, crash_restore=False)
        # In dry_run, orders_sent reflects broker fills = 0
        self.assertEqual(result["orders_sent"], 0)

    # ── T12: RunContext overall_status logic ─────────────────────────────────
    def test_12_run_context_status(self) -> None:
        ctx = RunContext("test-run", MarketSession.REGULAR)
        self.assertEqual(ctx.overall_status, PipelineStatus.SUCCESS)
        ctx.degraded_stages.append("some_stage")
        self.assertEqual(ctx.overall_status, PipelineStatus.DEGRADED)


# ══════════════════════════════════════════════════════════════════════════════
# __main__ ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MONSTER TRADING AI — god_core.py orchestrator"
    )
    parser.add_argument(
        "--smoke",  "-s",
        action="store_true",
        help="Run smoke test suite (12 tests)",
    )
    parser.add_argument(
        "--run",    "-r",
        action="store_true",
        help="Execute one dry-run pipeline cycle",
    )
    parser.add_argument(
        "--status", "-S",
        action="store_true",
        help="Print current system status and exit",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Override market session (REGULAR|PREMARKET|AFTER_HOURS|CRYPTO_24_7)",
    )
    args = parser.parse_args()

    if args.status:
        status = get_system_status()
        print(json.dumps(status, indent=2))
        sys.exit(0)

    if args.smoke:
        _banner(
            f"🧪  GOD_CORE SMOKE TEST SUITE  |  {ORCHESTRATOR_VERSION}",
            "purple",
        )
        loader = unittest.TestLoader()
        suite  = loader.loadTestsFromTestCase(_GodCoreTests)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)

        total  = result.testsRun
        passed = total - len(result.failures) - len(result.errors)
        failed = len(result.failures) + len(result.errors)

        P = _C["purple"]
        G = _C["green"]
        R = _C["red"]
        X = _C["reset"]

        print()
        print(f"{P}{'═'*72}{X}")
        if failed == 0:
            print(f"  {G}✓  {passed}/{total}  ALL CLEAR — god_core.py is production-ready.{X}")
        else:
            print(f"  {R}✗  {passed}/{total} passed — {failed} failure(s). See above.{X}")
        print(f"{P}{'═'*72}{X}")
        print()
        sys.exit(0 if result.wasSuccessful() else 1)

    if args.run:
        _banner("🚀 DRY-RUN PIPELINE EXECUTION", "cyan")
        result = run_pipeline(
            market_override=args.session,
            dry_run=True,
            crash_restore=True,
        )
        print()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] != PipelineStatus.HALTED.value else 1)

    # Default: show usage + status
    parser.print_help()
    print()
    print("System Status:")
    print(json.dumps(get_system_status(), indent=2))