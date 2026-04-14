"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              MONSTER TRADING AI — config.py                                 ║
║         Global Configuration Backbone · Single Source of Truth              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Imported by: news_engine · duplicate_filter · fake_news_validator          ║
║               source_registry · rss_sandbox · signal_engine                 ║
║               alert_router · execution_bridge · risk_guardian               ║
║               broker_sender · performance_analytics · mission_control        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Centralises every system-wide constant, threshold, environment toggle, broker
setting, risk control, timing interval, and file path.

NO module may hard-code values that belong here.  Import from config instead:

    from config import (
        PORTFOLIO_SIZE_USD, MIN_SIGNAL_CONFIDENCE, EXECUTION_TEMPLATES, ...
    )

Sections
--------
1.  Runtime Mode
2.  Portfolio + Risk
3.  Signal Thresholds
4.  Timing Controls
5.  Execution Templates   (per-asset-class sizing + SL/TP templates)
6.  External Integrations (env-var-backed secrets)
7.  File Paths
8.  Helper APIs           (as_dict, validate_config)
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Any, Final

# ══════════════════════════════════════════════════════════════════════════════
# 1.  RUNTIME MODE
# ══════════════════════════════════════════════════════════════════════════════

PAPER_MODE: Final[bool] = True
"""
Simulation / paper-trading mode.  No real orders reach any exchange.
All broker_sender adapters route to the Paper stub when this is True.
"""

LIVE_MODE: Final[bool] = False
"""
Live-trading mode.  When True, PAPER_MODE MUST be False.
validate_config() enforces this invariant at startup.
"""

DEFAULT_BROKER: Final[str] = "paper"
"""
Active broker adapter selected at runtime.
Must be one of: 'paper' | 'alpaca' | 'ibkr' | 'binance'.
broker_sender reads this to pick the correct adapter class.
"""

ENABLE_PLUGINS: Final[bool] = True
"""
Load optional plugin modules discovered in the /plugins directory at startup.
Set False in locked-down production environments.
"""

DEBUG: Final[bool] = True
"""
Enable verbose debug-level logging across all modules.
Should be False in production to reduce log noise and latency.
"""

# Internal lookup used by validate_config() and broker_sender.
_VALID_BROKERS: Final[frozenset] = frozenset({"paper", "alpaca", "ibkr", "binance"})

# ══════════════════════════════════════════════════════════════════════════════
# 2.  PORTFOLIO + RISK
# ══════════════════════════════════════════════════════════════════════════════

PORTFOLIO_SIZE_USD: Final[float] = 25_000.0
"""
Total notional capital allocated to the strategy (USD).
risk_guardian uses this as the base for all fractional-risk calculations.
"""

MAX_OPEN_POSITIONS: Final[int] = 5
"""
Hard ceiling on the number of concurrent open positions.
execution_bridge checks this before routing any new entry order.
"""

MAX_RISK_PER_TRADE: Final[float] = 0.02
"""
Maximum fraction of PORTFOLIO_SIZE_USD risked on a single trade (2 %).
Expressed as a decimal.  risk_guardian enforces this per-signal.
"""

MAX_TICKER_EXPOSURE: Final[float] = 0.10
"""
Maximum fraction of PORTFOLIO_SIZE_USD allocated to any single ticker (10 %).
Prevents outsized concentration in one name regardless of signal strength.
"""

DAILY_LOSS_LIMIT: Final[float] = 0.025
"""
Intraday drawdown fraction that triggers a full trading halt (2.5 %).
risk_guardian monitors realised + unrealised P&L and calls circuit-breaker.
"""

WARNING_DRAWDOWN: Final[float] = 0.015
"""
Intraday drawdown fraction that triggers a risk-warning alert (1.5 %).
alert_router fires a Discord/N8N notification at this level before halt.
"""

# ── UNIFIED DRAWDOWN CASCADE (single source of truth for all risk tiers) ──────
# This policy is the ONLY place max drawdown thresholds are defined.
# risk_guardian.py and self_learning_optimizer.py read this — never hardcode.
# Applied in descending order — first match wins.

DRAWDOWN_POLICY_TIERS: Final[list[tuple[float, str, float]]] = [
    # (threshold_pct, action_name, position_size_multiplier)
    (0.025, "FULL_KILL_SWITCH", 0.0),    # ≥ 2.5% → block all trades (DAILY_LOSS_LIMIT)
    (0.015, "HEAVY_REDUCTION", 0.4),     # 1.5–2.5% → 40% position size (WARNING_DRAWDOWN)
    (0.005, "WARNING_ALERT", 0.8),       # 0.5–1.5% → reduce 20%
    (0.0,   "NORMAL", 1.0),              # < 0.5% → full size
]

DRAWDOWN_ACTION_KILL_SWITCH = "FULL_KILL_SWITCH"
DRAWDOWN_ACTION_HEAVY_REDUCTION = "HEAVY_REDUCTION"
DRAWDOWN_ACTION_WARNING = "WARNING_ALERT"
DRAWDOWN_ACTION_NORMAL = "NORMAL"

MIN_SIGNAL_CONFIDENCE: Final[float] = 0.80
"""
Minimum model confidence required to treat a signal as actionable.
signal_engine discards anything below this without forwarding to execution.
"""

ELITE_SIGNAL_CONFIDENCE: Final[float] = 0.90
"""
Confidence level that qualifies a signal as 'elite'.
performance_analytics buckets signals at this boundary for tier reporting.
Execution may apply larger sizing for elite signals.
"""

CONVICTION_BONUS_THRESHOLD: Final[float] = 0.92
"""
Above this confidence execution_bridge applies a conviction-sizing bonus,
scaling position size by the multiplier defined in EXECUTION_TEMPLATES.
"""

SIGNAL_HALF_LIFE_MINUTES: Final[int] = 20
"""
Minutes after generation before a pending signal is considered stale.
signal_engine checks timestamp on every evaluation loop and purges expired signals.
"""

# ══════════════════════════════════════════════════════════════════════════════
# 4.  TIMING CONTROLS
# ══════════════════════════════════════════════════════════════════════════════

NEWS_POLL_INTERVAL_SECONDS: Final[int] = 300
"""
How often (seconds) news_engine and rss_sandbox fetch fresh headlines.
300 s = 5-minute cadence balances freshness vs. rate-limit safety.
"""

COOLDOWN_MINUTES: Final[int] = 15
"""
Minimum gap (minutes) between successive entries on the same ticker.
Prevents signal chasing after a rapid sequence of news events.
"""

MACRO_FREEZE_PRE_MINUTES: Final[int] = 30
"""
Minutes *before* a scheduled macro event (FOMC, NFP, CPI, etc.) during
which mission_control blocks new directional entries.
"""

MACRO_FREEZE_POST_MINUTES: Final[int] = 15
"""
Minutes *after* a macro event before normal entry flow resumes.
Allows the initial volatility spike to settle before new positions open.
"""

FAILURE_BACKOFF_BASE_SECONDS: Final[int] = 5
"""
Base delay (seconds) for exponential back-off on transient API failures.
Retry delay = FAILURE_BACKOFF_BASE_SECONDS * 2^(attempt - 1), capped at 300 s.
"""

# ══════════════════════════════════════════════════════════════════════════════
# 5.  EXECUTION TEMPLATES  (per-asset-class sizing + SL/TP templates)
# ══════════════════════════════════════════════════════════════════════════════

EXECUTION_TEMPLATES: Final[dict] = {
    # ── CRYPTO ───────────────────────────────────────────────────────────────
    "CRYPTO": {
        # Position sizing as fraction of PORTFOLIO_SIZE_USD
        "standard_size_pct": 0.05,          # 5 % — default entry allocation
        "aggressive_size_pct": 0.08,         # 8 % — used above CONVICTION_BONUS_THRESHOLD
        "max_position_pct": 0.10,            # hard ceiling per ticker
        "min_notional_usd": 20.0,            # smallest acceptable order value (USD)
        # Stop-loss / take-profit (distance from entry price)
        "stop_loss_pct": 0.04,              # 4 % below entry
        "take_profit_pct": 0.12,            # 12 % above entry  →  3 : 1 R/R
        "trailing_stop_pct": 0.025,         # 2.5 % trailing once position is in profit
        # Conviction-sizing multiplier (applied when confidence >= CONVICTION_BONUS_THRESHOLD)
        "conviction_multiplier": 1.40,
        # Order behaviour
        "order_type": "limit",
        "limit_slip_bps": 10,               # max acceptable slippage in basis points
        "time_in_force": "GTC",
        # Metadata
        "asset_class": "CRYPTO",
        "description": "Spot / perpetual crypto — higher volatility, wider SL/TP",
    },

    # ── EQUITY ───────────────────────────────────────────────────────────────
    "EQUITY": {
        "standard_size_pct": 0.06,
        "aggressive_size_pct": 0.09,
        "max_position_pct": 0.10,
        "min_notional_usd": 100.0,
        "stop_loss_pct": 0.025,             # tighter — equity vol is lower than crypto
        "take_profit_pct": 0.075,           # 3 : 1 R/R
        "trailing_stop_pct": 0.015,
        "conviction_multiplier": 1.30,
        "order_type": "limit",
        "limit_slip_bps": 5,
        "time_in_force": "DAY",
        "asset_class": "EQUITY",
        "description": "US / international equities — standard intraday or swing",
    },

    # ── FOREX ────────────────────────────────────────────────────────────────
    "FOREX": {
        "standard_size_pct": 0.04,
        "aggressive_size_pct": 0.06,
        "max_position_pct": 0.08,
        "min_notional_usd": 1_000.0,        # FX lot-sizing floor
        "stop_loss_pct": 0.010,             # tight — leverage amplifies moves
        "take_profit_pct": 0.020,           # 2 : 1 R/R — FX mean-reversion bias
        "trailing_stop_pct": 0.005,
        "conviction_multiplier": 1.20,
        "order_type": "market",             # FX executes best via market order
        "limit_slip_bps": 3,
        "time_in_force": "FOK",
        "asset_class": "FOREX",
        "description": "Major / minor FX pairs — tight spreads, high leverage",
    },

    # ── MACRO ────────────────────────────────────────────────────────────────
    "MACRO": {
        # Macro trades are larger directional bets held over days / weeks
        "standard_size_pct": 0.08,
        "aggressive_size_pct": 0.12,
        "max_position_pct": 0.15,
        "min_notional_usd": 500.0,
        "stop_loss_pct": 0.050,
        "take_profit_pct": 0.200,           # 4 : 1 R/R — macro moves are larger
        "trailing_stop_pct": 0.030,
        "conviction_multiplier": 1.50,
        "order_type": "limit",
        "limit_slip_bps": 15,
        "time_in_force": "GTC",
        "asset_class": "MACRO",
        "description": "Macro thematic — ETFs, indices, commodities, treasuries",
    },
}
"""
Per-asset-class execution templates consumed by execution_bridge.

Keys
----
standard_size_pct     : fraction of PORTFOLIO_SIZE_USD for a normal entry
aggressive_size_pct   : fraction used when confidence > CONVICTION_BONUS_THRESHOLD
max_position_pct      : absolute ceiling for a single ticker
min_notional_usd      : floor below which orders are rejected
stop_loss_pct         : distance below entry at which the stop is placed
take_profit_pct       : distance above entry for the profit target
trailing_stop_pct     : trailing-stop distance once position moves into profit
conviction_multiplier : multiplier applied to standard_size_pct for conviction bets
order_type            : 'limit' | 'market'
limit_slip_bps        : acceptable slippage in basis points for limit orders
time_in_force         : 'GTC' | 'DAY' | 'FOK' | 'IOC'
"""

# ══════════════════════════════════════════════════════════════════════════════
# 6.  EXTERNAL INTEGRATIONS  (env-var-backed secrets — never hard-code values)
# ══════════════════════════════════════════════════════════════════════════════

# ── Alpaca Markets ────────────────────────────────────────────────────────────
ALPACA_API_KEY: Final[str] = os.getenv("ALPACA_API_KEY", "")
"""Alpaca REST API key.  Set via environment variable ALPACA_API_KEY."""

ALPACA_SECRET: Final[str] = os.getenv("ALPACA_SECRET", "")
"""Alpaca REST API secret.  Set via environment variable ALPACA_SECRET."""

ALPACA_BASE_URL: Final[str] = os.getenv(
    "ALPACA_BASE_URL",
    "https://paper-api.alpaca.markets",   # safe default — paper endpoint
)
"""Alpaca base URL.  Override to https://api.alpaca.markets for live trading."""

# ── Interactive Brokers (IBKR) ────────────────────────────────────────────────
IBKR_HOST: Final[str] = os.getenv("IBKR_HOST", "127.0.0.1")
"""IBKR TWS / IB Gateway host address."""

IBKR_PORT: Final[int] = int(os.getenv("IBKR_PORT", "7497"))
"""IBKR TWS port.  7497 = paper, 7496 = live."""

IBKR_CLIENT_ID: Final[int] = int(os.getenv("IBKR_CLIENT_ID", "1"))
"""IBKR client connection ID — must be unique per simultaneous connection."""

# ── Binance ───────────────────────────────────────────────────────────────────
BINANCE_KEY: Final[str] = os.getenv("BINANCE_KEY", "")
"""Binance REST API key.  Set via environment variable BINANCE_KEY."""

BINANCE_SECRET: Final[str] = os.getenv("BINANCE_SECRET", "")
"""Binance REST API secret.  Set via environment variable BINANCE_SECRET."""

BINANCE_BASE_URL: Final[str] = os.getenv(
    "BINANCE_BASE_URL",
    "https://testnet.binance.vision",     # safe default — testnet
)
"""Binance base URL.  Override to https://api.binance.com for live trading."""

# ── Notification Webhooks ─────────────────────────────────────────────────────
DISCORD_WEBHOOK: Final[str] = os.getenv("DISCORD_WEBHOOK", "")
"""Discord webhook URL for alert_router trade notifications and risk warnings."""

N8N_WEBHOOK: Final[str] = os.getenv("N8N_WEBHOOK", "")
"""N8N automation webhook for downstream workflow triggers."""

# ── News / Data APIs ──────────────────────────────────────────────────────────
NEWSAPI_KEY: Final[str] = os.getenv("NEWSAPI_KEY", "")
"""NewsAPI.org key used by news_engine for headline ingestion."""

OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY", "")
"""OpenAI API key for any LLM-based sentiment or fake-news validation layers."""

# ══════════════════════════════════════════════════════════════════════════════
# 7.  FILE PATHS
# ══════════════════════════════════════════════════════════════════════════════

STATE_FILE: Final[str] = os.getenv("STATE_FILE", "state.json")
"""
Path to the persistent state JSON file.
mission_control and risk_guardian read/write positions, P&L, and session data here.
"""

LOG_FILE: Final[str] = os.getenv("LOG_FILE", "trading_ai.log")
"""
Rotating log file path.  All modules write structured log lines here.
"""

REPORTS_DIR: Final[str] = os.getenv("REPORTS_DIR", "reports")
"""
Directory where performance_analytics writes HTML / CSV / JSON reports.
Created automatically if it does not exist.
"""

PLUGIN_DIR: Final[str] = os.getenv("PLUGIN_DIR", "plugins")
"""
Directory scanned for optional plugin modules when ENABLE_PLUGINS is True.
"""

DUPLICATE_CACHE_FILE: Final[str] = os.getenv(
    "DUPLICATE_CACHE_FILE", "duplicate_cache.json"
)
"""
Persistent cache file for duplicate_filter's seen-headline fingerprints.
"""

# ══════════════════════════════════════════════════════════════════════════════
# 8.  HELPER APIs
# ══════════════════════════════════════════════════════════════════════════════

def as_dict() -> dict:
    """
    Return all public configuration values as a JSON-serialisable dictionary.

    Secrets (API keys, webhook URLs) are masked so the dict is safe to log
    or include in status reports without leaking credentials.

    Returns
    -------
    dict
        Flat + nested mapping of every config constant in this module.
    """
    def _mask(value: str) -> str:
        """Show first 4 chars then asterisks, or '<not set>' if empty."""
        if not value:
            return "<not set>"
        return value[:4] + "*" * (len(value) - 4) if len(value) > 4 else "****"

    return {
        # ── Runtime Mode ──────────────────────────────────────────────────
        "PAPER_MODE":                    PAPER_MODE,
        "LIVE_MODE":                     LIVE_MODE,
        "DEFAULT_BROKER":                DEFAULT_BROKER,
        "ENABLE_PLUGINS":                ENABLE_PLUGINS,
        "DEBUG":                         DEBUG,
        # ── Portfolio + Risk ──────────────────────────────────────────────
        "PORTFOLIO_SIZE_USD":            PORTFOLIO_SIZE_USD,
        "MAX_OPEN_POSITIONS":            MAX_OPEN_POSITIONS,
        "MAX_RISK_PER_TRADE":            MAX_RISK_PER_TRADE,
        "MAX_TICKER_EXPOSURE":           MAX_TICKER_EXPOSURE,
        "DAILY_LOSS_LIMIT":              DAILY_LOSS_LIMIT,
        "WARNING_DRAWDOWN":              WARNING_DRAWDOWN,
        # ── Signal Thresholds ─────────────────────────────────────────────
        "MIN_SIGNAL_CONFIDENCE":         MIN_SIGNAL_CONFIDENCE,
        "ELITE_SIGNAL_CONFIDENCE":       ELITE_SIGNAL_CONFIDENCE,
        "CONVICTION_BONUS_THRESHOLD":    CONVICTION_BONUS_THRESHOLD,
        "SIGNAL_HALF_LIFE_MINUTES":      SIGNAL_HALF_LIFE_MINUTES,
        # ── Timing Controls ───────────────────────────────────────────────
        "NEWS_POLL_INTERVAL_SECONDS":    NEWS_POLL_INTERVAL_SECONDS,
        "COOLDOWN_MINUTES":              COOLDOWN_MINUTES,
        "MACRO_FREEZE_PRE_MINUTES":      MACRO_FREEZE_PRE_MINUTES,
        "MACRO_FREEZE_POST_MINUTES":     MACRO_FREEZE_POST_MINUTES,
        "FAILURE_BACKOFF_BASE_SECONDS":  FAILURE_BACKOFF_BASE_SECONDS,
        # ── Execution Templates ───────────────────────────────────────────
        "EXECUTION_TEMPLATES":           EXECUTION_TEMPLATES,
        # ── External Integrations (masked) ────────────────────────────────
        "ALPACA_API_KEY":    _mask(ALPACA_API_KEY),
        "ALPACA_SECRET":     _mask(ALPACA_SECRET),
        "ALPACA_BASE_URL":   ALPACA_BASE_URL,
        "IBKR_HOST":         IBKR_HOST,
        "IBKR_PORT":         IBKR_PORT,
        "IBKR_CLIENT_ID":    IBKR_CLIENT_ID,
        "BINANCE_KEY":       _mask(BINANCE_KEY),
        "BINANCE_SECRET":    _mask(BINANCE_SECRET),
        "BINANCE_BASE_URL":  BINANCE_BASE_URL,
        "DISCORD_WEBHOOK":   _mask(DISCORD_WEBHOOK),
        "N8N_WEBHOOK":       _mask(N8N_WEBHOOK),
        "NEWSAPI_KEY":       _mask(NEWSAPI_KEY),
        "OPENAI_API_KEY":    _mask(OPENAI_API_KEY),
        # ── File Paths ────────────────────────────────────────────────────
        "STATE_FILE":            STATE_FILE,
        "LOG_FILE":              LOG_FILE,
        "REPORTS_DIR":           REPORTS_DIR,
        "PLUGIN_DIR":            PLUGIN_DIR,
        "DUPLICATE_CACHE_FILE":  DUPLICATE_CACHE_FILE,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9.  STANDARDIZED FIELD NAMES  (canonical keys across all modules)
# ══════════════════════════════════════════════════════════════════════════════

# Every module MUST use these exact field names when passing dicts between stages.
# NO module should hardcode alternative names like "gross_cap" or "hedge_ratio".

SIGNAL_FIELD_DIRECTION: Final[str] = "signal_direction"            # LONG | SHORT | FLAT
SIGNAL_FIELD_CONFIDENCE: Final[str] = "confidence_score"           # float [0-1]
SIGNAL_FIELD_STRENGTH: Final[str] = "signal_strength"              # float [0-1]
SIGNAL_FIELD_IMPACT: Final[str] = "impact_score"                   # float [0-1]
SIGNAL_FIELD_URGENCY: Final[str] = "urgency"                       # LOW | MEDIUM | HIGH | CRITICAL
SIGNAL_FIELD_EVENT_TYPE: Final[str] = "event_type"                 # EARNINGS | M&A | MACRO | etc
SIGNAL_FIELD_REASONS: Final[str] = "signal_reasons"                # list[str]

REGIME_FIELD_NAME: Final[str] = "market_regime"                    # bull_trend | crisis | etc
REGIME_FIELD_GROSS_CAP: Final[str] = "recommended_gross_cap"       # float [0.2-2.0]
REGIME_FIELD_HEDGE_RATIO: Final[str] = "recommended_hedge_ratio"   # float [0-1]
REGIME_FIELD_RISK_ON_SCORE: Final[str] = "risk_on_off_score"       # float [-1.0, +1.0]
REGIME_FIELD_VOL: Final[str] = "volatility_regime"                 # low | normal | elevated | extreme

PORTFOLIO_FIELD_OPTIONS_PREMIUM: Final[str] = "options_premium_at_risk"  # float [0-0.1]


def validate_config() -> bool:
    """
    Validate the configuration for logical consistency and safety.

    Rules enforced
    --------------
    1.  LIVE_MODE and PAPER_MODE cannot both be True simultaneously.
    2.  PORTFOLIO_SIZE_USD must be strictly positive.
    3.  All fractional thresholds must be in the range (0.0, 1.0].
    4.  DEFAULT_BROKER must be one of the recognised adapter names.
    5.  MAX_OPEN_POSITIONS must be >= 1.
    6.  Signal confidence thresholds must be in ascending order:
            MIN_SIGNAL_CONFIDENCE
            < ELITE_SIGNAL_CONFIDENCE
            <= CONVICTION_BONUS_THRESHOLD
    7.  DAILY_LOSS_LIMIT must be >= WARNING_DRAWDOWN (halt >= warning).

    Returns
    -------
    bool
        True if all checks pass.  Prints a detailed error report and
        returns False on any failure (does not raise — caller decides
        whether to abort).
    """
    errors: list = []

    # Rule 1 — mode exclusivity
    if LIVE_MODE and PAPER_MODE:
        errors.append(
            "MODE CONFLICT: LIVE_MODE and PAPER_MODE cannot both be True. "
            "Set exactly one to True."
        )

    # Rule 2 — portfolio size
    if PORTFOLIO_SIZE_USD <= 0:
        errors.append(
            f"PORTFOLIO_SIZE_USD must be > 0, got {PORTFOLIO_SIZE_USD}."
        )

    # Rule 3 — fractional thresholds in (0, 1]
    _fraction_checks = [
        ("MAX_RISK_PER_TRADE",           MAX_RISK_PER_TRADE),
        ("MAX_TICKER_EXPOSURE",          MAX_TICKER_EXPOSURE),
        ("DAILY_LOSS_LIMIT",             DAILY_LOSS_LIMIT),
        ("WARNING_DRAWDOWN",             WARNING_DRAWDOWN),
        ("MIN_SIGNAL_CONFIDENCE",        MIN_SIGNAL_CONFIDENCE),
        ("ELITE_SIGNAL_CONFIDENCE",      ELITE_SIGNAL_CONFIDENCE),
        ("CONVICTION_BONUS_THRESHOLD",   CONVICTION_BONUS_THRESHOLD),
    ]
    for name, value in _fraction_checks:
        if not (0.0 < value <= 1.0):
            errors.append(
                f"{name} must be in (0.0, 1.0], got {value}."
            )

    # Rule 4 — valid broker
    if DEFAULT_BROKER not in _VALID_BROKERS:
        errors.append(
            f"DEFAULT_BROKER '{DEFAULT_BROKER}' is not recognised. "
            f"Valid options: {sorted(_VALID_BROKERS)}."
        )

    # Rule 5 — position count
    if MAX_OPEN_POSITIONS < 1:
        errors.append(
            f"MAX_OPEN_POSITIONS must be >= 1, got {MAX_OPEN_POSITIONS}."
        )

    # Rule 6 — confidence ordering
    if not (MIN_SIGNAL_CONFIDENCE < ELITE_SIGNAL_CONFIDENCE):
        errors.append(
            "Confidence thresholds must satisfy: "
            "MIN_SIGNAL_CONFIDENCE < ELITE_SIGNAL_CONFIDENCE. "
            f"Got {MIN_SIGNAL_CONFIDENCE} vs {ELITE_SIGNAL_CONFIDENCE}."
        )
    if not (ELITE_SIGNAL_CONFIDENCE <= CONVICTION_BONUS_THRESHOLD):
        errors.append(
            "Confidence thresholds must satisfy: "
            "ELITE_SIGNAL_CONFIDENCE <= CONVICTION_BONUS_THRESHOLD. "
            f"Got {ELITE_SIGNAL_CONFIDENCE} vs {CONVICTION_BONUS_THRESHOLD}."
        )

    # Rule 7 — drawdown ordering
    if DAILY_LOSS_LIMIT < WARNING_DRAWDOWN:
        errors.append(
            "DAILY_LOSS_LIMIT must be >= WARNING_DRAWDOWN "
            "(halt threshold must be at or above the warning threshold). "
            f"Got halt={DAILY_LOSS_LIMIT}, warning={WARNING_DRAWDOWN}."
        )

    # ── Report ────────────────────────────────────────────────────────────────
    if errors:
        print("\n" + "═" * 72)
        print("  ✗  CONFIG VALIDATION FAILED")
        print("═" * 72)
        for i, err in enumerate(errors, 1):
            wrapped = textwrap.fill(err, width=68, subsequent_indent="      ")
            print(f"  {i}. {wrapped}")
        print("═" * 72 + "\n")
        return False

    print("  ✓  Config validation passed — all rules satisfied.")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTER  (used by smoke test and mission_control startup banner)
# ══════════════════════════════════════════════════════════════════════════════

def print_summary() -> None:
    """
    Print a formatted, human-readable summary of the active configuration.

    Called automatically in the __main__ smoke test and optionally by
    mission_control at startup to confirm the loaded settings.
    """
    W = 72  # total display width

    def _bar(char: str = "═") -> str:
        return char * W

    def _row(label: str, value: Any, pad_to: int = 44) -> str:
        dots = "." * max(pad_to - len(label), 1)
        return f"{label}{dots}{value}"

    def _pct(v: float) -> str:
        return f"{v * 100:.1f} %"

    mode_tag = (
        "LIVE  [!]"   if LIVE_MODE  else
        "PAPER [sim]" if PAPER_MODE else
        "UNKNOWN"
    )

    print()
    print(_bar())
    print("  MONSTER TRADING AI — Active Configuration".center(W))
    print(_bar())

    # ── Runtime Mode ──────────────────────────────────────────────────────────
    print()
    print("  ┌─ RUNTIME MODE " + "─" * 55 + "┐")
    print(_row("  │  Mode",                mode_tag))
    print(_row("  │  Default Broker",      DEFAULT_BROKER.upper()))
    print(_row("  │  Plugins Enabled",     ENABLE_PLUGINS))
    print(_row("  │  Debug Logging",       DEBUG))
    print("  └" + "─" * 70 + "┘")

    # ── Portfolio + Risk ──────────────────────────────────────────────────────
    print()
    print("  ┌─ PORTFOLIO + RISK " + "─" * 51 + "┐")
    print(_row("  │  Portfolio Size",          f"${PORTFOLIO_SIZE_USD:,.0f}"))
    print(_row("  │  Max Open Positions",      MAX_OPEN_POSITIONS))
    print(_row("  │  Max Risk / Trade",        _pct(MAX_RISK_PER_TRADE)))
    print(_row("  │  Max Ticker Exposure",     _pct(MAX_TICKER_EXPOSURE)))
    print(_row("  │  Daily Loss Limit [halt]", _pct(DAILY_LOSS_LIMIT)))
    print(_row("  │  Warning Drawdown",        _pct(WARNING_DRAWDOWN)))
    print("  └" + "─" * 70 + "┘")

    # ── Signal Thresholds ─────────────────────────────────────────────────────
    print()
    print("  ┌─ SIGNAL THRESHOLDS " + "─" * 50 + "┐")
    print(_row("  │  Min Signal Confidence",       _pct(MIN_SIGNAL_CONFIDENCE)))
    print(_row("  │  Elite Signal Confidence",     _pct(ELITE_SIGNAL_CONFIDENCE)))
    print(_row("  │  Conviction Bonus Threshold",  _pct(CONVICTION_BONUS_THRESHOLD)))
    print(_row("  │  Signal Half-Life",            f"{SIGNAL_HALF_LIFE_MINUTES} min"))
    print("  └" + "─" * 70 + "┘")

    # ── Timing Controls ───────────────────────────────────────────────────────
    print()
    print("  ┌─ TIMING CONTROLS " + "─" * 52 + "┐")
    print(_row("  │  News Poll Interval",       f"{NEWS_POLL_INTERVAL_SECONDS} s"))
    print(_row("  │  Trade Cooldown",           f"{COOLDOWN_MINUTES} min"))
    print(_row("  │  Macro Freeze Pre-Event",   f"{MACRO_FREEZE_PRE_MINUTES} min"))
    print(_row("  │  Macro Freeze Post-Event",  f"{MACRO_FREEZE_POST_MINUTES} min"))
    print(_row("  │  Failure Back-off Base",    f"{FAILURE_BACKOFF_BASE_SECONDS} s"))
    print("  └" + "─" * 70 + "┘")

    # ── Execution Templates ───────────────────────────────────────────────────
    print()
    print("  ┌─ EXECUTION TEMPLATES " + "─" * 48 + "┐")
    for asset_class, tpl in EXECUTION_TEMPLATES.items():
        print(f"  │")
        print(f"  │  [{asset_class}]  {tpl['description']}")
        print(_row("  │    Std / Aggressive Size",
                   f"{_pct(tpl['standard_size_pct'])} / {_pct(tpl['aggressive_size_pct'])}"))
        print(_row("  │    SL / TP",
                   f"{_pct(tpl['stop_loss_pct'])} / {_pct(tpl['take_profit_pct'])}"))
        print(_row("  │    Trailing Stop",        _pct(tpl['trailing_stop_pct'])))
        print(_row("  │    Conviction Multiplier", f"x{tpl['conviction_multiplier']:.2f}"))
        print(_row("  │    Order Type / TIF",
                   f"{tpl['order_type'].upper()} / {tpl['time_in_force']}"))
    print("  │")
    print("  └" + "─" * 70 + "┘")

    # ── External Integrations ─────────────────────────────────────────────────
    def _set(val: str) -> str:
        return "SET ✓" if val else "NOT SET ✗"

    print()
    print("  ┌─ EXTERNAL INTEGRATIONS " + "─" * 46 + "┐")
    print(_row("  │  Alpaca API Key",    _set(ALPACA_API_KEY)))
    print(_row("  │  Alpaca Secret",     _set(ALPACA_SECRET)))
    print(_row("  │  Alpaca Base URL",   ALPACA_BASE_URL))
    print(_row("  │  IBKR Host : Port",  f"{IBKR_HOST} : {IBKR_PORT}"))
    print(_row("  │  Binance Key",       _set(BINANCE_KEY)))
    print(_row("  │  Binance Base URL",  BINANCE_BASE_URL))
    print(_row("  │  Discord Webhook",   _set(DISCORD_WEBHOOK)))
    print(_row("  │  N8N Webhook",       _set(N8N_WEBHOOK)))
    print(_row("  │  NewsAPI Key",       _set(NEWSAPI_KEY)))
    print(_row("  │  OpenAI API Key",    _set(OPENAI_API_KEY)))
    print("  └" + "─" * 70 + "┘")

    # ── File Paths ────────────────────────────────────────────────────────────
    print()
    print("  ┌─ FILE PATHS " + "─" * 57 + "┐")
    print(_row("  │  State File",         STATE_FILE))
    print(_row("  │  Log File",           LOG_FILE))
    print(_row("  │  Reports Dir",        REPORTS_DIR))
    print(_row("  │  Plugin Dir",         PLUGIN_DIR))
    print(_row("  │  Duplicate Cache",    DUPLICATE_CACHE_FILE))
    print("  └" + "─" * 70 + "┘")
    print()
    print(_bar())
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python config.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_summary()

    print("  Running validation …")
    ok = validate_config()

    print()
    if ok:
        # Quick sanity-check on as_dict()
        cfg = as_dict()
        assert isinstance(cfg, dict), "as_dict() must return a dict"
        assert "EXECUTION_TEMPLATES" in cfg, "EXECUTION_TEMPLATES missing from as_dict()"
        assert all(
            k in cfg["EXECUTION_TEMPLATES"]
            for k in ("CRYPTO", "EQUITY", "FOREX", "MACRO")
        ), "Missing asset class in EXECUTION_TEMPLATES"
        print("  ✓  as_dict() integrity check passed.")
        print()
        print("  ✓  Smoke test PASSED — config.py is ready for production.")
        print()
        sys.exit(0)
    else:
        print("  ✗  Smoke test FAILED — fix the errors above before deploying.")
        print()
        sys.exit(1)