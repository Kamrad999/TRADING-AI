"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   ███████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗                                    ║
║   ██╔════╝██║██╔════╝ ████╗  ██║██╔══██╗██║                                    ║
║   ███████╗██║██║  ███╗██╔██╗ ██║███████║██║                                    ║
║   ╚════██║██║██║   ██║██║╚██╗██║██╔══██║██║                                    ║
║   ███████║██║╚██████╔╝██║ ╚████║██║  ██║███████╗                               ║
║   ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝                              ║
║                                                                                  ║
║   ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗                             ║
║   ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝                             ║
║   █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗                               ║
║   ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝                               ║
║   ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗                             ║
║   ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝                            ║
║                                                                                  ║
║   ════════════════════════════════════════════════════════════════════════════  ║
║                    MONSTER TRADING AI — SIGNAL ENGINE                           ║
║                          signal_engine.py  v1.0.0                               ║
║   ════════════════════════════════════════════════════════════════════════════  ║
║                                                                                  ║
║   PIPELINE POSITION                                                              ║
║   ─────────────────────────────────────────────────────────────────────────    ║
║   news_engine.py                                                                 ║
║     → duplicate_filter.py                                                       ║
║       → fake_news_validator.py                                                   ║
║         → signal_engine.py          ◄ YOU ARE HERE                              ║
║           → alert_router.py                                                      ║
║             → execution_bridge.py                                                ║
║                                                                                  ║
║   MISSION                                                                        ║
║   ─────────────────────────────────────────────────────────────────────────    ║
║   Transform validator-enriched financial news articles into actionable           ║
║   directional trading signals with confidence scores, market regime bias,        ║
║   position sizing guidance, and execution priority rankings.                     ║
║                                                                                  ║
║   This is NOT keyword matching. This module answers three questions:             ║
║     1. Should the system act on this event?                                      ║
║     2. In which direction?                                                       ║
║     3. With what size and urgency?                                                ║
║                                                                                  ║
║   ARCHITECTURE (10-Layer Stack)                                                  ║
║   ─────────────────────────────────────────────────────────────────────────    ║
║   L1  PUBLIC API          generate_signals()                                    ║
║   L2  ORCHESTRATOR        _generate_single_signal()                             ║
║   L3  EVENT CLASSIFIER    _classify_event_type()                                ║
║   L4  DIRECTIONAL ENGINE  _compute_signal_direction()                           ║
║   L5  IMPACT SCORER       _compute_impact_score()                               ║
║   L6  CONFIDENCE FUSION   _compute_confidence_score()                           ║
║   L7  REGIME BIAS         _infer_market_regime()                                ║
║   L8  POSITION SIZING     _compute_position_size_bias()                         ║
║   L9  EXECUTION PRIORITY  _compute_execution_priority()                         ║
║   L10 SAFETY HELPERS      _safe_str / _safe_float / _clamp / _log              ║
║                                                                                  ║
║   PERFORMANCE TARGET                                                             ║
║   ─────────────────────────────────────────────────────────────────────────    ║
║   ≥ 1,000 articles processed in < 100ms                                         ║
║   All regex precompiled at module level. O(1) lookups via frozensets/dicts.     ║
║                                                                                  ║
║   CONSTRAINTS                                                                    ║
║   ─────────────────────────────────────────────────────────────────────────    ║
║   • Pure Python standard library only — no numpy, no pandas, no async           ║
║   • Fully stateless — no global mutable state                                   ║
║   • Never mutates input dicts — shallow-copies all outputs                      ║
║   • Deterministic — same input always yields same output                        ║
║                                                                                  ║
║   AUTHOR:   Monster Trading AI Infrastructure Team                               ║
║   VERSION:  1.0.0                                                                ║
║   UPDATED:  2026-04-11                                                           ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import re
import sys
import time
import logging
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
#  MODULE LOGGER
#  Verbosity controlled via VERBOSITY constant below.
# ──────────────────────────────────────────────────────────────────────────────

_logger = logging.getLogger("signal_engine")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [SIGNAL] %(levelname)s — %(message)s", "%H:%M:%S"))
    _logger.addHandler(_handler)

VERBOSITY: int = 1
"""
Logging verbosity gate.
  0 = silent
  1 = INFO  (production default)
  2 = DEBUG (detailed layer tracing)
"""

# ──────────────────────────────────────────────────────────────────────────────
#  SIGNAL OUTPUT FIELD DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

_SIGNAL_DEFAULTS: dict[str, Any] = {
    "signal_direction":     "NO_TRADE",
    "signal_strength":      0.0,
    "impact_score":         0.0,
    "confidence_score":     0.0,
    "event_type":           "UNKNOWN",
    "urgency":              "LOW",
    "execution_priority":   5,
    "market_regime":        "NEUTRAL",
    "position_size_bias":   0.25,
    "signal_reasons":       [],
}

# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗      ██████╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ╚════██╗
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝      ███╔═╝
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗    ██╔══╝
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║    ███████╗
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    ╚══════╝
#
#  LAYER 3 — EVENT CLASSIFICATION PATTERNS
#  Precompiled at module load for O(1) amortised cost per article.
#
# ══════════════════════════════════════════════════════════════════════════════

_RE_FLAGS = re.IGNORECASE

# ── Event-type detection patterns ─────────────────────────────────────────────

_EVENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "EARNINGS",
        re.compile(
            r"\b(earnings|EPS|revenue|beat estimates|missed estimates|guidance|"
            r"quarterly results|net income|operating income|profit|loss per share|"
            r"raised guidance|lowered guidance|Q[1-4] results|fiscal year)\b",
            _RE_FLAGS,
        ),
    ),
    (
        "M&A",
        re.compile(
            r"\b(merger|acquisition|acquires|takeover|buyout|deal|bid|acquired by|"
            r"merger approved|hostile takeover|private equity|LBO|spin-off|divestiture)\b",
            _RE_FLAGS,
        ),
    ),
    (
        "REGULATORY",
        re.compile(
            r"\b(SEC|CFTC|DOJ|FTC|FDA|regulatory|investigation|subpoena|lawsuit|"
            r"settlement|fine|penalty|compliance|antitrust|probe|class action|"
            r"consent order|enforcement action|license revoked)\b",
            _RE_FLAGS,
        ),
    ),
    (
        "CRYPTO",
        re.compile(
            r"\b(bitcoin|BTC|ethereum|ETH|crypto|blockchain|DeFi|NFT|stablecoin|"
            r"altcoin|token|digital asset|web3|ETF approval|spot bitcoin ETF|"
            r"mining|hash rate|layer 2|solana|ripple|XRP)\b",
            _RE_FLAGS,
        ),
    ),
    (
        "MACRO",
        re.compile(
            r"\b(Fed|Federal Reserve|interest rate|rate cut|rate hike|inflation|CPI|"
            r"PPI|GDP|jobs report|unemployment|payroll|treasury|bond yield|FOMC|"
            r"monetary policy|fiscal policy|recession|stimulus|quantitative easing|"
            r"quantitative tightening|debt ceiling|trade war|tariff|sanctions|"
            r"geopolitical|central bank|ECB|IMF|World Bank|PMI|ISM)\b",
            _RE_FLAGS,
        ),
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗     ██╗  ██╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██║  ██║
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝    ███████║
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗    ╚════██║
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║         ██║
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝         ╚═╝
#
#  LAYER 4 — DIRECTIONAL SIGNAL PHRASE LEXICONS
#  Bullish / Bearish scored phrase banks with per-phrase weights.
#  frozenset membership used where weight uniformity allows O(1) check.
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Bullish signal phrase map: phrase → weight (0.0–1.0) ─────────────────────
_BULLISH_PHRASES: dict[str, float] = {
    # Monetary easing
    "rate cut":                 0.90,
    "rate cuts":                0.90,
    "dovish":                   0.75,
    "easing":                   0.70,
    "quantitative easing":      0.80,
    "stimulus":                 0.80,
    "bailout":                  0.60,
    # Earnings / guidance
    "beat estimates":           0.85,
    "beats estimates":          0.85,
    "raised guidance":          0.90,
    "raises guidance":          0.90,
    "record revenue":           0.85,
    "record profit":            0.85,
    "strong earnings":          0.80,
    "profit surge":             0.80,
    "earnings beat":            0.85,
    "revenue growth":           0.70,
    "margin expansion":         0.75,
    # Corporate actions
    "buyback":                  0.70,
    "share repurchase":         0.70,
    "dividend increase":        0.75,
    "special dividend":         0.65,
    "merger approved":          0.80,
    "acquisition completed":    0.70,
    "deal closed":              0.65,
    # Regulatory / macro positive
    "ETF approved":             0.95,
    "approved":                 0.55,
    "cleared":                  0.50,
    "lifted":                   0.55,
    "recovery":                 0.65,
    "upgrade":                  0.70,
    "upgraded":                 0.70,
    "buy rating":               0.75,
    "outperform":               0.70,
    "overweight":               0.65,
    # Macro positive
    "inflation falls":          0.80,
    "inflation eases":          0.80,
    "jobs added":               0.65,
    "unemployment falls":       0.70,
    "GDP growth":               0.70,
    "strong growth":            0.70,
    "easing inflation":         0.80,
}

# ── Bearish signal phrase map: phrase → weight (0.0–1.0) ─────────────────────
_BEARISH_PHRASES: dict[str, float] = {
    # Monetary tightening
    "rate hike":                0.90,
    "rate hikes":               0.90,
    "hawkish":                  0.75,
    "tightening":               0.70,
    "quantitative tightening":  0.80,
    # Earnings / guidance
    "missed estimates":         0.85,
    "misses estimates":         0.85,
    "lowered guidance":         0.90,
    "lowers guidance":          0.90,
    "earnings miss":            0.85,
    "profit warning":           0.90,
    "revenue decline":          0.75,
    "margin compression":       0.75,
    "loss reported":            0.80,
    # Corporate distress
    "layoffs":                  0.70,
    "job cuts":                 0.70,
    "restructuring":            0.60,
    "bankruptcy":               0.95,
    "default":                  0.90,
    "debt crisis":              0.85,
    "dilution":                 0.75,
    "secondary offering":       0.65,
    "share sale":               0.60,
    # Regulatory / legal
    "SEC investigation":        0.85,
    "fraud":                    0.90,
    "investigation":            0.65,
    "fine":                     0.60,
    "penalty":                  0.60,
    "lawsuit":                  0.55,
    "probe":                    0.65,
    "subpoena":                 0.70,
    # Analyst negative
    "downgrade":                0.75,
    "downgraded":               0.75,
    "sell rating":              0.75,
    "underperform":             0.70,
    "underweight":              0.65,
    "price target cut":         0.70,
    # Macro negative
    "recession":                0.85,
    "recession fears":          0.90,
    "bank crisis":              0.90,
    "inflation surge":          0.80,
    "supply shock":             0.75,
    "credit crunch":            0.85,
    "yield inversion":          0.75,
    "contagion":                0.80,
    "geopolitical risk":        0.65,
    "war":                      0.70,
    "sanctions":                0.65,
}

# ── Urgency detection patterns ────────────────────────────────────────────────
_URGENCY_CRITICAL: re.Pattern = re.compile(
    r"\b(breaking|urgent|flash|emergency|just in|halt|circuit breaker|"
    r"immediate|imminent|crisis|collapse|crash|historic|unprecedented)\b",
    _RE_FLAGS,
)

_URGENCY_HIGH: re.Pattern = re.compile(
    r"\b(warns|warning|alert|surges|plunges|soars|tumbles|spikes|skyrockets|"
    r"plummets|record high|record low|biggest|largest|massive|major)\b",
    _RE_FLAGS,
)

_URGENCY_MEDIUM: re.Pattern = re.compile(
    r"\b(rises|falls|drops|climbs|declines|beats|misses|reports|announces|"
    r"confirms|reveals|discloses)\b",
    _RE_FLAGS,
)

# ── Regime signal patterns ────────────────────────────────────────────────────
_REGIME_RISK_ON: re.Pattern = re.compile(
    r"\b(rate cut|easing inflation|stimulus|recovery|GDP growth|"
    r"ETF approved|strong jobs|unemployment falls|buyback|"
    r"merger approved|earnings beat|raised guidance|dovish)\b",
    _RE_FLAGS,
)

_REGIME_RISK_OFF: re.Pattern = re.compile(
    r"\b(recession|rate hike|bank crisis|inflation surge|credit crunch|"
    r"geopolitical|war|default|bankruptcy|fraud|investigation|"
    r"quantitative tightening|debt crisis|hawkish|yield inversion|contagion)\b",
    _RE_FLAGS,
)

# ── Event-type severity weights (used in impact scoring) ─────────────────────
_EVENT_SEVERITY: dict[str, float] = {
    "EARNINGS":    0.75,
    "M&A":         0.80,
    "REGULATORY":  0.85,
    "CRYPTO":      0.70,
    "MACRO":       0.90,
    "UNKNOWN":     0.40,
}

# ── Priority label → numeric priority ────────────────────────────────────────
_PRIORITY_STR_MAP: dict[str, int] = {
    "high":   1,
    "medium": 3,
    "low":    5,
    "normal": 3,
}

# ── Urgency → numeric tier for priority mapping ───────────────────────────────
_URGENCY_TIER: dict[str, int] = {
    "CRITICAL": 4,
    "HIGH":     3,
    "MEDIUM":   2,
    "LOW":      1,
}

# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗     ██╗ ██████╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ███║██╔═████╗
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝    ╚██║██║██╔██║
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗     ██║████╔╝██║
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║     ██║╚██████╔╝
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝     ╚═╝ ╚═════╝
#
#  LAYER 10 — SAFETY HELPERS
#  Defined first so all upper layers can reference them.
#
# ══════════════════════════════════════════════════════════════════════════════


def _safe_str(value: Any, default: str = "") -> str:
    """
    Safely coerce *value* to a non-None string.

    Parameters
    ----------
    value   : any incoming field value
    default : fallback when value is None or not string-coercible

    Returns
    -------
    str — guaranteed non-None
    """
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely coerce *value* to float.

    Parameters
    ----------
    value   : any incoming field value
    default : fallback on failure (NaN, None, non-numeric string)

    Returns
    -------
    float — guaranteed numeric
    """
    if value is None:
        return default
    try:
        result = float(value)
        # Guard against NaN / Inf
        if result != result or result == float("inf") or result == float("-inf"):
            return default
        return result
    except (ValueError, TypeError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Clamp *value* to the closed interval [lo, hi].

    Parameters
    ----------
    value : float to constrain
    lo    : lower bound (inclusive)
    hi    : upper bound (inclusive)

    Returns
    -------
    float in [lo, hi]
    """
    return max(lo, min(hi, value))


def _log(level: int, message: str) -> None:
    """
    Emit a log message if VERBOSITY is at or above *level*.

    Parameters
    ----------
    level   : minimum VERBOSITY required to emit (1=INFO, 2=DEBUG)
    message : log message body

    Returns
    -------
    None
    """
    if VERBOSITY >= level:
        if level == 1:
            _logger.info(message)
        else:
            _logger.debug(message)


# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗      ██████╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ╚════██╗
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝        ██╔╝
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗       ██╔╝
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║       ██║
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝       ╚═╝
#
#  LAYER 3 — EVENT CLASSIFICATION
#
# ══════════════════════════════════════════════════════════════════════════════


def _classify_event_type(text: str) -> tuple[str, list[str]]:
    """
    Classify a news article into one of the canonical event type buckets
    using precompiled regex patterns.

    Classification precedence (top = highest priority):
        EARNINGS → M&A → REGULATORY → CRYPTO → MACRO → UNKNOWN

    Parameters
    ----------
    text : concatenated title + summary of the article

    Returns
    -------
    tuple[str, list[str]]
        event_type : one of EARNINGS | M&A | REGULATORY | CRYPTO | MACRO | UNKNOWN
        reasons    : list of human-readable classification rationale strings

    Notes
    -----
    Patterns are checked in declared order. The first match wins.
    Multiple patterns may match; only the first is used for classification
    but all matches are noted in reasons for explainability.
    """
    reasons: list[str] = []

    for event_type, pattern in _EVENT_PATTERNS:
        match = pattern.search(text)
        if match:
            reasons.append(f"Event classified as {event_type} — matched term: '{match.group(0)}'")
            _log(2, f"Event type resolved → {event_type} (trigger: '{match.group(0)}')")
            return event_type, reasons

    _log(2, "Event type resolved → UNKNOWN (no pattern matched)")
    reasons.append("No event-type pattern matched — classified as UNKNOWN")
    return "UNKNOWN", reasons


# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗     ██╗  ██╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██║  ██║
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝    ███████║
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗    ╚════██║
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║         ██║
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝         ╚═╝
#
#  LAYER 4 — DIRECTIONAL SIGNAL ENGINE
#
# ══════════════════════════════════════════════════════════════════════════════

# Minimum net score differential required to commit a direction (vs NO_TRADE)
_DIRECTION_MIN_DELTA: float = 0.15


def _compute_signal_direction(
    text: str,
) -> tuple[str, float, list[str]]:
    """
    Score the article text against bullish and bearish phrase banks to
    determine the directional trading signal.

    Algorithm
    ---------
    1. Scan *text* for each phrase in _BULLISH_PHRASES.
       Accumulate total bullish weight.
    2. Repeat for _BEARISH_PHRASES.
    3. Normalize both scores to [0,1] by dividing by the maximum
       possible score in each bank (prevents bank-size bias).
    4. Compute net delta = bullish_norm − bearish_norm.
    5. If |delta| < _DIRECTION_MIN_DELTA → NO_TRADE (ambiguous signal).
    6. Otherwise → BUY (delta > 0) or SELL (delta < 0).

    Parameters
    ----------
    text : full article text (title + summary, lowered externally)

    Returns
    -------
    tuple[str, float, list[str]]
        signal_direction : BUY | SELL | NO_TRADE
        signal_strength  : float [0,1]
        reasons          : explainability list
    """
    reasons: list[str] = []

    # ── Bullish scoring ───────────────────────────────────────────────────────
    raw_bull: float = 0.0
    bull_hits: list[str] = []
    for phrase, weight in _BULLISH_PHRASES.items():
        if phrase in text:
            raw_bull += weight
            bull_hits.append(phrase)

    # ── Bearish scoring ───────────────────────────────────────────────────────
    raw_bear: float = 0.0
    bear_hits: list[str] = []
    for phrase, weight in _BEARISH_PHRASES.items():
        if phrase in text:
            raw_bear += weight
            bear_hits.append(phrase)

    # ── Normalization ─────────────────────────────────────────────────────────
    max_bull = sum(_BULLISH_PHRASES.values()) or 1.0
    max_bear = sum(_BEARISH_PHRASES.values()) or 1.0
    bull_norm = _clamp(raw_bull / max_bull)
    bear_norm = _clamp(raw_bear / max_bear)

    delta = bull_norm - bear_norm

    if bull_hits:
        reasons.append(f"Bullish signals detected: {', '.join(bull_hits[:5])}")
    if bear_hits:
        reasons.append(f"Bearish signals detected: {', '.join(bear_hits[:5])}")

    # ── Direction resolution ──────────────────────────────────────────────────
    if abs(delta) < _DIRECTION_MIN_DELTA:
        reasons.append(
            f"Signal conflict unresolved (delta={delta:.3f} < threshold={_DIRECTION_MIN_DELTA}) → NO_TRADE"
        )
        _log(2, f"Direction → NO_TRADE (delta={delta:.3f})")
        return "NO_TRADE", 0.0, reasons

    if delta > 0:
        strength = _clamp(delta * 2)  # scale delta to use full [0,1] range
        reasons.append(f"Net bullish bias (Δ={delta:.3f}) → BUY | strength={strength:.3f}")
        _log(2, f"Direction → BUY (Δ={delta:.3f}, strength={strength:.3f})")
        return "BUY", strength, reasons

    strength = _clamp(abs(delta) * 2)
    reasons.append(f"Net bearish bias (Δ={delta:.3f}) → SELL | strength={strength:.3f}")
    _log(2, f"Direction → SELL (Δ={delta:.3f}, strength={strength:.3f})")
    return "SELL", strength, reasons


# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗     ███████╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██╔════╝
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝    ███████╗
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗    ╚════██║
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║     ███████║
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝     ╚══════╝
#
#  LAYER 5 — IMPACT SCORING
#
# ══════════════════════════════════════════════════════════════════════════════


def _compute_urgency(text: str) -> tuple[str, float, list[str]]:
    """
    Determine the news urgency level from the article text.

    Urgency tiers (descending):
        CRITICAL → HIGH → MEDIUM → LOW

    Returns
    -------
    tuple[str, float, list[str]]
        urgency      : CRITICAL | HIGH | MEDIUM | LOW
        urgency_mult : multiplier to feed into impact scoring [0.5–1.0]
        reasons      : explainability strings
    """
    reasons: list[str] = []

    if _URGENCY_CRITICAL.search(text):
        match_word = _URGENCY_CRITICAL.search(text).group(0)
        reasons.append(f"CRITICAL urgency detected — trigger word: '{match_word}'")
        return "CRITICAL", 1.00, reasons

    if _URGENCY_HIGH.search(text):
        match_word = _URGENCY_HIGH.search(text).group(0)
        reasons.append(f"HIGH urgency detected — trigger word: '{match_word}'")
        return "HIGH", 0.80, reasons

    if _URGENCY_MEDIUM.search(text):
        match_word = _URGENCY_MEDIUM.search(text).group(0)
        reasons.append(f"MEDIUM urgency detected — trigger word: '{match_word}'")
        return "MEDIUM", 0.60, reasons

    reasons.append("LOW urgency — no urgency trigger words detected")
    return "LOW", 0.40, reasons


def _compute_impact_score(
    article: dict,
    event_type: str,
    urgency_mult: float,
    signal_strength: float,
) -> tuple[float, list[str]]:
    """
    Compute the composite market impact score for a validated article.

    Formula
    -------
    impact = (
        priority_weight  * 0.20 +
        event_severity   * 0.25 +
        cluster_depth    * 0.15 +
        validation_score * 0.25 +
        urgency_mult     * 0.15
    ) × signal_strength_boost

    Where signal_strength_boost ∈ [0.8, 1.2] — strong signals amplify impact,
    weak/no-trade signals slightly suppress it.

    Parameters
    ----------
    article         : validated article dict (upstream fields present)
    event_type      : classified event type string
    urgency_mult    : urgency tier multiplier [0.4–1.0]
    signal_strength : directional strength [0,1]

    Returns
    -------
    tuple[float, list[str]]
        impact_score : float [0,1]
        reasons      : explainability strings
    """
    reasons: list[str] = []

    # ── Article priority ──────────────────────────────────────────────────────
    raw_priority = _safe_str(article.get("priority", ""), "low").lower()
    priority_weight = _PRIORITY_STR_MAP.get(raw_priority, 3) / 5.0  # normalise 1–5 → 0.2–1.0
    reasons.append(f"Article priority='{raw_priority}' → weight={priority_weight:.2f}")

    # ── Event type severity ───────────────────────────────────────────────────
    event_severity = _EVENT_SEVERITY.get(event_type, 0.40)
    reasons.append(f"Event severity for {event_type} → {event_severity:.2f}")

    # ── Source cluster depth ──────────────────────────────────────────────────
    # cluster_size or source_count field from upstream (news_engine)
    cluster_raw = _safe_float(article.get("cluster_size", article.get("source_count", 1)), 1.0)
    cluster_depth = _clamp(cluster_raw / 10.0)   # normalise — 10+ sources = max
    reasons.append(f"Source cluster size={cluster_raw:.0f} → depth={cluster_depth:.2f}")

    # ── Upstream validation score ─────────────────────────────────────────────
    validation_score = _clamp(_safe_float(article.get("validation_score", 0.5)))

    # ── Weighted composite ────────────────────────────────────────────────────
    raw_impact = (
        priority_weight  * 0.20 +
        event_severity   * 0.25 +
        cluster_depth    * 0.15 +
        validation_score * 0.25 +
        urgency_mult     * 0.15
    )

    # Signal strength boost: high-conviction direction amplifies impact
    if signal_strength >= 0.70:
        boost = 1.15
        reasons.append("High-conviction direction detected — impact boosted ×1.15")
    elif signal_strength >= 0.40:
        boost = 1.00
    else:
        boost = 0.85
        reasons.append("Weak directional signal — impact suppressed ×0.85")

    impact_score = _clamp(raw_impact * boost)
    reasons.append(f"Composite impact_score={impact_score:.3f}")
    _log(2, f"Impact score computed → {impact_score:.3f}")
    return impact_score, reasons


# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗      ██████╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██╔════╝
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝     ██████╗
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗         ██╗
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║     ██████╔╝
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝     ╚═════╝
#
#  LAYER 6 — CONFIDENCE FUSION
#
# ══════════════════════════════════════════════════════════════════════════════


def _compute_confidence_score(
    validation_score: float,
    impact_score: float,
    regime_alignment: float,
) -> tuple[float, list[str]]:
    """
    Fuse upstream trust signals into a single confidence score.

    Formula (from spec)
    -------------------
    confidence = (
        validation_score * 0.35 +
        impact_score     * 0.45 +
        regime_alignment * 0.20
    )

    Clamped to [0,1].

    Parameters
    ----------
    validation_score : upstream fake_news_validator.py score [0,1]
    impact_score     : layer-5 market impact score [0,1]
    regime_alignment : regime alignment bonus [0,1] from layer 7

    Returns
    -------
    tuple[float, list[str]]
        confidence_score : float [0,1]
        reasons          : explainability strings
    """
    raw = (
        validation_score * 0.35 +
        impact_score     * 0.45 +
        regime_alignment * 0.20
    )
    score = _clamp(raw)
    reasons = [
        f"Confidence fusion: val={validation_score:.3f}×0.35 + "
        f"impact={impact_score:.3f}×0.45 + "
        f"regime_align={regime_alignment:.3f}×0.20 → {score:.3f}"
    ]
    _log(2, f"Confidence score → {score:.3f}")
    return score, reasons


# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗     ███████╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ╚════██╗
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝        ███╗
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗         ██║
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║     ██████╔╝
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝     ╚═════╝
#
#  LAYER 7 — MARKET REGIME BIAS
#
# ══════════════════════════════════════════════════════════════════════════════


def _infer_market_regime(
    text: str,
    signal_direction: str,
) -> tuple[str, float, list[str]]:
    """
    Infer the macro market regime that this article implies and return
    an alignment score indicating how well the article fits that regime.

    Regime taxonomy
    ---------------
    RISK_ON  : easing conditions, positive growth, pro-market events
    RISK_OFF : tightening, crisis, fear, fraud, geopolitical stress
    NEUTRAL  : ambiguous or no discernible macro signal

    Alignment score
    ---------------
    Used downstream in confidence fusion (layer 6).
    If signal direction and regime match logically, alignment = 0.80–1.00.
    If they conflict (e.g., BUY during RISK_OFF), alignment = 0.30–0.50.
    NEUTRAL regime → alignment = 0.50 always.

    Parameters
    ----------
    text             : full article text (lowercased)
    signal_direction : BUY | SELL | NO_TRADE from layer 4

    Returns
    -------
    tuple[str, float, list[str]]
        regime          : RISK_ON | RISK_OFF | NEUTRAL
        regime_alignment: float [0,1] for confidence fusion
        reasons         : explainability strings
    """
    reasons: list[str] = []

    risk_on_matches  = len(_REGIME_RISK_ON.findall(text))
    risk_off_matches = len(_REGIME_RISK_OFF.findall(text))

    _log(2, f"Regime scan → RISK_ON hits={risk_on_matches}, RISK_OFF hits={risk_off_matches}")

    if risk_on_matches == 0 and risk_off_matches == 0:
        reasons.append("No regime signals detected → NEUTRAL")
        return "NEUTRAL", 0.50, reasons

    if risk_on_matches > risk_off_matches:
        regime = "RISK_ON"
        reasons.append(f"RISK_ON dominant ({risk_on_matches} vs {risk_off_matches} RISK_OFF signals)")
        # Alignment: BUY aligns with RISK_ON
        if signal_direction == "BUY":
            alignment = 0.90
            reasons.append("Signal direction BUY aligns with RISK_ON regime → alignment=0.90")
        elif signal_direction == "SELL":
            alignment = 0.30
            reasons.append("Signal direction SELL conflicts with RISK_ON regime → alignment=0.30")
        else:
            alignment = 0.55
            reasons.append("Signal direction NO_TRADE in RISK_ON regime → alignment=0.55")

    elif risk_off_matches > risk_on_matches:
        regime = "RISK_OFF"
        reasons.append(f"RISK_OFF dominant ({risk_off_matches} vs {risk_on_matches} RISK_ON signals)")
        # Alignment: SELL aligns with RISK_OFF
        if signal_direction == "SELL":
            alignment = 0.90
            reasons.append("Signal direction SELL aligns with RISK_OFF regime → alignment=0.90")
        elif signal_direction == "BUY":
            alignment = 0.30
            reasons.append("Signal direction BUY conflicts with RISK_OFF regime → alignment=0.30")
        else:
            alignment = 0.55
            reasons.append("Signal direction NO_TRADE in RISK_OFF regime → alignment=0.55")

    else:
        # Equal matches → mixed regime
        regime = "NEUTRAL"
        alignment = 0.50
        reasons.append(
            f"Equal RISK_ON/RISK_OFF signals ({risk_on_matches} each) → NEUTRAL | alignment=0.50"
        )

    _log(2, f"Market regime → {regime} (alignment={alignment:.2f})")
    return regime, alignment, reasons


# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗      █████╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██╔══██╗
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝     ╚█████╔╝
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗    ██╔══██╗
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║    ╚█████╔╝
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝     ╚════╝
#
#  LAYER 8 — POSITION SIZING BIAS
#
# ══════════════════════════════════════════════════════════════════════════════


def _compute_position_size_bias(
    confidence_score: float,
    urgency: str,
    signal_direction: str,
) -> tuple[float, list[str]]:
    """
    Derive a position sizing multiplier that downstream execution_bridge.py
    can apply directly to base position sizes.

    Tier table
    ----------
    exceptional  (conf ≥ 0.80 AND urgency ∈ {CRITICAL, HIGH})  → 1.5
    strong       (conf ≥ 0.65 AND urgency ∈ {HIGH, MEDIUM})    → 1.2
    normal       (conf ≥ 0.45)                                  → 1.0
    cautious     (conf ≥ 0.30)                                  → 0.5
    minimal      (below 0.30 OR NO_TRADE)                       → 0.25

    Parameters
    ----------
    confidence_score : fused confidence [0,1]
    urgency          : CRITICAL | HIGH | MEDIUM | LOW
    signal_direction : BUY | SELL | NO_TRADE

    Returns
    -------
    tuple[float, list[str]]
        position_size_bias : float [0.25, 1.5]
        reasons            : explainability strings
    """
    reasons: list[str] = []

    if signal_direction == "NO_TRADE":
        reasons.append("NO_TRADE signal → position_size_bias=0.25 (no position)")
        return 0.25, reasons

    high_urgency = urgency in {"CRITICAL", "HIGH"}
    med_urgency  = urgency in {"HIGH", "MEDIUM"}

    if confidence_score >= 0.80 and high_urgency:
        bias = 1.50
        reasons.append(
            f"Exceptional conviction (conf={confidence_score:.3f}, urgency={urgency}) → bias=1.50"
        )
    elif confidence_score >= 0.65 and med_urgency:
        bias = 1.20
        reasons.append(
            f"Strong conviction (conf={confidence_score:.3f}, urgency={urgency}) → bias=1.20"
        )
    elif confidence_score >= 0.45:
        bias = 1.00
        reasons.append(f"Normal conviction (conf={confidence_score:.3f}) → bias=1.00")
    elif confidence_score >= 0.30:
        bias = 0.50
        reasons.append(f"Cautious conviction (conf={confidence_score:.3f}) → bias=0.50")
    else:
        bias = 0.25
        reasons.append(f"Minimal conviction (conf={confidence_score:.3f}) → bias=0.25")

    _log(2, f"Position size bias → {bias}")
    return bias, reasons


# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗      █████╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██╔══██╗
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝     ╚████╔╝
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗    ██╔══╝
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║    ███████╗
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    ╚══════╝
#
#  LAYER 9 — EXECUTION PRIORITY
#
# ══════════════════════════════════════════════════════════════════════════════

# Priority mapping: (urgency_tier, impact_bucket) → execution_priority 1–5
# urgency_tier : 4=CRITICAL, 3=HIGH, 2=MEDIUM, 1=LOW
# impact_bucket: 2=high (≥0.7), 1=medium (≥0.4), 0=low (<0.4)
_PRIORITY_MATRIX: dict[tuple[int, int], int] = {
    (4, 2): 1,  # CRITICAL + high impact  → immediate execution
    (4, 1): 1,  # CRITICAL + medium impact → immediate execution
    (4, 0): 2,  # CRITICAL + low impact   → near-immediate
    (3, 2): 1,  # HIGH     + high impact  → immediate
    (3, 1): 2,  # HIGH     + medium impact → high priority
    (3, 0): 3,  # HIGH     + low impact   → medium priority
    (2, 2): 2,  # MEDIUM   + high impact  → high priority
    (2, 1): 3,  # MEDIUM   + medium impact → medium priority
    (2, 0): 4,  # MEDIUM   + low impact   → low priority
    (1, 2): 3,  # LOW      + high impact  → medium priority
    (1, 1): 4,  # LOW      + medium impact → low priority
    (1, 0): 5,  # LOW      + low impact   → watchlist
}


def _compute_execution_priority(
    urgency: str,
    impact_score: float,
    signal_direction: str,
) -> tuple[int, list[str]]:
    """
    Map urgency and impact to a 1–5 execution priority ranking.

    Priority semantics
    ------------------
    1 = Execute immediately — send to execution_bridge with no delay
    2 = High priority     — route within seconds
    3 = Medium priority   — route within minutes
    4 = Low priority      — queue for next sweep
    5 = Watchlist         — log only, no active routing

    NO_TRADE signals always receive priority 5.

    Parameters
    ----------
    urgency          : CRITICAL | HIGH | MEDIUM | LOW
    impact_score     : float [0,1] from layer 5
    signal_direction : BUY | SELL | NO_TRADE

    Returns
    -------
    tuple[int, list[str]]
        execution_priority : int 1–5
        reasons            : explainability strings
    """
    reasons: list[str] = []

    if signal_direction == "NO_TRADE":
        reasons.append("NO_TRADE → execution_priority=5 (watchlist only)")
        return 5, reasons

    urgency_tier  = _URGENCY_TIER.get(urgency, 1)
    impact_bucket = 2 if impact_score >= 0.70 else (1 if impact_score >= 0.40 else 0)
    priority      = _PRIORITY_MATRIX.get((urgency_tier, impact_bucket), 5)

    reasons.append(
        f"Priority matrix: urgency_tier={urgency_tier}, "
        f"impact_bucket={impact_bucket} → execution_priority={priority}"
    )
    _log(2, f"Execution priority → {priority}")
    return priority, reasons


# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗      ██████╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ╚════██╗
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝        ██╔╝
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗       ██╔╝
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║       ██╔╝
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝       ╚═╝
#
#  LAYER 2 — ORCHESTRATOR
#  Calls all sub-layers in dependency order and assembles final output dict.
#
# ══════════════════════════════════════════════════════════════════════════════


def _generate_single_signal(article: dict) -> dict | None:
    """
    Orchestrate all signal-generation layers for a single validated article.

    Execution order
    ---------------
    1. Extract and normalise text fields
    2. Classify event type           (layer 3)
    3. Compute signal direction       (layer 4)
    4. Compute urgency               (layer 5, sub-step)
    5. Compute impact score          (layer 5)
    6. Infer market regime           (layer 7)
    7. Fuse confidence score         (layer 6)
    8. Compute position size bias    (layer 8)
    9. Compute execution priority    (layer 9)
    10. Assemble output dict

    Safety
    ------
    • Returns None if the article dict is malformed (no title/summary).
    • Never mutates the input dict — always shallow-copies first.

    Parameters
    ----------
    article : validator-enriched article dict (from fake_news_validator.py)

    Returns
    -------
    dict | None
        Enriched article dict with all signal fields injected.
        Returns None on fatal parse error (skipped by caller).
    """
    # ── 0. Safety extract ─────────────────────────────────────────────────────
    title   = _safe_str(article.get("title",   ""))
    summary = _safe_str(article.get("summary", article.get("description", "")))

    if not title and not summary:
        _log(1, "Skipping malformed article — no title or summary found")
        return None

    # Concatenate to a single searchable text blob (lowercased for phrase matching)
    text = (title + " " + summary).lower()

    # Accumulate all signal reasons for explainability
    all_reasons: list[str] = []

    # ── LAYER 3 — Event Classification ───────────────────────────────────────
    event_type, evt_reasons = _classify_event_type(text)
    all_reasons.extend(evt_reasons)

    # ── LAYER 4 — Directional Signal ─────────────────────────────────────────
    signal_direction, signal_strength, dir_reasons = _compute_signal_direction(text)
    all_reasons.extend(dir_reasons)

    # ── LAYER 5 — Urgency + Impact ────────────────────────────────────────────
    urgency, urgency_mult, urg_reasons = _compute_urgency(text)
    all_reasons.extend(urg_reasons)

    validation_score = _clamp(_safe_float(article.get("validation_score", 0.5)))
    impact_score, imp_reasons = _compute_impact_score(
        article, event_type, urgency_mult, signal_strength
    )
    all_reasons.extend(imp_reasons)

    # ── LAYER 7 — Market Regime ───────────────────────────────────────────────
    # Must precede confidence fusion — alignment feeds L6.
    market_regime_bias, regime_alignment, reg_reasons = _infer_market_regime(
        text, signal_direction
    )
    all_reasons.extend(reg_reasons)

    # ── LAYER 6 — Confidence Fusion ───────────────────────────────────────────
    confidence_score, conf_reasons = _compute_confidence_score(
        validation_score, impact_score, regime_alignment
    )
    all_reasons.extend(conf_reasons)

    # ── LAYER 8 — Position Size Bias ──────────────────────────────────────────
    position_size_bias, pos_reasons = _compute_position_size_bias(
        confidence_score, urgency, signal_direction
    )
    all_reasons.extend(pos_reasons)

    # ── LAYER 9 — Execution Priority ──────────────────────────────────────────
    execution_priority, pri_reasons = _compute_execution_priority(
        urgency, impact_score, signal_direction
    )
    all_reasons.extend(pri_reasons)

    # ── 10. Assemble output ───────────────────────────────────────────────────
    # Shallow copy — never mutate caller's dict
    output = dict(article)
    
    # Map signal_direction → signal_type for alert_router compatibility
    signal_type = signal_direction  # BUY/SELL/NO_TRADE
    
    # Map verification_status → verdict for alert_router compatibility
    verdict = article.get("verification_status", "UNVERIFIED")
    
    output.update({
        "signal_direction":     signal_direction,
        "signal_type":          signal_type,            # ← alert_router compat
        "signal_strength":      round(signal_strength, 4),
        "impact_score":         round(impact_score, 4),
        "confidence_score":     round(confidence_score, 4),
        "event_type":           event_type,
        "urgency":              urgency,
        "execution_priority":   execution_priority,
        "market_regime":        market_regime_bias,
        "position_size_bias":   round(position_size_bias, 4),
        "signal_reasons":       all_reasons,
        "verdict":              verdict,                # ← alert_router compat
    })

    return output


# ══════════════════════════════════════════════════════════════════════════════
#
#  ██╗      █████╗ ██╗   ██╗███████╗██████╗      ██╗
#  ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ███║
#  ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝    ╚██║
#  ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗     ██║
#  ███████╗██║  ██║   ██║   ███████╗██║  ██║     ██║
#  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝     ╚═╝
#
#  LAYER 1 — PUBLIC API
#
# ══════════════════════════════════════════════════════════════════════════════


def generate_signals(validated_articles: list[dict]) -> list[dict]:
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        PUBLIC API — ENTRY POINT                         │
    └─────────────────────────────────────────────────────────────────────────┘

    Transform a batch of validator-enriched news articles into tradeable
    directional signals with full explainability metadata.

    This function is the sole public surface of signal_engine.py.
    Downstream consumers (alert_router.py, execution_bridge.py) should call
    ONLY this function and never reference internal layer functions directly.

    Guarantees
    ----------
    • Order preservation — output list maintains input list ordering
    • Safe skipping      — malformed entries are logged and silently dropped
    • No mutation        — input dicts are never modified
    • Deterministic      — same input always produces same output

    Upstream contract (fields expected from fake_news_validator.py)
    --------------------------------------------------------------
    validation_score       : float
    trust_score            : float
    misinformation_risk    : float
    verification_status    : str
    is_high_confidence_news: bool
    needs_manual_review    : bool
    risk_reasons           : list[str]

    Injected output fields
    ----------------------
    signal_direction     : BUY | SELL | NO_TRADE
    signal_strength      : float [0, 1]
    impact_score         : float [0, 1]
    confidence_score     : float [0, 1]
    event_type           : MACRO | EARNINGS | CRYPTO | M&A | REGULATORY | UNKNOWN
    urgency              : LOW | MEDIUM | HIGH | CRITICAL
    execution_priority   : int 1–5
    market_regime        : RISK_ON | RISK_OFF | NEUTRAL
    position_size_bias   : float [0.25, 1.5]
    signal_reasons       : list[str]

    Parameters
    ----------
    validated_articles : list of dicts produced by fake_news_validator.py

    Returns
    -------
    list[dict]
        Enriched signal dicts — one per successfully processed article.
        Articles that fail safety validation are omitted.

    Performance
    -----------
    Designed for ≥ 1,000 articles in < 100ms.
    All regex patterns are module-level compiled constants.

    Example
    -------
    >>> signals = generate_signals(validated_articles)
    >>> for s in signals:
    ...     print(s["signal_direction"], s["confidence_score"])
    """
    if not isinstance(validated_articles, list):
        _log(1, f"generate_signals received non-list input ({type(validated_articles)}) — returning []")
        return []

    _log(1, f"generate_signals — processing {len(validated_articles)} article(s)")
    t_start = time.perf_counter()

    results: list[dict] = []
    skipped = 0

    for idx, article in enumerate(validated_articles):
        if not isinstance(article, dict):
            _log(2, f"Article[{idx}] is not a dict ({type(article)}) — skipping")
            skipped += 1
            continue

        try:
            signal = _generate_single_signal(article)
            if signal is None:
                skipped += 1
                continue
            results.append(signal)
        except Exception as exc:  # noqa: BLE001
            # Never let a single bad article crash the whole batch
            _log(1, f"Article[{idx}] raised an unhandled exception ({exc!r}) — skipping")
            skipped += 1

    elapsed_ms = (time.perf_counter() - t_start) * 1_000
    _log(
        1,
        f"generate_signals complete — "
        f"processed={len(results)}, skipped={skipped}, "
        f"elapsed={elapsed_ms:.2f}ms",
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST — 8 SYNTHETIC ARTICLES
#  Run directly: python signal_engine.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _logger.setLevel(logging.DEBUG)

    print("=" * 80)
    print("  SIGNAL ENGINE — SMOKE TEST")
    print("  8 Synthetic Articles | Institutional Validation Suite")
    print("=" * 80)
    print()

    _SYNTHETIC_ARTICLES: list[dict] = [
        # ── 1. High-confidence bullish MACRO ─────────────────────────────────
        {
            "id":                    "art-001",
            "title":                 "Federal Reserve announces surprise rate cut of 50bps — easing inflation",
            "summary":               "The FOMC voted unanimously for a dovish pivot, cutting the federal funds rate. "
                                     "Markets surged in response as recession fears eased and stimulus expectations rose.",
            "priority":              "high",
            "cluster_size":          9,
            "validation_score":      0.95,
            "trust_score":           0.92,
            "misinformation_risk":   0.04,
            "verification_status":   "VERIFIED",
            "is_high_confidence_news": True,
            "needs_manual_review":   False,
            "risk_reasons":          [],
        },
        # ── 2. Strong bearish EARNINGS ────────────────────────────────────────
        {
            "id":                    "art-002",
            "title":                 "MegaCorp Q3 earnings miss — lowers guidance, layoffs announced",
            "summary":               "MegaCorp missed estimates by 18%, lowered guidance for the full year, "
                                     "and announced 4,000 layoffs amid margin compression and revenue decline.",
            "priority":              "high",
            "cluster_size":          7,
            "validation_score":      0.88,
            "trust_score":           0.85,
            "misinformation_risk":   0.06,
            "verification_status":   "VERIFIED",
            "is_high_confidence_news": True,
            "needs_manual_review":   False,
            "risk_reasons":          [],
        },
        # ── 3. CRYPTO bullish — ETF approval ─────────────────────────────────
        {
            "id":                    "art-003",
            "title":                 "SEC approves spot Bitcoin ETF — crypto markets surge",
            "summary":               "The Securities and Exchange Commission has approved the first spot Bitcoin ETF, "
                                     "clearing the path for institutional investment. BTC soared on the news.",
            "priority":              "high",
            "cluster_size":          12,
            "validation_score":      0.91,
            "trust_score":           0.89,
            "misinformation_risk":   0.05,
            "verification_status":   "VERIFIED",
            "is_high_confidence_news": True,
            "needs_manual_review":   False,
            "risk_reasons":          [],
        },
        # ── 4. REGULATORY bearish — fraud investigation ───────────────────────
        {
            "id":                    "art-004",
            "title":                 "SEC launches fraud investigation into TechFirm Inc — stock halted",
            "summary":               "Regulators issued a subpoena to TechFirm Inc amid allegations of accounting fraud. "
                                     "The SEC investigation follows an enforcement action last quarter. Stock halted.",
            "priority":              "high",
            "cluster_size":          6,
            "validation_score":      0.87,
            "trust_score":           0.84,
            "misinformation_risk":   0.08,
            "verification_status":   "VERIFIED",
            "is_high_confidence_news": True,
            "needs_manual_review":   False,
            "risk_reasons":          [],
        },
        # ── 5. M&A bullish ────────────────────────────────────────────────────
        {
            "id":                    "art-005",
            "title":                 "Acquisition of GlobalSoft completed — merger approved by regulators",
            "summary":               "The $42B acquisition of GlobalSoft has been cleared by antitrust regulators. "
                                     "Deal closed after buyback program announced to boost shareholder value.",
            "priority":              "medium",
            "cluster_size":          5,
            "validation_score":      0.82,
            "trust_score":           0.80,
            "misinformation_risk":   0.10,
            "verification_status":   "VERIFIED",
            "is_high_confidence_news": True,
            "needs_manual_review":   False,
            "risk_reasons":          [],
        },
        # ── 6. MACRO bearish — recession + rate hike ─────────────────────────
        {
            "id":                    "art-006",
            "title":                 "Breaking: Recession fears surge as Fed signals aggressive rate hikes",
            "summary":               "The Federal Reserve issued a hawkish statement signalling quantitative tightening "
                                     "amid a credit crunch. Yield inversion deepened as recession fears hit a decade high.",
            "priority":              "high",
            "cluster_size":          10,
            "validation_score":      0.93,
            "trust_score":           0.91,
            "misinformation_risk":   0.03,
            "verification_status":   "VERIFIED",
            "is_high_confidence_news": True,
            "needs_manual_review":   False,
            "risk_reasons":          [],
        },
        # ── 7. Mixed signals → NO_TRADE expected ─────────────────────────────
        {
            "id":                    "art-007",
            "title":                 "Earnings beat overshadowed by SEC probe and layoff announcement",
            "summary":               "RetailChain beat estimates for Q2 and raised guidance. However, an SEC investigation "
                                     "into its CFO and a restructuring with 2,000 layoffs added uncertainty to the outlook.",
            "priority":              "medium",
            "cluster_size":          4,
            "validation_score":      0.70,
            "trust_score":           0.68,
            "misinformation_risk":   0.15,
            "verification_status":   "PARTIAL",
            "is_high_confidence_news": False,
            "needs_manual_review":   True,
            "risk_reasons":          ["conflicting signals", "partial verification only"],
        },
        # ── 8. Malformed article — should be skipped ──────────────────────────
        {
            "id":                    "art-008",
            "title":                 "",
            "summary":               "",
            "priority":              "low",
            "cluster_size":          1,
            "validation_score":      0.50,
            "trust_score":           0.50,
            "misinformation_risk":   0.50,
            "verification_status":   "UNVERIFIED",
            "is_high_confidence_news": False,
            "needs_manual_review":   True,
            "risk_reasons":          ["empty content"],
        },
    ]

    signals = generate_signals(_SYNTHETIC_ARTICLES)

    # ── Print results table ───────────────────────────────────────────────────
    print()
    print("─" * 80)
    print(f"  {'ID':<10} {'DIRECTION':<10} {'STRENGTH':>8} {'IMPACT':>8} {'CONF':>7} "
          f"{'URGENCY':<10} {'REGIME':<12} {'SZ BIAS':>8} {'PRI':>4} {'EVENT':<12}")
    print("─" * 80)

    for sig in signals:
        print(
            f"  {sig.get('id','?'):<10} "
            f"{sig['signal_direction']:<10} "
            f"{sig['signal_strength']:>8.4f} "
            f"{sig['impact_score']:>8.4f} "
            f"{sig['confidence_score']:>7.4f} "
            f"{sig['urgency']:<10} "
            f"{sig['market_regime']:<12} "
            f"{sig['position_size_bias']:>8.4f} "
            f"{sig['execution_priority']:>4} "
            f"{sig['event_type']:<12}"
        )

    print("─" * 80)
    print(f"\n  Signals generated : {len(signals)}")
    print(f"  Input articles    : {len(_SYNTHETIC_ARTICLES)}")
    print(f"  Skipped (malformed): {len(_SYNTHETIC_ARTICLES) - len(signals)}")
    print()

    # ── Verbose reason dump for first signal ─────────────────────────────────
    if signals:
        print("─" * 80)
        print(f"  SIGNAL REASONS — {signals[0].get('id', 'art-001')}")
        print("─" * 80)
        for i, reason in enumerate(signals[0]["signal_reasons"], 1):
            print(f"  [{i:02d}] {reason}")
        print()

    print("  ✓ Smoke test complete.")
    print("=" * 80)