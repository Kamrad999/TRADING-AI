"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║  ██████╗ ███████╗██████╗ ███████╗ ██████╗ ██████╗ ███╗   ███╗ █████╗ ███╗   ██╗ ██████╗ ║
║  ██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔══██╗████╗  ██║██╔════╝ ║
║  ██████╔╝█████╗  ██████╔╝█████╗  ██║   ██║██████╔╝██╔████╔██║███████║██╔██╗ ██║██║      ║
║  ██╔═══╝ ██╔══╝  ██╔══██╗██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║██╔══██║██║╚██╗██║██║      ║
║  ██║     ███████╗██║  ██║██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║██║  ██║██║ ╚████║╚██████╗ ║
║  ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ║
║                                                                                           ║
║   █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗████████╗██╗ ██████╗███████╗                   ║
║  ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══██╔══╝██║██╔════╝██╔════╝                   ║
║  ███████║██╔██╗ ██║███████║██║   ╚████╔╝    ██║   ██║██║     ███████╗                   ║
║  ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝     ██║   ██║██║     ╚════██║                   ║
║  ██║  ██║██║ ╚████║██║  ██║███████╗██║      ██║   ██║╚██████╗███████║                   ║
║  ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝      ╚═╝   ╚═╝ ╚═════╝╚══════╝                   ║
║                                                                                           ║
║  ┌───────────────────────────────────────────────────────────────────────────────────┐   ║
║  │         📊  P E R F O R M A N C E   A N A L Y T I C S  📊                        │   ║
║  │      Pipeline Terminal Intelligence — Alpha Discovery Infrastructure              │   ║
║  │   broker_sender → [YOU]                                                           │   ║
║  │   "Which signals, sources, regimes, and assets actually make money?"              │   ║
║  └───────────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                           ║
║  Module   : performance_analytics.py                                                      ║
║  Version  : 1.0.0                                                                         ║
║  Mission  : Transform broker fills into a self-improving quant feedback engine            ║
║  Layers   : 8 Analytics + Fill Normalization + Recommendation Engine                     ║
║  Complexity: O(n) over fills · Standard library only · No pandas · No numpy              ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import statistics
import sys
import textwrap
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0 ▸ CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ANALYTICS_VERSION = "1.0.0"
ANALYTICS_BUILD   = "MONSTER-TRADING-AI"

# ── Sharpe-like score ─────────────────────────────────────────────────────────
RISK_FREE_RATE_ANNUAL  = 0.053    # approx T-bill rate (annualised)
TRADING_DAYS_PER_YEAR  = 252

# ── Confidence buckets ────────────────────────────────────────────────────────
CONFIDENCE_BUCKETS: list[tuple[float, float, str]] = [
    (0.90, 1.01, "0.90–1.00 (elite)"),
    (0.85, 0.90, "0.85–0.90 (high)"),
    (0.80, 0.85, "0.80–0.85 (standard)"),
    (0.00, 0.80, "< 0.80 (subthreshold)"),
]

# ── Alpha source tiers (for ranking) ─────────────────────────────────────────
SOURCE_TIER: dict[str, int] = {
    "reuters":      1,
    "bloomberg":    1,
    "fed":          1,
    "wsj":          1,
    "ft":           1,
    "cnbc":         2,
    "barrons":      2,
    "marketwatch":  2,
    "seekingalpha": 3,
    "twitter":      4,
    "reddit":       4,
    "unknown":      5,
}

# ── Asset class keyword map for normalization ─────────────────────────────────
_CRYPTO_TICKERS = frozenset({
    "BTC","ETH","SOL","BNB","XRP","ADA","DOGE","AVAX","DOT","MATIC",
    "LINK","LTC","BCH","ATOM","UNI","COIN","MSTR","GBTC",
    "BTCUSD","ETHUSD","SOLUSD",
})
_MACRO_TICKERS = frozenset({
    "SPY","QQQ","IVV","VTI","DIA","GLD","SLV","TLT","VIX",
    "ES","NQ","YM","RTY","CL","GC","SI","SPX","NDX","RUT",
    "XAU","USO","BND","AGG","IEF","SHY","TLH",
})
_FOREX_TICKERS = frozenset({
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD",
    "USDCHF","NZDUSD","EURGBP","EURJPY","GBPJPY",
})

# ── Regime keywords ───────────────────────────────────────────────────────────
_REGIME_KEYWORDS: dict[str, list[str]] = {
    "TRENDING":    ["trending","momentum","breakout","rally","rip"],
    "CHOPPY":      ["choppy","range","sideways","consolidat","noise"],
    "PANIC":       ["crash","panic","flash","circuit","halt","collapse"],
    "POST_FOMC":   ["fomc","fed","rate decision","powell","basis point"],
    "CRYPTO_SQUEEZE": ["squeeze","liquidat","crypto","btc","eth","leverage","short"],
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 ▸ FILL NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _safe_str(val: Any, default: str = "") -> str:
    return str(val).strip() if val is not None else default


def _parse_ts(val: Any) -> datetime | None:
    """Parse ISO-8601 timestamp → aware UTC datetime, or None."""
    if not val:
        return None
    s = str(val).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _classify_asset(ticker: str) -> str:
    t = ticker.upper().replace("USDT", "").replace("/", "").replace("-", "")
    if t in _CRYPTO_TICKERS or t.endswith("BTC") or t.endswith("ETH"):
        return "CRYPTO"
    if t in _MACRO_TICKERS:
        return "MACRO"
    if t in _FOREX_TICKERS or (len(t) == 6 and t.isalpha()):
        return "FOREX"
    return "EQUITY"


def _classify_regime(text: str) -> str:
    low = text.lower()
    for regime, keywords in _REGIME_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return regime
    return "NORMAL"


def _classify_news_category(text: str) -> str:
    low = text.lower()
    cats = {
        "EARNINGS":        ["earn","eps","revenue","guidance","beat","miss","profit"],
        "MACRO_POLICY":    ["fed","fomc","rate","inflation","cpi","nfp","gdp","powell"],
        "GEOPOLITICAL":    ["war","sanction","tariff","geopolit","conflict","military"],
        "CRYPTO_NEWS":     ["bitcoin","crypto","btc","eth","defi","nft","blockchain","web3"],
        "BANKING_CRISIS":  ["bank","fdic","collapse","bailout","deposit","svb","credit"],
        "ENERGY_SHOCK":    ["oil","opec","crude","natural gas","energy","pipeline"],
        "CORPORATE_ACTION":["merger","acquisition","buyback","ipo","spinoff","dividend"],
        "CYBER_EVENT":     ["hack","breach","ransomware","cybersecurity","attack"],
    }
    for cat, kws in cats.items():
        if any(k in low for k in kws):
            return cat
    return "GENERAL"


def normalize_fill(raw: dict) -> dict | None:
    """
    Convert any broker_sender output dict into a canonical fill record.
    Returns None if the fill is not tradeable (status not a fill event).
    """
    if not isinstance(raw, dict):
        return None

    status = _safe_str(raw.get("broker_status")).upper()
    if status not in ("SENT", "PAPER_FILLED", "PARTIAL_FILL"):
        return None

    # ── Ticker resolution ─────────────────────────────────────────────────────
    ticker = _safe_str(
        raw.get("_ticker")
        or (raw.get("broker_payload") or {}).get("symbol")
        or "UNKNOWN"
    ).upper()

    # ── Side ──────────────────────────────────────────────────────────────────
    bp   = raw.get("broker_payload") or {}
    side = _safe_str(
        bp.get("side") or raw.get("_signal_type") or "BUY"
    ).upper()[:4]

    # ── Prices & size ─────────────────────────────────────────────────────────
    fill_price    = _safe_float(raw.get("fill_simulated_price"))
    expected_price = _safe_float(
        (raw.get("broker_payload") or {}).get("_expected_price") or fill_price
    )
    notional = _safe_float(
        bp.get("notional") or bp.get("cashQty") or bp.get("quoteOrderQty")
        or (_safe_float(raw.get("adjusted_position_size"), 0.05) * 100_000)
    )
    position_size = _safe_float(raw.get("adjusted_position_size"), 0.05)

    # ── Metadata ──────────────────────────────────────────────────────────────
    fill_ts   = _parse_ts(raw.get("sent_at") or raw.get("_evaluated_at"))
    signal_ts = _parse_ts(raw.get("signal_timestamp"))
    close_ts  = _parse_ts(raw.get("close_timestamp"))

    confidence = _safe_float(
        raw.get("confidence_score")
        or (raw.get("formatted_alert") or {}).get("raw_signal", {}).get("confidence_score")
        or 0.85
    )

    headline = _safe_str(
        (raw.get("formatted_alert") or {}).get("raw_signal", {}).get("title")
        or raw.get("headline")
        or ticker
    )
    source = _safe_str(
        (raw.get("formatted_alert") or {}).get("raw_signal", {}).get("source")
        or raw.get("source")
        or "unknown"
    ).lower()

    # Strip domain prefixes for cleaner grouping
    for prefix in ("www.", "https://", "http://"):
        source = source.replace(prefix, "")
    source = source.split(".")[0] if "." in source else source

    # ── Classifications ───────────────────────────────────────────────────────
    asset_class   = _classify_asset(ticker)
    news_category = _classify_news_category(headline)
    regime        = _classify_regime(
        _safe_str(raw.get("router_reason"))
        + " "
        + _safe_str(raw.get("execution_reason"))
        + " "
        + headline
    )

    # ── Slippage ──────────────────────────────────────────────────────────────
    if expected_price > 0 and fill_price > 0:
        direction   = 1 if side == "BUY" else -1
        slippage_bps = direction * (fill_price - expected_price) / expected_price * 10_000
    else:
        slippage_bps = _safe_float(
            (raw.get("broker_payload") or {}).get("slippage_bps"), 0.0
        )

    # ── Realized PnL (if close_price provided) ────────────────────────────────
    close_price = _safe_float(raw.get("close_price"), 0.0)
    realized_pnl = 0.0
    if close_price > 0 and fill_price > 0 and notional > 0:
        shares = notional / fill_price if fill_price > 0 else 0
        if side == "BUY":
            realized_pnl = (close_price - fill_price) * shares
        else:
            realized_pnl = (fill_price - close_price) * shares

    hold_minutes = 0.0
    if fill_ts and close_ts:
        hold_minutes = max(0.0, (close_ts - fill_ts).total_seconds() / 60.0)

    order_id = _safe_str(raw.get("_order_id") or raw.get("broker_order_id") or str(uuid.uuid4()))

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "order_id":       order_id,
        "ticker":         ticker,
        "side":           side,
        "status":         status,
        # ── Price & size ──────────────────────────────────────────────────────
        "fill_price":     fill_price,
        "expected_price": expected_price,
        "close_price":    close_price,
        "notional":       notional,
        "position_size":  position_size,
        # ── PnL ───────────────────────────────────────────────────────────────
        "realized_pnl":   realized_pnl,
        "is_closed":      close_price > 0,
        # ── Risk & slippage ───────────────────────────────────────────────────
        "slippage_bps":   slippage_bps,
        "retry_count":    int(raw.get("retry_count", 0)),
        "route_latency_ms": _safe_float(raw.get("route_latency_ms")),
        # ── Time ──────────────────────────────────────────────────────────────
        "fill_ts":        fill_ts,
        "signal_ts":      signal_ts,
        "close_ts":       close_ts,
        "hold_minutes":   hold_minutes,
        # ── Intelligence ──────────────────────────────────────────────────────
        "confidence":     confidence,
        "source":         source,
        "headline":       headline,
        "asset_class":    asset_class,
        "news_category":  news_category,
        "regime":         regime,
        "risk_score":     _safe_float(raw.get("risk_score"), 0.0),
        "signal_strength": _safe_str(raw.get("alert_priority") or raw.get("_signal_type"), "UNKNOWN"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 ▸ PnL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _bucket_confidence(conf: float) -> str:
    for lo, hi, label in CONFIDENCE_BUCKETS:
        if lo <= conf < hi:
            return label
    return "< 0.80 (subthreshold)"


def _compute_pnl_series(fills: list[dict]) -> list[float]:
    """Return ordered list of realized PnL values for closed trades."""
    return [f["realized_pnl"] for f in fills if f["is_closed"]]


def _running_peak_trough(pnl_series: list[float]) -> float:
    """Peak-to-trough max drawdown (dollar terms) over PnL equity curve."""
    if not pnl_series:
        return 0.0
    equity   = 0.0
    peak     = 0.0
    max_dd   = 0.0
    for pnl in pnl_series:
        equity += pnl
        peak    = max(peak, equity)
        dd      = peak - equity
        max_dd  = max(max_dd, dd)
    return round(max_dd, 4)


def _streaks(pnl_series: list[float]) -> dict:
    """Compute best winning streak, worst losing streak, current streak."""
    if not pnl_series:
        return {"best_win_streak": 0, "worst_loss_streak": 0,
                "current_streak": 0, "current_direction": "NONE"}

    best_win  = 0
    worst_loss = 0
    cur_win   = 0
    cur_loss  = 0

    for p in pnl_series:
        if p > 0:
            cur_win  += 1
            cur_loss  = 0
            best_win  = max(best_win, cur_win)
        elif p < 0:
            cur_loss += 1
            cur_win   = 0
            worst_loss = max(worst_loss, cur_loss)
        else:
            cur_win = cur_loss = 0

    last     = pnl_series[-1]
    cur_dir  = "WIN" if last > 0 else ("LOSS" if last < 0 else "FLAT")
    cur_val  = cur_win if last > 0 else -cur_loss

    return {
        "best_win_streak":   best_win,
        "worst_loss_streak": worst_loss,
        "current_streak":    cur_val,
        "current_direction": cur_dir,
    }


def _sharpe_like(pnl_series: list[float], hold_minutes_series: list[float]) -> float:
    """
    Annualised Sharpe-like score using per-trade PnL as returns.
    Scaled by average hold time to annualise.
    Returns 0 if fewer than 2 closed trades.
    """
    if len(pnl_series) < 2:
        return 0.0
    try:
        mean_pnl  = statistics.mean(pnl_series)
        stdev_pnl = statistics.stdev(pnl_series)
        if stdev_pnl == 0:
            return 0.0
        # Average hold in minutes → trades per year approximation
        avg_hold  = statistics.mean(hold_minutes_series) if hold_minutes_series else 60.0
        avg_hold  = max(avg_hold, 1.0)
        trades_per_year = (TRADING_DAYS_PER_YEAR * 390) / avg_hold   # 390 min/day
        rf_per_trade    = RISK_FREE_RATE_ANNUAL / max(trades_per_year, 1)
        sharpe = (mean_pnl - rf_per_trade) / stdev_pnl * math.sqrt(trades_per_year)
        return round(sharpe, 4)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 ▸ ALPHA ATTRIBUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _group_by(fills: list[dict], key: str) -> dict[str, list[dict]]:
    """O(n) groupBy — returns dict of lists."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in fills:
        groups[f.get(key, "UNKNOWN")].append(f)
    return dict(groups)


def _pnl_stats(fills: list[dict]) -> dict:
    """Compute PnL statistics for a group of fills."""
    closed      = [f for f in fills if f["is_closed"]]
    all_pnl     = [f["realized_pnl"] for f in closed]
    winners     = [p for p in all_pnl if p > 0]
    losers      = [p for p in all_pnl if p < 0]
    n           = len(all_pnl)

    total       = round(sum(all_pnl), 4)
    win_rate    = round(len(winners) / n, 4) if n > 0 else 0.0
    avg         = round(total / n, 4) if n > 0 else 0.0
    avg_win     = round(statistics.mean(winners), 4) if winners else 0.0
    avg_loss    = round(statistics.mean(losers), 4) if losers else 0.0
    expectancy  = round(
        win_rate * avg_win + (1 - win_rate) * avg_loss, 4
    ) if n > 0 else 0.0
    profit_factor = (
        round(sum(winners) / abs(sum(losers)), 4)
        if losers and sum(winners) > 0
        else (float("inf") if winners else 0.0)
    )

    return {
        "total_pnl":      total,
        "trade_count":    len(fills),
        "closed_trades":  n,
        "win_rate":       win_rate,
        "avg_pnl":        avg,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "expectancy":     expectancy,
        "profit_factor":  profit_factor,
    }


def _source_alpha_rankings(fills: list[dict]) -> list[dict]:
    """
    Rank news sources by expectancy × trade count — the two-dimensional alpha score.
    Higher trade count means statistical reliability; higher expectancy means edge.
    """
    groups = _group_by(fills, "source")
    ranked = []
    for source, group in groups.items():
        stats = _pnl_stats(group)
        tier  = SOURCE_TIER.get(source, 5)
        # alpha_score = expectancy × sqrt(n) — rewards both edge AND volume
        n     = stats["closed_trades"]
        alpha = round(stats["expectancy"] * math.sqrt(max(n, 1)), 4)
        ranked.append({
            "source":        source,
            "tier":          tier,
            "alpha_score":   alpha,
            **stats,
        })
    ranked.sort(key=lambda x: x["alpha_score"], reverse=True)
    return ranked


def _false_signal_clusters(fills: list[dict]) -> list[dict]:
    """
    Identify clusters of consecutive losing trades that share a common attribute.
    Returns grouped loss clusters by (source, news_category, regime).
    """
    losers = [f for f in fills if f["is_closed"] and f["realized_pnl"] < 0]
    if not losers:
        return []

    cluster_key = lambda f: (f["source"], f["news_category"], f["regime"])
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for f in losers:
        groups[cluster_key(f)].append(f)

    clusters = []
    for (src, cat, regime), group in groups.items():
        if len(group) < 2:
            continue
        total_loss = sum(f["realized_pnl"] for f in group)
        avg_conf   = statistics.mean(f["confidence"] for f in group)
        clusters.append({
            "source":      src,
            "category":    cat,
            "regime":      regime,
            "loss_count":  len(group),
            "total_loss":  round(total_loss, 4),
            "avg_confidence": round(avg_conf, 4),
            "signal":      "FALSE_SIGNAL_CLUSTER",
        })
    clusters.sort(key=lambda x: x["total_loss"])
    return clusters


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 ▸ CONFIDENCE CALIBRATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _confidence_calibration(fills: list[dict]) -> dict[str, dict]:
    """
    For each confidence bucket, compute realized win_rate and avg_pnl.
    A well-calibrated model has monotonically increasing win_rate with confidence.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for f in fills:
        label = _bucket_confidence(f["confidence"])
        buckets[label].append(f)

    result = {}
    for lo, hi, label in CONFIDENCE_BUCKETS:
        group = buckets.get(label, [])
        result[label] = _pnl_stats(group)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 ▸ REGIME ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _regime_analysis(fills: list[dict]) -> dict[str, dict]:
    groups = _group_by(fills, "regime")
    return {regime: _pnl_stats(group) for regime, group in groups.items()}


def _best_worst_regime(regime_stats: dict[str, dict]) -> tuple[str, str]:
    if not regime_stats:
        return "UNKNOWN", "UNKNOWN"
    scored = {r: s["expectancy"] for r, s in regime_stats.items() if s["closed_trades"] > 0}
    if not scored:
        return "UNKNOWN", "UNKNOWN"
    best  = max(scored, key=scored.get)
    worst = min(scored, key=scored.get)
    return best, worst


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 ▸ SLIPPAGE FORENSICS
# ══════════════════════════════════════════════════════════════════════════════

def _slippage_forensics(fills: list[dict]) -> dict:
    """
    Measure total slippage drag across all fills.
    Computes expected_vs_actual cost and its impact on PnL.
    """
    bps_vals  = [f["slippage_bps"] for f in fills if f["fill_price"] > 0]
    if not bps_vals:
        return {
            "avg_slippage_bps":    0.0,
            "max_slippage_bps":    0.0,
            "total_slippage_cost": 0.0,
            "worst_ticker":        "N/A",
            "slippage_by_asset":   {},
        }

    # Total dollar slippage cost
    total_cost = sum(
        abs(f["slippage_bps"]) / 10_000 * f["notional"]
        for f in fills if f["fill_price"] > 0
    )

    # Worst single ticker by avg slippage bps
    by_ticker: dict[str, list[float]] = defaultdict(list)
    for f in fills:
        if f["fill_price"] > 0:
            by_ticker[f["ticker"]].append(abs(f["slippage_bps"]))
    worst_ticker = max(
        by_ticker,
        key=lambda t: statistics.mean(by_ticker[t]) if by_ticker[t] else 0,
        default="N/A",
    )

    # By asset class
    by_asset: dict[str, list[float]] = defaultdict(list)
    for f in fills:
        if f["fill_price"] > 0:
            by_asset[f["asset_class"]].append(abs(f["slippage_bps"]))
    slippage_by_asset = {
        k: round(statistics.mean(v), 2)
        for k, v in by_asset.items() if v
    }

    return {
        "avg_slippage_bps":    round(statistics.mean(bps_vals), 4),
        "max_slippage_bps":    round(max(bps_vals), 4),
        "total_slippage_cost": round(total_cost, 4),
        "worst_ticker":        worst_ticker,
        "slippage_by_asset":   slippage_by_asset,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 ▸ DRAWDOWN + STREAK ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# (Helpers already defined in LAYER 2 — _running_peak_trough, _streaks)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8 ▸ RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _generate_recommendations(
    pnl_by_asset:       dict[str, dict],
    pnl_by_source:      dict[str, dict],
    conf_calibration:   dict[str, dict],
    regime_stats:       dict[str, dict],
    worst_regime:       str,
    slippage:           dict,
    false_clusters:     list[dict],
    source_rankings:    list[dict],
    fills:              list[dict],
) -> list[str]:
    """
    O(k) recommendation engine — each heuristic fires independently.
    Returns ranked list of actionable strategy advice strings.
    """
    recs: list[tuple[int, str]] = []   # (priority, text)

    # ── Asset class recs ──────────────────────────────────────────────────────
    for asset, stats in pnl_by_asset.items():
        if stats["closed_trades"] < 2:
            continue
        if stats["win_rate"] < 0.40:
            recs.append((1, f"🔴 REDUCE {asset} — win rate {stats['win_rate']:.0%} below 40%. "
                           "Consider halving position size or pausing allocation."))
        elif stats["expectancy"] < -50:
            recs.append((1, f"🔴 REVIEW {asset} signal filter — avg expectancy "
                           f"${stats['expectancy']:.0f}/trade is deeply negative."))
        elif stats["win_rate"] > 0.65 and stats["expectancy"] > 100:
            recs.append((3, f"⭐ INCREASE {asset} allocation — "
                           f"win rate {stats['win_rate']:.0%}, "
                           f"expectancy ${stats['expectancy']:.0f}/trade."))

    # ── Source alpha recs ─────────────────────────────────────────────────────
    for entry in source_rankings:
        src   = entry["source"]
        tier  = entry["tier"]
        ct    = entry["closed_trades"]
        if ct < 2:
            continue
        if entry["win_rate"] < 0.35 and tier >= 3:
            recs.append((2, f"🚫 IGNORE '{src}' signals — tier {tier} source with "
                           f"{entry['win_rate']:.0%} win rate. Likely noise."))
        elif entry["alpha_score"] > 0 and tier == 1:
            recs.append((3, f"📈 BOOST '{src}' (tier {tier}) — "
                           f"alpha score {entry['alpha_score']:.2f}, "
                           f"expectancy ${entry['expectancy']:.0f}/trade."))

    # ── Confidence calibration recs ───────────────────────────────────────────
    elite_stats    = conf_calibration.get("0.90–1.00 (elite)", {})
    standard_stats = conf_calibration.get("0.80–0.85 (standard)", {})
    if (elite_stats.get("win_rate", 0) < standard_stats.get("win_rate", 0)
            and elite_stats.get("closed_trades", 0) >= 3):
        recs.append((2, "⚠️  CONFIDENCE MISCALIBRATION — elite signals (0.90+) underperforming "
                        "standard (0.80–0.85). Re-examine signal_engine scoring model."))
    if elite_stats.get("win_rate", 0) > 0.70 and elite_stats.get("closed_trades", 0) >= 3:
        recs.append((3, "✅ HIGH CONFIDENCE signals well-calibrated — "
                        f"{elite_stats.get('win_rate', 0):.0%} win rate at 0.90+ bucket. "
                        "Maintain or increase size threshold."))

    # ── Regime recs ───────────────────────────────────────────────────────────
    worst_s = regime_stats.get(worst_regime, {})
    if worst_s.get("closed_trades", 0) >= 2 and worst_s.get("expectancy", 0) < -30:
        recs.append((1, f"🚨 AVOID {worst_regime} regime — "
                        f"expectancy ${worst_s.get('expectancy', 0):.0f}/trade. "
                        "Engage regime filter or reduce size to 25% during this market state."))
    elevated_stats = regime_stats.get("CHOPPY", {})
    if elevated_stats.get("closed_trades", 0) >= 2 and elevated_stats.get("win_rate", 0) < 0.40:
        recs.append((2, "⚠️  REDUCE size in CHOPPY regime — "
                        f"{elevated_stats.get('win_rate', 0):.0%} win rate. "
                        "Signals lose edge in sideways markets."))

    # ── Slippage recs ─────────────────────────────────────────────────────────
    if slippage["avg_slippage_bps"] > 15:
        recs.append((2, f"💸 HIGH SLIPPAGE — avg {slippage['avg_slippage_bps']:.1f} bps. "
                        f"Worst ticker: {slippage['worst_ticker']}. "
                        "Consider limit orders or smaller lot sizes."))
    if slippage["total_slippage_cost"] > 500:
        recs.append((2, f"💸 SLIPPAGE DRAG ${slippage['total_slippage_cost']:,.0f} — "
                        "above $500 total. Review execution timing and liquidity screening."))

    # ── False signal cluster recs ─────────────────────────────────────────────
    for cluster in false_clusters[:3]:
        recs.append((1, f"🔴 FALSE SIGNAL CLUSTER — {cluster['loss_count']} losses "
                        f"from '{cluster['source']}' / '{cluster['category']}' "
                        f"in {cluster['regime']} regime. "
                        f"Total loss: ${abs(cluster['total_loss']):,.0f}. "
                        "Add a filter gate for this combination."))

    # ── Latency recs ──────────────────────────────────────────────────────────
    latencies = [f["route_latency_ms"] for f in fills if f["route_latency_ms"] > 0]
    if latencies and statistics.mean(latencies) > 200:
        recs.append((2, f"⏱  HIGH ROUTE LATENCY — avg {statistics.mean(latencies):.0f}ms. "
                        "Co-locate broker adapter or upgrade network path."))

    # Deduplicate and sort by priority
    seen = set()
    unique = []
    for p, r in sorted(recs, key=lambda x: x[0]):
        if r not in seen:
            seen.add(r)
            unique.append(r)

    return unique if unique else ["✅ No critical issues detected — maintain current strategy parameters."]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def analyze_fills(fills: list[dict]) -> dict:
    """
    Transform raw broker_sender output into a comprehensive alpha discovery report.

    Parameters
    ----------
    fills : list[dict]
        Output from broker_sender.send_orders().  Non-fill entries are safely skipped.
        Closed trades (with 'close_price' set) contribute to PnL metrics.

    Returns
    -------
    dict
        Full analytics report with 19+ top-level keys covering PnL, attribution,
        calibration, slippage, drawdown, streaks, regime, and recommendations.
    """
    if not fills:
        return _empty_report()

    # ── Step 1: Normalize ─────────────────────────────────────────────────────
    normalized: list[dict] = []
    for raw in fills:
        n = normalize_fill(raw)
        if n:
            normalized.append(n)

    if not normalized:
        return _empty_report()

    # ── Step 2: Core PnL series ───────────────────────────────────────────────
    pnl_series      = _compute_pnl_series(normalized)
    closed_fills    = [f for f in normalized if f["is_closed"]]
    open_fills      = [f for f in normalized if not f["is_closed"]]
    all_pnl         = sum(pnl_series)
    closed_n        = len(pnl_series)
    winners_pnl     = [p for p in pnl_series if p > 0]
    losers_pnl      = [p for p in pnl_series if p < 0]
    win_rate        = round(len(winners_pnl) / closed_n, 4) if closed_n else 0.0
    avg_pnl         = round(all_pnl / closed_n, 4) if closed_n else 0.0
    avg_win         = round(statistics.mean(winners_pnl), 4) if winners_pnl else 0.0
    avg_loss        = round(statistics.mean(losers_pnl), 4) if losers_pnl else 0.0
    expectancy      = round(win_rate * avg_win + (1 - win_rate) * avg_loss, 4) if closed_n else 0.0

    hold_minutes    = [f["hold_minutes"] for f in closed_fills if f["hold_minutes"] > 0]
    avg_hold        = round(statistics.mean(hold_minutes), 2) if hold_minutes else 0.0
    latencies       = [f["route_latency_ms"] for f in normalized if f["route_latency_ms"] > 0]
    avg_latency     = round(statistics.mean(latencies), 2) if latencies else 0.0

    # ── Step 3: Attribution ───────────────────────────────────────────────────
    pnl_by_asset    = {k: _pnl_stats(v) for k, v in _group_by(normalized, "asset_class").items()}
    pnl_by_cat      = {k: _pnl_stats(v) for k, v in _group_by(normalized, "news_category").items()}
    pnl_by_source   = {k: _pnl_stats(v) for k, v in _group_by(normalized, "source").items()}
    pnl_by_conf     = _confidence_calibration(normalized)
    source_rankings = _source_alpha_rankings(normalized)
    false_clusters  = _false_signal_clusters(normalized)

    # ── Step 4: Regime ────────────────────────────────────────────────────────
    regime_stats    = _regime_analysis(normalized)
    best_regime, worst_regime = _best_worst_regime(regime_stats)

    # ── Step 5: Slippage ──────────────────────────────────────────────────────
    slippage        = _slippage_forensics(normalized)

    # ── Step 6: Risk metrics ──────────────────────────────────────────────────
    max_dd          = _running_peak_trough(pnl_series)
    streaks         = _streaks(pnl_series)
    sharpe          = _sharpe_like(pnl_series, hold_minutes)

    # ── Step 7: Recommendations ───────────────────────────────────────────────
    recommendations = _generate_recommendations(
        pnl_by_asset, pnl_by_source, pnl_by_conf,
        regime_stats, worst_regime, slippage,
        false_clusters, source_rankings, normalized,
    )

    return {
        # ── Core PnL ──────────────────────────────────────────────────────────
        "total_pnl":               round(all_pnl, 4),
        "realized_win_rate":       win_rate,
        "avg_pnl_per_trade":       avg_pnl,
        "expectancy":              expectancy,
        "avg_hold_time_minutes":   avg_hold,
        # ── Attribution ───────────────────────────────────────────────────────
        "pnl_by_asset_class":      pnl_by_asset,
        "pnl_by_news_category":    pnl_by_cat,
        "pnl_by_source":           pnl_by_source,
        "pnl_by_confidence_bucket": pnl_by_conf,
        "source_alpha_rankings":   source_rankings,
        "false_signal_clusters":   false_clusters,
        # ── Regime ────────────────────────────────────────────────────────────
        "regime_stats":            regime_stats,
        "best_regime":             best_regime,
        "worst_regime":            worst_regime,
        # ── Slippage ──────────────────────────────────────────────────────────
        "slippage_drag":           slippage,
        # ── Latency ───────────────────────────────────────────────────────────
        "avg_route_latency":       avg_latency,
        # ── Risk ──────────────────────────────────────────────────────────────
        "sharpe_like_score":       sharpe,
        "max_drawdown":            max_dd,
        "streaks":                 streaks,
        # ── Intelligence ──────────────────────────────────────────────────────
        "recommendations":         recommendations,
        # ── Meta ──────────────────────────────────────────────────────────────
        "_meta": {
            "version":        ANALYTICS_VERSION,
            "total_fills":    len(normalized),
            "closed_trades":  closed_n,
            "open_trades":    len(open_fills),
            "analyzed_at":    datetime.now(timezone.utc).isoformat(),
        },
    }


def _empty_report() -> dict:
    return {
        "total_pnl": 0.0, "realized_win_rate": 0.0, "avg_pnl_per_trade": 0.0,
        "expectancy": 0.0, "avg_hold_time_minutes": 0.0,
        "pnl_by_asset_class": {}, "pnl_by_news_category": {}, "pnl_by_source": {},
        "pnl_by_confidence_bucket": {}, "source_alpha_rankings": [],
        "false_signal_clusters": [], "regime_stats": {},
        "best_regime": "N/A", "worst_regime": "N/A",
        "slippage_drag": {"avg_slippage_bps": 0.0, "total_slippage_cost": 0.0},
        "avg_route_latency": 0.0, "sharpe_like_score": 0.0,
        "max_drawdown": 0.0, "streaks": {},
        "recommendations": ["⚠️  No fills to analyze."],
        "_meta": {
            "version": ANALYTICS_VERSION, "total_fills": 0,
            "closed_trades": 0, "open_trades": 0,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLUGIN REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_PLUGIN_REGISTRY: list[Callable[[dict], None]] = []


def register_plugin(fn: Callable[[dict], None]) -> None:
    """
    Register a post-analysis plugin.

    Examples:
        register_plugin(discord_report_sender)   # posts summary embed
        register_plugin(n8n_dashboard_webhook)   # triggers n8n workflow
        register_plugin(db_metrics_writer)       # persists to TimescaleDB
        register_plugin(grafana_push_gateway)    # sends to Prometheus PushGateway
    """
    _PLUGIN_REGISTRY.append(fn)


def analyze_and_dispatch(fills: list[dict]) -> dict:
    """
    Run analyze_fills() then fire all registered plugins with the report.
    """
    report = analyze_fills(fills)
    for plugin in _PLUGIN_REGISTRY:
        try:
            plugin(report)
        except Exception as exc:  # noqa: BLE001
            print(f"\033[1;33m⚠  ANALYTICS PLUGIN '{plugin.__name__}': {exc}\033[0m",
                  file=sys.stderr)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL TERMINAL PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_analytics_report(report: dict) -> None:  # noqa: C901 — intentionally rich
    """
    Full-colour institutional performance summary.
    Designed to be called after analyze_fills().
    """
    P  = "\033[1;35m"   # purple
    G  = "\033[1;32m"   # green
    R  = "\033[1;31m"   # red
    Y  = "\033[1;33m"   # yellow
    C  = "\033[1;36m"   # cyan
    B  = "\033[1m"      # bold
    D  = "\033[2m"      # dim
    X  = "\033[0m"      # reset

    def _col(val: float, pos_good: bool = True) -> str:
        if val > 0:
            return G if pos_good else R
        if val < 0:
            return R if pos_good else G
        return D

    def _bar(val: float, scale: float, width: int = 24, char_pos: str = "█", char_neg: str = "▓") -> str:
        if scale == 0:
            return "░" * width
        ratio = min(abs(val) / scale, 1.0)
        filled = round(ratio * width)
        ch = char_pos if val >= 0 else char_neg
        col = G if val >= 0 else R
        return col + ch * filled + D + "░" * (width - filled) + X

    def _pct(v: float) -> str:
        return f"{v:.1%}"

    def _dollar(v: float, sign: bool = True) -> str:
        prefix = "+" if (sign and v > 0) else ""
        return f"{prefix}${v:>10,.2f}"

    meta     = report.get("_meta", {})
    streaks  = report.get("streaks", {})
    slip     = report.get("slippage_drag", {})
    wr       = report.get("realized_win_rate", 0)
    total_pnl = report.get("total_pnl", 0)
    sharpe   = report.get("sharpe_like_score", 0)
    max_dd   = report.get("max_drawdown", 0)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{P}{'╔' + '═'*82 + '╗'}{X}")
    print(f"{P}║{X}  📊  {B}MONSTER TRADING AI — PERFORMANCE ANALYTICS REPORT{X}"
          f"  {D}v{ANALYTICS_VERSION}{X}{P}{'':>26}║{X}")
    print(f"{P}║{X}  {D}Analyzed: {meta.get('analyzed_at','?')[:19]}Z  |  "
          f"Fills: {meta.get('total_fills',0)}  |  "
          f"Closed: {meta.get('closed_trades',0)}  |  "
          f"Open: {meta.get('open_trades',0)}{X}"
          f"{P}{'':>15}║{X}")
    print(f"{P}{'╠' + '═'*82 + '╣'}{X}")

    # ── Core KPIs ─────────────────────────────────────────────────────────────
    print(f"{P}║{X}  {B}CORE PERFORMANCE{X}{P}{'':>64}║{X}")
    wr_col = G if wr >= 0.55 else (Y if wr >= 0.45 else R)
    sh_col = G if sharpe >= 1.0 else (Y if sharpe >= 0.5 else R)
    dd_col = G if max_dd < 500 else (Y if max_dd < 2000 else R)

    kpis = [
        ("Total P&L",       f"{_col(total_pnl)}{_dollar(total_pnl)}{X}"),
        ("Win Rate",         f"{wr_col}{_pct(wr)}{X}"),
        ("Expectancy/Trade", f"{_col(report.get('expectancy',0))}{_dollar(report.get('expectancy',0))}{X}"),
        ("Avg P&L/Trade",   f"{_col(report.get('avg_pnl_per_trade',0))}{_dollar(report.get('avg_pnl_per_trade',0))}{X}"),
        ("Sharpe-Like",     f"{sh_col}{sharpe:>+.3f}{X}"),
        ("Max Drawdown",    f"{dd_col}${max_dd:>10,.2f}{X}"),
        ("Avg Hold (min)",  f"{C}{report.get('avg_hold_time_minutes',0):>10.1f}{X}"),
        ("Avg Latency",     f"{C}{report.get('avg_route_latency',0):>9.1f}ms{X}"),
    ]
    for i in range(0, len(kpis), 2):
        l1, v1 = kpis[i]
        l2, v2 = kpis[i+1] if i+1 < len(kpis) else ("", "")
        print(f"{P}║{X}  {D}{l1:<20}{X}{v1}    {D}{l2:<20}{X}{v2}{P}{'':>5}║{X}")

    # ── Streaks ───────────────────────────────────────────────────────────────
    best_s  = streaks.get("best_win_streak", 0)
    worst_s = streaks.get("worst_loss_streak", 0)
    cur_s   = streaks.get("current_streak", 0)
    cur_d   = streaks.get("current_direction", "NONE")
    cur_col = G if cur_d == "WIN" else (R if cur_d == "LOSS" else D)
    print(f"{P}║{X}  {D}Best Win Streak{X}: {G}{best_s}{X}   "
          f"{D}Worst Loss Streak{X}: {R}{worst_s}{X}   "
          f"{D}Current{X}: {cur_col}{cur_s:+d} ({cur_d}){X}"
          f"{P}{'':>28}║{X}")

    # ── Slippage ──────────────────────────────────────────────────────────────
    print(f"{P}{'╠' + '═'*82 + '╣'}{X}")
    print(f"{P}║{X}  {B}SLIPPAGE FORENSICS{X}{P}{'':>62}║{X}")
    avg_slip = slip.get("avg_slippage_bps", 0)
    tot_slip = slip.get("total_slippage_cost", 0)
    slip_col = G if avg_slip < 10 else (Y if avg_slip < 20 else R)
    print(f"{P}║{X}  {D}Avg Slip{X}: {slip_col}{avg_slip:.1f} bps{X}   "
          f"{D}Total Drag{X}: {R}${tot_slip:,.2f}{X}   "
          f"{D}Worst Ticker{X}: {Y}{slip.get('worst_ticker','N/A')}{X}"
          f"{P}{'':>26}║{X}")
    for asset_cls, bps in slip.get("slippage_by_asset", {}).items():
        col = G if bps < 10 else (Y if bps < 20 else R)
        print(f"{P}║{X}    {D}{asset_cls:<10}{X} {col}{bps:.1f} bps{X}"
              f"  {_bar(bps, 30, 20)}"
              f"{P}{'':>26}║{X}")

    # ── Alpha by Asset Class ──────────────────────────────────────────────────
    print(f"{P}{'╠' + '═'*82 + '╣'}{X}")
    print(f"{P}║{X}  {B}PnL BY ASSET CLASS{X}{P}{'':>62}║{X}")
    max_abs_pnl = max((abs(v["total_pnl"]) for v in report["pnl_by_asset_class"].values()), default=1)
    for asset, stats in sorted(report["pnl_by_asset_class"].items(),
                                key=lambda x: x[1]["total_pnl"], reverse=True):
        bar  = _bar(stats["total_pnl"], max_abs_pnl)
        pnl  = stats["total_pnl"]
        wr_a = stats["win_rate"]
        ct   = stats["closed_trades"]
        exp  = stats["expectancy"]
        col  = _col(pnl)
        print(f"{P}║{X}  {B}{asset:<10}{X}  {bar}  "
              f"{col}{_dollar(pnl)}{X}  "
              f"{D}WR:{X} {_pct(wr_a):<7}  "
              f"{D}E:{X} {_col(exp)}{_dollar(exp,False)}{X}  "
              f"{D}n={ct}{X}"
              f"{P}  ║{X}")

    # ── Source Alpha Rankings ─────────────────────────────────────────────────
    print(f"{P}{'╠' + '═'*82 + '╣'}{X}")
    print(f"{P}║{X}  {B}SOURCE ALPHA RANKINGS{X}{P}{'':>59}║{X}")
    print(f"{P}║{X}  {D}{'RANK':<5}{'SOURCE':<18}{'TIER':<7}{'ALPHA':>8}{'WIN%':>8}"
          f"{'EXPECT':>10}{'TRADES':>8}{X}{P}{'':>19}║{X}")
    for rank, entry in enumerate(report["source_alpha_rankings"][:8], 1):
        tier     = entry["tier"]
        tier_str = "★" * (6 - tier) if tier <= 5 else "·"
        col      = G if entry["alpha_score"] > 0 else R
        alpha    = entry["alpha_score"]
        wr_s     = entry["win_rate"]
        exp_s    = entry["expectancy"]
        ct_s     = entry["closed_trades"]
        print(f"{P}║{X}  {D}{rank:<5}{X}{B}{entry['source']:<18}{X}"
              f"{D}{tier_str:<7}{X}"
              f"{col}{alpha:>8.2f}{X}"
              f"{'':>1}{_pct(wr_s):>7}"
              f"  {_col(exp_s)}{_dollar(exp_s,False):>9}{X}"
              f"  {D}{ct_s:>6}{X}"
              f"{P}{'':>2}║{X}")

    # ── Confidence Calibration ────────────────────────────────────────────────
    print(f"{P}{'╠' + '═'*82 + '╣'}{X}")
    print(f"{P}║{X}  {B}CONFIDENCE CALIBRATION{X}{P}{'':>58}║{X}")
    for label, stats in report["pnl_by_confidence_bucket"].items():
        ct   = stats.get("closed_trades", 0)
        wr_c = stats.get("win_rate", 0)
        ep_c = stats.get("expectancy", 0)
        bar  = _bar(wr_c - 0.5, 0.5, 16) if ct > 0 else D + "── no data ──" + X
        col  = _col(ep_c)
        print(f"{P}║{X}  {D}{label:<26}{X}  {bar}  "
              f"WR: {_pct(wr_c):<8}"
              f"E: {col}{_dollar(ep_c,False):>9}{X}  "
              f"{D}n={ct}{X}"
              f"{P}{'':>4}║{X}")

    # ── Regime Analysis ───────────────────────────────────────────────────────
    print(f"{P}{'╠' + '═'*82 + '╣'}{X}")
    print(f"{P}║{X}  {B}REGIME PERFORMANCE{X}  "
          f"{G}Best: {report['best_regime']}{X}  "
          f"{R}Worst: {report['worst_regime']}{X}"
          f"{P}{'':>36}║{X}")
    for regime, stats in sorted(report["regime_stats"].items(),
                                 key=lambda x: x[1]["expectancy"], reverse=True):
        ct   = stats["closed_trades"]
        ep_r = stats["expectancy"]
        wr_r = stats["win_rate"]
        col  = _col(ep_r)
        icon = "🏆" if regime == report["best_regime"] else ("💀" if regime == report["worst_regime"] else "  ")
        print(f"{P}║{X}  {icon} {regime:<18}  "
              f"WR: {_pct(wr_r):<8}"
              f"E: {col}{_dollar(ep_r,False)}{X}  "
              f"{D}n={ct}{X}"
              f"{P}{'':>18}║{X}")

    # ── False Signal Clusters ─────────────────────────────────────────────────
    if report["false_signal_clusters"]:
        print(f"{P}{'╠' + '═'*82 + '╣'}{X}")
        print(f"{P}║{X}  {B}🔴 FALSE SIGNAL CLUSTERS{X}{P}{'':>57}║{X}")
        for cluster in report["false_signal_clusters"][:4]:
            line = (f"  {cluster['source']:<12} · {cluster['category']:<20} · "
                    f"{cluster['regime']:<16} "
                    f"n={cluster['loss_count']} "
                    f"loss=${abs(cluster['total_loss']):,.0f}")
            print(f"{P}║{X}{R}{line[:82]}{X}{P}{'':>2}║{X}")

    # ── Recommendations ───────────────────────────────────────────────────────
    print(f"{P}{'╠' + '═'*82 + '╣'}{X}")
    print(f"{P}║{X}  {B}🧠 STRATEGY RECOMMENDATIONS{X}{P}{'':>54}║{X}")
    for rec in report["recommendations"][:6]:
        wrapped = textwrap.wrap(rec, 78)
        for i, line in enumerate(wrapped):
            indent = "  " if i == 0 else "    "
            print(f"{P}║{X}{indent}{line}{P}{'':>{max(2, 84-len(indent)-len(line))}}║{X}")

    print(f"{P}{'╚' + '═'*82 + '╝'}{X}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FILL GENERATOR  (for smoke tests)
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_fill(
    ticker:       str,
    side:         str         = "BUY",
    fill_price:   float       = 100.0,
    close_price:  float       = 0.0,
    notional:     float       = 5000.0,
    confidence:   float       = 0.88,
    source:       str         = "reuters",
    headline:     str         = "Market update",
    regime_hint:  str         = "",
    slippage_bps: float       = 8.0,
    latency_ms:   float       = 45.0,
    age_minutes:  float       = 30.0,
    broker_status: str        = "PAPER_FILLED",
) -> dict:
    """Build a minimal broker_sender-shaped dict for testing."""
    now = datetime.now(timezone.utc)
    fill_ts  = (now - timedelta(minutes=age_minutes)).isoformat()
    close_ts = now.isoformat() if close_price > 0 else None
    return {
        "_order_id":             str(uuid.uuid4()),
        "_ticker":               ticker,
        "broker_status":         broker_status,
        "fill_simulated_price":  fill_price,
        "close_price":           close_price,
        "close_timestamp":       close_ts,
        "sent_at":               fill_ts,
        "signal_timestamp":      fill_ts,
        "adjusted_position_size": notional / 100_000,
        "route_latency_ms":      latency_ms,
        "retry_count":           0,
        "confidence_score":      confidence,
        "source":                source,
        "risk_score":            confidence,
        "headline":              headline + " " + regime_hint,
        "alert_priority":        "CRITICAL",
        "_signal_type":          side,
        "broker_payload": {
            "symbol":    ticker,
            "side":      side,
            "notional":  notional,
            "slippage_bps": slippage_bps,
            "_expected_price": fill_price * (1 - slippage_bps / 10_000 * (1 if side == "BUY" else -1)),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  —  20 SYNTHETIC FILLS
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    GREEN  = "\033[1;32m"
    RED    = "\033[1;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[1;36m"
    PURPLE = "\033[1;35m"
    RESET  = "\033[0m"

    print(f"\n{PURPLE}{'▓'*86}")
    print("  🧪  PERFORMANCE ANALYTICS — SMOKE TEST SUITE  (20 synthetic fills)")
    print(f"{'▓'*86}{RESET}\n")

    def _check(label: str, condition: bool, detail: str = "") -> None:
        ok = bool(condition)
        tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{tag}]  {label}")
        if detail:
            print(f"          {CYAN}↳ {detail}{RESET}")
        if ok:
            _check.passed += 1
        else:
            _check.failed += 1
    _check.passed = 0
    _check.failed = 0

    # ── Build 20 synthetic fills ──────────────────────────────────────────────
    fills = [
        # Winning equity trades — Bloomberg, high confidence
        _synthetic_fill("NVDA", "BUY",  138.0, 151.0, 8000, 0.94, "bloomberg",
                        "NVDA earnings beat EPS", "", 6.0, 38.0, 45.0),
        _synthetic_fill("AAPL", "BUY",  228.0, 235.0, 5000, 0.91, "bloomberg",
                        "Apple announces buyback", "", 7.0, 42.0, 60.0),
        _synthetic_fill("MSFT", "SELL", 415.0, 400.0, 6000, 0.90, "reuters",
                        "Microsoft guidance cut", "", 8.0, 35.0, 30.0),
        _synthetic_fill("TSLA", "BUY",  248.0, 265.0, 4000, 0.88, "reuters",
                        "Tesla deliveries record", "", 10.0, 55.0, 90.0),
        _synthetic_fill("JPM",  "BUY",  240.0, 250.0, 7000, 0.87, "reuters",
                        "JPM beats on revenue", "earnings", 7.0, 40.0, 50.0),
        # Losing equity trades — Twitter source, lower confidence
        _synthetic_fill("GME",  "BUY",   20.0,  14.0, 3000, 0.81, "twitter",
                        "GME short squeeze incoming", "choppy sideways", 18.0, 180.0, 120.0),
        _synthetic_fill("AMC",  "BUY",    5.0,   3.0, 2000, 0.82, "reddit",
                        "AMC breakout imminent", "choppy range", 22.0, 210.0, 110.0),
        _synthetic_fill("BBBY", "BUY",    2.0,   0.5, 1500, 0.80, "twitter",
                        "BBBY meme momentum", "choppy noise", 25.0, 230.0, 100.0),
        # Crypto — high vol, wider slippage
        _synthetic_fill("BTC",  "BUY",  95000, 102000, 10000, 0.93, "bloomberg",
                        "Bitcoin ETF inflows surge", "crypto squeeze", 28.0, 120.0, 20.0),
        _synthetic_fill("ETH",  "BUY",   3500,   3200,  8000, 0.89, "cnbc",
                        "Ethereum staking yield drops", "crypto squeeze", 30.0, 115.0, 25.0),
        _synthetic_fill("SOL",  "SELL",   200,    170,  5000, 0.85, "cnbc",
                        "Solana network congestion", "", 26.0, 108.0, 15.0),
        _synthetic_fill("COIN", "BUY",   235,    250,  4000, 0.92, "reuters",
                        "Coinbase trading volume record", "crypto", 12.0, 95.0, 30.0),
        # Macro / Fed — post-FOMC regime
        _synthetic_fill("SPY",  "SELL", 545.0,  530.0, 12000, 0.95, "reuters",
                        "Fed signals 50bps cut rate decision fomc", "post-FOMC", 4.0, 22.0, 5.0),
        _synthetic_fill("TLT",  "BUY",   95.0,  100.0,  6000, 0.91, "bloomberg",
                        "Treasury bonds rally on Fed pivot", "", 5.0, 28.0, 10.0),
        _synthetic_fill("GLD",  "BUY",  237.0,  245.0,  5000, 0.88, "reuters",
                        "Gold surges on inflation data CPI", "macro", 6.0, 30.0, 20.0),
        # Panic regime
        _synthetic_fill("QQQ",  "SELL", 472.0,  451.0,  9000, 0.93, "bloomberg",
                        "Market crash panic circuit halt", "panic collapse", 8.0, 18.0, 2.0),
        _synthetic_fill("XLF",  "SELL",  47.0,   44.0,  4000, 0.87, "reuters",
                        "Banking sector bank collapse fears", "banking crisis panic", 9.0, 35.0, 8.0),
        # False signals — seekingalpha, low tier
        _synthetic_fill("MSTR", "BUY",  388.0,  370.0,  3000, 0.82, "seekingalpha",
                        "MSTR crypto treasury strategy", "choppy", 14.0, 150.0, 75.0),
        # Open trades (no close_price) — should not affect PnL
        _synthetic_fill("AMD",  "BUY",  152.0,    0.0,  5000, 0.86, "cnbc",
                        "AMD AI chip demand surge", "", 9.0, 60.0, 5.0),
        # Rejected fill (should be normalized out)
        _synthetic_fill("AVGO", "BUY", 1720.0,    0.0,  3000, 0.90, "bloomberg",
                        "AVGO dividend raise", "", 7.0, 45.0, 10.0, "REJECTED"),
    ]

    report = analyze_fills(fills)

    # ── Test assertions ────────────────────────────────────────────────────────
    meta = report["_meta"]
    _check(
        "T01 — Correct fill count (19 valid, 1 rejected)",
        meta["total_fills"] == 19,
        f"total_fills={meta['total_fills']} (expected 19)",
    )
    _check(
        "T02 — Closed trade count correct",
        meta["closed_trades"] >= 17,
        f"closed_trades={meta['closed_trades']} (expected >=17)",
    )
    _check(
        "T03 — Open trade count >= 1",
        meta["open_trades"] >= 1,
        f"open_trades={meta['open_trades']}",
    )
    _check(
        "T04 — total_pnl is non-zero float",
        isinstance(report["total_pnl"], float) and report["total_pnl"] != 0.0,
        f"total_pnl={report['total_pnl']:.2f}",
    )
    _check(
        "T05 — realized_win_rate in [0, 1]",
        0.0 <= report["realized_win_rate"] <= 1.0,
        f"win_rate={report['realized_win_rate']:.2%}",
    )
    _check(
        "T06 — pnl_by_asset_class has EQUITY, CRYPTO, MACRO",
        all(k in report["pnl_by_asset_class"] for k in ("EQUITY", "CRYPTO", "MACRO")),
        f"asset_classes={list(report['pnl_by_asset_class'].keys())}",
    )
    _check(
        "T07 — pnl_by_source has bloomberg and reuters",
        all(k in report["pnl_by_source"] for k in ("bloomberg", "reuters")),
        f"sources={list(report['pnl_by_source'].keys())}",
    )
    _check(
        "T08 — source_alpha_rankings is ordered list",
        (isinstance(report["source_alpha_rankings"], list)
         and len(report["source_alpha_rankings"]) > 0),
        f"rankings count={len(report['source_alpha_rankings'])}",
    )
    _check(
        "T09 — Bloomberg ranked above Twitter (tier 1 > tier 4)",
        (lambda ranks: (
            next((r for r in ranks if r["source"] == "bloomberg"), {}).get("alpha_score", -1)
            > next((r for r in ranks if r["source"] == "twitter"), {}).get("alpha_score", -99)
        ))(report["source_alpha_rankings"]),
        "bloomberg.alpha_score > twitter.alpha_score",
    )
    _check(
        "T10 — pnl_by_confidence_bucket has elite bucket",
        any("elite" in k for k in report["pnl_by_confidence_bucket"]),
        f"buckets={list(report['pnl_by_confidence_bucket'].keys())}",
    )
    _check(
        "T11 — best_regime and worst_regime are strings",
        isinstance(report["best_regime"], str) and isinstance(report["worst_regime"], str),
        f"best={report['best_regime']} | worst={report['worst_regime']}",
    )
    _check(
        "T12 — max_drawdown >= 0",
        report["max_drawdown"] >= 0,
        f"max_drawdown=${report['max_drawdown']:,.2f}",
    )
    _check(
        "T13 — streaks dict has required keys",
        all(k in report["streaks"] for k in (
            "best_win_streak", "worst_loss_streak",
            "current_streak", "current_direction"
        )),
        f"streak_keys={list(report['streaks'].keys())}",
    )
    _check(
        "T14 — slippage_drag has avg_slippage_bps > 0",
        report["slippage_drag"]["avg_slippage_bps"] > 0,
        f"avg_slippage_bps={report['slippage_drag']['avg_slippage_bps']:.2f}",
    )
    _check(
        "T15 — avg_route_latency > 0",
        report["avg_route_latency"] > 0,
        f"avg_latency={report['avg_route_latency']:.1f}ms",
    )
    _check(
        "T16 — recommendations is non-empty list",
        isinstance(report["recommendations"], list) and len(report["recommendations"]) >= 1,
        f"rec_count={len(report['recommendations'])}",
    )
    # T17: inject two fills that share exact (source, category, regime) to trigger cluster
    cluster_fills = fills + [
        _synthetic_fill("WISH", "BUY", 1.0, 0.3, 1000, 0.81, "twitter",
                        "WISH meme incoming", "choppy sideways", 18.0, 180.0, 110.0),
        _synthetic_fill("CLOV", "BUY", 3.0, 1.0, 800, 0.80, "twitter",
                        "CLOV short squeeze choppy", "choppy sideways", 20.0, 200.0, 108.0),
    ]
    cluster_report = analyze_fills(cluster_fills)
    _check(
        "T17 — false_signal_clusters detected with matching (source+category+regime)",
        len(cluster_report["false_signal_clusters"]) >= 1,
        f"cluster_count={len(cluster_report['false_signal_clusters'])}",
    )
    _check(
        "T18 — empty fills returns safe empty report",
        analyze_fills([])["total_pnl"] == 0.0
        and analyze_fills([])["_meta"]["total_fills"] == 0,
        "empty report structure valid",
    )
    _check(
        "T19 — non-dict fills safely skipped",
        analyze_fills(["bad", None, 42])["total_pnl"] == 0.0,
        "non-dict entries produce empty report",
    )
    # ── Plugin test ───────────────────────────────────────────────────────────
    _PLUGIN_REGISTRY.clear()
    plugin_results = []

    def _mock_plugin(rep: dict) -> None:
        plugin_results.append(rep.get("total_pnl"))

    register_plugin(_mock_plugin)
    analyze_and_dispatch(fills)
    _check(
        "T20 — Plugin fires with full report",
        len(plugin_results) == 1 and isinstance(plugin_results[0], float),
        f"plugin received total_pnl={plugin_results[0]:.2f}" if plugin_results else "no result",
    )

    # ── Print full report ──────────────────────────────────────────────────────
    print(f"\n  {YELLOW}── Full Analytics Report ───────────────────────────────────────{RESET}")
    print_analytics_report(report)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = _check.passed + _check.failed
    print(f"\n{PURPLE}{'═'*86}")
    print(
        f"  🏁  SMOKE TEST RESULTS — {_check.passed}/{total} passed   "
        f"{'✅ ALL CLEAR' if _check.failed == 0 else '❌ FAILURES DETECTED'}"
    )
    print(f"{'═'*86}{RESET}\n")
    sys.exit(0 if _check.failed == 0 else 1)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _startup_banner() -> None:
    print("""
\033[1;32m╔═════════════════════════════════════════════════════════════════════════════════╗
║   📊  MONSTER TRADING AI — PERFORMANCE ANALYTICS  v1.0.0                       ║
║   Alpha Discovery Infrastructure — O(n) · Standard Library · No pandas        ║
║   8 Layers: Normalization · PnL · Attribution · Calibration ·                 ║
║             Regime · Slippage · Drawdown/Streak · Recommendations             ║
╚═════════════════════════════════════════════════════════════════════════════════╝\033[0m
""")


if __name__ == "__main__":
    if "--smoke" in sys.argv or "-s" in sys.argv:
        _smoke_test()
    else:
        _startup_banner()
        print("Usage:")
        print("  python performance_analytics.py --smoke    # run 20-fill smoke test")
        print()
        print("API:")
        print("  from performance_analytics import analyze_fills, print_analytics_report")
        print("  report = analyze_fills(broker_sender_output)")
        print("  print_analytics_report(report)")
        print()
        print("Plugins:")
        print("  from performance_analytics import register_plugin, analyze_and_dispatch")
        print("  register_plugin(discord_reporter)")
        print("  register_plugin(n8n_dashboard_hook)")
        print("  report = analyze_and_dispatch(fills)")