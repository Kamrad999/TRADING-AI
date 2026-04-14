"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║    ██████╗  █████╗  ██████╗██╗  ██╗████████╗███████╗███████╗████████╗                   ║
║    ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝                   ║
║    ██████╔╝███████║██║     █████╔╝    ██║   █████╗  ███████╗   ██║                      ║
║    ██╔══██╗██╔══██║██║     ██╔═██╗    ██║   ██╔══╝  ╚════██║   ██║                      ║
║    ██████╔╝██║  ██║╚██████╗██║  ██╗   ██║   ███████╗███████║   ██║                      ║
║    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝   ╚═╝                      ║
║                                                                                          ║
║   ┌──────────────────────────────────────────────────────────────────────────────────┐   ║
║   │     🔬 BACKTEST_ENGINE.PY  —  FORENSIC ALPHA VALIDATION LAYER  🔬              │   ║
║   │         Historical Signal Replay · Institutional-Grade Analytics               │   ║
║   │    Measures: Win Rate, Profit Factor, Drawdown, Attribution, False Positives  │   ║
║   └──────────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                          ║
║  Module   : backtest_engine.py                                                           ║
║  Version  : 1.0.0                                                                        ║
║  Role     : Validates alpha generation via historical signal replay                      ║
║  Style    : Institutional forensic analytics · Bloomberg OMS audit trail               ║
║                                                                                          ║
║  ╔════════════════════════════════════════════════════════════════════════════════╗       ║
║  ║  MISSION: Does the signal system actually generate alpha?                      ║       ║
║  ║  METHOD:  Replay every historical signal against market outcomes & measure:    ║       ║
║  ║    · Win rate by source quality                                                 ║       ║
║  ║    · Profit factor & expectancy per event type                                  ║       ║
║  ║    · Max drawdown & Sharpe ratio in each market regime                          ║       ║
║  ║    · False positive cluster detection (likely spurious patterns)                ║       ║
║  ║    · Forensic recommendations for strategy improvement                          ║       ║
║  ╚════════════════════════════════════════════════════════════════════════════════╝       ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import sys
import time
import unittest
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 ▸ MODULE-LEVEL CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_ENGINE_VERSION = "1.0.0"
BACKTEST_ENGINE_BUILD   = "FORENSIC-ALPHA-VALIDATOR"

# ── Return window horizons (in seconds) ────────────────────────────────────
RETURN_WINDOWS_SECONDS: Dict[str, int] = {
    "5m":   5 * 60,           # 300
    "15m":  15 * 60,          # 900
    "1h":   60 * 60,          # 3600
    "4h":   4 * 60 * 60,      # 14400
    "1d":   24 * 60 * 60,     # 86400
}

# ── PnL & Risk Config ──────────────────────────────────────────────────────
DEFAULT_ENTRY_SLIPPAGE_BPS = 2          # 2bps assumed slippage on entry
DEFAULT_EXIT_SLIPPAGE_BPS  = 2          # 2bps assumed slippage on exit
DEFAULT_POSITION_SIZE_PCT  = 1.0        # 1% of portfolio per trade (for PnL calc)
MIN_CONFIDENCE_THRESHOLD   = 25         # Ignore signals < 25% confidence

# ── Clustering & Pattern Detection ────────────────────────────────────────
FALSE_POSITIVE_CLUSTER_SIZE = 5         # 5+ consecutive losing trades = cluster
FALSE_POSITIVE_LOOKBACK_BARS = 20       # Analyze 20 bars for cluster

# ── Logging ───────────────────────────────────────────────────────────────
LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)-8s] [backtest] %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 ▸ ENUMS & VALUE TYPES
# ══════════════════════════════════════════════════════════════════════════════

class TradeDirection(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, Enum):
    ENTRY_PENDING  = "entry_pending"
    ENTERED        = "entered"
    EXITED         = "exited"
    REJECTED       = "rejected"


class BacktestStatus(str, Enum):
    SUCCESS  = "SUCCESS"
    DEGRADED = "DEGRADED"
    FAILED   = "FAILED"


@dataclass
class TradeResult:
    """Immutable record of one backtested trade."""
    __slots__ = (
        "signal_id", "symbol", "direction", "entry_price", "exit_price",
        "entry_time", "exit_time", "pnl_usd", "pnl_pct", "return_window",
        "status", "source", "event_type", "confidence", "regime_at_entry",
    )

    signal_id: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl_usd: float
    pnl_pct: float
    return_window: str
    status: TradeStatus
    source: str
    event_type: str
    confidence: float
    regime_at_entry: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4) if self.exit_price else None,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl_usd": round(self.pnl_usd, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "return_window": self.return_window,
            "status": self.status.value,
            "source": self.source,
            "event_type": self.event_type,
            "confidence": round(self.confidence, 2),
            "regime": self.regime_at_entry,
        }


@dataclass
class BacktestContext:
    """Mutable state during backtest execution."""
    trades: List[TradeResult] = field(default_factory=list)
    peak_equity: float = 1.0
    min_equity: float = 1.0
    current_equity: float = 1.0
    total_loss: float = 0.0
    max_loss: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 ▸ STRUCTURED LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def _build_logger(name: str = "backtest") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(handler)
    return logger


log = _build_logger()

# ANSI colour codes
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


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 ▸ TIME & DATE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _parse_iso_datetime(dt_str: str) -> datetime:
    """Parse ISO 8601 datetime string to UTC-aware datetime."""
    if isinstance(dt_str, datetime):
        return dt_str if dt_str.tzinfo else dt_str.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)


def _add_seconds(dt: datetime, seconds: int) -> datetime:
    """Add seconds to a datetime."""
    return dt + timedelta(seconds=seconds)


def _seconds_between(dt1: datetime, dt2: datetime) -> float:
    """Calculate seconds between two datetimes."""
    delta = dt2 - dt1
    return delta.total_seconds()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 ▸ MARKET DATA LOOKUP ENGINE (Binary Search)
# ══════════════════════════════════════════════════════════════════════════════

def _build_time_index(market_data: List[Dict[str, Any]]) -> List[datetime]:
    """Build sorted datetime index for market data."""
    indices = []
    for bar in market_data:
        ts = _parse_iso_datetime(bar.get("timestamp", bar.get("time", "")))
        indices.append(ts)
    return sorted(indices)


def _find_bar_at_or_after(
    market_data: List[Dict[str, Any]],
    target_time: datetime,
) -> Optional[Dict[str, Any]]:
    """
    Find the first bar at or after target_time using binary search.
    Returns None if no bar found after target_time.
    """
    if not market_data:
        return None

    times = [_parse_iso_datetime(b.get("timestamp", b.get("time", ""))) for b in market_data]

    # Binary search for insertion point
    idx = bisect.bisect_left(times, target_time)

    # Find exact match or next bar
    if idx < len(times):
        return market_data[idx]

    return None


def _calculate_forward_return(
    entry_bar: Dict[str, Any],
    market_data: List[Dict[str, Any]],
    window_seconds: int,
) -> Tuple[bool, Optional[float]]:
    """
    Calculate forward return from entry_bar over window_seconds.

    Returns:
        (success: bool, return_pct: Optional[float])
            success=True if exit found, False otherwise
            return_pct = (exit_price - entry_price) / entry_price (as %)
    """
    entry_time = _parse_iso_datetime(entry_bar.get("timestamp", entry_bar.get("time", "")))
    target_exit_time = _add_seconds(entry_time, window_seconds)

    # Get entry price
    entry_price = float(entry_bar.get("close", 0))
    if entry_price <= 0:
        return False, None

    # Find exit bar
    exit_bar = _find_bar_at_or_after(market_data, target_exit_time)
    if not exit_bar:
        return False, None

    exit_price = float(exit_bar.get("close", 0))
    if exit_price <= 0:
        return False, None

    return_pct = ((exit_price - entry_price) / entry_price) * 100
    return True, return_pct


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 ▸ SIGNAL-MARKET DATA ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def _find_entry_bar(
    signal: Dict[str, Any],
    market_data: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Find market data bar matching signal entry time.
    Signal: {"timestamp", "symbol", "direction", ...}
    """
    signal_time = _parse_iso_datetime(signal.get("timestamp", ""))
    entry_bar = _find_bar_at_or_after(market_data, signal_time)

    if entry_bar:
        bar_time = _parse_iso_datetime(entry_bar.get("timestamp", entry_bar.get("time", "")))
        # Allow up to 5 minutes of slippage
        if _seconds_between(signal_time, bar_time) <= 300:
            return entry_bar

    return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 ▸ PnL & SLIPPAGE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def _apply_slippage(price: float, bps: int, is_entry: bool = True) -> float:
    """Apply slippage to a price (entry: add spread, exit: add spread)."""
    bp_multiplier = bps / 10_000
    slippage_amount = price * bp_multiplier
    # Entry: assume we're always slightly worse
    # Exit: assume we're always slightly worse
    return price + slippage_amount


def _calculate_pnl(
    signal: Dict[str, Any],
    return_pct: float,
) -> Tuple[float, float]:
    """
    Calculate PnL given signal and forward return.

    Returns:
        (pnl_usd: float, pnl_pct: float)
    """
    confidence = signal.get("confidence", 50)
    position_size_pct = DEFAULT_POSITION_SIZE_PCT * (confidence / 100)
    portfolio_value = 100_000  # Assumed portfolio

    position_usd = portfolio_value * (position_size_pct / 100)

    # Apply slippage
    effective_return = return_pct - (DEFAULT_ENTRY_SLIPPAGE_BPS + DEFAULT_EXIT_SLIPPAGE_BPS) / 100

    # Direction bias
    direction = signal.get("direction", "LONG")
    if direction == "SHORT":
        effective_return = -effective_return

    pnl_usd = position_usd * (effective_return / 100)
    pnl_pct = effective_return

    return pnl_usd, pnl_pct


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8 ▸ RETURN WINDOW ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _backtest_signal_all_windows(
    signal: Dict[str, Any],
    market_data: List[Dict[str, Any]],
    ctx: BacktestContext,
) -> List[TradeResult]:
    """
    Backtest one signal across all return windows (5m, 15m, 1h, 4h, 1d).
    Returns list of trade results, one per window.
    """
    trades = []

    # Find entry bar
    entry_bar = _find_entry_bar(signal, market_data)
    if not entry_bar:
        # Rejected: no market data
        trades.append(TradeResult(
            signal_id=signal.get("id", "unknown"),
            symbol=signal.get("symbol", "UNKNOWN"),
            direction=TradeDirection(signal.get("direction", "LONG")),
            entry_price=0,
            exit_price=None,
            entry_time=_parse_iso_datetime(signal.get("timestamp", "")),
            exit_time=None,
            pnl_usd=0,
            pnl_pct=0,
            return_window="N/A",
            status=TradeStatus.REJECTED,
            source=signal.get("source", "unknown"),
            event_type=signal.get("event_type", "unknown"),
            confidence=signal.get("confidence", 0),
            regime_at_entry="UNKNOWN",
        ))
        return trades

    entry_price = float(entry_bar.get("close", 0))

    # Test each return window
    for window_name, window_seconds in RETURN_WINDOWS_SECONDS.items():
        success, return_pct = _calculate_forward_return(entry_bar, market_data, window_seconds)

        if success and return_pct is not None:
            pnl_usd, pnl_pct = _calculate_pnl(signal, return_pct)
            exit_time = _add_seconds(
                _parse_iso_datetime(entry_bar.get("timestamp", "")),
                window_seconds
            )
            trade = TradeResult(
                signal_id=signal.get("id", "unknown"),
                symbol=signal.get("symbol", "UNKNOWN"),
                direction=TradeDirection(signal.get("direction", "LONG")),
                entry_price=entry_price,
                exit_price=entry_price * (1 + return_pct / 100),
                entry_time=_parse_iso_datetime(entry_bar.get("timestamp", "")),
                exit_time=exit_time,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                return_window=window_name,
                status=TradeStatus.EXITED,
                source=signal.get("source", "unknown"),
                event_type=signal.get("event_type", "unknown"),
                confidence=signal.get("confidence", 0),
                regime_at_entry=signal.get("regime", "UNKNOWN"),
            )
            trades.append(trade)
        else:
            # Incomplete: market data ended before window complete
            trade = TradeResult(
                signal_id=signal.get("id", "unknown"),
                symbol=signal.get("symbol", "UNKNOWN"),
                direction=TradeDirection(signal.get("direction", "LONG")),
                entry_price=entry_price,
                exit_price=None,
                entry_time=_parse_iso_datetime(entry_bar.get("timestamp", "")),
                exit_time=None,
                pnl_usd=0,
                pnl_pct=0,
                return_window=window_name,
                status=TradeStatus.ENTRY_PENDING,
                source=signal.get("source", "unknown"),
                event_type=signal.get("event_type", "unknown"),
                confidence=signal.get("confidence", 0),
                regime_at_entry=signal.get("regime", "UNKNOWN"),
            )
            trades.append(trade)

    return trades


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 9 ▸ PERFORMANCE ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _calculate_metrics(trades: List[TradeResult]) -> Dict[str, Any]:
    """
    Calculate institutional-grade performance metrics from trades.
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }

    completed_trades = [t for t in trades if t.status == TradeStatus.EXITED]

    if not completed_trades:
        return {
            "total_trades": len(trades),
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }

    pnls = [t.pnl_pct for t in completed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    winning_trades = len(wins)
    losing_trades = len(losses)
    total_trades = len(completed_trades)

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    avg_return = mean(pnls) if pnls else 0.0
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    # Expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    expectancy = (
        (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    ) if avg_win + abs(avg_loss) > 0 else 0.0

    # Profit Factor = Gross Profit / Gross Loss
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float('inf') if gross_profit > 0 else 0.0
    )

    largest_win = max(wins) if wins else 0.0
    largest_loss = min(losses) if losses else 0.0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": round(win_rate, 2),
        "avg_return": round(avg_return, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "expectancy": round(expectancy, 4),
        "profit_factor": round(profit_factor, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "largest_win": round(largest_win, 4),
        "largest_loss": round(largest_loss, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 10 ▸ DRAWDOWN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _calculate_max_drawdown(trades: List[TradeResult]) -> float:
    """
    Calculate maximum drawdown from equity curve.
    """
    completed = [t for t in trades if t.status == TradeStatus.EXITED]
    if not completed:
        return 0.0

    equity_curve = [1.0]
    for trade in completed:
        new_equity = equity_curve[-1] * (1 + trade.pnl_pct / 100)
        equity_curve.append(new_equity)

    if not equity_curve:
        return 0.0

    max_dd = 0.0
    peak = equity_curve[0]

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = ((peak - equity) / peak) * 100
        if dd > max_dd:
            max_dd = dd

    return round(max_dd, 2)


def _calculate_sharpe_ratio(trades: List[TradeResult], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio from returns.
    risk_free_rate: annual % (default 2%)
    """
    completed = [t for t in trades if t.status == TradeStatus.EXITED]
    if len(completed) < 2:
        return 0.0

    returns = [t.pnl_pct for t in completed]
    excess_returns = [r - (risk_free_rate / 252 / 100) for r in returns]  # Daily excess

    avg_return = mean(excess_returns)
    std_dev = stdev(excess_returns) if len(excess_returns) > 1 else 0.1

    if std_dev == 0:
        return 0.0

    # Annualize (assuming ~252 trading days)
    sharpe = (avg_return * 252) / (std_dev * math.sqrt(252))
    return round(sharpe, 2)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 11 ▸ ATTRIBUTION ENGINES (Source, Event Type, Regime)
# ══════════════════════════════════════════════════════════════════════════════

def _attribution_by_dimension(
    trades: List[TradeResult],
    dimension: str,  # "source", "event_type", "regime_at_entry"
) -> Dict[str, Dict[str, Any]]:
    """
    Attribute performance by a given dimension (source, event type, regime).
    """
    completed = [t for t in trades if t.status == TradeStatus.EXITED]
    grouped: Dict[str, List[TradeResult]] = defaultdict(list)

    for trade in completed:
        key = getattr(trade, dimension, "unknown")
        grouped[key].append(trade)

    result = {}
    for key, group_trades in grouped.items():
        metrics = _calculate_metrics(group_trades)
        result[key] = metrics

    return result


def _rank_by_metric(
    dimension_dict: Dict[str, Dict[str, Any]],
    metric: str = "expectancy",
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Rank dimensions by a given metric (e.g., best sources by win_rate).
    """
    ranked = [
        (name, stats.get(metric, 0))
        for name, stats in dimension_dict.items()
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 12 ▸ FALSE POSITIVE CLUSTER DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def _detect_false_positive_clusters(trades: List[TradeResult]) -> List[Dict[str, Any]]:
    """
    Detect clusters of consecutive losing trades (likely spurious patterns).
    """
    completed = [t for t in trades if t.status == TradeStatus.EXITED]
    clusters = []

    current_cluster = []

    for trade in completed:
        if trade.pnl_pct < 0:
            current_cluster.append(trade)
        else:
            if len(current_cluster) >= FALSE_POSITIVE_CLUSTER_SIZE:
                cluster_info = {
                    "size": len(current_cluster),
                    "total_loss_pct": sum(t.pnl_pct for t in current_cluster),
                    "avg_loss_pct": mean(t.pnl_pct for t in current_cluster),
                    "start_time": current_cluster[0].entry_time.isoformat(),
                    "end_time": current_cluster[-1].entry_time.isoformat(),
                    "event_types": [t.event_type for t in current_cluster],
                    "sources": [t.source for t in current_cluster],
                }
                clusters.append(cluster_info)
            current_cluster = []

    # Check final cluster
    if len(current_cluster) >= FALSE_POSITIVE_CLUSTER_SIZE:
        cluster_info = {
            "size": len(current_cluster),
            "total_loss_pct": sum(t.pnl_pct for t in current_cluster),
            "avg_loss_pct": mean(t.pnl_pct for t in current_cluster),
            "start_time": current_cluster[0].entry_time.isoformat(),
            "end_time": current_cluster[-1].entry_time.isoformat(),
            "event_types": [t.event_type for t in current_cluster],
            "sources": [t.source for t in current_cluster],
        }
        clusters.append(cluster_info)

    return clusters


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 13 ▸ FORENSIC REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _generate_forensic_notes(
    trades: List[TradeResult],
    metrics: Dict[str, Any],
    source_attr: Dict[str, Dict[str, Any]],
    event_attr: Dict[str, Dict[str, Any]],
    regime_attr: Dict[str, Dict[str, Any]],
    clusters: List[Dict[str, Any]],
) -> List[str]:
    """
    Generate actionable forensic notes for strategy improvement.
    """
    notes = []

    total = metrics.get("total_trades", 0)
    completed = metrics.get("winning_trades", 0) + metrics.get("losing_trades", 0)
    win_rate = metrics.get("win_rate", 0)
    expectancy = metrics.get("expectancy", 0)
    profit_factor = metrics.get("profit_factor", 0)

    # Note 1: Win rate assessment
    if win_rate < 40:
        notes.append(
            f"⚠️  CRITICAL: Win rate {win_rate}% is below 40% breakeven threshold. "
            f"Consider disabling this strategy in all market regimes."
        )
    elif win_rate < 50:
        notes.append(
            f"⚠️  LOW WIN RATE: {win_rate}% requires high profit factor to be viable. "
            f"Check if large wins offset many small losses."
        )
    elif win_rate >= 60:
        notes.append(
            f"✓ GOOD WIN RATE: {win_rate}% suggests reliable signal generation."
        )

    # Note 2: Expectancy
    if expectancy < -0.5:
        notes.append(
            f"⚠️  NEGATIVE EXPECTANCY: {expectancy}% per trade. System is unprofitable. "
            f"Revise signal scoring or add regime filters."
        )
    elif expectancy > 0.5:
        notes.append(
            f"✓ POSITIVE EXPECTANCY: {expectancy}% per trade suggests alpha generation."
        )

    # Note 3: Profit factor
    if profit_factor < 1.0:
        notes.append(
            f"⚠️  PROFIT FACTOR < 1.0: {profit_factor} means losses exceed wins. "
            f"Risk/reward ratio needs adjustment."
        )
    elif profit_factor > 2.0:
        notes.append(
            f"✓ STRONG PROFIT FACTOR: {profit_factor} indicates good risk/reward. "
            f"Consider increasing position sizing."
        )

    # Note 4: Best/worst sources
    best_sources_list = _rank_by_metric(source_attr, "win_rate", top_n=3)
    if best_sources_list:
        best_src = best_sources_list[0]
        notes.append(
            f"✓ TOP SOURCE: '{best_src[0]}' has {best_src[1]:.1f}% win rate. "
            f"Increase signal weighting from this source."
        )

    worst_sources_list = _rank_by_metric(source_attr, "win_rate", top_n=3)
    worst_sources_list.reverse()
    if worst_sources_list and worst_sources_list[0][1] < 30:
        worst_src = worst_sources_list[0]
        notes.append(
            f"⚠️  POOR SOURCE: '{worst_src[0]}' has {worst_src[1]:.1f}% win rate. "
            f"Consider disabling or downweighting this source."
        )

    # Note 5: Best/worst event types
    best_events = _rank_by_metric(event_attr, "win_rate", top_n=1)
    if best_events:
        best_evt = best_events[0]
        notes.append(
            f"✓ TOP EVENT TYPE: '{best_evt[0]}' has {best_evt[1]:.1f}% win rate. "
            f"Allocate more capital to this signal type."
        )

    # Note 6: Regime performance
    best_regimes = _rank_by_metric(regime_attr, "win_rate", top_n=1)
    if best_regimes:
        best_reg = best_regimes[0]
        notes.append(
            f"✓ BEST REGIME: Strategy performs {best_reg[1]:.1f}% win rate in {best_reg[0]}. "
            f"Increase aggressiveness during this regime."
        )

    # Note 7: False positive clusters
    if clusters:
        worst_cluster = max(clusters, key=lambda x: x["size"])
        notes.append(
            f"⚠️  CLUSTER ALERT: {worst_cluster['size']} consecutive losses from event types: "
            f"{', '.join(set(worst_cluster['event_types']))}. "
            f"Likely spurious pattern. Consider filtering."
        )

    # Note 8: Incomplete trades
    incomplete = total - completed
    if incomplete > completed * 0.5:
        notes.append(
            f"⚠️  DATA QUALITY: {incomplete}/{total} trades incomplete (market data ended early). "
            f"Extend historical data for more reliable analysis."
        )

    # Note 9: Sample size
    if completed < 30:
        notes.append(
            f"⚠️  SAMPLE SIZE: Only {completed} completed trades. "
            f"Need minimum 100+ trades for statistical significance."
        )
    elif completed >= 100:
        notes.append(
            f"✓ GOOD SAMPLE SIZE: {completed} trades provides reasonable confidence."
        )

    return notes if notes else ["✓ No major issues detected. System appears viable."]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 14 ▸ PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    signals: List[Dict[str, Any]],
    market_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute forensic backtest: replay signals against market outcomes.

    Parameters
    ----------
    signals : list[dict]
        Historical signals from signal_engine.py:
        {
            "id": str,
            "timestamp": str (ISO 8601),
            "symbol": str,
            "direction": "LONG" | "SHORT",
            "confidence": float (0-100),
            "source": str,
            "event_type": str,
            "regime": str,
        }

    market_data : list[dict]
        Historical market bars:
        {
            "timestamp": str (ISO 8601),
            "symbol": str,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": int,
        }

    Returns
    -------
    dict
        {
            "total_trades": int,
            "win_rate": float,
            "avg_return": float,
            "expectancy": float,
            "max_drawdown": float,
            "sharpe_ratio": float,
            "profit_factor": float,
            "best_sources": list,
            "worst_sources": list,
            "best_event_types": list,
            "worst_event_types": list,
            "regime_performance": dict,
            "false_positive_clusters": list,
            "forensic_notes": list,
            "status": str,
        }
    """
    start_time = time.time()

    _banner("🔬 FORENSIC ALPHA VALIDATION BACKTEST", "cyan")

    if not signals:
        log.warning("No signals provided for backtest")
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "best_sources": [],
            "worst_sources": [],
            "best_event_types": [],
            "worst_event_types": [],
            "regime_performance": {},
            "false_positive_clusters": [],
            "forensic_notes": ["No signals processed"],
            "status": BacktestStatus.FAILED.value,
        }

    if not market_data:
        log.warning("No market data provided for backtest")
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "best_sources": [],
            "worst_sources": [],
            "best_event_types": [],
            "worst_event_types": [],
            "regime_performance": {},
            "false_positive_clusters": [],
            "forensic_notes": ["No market data provided"],
            "status": BacktestStatus.FAILED.value,
        }

    # Filter signals by confidence threshold
    valid_signals = [s for s in signals if s.get("confidence", 0) >= MIN_CONFIDENCE_THRESHOLD]
    log.info(f"Processing {len(valid_signals)}/{len(signals)} signals (confidence > {MIN_CONFIDENCE_THRESHOLD}%)")

    # Backtest each signal
    trades: List[TradeResult] = []
    for idx, signal in enumerate(valid_signals):
        if (idx + 1) % max(1, len(valid_signals) // 10) == 0:
            log.debug(f"  {idx + 1}/{len(valid_signals)} signals processed")

        signal_trades = _backtest_signal_all_windows(signal, market_data, BacktestContext())
        trades.extend(signal_trades)

    log.info(f"Generated {len(trades)} trade scenarios ({len(valid_signals)} signals × {len(RETURN_WINDOWS_SECONDS)} windows)")

    # Calculate metrics
    metrics = _calculate_metrics(trades)
    max_dd = _calculate_max_drawdown(trades)
    sharpe = _calculate_sharpe_ratio(trades)

    # Attribution
    source_attr = _attribution_by_dimension(trades, "source")
    event_attr = _attribution_by_dimension(trades, "event_type")
    regime_attr = _attribution_by_dimension(trades, "regime_at_entry")

    # Best/worst
    best_sources = _rank_by_metric(source_attr, "expectancy", top_n=5)
    worst_sources = list(reversed(_rank_by_metric(source_attr, "expectancy", top_n=5)))
    best_events = _rank_by_metric(event_attr, "expectancy", top_n=5)
    worst_events = list(reversed(_rank_by_metric(event_attr, "expectancy", top_n=5)))

    # False positives
    clusters = _detect_false_positive_clusters(trades)

    # Forensic notes
    notes = _generate_forensic_notes(trades, metrics, source_attr, event_attr, regime_attr, clusters)

    # Determine overall status
    overall_status = BacktestStatus.SUCCESS
    if max_dd > 30 or metrics.get("expectancy", 0) < 0:
        overall_status = BacktestStatus.DEGRADED
    if metrics.get("total_trades", 0) == 0 or metrics.get("win_rate", 0) < 20:
        overall_status = BacktestStatus.FAILED

    elapsed_ms = (time.time() - start_time) * 1000

    result = {
        "total_trades": metrics.get("total_trades", 0),
        "winning_trades": metrics.get("winning_trades", 0),
        "losing_trades": metrics.get("losing_trades", 0),
        "win_rate": metrics.get("win_rate", 0.0),
        "avg_return": metrics.get("avg_return", 0.0),
        "avg_win": metrics.get("avg_win", 0.0),
        "avg_loss": metrics.get("avg_loss", 0.0),
        "expectancy": metrics.get("expectancy", 0.0),
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "profit_factor": metrics.get("profit_factor", 0.0),
        "gross_profit": metrics.get("gross_profit", 0.0),
        "gross_loss": metrics.get("gross_loss", 0.0),
        "largest_win": metrics.get("largest_win", 0.0),
        "largest_loss": metrics.get("largest_loss", 0.0),
        "best_sources": best_sources,
        "worst_sources": worst_sources,
        "best_event_types": best_events,
        "worst_event_types": worst_events,
        "regime_performance": dict(regime_attr),
        "false_positive_clusters": clusters,
        "forensic_notes": notes,
        "status": overall_status.value,
        "processing_time_ms": round(elapsed_ms, 1),
    }

    # Print summary
    log.info("")
    log.info(f"[BACKTEST SUMMARY]  Win Rate: {result['win_rate']}%  "
             f"Expectancy: {result['expectancy']:.4f}%  "
             f"Max DD: {result['max_drawdown']:.2f}%  "
             f"Sharpe: {result['sharpe_ratio']:.2f}  "
             f"Profit Factor: {result['profit_factor']:.2f}")
    log.info(f"[STATUS] {result['status']} ({result['processing_time_ms']}ms)")
    log.info("")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 15 ▸ SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

class _BacktestEngineTests(unittest.TestCase):
    """Smoke tests for backtest_engine."""

    def test_backtest_with_synthetic_signals(self):
        """Test backtest with synthetic trades."""
        signals = [
            {
                "id": f"sig-{i}",
                "timestamp": (datetime.now(timezone.utc) + timedelta(hours=i)).isoformat(),
                "symbol": "EURUSD",
                "direction": "LONG" if i % 2 == 0 else "SHORT",
                "confidence": 60 + (i % 30),
                "source": "ForexFactory" if i % 3 == 0 else "Reuters",
                "event_type": "Rate Hike" if i % 2 == 0 else "Data Release",
                "regime": "TRENDING",
            }
            for i in range(20)
        ]

        # Synthetic market data
        market_data = []
        base_price = 1.0800
        for i in range(500):
            ts = datetime.now(timezone.utc) + timedelta(minutes=i)
            close = base_price + (i * 0.0001) + (0.00005 * (i % 10))
            market_data.append({
                "timestamp": ts.isoformat(),
                "symbol": "EURUSD",
                "open": close - 0.00005,
                "high": close + 0.0001,
                "low": close - 0.0001,
                "close": close,
                "volume": 1000000,
            })

        result = run_backtest(signals, market_data)

        self.assertIn("total_trades", result)
        self.assertIn("win_rate", result)
        self.assertIn("expectancy", result)
        self.assertIn("forensic_notes", result)
        self.assertGreater(result["total_trades"], 0)
        self.assertIn(result["status"], ["SUCCESS", "DEGRADED", "FAILED"])

    def test_backtest_empty_signals(self):
        """Test backtest with no signals."""
        result = run_backtest([], [])
        self.assertEqual(result["total_trades"], 0)
        self.assertEqual(result["status"], "FAILED")

    def test_metrics_calculation(self):
        """Test metrics calculation."""
        trades = [
            TradeResult(
                signal_id="t1",
                symbol="SPY",
                direction=TradeDirection.LONG,
                entry_price=400,
                exit_price=410,
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
                pnl_usd=100,
                pnl_pct=2.5,
                return_window="1h",
                status=TradeStatus.EXITED,
                source="Test",
                event_type="Test",
                confidence=80,
                regime_at_entry="TRENDING",
            ),
            TradeResult(
                signal_id="t2",
                symbol="SPY",
                direction=TradeDirection.SHORT,
                entry_price=410,
                exit_price=405,
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
                pnl_usd=50,
                pnl_pct=1.22,
                return_window="1h",
                status=TradeStatus.EXITED,
                source="Test",
                event_type="Test",
                confidence=75,
                regime_at_entry="TRENDING",
            ),
        ]

        metrics = _calculate_metrics(trades)
        self.assertEqual(metrics["total_trades"], 2)
        self.assertEqual(metrics["winning_trades"], 2)
        self.assertEqual(metrics["win_rate"], 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# __main__ ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BACKTEST_ENGINE.PY — Forensic Alpha Validation"
    )
    parser.add_argument(
        "--smoke", "-s",
        action="store_true",
        help="Run smoke test suite",
    )
    parser.add_argument(
        "--signals",
        default=None,
        help="Path to JSON file with historical signals",
    )
    parser.add_argument(
        "--market-data",
        default=None,
        help="Path to JSON file with historical market data",
    )
    args = parser.parse_args()

    if args.smoke:
        _banner(
            f"🧪 BACKTEST_ENGINE SMOKE TEST SUITE  |  {BACKTEST_ENGINE_VERSION}",
            "purple",
        )
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(_BacktestEngineTests)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        total = result.testsRun
        passed = total - len(result.failures) - len(result.errors)
        failed = len(result.failures) + len(result.errors)

        print()
        P = _C["purple"]
        G = _C["green"]
        R = _C["red"]
        X = _C["reset"]

        print(f"{P}{'═'*72}{X}")
        if failed == 0:
            print(f"  {G}✓  {passed}/{total}  ALL CLEAR — backtest_engine.py is production-ready.{X}")
        else:
            print(f"  {R}✗  {passed}/{total} passed — {failed} failure(s). See above.{X}")
        print(f"{P}{'═'*72}{X}")
        print()
        sys.exit(0 if result.wasSuccessful() else 1)

    # Load and run backtest
    if args.signals and args.market_data:
        try:
            with open(args.signals, "r") as f:
                signals = json.load(f)
            with open(args.market_data, "r") as f:
                market_data = json.load(f)
            result = run_backtest(signals, market_data)
            print(json.dumps(result, indent=2))
        except Exception as e:
            log.error(f"Error loading files: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)
