"""
self_learning_optimizer.py
==========================
MONSTER TRADING AI — Institutional Self-Learning Optimization Layer

Pipeline position:
  performance_analytics.py
  → validation_memory.py
  → self_learning_optimizer.py   ← THIS MODULE
  → signal_engine.py (adaptive thresholds)
  → regime_detector.py
  → portfolio_brain.py
  → execution_bridge.py

Deterministic, standard-library-only, O(n) historical pass.
No ML libraries. No numpy/pandas. No randomness.
All outputs bounded by policy ranges.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# =============================================================================
# 1. CONSTANTS + POLICY TUNING RANGES
# =============================================================================

# ── Exponential decay ────────────────────────────────────────────────────────
EWMA_HALFLIFE_DAYS: float = 30.0          # 30-day half-life for recency weighting
DECAY_LAMBDA: float = math.log(2) / EWMA_HALFLIFE_DAYS

# ── Signal confidence thresholds ─────────────────────────────────────────────
THRESHOLD_MIN: float = 0.40               # floor — never go below 40 % confidence
THRESHOLD_MAX: float = 0.90               # ceiling — never demand > 90 %
THRESHOLD_DEFAULT: float = 0.60
THRESHOLD_TIGHTEN_STEP: float = 0.05      # max single-pass tightening
THRESHOLD_LOOSEN_STEP: float = 0.03      # max single-pass loosening

# ── Source credibility weights ────────────────────────────────────────────────
SOURCE_WEIGHT_MIN: float = 0.10
SOURCE_WEIGHT_MAX: float = 2.00
SOURCE_WEIGHT_DEFAULT: float = 1.00

# ── Gross exposure caps (fraction of NAV) ────────────────────────────────────
GROSS_CAP_MIN: float = 0.20
GROSS_CAP_MAX: float = 2.00
GROSS_CAP_DEFAULT: float = 1.20

# ── Hedge ratio bounds ────────────────────────────────────────────────────────
HEDGE_RATIO_MIN: float = 0.00
HEDGE_RATIO_MAX: float = 1.00
HEDGE_RATIO_DEFAULT: float = 0.20

# ── Options premium risk ceiling (fraction of NAV) ────────────────────────────
OPTIONS_CEIL_MIN: float = 0.01
OPTIONS_CEIL_MAX: float = 0.10
OPTIONS_CEIL_DEFAULT: float = 0.03

# ── Crisis escalation sensitivity ────────────────────────────────────────────
CRISIS_SENSITIVITY_MIN: float = 0.50
CRISIS_SENSITIVITY_MAX: float = 2.00
CRISIS_SENSITIVITY_DEFAULT: float = 1.00

# ── Drawdown thresholds ───────────────────────────────────────────────────────
# UNIFIED: read from config.DRAWDOWN_POLICY_TIERS (never hardcode locally)
from config import DRAWDOWN_POLICY_TIERS
# Map policy tiers to helper constants for backward compatibility
DRAWDOWN_WARN: float       = DRAWDOWN_POLICY_TIERS[2][0]      # ~0.005 (0.5%)
DRAWDOWN_REDUCE: float     = DRAWDOWN_POLICY_TIERS[1][0]      # 0.015 (1.5%)
DRAWDOWN_DEFENSIVE: float  = DRAWDOWN_POLICY_TIERS[1][0]      # 0.015 (1.5%)
DRAWDOWN_KILLSWITCH: float = DRAWDOWN_POLICY_TIERS[0][0]      # 0.025 (2.5%)

# ── Hit-rate thresholds ───────────────────────────────────────────────────────
HIT_RATE_STRONG: float = 0.60            # ≥ 60 % → loosen thresholds slightly
HIT_RATE_WEAK: float = 0.45              # < 45 % → tighten + trigger rollback
HIT_RATE_COLLAPSE: float = 0.35          # < 35 % → confidence rollback + safe mode

# ── False-positive penalty ────────────────────────────────────────────────────
FP_RECENT_MULTIPLIER: float = 2.0        # recent FPs count double
FP_HEAVY_THRESHOLD: float = 0.30         # FP-rate > 30 % → heavy penalty
FP_MODERATE_THRESHOLD: float = 0.15      # FP-rate > 15 % → moderate penalty

# ── Regime buckets ────────────────────────────────────────────────────────────
REGIME_LABELS: tuple[str, ...] = (
    "bull_trend",
    "bear_trend",
    "high_volatility",
    "low_volatility",
    "sideways",
    "crisis",
    "recovery",
    "unknown",
)

# ── Safe leverage jump guard ──────────────────────────────────────────────────
MAX_LEVERAGE_JUMP: float = 0.20          # no single-step gross cap increase > 20 %
MAX_HEDGE_JUMP: float = 0.10             # no single-step hedge ratio change > 10 %

# ── Overfitting guard ─────────────────────────────────────────────────────────
MIN_TRADES_FOR_LEARNING: int = 10        # need ≥ 10 trades per regime to learn
MIN_REGIME_SAMPLE_FRACTION: float = 0.05  # regime bucket < 5 % of total → skip

# =============================================================================
# 2. TYPED HELPER STRUCTURES
# =============================================================================

@dataclass
class TradeRecord:
    """
    Normalised representation of a single historical trade.
    Maps directly onto the output contracts of signal_engine.py,
    portfolio_brain.py, and performance_analytics.py.
    """
    trade_id: str
    timestamp: float                   # Unix epoch seconds
    asset: str
    direction: str                     # "long" | "short"
    regime_tag: str                    # from regime_detector.py
    source: str                        # news / signal source identifier
    confidence_score: float            # signal_engine output [0-1]
    signal_strength: float             # signal_engine output [0-1]
    false_positive_flag: bool          # signal_engine output
    pnl: float                         # performance_analytics output
    hit: bool                          # win = True
    gross_exposure: float              # portfolio_brain output
    hedge_allocation: float            # portfolio_brain output
    options_premium_at_risk: float     # portfolio_brain output
    drawdown_at_entry: float           # running max-dd at time of trade
    validation_score: float            # fake_news_validator output [0-1]
    trust_score: float                 # fake_news_validator output [0-1]
    misinformation_risk: float         # fake_news_validator output [0-1]


@dataclass
class RegimeBucket:
    """Accumulated statistics for one market regime."""
    regime: str
    trade_count: int = 0
    weighted_hits: float = 0.0
    weighted_total: float = 0.0
    weighted_pnl: float = 0.0
    weighted_fp: float = 0.0
    weighted_fp_total: float = 0.0
    avg_confidence: float = 0.0
    confidence_sum: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.weighted_total < 1e-9:
            return 0.5
        return self.weighted_hits / self.weighted_total

    @property
    def fp_rate(self) -> float:
        if self.weighted_fp_total < 1e-9:
            return 0.0
        return self.weighted_fp / self.weighted_fp_total

    @property
    def avg_pnl(self) -> float:
        if self.weighted_total < 1e-9:
            return 0.0
        return self.weighted_pnl / self.weighted_total


@dataclass
class SourceAlpha:
    """Per-source accumulated alpha statistics."""
    source: str
    weighted_hits: float = 0.0
    weighted_total: float = 0.0
    weighted_pnl: float = 0.0
    weighted_fp: float = 0.0
    weighted_fp_total: float = 0.0
    recent_fp_penalty: float = 0.0     # extra weight on recent FPs

    @property
    def hit_rate(self) -> float:
        if self.weighted_total < 1e-9:
            return 0.5
        return self.weighted_hits / self.weighted_total

    @property
    def fp_rate(self) -> float:
        if self.weighted_fp_total < 1e-9:
            return 0.0
        return self.weighted_fp / self.weighted_fp_total

    @property
    def alpha_score(self) -> float:
        """Composite alpha: hit-rate adjusted for FP penalty and PnL quality."""
        if self.weighted_total < 1e-9:
            return SOURCE_WEIGHT_DEFAULT
        base = self.hit_rate - (self.fp_rate * FP_RECENT_MULTIPLIER)
        pnl_quality = math.tanh(self.weighted_pnl / max(self.weighted_total, 1.0))
        raw = (base + pnl_quality) / 2.0
        # Map [-1, 1] → [SOURCE_WEIGHT_MIN, SOURCE_WEIGHT_MAX]
        normalised = (raw + 1.0) / 2.0
        return _clamp(
            SOURCE_WEIGHT_MIN + normalised * (SOURCE_WEIGHT_MAX - SOURCE_WEIGHT_MIN),
            SOURCE_WEIGHT_MIN,
            SOURCE_WEIGHT_MAX,
        )


@dataclass
class OptimizerState:
    """Mutable accumulation state during the O(n) historical pass."""
    regime_buckets: dict[str, RegimeBucket] = field(
        default_factory=lambda: {r: RegimeBucket(regime=r) for r in REGIME_LABELS}
    )
    source_alphas: dict[str, SourceAlpha] = field(default_factory=dict)
    total_weighted: float = 0.0
    total_weighted_hits: float = 0.0
    total_weighted_pnl: float = 0.0
    total_fp_weighted: float = 0.0
    total_fp_total: float = 0.0
    recent_drawdown: float = 0.0
    crisis_fp_count: float = 0.0
    crisis_total: float = 0.0
    now_ts: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

# =============================================================================
# 3. UTILITY HELPERS
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if abs(denom) > 1e-12 else default


def _bounded_step(current: float, proposed: float, max_jump: float,
                  lo: float, hi: float) -> float:
    """Apply proposed change but cap the single-step jump and clamp to [lo, hi]."""
    delta = proposed - current
    capped = current + _clamp(delta, -max_jump, max_jump)
    return _clamp(capped, lo, hi)

# =============================================================================
# 4. EXPONENTIAL DECAY ENGINE
# =============================================================================

def decay_weight(trade_ts: float, now_ts: float) -> float:
    """
    Return an exponential decay weight for a trade at `trade_ts`.
    Weight = exp(-λ * age_in_days).
    Recent trades → weight ≈ 1.0.  Old trades → weight → 0.
    """
    age_days = max(0.0, (now_ts - trade_ts) / 86_400.0)
    return math.exp(-DECAY_LAMBDA * age_days)


def recent_penalty_multiplier(trade_ts: float, now_ts: float,
                               recency_window_days: float = 7.0) -> float:
    """
    Extra penalty multiplier for false positives that occurred within
    `recency_window_days`.  Returns FP_RECENT_MULTIPLIER if recent, else 1.0.
    """
    age_days = (now_ts - trade_ts) / 86_400.0
    return FP_RECENT_MULTIPLIER if age_days <= recency_window_days else 1.0

# =============================================================================
# 5. EWMA PERFORMANCE SCORER
# =============================================================================

def ewma_score(values: list[tuple[float, float]]) -> float:
    """
    Compute exponentially-weighted moving average.
    `values` = list of (weight, value) tuples already decay-weighted.
    Returns the weighted mean.  O(n).
    """
    total_w = sum(w for w, _ in values)
    if total_w < 1e-12:
        return 0.0
    return sum(w * v for w, v in values) / total_w


def ewma_hit_rate(trades: list[TradeRecord], now_ts: float) -> float:
    """Overall EWMA hit-rate across all trades."""
    pairs = [(decay_weight(t.timestamp, now_ts), 1.0 if t.hit else 0.0)
             for t in trades]
    return ewma_score(pairs)


def ewma_sharpe_proxy(trades: list[TradeRecord], now_ts: float) -> float:
    """
    Lightweight Sharpe proxy: EWMA(PnL) / std(PnL).
    Uses population stddev over the decay-weighted sample.
    """
    weighted_pnls = [(decay_weight(t.timestamp, now_ts), t.pnl) for t in trades]
    if not weighted_pnls:
        return 0.0
    mean_pnl = ewma_score(weighted_pnls)
    variance_pairs = [(w, (v - mean_pnl) ** 2) for w, v in weighted_pnls]
    variance = ewma_score(variance_pairs)
    std = math.sqrt(max(variance, 1e-12))
    return mean_pnl / std

# =============================================================================
# 6. FALSE-POSITIVE SUPPRESSION ENGINE
# =============================================================================

def compute_fp_suppression_factor(fp_rate: float,
                                  recent_fp_penalty: float) -> float:
    """
    Map a source/regime false-positive rate to a suppression multiplier
    applied to that source's threshold recommendation.

    fp_rate:            weighted FP fraction [0-1]
    recent_fp_penalty:  extra penalty accumulated from recent window

    Returns a factor in [0.5, 1.0]:
      1.0 = no suppression needed
      0.5 = maximum suppression (threshold +25 % tighter)
    """
    base_penalty = 0.0
    if fp_rate > FP_HEAVY_THRESHOLD:
        base_penalty = 0.5
    elif fp_rate > FP_MODERATE_THRESHOLD:
        base_penalty = 0.25

    # Recent FPs add proportional extra suppression
    recent_adj = _clamp(recent_fp_penalty * 0.10, 0.0, 0.25)
    raw_factor = 1.0 - base_penalty - recent_adj
    return _clamp(raw_factor, 0.50, 1.00)

# =============================================================================
# 7. REGIME BUCKET LEARNER  (O(n) single pass, called inside main loop)
# =============================================================================

def accumulate_regime(bucket: RegimeBucket, trade: TradeRecord,
                      w: float, fp_penalty_w: float) -> None:
    """Update a RegimeBucket with one trade's decay-weighted contribution."""
    bucket.trade_count += 1
    bucket.weighted_total += w
    bucket.weighted_hits += w * (1.0 if trade.hit else 0.0)
    bucket.weighted_pnl += w * trade.pnl
    bucket.weighted_fp += fp_penalty_w * (1.0 if trade.false_positive_flag else 0.0)
    bucket.weighted_fp_total += fp_penalty_w
    bucket.confidence_sum += w * trade.confidence_score


def derive_regime_gross_cap(bucket: RegimeBucket,
                             current_cap: float,
                             regime_history: list[dict]) -> float:
    """
    Recommend a new gross-cap for a regime based on its EWMA hit-rate and PnL.
    Conservative: only step up if evidence is strong; step down quickly.
    """
    if bucket.trade_count < MIN_TRADES_FOR_LEARNING:
        return current_cap

    hr = bucket.hit_rate
    avg_pnl = bucket.avg_pnl

    if hr >= HIT_RATE_STRONG and avg_pnl > 0:
        # Evidence of edge → allow modest cap increase
        proposed = current_cap + 0.05
    elif hr < HIT_RATE_WEAK or avg_pnl < 0:
        # Poor performance → reduce cap
        reduction = 0.10 if hr < HIT_RATE_COLLAPSE else 0.05
        proposed = current_cap - reduction
    else:
        proposed = current_cap

    return _bounded_step(current_cap, proposed, MAX_LEVERAGE_JUMP,
                         GROSS_CAP_MIN, GROSS_CAP_MAX)


def derive_regime_hedge_ratio(bucket: RegimeBucket,
                               current_ratio: float) -> float:
    """
    Recommend a new hedge ratio for a regime.
    Crisis / bear / high-vol regimes get higher floors.
    """
    if bucket.trade_count < MIN_TRADES_FOR_LEARNING:
        return current_ratio

    crisis_regimes = {"crisis", "bear_trend", "high_volatility"}
    is_risky = bucket.regime in crisis_regimes

    if bucket.hit_rate < HIT_RATE_WEAK or bucket.avg_pnl < 0:
        step = 0.05 if not is_risky else 0.08
        proposed = current_ratio + step
    elif bucket.hit_rate >= HIT_RATE_STRONG and bucket.avg_pnl > 0 and not is_risky:
        proposed = current_ratio - 0.03
    else:
        proposed = current_ratio

    return _bounded_step(current_ratio, proposed, MAX_HEDGE_JUMP,
                         HEDGE_RATIO_MIN, HEDGE_RATIO_MAX)

# =============================================================================
# 8. SOURCE ALPHA LEADERBOARD
# =============================================================================

def accumulate_source(state: OptimizerState, trade: TradeRecord,
                      w: float, fp_penalty_w: float, now_ts: float) -> None:
    """Update SourceAlpha for a single trade."""
    src = trade.source
    if src not in state.source_alphas:
        state.source_alphas[src] = SourceAlpha(source=src)
    sa = state.source_alphas[src]

    sa.weighted_total += w
    sa.weighted_hits += w * (1.0 if trade.hit else 0.0)
    sa.weighted_pnl += w * trade.pnl
    sa.weighted_fp += fp_penalty_w * (1.0 if trade.false_positive_flag else 0.0)
    sa.weighted_fp_total += fp_penalty_w

    if trade.false_positive_flag:
        rp = recent_penalty_multiplier(trade.timestamp, now_ts)
        sa.recent_fp_penalty += rp - 1.0   # accumulate only the extra portion


def build_source_alpha_scores(state: OptimizerState) -> dict[str, float]:
    """Return {source: alpha_score} dict, bounded to [SOURCE_WEIGHT_MIN, SOURCE_WEIGHT_MAX]."""
    return {src: sa.alpha_score for src, sa in state.source_alphas.items()}


def worst_fp_source(state: OptimizerState) -> str:
    """Return the source identifier with the highest weighted false-positive rate."""
    if not state.source_alphas:
        return "unknown"
    return max(state.source_alphas.values(), key=lambda sa: sa.fp_rate).source

# =============================================================================
# 9. THRESHOLD OPTIMIZER
# =============================================================================

def compute_adaptive_thresholds(
    state: OptimizerState,
    threshold_config: dict,
    global_hit_rate: float,
    validation_memory: dict,
) -> dict[str, float]:
    """
    Return per-regime adaptive confidence thresholds.

    Logic:
    - Base threshold from threshold_config (or DEFAULT).
    - Adjust upward if regime FP-rate is high or hit-rate is collapsing.
    - Adjust downward if regime has strong hit-rate and positive PnL.
    - Apply FP suppression factor.
    - Cap single-step changes.
    - Confidence rollback if recent global hit-rate collapses.
    """
    thresholds: dict[str, float] = {}
    rollback_mode = global_hit_rate < HIT_RATE_COLLAPSE

    for regime, bucket in state.regime_buckets.items():
        base = threshold_config.get(f"threshold_{regime}", THRESHOLD_DEFAULT)

        if rollback_mode:
            # Confidence rollback: tighten all thresholds by max step
            proposed = base + THRESHOLD_TIGHTEN_STEP
        elif bucket.trade_count < MIN_TRADES_FOR_LEARNING:
            proposed = base
        else:
            fp_suppression = compute_fp_suppression_factor(
                bucket.fp_rate, 0.0
            )
            if bucket.hit_rate >= HIT_RATE_STRONG and bucket.avg_pnl > 0:
                proposed = base - (THRESHOLD_LOOSEN_STEP * fp_suppression)
            elif bucket.hit_rate < HIT_RATE_WEAK or bucket.avg_pnl < 0:
                proposed = base + (THRESHOLD_TIGHTEN_STEP / fp_suppression)
            else:
                proposed = base

        # Apply validation_memory: if threshold_change_history shows this
        # regime has been repeatedly tightened, don't loosen too fast
        history = validation_memory.get("threshold_change_history", {})
        if isinstance(history, dict):
            tighten_count = history.get(f"{regime}_tighten_count", 0)
            if tighten_count >= 3 and proposed < base:
                # Require 2 × the normal evidence before loosening
                proposed = (proposed + base) / 2.0

        thresholds[regime] = _clamp(
            _bounded_step(base, proposed, THRESHOLD_TIGHTEN_STEP,
                          THRESHOLD_MIN, THRESHOLD_MAX),
            THRESHOLD_MIN, THRESHOLD_MAX
        )

    return thresholds

# =============================================================================
# 10. GROSS CAP OPTIMIZER
# =============================================================================

def compute_gross_caps(
    state: OptimizerState,
    threshold_config: dict,
    recent_drawdown: float,
) -> dict[str, float]:
    """
    Return per-regime recommended gross exposure caps.
    Drawdown guard: at DRAWDOWN_KILLSWITCH, all caps floor to GROSS_CAP_MIN.
    """
    caps: dict[str, float] = {}

    # Kill-switch: deep drawdown → set every regime to minimum cap
    killswitch = recent_drawdown <= DRAWDOWN_KILLSWITCH

    for regime, bucket in state.regime_buckets.items():
        current = threshold_config.get(f"gross_cap_{regime}", GROSS_CAP_DEFAULT)

        if killswitch:
            caps[regime] = GROSS_CAP_MIN
        else:
            new_cap = derive_regime_gross_cap(bucket, current,
                                              regime_history=[])
            # Additional drawdown-aware reduction
            if recent_drawdown <= DRAWDOWN_DEFENSIVE:
                new_cap = min(new_cap, 0.60)
            elif recent_drawdown <= DRAWDOWN_REDUCE:
                new_cap = min(new_cap, 0.90)
            elif recent_drawdown <= DRAWDOWN_WARN:
                new_cap = min(new_cap, 1.10)

            caps[regime] = new_cap

    return caps

# =============================================================================
# 11. HEDGE RATIO OPTIMIZER
# =============================================================================

def compute_hedge_ratios(
    state: OptimizerState,
    threshold_config: dict,
    recent_drawdown: float,
    global_fp_rate: float,
) -> dict[str, float]:
    """
    Return per-regime recommended hedge ratios.
    """
    ratios: dict[str, float] = {}

    for regime, bucket in state.regime_buckets.items():
        current = threshold_config.get(f"hedge_ratio_{regime}", HEDGE_RATIO_DEFAULT)

        new_ratio = derive_regime_hedge_ratio(bucket, current)

        # Drawdown overlay: severe drawdown → force higher hedging
        if recent_drawdown <= DRAWDOWN_KILLSWITCH:
            new_ratio = max(new_ratio, 0.80)
        elif recent_drawdown <= DRAWDOWN_DEFENSIVE:
            new_ratio = max(new_ratio, 0.50)
        elif recent_drawdown <= DRAWDOWN_REDUCE:
            new_ratio = max(new_ratio, 0.30)

        # High global FP rate → also hedge more
        if global_fp_rate > FP_HEAVY_THRESHOLD:
            new_ratio = max(new_ratio, HEDGE_RATIO_DEFAULT + 0.10)

        ratios[regime] = _clamp(new_ratio, HEDGE_RATIO_MIN, HEDGE_RATIO_MAX)

    return ratios

# =============================================================================
# 12. OPTIONS PREMIUM RISK OPTIMIZER
# =============================================================================

def compute_options_risk_adjustments(
    state: OptimizerState,
    threshold_config: dict,
    recent_drawdown: float,
    global_hit_rate: float,
) -> dict[str, float]:
    """
    Return per-regime recommended options premium risk ceilings
    (as fraction of NAV).

    Logic:
    - In crisis / high-vol regimes with poor hit-rate → lower ceiling.
    - In bull_trend with strong hit-rate → allow slightly higher ceiling.
    - Drawdown guard applied.
    """
    adjustments: dict[str, float] = {}
    defensive_regimes = {"crisis", "high_volatility", "bear_trend"}

    for regime, bucket in state.regime_buckets.items():
        current = threshold_config.get(
            f"options_risk_ceiling_{regime}", OPTIONS_CEIL_DEFAULT
        )

        if recent_drawdown <= DRAWDOWN_KILLSWITCH:
            proposed = OPTIONS_CEIL_MIN
        elif regime in defensive_regimes:
            if bucket.trade_count >= MIN_TRADES_FOR_LEARNING:
                if bucket.hit_rate < HIT_RATE_WEAK:
                    proposed = max(OPTIONS_CEIL_MIN, current - 0.005)
                else:
                    proposed = current
            else:
                proposed = current
        else:
            if (bucket.trade_count >= MIN_TRADES_FOR_LEARNING
                    and bucket.hit_rate >= HIT_RATE_STRONG
                    and bucket.avg_pnl > 0
                    and global_hit_rate >= HIT_RATE_STRONG):
                proposed = min(OPTIONS_CEIL_MAX, current + 0.002)
            else:
                proposed = current

        adjustments[regime] = _clamp(proposed, OPTIONS_CEIL_MIN, OPTIONS_CEIL_MAX)

    return adjustments

# =============================================================================
# 13. CRISIS ESCALATION TUNER
# =============================================================================

def compute_crisis_sensitivity(
    state: OptimizerState,
    threshold_config: dict,
    validation_memory: dict,
) -> dict[str, float]:
    """
    Tune crisis escalation sensitivity.

    Overfitting guard:
    - If crisis_trigger_accuracy (from validation_memory) is high but
      crisis false-alarm rate is also high → do NOT increase sensitivity further.
    - If there have been many false alarms in crisis regime → lower sensitivity
      slightly to reduce unnecessary de-risking costs.
    """
    current = threshold_config.get("crisis_sensitivity", CRISIS_SENSITIVITY_DEFAULT)

    crisis_accuracy = validation_memory.get("crisis_trigger_accuracy", 0.5)
    if not isinstance(crisis_accuracy, (int, float)):
        crisis_accuracy = 0.5

    crisis_bucket = state.regime_buckets.get("crisis", RegimeBucket(regime="crisis"))
    crisis_fp = crisis_bucket.fp_rate

    # Overfitting prevention: if recent crisis FP > 30 % despite accuracy signal,
    # don't amplify sensitivity
    if crisis_fp > FP_HEAVY_THRESHOLD:
        # False alarms are expensive: reduce sensitivity slightly
        proposed = current - 0.10
    elif crisis_accuracy >= 0.70 and crisis_fp <= FP_MODERATE_THRESHOLD:
        # Good accuracy + low FP → can sharpen sensitivity
        proposed = current + 0.10
    elif crisis_accuracy < 0.40:
        # Poor accuracy → desensitise to avoid false alarms
        proposed = current - 0.15
    else:
        proposed = current

    bounded = _clamp(proposed, CRISIS_SENSITIVITY_MIN, CRISIS_SENSITIVITY_MAX)
    return {"crisis_sensitivity_multiplier": bounded}

# =============================================================================
# 14. DRAWDOWN RESPONSE TUNER
# =============================================================================

def compute_drawdown_response_curve(
    state: OptimizerState,
    threshold_config: dict,
    recent_drawdown: float,
    global_hit_rate: float,
) -> dict[str, Any]:
    """
    Return a drawdown response curve: at each drawdown level, what actions
    should the system take?

    The curve adapts based on:
    - How well the system has performed in drawdown recovery historically
    - Current regime hit-rate (poor hit-rate → more aggressive de-risking)
    """
    recovery_bucket = state.regime_buckets.get("recovery", RegimeBucket(regime="recovery"))
    crisis_bucket = state.regime_buckets.get("crisis", RegimeBucket(regime="crisis"))

    # Recovery quality: can we trust the system to re-risk after drawdown?
    recovery_confidence = recovery_bucket.hit_rate if recovery_bucket.trade_count >= MIN_TRADES_FOR_LEARNING else 0.5

    # Capital preservation mode trigger
    capital_preservation = (
        recent_drawdown <= DRAWDOWN_KILLSWITCH
        or global_hit_rate < HIT_RATE_COLLAPSE
        or crisis_bucket.fp_rate > FP_HEAVY_THRESHOLD
    )

    # Drawdown response speed: how quickly to re-risk after recovery?
    # If recovery hit-rate is strong → faster re-risk; else cautious
    rerisk_speed = "fast" if recovery_confidence >= HIT_RATE_STRONG else (
        "moderate" if recovery_confidence >= HIT_RATE_WEAK else "slow"
    )

    curve = {
        "warn_threshold": DRAWDOWN_WARN,
        "reduce_threshold": DRAWDOWN_REDUCE,
        "defensive_threshold": DRAWDOWN_DEFENSIVE,
        "killswitch_threshold": DRAWDOWN_KILLSWITCH,
        "current_drawdown": recent_drawdown,
        "active_level": _drawdown_level(recent_drawdown),
        "capital_preservation_mode": capital_preservation,
        "rerisk_speed": rerisk_speed,
        "recovery_regime_hit_rate": round(recovery_confidence, 4),
        "recommended_actions": _drawdown_actions(recent_drawdown, capital_preservation),
    }
    return curve


def _drawdown_level(dd: float) -> str:
    if dd <= DRAWDOWN_KILLSWITCH:
        return "KILLSWITCH"
    if dd <= DRAWDOWN_DEFENSIVE:
        return "DEFENSIVE"
    if dd <= DRAWDOWN_REDUCE:
        return "REDUCE"
    if dd <= DRAWDOWN_WARN:
        return "WARN"
    return "NORMAL"


def _drawdown_actions(dd: float, capital_preservation: bool) -> list[str]:
    actions: list[str] = []
    if capital_preservation:
        actions.append("Activate full capital preservation mode — all gross caps to minimum.")
        actions.append("Halt new position entries until drawdown recovery confirmed.")
        actions.append("Increase all hedge ratios to maximum policy floor (80%).")
    elif dd <= DRAWDOWN_DEFENSIVE:
        actions.append("Reduce gross exposure to 60% cap across all regimes.")
        actions.append("Elevate hedge ratios to 50% minimum.")
        actions.append("Suspend options premium risk — ceiling at minimum.")
    elif dd <= DRAWDOWN_REDUCE:
        actions.append("Reduce gross exposure cap to 90%.")
        actions.append("Increase hedge ratio to 30% minimum.")
    elif dd <= DRAWDOWN_WARN:
        actions.append("Cap gross exposure at 110%.")
        actions.append("Monitor regime hit-rates closely — tighten on further deterioration.")
    else:
        actions.append("Normal operations — continue monitoring.")
    return actions

# =============================================================================
# 15. EXPLAINABILITY SUMMARY BUILDER
# =============================================================================

def build_explanations(
    state: OptimizerState,
    adaptive_thresholds: dict[str, float],
    gross_caps: dict[str, float],
    hedge_ratios: dict[str, float],
    crisis_adj: dict[str, float],
    drawdown_curve: dict[str, Any],
    global_hit_rate: float,
    global_fp_rate: float,
    threshold_config: dict,
) -> list[str]:
    """Generate plain-English explanations for all recommendations."""
    explanations: list[str] = []

    # ── Global hit-rate commentary ────────────────────────────────────────────
    if global_hit_rate < HIT_RATE_COLLAPSE:
        explanations.append(
            f"CRITICAL: Global hit-rate has collapsed to {global_hit_rate:.1%}. "
            "Confidence rollback activated — all thresholds tightened to maximum step. "
            "Do not loosen thresholds until hit-rate recovers above 45%."
        )
    elif global_hit_rate < HIT_RATE_WEAK:
        explanations.append(
            f"WARNING: Global hit-rate ({global_hit_rate:.1%}) is below the weak threshold "
            f"({HIT_RATE_WEAK:.0%}). Tightening signal confidence thresholds system-wide."
        )
    elif global_hit_rate >= HIT_RATE_STRONG:
        explanations.append(
            f"System performing well — global hit-rate at {global_hit_rate:.1%}. "
            "Modest threshold relaxation permitted in high-performing regimes."
        )

    # ── False-positive commentary ─────────────────────────────────────────────
    if global_fp_rate > FP_HEAVY_THRESHOLD:
        explanations.append(
            f"High false-positive rate ({global_fp_rate:.1%} > {FP_HEAVY_THRESHOLD:.0%}). "
            "Source credibility weights penalised. Recent FPs weighted double."
        )

    # ── Per-regime threshold changes ──────────────────────────────────────────
    for regime, new_thresh in adaptive_thresholds.items():
        old = threshold_config.get(f"threshold_{regime}", THRESHOLD_DEFAULT)
        delta = new_thresh - old
        bucket = state.regime_buckets[regime]
        if abs(delta) > 1e-4 and bucket.trade_count >= MIN_TRADES_FOR_LEARNING:
            direction = "tightened" if delta > 0 else "loosened"
            explanations.append(
                f"Regime '{regime}': Confidence threshold {direction} from "
                f"{old:.2f} → {new_thresh:.2f} "
                f"(hit-rate={bucket.hit_rate:.1%}, FP-rate={bucket.fp_rate:.1%}, "
                f"avg-PnL={bucket.avg_pnl:.4f}, n={bucket.trade_count})."
            )

    # ── Gross cap changes ─────────────────────────────────────────────────────
    for regime, new_cap in gross_caps.items():
        old = threshold_config.get(f"gross_cap_{regime}", GROSS_CAP_DEFAULT)
        delta = new_cap - old
        if abs(delta) > 1e-4:
            direction = "increased" if delta > 0 else "reduced"
            explanations.append(
                f"Regime '{regime}': Gross exposure cap {direction} "
                f"from {old:.2f}x → {new_cap:.2f}x NAV."
            )

    # ── Hedge ratio changes ───────────────────────────────────────────────────
    for regime, new_ratio in hedge_ratios.items():
        old = threshold_config.get(f"hedge_ratio_{regime}", HEDGE_RATIO_DEFAULT)
        delta = new_ratio - old
        if abs(delta) > 1e-4:
            direction = "raised" if delta > 0 else "reduced"
            explanations.append(
                f"Regime '{regime}': Hedge ratio {direction} "
                f"from {old:.2f} → {new_ratio:.2f}."
            )

    # ── Crisis sensitivity ────────────────────────────────────────────────────
    cs = crisis_adj.get("crisis_sensitivity_multiplier", CRISIS_SENSITIVITY_DEFAULT)
    old_cs = threshold_config.get("crisis_sensitivity", CRISIS_SENSITIVITY_DEFAULT)
    if abs(cs - old_cs) > 1e-4:
        direction = "increased" if cs > old_cs else "reduced"
        explanations.append(
            f"Crisis escalation sensitivity {direction} from {old_cs:.2f} → {cs:.2f}. "
            "Overfitting guard applied — false-alarm suppression active."
        )

    # ── Drawdown level commentary ─────────────────────────────────────────────
    level = drawdown_curve.get("active_level", "NORMAL")
    dd_val = drawdown_curve.get("current_drawdown", 0.0)
    if level != "NORMAL":
        explanations.append(
            f"Drawdown alert — current drawdown: {dd_val:.1%}. "
            f"Response level: {level}. "
            f"Re-risk speed when recovered: {drawdown_curve.get('rerisk_speed', 'moderate')}."
        )

    return explanations

# =============================================================================
# 15b. SUMMARY BUILDER HELPERS
# =============================================================================

def top_winning_regime(state: OptimizerState) -> str:
    """Return regime with highest EWMA hit-rate (minimum sample enforced)."""
    eligible = [
        b for b in state.regime_buckets.values()
        if b.trade_count >= MIN_TRADES_FOR_LEARNING
    ]
    if not eligible:
        return "unknown (insufficient data)"
    best = max(eligible, key=lambda b: b.hit_rate)
    return f"{best.regime} (hit-rate={best.hit_rate:.1%}, n={best.trade_count})"


def recommended_signal_tightening(adaptive_thresholds: dict[str, float],
                                   threshold_config: dict) -> float:
    """
    Return the average tightening across all regimes as a single scalar.
    Positive = net tightening; negative = net loosening.
    """
    deltas = []
    for regime, new_t in adaptive_thresholds.items():
        old = threshold_config.get(f"threshold_{regime}", THRESHOLD_DEFAULT)
        deltas.append(new_t - old)
    return round(statistics.mean(deltas) if deltas else 0.0, 4)

# =============================================================================
# 16. O(n) SINGLE HISTORICAL PASS — ACCUMULATION LOOP
# =============================================================================

def _accumulate(
    historical_trades: list[dict],
    state: OptimizerState,
) -> None:
    """
    Single O(n) pass over all historical trades.
    Converts raw dicts to TradeRecord, applies decay weights, and
    accumulates into regime buckets, source alphas, and global counters.
    """
    now_ts = state.now_ts

    for raw in historical_trades:
        trade = _parse_trade(raw)

        w = decay_weight(trade.timestamp, now_ts)
        rp_mult = recent_penalty_multiplier(trade.timestamp, now_ts)
        fp_penalty_w = w * (rp_mult if trade.false_positive_flag else 1.0)

        # ── Global accumulators ───────────────────────────────────────────────
        state.total_weighted += w
        state.total_weighted_hits += w * (1.0 if trade.hit else 0.0)
        state.total_weighted_pnl += w * trade.pnl
        state.total_fp_weighted += fp_penalty_w * (1.0 if trade.false_positive_flag else 0.0)
        state.total_fp_total += fp_penalty_w

        # Track most severe recent drawdown
        if trade.drawdown_at_entry < state.recent_drawdown:
            state.recent_drawdown = trade.drawdown_at_entry

        # ── Regime bucket ─────────────────────────────────────────────────────
        regime = trade.regime_tag if trade.regime_tag in state.regime_buckets else "unknown"
        accumulate_regime(state.regime_buckets[regime], trade, w, fp_penalty_w)

        # ── Source alpha ──────────────────────────────────────────────────────
        accumulate_source(state, trade, w, fp_penalty_w, now_ts)

        # ── Crisis tracking ───────────────────────────────────────────────────
        if regime == "crisis":
            state.crisis_total += w
            if trade.false_positive_flag:
                state.crisis_fp_count += fp_penalty_w


def _parse_trade(raw: dict) -> TradeRecord:
    """
    Safely parse a raw dict into a TradeRecord, applying defaults
    for any missing fields.  Matches the output contracts of all
    upstream modules.
    """
    ts = raw.get("timestamp")
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts).timestamp()
        except ValueError:
            ts = datetime.now(timezone.utc).timestamp()
    elif not isinstance(ts, (int, float)):
        ts = datetime.now(timezone.utc).timestamp()

    return TradeRecord(
        trade_id=str(raw.get("trade_id", "")),
        timestamp=float(ts),
        asset=str(raw.get("asset", "unknown")),
        direction=str(raw.get("direction", "long")),
        regime_tag=str(raw.get("regime_tag", "unknown")),
        source=str(raw.get("source", "unknown")),
        confidence_score=float(raw.get("confidence_score", 0.5)),
        signal_strength=float(raw.get("signal_strength", 0.5)),
        false_positive_flag=bool(raw.get("false_positive_flag", False)),
        pnl=float(raw.get("pnl", 0.0)),
        hit=bool(raw.get("hit", raw.get("pnl", 0.0) > 0)),
        gross_exposure=float(raw.get("gross_exposure", 1.0)),
        hedge_allocation=float(raw.get("hedge_allocation", 0.2)),
        options_premium_at_risk=float(raw.get("options_premium_at_risk", 0.0)),
        drawdown_at_entry=float(raw.get("drawdown_at_entry", 0.0)),
        validation_score=float(raw.get("validation_score", 1.0)),
        trust_score=float(raw.get("trust_score", 1.0)),
        misinformation_risk=float(raw.get("misinformation_risk", 0.0)),
    )

# =============================================================================
# 15. PUBLIC API — optimize_learning()
# =============================================================================

def optimize_learning(
    historical_trades: list[dict],
    signal_history: list[dict],
    regime_history: list[dict],
    source_outcomes: dict,
    threshold_config: dict,
    validation_memory: dict | None = None,
) -> dict[str, Any]:
    """
    Main entry point for the self-learning optimizer.

    Parameters
    ----------
    historical_trades : list[dict]
        Each dict maps to the output contracts of signal_engine.py,
        portfolio_brain.py, performance_analytics.py.
    signal_history : list[dict]
        Past signal records from signal_engine.py (used for regime cross-ref).
    regime_history : list[dict]
        Past regime snapshots from regime_detector.py.
    source_outcomes : dict
        Aggregated source statistics from validation_memory.py.
    threshold_config : dict
        Current policy thresholds (baseline for bounded adjustments).
    validation_memory : dict | None
        Output of validation_memory.py — optional but recommended.

    Returns
    -------
    dict conforming to the OUTPUT CONTRACT defined in the module spec.
    """
    if validation_memory is None:
        validation_memory = {}

    # ── Build mutable state ───────────────────────────────────────────────────
    state = OptimizerState()

    # ── O(n) accumulation pass ────────────────────────────────────────────────
    _accumulate(historical_trades, state)

    # ── Derived global metrics ────────────────────────────────────────────────
    global_hit_rate = _safe_div(
        state.total_weighted_hits, state.total_weighted, 0.5
    )
    global_fp_rate = _safe_div(
        state.total_fp_weighted, state.total_fp_total, 0.0
    )
    recent_drawdown = state.recent_drawdown  # most severe in dataset

    # ── Component optimizers ──────────────────────────────────────────────────
    adaptive_thresholds = compute_adaptive_thresholds(
        state, threshold_config, global_hit_rate, validation_memory
    )
    source_alpha_scores = build_source_alpha_scores(state)
    recommended_gross_caps = compute_gross_caps(
        state, threshold_config, recent_drawdown
    )
    recommended_hedge_ratios = compute_hedge_ratios(
        state, threshold_config, recent_drawdown, global_fp_rate
    )
    options_risk_adjustments = compute_options_risk_adjustments(
        state, threshold_config, recent_drawdown, global_hit_rate
    )
    crisis_sensitivity_adjustment = compute_crisis_sensitivity(
        state, threshold_config, validation_memory
    )
    drawdown_response_curve = compute_drawdown_response_curve(
        state, threshold_config, recent_drawdown, global_hit_rate
    )

    # ── Capital preservation mode ─────────────────────────────────────────────
    capital_preservation = drawdown_response_curve["capital_preservation_mode"]

    # ── Explanations ──────────────────────────────────────────────────────────
    explanations = build_explanations(
        state, adaptive_thresholds, recommended_gross_caps,
        recommended_hedge_ratios, crisis_sensitivity_adjustment,
        drawdown_response_curve, global_hit_rate, global_fp_rate,
        threshold_config,
    )

    # ── Optimizer summary ─────────────────────────────────────────────────────
    optimizer_summary = {
        "top_winning_regime": top_winning_regime(state),
        "worst_false_positive_source": worst_fp_source(state),
        "recommended_signal_tightening": recommended_signal_tightening(
            adaptive_thresholds, threshold_config
        ),
        "capital_preservation_mode": capital_preservation,
        "explanations": explanations,
    }

    return {
        "adaptive_thresholds": adaptive_thresholds,
        "source_alpha_scores": source_alpha_scores,
        "recommended_gross_caps": recommended_gross_caps,
        "recommended_hedge_ratios": recommended_hedge_ratios,
        "options_risk_adjustments": options_risk_adjustments,
        "drawdown_response_curve": drawdown_response_curve,
        "crisis_sensitivity_adjustment": crisis_sensitivity_adjustment,
        "optimizer_summary": optimizer_summary,
    }

# =============================================================================
# 16. SMOKE TESTS + CLI DEMO
# =============================================================================

def _make_smoke_trades(n: int = 80) -> list[dict]:
    """
    Generate a deterministic set of synthetic trade records for smoke testing.
    Uses integer arithmetic seeded from index — no randomness.
    """
    regimes = list(REGIME_LABELS)
    sources = ["reuters", "bloomberg", "wsj", "twitter", "sec_filing", "internal_model"]
    trades = []
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()

    for i in range(n):
        day_offset = i * 86_400 * 3       # trade every 3 days
        regime = regimes[i % len(regimes)]
        source = sources[i % len(sources)]

        # Deterministic synthetic signals
        hit = (i % 3) != 0                # ~67 % hit-rate overall
        fp = (i % 7) == 0                 # ~14 % FP rate
        pnl = 0.008 if hit else -0.004
        drawdown = -0.005 * (i % 5)       # cycles through mild drawdowns

        trades.append({
            "trade_id": f"T{i:04d}",
            "timestamp": base_ts + day_offset,
            "asset": f"ASSET_{i % 10}",
            "direction": "long" if i % 2 == 0 else "short",
            "regime_tag": regime,
            "source": source,
            "confidence_score": 0.50 + (i % 10) * 0.04,
            "signal_strength": 0.40 + (i % 8) * 0.05,
            "false_positive_flag": fp,
            "pnl": pnl,
            "hit": hit,
            "gross_exposure": 1.0 + (i % 5) * 0.10,
            "hedge_allocation": 0.10 + (i % 4) * 0.05,
            "options_premium_at_risk": 0.01 + (i % 3) * 0.005,
            "drawdown_at_entry": drawdown,
            "validation_score": 0.70 + (i % 5) * 0.05,
            "trust_score": 0.65 + (i % 6) * 0.05,
            "misinformation_risk": 0.05 + (i % 4) * 0.03,
        })

    # Inject a recent FP cluster (last 5 trades, 7-day window)
    now_ts = datetime.now(timezone.utc).timestamp()
    for j in range(5):
        trades.append({
            "trade_id": f"RECENT_FP_{j}",
            "timestamp": now_ts - j * 86_400,   # within last 7 days
            "asset": "ASSET_X",
            "direction": "long",
            "regime_tag": "crisis",
            "source": "twitter",
            "confidence_score": 0.62,
            "signal_strength": 0.55,
            "false_positive_flag": True,
            "pnl": -0.010,
            "hit": False,
            "gross_exposure": 1.30,
            "hedge_allocation": 0.15,
            "options_premium_at_risk": 0.025,
            "drawdown_at_entry": -0.08,
            "validation_score": 0.45,
            "trust_score": 0.40,
            "misinformation_risk": 0.35,
        })

    return trades


def _run_smoke_tests() -> None:
    """Run a suite of deterministic smoke tests and print results."""
    import json

    print("=" * 70)
    print("MONSTER TRADING AI — self_learning_optimizer.py  SMOKE TESTS")
    print("=" * 70)

    # ── Test 1: Basic run with synthetic trades ────────────────────────────────
    print("\n[TEST 1] Basic optimization run with 85 synthetic trades...")
    trades = _make_smoke_trades(80)

    threshold_config = {
        # Signal thresholds
        "threshold_bull_trend": 0.55,
        "threshold_bear_trend": 0.65,
        "threshold_high_volatility": 0.70,
        "threshold_low_volatility": 0.55,
        "threshold_sideways": 0.60,
        "threshold_crisis": 0.75,
        "threshold_recovery": 0.58,
        "threshold_unknown": 0.60,
        # Gross caps
        "gross_cap_bull_trend": 1.40,
        "gross_cap_bear_trend": 0.80,
        "gross_cap_high_volatility": 0.70,
        "gross_cap_low_volatility": 1.20,
        "gross_cap_sideways": 1.00,
        "gross_cap_crisis": 0.40,
        "gross_cap_recovery": 0.90,
        "gross_cap_unknown": 1.00,
        # Hedge ratios
        "hedge_ratio_bull_trend": 0.10,
        "hedge_ratio_bear_trend": 0.40,
        "hedge_ratio_high_volatility": 0.50,
        "hedge_ratio_low_volatility": 0.10,
        "hedge_ratio_sideways": 0.20,
        "hedge_ratio_crisis": 0.70,
        "hedge_ratio_recovery": 0.30,
        "hedge_ratio_unknown": 0.20,
        # Options ceilings
        "options_risk_ceiling_bull_trend": 0.04,
        "options_risk_ceiling_bear_trend": 0.02,
        "options_risk_ceiling_high_volatility": 0.02,
        "options_risk_ceiling_crisis": 0.01,
        # Crisis sensitivity
        "crisis_sensitivity": 1.00,
    }

    validation_memory = {
        "crisis_trigger_accuracy": 0.55,
        "threshold_change_history": {
            "crisis_tighten_count": 2,
            "bull_trend_tighten_count": 1,
        },
        "source_false_positive_count": {"twitter": 12, "reuters": 2},
        "source_success_count": {"reuters": 45, "bloomberg": 38},
        "regime_success_memory": {"bull_trend": 0.63, "crisis": 0.44},
    }

    result = optimize_learning(
        historical_trades=trades,
        signal_history=[],
        regime_history=[],
        source_outcomes={},
        threshold_config=threshold_config,
        validation_memory=validation_memory,
    )

    print("  ✓ optimize_learning() completed without error.\n")

    # ── Validate output schema ─────────────────────────────────────────────────
    required_keys = [
        "adaptive_thresholds", "source_alpha_scores", "recommended_gross_caps",
        "recommended_hedge_ratios", "options_risk_adjustments",
        "drawdown_response_curve", "crisis_sensitivity_adjustment",
        "optimizer_summary",
    ]
    for key in required_keys:
        assert key in result, f"Missing output key: {key}"
    print("  ✓ All required output keys present.")

    # ── Validate bounds ────────────────────────────────────────────────────────
    for regime, t in result["adaptive_thresholds"].items():
        assert THRESHOLD_MIN <= t <= THRESHOLD_MAX, (
            f"Threshold out of bounds: {regime}={t}"
        )
    print("  ✓ All adaptive thresholds within policy bounds.")

    for regime, cap in result["recommended_gross_caps"].items():
        assert GROSS_CAP_MIN <= cap <= GROSS_CAP_MAX, (
            f"Gross cap out of bounds: {regime}={cap}"
        )
    print("  ✓ All gross caps within policy bounds.")

    for regime, hr in result["recommended_hedge_ratios"].items():
        assert HEDGE_RATIO_MIN <= hr <= HEDGE_RATIO_MAX, (
            f"Hedge ratio out of bounds: {regime}={hr}"
        )
    print("  ✓ All hedge ratios within policy bounds.")

    for regime, ceil in result["options_risk_adjustments"].items():
        assert OPTIONS_CEIL_MIN <= ceil <= OPTIONS_CEIL_MAX, (
            f"Options ceiling out of bounds: {regime}={ceil}"
        )
    print("  ✓ All options ceilings within policy bounds.")

    cs = result["crisis_sensitivity_adjustment"]["crisis_sensitivity_multiplier"]
    assert CRISIS_SENSITIVITY_MIN <= cs <= CRISIS_SENSITIVITY_MAX, (
        f"Crisis sensitivity out of bounds: {cs}"
    )
    print("  ✓ Crisis sensitivity within policy bounds.")

    # ── Validate summary ───────────────────────────────────────────────────────
    summary = result["optimizer_summary"]
    assert isinstance(summary["top_winning_regime"], str)
    assert isinstance(summary["worst_false_positive_source"], str)
    assert isinstance(summary["recommended_signal_tightening"], float)
    assert isinstance(summary["capital_preservation_mode"], bool)
    assert isinstance(summary["explanations"], list)
    print("  ✓ Optimizer summary schema valid.")
    print(f"    → Top winning regime  : {summary['top_winning_regime']}")
    print(f"    → Worst FP source     : {summary['worst_false_positive_source']}")
    print(f"    → Signal tightening   : {summary['recommended_signal_tightening']:+.4f}")
    print(f"    → Capital preservation: {summary['capital_preservation_mode']}")

    # ── Test 2: Killswitch scenario ────────────────────────────────────────────
    print("\n[TEST 2] Killswitch scenario — extreme drawdown trades...")
    killswitch_trades = []
    now_ts = datetime.now(timezone.utc).timestamp()
    for i in range(15):
        killswitch_trades.append({
            "trade_id": f"KS_{i}",
            "timestamp": now_ts - i * 86_400,
            "asset": "SPY",
            "direction": "long",
            "regime_tag": "crisis",
            "source": "internal_model",
            "confidence_score": 0.72,
            "signal_strength": 0.60,
            "false_positive_flag": i % 2 == 0,
            "pnl": -0.025,
            "hit": False,
            "gross_exposure": 1.50,
            "hedge_allocation": 0.10,
            "options_premium_at_risk": 0.04,
            "drawdown_at_entry": -0.22,   # beyond killswitch
            "validation_score": 0.80,
            "trust_score": 0.75,
            "misinformation_risk": 0.10,
        })

    ks_result = optimize_learning(
        historical_trades=killswitch_trades,
        signal_history=[], regime_history=[], source_outcomes={},
        threshold_config=threshold_config, validation_memory={},
    )

    assert ks_result["optimizer_summary"]["capital_preservation_mode"], (
        "Killswitch should activate capital preservation mode."
    )
    for regime, cap in ks_result["recommended_gross_caps"].items():
        assert cap == GROSS_CAP_MIN, (
            f"Killswitch: expected min gross cap for {regime}, got {cap}"
        )
    print("  ✓ Killswitch: capital preservation mode activated.")
    print(f"  ✓ Killswitch: all gross caps at minimum ({GROSS_CAP_MIN}).")

    # ── Test 3: Empty trades (edge case) ──────────────────────────────────────
    print("\n[TEST 3] Edge case — empty trade history...")
    empty_result = optimize_learning(
        historical_trades=[],
        signal_history=[], regime_history=[], source_outcomes={},
        threshold_config={}, validation_memory={},
    )
    assert "adaptive_thresholds" in empty_result
    print("  ✓ Empty history handled gracefully.")

    # ── Test 4: Safe leverage jump guard ──────────────────────────────────────
    print("\n[TEST 4] Safe leverage jump guard...")
    large_cap_config = {f"gross_cap_{r}": 0.30 for r in REGIME_LABELS}   # very low baseline
    large_cap_config.update({f"threshold_{r}": 0.60 for r in REGIME_LABELS})
    large_cap_config.update({f"hedge_ratio_{r}": 0.20 for r in REGIME_LABELS})

    strong_trades = []
    for i in range(20):
        strong_trades.append({
            "trade_id": f"ST_{i}",
            "timestamp": now_ts - i * 86_400 * 2,
            "asset": "MSFT",
            "direction": "long",
            "regime_tag": "bull_trend",
            "source": "reuters",
            "confidence_score": 0.80,
            "signal_strength": 0.75,
            "false_positive_flag": False,
            "pnl": 0.020,
            "hit": True,
            "gross_exposure": 1.20,
            "hedge_allocation": 0.10,
            "options_premium_at_risk": 0.02,
            "drawdown_at_entry": 0.0,
            "validation_score": 0.90,
            "trust_score": 0.88,
            "misinformation_risk": 0.02,
        })

    jump_result = optimize_learning(
        historical_trades=strong_trades,
        signal_history=[], regime_history=[], source_outcomes={},
        threshold_config=large_cap_config, validation_memory={},
    )
    bull_cap = jump_result["recommended_gross_caps"]["bull_trend"]
    assert bull_cap <= 0.30 + MAX_LEVERAGE_JUMP, (
        f"Leverage jump too large: from 0.30 to {bull_cap}"
    )
    print(f"  ✓ Leverage jump bounded: bull_trend cap = {bull_cap:.2f} "
          f"(max allowed jump = {MAX_LEVERAGE_JUMP}).")

    # ── Final output print ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FULL OUTPUT (TEST 1) — pretty-printed summary:")
    print("=" * 70)
    print("\nAdaptive Thresholds:")
    for r, v in result["adaptive_thresholds"].items():
        print(f"  {r:<22s}: {v:.4f}")
    print("\nSource Alpha Scores:")
    for src, v in sorted(result["source_alpha_scores"].items(),
                         key=lambda x: -x[1]):
        print(f"  {src:<20s}: {v:.4f}")
    print("\nRecommended Gross Caps:")
    for r, v in result["recommended_gross_caps"].items():
        print(f"  {r:<22s}: {v:.4f}x NAV")
    print("\nRecommended Hedge Ratios:")
    for r, v in result["recommended_hedge_ratios"].items():
        print(f"  {r:<22s}: {v:.4f}")
    print("\nOptions Risk Adjustments:")
    for r, v in result["options_risk_adjustments"].items():
        print(f"  {r:<22s}: {v:.4f}")
    print("\nCrisis Sensitivity:")
    for k, v in result["crisis_sensitivity_adjustment"].items():
        print(f"  {k}: {v:.4f}")
    print("\nDrawdown Response Curve:")
    drc = result["drawdown_response_curve"]
    for k in ["current_drawdown", "active_level", "capital_preservation_mode",
              "rerisk_speed"]:
        print(f"  {k}: {drc[k]}")
    print("  Recommended actions:")
    for act in drc.get("recommended_actions", []):
        print(f"    • {act}")
    print("\nOptimizer Summary:")
    for k, v in result["optimizer_summary"].items():
        if k != "explanations":
            print(f"  {k}: {v}")
    print("\nExplanations:")
    for e in result["optimizer_summary"]["explanations"]:
        print(f"  • {e}")

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED ✓")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    _run_smoke_tests()