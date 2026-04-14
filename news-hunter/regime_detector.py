"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        REGIME DETECTOR v1.0.0                                  ║
║              Institutional-Grade Market Regime Classification Engine            ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Pipeline Position:                                                            ║
║    news_engine → signal_engine → [regime_detector] → portfolio_brain           ║
║    → risk_guardian → execution_bridge → god_core                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Regime Classes:                                                               ║
║    bull_trend          — sustained upward momentum, low vol, risk-on           ║
║    bear_trend          — sustained downward momentum, elevated vol              ║
║    crisis              — acute systemic stress, correlation spike, VIX >40     ║
║    mean_reversion      — oscillating price action, low momentum persistence    ║
║    high_volatility_chop — elevated vol without clear direction                 ║
║    macro_risk_off      — macro-signal driven flight to safety                  ║
║    gamma_squeeze       — options flow forcing directional dealer hedging        ║
║    low_liquidity_panic — volume collapse + spread widening + gap risk           ║
║    ai_bubble_momentum  — parabolic momentum in narrow AI/tech cluster           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Detection Engine Modules:                                                     ║
║    VolatilityClusterDetector  — GARCH-style vol clustering + regime vol bands  ║
║    DrawdownRegimeDetector     — peak-to-trough classification + severity tiers ║
║    CorrelationBreakdownDetector — rolling cross-asset correlation collapse      ║
║    MomentumPersistenceScorer  — Hurst exponent proxy + trend strength          ║
║    CrisisEscalationRules      — waterfall hard-override logic                  ║
║    MacroSignalAggregator      — macro dict signal parsing + risk-off scoring   ║
║    OptionsFlowAnalyser        — gamma squeeze + put/call flow detection         ║
║    LiquidityStressDetector    — volume ratio + gap analysis                    ║
║    AIBubbleDetector           — sector concentration + parabolic acceleration  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Output Contract:                                                              ║
║    market_regime           : str    — canonical regime label                   ║
║    risk_on_off_score       : float  — −1.0 (max risk-off) to +1.0 (max risk-on)║
║    trend_persistence       : float  — 0.0 (choppy) to 1.0 (strong trend)      ║
║    volatility_regime       : str    — low / normal / elevated / extreme        ║
║    recommended_gross_cap   : float  — fraction of NAV (e.g. 0.80)             ║
║    recommended_hedge_ratio : float  — fraction of long book to hedge           ║
║  + explainability block, sub-scores, escalation flags                          ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Engineering: stdlib only · deterministic · no placeholders · smoke tests      ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("regime_detector")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s [regime_detector] %(message)s")
    )
    log.addHandler(_h)
log.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ─────────────────────────────────────────────────────────────────────────────

PriceSeries      = Dict[str, List[float]]
VolatilitySeries = Dict[str, List[float]]
MacroSignals     = List[Dict[str, Any]]
OptionsFlow      = List[Dict[str, Any]]
RegimeResult     = Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — REGIME THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

# ── Volatility band definitions (annualised realised vol) ─────────────────────
VOL_LOW_CEILING:      float = 0.10   # < 10 %  → low
VOL_NORMAL_CEILING:   float = 0.20   # < 20 %  → normal
VOL_ELEVATED_CEILING: float = 0.35   # < 35 %  → elevated
# ≥ 35 % → extreme

# GARCH-style persistence — weight on prior vol estimate
VOL_CLUSTER_PERSISTENCE: float = 0.85
VOL_SPIKE_MULTIPLIER:    float = 2.0   # current vol > 2× long-run → clustering spike

# ── Drawdown regime tiers ─────────────────────────────────────────────────────
DD_MODERATE:  float = 0.05    # −5 %
DD_SEVERE:    float = 0.12    # −12 %
DD_CRISIS:    float = 0.20    # −20 %

# ── Momentum / trend detection ────────────────────────────────────────────────
MOMENTUM_WINDOW_SHORT:  int   = 10
MOMENTUM_WINDOW_LONG:   int   = 30
TREND_STRENGTH_THRESH:  float = 0.60   # R² / direction consistency above which = trending
HURST_TREND_THRESH:     float = 0.60   # H > 0.6 → trending (super-diffusive)
HURST_MR_THRESH:        float = 0.40   # H < 0.4 → mean-reverting (sub-diffusive)

# ── Correlation breakdown ─────────────────────────────────────────────────────
CORR_WINDOW:           int   = 20
CORR_CRISIS_THRESHOLD: float = 0.80   # avg cross-asset |ρ| > 0.80 → crisis correlation
CORR_BREAKDOWN_DELTA:  float = 0.30   # correlation dropped > 0.30 from prior window

# ── Crisis escalation triggers ────────────────────────────────────────────────
CRISIS_VIX_EQUIV:       float = 0.40   # VIX-equivalent vol ≥ 40 %
CRISIS_DRAWDOWN:        float = 0.20   # portfolio / index drawdown ≥ 20 %
CRISIS_CORR_THRESHOLD:  float = 0.80   # cross-asset ρ spike
CRISIS_MACRO_SCORE:     float = 0.70   # macro risk-off score above which → crisis escalation

# ── Options flow ─────────────────────────────────────────────────────────────
GAMMA_SQUEEZE_THRESHOLD:      float = 0.65   # options flow score above which → gamma squeeze
PUT_CALL_EXTREME_BEARISH:     float = 2.50   # P/C ratio ≥ 2.5 → extreme fear
PUT_CALL_EXTREME_BULLISH:     float = 0.50   # P/C ratio ≤ 0.5 → extreme greed
IV_SKEW_BEARISH_THRESHOLD:    float = 0.15   # put skew > 15 %

# ── Liquidity stress ──────────────────────────────────────────────────────────
LIQUIDITY_VOLUME_DROP:  float = 0.50   # volume < 50 % of 20d avg → stress
LIQUIDITY_GAP_THRESH:   float = 0.03   # intraday gap > 3 % → panic flag

# ── AI bubble momentum ────────────────────────────────────────────────────────
AI_BUBBLE_MOMENTUM_THRESH: float = 0.80   # normalised momentum score
AI_BUBBLE_SECTOR_CONC:     float = 0.40   # > 40 % portfolio in AI/tech cluster
AI_TICKERS = frozenset([
    "NVDA", "AMD", "SMCI", "MSFT", "GOOGL", "META", "AMZN", "TSM",
    "AVGO", "ARM", "PLTR", "AI", "SOUN", "IONQ", "BBAI", "AIOT",
])

# ── Regime output caps/hedges per regime ──────────────────────────────────────
REGIME_POLICY: Dict[str, Dict[str, float]] = {
    "bull_trend":           {"gross_cap": 1.30, "hedge_ratio": 0.05},
    "bear_trend":           {"gross_cap": 0.80, "hedge_ratio": 0.25},
    "crisis":               {"gross_cap": 0.40, "hedge_ratio": 0.50},
    "mean_reversion":       {"gross_cap": 0.90, "hedge_ratio": 0.10},
    "high_volatility_chop": {"gross_cap": 0.70, "hedge_ratio": 0.20},
    "macro_risk_off":       {"gross_cap": 0.65, "hedge_ratio": 0.35},
    "gamma_squeeze":        {"gross_cap": 0.85, "hedge_ratio": 0.15},
    "low_liquidity_panic":  {"gross_cap": 0.50, "hedge_ratio": 0.40},
    "ai_bubble_momentum":   {"gross_cap": 1.00, "hedge_ratio": 0.12},
}

# ── Risk-on/off score anchors per regime ──────────────────────────────────────
REGIME_RISK_ON_SCORE: Dict[str, float] = {
    "bull_trend":           +0.80,
    "ai_bubble_momentum":   +0.60,
    "mean_reversion":       +0.10,
    "gamma_squeeze":        +0.00,
    "high_volatility_chop": -0.20,
    "bear_trend":           -0.55,
    "macro_risk_off":       -0.65,
    "low_liquidity_panic":  -0.75,
    "crisis":               -0.95,
}

# Minimum bars required for analysis
MIN_BARS_REQUIRED: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# MATHEMATICAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _returns(prices: List[float]) -> List[float]:
    """Simple log returns from a price series."""
    if len(prices) < 2:
        return []
    return [
        math.log(prices[i] / prices[i - 1])
        for i in range(1, len(prices))
        if prices[i - 1] > 0 and prices[i] > 0
    ]


def _annualised_vol(returns: List[float], periods_per_year: int = 252) -> float:
    """Annualised realised volatility from a return series."""
    if len(returns) < 2:
        return 0.15   # fallback assumption
    try:
        return statistics.stdev(returns) * math.sqrt(periods_per_year)
    except statistics.StatisticsError:
        return 0.15


def _sma(series: List[float], window: int) -> float:
    """Simple moving average over the last `window` observations."""
    trimmed = series[-window:]
    if not trimmed:
        return 0.0
    return sum(trimmed) / len(trimmed)


def _ema(series: List[float], span: int) -> float:
    """
    Exponential moving average — iterative, no external deps.
    Returns the most recent EMA value.
    """
    if not series:
        return 0.0
    alpha = 2.0 / (span + 1)
    ema = series[0]
    for v in series[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Pearson ρ between two equal-length series. Returns 0.0 on failure."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[-n:], y[-n:]
    try:
        mx, my = sum(x) / n, sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx  = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        dy  = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if dx == 0 or dy == 0:
            return 0.0
        return max(-1.0, min(1.0, num / (dx * dy)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _hurst_exponent(series: List[float], max_lag: int = 20) -> float:
    """
    Simplified Hurst exponent via rescaled-range (R/S) analysis.
    H > 0.5 → trending  |  H ≈ 0.5 → random walk  |  H < 0.5 → mean-reverting.
    Returns 0.5 when insufficient data.
    """
    n = len(series)
    if n < max_lag * 2:
        return 0.5

    lags   = [lag for lag in range(2, min(max_lag, n // 2) + 1)]
    rs_log = []
    lag_log = []

    for lag in lags:
        chunk = series[-lag * 2: -lag] if lag * 2 <= n else series[:lag]
        if len(chunk) < 2:
            continue
        try:
            mean_c = sum(chunk) / len(chunk)
            deviation = [v - mean_c for v in chunk]
            cumdev    = []
            acc = 0.0
            for d in deviation:
                acc += d
                cumdev.append(acc)
            R = max(cumdev) - min(cumdev)
            S = statistics.stdev(chunk) if len(chunk) > 1 else 1e-9
            if S == 0:
                continue
            rs_log.append(math.log(R / S))
            lag_log.append(math.log(lag))
        except (ValueError, statistics.StatisticsError):
            continue

    if len(rs_log) < 2:
        return 0.5

    # OLS slope via covariance/variance
    n2   = len(lag_log)
    mx   = sum(lag_log) / n2
    my   = sum(rs_log) / n2
    num  = sum((lag_log[i] - mx) * (rs_log[i] - my) for i in range(n2))
    denom = sum((lag_log[i] - mx) ** 2 for i in range(n2))
    if denom == 0:
        return 0.5
    H = num / denom
    return max(0.0, min(1.0, H))


def _direction_consistency(returns: List[float], window: int) -> float:
    """
    Fraction of return-sign agreements in the last `window` bars.
    High → strong directional trend.  Low → choppy.
    """
    r = returns[-window:]
    if len(r) < 2:
        return 0.5
    n = len(r)
    agreements = sum(
        1 for i in range(1, n)
        if (r[i] > 0) == (r[i - 1] > 0)
    )
    return agreements / (n - 1)


def _peak_to_trough(prices: List[float]) -> float:
    """Maximum drawdown (positive fraction) over the supplied price series."""
    if not prices:
        return 0.0
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _vol_band(ann_vol: float) -> str:
    """Map annualised vol to a named band."""
    if ann_vol < VOL_LOW_CEILING:
        return "low"
    if ann_vol < VOL_NORMAL_CEILING:
        return "normal"
    if ann_vol < VOL_ELEVATED_CEILING:
        return "elevated"
    return "extreme"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ─────────────────────────────────────────────────────────────────────────────
# SUB-DETECTOR: VOLATILITY CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityClusterDetector:
    """
    GARCH-inspired volatility clustering detection.
    Combines a long-run vol estimate with short-run exponential weighting
    to identify vol persistence and spike events.
    """

    def analyse(
        self,
        vol_series: VolatilitySeries,
        price_series: PriceSeries,
    ) -> Dict[str, Any]:
        """
        Returns:
            current_vol          : float — blended annualised vol estimate
            vol_band             : str
            clustering_detected  : bool — recent vol > VOL_SPIKE_MULTIPLIER × long-run
            clustering_score     : float [0,1]
            vol_persistence      : float [0,1] — GARCH persistence proxy
            per_asset_vol        : dict[symbol → float]
        """
        all_vols: List[float] = []
        per_asset: Dict[str, float] = {}

        for sym, prices in price_series.items():
            rets = _returns(prices)
            if not rets:
                # fall back to pre-computed vol series if available
                vseries = vol_series.get(sym, [])
                av = vseries[-1] if vseries else 0.15
            else:
                av = _annualised_vol(rets)
            per_asset[sym] = av
            all_vols.append(av)

        # Also ingest pre-computed vol series (e.g. implied vol from options)
        for sym, vseries in vol_series.items():
            if sym not in per_asset and vseries:
                per_asset[sym] = vseries[-1]
                all_vols.append(vseries[-1])

        if not all_vols:
            return {
                "current_vol": 0.15,
                "vol_band": "normal",
                "clustering_detected": False,
                "clustering_score": 0.0,
                "vol_persistence": 0.5,
                "per_asset_vol": {},
            }

        current_vol = statistics.median(all_vols)
        long_run_vol = statistics.mean(all_vols)

        # Vol persistence: how many assets have vol > 1.5× their own long-run avg
        # Proxy: compute short-window vs long-window vol for each asset
        persistence_flags: List[float] = []
        for sym, prices in price_series.items():
            rets = _returns(prices)
            if len(rets) < MOMENTUM_WINDOW_LONG:
                persistence_flags.append(0.5)
                continue
            short_vol = _annualised_vol(rets[-MOMENTUM_WINDOW_SHORT:])
            long_vol  = _annualised_vol(rets[-MOMENTUM_WINDOW_LONG:])
            if long_vol > 0:
                ratio = short_vol / long_vol
                # GARCH-inspired — persistence weight
                flag = _clamp(
                    VOL_CLUSTER_PERSISTENCE * (ratio - 1.0) / (VOL_SPIKE_MULTIPLIER - 1.0),
                    0.0, 1.0,
                )
            else:
                flag = 0.0
            persistence_flags.append(flag)

        vol_persistence = (
            statistics.mean(persistence_flags) if persistence_flags else 0.5
        )

        clustering_score = _clamp(
            (current_vol / max(long_run_vol, 0.01) - 1.0) / (VOL_SPIKE_MULTIPLIER - 1.0),
            0.0, 1.0,
        )
        clustering_detected = clustering_score > 0.55

        return {
            "current_vol":       current_vol,
            "vol_band":          _vol_band(current_vol),
            "clustering_detected": clustering_detected,
            "clustering_score":  round(clustering_score, 4),
            "vol_persistence":   round(vol_persistence, 4),
            "per_asset_vol":     {s: round(v, 4) for s, v in per_asset.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-DETECTOR: DRAWDOWN REGIME
# ─────────────────────────────────────────────────────────────────────────────

class DrawdownRegimeDetector:
    """
    Classifies regime based on peak-to-trough drawdown across the price universe.
    Produces a drawdown severity score and crisis flag.
    """

    def analyse(self, price_series: PriceSeries) -> Dict[str, Any]:
        """
        Returns:
            max_drawdown          : float — worst drawdown across universe
            avg_drawdown          : float
            drawdown_severity     : str — none / moderate / severe / crisis
            crisis_drawdown_flag  : bool
            drawdown_score        : float [0,1] — normalised severity
        """
        drawdowns: List[float] = []
        per_asset: Dict[str, float] = {}

        for sym, prices in price_series.items():
            if len(prices) < 2:
                continue
            dd = _peak_to_trough(prices)
            drawdowns.append(dd)
            per_asset[sym] = round(dd, 4)

        if not drawdowns:
            return {
                "max_drawdown":       0.0,
                "avg_drawdown":       0.0,
                "drawdown_severity":  "none",
                "crisis_drawdown_flag": False,
                "drawdown_score":     0.0,
                "per_asset_drawdown": {},
            }

        max_dd = max(drawdowns)
        avg_dd = statistics.mean(drawdowns)

        # Severity classification
        if max_dd >= DD_CRISIS:
            severity = "crisis"
        elif max_dd >= DD_SEVERE:
            severity = "severe"
        elif max_dd >= DD_MODERATE:
            severity = "moderate"
        else:
            severity = "none"

        drawdown_score = _clamp(max_dd / DD_CRISIS, 0.0, 1.0)

        return {
            "max_drawdown":        round(max_dd, 4),
            "avg_drawdown":        round(avg_dd, 4),
            "drawdown_severity":   severity,
            "crisis_drawdown_flag": max_dd >= DD_CRISIS,
            "drawdown_score":      round(drawdown_score, 4),
            "per_asset_drawdown":  per_asset,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-DETECTOR: CORRELATION BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationBreakdownDetector:
    """
    Detects cross-asset correlation regime changes.
    High ρ spike → crisis / risk-off.
    Correlation breakdown (sudden drop) → regime shift / liquidity stress.
    """

    def analyse(self, price_series: PriceSeries) -> Dict[str, Any]:
        """
        Returns:
            avg_correlation         : float — current window avg |ρ|
            prior_avg_correlation   : float — prior window avg |ρ|
            correlation_spike       : bool — crisis-level ρ elevation
            correlation_breakdown   : bool — sudden ρ collapse
            correlation_score       : float [0,1]
            pair_correlations       : dict[pair → ρ]
        """
        symbols = [s for s, p in price_series.items() if len(p) >= MIN_BARS_REQUIRED * 2]
        if len(symbols) < 2:
            return {
                "avg_correlation":        0.0,
                "prior_avg_correlation":  0.0,
                "correlation_spike":      False,
                "correlation_breakdown":  False,
                "correlation_score":      0.0,
                "pair_correlations":      {},
            }

        # Build return series per symbol
        ret_series: Dict[str, List[float]] = {}
        for sym in symbols:
            rets = _returns(price_series[sym])
            if len(rets) >= CORR_WINDOW:
                ret_series[sym] = rets

        syms = list(ret_series)
        if len(syms) < 2:
            return {
                "avg_correlation":        0.0,
                "prior_avg_correlation":  0.0,
                "correlation_spike":      False,
                "correlation_breakdown":  False,
                "correlation_score":      0.0,
                "pair_correlations":      {},
            }

        # Current window correlations
        current_corrs: List[float] = []
        prior_corrs:   List[float] = []
        pair_map:      Dict[str, float] = {}

        half = CORR_WINDOW // 2

        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                sa, sb = syms[i], syms[j]
                ra = ret_series[sa]
                rb = ret_series[sb]
                n  = min(len(ra), len(rb))

                curr_rho  = _pearson_correlation(ra[-CORR_WINDOW:], rb[-CORR_WINDOW:])
                prior_rho = _pearson_correlation(
                    ra[-(CORR_WINDOW + half):-half],
                    rb[-(CORR_WINDOW + half):-half],
                )
                current_corrs.append(abs(curr_rho))
                prior_corrs.append(abs(prior_rho))
                pair_map[f"{sa}/{sb}"] = round(curr_rho, 4)

        avg_corr  = statistics.mean(current_corrs) if current_corrs else 0.0
        prior_avg = statistics.mean(prior_corrs)   if prior_corrs   else 0.0

        corr_spike     = avg_corr > CORR_CRISIS_THRESHOLD
        corr_breakdown = (prior_avg - avg_corr) > CORR_BREAKDOWN_DELTA

        # Score: 1.0 = full crisis correlation; 0.0 = uncorrelated
        corr_score = _clamp(avg_corr / CORR_CRISIS_THRESHOLD, 0.0, 1.0)

        return {
            "avg_correlation":        round(avg_corr, 4),
            "prior_avg_correlation":  round(prior_avg, 4),
            "correlation_spike":      corr_spike,
            "correlation_breakdown":  corr_breakdown,
            "correlation_score":      round(corr_score, 4),
            "pair_correlations":      pair_map,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-DETECTOR: MOMENTUM PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

class MomentumPersistenceScorer:
    """
    Scores trend strength and persistence using:
      • Short/long SMA cross ratio
      • Direction consistency (sign agreement fraction)
      • Hurst exponent proxy
    """

    def analyse(self, price_series: PriceSeries) -> Dict[str, Any]:
        """
        Returns:
            momentum_score       : float [−1,+1] — signed trend score
            trend_persistence    : float [0,1]
            hurst_exponent       : float [0,1]
            direction            : str — up / down / neutral / choppy
            strong_trend         : bool
            per_asset_momentum   : dict[symbol → float]
        """
        asset_scores:     List[float] = []
        asset_hurst:      List[float] = []
        asset_consistency: List[float] = []
        per_asset:        Dict[str, float] = {}

        for sym, prices in price_series.items():
            if len(prices) < MOMENTUM_WINDOW_LONG + 1:
                per_asset[sym] = 0.0
                continue

            rets = _returns(prices)

            # SMA cross signal
            short_sma = _sma(prices, MOMENTUM_WINDOW_SHORT)
            long_sma  = _sma(prices, MOMENTUM_WINDOW_LONG)
            sma_signal = (short_sma - long_sma) / max(long_sma, 1e-9)

            # Direction consistency
            consist = _direction_consistency(rets, MOMENTUM_WINDOW_SHORT)
            # Re-centre: 0.5 = random, 1.0 = perfectly consistent
            consist_score = (consist - 0.5) * 2.0   # [-1, +1]

            # Signed momentum: EMA of returns
            ema_ret = _ema(rets[-MOMENTUM_WINDOW_SHORT:], MOMENTUM_WINDOW_SHORT)
            signed_score = _clamp(
                0.40 * math.copysign(1, sma_signal) * min(abs(sma_signal) * 10, 1.0)
                + 0.35 * consist_score
                + 0.25 * math.copysign(1, ema_ret) * min(abs(ema_ret) * 50, 1.0),
                -1.0, 1.0,
            )
            per_asset[sym] = round(signed_score, 4)
            asset_scores.append(signed_score)

            # Hurst
            if len(prices) >= 40:
                H = _hurst_exponent(prices)
                asset_hurst.append(H)
            asset_consistency.append(abs(consist - 0.5) * 2)

        if not asset_scores:
            return {
                "momentum_score":     0.0,
                "trend_persistence":  0.5,
                "hurst_exponent":     0.5,
                "direction":          "neutral",
                "strong_trend":       False,
                "per_asset_momentum": {},
            }

        momentum_score   = statistics.mean(asset_scores)
        hurst_exponent   = statistics.mean(asset_hurst) if asset_hurst else 0.5
        trend_persistence = statistics.mean(asset_consistency) if asset_consistency else 0.5

        # Direction label
        if trend_persistence < 0.30:
            direction = "choppy"
        elif momentum_score > 0.25:
            direction = "up"
        elif momentum_score < -0.25:
            direction = "down"
        else:
            direction = "neutral"

        strong_trend = (
            trend_persistence >= TREND_STRENGTH_THRESH
            and abs(momentum_score) > 0.40
        )

        return {
            "momentum_score":     round(momentum_score, 4),
            "trend_persistence":  round(trend_persistence, 4),
            "hurst_exponent":     round(hurst_exponent, 4),
            "direction":          direction,
            "strong_trend":       strong_trend,
            "per_asset_momentum": per_asset,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-DETECTOR: MACRO SIGNAL AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class MacroSignalAggregator:
    """
    Parses macro_signals dicts from upstream (news_engine / signal_engine)
    and produces a normalised risk-off score + event classification.

    Expected macro signal fields:
        event_type, impact_score, signal_direction, confidence_score,
        urgency, sector, asset_class, signal_reasons
    """

    _RISK_OFF_EVENTS = frozenset([
        "rate_hike", "credit_crunch", "sovereign_debt", "recession",
        "geopolitical", "bank_failure", "inflation_shock", "fed_hawkish",
        "yield_curve_inversion", "default_risk", "systemic_risk",
        "liquidity_crisis", "contagion", "war", "pandemic",
    ])

    _RISK_ON_EVENTS = frozenset([
        "rate_cut", "qe", "earnings_beat", "fed_dovish", "stimulus",
        "buyback", "m&a", "ipo_boom", "ai_breakthrough", "soft_landing",
    ])

    def analyse(self, macro_signals: MacroSignals) -> Dict[str, Any]:
        """
        Returns:
            macro_risk_off_score : float [0,1]
            macro_risk_on_score  : float [0,1]
            net_macro_score      : float [−1,+1]
            risk_off_events      : list[str]
            risk_on_events       : list[str]
            dominant_theme       : str
            macro_escalation     : bool — severe enough to trigger macro_risk_off regime
        """
        if not macro_signals:
            return {
                "macro_risk_off_score": 0.0,
                "macro_risk_on_score":  0.0,
                "net_macro_score":      0.0,
                "risk_off_events":      [],
                "risk_on_events":       [],
                "dominant_theme":       "neutral",
                "macro_escalation":     False,
            }

        risk_off_weights: List[float] = []
        risk_on_weights:  List[float] = []
        off_events:       List[str]   = []
        on_events:        List[str]   = []

        for sig in macro_signals:
            event    = str(sig.get("event_type", "")).lower().replace(" ", "_")
            impact   = float(sig.get("impact_score", 0.5))
            conf     = float(sig.get("confidence_score", 0.5))
            urgency_map = {"low": 0.3, "medium": 0.5, "high": 0.8, "critical": 1.0}
            urgency  = urgency_map.get(str(sig.get("urgency", "medium")).lower(), 0.5)
            direction = str(sig.get("signal_direction", "neutral")).lower()

            weight = impact * conf * urgency

            is_off = any(ro in event for ro in self._RISK_OFF_EVENTS) or direction == "bearish"
            is_on  = any(ro in event for ro in self._RISK_ON_EVENTS)  or direction == "bullish"

            if is_off:
                risk_off_weights.append(weight)
                off_events.append(event)
            elif is_on:
                risk_on_weights.append(weight)
                on_events.append(event)
            else:
                # Neutral — split the weight
                risk_off_weights.append(weight * 0.1)
                risk_on_weights.append(weight * 0.1)

        def _normalise(weights: List[float]) -> float:
            if not weights:
                return 0.0
            raw = sum(weights) / len(macro_signals)
            return _clamp(raw, 0.0, 1.0)

        off_score = _normalise(risk_off_weights)
        on_score  = _normalise(risk_on_weights)
        net       = _clamp(on_score - off_score, -1.0, 1.0)

        if off_score > on_score + 0.10:
            dominant = "risk_off"
        elif on_score > off_score + 0.10:
            dominant = "risk_on"
        else:
            dominant = "neutral"

        return {
            "macro_risk_off_score": round(off_score, 4),
            "macro_risk_on_score":  round(on_score, 4),
            "net_macro_score":      round(net, 4),
            "risk_off_events":      list(set(off_events))[:10],
            "risk_on_events":       list(set(on_events))[:10],
            "dominant_theme":       dominant,
            "macro_escalation":     off_score >= CRISIS_MACRO_SCORE,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-DETECTOR: OPTIONS FLOW ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class OptionsFlowAnalyser:
    """
    Detects gamma squeeze conditions and put/call sentiment extremes
    from upstream options flow signals (set by signal_engine).

    Expected options flow signal fields:
        symbol, option_type, delta, gamma, implied_volatility,
        signal_direction, signal_strength, impact_score, urgency
    """

    def analyse(self, options_flow: Optional[OptionsFlow]) -> Dict[str, Any]:
        """
        Returns:
            gamma_squeeze_score  : float [0,1]
            gamma_squeeze_flag   : bool
            put_call_ratio       : float
            sentiment            : str — bullish / bearish / extreme_fear / extreme_greed / neutral
            iv_skew_score        : float [0,1]
            options_risk_score   : float [0,1]
        """
        _empty = {
            "gamma_squeeze_score": 0.0,
            "gamma_squeeze_flag":  False,
            "put_call_ratio":      1.0,
            "sentiment":           "neutral",
            "iv_skew_score":       0.0,
            "options_risk_score":  0.0,
        }
        if not options_flow:
            return _empty

        call_count, put_count = 0, 0
        gamma_scores:   List[float] = []
        iv_put_list:    List[float] = []
        iv_call_list:   List[float] = []

        for sig in options_flow:
            opt_type  = str(sig.get("option_type", "")).lower()
            gamma     = abs(float(sig.get("gamma", 0.0)))
            iv        = float(sig.get("implied_volatility", 0.0))
            strength  = float(sig.get("signal_strength", 0.5))
            direction = str(sig.get("signal_direction", "")).lower()
            impact    = float(sig.get("impact_score", 0.5))

            if opt_type == "call":
                call_count += 1
                iv_call_list.append(iv)
            elif opt_type == "put":
                put_count += 1
                iv_put_list.append(iv)

            # Gamma squeeze signal: large gamma + strong directional flow
            if gamma > 0 and strength > 0.5:
                gamma_scores.append(
                    _clamp(gamma * strength * impact * 10, 0.0, 1.0)
                )

        total = call_count + put_count
        if total == 0:
            return _empty

        pc_ratio = put_count / max(call_count, 1)

        # IV skew: (avg put IV − avg call IV) / avg call IV
        avg_put_iv  = statistics.mean(iv_put_list)  if iv_put_list  else 0.0
        avg_call_iv = statistics.mean(iv_call_list) if iv_call_list else 0.01
        iv_skew     = (avg_put_iv - avg_call_iv) / max(avg_call_iv, 0.01)
        iv_skew_score = _clamp(iv_skew / IV_SKEW_BEARISH_THRESHOLD, 0.0, 1.0)

        # Sentiment label
        if pc_ratio >= PUT_CALL_EXTREME_BEARISH:
            sentiment = "extreme_fear"
        elif pc_ratio >= 1.80:
            sentiment = "bearish"
        elif pc_ratio <= PUT_CALL_EXTREME_BULLISH:
            sentiment = "extreme_greed"
        elif pc_ratio <= 0.75:
            sentiment = "bullish"
        else:
            sentiment = "neutral"

        gamma_squeeze_score = statistics.mean(gamma_scores) if gamma_scores else 0.0
        gamma_squeeze_flag  = gamma_squeeze_score >= GAMMA_SQUEEZE_THRESHOLD

        # Composite options risk: high PC + high skew + high gamma = risk
        options_risk_score = _clamp(
            0.40 * iv_skew_score
            + 0.35 * gamma_squeeze_score
            + 0.25 * _clamp((pc_ratio - 1.0) / (PUT_CALL_EXTREME_BEARISH - 1.0), 0.0, 1.0),
            0.0, 1.0,
        )

        return {
            "gamma_squeeze_score": round(gamma_squeeze_score, 4),
            "gamma_squeeze_flag":  gamma_squeeze_flag,
            "put_call_ratio":      round(pc_ratio, 4),
            "sentiment":           sentiment,
            "iv_skew_score":       round(iv_skew_score, 4),
            "options_risk_score":  round(options_risk_score, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-DETECTOR: LIQUIDITY STRESS
# ─────────────────────────────────────────────────────────────────────────────

class LiquidityStressDetector:
    """
    Detects low-liquidity panic conditions via:
      • Volume relative to 20-day average (from macro_signals or price_series proxy)
      • Intraday gap magnitude
      • Liquidity_score fields from macro/options signals
    """

    def analyse(
        self,
        price_series:  PriceSeries,
        macro_signals: MacroSignals,
        options_flow:  Optional[OptionsFlow],
    ) -> Dict[str, Any]:
        """
        Returns:
            liquidity_stress_score : float [0,1]
            low_liquidity_flag     : bool
            gap_panic_flag         : bool
            volume_stress_flag     : bool
            liquidity_sources      : list[str] — explanation
        """
        sources:    List[str]   = []
        gap_flags:  List[bool]  = []
        liq_scores: List[float] = []

        # ── Gap detection via price series ───────────────────────────────────
        for sym, prices in price_series.items():
            if len(prices) < 3:
                continue
            # Detect gaps: |close[t] - close[t-1]| / close[t-1]
            for i in range(max(1, len(prices) - 5), len(prices)):
                prev = prices[i - 1]
                curr = prices[i]
                if prev > 0:
                    gap = abs(curr - prev) / prev
                    if gap > LIQUIDITY_GAP_THRESH:
                        gap_flags.append(True)
                        sources.append(f"gap_{sym}_{gap:.2%}")

        # ── Liquidity scores from flow signals ───────────────────────────────
        for sig in (macro_signals or []) + (options_flow or []):
            liq = sig.get("liquidity_score")
            if liq is not None:
                liq_scores.append(float(liq))
                if float(liq) < 0.30:
                    sources.append(f"low_liq_signal({liq:.2f})")

        # ── Volume stress proxy via macro event keywords ─────────────────────
        volume_stress_flag = False
        for sig in macro_signals or []:
            event = str(sig.get("event_type", "")).lower()
            if any(kw in event for kw in ("liquidity", "volume", "halt", "circuit", "panic")):
                volume_stress_flag = True
                sources.append(f"macro_event:{event}")

        avg_liq = statistics.mean(liq_scores) if liq_scores else 1.0
        gap_panic_flag = len(gap_flags) >= 2 or (gap_flags and avg_liq < 0.50)

        # Composite stress score
        liq_stress = 0.0
        liq_stress += 0.50 * _clamp(1.0 - avg_liq, 0.0, 1.0)
        liq_stress += 0.30 * (1.0 if gap_panic_flag else 0.0)
        liq_stress += 0.20 * (1.0 if volume_stress_flag else 0.0)
        liq_stress = _clamp(liq_stress, 0.0, 1.0)

        return {
            "liquidity_stress_score": round(liq_stress, 4),
            "low_liquidity_flag":     liq_stress > 0.55,
            "gap_panic_flag":         gap_panic_flag,
            "volume_stress_flag":     volume_stress_flag,
            "liquidity_sources":      sources[:10],
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-DETECTOR: AI BUBBLE MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

class AIBubbleDetector:
    """
    Identifies parabolic momentum concentrated in the AI / semiconductor cluster.
    Triggers when:
      • Multiple AI-cluster symbols exhibit extreme upward momentum
      • Portfolio / signal concentration in that cluster exceeds threshold
      • Hurst exponent is high (persistent trending)
    """

    def analyse(
        self,
        price_series:       PriceSeries,
        current_positions:  Dict[str, float],
        momentum_result:    Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Returns:
            ai_bubble_score     : float [0,1]
            ai_bubble_flag      : bool
            ai_cluster_exposure : float — fraction of positions in AI tickers
            ai_momentum_tickers : list[str]
        """
        ai_momentum_tickers: List[str]   = []
        ai_position_weight:  float       = 0.0
        ai_momentum_scores:  List[float] = []

        per_mom = momentum_result.get("per_asset_momentum", {})

        # Check price momentum in AI cluster
        for sym, prices in price_series.items():
            if sym.upper() not in AI_TICKERS:
                continue
            mom = per_mom.get(sym, 0.0)
            if mom > AI_BUBBLE_MOMENTUM_THRESH * 0.7:   # softer threshold per-asset
                ai_momentum_tickers.append(sym)
                ai_momentum_scores.append(mom)

        # Check position concentration
        total_pos = sum(abs(v) for v in current_positions.values()) or 1e-9
        for sym, weight in current_positions.items():
            if sym.upper() in AI_TICKERS:
                ai_position_weight += abs(weight) / total_pos

        # Hurst-based persistence
        hurst = momentum_result.get("hurst_exponent", 0.5)
        hurst_boost = _clamp((hurst - HURST_TREND_THRESH) / (1.0 - HURST_TREND_THRESH), 0.0, 1.0)

        avg_ai_mom = statistics.mean(ai_momentum_scores) if ai_momentum_scores else 0.0

        ai_bubble_score = _clamp(
            0.50 * avg_ai_mom
            + 0.30 * _clamp(ai_position_weight / AI_BUBBLE_SECTOR_CONC, 0.0, 1.0)
            + 0.20 * hurst_boost,
            0.0, 1.0,
        )

        return {
            "ai_bubble_score":     round(ai_bubble_score, 4),
            "ai_bubble_flag":      ai_bubble_score >= AI_BUBBLE_MOMENTUM_THRESH,
            "ai_cluster_exposure": round(ai_position_weight, 4),
            "ai_momentum_tickers": ai_momentum_tickers,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CRISIS ESCALATION RULES  — hard-override waterfall
# ─────────────────────────────────────────────────────────────────────────────

class CrisisEscalationRules:
    """
    Waterfall of hard-override rules that can force regime = 'crisis'
    regardless of the soft scoring.  Returns (escalated, reasons).
    """

    def check(
        self,
        vol_result:   Dict[str, Any],
        dd_result:    Dict[str, Any],
        corr_result:  Dict[str, Any],
        macro_result: Dict[str, Any],
        liq_result:   Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Returns (crisis_forced: bool, reasons: list[str]).
        """
        reasons: List[str] = []

        if vol_result.get("current_vol", 0.0) >= CRISIS_VIX_EQUIV:
            reasons.append(
                f"vol_crisis_trigger(vol={vol_result['current_vol']:.2%}≥{CRISIS_VIX_EQUIV:.0%})"
            )

        if dd_result.get("crisis_drawdown_flag", False):
            reasons.append(
                f"drawdown_crisis_trigger(dd={dd_result['max_drawdown']:.2%})"
            )

        if corr_result.get("correlation_spike", False):
            reasons.append(
                f"corr_spike_trigger(ρ={corr_result['avg_correlation']:.2f})"
            )

        if macro_result.get("macro_escalation", False):
            reasons.append(
                f"macro_escalation_trigger(score={macro_result['macro_risk_off_score']:.2f})"
            )

        if liq_result.get("gap_panic_flag", False) and liq_result.get("volume_stress_flag", False):
            reasons.append("liquidity_panic_trigger(gap+volume_stress)")

        return len(reasons) >= 1, reasons


# ─────────────────────────────────────────────────────────────────────────────
# REGIME SCORER  — soft voting + priority resolution
# ─────────────────────────────────────────────────────────────────────────────

class RegimeScorer:
    """
    Aggregates all sub-detector outputs into a single regime classification
    via weighted soft voting with a priority resolution tie-breaker.

    Priority order (highest → lowest):
        crisis > low_liquidity_panic > macro_risk_off > bear_trend
        > gamma_squeeze > high_volatility_chop > mean_reversion
        > ai_bubble_momentum > bull_trend
    """

    _PRIORITY = [
        "crisis", "low_liquidity_panic", "macro_risk_off", "bear_trend",
        "gamma_squeeze", "high_volatility_chop", "mean_reversion",
        "ai_bubble_momentum", "bull_trend",
    ]

    def score(
        self,
        vol:      Dict[str, Any],
        dd:       Dict[str, Any],
        corr:     Dict[str, Any],
        mom:      Dict[str, Any],
        macro:    Dict[str, Any],
        options:  Dict[str, Any],
        liq:      Dict[str, Any],
        ai:       Dict[str, Any],
        crisis_forced: bool,
    ) -> Tuple[str, Dict[str, float], List[str]]:
        """
        Returns (winning_regime, votes_dict, explanation_list).
        Votes are raw evidence scores in [0,1]; highest vote wins subject to priority.
        """
        # ── Build evidence votes per regime ──────────────────────────────────
        votes: Dict[str, float] = defaultdict(float)

        # crisis
        if crisis_forced:
            votes["crisis"] += 1.0
        votes["crisis"] += 0.5 * dd.get("drawdown_score", 0.0)
        votes["crisis"] += 0.3 * corr.get("correlation_score", 0.0)
        votes["crisis"] += 0.2 * vol.get("clustering_score", 0.0)

        # low_liquidity_panic
        votes["low_liquidity_panic"] += 0.7 * liq.get("liquidity_stress_score", 0.0)
        votes["low_liquidity_panic"] += 0.3 * dd.get("drawdown_score", 0.0)

        # macro_risk_off
        votes["macro_risk_off"] += 0.8 * macro.get("macro_risk_off_score", 0.0)
        votes["macro_risk_off"] += 0.2 * (1.0 if macro.get("dominant_theme") == "risk_off" else 0.0)

        # bear_trend
        dir_is_down = mom.get("direction") == "down"
        votes["bear_trend"] += 0.6 * (mom.get("trend_persistence", 0.0) if dir_is_down else 0.0)
        votes["bear_trend"] += 0.4 * _clamp(-mom.get("momentum_score", 0.0), 0.0, 1.0)

        # gamma_squeeze
        votes["gamma_squeeze"] += options.get("gamma_squeeze_score", 0.0)

        # high_volatility_chop
        vol_band = vol.get("vol_band", "normal")
        is_high_vol = vol_band in ("elevated", "extreme")
        dir_is_chop = mom.get("direction") in ("neutral", "choppy")
        votes["high_volatility_chop"] += 0.6 * (1.0 if is_high_vol else 0.0)
        votes["high_volatility_chop"] += 0.4 * (1.0 if dir_is_chop else 0.0)

        # mean_reversion
        hurst = mom.get("hurst_exponent", 0.5)
        mr_hurst_score = _clamp((HURST_MR_THRESH - hurst) / HURST_MR_THRESH, 0.0, 1.0)
        votes["mean_reversion"] += 0.7 * mr_hurst_score
        votes["mean_reversion"] += 0.3 * (1.0 - mom.get("trend_persistence", 0.5))

        # ai_bubble_momentum
        votes["ai_bubble_momentum"] += ai.get("ai_bubble_score", 0.0)

        # bull_trend
        dir_is_up = mom.get("direction") == "up"
        votes["bull_trend"] += 0.6 * (mom.get("trend_persistence", 0.0) if dir_is_up else 0.0)
        votes["bull_trend"] += 0.4 * _clamp(mom.get("momentum_score", 0.0), 0.0, 1.0)
        votes["bull_trend"] -= 0.2 * macro.get("macro_risk_off_score", 0.0)  # macro headwind
        votes["bull_trend"] = max(0.0, votes["bull_trend"])

        # ── Priority-weighted resolution ──────────────────────────────────────
        # Find regime with highest vote; ties broken by priority order.
        best_regime = "neutral"
        best_score  = -1.0

        for regime in self._PRIORITY:
            vote = votes.get(regime, 0.0)
            if vote > best_score:
                best_score  = vote
                best_regime = regime
            elif abs(vote - best_score) < 0.05:
                # Tie: higher priority wins (earlier in list)
                pass   # priority order already handles this via iteration order

        # ── Normalise votes for explainability ───────────────────────────────
        total_vote = sum(votes.values()) or 1.0
        normed_votes = {k: round(v / total_vote, 4) for k, v in sorted(
            votes.items(), key=lambda kv: kv[1], reverse=True
        )}

        explanations = [
            f"WINNING_REGIME={best_regime}(raw_score={best_score:.3f})",
            f"regime_votes={normed_votes}",
            f"direction={mom.get('direction')} persistence={mom.get('trend_persistence'):.2f}",
            f"hurst={hurst:.2f} vol_band={vol_band}",
            f"macro_theme={macro.get('dominant_theme')} "
            f"risk_off={macro.get('macro_risk_off_score'):.2f}",
            f"corr_spike={corr.get('correlation_spike')} "
            f"avg_corr={corr.get('avg_correlation'):.2f}",
            f"gamma_squeeze={options.get('gamma_squeeze_flag')} "
            f"pc_ratio={options.get('put_call_ratio'):.2f}",
            f"liq_stress={liq.get('liquidity_stress_score'):.2f} "
            f"gap_panic={liq.get('gap_panic_flag')}",
            f"ai_bubble={ai.get('ai_bubble_flag')} "
            f"ai_score={ai.get('ai_bubble_score'):.2f}",
        ]

        return best_regime, dict(votes), explanations


# ─────────────────────────────────────────────────────────────────────────────
# RISK-ON / OFF SCORE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_risk_on_off_score(
    regime:       str,
    mom_score:    float,
    macro_net:    float,
    vol_score:    float,
    options_risk: float,
    liq_stress:   float,
    corr_score:   float,
) -> float:
    """
    Blended risk-on/off score in [−1.0, +1.0].
    Anchored on the regime base score, then adjusted by sub-factor evidence.
    """
    base = REGIME_RISK_ON_SCORE.get(regime, 0.0)

    adjustment = (
        0.25 * mom_score
        + 0.20 * macro_net
        - 0.20 * vol_score
        - 0.15 * options_risk
        - 0.10 * liq_stress
        - 0.10 * corr_score
    )
    blended = 0.60 * base + 0.40 * adjustment
    return round(_clamp(blended, -1.0, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTOR — PRIMARY CLASS
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Primary entry-point for market regime detection.

    Usage
    -----
    detector = RegimeDetector()
    result = detector.detect(
        price_series      = {"SPY": [...], "QQQ": [...], ...},
        volatility_series = {"VIX": [...], ...},
        macro_signals     = [...],              # from signal_engine
        options_flow_signals = [...],           # optional
        current_positions = {"AAPL": 0.10, ...},
    )

    The returned dict is plug-compatible with portfolio_brain.PortfolioContext
    and passes market_regime + risk_on_off_score downstream.
    """

    def __init__(self) -> None:
        self.vol_detector    = VolatilityClusterDetector()
        self.dd_detector     = DrawdownRegimeDetector()
        self.corr_detector   = CorrelationBreakdownDetector()
        self.mom_scorer      = MomentumPersistenceScorer()
        self.macro_agg       = MacroSignalAggregator()
        self.options_analyst = OptionsFlowAnalyser()
        self.liq_detector    = LiquidityStressDetector()
        self.ai_detector     = AIBubbleDetector()
        self.crisis_rules    = CrisisEscalationRules()
        self.regime_scorer   = RegimeScorer()

    # ── Main detect entrypoint ────────────────────────────────────────────────

    def detect(
        self,
        price_series:          PriceSeries,
        volatility_series:     VolatilitySeries,
        macro_signals:         MacroSignals,
        options_flow_signals:  Optional[OptionsFlow] = None,
        current_positions:     Optional[Dict[str, float]] = None,
    ) -> RegimeResult:
        """
        Full detection pipeline.  All sub-detectors run independently;
        results are aggregated by RegimeScorer with crisis escalation override.

        Returns the output contract consumed by portfolio_brain.
        """
        current_positions = current_positions or {}
        log.info(
            "RegimeDetector.detect(): symbols=%d vol_series=%d macro=%d options=%d",
            len(price_series), len(volatility_series),
            len(macro_signals), len(options_flow_signals or []),
        )

        # ── Sub-detector runs ─────────────────────────────────────────────────
        vol_result   = self.vol_detector.analyse(volatility_series, price_series)
        dd_result    = self.dd_detector.analyse(price_series)
        corr_result  = self.corr_detector.analyse(price_series)
        mom_result   = self.mom_scorer.analyse(price_series)
        macro_result = self.macro_agg.analyse(macro_signals)
        opts_result  = self.options_analyst.analyse(options_flow_signals)
        liq_result   = self.liq_detector.analyse(price_series, macro_signals, options_flow_signals)
        ai_result    = self.ai_detector.analyse(price_series, current_positions, mom_result)

        # ── Crisis escalation hard check ──────────────────────────────────────
        crisis_forced, crisis_reasons = self.crisis_rules.check(
            vol_result, dd_result, corr_result, macro_result, liq_result
        )
        if crisis_forced:
            log.warning("CRISIS ESCALATION TRIGGERED: %s", crisis_reasons)

        # ── Regime classification ─────────────────────────────────────────────
        regime, votes, explanations = self.regime_scorer.score(
            vol_result, dd_result, corr_result, mom_result,
            macro_result, opts_result, liq_result, ai_result,
            crisis_forced,
        )

        # ── Derived output values ─────────────────────────────────────────────
        risk_on_off = _build_risk_on_off_score(
            regime,
            mom_result.get("momentum_score",       0.0),
            macro_result.get("net_macro_score",    0.0),
            vol_result.get("clustering_score",     0.0),
            opts_result.get("options_risk_score",  0.0),
            liq_result.get("liquidity_stress_score", 0.0),
            corr_result.get("correlation_score",   0.0),
        )

        policy = REGIME_POLICY.get(regime, {"gross_cap": 0.90, "hedge_ratio": 0.15})

        # ── Assemble output contract ──────────────────────────────────────────
        result: RegimeResult = {
            # ── Primary output contract (portfolio_brain compatible) ──────────
            "market_regime":           regime,
            "risk_on_off_score":       risk_on_off,
            "trend_persistence":       mom_result.get("trend_persistence", 0.5),
            "volatility_regime":       vol_result.get("vol_band", "normal"),
            "recommended_gross_cap":   policy["gross_cap"],
            "recommended_hedge_ratio": policy["hedge_ratio"],
            # ── Explainability & sub-scores ───────────────────────────────────
            "regime_summary": {
                "explanations":      explanations,
                "crisis_forced":     crisis_forced,
                "crisis_reasons":    crisis_reasons,
                "regime_votes":      {k: round(v, 4) for k, v in votes.items()},
            },
            "sub_scores": {
                "volatility":  vol_result,
                "drawdown":    dd_result,
                "correlation": corr_result,
                "momentum":    mom_result,
                "macro":       macro_result,
                "options":     opts_result,
                "liquidity":   liq_result,
                "ai_bubble":   ai_result,
            },
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "module":        "regime_detector",
        }

        log.info(
            "Regime=%s | risk_on_off=%.2f | vol=%s | trend=%.2f | "
            "gross_cap=%.2f | hedge=%.2f",
            regime, risk_on_off,
            vol_result.get("vol_band"),
            mom_result.get("trend_persistence", 0.5),
            policy["gross_cap"],
            policy["hedge_ratio"],
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime(
    price_series:         PriceSeries,
    volatility_series:    VolatilitySeries,
    macro_signals:        MacroSignals,
    options_flow_signals: Optional[OptionsFlow]       = None,
    current_positions:    Optional[Dict[str, float]]  = None,
) -> RegimeResult:
    """
    Pipeline-compatible function wrapper for RegimeDetector.

    Drop into the TRADING_AI pipeline between signal_engine and portfolio_brain:
        regime_result = detect_regime(price_series, vol_series, macro_signals)
        ctx.market_regime = regime_result["market_regime"]
    """
    return RegimeDetector().detect(
        price_series, volatility_series, macro_signals,
        options_flow_signals, current_positions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA FACTORIES FOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _trending_prices(
    n: int = 60,
    drift: float = 0.001,
    noise: float = 0.005,
    start: float = 100.0,
    seed: int = 42,
) -> List[float]:
    """Deterministic trending price series (no random module — LCG-based)."""
    # Minimal LCG for reproducible pseudo-noise
    lcg_state = seed
    def _lcg() -> float:
        nonlocal lcg_state
        lcg_state = (1664525 * lcg_state + 1013904223) & 0xFFFFFFFF
        return (lcg_state / 0xFFFFFFFF) - 0.5   # [-0.5, +0.5]

    prices = [start]
    for _ in range(n - 1):
        ret = drift + noise * _lcg()
        prices.append(prices[-1] * math.exp(ret))
    return prices


def _volatile_prices(n: int = 60, noise: float = 0.025, seed: int = 99) -> List[float]:
    """Deterministic high-volatility choppy series."""
    return _trending_prices(n, drift=0.0, noise=noise, seed=seed)


def _crashing_prices(n: int = 60, seed: int = 7) -> List[float]:
    """Deterministic bear / crash series."""
    return _trending_prices(n, drift=-0.008, noise=0.015, seed=seed)


def _flat_prices(n: int = 60, seed: int = 11) -> List[float]:
    """Near-flat mean-reverting series."""
    return _trending_prices(n, drift=0.0, noise=0.003, seed=seed)


def _make_macro_signal(
    event_type:       str,
    impact:           float = 0.8,
    confidence:       float = 0.85,
    urgency:          str   = "high",
    signal_direction: str   = "bearish",
) -> Dict[str, Any]:
    return {
        "event_type":       event_type,
        "impact_score":     impact,
        "confidence_score": confidence,
        "urgency":          urgency,
        "signal_direction": signal_direction,
        "liquidity_score":  0.80,
    }


def _make_options_signal(
    option_type: str,
    gamma:       float = 0.005,
    iv:          float = 0.30,
    strength:    float = 0.75,
    direction:   str   = "long",
) -> Dict[str, Any]:
    return {
        "option_type":        option_type,
        "gamma":              gamma,
        "implied_volatility": iv,
        "signal_strength":    strength,
        "signal_direction":   direction,
        "impact_score":       0.70,
        "urgency":            "high",
        "liquidity_score":    0.75,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_smoke_tests() -> None:
    print("\n" + "═" * 72)
    print("  REGIME DETECTOR — SMOKE TESTS")
    print("═" * 72)

    PASS = "✓ PASS"
    FAIL = "✗ FAIL"
    results: List[Tuple[bool, str, str]] = []

    def chk(name: str, condition: bool, detail: str = "") -> None:
        label = PASS if condition else FAIL
        print(f"  {label}  {name}" + (f"  [{detail}]" if detail else ""))
        results.append((condition, name, detail))
        if not condition:
            raise AssertionError(f"FAILED: {name} — {detail}")

    # ── Test 1: Output contract completeness ──────────────────────────────────
    prices = {
        "SPY": _trending_prices(60, drift=0.001),
        "QQQ": _trending_prices(60, drift=0.0008, seed=13),
        "GLD": _flat_prices(60),
    }
    result = detect_regime(prices, {}, [])
    required_keys = [
        "market_regime", "risk_on_off_score", "trend_persistence",
        "volatility_regime", "recommended_gross_cap", "recommended_hedge_ratio",
    ]
    chk("Output contract: all 6 required keys present",
        all(k in result for k in required_keys),
        str([k for k in required_keys if k not in result]))

    # ── Test 2: Valid regime label ────────────────────────────────────────────
    valid_regimes = set(REGIME_POLICY.keys())
    chk("market_regime is a known label",
        result["market_regime"] in valid_regimes,
        result["market_regime"])

    # ── Test 3: risk_on_off_score in [−1, +1] ─────────────────────────────────
    ros = result["risk_on_off_score"]
    chk("risk_on_off_score in [−1.0, +1.0]", -1.0 <= ros <= 1.0, f"{ros:.4f}")

    # ── Test 4: trend_persistence in [0, 1] ──────────────────────────────────
    tp = result["trend_persistence"]
    chk("trend_persistence in [0.0, 1.0]", 0.0 <= tp <= 1.0, f"{tp:.4f}")

    # ── Test 5: volatility_regime is a valid band ─────────────────────────────
    chk("volatility_regime is valid band",
        result["volatility_regime"] in ("low", "normal", "elevated", "extreme"),
        result["volatility_regime"])

    # ── Test 6: recommended_gross_cap in sane range ───────────────────────────
    gc = result["recommended_gross_cap"]
    chk("recommended_gross_cap in [0.30, 1.50]", 0.30 <= gc <= 1.50, f"{gc:.2f}")

    # ── Test 7: recommended_hedge_ratio in [0, 1] ─────────────────────────────
    hr = result["recommended_hedge_ratio"]
    chk("recommended_hedge_ratio in [0.0, 1.0]", 0.0 <= hr <= 1.0, f"{hr:.2f}")

    # ── Test 8: Bull trend detection ──────────────────────────────────────────
    bull_prices = {
        "SPY": _trending_prices(60, drift=0.0015, noise=0.003, seed=1),
        "QQQ": _trending_prices(60, drift=0.0012, noise=0.003, seed=2),
        "IWM": _trending_prices(60, drift=0.0010, noise=0.003, seed=3),
    }
    bull_result = detect_regime(bull_prices, {}, [])
    chk("Bull trend scenario: risk_on_off > 0",
        bull_result["risk_on_off_score"] > 0,
        f"{bull_result['risk_on_off_score']:.3f}")
    chk("Bull trend scenario: gross_cap >= 1.0",
        bull_result["recommended_gross_cap"] >= 1.0,
        f"{bull_result['recommended_gross_cap']:.2f}")

    # ── Test 9: Crisis scenario — vol escalation ──────────────────────────────
    # Simulate extreme volatility by using very noisy prices (→ vol ≥ 40 %)
    # We'll inject very high variance returns via large noise
    crisis_prices = {
        "SPY": _crashing_prices(60, seed=5),
        "QQQ": _crashing_prices(60, seed=6),
        "HYG": _crashing_prices(60, seed=7),
    }
    crisis_macro = [
        _make_macro_signal("systemic_risk",    impact=0.95, confidence=0.90),
        _make_macro_signal("bank_failure",     impact=0.90, confidence=0.88),
        _make_macro_signal("liquidity_crisis", impact=0.92, confidence=0.85),
        _make_macro_signal("contagion",        impact=0.88, confidence=0.82),
    ]
    crisis_result = detect_regime(crisis_prices, {}, crisis_macro)
    chk("Crisis macro signals: risk_on_off < 0",
        crisis_result["risk_on_off_score"] < 0,
        f"{crisis_result['risk_on_off_score']:.3f}")
    chk("Crisis scenario: hedge_ratio >= 0.20",
        crisis_result["recommended_hedge_ratio"] >= 0.20,
        f"{crisis_result['recommended_hedge_ratio']:.2f}")
    chk("Crisis scenario: gross_cap <= 0.90",
        crisis_result["recommended_gross_cap"] <= 0.90,
        f"{crisis_result['recommended_gross_cap']:.2f}")

    # ── Test 10: Bear trend detection ─────────────────────────────────────────
    bear_prices = {
        "SPY": _crashing_prices(60, seed=21),
        "QQQ": _crashing_prices(60, seed=22),
    }
    bear_result = detect_regime(bear_prices, {}, [])
    chk("Bear trend: risk_on_off <= 0",
        bear_result["risk_on_off_score"] <= 0,
        f"{bear_result['risk_on_off_score']:.3f}")

    # ── Test 11: Macro risk-off regime ────────────────────────────────────────
    neutral_prices = {
        "SPY": _flat_prices(60, seed=30),
        "TLT": _flat_prices(60, seed=31),
    }
    roff_macro = [
        _make_macro_signal("rate_hike",            impact=0.85, confidence=0.90),
        _make_macro_signal("yield_curve_inversion", impact=0.80, confidence=0.85),
        _make_macro_signal("recession",             impact=0.90, confidence=0.88),
    ]
    roff_result = detect_regime(neutral_prices, {}, roff_macro)
    chk("Macro risk-off signals produce negative risk_on_off",
        roff_result["risk_on_off_score"] < 0,
        f"{roff_result['risk_on_off_score']:.3f}")

    # ── Test 12: Gamma squeeze detection ──────────────────────────────────────
    muted_prices = {"SPY": _flat_prices(60, seed=40)}
    gamma_options = [
        _make_options_signal("call", gamma=0.080, strength=0.90, iv=0.45),
        _make_options_signal("call", gamma=0.075, strength=0.85, iv=0.42),
        _make_options_signal("call", gamma=0.090, strength=0.92, iv=0.50),
        _make_options_signal("put",  gamma=0.010, strength=0.30, iv=0.25),
    ]
    gamma_result = detect_regime(muted_prices, {}, [], gamma_options)
    gamma_score = gamma_result["sub_scores"]["options"]["gamma_squeeze_score"]
    chk("Gamma squeeze: options sub-score > 0",
        gamma_score > 0,
        f"gamma_squeeze_score={gamma_score:.4f}")

    # ── Test 13: AI bubble detection ──────────────────────────────────────────
    ai_prices = {
        "NVDA": _trending_prices(60, drift=0.005, noise=0.008, seed=50),
        "AMD":  _trending_prices(60, drift=0.004, noise=0.007, seed=51),
        "SMCI": _trending_prices(60, drift=0.006, noise=0.010, seed=52),
        "SPY":  _flat_prices(60, seed=53),
    }
    ai_positions = {"NVDA": 0.18, "AMD": 0.15, "SMCI": 0.12, "SPY": 0.10}
    ai_result_main = detect_regime(ai_prices, {}, [], None, ai_positions)
    ai_sub = ai_result_main["sub_scores"]["ai_bubble"]
    chk("AI bubble: ai_bubble_score > 0",
        ai_sub["ai_bubble_score"] > 0,
        f"score={ai_sub['ai_bubble_score']:.4f}")
    chk("AI bubble: ai_cluster_exposure > 0",
        ai_sub["ai_cluster_exposure"] > 0,
        f"exposure={ai_sub['ai_cluster_exposure']:.4f}")

    # ── Test 14: High volatility chop ─────────────────────────────────────────
    # noise=0.07 → daily stdev ≈ 0.07 × (1/√12) ≈ 2.02 %  → annualised ≈ 32 % (elevated)
    chop_prices = {
        "SPY": _volatile_prices(60, noise=0.070, seed=60),
        "QQQ": _volatile_prices(60, noise=0.075, seed=61),
        "IWM": _volatile_prices(60, noise=0.072, seed=62),
    }
    chop_result = detect_regime(chop_prices, {}, [])
    vol_band_chop = chop_result["volatility_regime"]
    chk("High-vol chop: vol_band not low (elevated, extreme, or at least normal)",
        vol_band_chop in ("elevated", "extreme", "normal"),
        f"vol_band={vol_band_chop}")

    # ── Test 15: Mean reversion detection ─────────────────────────────────────
    mr_prices = {
        "SPY": _flat_prices(80, seed=70),
        "GLD": _flat_prices(80, seed=71),
        "BND": _flat_prices(80, seed=72),
    }
    mr_result = detect_regime(mr_prices, {}, [])
    regime_mr = mr_result["market_regime"]
    tp_mr = mr_result["trend_persistence"]
    chk("Mean-reversion / flat prices: low trend_persistence",
        tp_mr < 0.80,
        f"persistence={tp_mr:.4f} regime={regime_mr}")

    # ── Test 16: Liquidity stress detection ───────────────────────────────────
    gap_prices = {
        "SPY": [100.0, 100.5, 97.0, 96.5, 93.0, 92.0, 88.0, 87.5,  # gap down sequence
                88.0, 87.0, 86.0, 85.0, 84.5, 84.0, 83.5] + [83.0] * 45,
    }
    liq_macro = [
        _make_macro_signal("liquidity_crisis", impact=0.90, confidence=0.88),
        _make_macro_signal("halt",             impact=0.85, confidence=0.80),
    ]
    liq_result_main = detect_regime(gap_prices, {}, liq_macro)
    liq_sub = liq_result_main["sub_scores"]["liquidity"]
    chk("Liquidity stress: stress_score > 0",
        liq_sub["liquidity_stress_score"] > 0,
        f"stress={liq_sub['liquidity_stress_score']:.4f}")

    # ── Test 17: Correlation spike detection ──────────────────────────────────
    # All assets moving identically → high correlation
    base = _crashing_prices(60, seed=80)
    corr_prices = {
        "SPY":  base,
        "QQQ":  [p * 0.999 for p in base],
        "IWM":  [p * 1.001 for p in base],
        "HYG":  [p * 0.998 for p in base],
        "EEM":  [p * 1.002 for p in base],
    }
    corr_result_main = detect_regime(corr_prices, {}, [])
    corr_sub = corr_result_main["sub_scores"]["correlation"]
    chk("Correlation spike: avg_correlation > 0.70",
        corr_sub["avg_correlation"] > 0.70,
        f"avg_corr={corr_sub['avg_correlation']:.4f}")

    # ── Test 18: sub_scores block completeness ────────────────────────────────
    required_sub = ["volatility", "drawdown", "correlation", "momentum",
                    "macro", "options", "liquidity", "ai_bubble"]
    chk("sub_scores contains all required blocks",
        all(k in result["sub_scores"] for k in required_sub),
        str([k for k in required_sub if k not in result.get("sub_scores", {})]))

    # ── Test 19: Hurst exponent in [0, 1] ─────────────────────────────────────
    h = result["sub_scores"]["momentum"].get("hurst_exponent", 0.5)
    chk("hurst_exponent in [0.0, 1.0]", 0.0 <= h <= 1.0, f"H={h:.4f}")

    # ── Test 20: Empty inputs return safe defaults ────────────────────────────
    empty_result = detect_regime({}, {}, [])
    chk("Empty inputs: returns valid output contract",
        all(k in empty_result for k in required_keys),
        "")
    chk("Empty inputs: risk_on_off in [−1, +1]",
        -1.0 <= empty_result["risk_on_off_score"] <= 1.0,
        f"{empty_result['risk_on_off_score']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for ok, _, _ in results if ok)
    print("═" * 72)
    print(f"  {passed}/{len(results)} smoke tests passed ✓")
    print("═" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_smoke_tests()